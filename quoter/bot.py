import logging
from typing import Callable, Any, List, Dict
from framework_config import QuoterConfig
from quoter import calculate_quote_levels
from models import QuoteSide

log = logging.getLogger(__name__)

# G3 buffer multiplier now reads from `config.g3_edge_mult` (per-row knob in
# market_parameters.csv). Default 0.5 set in QuoterConfig. Falls back to the
# legacy per-prefix table for any row that doesn't have the column populated
# (CSV reload sets g3_edge_mult=0.5 by default, so the fallback only fires
# on configs missing the attribute entirely — defensive guard for tests).
def _g3_edge_mult_for(config) -> float:
    v = getattr(config, "g3_edge_mult", None)
    if v is not None:
        return float(v)
    mid = (getattr(config, "market_id", "") or "").upper()
    if "CS2" in mid and "_T1" in mid:
        return 0.25
    return 0.5

class QuoterBot:
    """
    Instance representing a single quoting strategy hooked to a specific market/ticker.
    """
    def __init__(self, config: QuoterConfig):
        """
        :param config: QuoterConfig object populated from CSV.
        """
        self.config = config
        self.active = True

        # State tracking to minimize API calls
        self.last_bid_theo = None
        self.last_offer_theo = None
        self.last_top_bid = None
        self.last_top_offer = None
        self.last_desired_quotes = None
        self._last_position = None
        self._reduce_only_active = False

        # Per-side maker-fill watermark for the refill-cooldown feature. We
        # compare position_store.get_last_maker_fill_ts against this; any newer
        # ts means a fill just happened on (ticker, side) since we last looked.
        self._last_seen_maker_fill_ts = {"yes": 0.0, "no": 0.0}

    def evaluate(self, market_state: Any, top_level_bid: float, top_level_offer: float, bid_theo: float, offer_theo: float, **kwargs) -> List[Dict[str, Any]]:
        """
        Calculates theoretical edges and desired bounds using the provided evaluated theoreticals.
        """
        if not self.active:
            return []

        # ── Theo-sanity guards (added 2026-06-16 after VGYB blowup) ──
        # Two structural invariants that, when violated, mean upstream data is
        # in mid-update / unsynced and theos are garbage. Quoting off garbage
        # theos burned ~700 lots of self-trading on VGYB at 13:33-13:34.
        #
        # Guard 1: bid_theo > offer_theo is mathematically impossible in a
        #   healthy market. When it happens, quoter posts crossed BIDS and
        #   OFFERS that the arber's taker simultaneously hits — guaranteed loss.
        # Guard 2: cons_map_no_ask + cons_map_yes_ask < 98 means one ticker's
        #   no_ask is BELOW the other ticker's yes_ask — structurally impossible
        #   without one side being stale. Sums >100 are fine — that's just a
        #   wide market with both team books quoted loose; the math still holds.
        try:
            if (isinstance(bid_theo, (int, float)) and isinstance(offer_theo, (int, float))
                    and bid_theo > offer_theo):
                log.error(
                    f"[THEO-INVERTED HALT] {self.config.ticker} "
                    f"bid_theo={bid_theo:.1f} > offer_theo={offer_theo:.1f} — refusing to quote"
                )
                return []
            _se_g = kwargs.get("series_eval") or {}
            _cmn = _se_g.get("cons_map_no_ask")
            _cmy = _se_g.get("cons_map_yes_ask")
            if (isinstance(_cmn, (int, float)) and isinstance(_cmy, (int, float))
                    and (_cmn + _cmy) < 98.0):
                log.error(
                    f"[MAP-INCONSISTENT HALT] {self.config.ticker} "
                    f"cons_map_no={_cmn:.1f} + cons_map_yes={_cmy:.1f} = {_cmn + _cmy:.1f} (<98) — "
                    f"one ticker is stale, refusing to quote"
                )
                return []
        except Exception:
            pass  # never let sanity guards break quoting on edge-case types

        # Hedge-staleness suppression / Kalshi map-book fallback (26AUG13).
        # mode() == "OK":         quote normally (identical to old is_stale=False).
        # mode() == "SUPPRESSED": don't quote (identical to old is_stale=True).
        # mode() == "FALLBACK":   Poly frozen but Kalshi's own map books are
        #   fresh/tight AND the operator armed enable_stale_kalshi_fallback.flag
        #   — keep quoting makers at HALF clips (halving applied below), with a
        #   10c deviation guard: if the map inputs our theo was built from have
        #   drifted >10c from the live Kalshi map book, the theo is poisoned by
        #   the frozen feed — refuse (VITBBL chasing-theo protection).
        # We only consume here — the arber is the producer of snapshot data.
        _fb_halve = False
        try:
            _series_base = self.config.ticker.rsplit("-", 1)[0]
            import hedge_staleness
            _hs_mode = hedge_staleness.mode(_series_base)
            if _hs_mode == "SUPPRESSED":
                return []
            if _hs_mode == "FALLBACK":
                _fb = hedge_staleness.fallback_books(_series_base) or {}
                _se_f = kwargs.get("series_eval") or {}
                _am_f = _se_f.get("active_map")
                _ftb_f = kwargs.get("full_top_bids") or {}
                _fto_f = kwargs.get("full_top_offers") or {}
                if _am_f and _am_f in _fb:
                    _kb, _ka = _fb[_am_f]
                    _pb = _ftb_f.get(_am_f)
                    _pa = _fto_f.get(_am_f)
                    if (_pb is not None and _pa is not None
                            and abs((_kb + _ka) / 2.0 - (_pb + _pa) / 2.0) > 10.0):
                        log.warning(
                            f"[POLY-STALE FALLBACK DEV-GUARD] {self.config.ticker} "
                            f"theo map inputs ({_pb:.0f}/{_pa:.0f}) deviate >10c "
                            f"from live kalshi map book ({_kb}/{_ka}) — not quoting")
                        return []
                _fb_halve = True
        except Exception:
            pass  # never let staleness lookup break quoting

        # BO5 safety: halt if active map has no real Kalshi orderbook (defaults
        # to ask=100 / bid=0). Quoter's theos still flow from p1/p2/p3/p4/p5
        # static values — fine during G1-G3, dangerous in G4/G5 where the
        # model has no live grounding. Auto-resumes when ticker appears.
        try:
            _se = kwargs.get("series_eval") or {}
            if _se.get("is_bo5"):
                _am = _se.get("active_map")
                _ftb = kwargs.get("full_top_bids") or {}
                _fto = kwargs.get("full_top_offers") or {}
                if _am:
                    _am_ask = _fto.get(_am, 100.0)
                    _am_bid = _ftb.get(_am, 0.0)
                    if _am_ask >= 99 and _am_bid <= 0:
                        log.warning(
                            f"[BO5 HALT — NO KALSHI MAP] {self.config.ticker} "
                            f"state={_se.get('state')} active_map={_am} "
                            f"ask={_am_ask:.1f} bid={_am_bid:.1f} — quoter refusing"
                        )
                        return []
        except Exception:
            pass  # never let BO5 safety lookup break quoting

        # ── Maker refill cooldown: detect new maker fills on our (ticker, side)
        # and trigger a direction-keyed cooldown. The cooldown is shared with
        # the opposite team's QuoterBot via position_store, so a YES fill on
        # us also blocks the other side's mirror BID NO. Skipped if cooldown
        # is configured to 0 (legacy / disabled).
        cooldown_sec = getattr(self.config, "maker_refill_cooldown_sec", 0.0)
        event_base = None
        our_suffix = None
        opp_suffix = None
        if cooldown_sec > 0:
            try:
                import position_store
                parts = self.config.ticker.split("-")
                if len(parts) >= 3:
                    event_base = "-".join(parts[:2])
                    our_suffix = parts[-1]
                    # Look for sibling tickers under the same event_base in
                    # full_market_state; the opponent is any ticker that shares
                    # event_base prefix but has a different suffix.
                    full_state = kwargs.get("full_market_state") or {}
                    for t in full_state.keys():
                        if isinstance(t, str) and t.startswith(event_base + "-") and t != self.config.ticker:
                            t_parts = t.split("-")
                            # Filter to series-game tickers only (skip MAP/SETWINNER/etc.
                            # which have an extra map-number segment between event_base
                            # and team suffix). Series-game tickers have exactly 3 parts.
                            if len(t_parts) == 3:
                                opp_suffix = t_parts[-1]
                                break

                    # For each side, check the global maker-fill watermark and
                    # trigger a cooldown for the matching exposure direction if
                    # there's been a new fill since we last looked.
                    for side, long_team in (("yes", our_suffix), ("no", opp_suffix)):
                        if long_team is None:
                            continue  # opponent not yet known — skip cross-ticker trigger
                        last_ts = position_store.get_last_maker_fill_ts(self.config.ticker, side)
                        if last_ts > self._last_seen_maker_fill_ts.get(side, 0.0):
                            position_store.trigger_direction_cooldown(event_base, long_team, cooldown_sec)
                            log.warning(f"[QUOTER COOLDOWN] {self.config.ticker} side={side} → "
                                        f"cool (event={event_base}, long={long_team}) for {cooldown_sec}s")
                            self._last_seen_maker_fill_ts[side] = last_ts
            except Exception as e:
                log.exception(f"[QUOTER COOLDOWN] {self.config.ticker}: error in fill detection: {e}")

        # Hard position cap at 1.5× max_position with 1.4× release hysteresis.
        # Once |pos| ≥ 1.5×max_position, switch to reduce-only quoting (only the
        # side that shrinks |pos|: offers when long YES, bids when short YES).
        # Stay in reduce-only mode until fills bring |pos| < 1.4×max_position,
        # then resume two-sided quoting. The gap prevents flapping right at the
        # boundary. Game-decided / stale-book detection is handled centrally in
        # hedge_engine.detect_bo3_state via the loose-5c rule + live-M2 hedge math.
        reduce_only_side = None  # None = unrestricted; "bids" or "offers" = only that side
        try:
            import position_store
            cur = position_store.get_position(self.config.ticker)
            hard_cap = int(1.5 * self.config.max_position)
            release_cap = int(1.4 * self.config.max_position)
            abs_pos = abs(cur) if cur is not None else 0
            if self._reduce_only_active:
                if abs_pos < release_cap:
                    self._reduce_only_active = False
                    log.warning(f"[STOP-CAP] {self.config.ticker}: |pos|={abs_pos} < "
                                f"1.4×max_position={release_cap} — released, resuming both sides")
            elif cur is not None and abs_pos >= hard_cap:
                self._reduce_only_active = True
                entry_side = "offers" if cur > 0 else "bids"
                log.warning(f"[STOP-CAP] {self.config.ticker}: |pos|={abs_pos} ≥ "
                            f"1.5×max_position={hard_cap} — reduce-only ({entry_side})")
            if self._reduce_only_active and cur is not None:
                reduce_only_side = "offers" if cur > 0 else "bids"
        except Exception:
            pass

        # G3 edge buffer: during G2 of a BO3 (state 1 = our team up 1-0,
        # state 2 = our team down 0-1; both = G2 active), widen min_edge by
        # P(G3)*min_edge to protect against systematic G3 mispricing.
        # P(G3) = P(G1 loser wins G2), derived from current m2 mid + m1_won
        # flag. Mirrors the same buffer applied in arber_bot. Added 2026-05-27.
        effective_min_edge = self.config.min_edge
        se = kwargs.get("series_eval")
        if se and not se.get("is_bo5") and se.get("state") in (1, 2):
            am = se.get("active_map")
            ftb = kwargs.get("full_top_bids") or {}
            fto = kwargs.get("full_top_offers") or {}
            if am and am in ftb and am in fto:
                m2_mid = (ftb[am] + fto[am]) / 2.0
                if se.get("m1_won"):
                    p_g3 = max(0.0, (100.0 - m2_mid) / 100.0)
                else:
                    p_g3 = max(0.0, m2_mid / 100.0)
                effective_min_edge = self.config.min_edge + _g3_edge_mult_for(self.config) * p_g3 * self.config.min_edge

        # Use general routing math to determine acceptable target limits
        bids, offers = calculate_quote_levels(
            bid_theo=bid_theo,
            offer_theo=offer_theo,
            top_level_bid=top_level_bid,
            top_level_offer=top_level_offer,
            min_distance_from_top_level=self.config.min_distance_from_top_level,
            min_edge=effective_min_edge,
            min_absolute_edge=self.config.min_absolute_edge,
            volumes=self.config.volumes,
            tick_step=self.config.tick_step,
            fav_edge_k=self.config.fav_edge_k,
        )

        desired_quotes = []

        # Step 3: Bundle Bids
        # YES-asymmetric widening: when yes_extra_edge_cents > 0, lower the YES
        # bid by that many cents (no effect on NO bids below). Floor at 1c to
        # keep the order valid. Default 0 = symmetric (legacy).
        yes_widen = int(round(self.config.yes_extra_edge_cents))
        if self.config.quote_side in ["both", "bids"] and reduce_only_side in (None, "bids"):
            for price, vol in bids:
                adj_price = max(1, price - yes_widen) if yes_widen > 0 else price
                desired_quotes.append({
                    "side": QuoteSide.BID,
                    "ticker": self.config.ticker,
                    "kalshi_side": "yes",
                    "size": vol,
                    "limit_cents": adj_price
                })

        # Step 4: Bundle Offers
        # On Kalshi, an Offer (Selling YES at price X) typically requires buying the NO 
        # contract at price (100 - X). We translate that structure here for execution ease.
        if self.config.quote_side in ["both", "offers"] and reduce_only_side in (None, "offers"):
            for price, vol in offers:
                desired_quotes.append({
                    "side": QuoteSide.OFFER,
                    "ticker": self.config.ticker,
                    "kalshi_side": "no",
                    "size": vol,
                    "limit_cents": 100 - price
                })

        # Drop any desired quote whose exposure direction is currently in cooldown.
        # YES bid → "long our_suffix" direction; NO bid → "long opp_suffix" direction.
        # (Both QuoterBot quote types are bids — see bot.py offer-as-NO-bid translation
        # above.) If filtering empties the list, return [] so the manager cancels
        # any resting orders for this bot rather than leaving them on the book.
        if cooldown_sec > 0 and event_base is not None and our_suffix is not None and desired_quotes:
            try:
                import position_store
                kept = []
                for q in desired_quotes:
                    ks = q.get("kalshi_side", "yes").lower()
                    long_team = our_suffix if ks == "yes" else opp_suffix
                    if long_team is None:
                        # opponent unknown — only same-ticker cooldown applies
                        long_team = our_suffix if ks == "yes" else None
                    if long_team is not None and position_store.is_direction_in_cooldown(event_base, long_team):
                        continue
                    kept.append(q)
                if len(kept) < len(desired_quotes):
                    log.warning(f"[QUOTER COOLDOWN] {self.config.ticker}: "
                                f"filtered {len(desired_quotes) - len(kept)}/{len(desired_quotes)} quotes (cooldown active)")
                desired_quotes = kept
            except Exception as e:
                log.exception(f"[QUOTER COOLDOWN] {self.config.ticker}: filter error: {e}")

        # Kalshi-fallback mode: makers only at HALF clips (scope Phase 1).
        # Applied before the reprice/dedup logic so cached last_desired_quotes
        # compare like-for-like across cycles.
        if _fb_halve and desired_quotes:
            for _q in desired_quotes:
                _q["size"] = max(1, int(_q["size"] * 0.5))

        # Check if position changed (fill happened) — always reprice after fills
        try:
            import position_store
            current_pos = position_store.get_position(self.config.ticker)
        except Exception:
            current_pos = None

        position_changed = (self._last_position is not None and
                           current_pos is not None and
                           current_pos != self._last_position)
        self._last_position = current_pos

        # Final Anchor Check: Track strict physical differences over the reprice_buffer
        #
        # Buffer is ADAPTIVELY capped at REPRICE_BUFFER_EDGE_FRACTION × effective
        # edge at the quote's current price level. This enforces the invariant:
        #
        #   reprice_buffer < effective_min_edge   (always, at all price levels)
        #
        # which prevents the case where a quote sits at "zero effective edge"
        # before we reprice — getting filled at theo = guaranteed loss.
        #
        # effective_edge = max(min_edge × variance_scale, min_absolute_edge)
        # where variance_scale = 4p(1-p) is the standard Kalshi binary-outcome
        # variance scaling. At extreme prices the floor (min_absolute_edge)
        # dominates; at midprice the scaled edge does.
        REPRICE_BUFFER_EDGE_FRACTION = 0.5  # never reprice on > half the edge

        def _adaptive_buffer(quote_price_c):
            """Reprice buffer = min(configured, EDGE_FRACTION × effective edge)."""
            p = max(0.01, min(0.99, quote_price_c / 100.0))
            variance_scaled = self.config.min_edge * 4.0 * p * (1.0 - p)
            effective_edge = max(variance_scaled, self.config.min_absolute_edge)
            return min(self.config.reprice_buffer, effective_edge * REPRICE_BUFFER_EDGE_FRACTION)

        if not position_changed and self.last_desired_quotes is not None and len(self.last_desired_quotes) == len(desired_quotes):
            should_reprice = False
            for i in range(len(desired_quotes)):
                new_q = desired_quotes[i]
                old_q = self.last_desired_quotes[i]

                # Adaptive buffer based on the old quote's price level — that's
                # where the quote currently sits and where the edge math applies.
                adaptive_buf = _adaptive_buffer(old_q["limit_cents"])

                # Check absolute pricing delta against the adaptive buffer
                if abs(new_q["limit_cents"] - old_q["limit_cents"]) > adaptive_buf:
                    should_reprice = True
                    break

                # Standard catch if side or volume definitions drastically shift underneath
                if new_q["size"] != old_q["size"] or new_q["kalshi_side"] != old_q["kalshi_side"]:
                    should_reprice = True
                    break

            if not should_reprice:
                return None

        self.last_desired_quotes = desired_quotes

        if desired_quotes:
            se = kwargs.get("series_eval") or {}
            yes_bids = [(q["limit_cents"], q["size"]) for q in desired_quotes
                        if q["side"] == QuoteSide.BID and q["kalshi_side"] == "yes"]
            yes_offers = [(100 - q["limit_cents"], q["size"]) for q in desired_quotes
                          if q["side"] == QuoteSide.OFFER and q["kalshi_side"] == "no"]
            state = se.get("state", "?")
            active_map = se.get("active_map", "?")
            cons_no = se.get("cons_map_no_ask", "?")
            cons_yes = se.get("cons_map_yes_ask", "?")
            syn_l = se.get("synthetic_long_cost")
            syn_s = se.get("synthetic_short_cost")
            p1 = se.get("p1"); p2 = se.get("p2"); ser = se.get("series_prob")
            syn_l_s = f"{syn_l:.1f}" if isinstance(syn_l, (int, float)) else "?"
            syn_s_s = f"{syn_s:.1f}" if isinstance(syn_s, (int, float)) else "?"
            p1_s = f"{p1:.3f}" if isinstance(p1, (int, float)) else "?"
            p2_s = f"{p2:.3f}" if isinstance(p2, (int, float)) else "?"
            ser_s = f"{ser:.3f}" if isinstance(ser, (int, float)) else "?"
            bt = f"{bid_theo:.1f}" if isinstance(bid_theo, (int, float)) else "?"
            ot = f"{offer_theo:.1f}" if isinstance(offer_theo, (int, float)) else "?"
            log.info(
                f"[ARBER MAKER FIRE] {self.config.ticker} state={state} active_map={active_map} | "
                f"bid_theo={bt} offer_theo={ot} | "
                f"bids={yes_bids} offers={yes_offers} | "
                f"top={top_level_bid}/{top_level_offer} | "
                f"cons_map_no={cons_no} cons_map_yes={cons_yes} | "
                f"syn_long={syn_l_s} syn_short={syn_s_s} | "
                f"p1={p1_s} p2={p2_s} series={ser_s}"
            )

        return desired_quotes
