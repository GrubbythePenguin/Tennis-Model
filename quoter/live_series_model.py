"""
Live Series Theo Generator — central evaluator.

Per cycle, for each esports series ticker, produces a SeriesEvaluation dict:
  {
    "bid_theo", "offer_theo", "raw_bid_theo", "raw_offer_theo",   # quoter-facing
    "state", "active_map", "active_map_opp",                       # state from detect_bo3_state
    "m1_won", "m1_lost", "m2_won", "m2_lost",                      # map outcomes
    "cons_map_no_ask", "cons_map_yes_ask",                          # cross-book map prices
    "hedge_profile_long", "hedge_profile_short",                    # hedge math (BO3 state)
    "synthetic_long_cost", "synthetic_short_cost",                  # series_loaded + hedge cost
    "edge_long_c", "edge_short_c",                                  # 100 - synthetic
    "p1", "p2", "p3", "series_prob", "g2_momentum", "g3_momentum",  # probabilities
    "is_bo5", "verified",                                           # CSV flags
  }

Inactive/skipped tickers are absent from the result dict.

Existing consumers (HybridTheoGenerator) only read theo fields from the
returned dict, so the richer shape is backward-compatible. Bots that want
the full evaluation can read the additional fields once Phase 5+ wiring
hands `series_eval` through manager.evaluate.
"""
import logging
from typing import Dict, List, Any
import manual_score_gate
import bo3_score_feed
from base_model import BaseTheoGenerator
from hedge_engine import (
    detect_bo3_state,
    detect_bo5_state,
    get_cross_book_map_prices,
    get_cross_book_series_prices,
    compute_hedge_profiles,
    compute_synthetic_costs,
    compute_series_theos,
    compute_map_hedge_profiles,
    compute_map_synthetic_costs,
    compute_map_theos,
    parse_series_ticker,
    load_probabilities,
    get_loaded_cost_cents,
    bo5_disabled,
)

log = logging.getLogger(__name__)


class LiveSeriesTheoGenerator(BaseTheoGenerator):
    """
    Generates bid/offer theos for series tickers during live games
    by deriving fair value from the active map price using the arber's
    hedge cost functions.
    """

    def __init__(self, client, configs=None):
        super().__init__(client, configs)
        self._prob_cache = load_probabilities()
        self._skipped_tickers = set()
        self._bo3_toxic_skipped = set()   # tickers halted by the bo3 toxicity gate
        log.info(f"[LIVE SERIES] Loaded {len(self._prob_cache)} probabilities")

    def _batch_generate(self, tickers: List[str], dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        # Reload probs if any tickers are missing from cache
        if any(t not in self._prob_cache for t in tickers):
            self._prob_cache = load_probabilities()

        results = {}

        # Build full_top_bids/offers from raw_ob when available,
        # then fill in from injected top-of-book (covers Poly CLOB path
        # which writes to top_level_bids/offers but not raw_ob)
        full_top_bids = {}
        full_top_offers = {}
        for ticker, state in dt_market_state.items():
            if isinstance(state, dict):
                raw_ob = state.get("raw_ob", {}).get("orderbook_fp", {})
                if raw_ob:
                    y = raw_ob.get("yes_dollars", [])
                    n = raw_ob.get("no_dollars", [])
                    if y:
                        y_sorted = sorted(y, key=lambda x: float(x[0]), reverse=True)
                        full_top_bids[ticker] = round(float(y_sorted[0][0]) * 100)
                    if n:
                        n_sorted = sorted(n, key=lambda x: float(x[0]), reverse=True)
                        full_top_offers[ticker] = 100 - round(float(n_sorted[0][0]) * 100)

        # Fallback: use manager-injected top-of-book for tickers without raw_ob
        injected_bids = dt_market_state.get("__top_bids__", {})
        injected_offers = dt_market_state.get("__top_offers__", {})
        for t, v in injected_bids.items():
            if t not in full_top_bids:
                full_top_bids[t] = v
        for t, v in injected_offers.items():
            if t not in full_top_offers:
                full_top_offers[t] = v

        for ticker in tickers:
            probs = self._prob_cache.get(ticker)
            if not probs:
                if ticker not in self._skipped_tickers:
                    log.info(f"[LIVE SERIES] {ticker}: no probs — inactive")
                    self._skipped_tickers.add(ticker)
                continue

            parsed = parse_series_ticker(ticker)
            if not parsed:
                continue
            series_base, team_suffix, map_base = parsed

            p1 = probs["p1"]
            p2 = probs["p2"]
            series = probs["series"]
            g2_momentum = probs.get("g2_momentum", 0.0)
            g3_momentum = probs.get("g3_momentum", 0.0)
            is_bo5 = probs.get("bo5", False)

            # BO5 runtime kill-switch. When disable_bo5.flag is present, skip
            # BO5 tickers entirely so no series_eval is published — quoter has
            # no theos to quote on, and arber's `elif is_bo5: return []`
            # short-circuits before any orders fire.
            if is_bo5 and bo5_disabled():
                if ticker not in self._skipped_tickers:
                    log.info(f"[LIVE SERIES] {ticker}: BO5 disabled via disable_bo5.flag — skipping")
                    self._skipped_tickers.add(ticker)
                # HALT SIGNAL (2026-07-18): affirmatively tell the manager we have
                # NO valid fair value → cancel all resting orders + fire nothing,
                # for every bot on this ticker (arbers included). See manager.py.
                results[ticker] = {"halt": True, "halt_reason": "bo5_disabled"}
                continue

            p3 = probs.get("p3", (p1 + p2) / 2.0)
            # p4/p5 from loaded probs when populate_configs captured them
            # from Polymarket G4/G5 Winner markets. None falls back to the
            # (p1+p2+p3)/3 stand-in inside calculate_bo5_hedge.
            p4 = probs.get("p4")
            p5 = probs.get("p5")

            # If the series itself is at 99+ or 1-, it's decided — stop quoting.
            series_bid = full_top_bids.get(ticker, 0)
            series_ask = full_top_offers.get(ticker, 100)
            if series_bid >= 99 or series_ask <= 1:
                if ticker not in self._skipped_tickers:
                    log.info(f"[LIVE SERIES] {ticker}: series at {series_bid}/{series_ask} — decided, skipping")
                    self._skipped_tickers.add(ticker)
                # HALT SIGNAL: series decided → no fair value → cancel all + no fire.
                results[ticker] = {"halt": True, "halt_reason": "series_decided"}
                continue

            # Detect state and active map via centralized engine.
            # BO5: detect_bo5_state walks M1-M5 (returns tuple state, blocks G5).
            # BO3: detect_bo3_state walks M1/M2 (returns int state 0-3).
            # Both bots (arber + quoter) read the same state_info — single source
            # of truth for series state.
            # data_source registry is injected by manager.py; falls back to default if absent.
            ds_registry = dt_market_state.get("__data_source__", {}) or {}
            ds = ds_registry.get(ticker, "kalshi_na_poly_eu")
            disable_registry = dt_market_state.get("__disable_forfeit__", {}) or {}
            disable_ff = disable_registry.get(ticker, False)
            if is_bo5:
                state_info = detect_bo5_state(
                    map_base, team_suffix, dt_market_state,
                    full_top_bids, full_top_offers, series_base=series_base,
                    data_source=ds,
                    disable_forfeit_check=disable_ff)
            else:
                state_info = detect_bo3_state(
                    map_base, team_suffix, dt_market_state,
                    full_top_bids, full_top_offers, series_base=series_base,
                    data_source=ds,
                    disable_forfeit_check=disable_ff)

            if state_info is None or state_info["active_map"] is None:
                if ticker not in self._skipped_tickers:
                    log.info(f"[LIVE SERIES] {ticker}: cannot detect state — inactive")
                    self._skipped_tickers.add(ticker)
                # HALT SIGNAL: forfeit / undetectable state / no active map → no
                # fair value → cancel all resting orders + fire nothing. This is the
                # forfeit-freeze case that ran the arber over on ATRNTR 2026-07-18.
                results[ticker] = {"halt": True, "halt_reason": "no_state_or_forfeit"}
                continue
            state = state_info["state"]
            active_map = state_info["active_map"]
            opp_map = state_info["active_map_opp"]

            # If we previously skipped this ticker but now it's active again, log it
            if ticker in self._skipped_tickers:
                log.info(f"[LIVE SERIES] {ticker}: reactivated (state={state})")
                self._skipped_tickers.discard(ticker)

            # Cross-book map prices via centralized engine
            cons_map_no_ask, cons_map_yes_ask = get_cross_book_map_prices(
                active_map, opp_map, full_top_bids, full_top_offers)

            map_bid = full_top_bids.get(active_map, 0)
            map_ask = full_top_offers.get(active_map, 100)
            opp_map_bid = full_top_bids.get(opp_map, 0)
            opp_map_ask = full_top_offers.get(opp_map, 100)
            both_empty = (map_bid == 0 and map_ask == 100 and
                          opp_map_bid == 0 and opp_map_ask == 100)

            # data_source=poly strictness (2026-07-15, EFTL post-mortem): for a
            # poly-source event the feed injects BOTH map tickers (the real,
            # fresh Poly CLOB top-of-book — NOT a synthetic mid±1c) when Poly is
            # healthy. If the ACTIVE map's OWN ticker is empty (0/100),
            # Poly data is incomplete for this leg — and get_cross_book_map_prices
            # would otherwise synthesize the active price PURELY from the opp
            # ticker via min() (e.g. our EF empty + a stale/partial TL book →
            # cons_map_yes = 100-43 = 57), manufacturing a phantom pregame-priced
            # "56 at 57" market and a fake ~20c edge. Never guess: suppress.
            active_own_empty = (map_bid == 0 and map_ask == 100)

            # ── Manual operator-score plausibility gate (TEMPORARY 2026-07-21) ──
            # When enable_manual_score_gate.flag is present AND the operator has
            # entered this event's live score in manual_scores.csv, halt the
            # event if the map-implied prob (cons_map_yes_ask) is impossible for
            # that score — the KCGX/WALJUS phantom-hedge-blip class. Only checks
            # a real book (skip empty/one-sided, other guards own those). No
            # flag or no CSV row → total no-op. See manual_score_gate.py.
            if (manual_score_gate.enabled() and not both_empty
                    and not active_own_empty):
                _ok, _why = manual_score_gate.check(
                    series_base, team_suffix, float(cons_map_yes_ask))
                if not _ok:
                    if ticker not in self._skipped_tickers:
                        log.warning(f"[MANUAL SCORE GATE] {ticker}: {_why} — halting")
                        self._skipped_tickers.add(ticker)
                    results[ticker] = {"halt": True, "halt_reason": "score_implausible"}
                    continue

            # ── bo3 round-score toxicity gate (OFF by default 2026-07-22) ──
            # When enable_bo3_toxicity_gate.flag is present AND bo3 reports the
            # active map is in the pickoff regime (one team >=10 AND diff <=3,
            # live & fresh), MAKER-SUPPRESS this event: pull our resting quotes,
            # which get run over in the coinflip zone no matter how fast we are
            # (FNC-KC g1 13-13 OT class; VAL maker ultra-toxic tail settled
            # ~-32c/lot). This emits `maker_suppress`, NOT `halt`: the manager
            # cancels ONLY the dedicated quoter, so arber / momentum TAKERS keep
            # firing — 7d analysis showed map-triggered takers are the informed,
            # profitable side of the very same move. Symmetric (score order
            # irrelevant). Flag absent (or event out of the flag's scope list) ->
            # toxicity_active() returns False without ever starting the feed
            # thread => total no-op. Unknown/stale bo3 -> proceed (additive
            # safety, never suppress a market bo3 can't see).
            _bo3_toxic = bo3_score_feed.toxicity_active(series_base)
            # Throttled heartbeat: what bo3 reports for this gated event (score,
            # map, toxic decision). Only fires for an ON + in-scope event and
            # only on change / every DEBUG_HEARTBEAT_SEC — a no-op when off.
            _bo3_dbg = bo3_score_feed.gate_debug(series_base)
            if _bo3_dbg:
                log.info(f"[BO3 GATE DEBUG] {_bo3_dbg}")
            if _bo3_toxic:
                if ticker not in self._bo3_toxic_skipped:
                    _bands = bo3_score_feed.describe_thresholds(
                        bo3_score_feed.event_sport(series_base))
                    log.warning(f"[BO3 TOXICITY GATE] {ticker}: active map in "
                                f"pickoff regime ({_bands}) — suppressing "
                                f"QUOTES (takers keep firing)")
                    self._bo3_toxic_skipped.add(ticker)
                results[ticker] = {"maker_suppress": True,
                                   "suppress_reason": "map_pickoff_toxic"}
                continue
            elif ticker in self._bo3_toxic_skipped:
                log.info(f"[BO3 TOXICITY GATE] {ticker}: pickoff regime cleared "
                         f"— resuming quotes")
                self._bo3_toxic_skipped.discard(ticker)

            # Guard: when the active-map books are both 0/100, we have no map
            # data to derive theos from. State 1/2: M2 just settled and Kalshi's
            # status hasn't propagated. State 0: either truly pre-game (nothing
            # useful to quote anyway) or — the dangerous case — stale post-
            # settlement where M1/M2 both cleared and detect_bo3_state fell
            # back to state 0 because the loose-decided check requires 0<bid<5
            # and a fully empty book fails that. Using baseline probabilities
            # here publishes phantom ~50c theos against the real series price,
            # which is how BBB8 got picked off at 3c during g3 on 2026-05-11.
            # BO5 tuple states are treated as live-data-required as well — a
            # BO5 with no active-map data is no safer to quote than BO3.
            bo3_active_state = state in (0, 1, 2)
            bo5_any_state = isinstance(state, tuple)
            # Effective poly-source right now — mirrors hedge_engine's resolver
            # (poly → always; kalshi_na_poly_eu → poly only outside NA hours).
            if ds == "poly":
                _poly_now = True
            elif ds == "kalshi_na_poly_eu":
                from esports_config import is_na_hours
                _poly_now = not is_na_hours(series_base)
            else:
                _poly_now = False
            suppress_map = both_empty or (_poly_now and active_own_empty)
            if (bo3_active_state or bo5_any_state) and suppress_map:
                if ticker not in self._skipped_tickers:
                    _why = ("both active-map books empty (0/100)" if both_empty
                            else f"poly active-map own book empty "
                                 f"(bid={map_bid},ask={map_ask}) — Poly leg incomplete")
                    log.info(f"[LIVE SERIES] {ticker}: state={state} {_why} — suppressing theos")
                    self._skipped_tickers.add(ticker)
                continue

            # In state 0, optionally swap baseline-p2 for live M2 ask in
            # hedge math (when M1 essentially decided but M2 still trading).
            # See detect_bo3_state for the trigger logic.
            use_live_m2 = bool(state_info.get("use_live_m2_state0", False))
            m2_no_ask, m2_yes_ask = None, None
            if use_live_m2:
                m2_ticker = f"{map_base}-2-{team_suffix}"
                m2_our_bid = full_top_bids.get(m2_ticker, 0)
                m2_our_ask = full_top_offers.get(m2_ticker, 100)
                # Same convention as get_cross_book_map_prices: NO ask = 100 - YES bid.
                # For better cross-book pricing we'd compute against opp ticker too,
                # but for state-0 fallback the our-ticker derivation is sufficient.
                if m2_our_bid > 0:
                    m2_no_ask = 100.0 - m2_our_bid
                if m2_our_ask < 100:
                    m2_yes_ask = float(m2_our_ask)

            # Hedge profiles + theos via centralized engine.
            # BO5: detect_bo5_state provides wins_a / wins_b directly (walks
            # M1-M5). For BO3 events, derive wa/wb from M1/M2 outcomes — same
            # behavior as before this refactor.
            if is_bo5:
                wa = state_info["wins_a"]
                wb = state_info["wins_b"]
                # compute_hedge_profiles ignores `state` (int) on the BO5 path —
                # it dispatches purely on is_bo5 + (wins_a, wins_b). Passing 0
                # here is safe; preserves the original signature.
                state_for_hedge = 0
            else:
                wa = int(state_info["m1_won"]) + int(state_info["m2_won"])
                wb = int(state_info["m1_lost"]) + int(state_info["m2_lost"])
                state_for_hedge = state
            hp1, hp2 = compute_hedge_profiles(
                state_for_hedge, p1, p2, series, g2_momentum, g3_momentum,
                cons_map_no_ask, cons_map_yes_ask,
                m2_no_ask=m2_no_ask, m2_yes_ask=m2_yes_ask,
                use_live_m2_state0=use_live_m2,
                is_bo5=is_bo5, p3=p3, p4=p4, p5=p5,
                wins_a=wa, wins_b=wb)

            bid_theo, offer_theo = compute_series_theos(hp1, hp2)

            # Synthetic costs + edges (relative to current series book).
            # Synthetic_long = loaded(series_yes_ask) + hedge_long_cost
            # Synthetic_short = loaded(series_no_ask) + hedge_short_cost
            series_no_ask = 100.0 - series_bid
            synthetic_long_cost, synthetic_short_cost, edge_long_c, edge_short_c = compute_synthetic_costs(
                float(series_ask), series_no_ask, hp1, hp2,
            )

            # ── Map-arber additions (wrapped in try/except so any failure here
            # cannot break the series-side path that quoter/arber depend on).
            # BO5 not supported by map_arber — only computed for BO3.
            map_fields = {}
            if not is_bo5:
                try:
                    opp_suffix = opp_map.split("-")[-1] if opp_map else None
                    opp_series_ticker = f"{series_base}-{opp_suffix}" if opp_suffix else None
                    if opp_series_ticker:
                        cons_series_no_ask, cons_series_yes_ask = get_cross_book_series_prices(
                            ticker, opp_series_ticker, full_top_bids, full_top_offers,
                        )
                    else:
                        cons_series_no_ask, cons_series_yes_ask = series_no_ask, float(series_ask)

                    map_hp1, map_hp2 = compute_map_hedge_profiles(
                        state, p1, p2, series, g2_momentum, g3_momentum,
                        series_no_ask=cons_series_no_ask,
                        series_yes_ask=cons_series_yes_ask,
                    )
                    map_bid_theo, map_offer_theo = compute_map_theos(map_hp1, map_hp2)
                    (map_synthetic_long_cost, map_synthetic_short_cost,
                     map_edge_long_c, map_edge_short_c) = compute_map_synthetic_costs(
                        float(cons_map_yes_ask), float(cons_map_no_ask), map_hp1, map_hp2,
                    )
                    map_fields = {
                        "cons_series_yes_ask": cons_series_yes_ask,
                        "cons_series_no_ask": cons_series_no_ask,
                        "map_hedge_profile_long": map_hp1,
                        "map_hedge_profile_short": map_hp2,
                        "map_synthetic_long_cost": map_synthetic_long_cost,
                        "map_synthetic_short_cost": map_synthetic_short_cost,
                        "map_edge_long_c": map_edge_long_c,
                        "map_edge_short_c": map_edge_short_c,
                        "map_bid_theo": map_bid_theo,
                        "map_offer_theo": map_offer_theo,
                    }
                except Exception as e:
                    log.debug(f"[LIVE SERIES] {ticker}: map-side compute failed (series-side OK): {e}")

            results[ticker] = {
                # Theo fields (consumed by HybridTheoGenerator / quoter)
                "bid_theo": bid_theo,
                "offer_theo": offer_theo,
                "raw_bid_theo": bid_theo,
                "raw_offer_theo": offer_theo,
                # State (consumed by future arber/hedger migrations)
                "state": state,
                "active_map": active_map,
                "active_map_opp": opp_map,
                "m1_won": state_info["m1_won"],
                "m1_lost": state_info["m1_lost"],
                "m2_won": state_info["m2_won"],
                "m2_lost": state_info["m2_lost"],
                # Cross-book map prices
                "cons_map_no_ask": cons_map_no_ask,
                "cons_map_yes_ask": cons_map_yes_ask,
                # Hedge profiles
                "hedge_profile_long": hp1,
                "hedge_profile_short": hp2,
                # Synthetic costs and edges
                "synthetic_long_cost": synthetic_long_cost,
                "synthetic_short_cost": synthetic_short_cost,
                "edge_long_c": edge_long_c,
                "edge_short_c": edge_short_c,
                # Probabilities (echoed for consumer convenience)
                "p1": p1,
                "p2": p2,
                "p3": probs.get("p3", (p1 + p2) / 2.0),
                "series_prob": series,
                "g2_momentum": g2_momentum,
                "g3_momentum": g3_momentum,
                "is_bo5": probs.get("bo5", False),
                "verified": probs.get("verified", False),
                # Map-arber fields (populated for BO3 only; map_fields is {} otherwise)
                **map_fields,
            }

            # Publish a second results entry keyed by the active_map ticker so a
            # QuoterBot registered against the map can consume map_bid_theo /
            # map_offer_theo via the standard live_series model dispatch. This
            # mirrors the series-side path (QuoterBot reads bid_theo/offer_theo
            # from results[series_ticker]) and lets us run a map-side quoter
            # alongside map_arber with no new bot class.
            if active_map and map_fields and "map_bid_theo" in map_fields:
                results[active_map] = {
                    "bid_theo": map_fields["map_bid_theo"],
                    "offer_theo": map_fields["map_offer_theo"],
                    "raw_bid_theo": map_fields["map_bid_theo"],
                    "raw_offer_theo": map_fields["map_offer_theo"],
                }

            log.debug(f"[LIVE SERIES] {ticker} state={state} "
                      f"map_no_ask={cons_map_no_ask:.0f} map_yes_ask={cons_map_yes_ask:.0f} "
                      f"bid_theo={bid_theo:.1f}c offer_theo={offer_theo:.1f}c "
                      f"edge_long={edge_long_c:.1f}c edge_short={edge_short_c:.1f}c")

        return results
