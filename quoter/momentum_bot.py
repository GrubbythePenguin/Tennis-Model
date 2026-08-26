"""Momentum strategy bot — fires top-of-book IOCs in the band BELOW the regular
arber's threshold, triggered by sudden hedge-cost moves (map jumps).

Design rationale and full spec in commit message / project notes. Key invariants:

  Sport whitelist (hardcoded): LoL, CS2, VALORANT. Other prefixes are rejected
  at evaluate time even if a row with execution_type=momentum is added for them.

  Edge band: [momentum_edge_floor, momentum_edge_ceiling] where
    momentum_edge_ceiling = config.min_edge  (same for BO3 and BO5 — the 1.5×
    BO5 widening was reverted; BO5 uses the unscaled min_edge ceiling)
    — i.e., arber's PRE-G3-buffer threshold. G3 buffer intentionally NOT included
    so the buffer's protection during BO3 G2 (state 1/2) remains intact: nothing
    fires in [base, base+g3_buf] band. series_edge_mult does NOT apply here:
    momentum sizing/firing is independent of the arber's series-vs-map
    edge-mult lever (2026-06-18 — was a copy-paste artifact from arber math).

  Trigger: |Δhedge| ≥ momentum_delta_hedge_min vs baseline cycle ≥ lookback_sec
  old, AND (momentum_require_hedge_driven=False OR |Δhedge| > |Δbest|).

  Execution: top-of-book IOC on each economically-equivalent book (cross-book),
  capped at momentum_volumes (or momentum_reducing_cap if the fire reduces
  |position|). No walking deeper levels — backtest used top-level only.

  Position safety: 1.0×max_position hard cap with 0.9× release. Momentum hard-
  stops at 1× (no edge exception, unlike the arber's hedge zone).

  Safety counters: per-direction per-series fire cap, per-direction throttle.

Reads from `series_eval` published by live_series_model (synthetic costs +
hedge profiles + state). Manager passes it via kwargs the same way the arber
and quoter do.
"""
import logging
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

import position_store
from framework_config import QuoterConfig
from models import QuoteSide
from arber_bot import get_loaded_cost_cents
from position_adjuster import compute_position_skew_shift_cents, variance_scale_at_theo

log = logging.getLogger(__name__)

# Hard sport whitelist — refusing to fire even if misconfigured.
ALLOWED_SPORT_PREFIXES = ("KXLOLGAME", "KXCS2GAME", "KXVALORANTGAME", "KXDOTA2GAME")


class MomentumBot:
    """Top-of-book cross-book IOC bot triggered by hedge-cost moves.

    Bound 1:1 with a series ticker (same as EsportsArberBot). Evaluates both
    directions (long-self via D1, long-opp via D2) per cycle and emits at most
    2 IOC orders per direction (one per book in the cross-set).
    """

    def __init__(self, config: QuoterConfig):
        self.config = config
        self.active = True

        # Hard sport guard — log once and stay disabled if misconfigured.
        if not any(self.config.ticker.startswith(p) for p in ALLOWED_SPORT_PREFIXES):
            log.warning(f"[MOMENTUM] {self.config.ticker}: sport not in whitelist "
                        f"{ALLOWED_SPORT_PREFIXES} — bot will refuse to fire")
            self._sport_ok = False
        else:
            self._sport_ok = True

        # Sliding window of past series_eval snapshots for Δhedge baseline.
        # Each entry: (ts, hedge_long, hedge_short, best_d1_price, best_d2_price)
        self._history: deque = deque(maxlen=200)

        # Per-direction throttle and per-event-direction fire counters.
        self._last_fire_ts: Dict[str, float] = {"d1": 0.0, "d2": 0.0}
        self._event_fires: Dict[str, int] = {"d1": 0, "d2": 0}
        # Track current event_base so we can reset event_fires on series change
        # (manager re-binds bots when the template changes; this is a defensive
        # secondary reset in case the bot somehow persists across events).
        self._current_event_base: Optional[str] = None

        # Reduce-only state: 1.0× max_position hard cap, 0.9× release
        # hysteresis. Set when |pos| ≥ 1.0×max; cleared when |pos| falls
        # below 0.9×max.
        self._reduce_only_active = False

        # Dedup for [MOMENTUM SKIP] warnings (one log per (dir, reason) per session)
        self._skip_warned: set = set()

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _ceiling(self, is_bo5: bool, d_trigger: str) -> float:
        """Edge ceiling = arber's pre-G3-buffer threshold for this direction.

        Note: series_edge_mult does NOT apply here. Momentum operates on
        the unscaled min_edge floor; the d_trigger arg is retained for
        signature symmetry with arber but is unused (2026-06-18).
        """
        # BO5 uses the SAME unscaled ceiling as BO3 — no 1.5× widening.
        # (is_bo5 retained for signature symmetry; no longer changes the band.)
        return self.config.min_edge

    def _skip_log(self, direction: str, reason: str, ctx: str = "") -> None:
        key = (direction, reason)
        if key in self._skip_warned:
            return
        self._skip_warned.add(key)
        log.info(f"[MOMENTUM SKIP] {self.config.ticker} dir={direction} reason={reason} "
                 f"(further skips for this reason silenced){' | ' + ctx if ctx else ''}")

    def _find_baseline(self, current_ts: float) -> Optional[tuple]:
        """Return the most recent snapshot ≥ lookback_sec old, or None."""
        lookback = self.config.momentum_lookback_sec
        for past in reversed(self._history):
            if current_ts - past[0] >= lookback:
                return past
        return None

    # ────────────────────────────────────────────────────────────────────
    # Main evaluation
    # ────────────────────────────────────────────────────────────────────

    def evaluate(self, market_state: Any, top_level_bid: float, top_level_offer: float,
                 bid_theo: float, offer_theo: float, **kwargs) -> List[Dict[str, Any]]:
        """Evaluate momentum triggers and return list of IOC orders to submit.

        Matches the signature of QuoterBot / EsportsArberBot so manager can
        dispatch without special-casing. Returns [] on any of:
          - bot inactive / sport guard failed
          - momentum disabled (momentum_delta_hedge_min ≤ 0 or volumes ≤ 0)
          - no series_eval available
          - state not eligible (decided / pre-game with no maps)
          - no baseline cycle ≥ lookback_sec old yet
          - no direction meets the gate
        """
        if not self.active or not self._sport_ok:
            return []

        cfg = self.config
        # "Disabled" check: either of these zero/empty means the bot is a no-op.
        if cfg.momentum_delta_hedge_min <= 0 or cfg.momentum_volumes <= 0:
            return []

        if cfg.stop_quoting:
            return []

        series_eval = kwargs.get("series_eval")
        full_market_state = kwargs.get("full_market_state") or {}
        full_top_bids = kwargs.get("full_top_bids") or {}
        full_top_offers = kwargs.get("full_top_offers") or {}

        if not series_eval or "hedge_profile_long" not in series_eval:
            return []

        state = series_eval.get("state")
        active_map = series_eval.get("active_map")
        is_bo5 = bool(series_eval.get("is_bo5", False))
        series_ticker = cfg.ticker  # defined early so the BO5-halt log below can reference it

        # Eligibility: BO3 needs state in {0,1,2} + active map; BO5 needs an
        # active map ticker. live_series_model already returns None for state
        # when the series is decided / pre-game-blank, so we just check it.
        if active_map is None:
            return []
        if not is_bo5 and state not in (0, 1, 2):
            return []

        # BO5 safety: halt if active map has no real Kalshi orderbook (defaults
        # to ask=100 / bid=0). Without live map data we'd be trading off static
        # p4/p5 stale assumptions — auto-resumes when Kalshi lists the ticker.
        if is_bo5:
            _am_ask = full_top_offers.get(active_map, 100.0)
            _am_bid = full_top_bids.get(active_map, 0.0)
            if _am_ask >= 99 and _am_bid <= 0:
                log.warning(
                    f"[BO5 HALT — NO KALSHI MAP] {series_ticker} state={state} "
                    f"active_map={active_map} ask={_am_ask:.1f} bid={_am_bid:.1f} "
                    f"— momentum refusing"
                )
                return []

        # Pull synthetic costs (already loaded with fees) from series_eval.
        syn_long = float(series_eval.get("synthetic_long_cost", 0.0))
        syn_short = float(series_eval.get("synthetic_short_cost", 0.0))
        hp_long = series_eval["hedge_profile_long"]
        hp_short = series_eval["hedge_profile_short"]
        hedge_long_cost = float(hp_long.get("synthetic_cost", 0.0))
        hedge_short_cost = float(hp_short.get("synthetic_cost", 0.0))

        # Series book — our team's series YES ask + opp's series YES ask
        # (which equals our team's NO ask economically).
        ticker_parts = series_ticker.split("-")
        if len(ticker_parts) < 3:
            return []
        series_base = "-".join(ticker_parts[:2])
        team_suffix = ticker_parts[-1]
        # Find opponent series ticker
        opp_series_ticker = None
        for k in full_top_offers:
            if isinstance(k, str) and k.startswith(series_base + "-") and k != series_ticker:
                opp_series_ticker = k
                break

        our_series_ask = full_top_offers.get(series_ticker, 100.0)
        our_series_bid = full_top_bids.get(series_ticker, 0.0)
        opp_series_ask = full_top_offers.get(opp_series_ticker, 100.0) if opp_series_ticker else 100.0
        opp_series_bid = full_top_bids.get(opp_series_ticker, 0.0) if opp_series_ticker else 0.0

        # D1 = long self (buy our YES OR buy opp NO); D2 = long opp.
        # best_d1_price = cheapest cost to go long self at top of book.
        # On Kalshi: opp NO ask = 100 - opp YES bid (since YES+NO=100).
        opp_no_ask_from_bid = max(0, 100 - opp_series_bid)
        best_d1_price = min(our_series_ask, opp_no_ask_from_bid)
        # best_d2_price = cheapest cost to go long opp (buy opp YES OR buy our NO).
        our_no_ask_from_bid = max(0, 100 - our_series_bid)
        best_d2_price = min(opp_series_ask, our_no_ask_from_bid)

        now = time.time()
        # Push current snapshot into history regardless of whether we fire.
        self._history.append((now, hedge_long_cost, hedge_short_cost,
                              best_d1_price, best_d2_price))

        # Reset event-fire counter if event_base changed (defensive).
        if self._current_event_base != series_base:
            self._current_event_base = series_base
            self._event_fires = {"d1": 0, "d2": 0}

        # Reduce-only state: momentum hard-stops at 1.0× max_position (no edge
        # exception, unlike the arber's hedge zone). 0.9× release hysteresis.
        current_pos = position_store.get_position(series_ticker) or 0
        hard_cap = int(1.0 * cfg.max_position)
        release_cap = int(0.9 * cfg.max_position)
        abs_pos = abs(current_pos)
        if self._reduce_only_active:
            if abs_pos < release_cap:
                self._reduce_only_active = False
                log.warning(f"[MOMENTUM STOP-CAP] {series_ticker}: |pos|={abs_pos} "
                            f"< 0.9×max={release_cap} — released")
        elif abs_pos >= hard_cap:
            self._reduce_only_active = True
            log.warning(f"[MOMENTUM STOP-CAP] {series_ticker}: |pos|={abs_pos} "
                        f"≥ 1.0×max={hard_cap} — reduce-only mode")

        # Need a baseline before we can compute deltas.
        baseline = self._find_baseline(now)
        if baseline is None:
            return []
        _, base_hedge_long, base_hedge_short, base_best_d1, base_best_d2 = baseline

        # Build candidate fires per direction
        desired_quotes: List[Dict[str, Any]] = []
        # For each direction, decide: gates pass → top-level cross-book sweep.
        d1_orders = self._maybe_fire_direction(
            direction="d1",
            full_market_state=full_market_state,
            current_pos=current_pos,
            edge=100.0 - syn_long,                # synthetic_long_cost is fees-loaded
            hedge_now=hedge_long_cost, hedge_base=base_hedge_long,
            best_now=best_d1_price, best_base=base_best_d1,
            primary_ticker=series_ticker, primary_side="yes",
            primary_top_price=int(round(our_series_ask)),
            # Buying YES at ask P → take NO-bid liquidity at (100-P). Kalshi's
            # yes_dollars=YES bids, no_dollars=NO bids; sell-side YES sits in
            # no_dollars at the complement price.
            primary_top_size=self._get_top_size(full_market_state, series_ticker, "no_dollars", 100 - int(round(our_series_ask))),
            mirror_ticker=opp_series_ticker, mirror_side="no",
            mirror_top_price=int(round(opp_no_ask_from_bid)),
            # Buying NO at ask P → take YES-bid liquidity at (100-P).
            mirror_top_size=self._get_top_size(full_market_state, opp_series_ticker, "yes_dollars", 100 - int(round(opp_no_ask_from_bid))) if opp_series_ticker else 0,
            shared_hedge_cost=float(hp_long.get("synthetic_cost", 0.0)),
            is_bo5=is_bo5, state=state, series_eval=series_eval,
            now=now,
        )
        d2_orders = self._maybe_fire_direction(
            direction="d2",
            full_market_state=full_market_state,
            current_pos=current_pos,
            edge=100.0 - syn_short,
            hedge_now=hedge_short_cost, hedge_base=base_hedge_short,
            best_now=best_d2_price, best_base=base_best_d2,
            primary_ticker=series_ticker, primary_side="no",
            primary_top_price=int(round(our_no_ask_from_bid)),
            # Buying NO at ask P → take YES-bid liquidity at (100-P).
            primary_top_size=self._get_top_size(full_market_state, series_ticker, "yes_dollars", 100 - int(round(our_no_ask_from_bid))),
            mirror_ticker=opp_series_ticker, mirror_side="yes",
            mirror_top_price=int(round(opp_series_ask)),
            # Buying YES at ask P → take NO-bid liquidity at (100-P).
            mirror_top_size=self._get_top_size(full_market_state, opp_series_ticker, "no_dollars", 100 - int(round(opp_series_ask))) if opp_series_ticker else 0,
            shared_hedge_cost=float(hp_short.get("synthetic_cost", 0.0)),
            is_bo5=is_bo5, state=state, series_eval=series_eval,
            now=now,
        )
        desired_quotes.extend(d1_orders)
        desired_quotes.extend(d2_orders)
        return desired_quotes

    # ────────────────────────────────────────────────────────────────────
    # Per-direction fire decision
    # ────────────────────────────────────────────────────────────────────

    def _maybe_fire_direction(self, *,
                              direction: str,
                              full_market_state: dict,
                              current_pos: int,
                              edge: float,
                              hedge_now: float, hedge_base: float,
                              best_now: float, best_base: float,
                              primary_ticker: str, primary_side: str,
                              primary_top_price: int, primary_top_size: int,
                              mirror_ticker: Optional[str], mirror_side: str,
                              mirror_top_price: int, mirror_top_size: int,
                              shared_hedge_cost: float,
                              is_bo5: bool, state: Any,
                              series_eval: dict, now: float) -> List[Dict[str, Any]]:
        cfg = self.config

        # Throttle check
        if now - self._last_fire_ts[direction] < cfg.momentum_throttle_sec:
            return []
        # Event-direction fire cap
        if self._event_fires[direction] >= cfg.momentum_event_cap_per_dir:
            self._skip_log(direction, "event_cap")
            return []

        # Edge band gate (use raw edge from synthetic, NOT including G3 buffer)
        # d_trigger inferred as "map" since momentum fires on map-driven hedge moves
        ceiling = self._ceiling(is_bo5=is_bo5, d_trigger="map")
        # ── Position skew (2026-07-25): momentum now respects the SAME per-price
        # variance-scaled position adjuster the arber/quoter use. RAISE the edge
        # floor to brake ADDS as |position| grows; LOWER it to ease REDUCES. This
        # honors the skew_* fields already on every MOM row (0/1.0/3.0); with
        # skew_max_shift_cents=0 the skew is 0.0 and floor == the legacy
        # momentum_edge_floor (exact no-op). direction_sign/is_reducing are also
        # used below for the reduce-only + cap logic; hoisted here so the skew can
        # pick the add-vs-reduce sign.
        direction_sign = 1 if direction == "d1" else -1
        is_reducing = (direction_sign * current_pos) < 0
        skew_c = (compute_position_skew_shift_cents(current_pos, cfg)
                  * variance_scale_at_theo(primary_top_price))
        # ── Taker retreat knob (TRADE_RETREAT_SCOPE.md, 26AUG18) ──
        # retreat_cap_cents > 0 on THIS momentum row makes the floor honor the
        # recency-flow theo shift: SIGNED per direction (+ = this direction is
        # the recently-accumulated side, edge overstated), applied the same for
        # add and reduce (a theo shift is add/reduce-agnostic, unlike skew_c).
        # All MOM rows ship with retreat_cap_cents=0 → retreat_c=0.0, exact
        # no-op. Direction axis: f is toward cfg.ticker's own team = d1.
        retreat_c = 0.0
        _r_cap = getattr(cfg, "retreat_cap_cents", 0.0) or 0.0
        if _r_cap > 0:
            _r_parts = cfg.ticker.split("-")
            if len(_r_parts) == 3:
                try:
                    import retreat_shadow
                    _rc, _rf0, _rfcap, _rhl, _rtk = retreat_shadow.params_from_conf(cfg)
                    _r_f = retreat_shadow.get_f(
                        "-".join(_r_parts[:2]), _r_parts[-1], cfg.max_position,
                        half_life=_rhl, include_takers=_rtk)
                    _r_mag = retreat_shadow.shift_cents(
                        abs(_r_f), cap_cents=_rc, f0=_rf0, f_cap=_rfcap)
                    retreat_c = (_r_mag if _r_f > 0 else -_r_mag) * direction_sign
                    if abs(retreat_c) > 0.05:
                        log.info(f"[RETREAT-TAKER] {cfg.ticker} MOM dir={direction} "
                                 f"f={_r_f:+.3f} floor_shift={retreat_c:+.2f}c")
                except Exception:
                    # Loud by design — never silently demote to no-retreat.
                    log.exception(f"[RETREAT-TAKER] MOM shift computation FAILED "
                                  f"for {cfg.ticker} — floor running WITHOUT retreat")
        floor = cfg.momentum_edge_floor + (-skew_c if is_reducing else skew_c) + retreat_c
        if edge < floor:
            return []  # Below (skew-adjusted) floor — no log (very common)
        if edge > ceiling:
            return []  # Above ceiling — regular arber's job

        # Δhedge gate
        delta_hedge = hedge_now - hedge_base
        delta_best = best_now - best_base
        if abs(delta_hedge) < cfg.momentum_delta_hedge_min:
            return []
        if cfg.momentum_require_hedge_driven and abs(delta_hedge) <= abs(delta_best):
            return []  # Series-book-driven move, not map-hedge-driven

        # Reduce-only mode check (1.0× cap breached)
        # direction_sign: +1 for d1 (long self), -1 for d2 (long opp) — hoisted above.
        if self._reduce_only_active:
            # Only allow fires that reduce |position|.
            projected_pos = current_pos + direction_sign  # nominal: would this push toward 0?
            if abs(projected_pos) >= abs(current_pos):
                self._skip_log(direction, "reduce_only_mode")
                return []

        # is_reducing (does THIS fire reduce |position| → unlocks larger cap) was
        # computed with the skew gate above.

        # ── Hedge-CONFIRMS-direction gate (wrong-direction fire fix, 2026-07-18) ──
        # The bought side's fair value = (100 - hedge_now)  [see arb_theo_yes below].
        # So Δhedge = hedge_now - hedge_base moves OPPOSITE to that fair value:
        #   Δhedge < 0  → bought side's fair value ROSE → team improving → CONFIRMS a long.
        #   Δhedge > 0  → bought side's fair value FELL → team worsening → OPPOSES the long.
        # The magnitude gate above (abs(delta_hedge)) is sign-blind, so a hedge move
        # that marks the team DOWN can still open a long whenever a residual synthetic
        # edge survives (series overshot, or — worse — a phantom/pregame hedge revert).
        # Empirically 5.3% of opening fires (41k lots, EFKC among them; 67% with the
        # market ALSO marking the team down) were this adverse-selection pattern.
        # Only gate OPENING fires — a reducing fire that closes into an adverse move
        # is desirable. Kill switch: momentum_require_hedge_confirms=false per row.
        if (cfg.momentum_require_hedge_confirms
                and not is_reducing
                and delta_hedge >= 0.0):
            self._skip_log(direction, "hedge_opposes_direction")
            return []

        cap = cfg.momentum_reducing_cap if (is_reducing and cfg.momentum_reducing_cap > 0) else cfg.momentum_volumes
        if cap <= 0:
            return []

        # HARD position bound: after this fire, |position| must never exceed
        # 1.0×max_position — in EITHER direction. Momentum hard-stops at 1× (no
        # edge exception). This applies to reducing fires too: a large reducing
        # fire (reducing_cap can exceed |pos|) could otherwise overshoot through
        # zero and build an oversized position on the opposite side, blowing past
        # the cap. Cap by SIGNED room:
        #   new_pos = current_pos + direction_sign * size ;  |new_pos| ≤ hard_cap
        #   d1 (sign +1): size ≤ hard_cap - current_pos
        #   d2 (sign -1): size ≤ hard_cap + current_pos
        #   unified:      size ≤ hard_cap - direction_sign * current_pos
        hard_cap = int(1.0 * cfg.max_position)
        room = max(0, hard_cap - direction_sign * current_pos)
        if room <= 0:
            self._skip_log(direction, "position_cap_hit")
            return []
        cap = min(cap, room)

        # Build the FULL cross-book ladder. Both books carry "go-long-target"
        # liquidity at every level — not just at the top. Bug 2026-06-27:
        # previously only the top of each book was considered, which would
        # skip cheaper deeper levels on the primary book in favor of the top
        # of the mirror book when those deeper levels were cheaper. Example:
        # primary NO_DFAM stack 69c/70c/71c (huge sizes), mirror YES_MEN top
        # 72c. We'd lift only 69c primary then jump to 72c mirror, leaving
        # the 70c/71c primary depth untouched.
        #
        # The semantics:
        #   primary_side="yes": BUY YES → cost = (100 - no_p) for each NO bid
        #   primary_side="no":  BUY NO  → cost = (100 - yes_p) for each YES bid
        # (mirror leg is analogous; both legs go LONG the target team.)
        def _book_levels(ticker, side_key):
            if not ticker:
                return []
            mkt = full_market_state.get(ticker, {})
            ob = mkt.get("raw_ob", {}).get("orderbook_fp", {})
            out = []
            for pt in ob.get(side_key, []):
                try:
                    p = int(round(float(pt[0]) * 100))
                    s = int(float(pt[1]))
                    if s > 0:
                        out.append((p, s))
                except (IndexError, ValueError, TypeError):
                    continue
            return out

        merged: List[tuple] = []  # (cost_c, size, ticker, side)
        # Primary leg
        if primary_side == "yes":
            # BUY YES → consume no_dollars (NO bids). Each NO bid at no_p
            # is a YES ask at (100 - no_p).
            for no_p, sz in _book_levels(primary_ticker, "no_dollars"):
                merged.append((100 - no_p, sz, primary_ticker, "yes"))
        elif primary_side == "no":
            # BUY NO → consume yes_dollars (YES bids).
            for yes_p, sz in _book_levels(primary_ticker, "yes_dollars"):
                merged.append((100 - yes_p, sz, primary_ticker, "no"))
        # Mirror leg (same semantics, different book)
        if mirror_ticker:
            if mirror_side == "yes":
                for no_p, sz in _book_levels(mirror_ticker, "no_dollars"):
                    merged.append((100 - no_p, sz, mirror_ticker, "yes"))
            elif mirror_side == "no":
                for yes_p, sz in _book_levels(mirror_ticker, "yes_dollars"):
                    merged.append((100 - yes_p, sz, mirror_ticker, "no"))

        # Drop degenerate prices (0/100 = no actionable liquidity).
        merged = [m for m in merged if 0 < m[0] < 100]
        if not merged:
            self._skip_log(direction, "empty_book")
            return []
        merged.sort(key=lambda x: (x[0], -x[1]))  # cheapest first, larger tier first on ties

        # Generate one shared sweep_id so cross-leg fills aggregate in trade_logger.
        sweep_id = str(uuid.uuid4())
        # Sweep cheapest-first. Accumulate per-(ticker, side) → emit ONE IOC
        # per (ticker, side) at the worst (highest) cost we crossed. This
        # matches the arber cross-book pattern.
        per_leg: dict = {}  # (ticker, side) -> {"vol": int, "worst_cost": int}
        cap_remaining = cap
        for cost, size_avail, lg_ticker, lg_side in merged:
            if cap_remaining <= 0:
                break
            # Edge gate at this level using shared hedge cost
            loaded = get_loaded_cost_cents(float(cost))
            edge_here = 100.0 - loaded - shared_hedge_cost
            if edge_here < floor:
                break  # remaining levels are even worse
            take = min(size_avail, cap_remaining)
            if take <= 0:
                continue
            key = (lg_ticker, lg_side)
            slot = per_leg.setdefault(key, {"vol": 0, "worst_cost": 0})
            slot["vol"] += int(take)
            if cost > slot["worst_cost"]:
                slot["worst_cost"] = cost
            cap_remaining -= take

        if not per_leg:
            self._skip_log(direction, "edge_at_book_below_floor")
            return []

        # Build the orders. theo (in YES terms) per leg:
        #   side=no  → arb_theo_yes = shared_hedge_cost (NO-flip)
        #   side=yes → arb_theo_yes = 100 - shared_hedge_cost
        orders: List[Dict[str, Any]] = []
        for (lg_ticker, lg_side), slot in per_leg.items():
            arb_theo_yes = shared_hedge_cost if lg_side == "no" else 100.0 - shared_hedge_cost
            orders.append({
                "side": QuoteSide.BID,
                "ticker": lg_ticker,
                "kalshi_side": lg_side,
                "size": slot["vol"],
                "limit_cents": int(slot["worst_cost"]),
                "time_in_force": "immediate_or_cancel",
                "_raw_theo": arb_theo_yes,
                "_adj_theo": arb_theo_yes,
                "_was_size_capped": (cap_remaining <= 0),
                "_sweep_id": sweep_id,
                "_sweep_total_vol": 0,  # set below after summing
                "_trigger_type": "momentum",
            })

        if not orders:
            self._skip_log(direction, "edge_at_book_below_floor")
            return []

        # Backfill sweep_total_vol on each order so trade_logger aggregates correctly.
        sweep_total = sum(o["size"] for o in orders)
        for o in orders:
            o["_sweep_total_vol"] = sweep_total

        # Mark fire and log.
        self._last_fire_ts[direction] = now
        self._event_fires[direction] += 1
        log.info(
            f"[MOMENTUM FIRE] {self.config.ticker} dir={direction.upper()} "
            f"edge={edge:+.2f}c ceiling={ceiling:.2f}c floor={floor:.2f}c "
            f"Δhedge={delta_hedge:+.2f}c Δbest={delta_best:+.2f}c "
            f"baseline_age={(now - (now - cfg.momentum_lookback_sec)):.1f}s "
            f"is_reducing={is_reducing} cap={cap} "
            f"fires_dir={self._event_fires[direction]}/{cfg.momentum_event_cap_per_dir} "
            f"orders={[(o['ticker'], o['kalshi_side'], o['size'], o['limit_cents']) for o in orders]}"
        )
        return orders

    # ────────────────────────────────────────────────────────────────────
    # Misc helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_top_size(full_market_state: dict, ticker: Optional[str], side_key: str,
                     target_price: int) -> int:
        """Look up the top-level size in cents at target_price on the given side
        of the raw orderbook. Returns 0 if not present."""
        if not ticker:
            return 0
        mkt = full_market_state.get(ticker, {})
        ob = mkt.get("raw_ob", {}).get("orderbook_fp", {})
        pts = ob.get(side_key, [])
        # pts is a list of [price_dollars, size] for the YES/NO side
        for pt in pts:
            try:
                p = float(pt[0])
                # Convert dollar price to cents and compare
                if int(round(p * 100)) == target_price:
                    return int(float(pt[1]))
            except (IndexError, ValueError, TypeError):
                continue
        return 0
