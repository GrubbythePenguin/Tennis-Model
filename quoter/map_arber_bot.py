"""Map-arber: lead-lag arb where SERIES is the leading book and the live MAP
is the lagging instrument we trade.

Inverse of arber_bot.py. Used for esports where series volume dominates map
volume (CS2: ~5–7× series vs maps), making series the price-discovery
instrument and maps the lagging consumer.

Same scaffolding as series-arber via ArberBotBase. Differences:
  - Trade leg = active_map; hedge leg = series.
  - Position cap is on the active map ticker, not the series ticker.
  - Fake-move detection watches the series book (the leading instrument).
  - State 3 (1-1, no live map) is skipped — no separate G3 map market in BO3.
  - No tennis / no BO5 / no momentum strategy.
  - Reads pre-computed map-side hedge profiles from series_eval (live_series_model).

The bot is registered against the SERIES ticker (same as series-arber) so
the alphabetical-dedup rule naturally inherits — only the alphabetically
earlier team's bot evaluates each event_base.

Env var MAP_ARBER_NO_FORFEIT=1 disables forfeit protection for this bot
(intended for dry-run observation only). With the flag set:
  - The inline evaluate_forfeit check is skipped.
  - Inline detect_bo3_state fallback is called with series_base=None, which
    bypasses the secondary forfeit gate inside that function.
  - The series_eval path still honors live_series_model's own forfeit gate
    (so if live_series suppressed this ticker for forfeit, the bot drops
    naturally to the inline-no-forfeit path).
"""
import logging
import os
import time
from typing import Any, Dict, List

import position_store
import hedge_staleness
from arber_bot_base import (
    ArberBotBase,
    calculate_cross_book_sweep,
)
from hedge_engine import (
    get_loaded_cost_cents,
    detect_bo3_state,
    evaluate_forfeit,
    parse_series_ticker,
    compute_map_hedge_profiles,
    compute_map_synthetic_costs,
    get_cross_book_series_prices,
)
from models import QuoteSide

log = logging.getLogger(__name__)

DEBUG_MODE = False
NO_FORFEIT = os.environ.get("MAP_ARBER_NO_FORFEIT", "0") == "1"


class EsportsMapArberBot(ArberBotBase):
    """Lead-lag arber that trades the live map using series as the leading signal."""

    def evaluate(
        self,
        market_state: Any,
        top_level_bid: float,
        top_level_offer: float,
        bid_theo: float,
        offer_theo: float,
        full_market_state: Dict[str, Any] = None,
        full_top_bids: Dict[str, float] = None,
        full_top_offers: Dict[str, float] = None,
        series_eval: Dict[str, Any] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:

        # Probability hot-reload retry (mirror of series-arber)
        if not self.active and not self.baseline_loaded:
            import time
            now = time.time()
            if now - self._last_prob_retry >= 600:
                self._last_prob_retry = now
                self._load_pre_game_probabilities()
                if self.baseline_loaded:
                    self.active = True
                    log.info(f"[MAP-ARBER REVIVED] {self.config.ticker} | "
                             f"Probabilities found on retry — reactivating.")

        if not self.active or not full_top_bids or not full_top_offers:
            return []

        # config.ticker is the series ticker (same registration as arber_bot)
        series_ticker = self.config.ticker
        if "GAME" not in series_ticker:
            # Map-arber only supports GAME (esports BO3) for now. Tennis MATCH
            # markets would need a different decomposition (set-arb already
            # exists in arber_bot; not duplicated here).
            return []

        parsed = parse_series_ticker(series_ticker)
        if not parsed:
            return []
        series_base, team_suffix, map_base = parsed

        # Forfeit detection — share oracle with series-arber
        series_ask = full_top_offers.get(series_ticker, 100.0)
        series_bid = full_top_bids.get(series_ticker, 0.0)
        series_no_ask = 100.0 - series_bid

        map1_ticker = f"{map_base}-1-{team_suffix}"
        map2_ticker = f"{map_base}-2-{team_suffix}"

        if not NO_FORFEIT:
            forfeited, forfeit_reason = evaluate_forfeit(
                series_base, map1_ticker, map2_ticker,
                full_top_bids, full_top_offers,
                series_ask=series_ask, series_no_ask=series_no_ask,
                data_source=self.config.data_source,
            )
            if forfeited:
                if DEBUG_MODE:
                    log.info(f"[MAP-ARBER FORFEIT] {series_ticker} reason={forfeit_reason}")
                return []

        # ── State + active map ──
        # Prefer series_eval (Phase 5c path); fall back to inline detect.
        if series_eval is not None and "state" in series_eval:
            state = series_eval["state"]
            active_map_ticker = series_eval["active_map"]
            active_map_opp = series_eval.get("active_map_opp")
        else:
            # NO_FORFEIT: pass series_base=None so detect_bo3_state's internal
            # forfeit gate is bypassed (evaluate_forfeit only runs when
            # series_base is supplied). All other state-detection logic
            # (settlements, loose-decided rules, BO3 endgame fog) still applies.
            sb_for_detect = None if NO_FORFEIT else series_base
            state_info = detect_bo3_state(
                map_base, team_suffix, full_market_state,
                full_top_bids, full_top_offers, series_base=sb_for_detect,
                data_source=self.config.data_source,
            )
            if state_info is None:
                return []
            state = state_info["state"]
            active_map_ticker = state_info["active_map"]
            active_map_opp = state_info.get("active_map_opp")

        # State 3 (1-1) has no separate live map market — skip
        if state == 3 or active_map_ticker is None:
            return []

        # Find opp series ticker and apply alphabetical dedup at the series
        # level (mirror of arber_bot — only earlier-suffix bot runs).
        opp_series_ticker = None
        for k in (full_top_bids or {}):
            if k.startswith(series_base + "-") and k != series_ticker:
                opp_series_ticker = k
                break
        # NOTE: alphabetical dedup moved BELOW hedge_staleness.update_snapshot
        # so both teams contribute to the freeze detector. See arber_bot.py
        # for the parallel fix (2026-06-19 MENGRIND incident).
        opp_suffix = opp_series_ticker.split("-")[-1] if opp_series_ticker else ""

        # ── Compute / fetch map-side hedge profiles ──
        if (series_eval is not None and "map_hedge_profile_long" in series_eval
                and series_eval.get("map_hedge_profile_long") is not None):
            map_hp1 = series_eval["map_hedge_profile_long"]
            map_hp2 = series_eval["map_hedge_profile_short"]
            cons_series_yes_ask = series_eval["cons_series_yes_ask"]
            cons_series_no_ask = series_eval["cons_series_no_ask"]
        else:
            # Fallback: inline compute against current snapshot
            cons_series_no_ask, cons_series_yes_ask = get_cross_book_series_prices(
                series_ticker, opp_series_ticker or "", full_top_bids, full_top_offers,
            ) if opp_series_ticker else (series_no_ask, series_ask)
            map_hp1, map_hp2 = compute_map_hedge_profiles(
                state,
                self.p1_start, self.p2_start, self.series_start,
                self.g2_momentum, self.g3_momentum,
                series_no_ask=cons_series_no_ask,
                series_yes_ask=cons_series_yes_ask,
            )

        # ── Active-map prices and books ──
        active_map_bid = full_top_bids.get(active_map_ticker, 0.0)
        active_map_ask = full_top_offers.get(active_map_ticker, 100.0)
        active_map_no_ask = 100.0 - active_map_bid

        # BO5 safety: halt if active map has no real Kalshi orderbook (defaults
        # to ask=100 / bid=0). Implicit gates (< 99) below also block fires —
        # this explicit halt adds a grep'able log and consistency with arber/quoter.
        is_bo5 = isinstance(state, tuple)
        if is_bo5 and active_map_ask >= 99 and active_map_bid <= 0:
            log.warning(
                f"[BO5 HALT — NO KALSHI MAP] {series_ticker} state={state} "
                f"active_map={active_map_ticker} ask={active_map_ask:.1f} "
                f"bid={active_map_bid:.1f} — map_arber refusing to trade"
            )
            return []

        opp_map_bid = full_top_bids.get(active_map_opp, 0.0) if active_map_opp else 0.0
        opp_map_ask = full_top_offers.get(active_map_opp, 100.0) if active_map_opp else 100.0

        # Cross-book best prices on the trade leg (= active map). Mirror of
        # arber's cross-book series prices, but the trade leg is the map.
        best_d1_price = int(min(round(active_map_ask), round(100.0 - opp_map_bid)))  # long map_YES
        best_d2_price = int(min(round(active_map_no_ask), round(opp_map_ask)))       # long map_NO

        # ── Position cap on the active map ticker ──
        cap_filter, _, _, _ = self._build_position_cap_filter(active_map_ticker)
        net_map_pos = position_store.get_position(active_map_ticker) or 0

        # ── Forge synthetic costs for snapshot caching ──
        h1_cost = map_hp1["synthetic_cost"]
        h2_cost = map_hp2["synthetic_cost"]
        h1_tenths = int(float(f"{h1_cost:.1f}") * 10)
        h2_tenths = int(float(f"{h2_cost:.1f}") * 10)
        d1_snap = (best_d1_price, h1_tenths)
        d2_snap = (best_d2_price, h2_tenths)

        # Per-event hedge-freeze detector (see hedge_staleness.py). Each per-team
        # arber feeds its OWN team's snapshot; both teams together populate
        # event-level state. If flagged, suppress all fires this cycle.
        _now_ts = time.time()
        hedge_staleness.update_snapshot(series_base, team_suffix, _now_ts, best_d1_price, h1_cost)
        if hedge_staleness.is_stale(series_base):
            return []

        # Alphabetical dedup — moved here after update_snapshot so both teams
        # feed the staleness detector (see note above the opp_suffix calc).
        if opp_suffix and team_suffix > opp_suffix:
            return []

        d1_improved, d1_trigger = self._check_d_improved(d1_snap, "_last_d1", state)
        d2_improved, d2_trigger = self._check_d_improved(d2_snap, "_last_d2", state)
        prev_d1 = self._last_d1
        prev_d2 = self._last_d2
        self._last_d1 = d1_snap
        self._last_d2 = d2_snap

        # ── Watched-book fake-move detection on SERIES (the leading book) ──
        if self._is_fake_move(series_bid, series_ask):
            return []

        # ── Status gating ──
        series_status = full_market_state.get(series_ticker, {}).get("status", "")
        active_map_status = full_market_state.get(active_map_ticker, {}).get("status", "")
        can_trade = (series_status == "active" and active_map_status == "active")
        if not can_trade:
            return []

        # ── Sizing knobs ──
        base_vol = self.config.volumes[0] if self.config.volumes else 100
        fire_cap = getattr(self.config, "max_fire_size", 500)
        scale_step = getattr(self.config, "arb_scale_step_cents", 1.0)

        desired_quotes: List[Dict[str, Any]] = []

        # ── Direction 1: long active map_YES (our team wins this map) ──
        # Sweep: our map YES book + opp map NO book (both = "we win this map")
        edge_1_approx = 100.0 - (get_loaded_cost_cents(best_d1_price) + h1_cost)
        if (active_map_no_ask < 99 and d1_improved
                and self._check_fire_throttle(edge_1_approx)):
            map_yes_pts = (full_market_state.get(active_map_ticker, {})
                           .get("raw_ob", {}).get("orderbook_fp", {})
                           .get("no_dollars", []))
            opp_map_no_pts = []
            if active_map_opp:
                opp_map_no_pts = (full_market_state.get(active_map_opp, {})
                                  .get("raw_ob", {}).get("orderbook_fp", {})
                                  .get("yes_dollars", []))
            books_1 = [(map_yes_pts, active_map_ticker, "yes")]
            if active_map_opp and opp_map_no_pts:
                books_1.append((opp_map_no_pts, active_map_opp, "no"))

            cross_orders_1 = calculate_cross_book_sweep(
                books=books_1,
                hedge_cost=h1_cost,
                min_edge=self.config.min_edge,
                scale_step_c=scale_step,
                base_vol=base_vol,
                max_fire_size=fire_cap,
                max_position=self.config.max_position,
                current_pos=net_map_pos,
                direction_sign=1,
                min_absolute_edge=self.config.min_absolute_edge,
                fav_edge_k=self.config.fav_edge_k,
            )
            total_1 = sum(o["size"] for o in cross_orders_1)
            if total_1 > 0:
                self._record_fire(edge_1_approx)
            for order in cross_orders_1:
                order["_trigger_type"] = d1_trigger
                desired_quotes.append(order)
                log.info(f"MAP-LEAD ARBITRAGE [{d1_trigger.upper()}] (S{state}): "
                         f"{order['ticker']} {order['kalshi_side'].upper()} "
                         f"{order['size']}x at {order['limit_cents']}c "
                         f"[hedge={h1_cost:.1f}c series={cons_series_yes_ask:.1f}/{cons_series_no_ask:.1f} "
                         f"sweep={total_1}]")

        # ── Direction 2: long active map_NO (opp wins this map) ──
        edge_2_approx = 100.0 - (get_loaded_cost_cents(best_d2_price) + h2_cost)
        if (active_map_ask < 99 and d2_improved
                and self._check_fire_throttle(edge_2_approx)):
            map_no_pts = (full_market_state.get(active_map_ticker, {})
                          .get("raw_ob", {}).get("orderbook_fp", {})
                          .get("yes_dollars", []))
            opp_map_yes_pts = []
            if active_map_opp:
                opp_map_yes_pts = (full_market_state.get(active_map_opp, {})
                                   .get("raw_ob", {}).get("orderbook_fp", {})
                                   .get("no_dollars", []))
            books_2 = [(map_no_pts, active_map_ticker, "no")]
            if active_map_opp and opp_map_yes_pts:
                books_2.append((opp_map_yes_pts, active_map_opp, "yes"))

            cross_orders_2 = calculate_cross_book_sweep(
                books=books_2,
                hedge_cost=h2_cost,
                min_edge=self.config.min_edge,
                scale_step_c=scale_step,
                base_vol=base_vol,
                max_fire_size=fire_cap,
                max_position=self.config.max_position,
                current_pos=net_map_pos,
                direction_sign=-1,
                min_absolute_edge=self.config.min_absolute_edge,
                fav_edge_k=self.config.fav_edge_k,
            )
            total_2 = sum(o["size"] for o in cross_orders_2)
            if total_2 > 0:
                self._record_fire(edge_2_approx)
            for order in cross_orders_2:
                order["_trigger_type"] = d2_trigger
                desired_quotes.append(order)
                log.info(f"MAP-LEAD ARBITRAGE [{d2_trigger.upper()}] (S{state}): "
                         f"{order['ticker']} {order['kalshi_side'].upper()} "
                         f"{order['size']}x at {order['limit_cents']}c "
                         f"[hedge={h2_cost:.1f}c series={cons_series_yes_ask:.1f}/{cons_series_no_ask:.1f} "
                         f"sweep={total_2}]")

        # ── Per-cycle context log ──
        opp_label = active_map_opp.split("-")[-1] if active_map_opp else "?"
        best_d1_cost = get_loaded_cost_cents(best_d1_price) + h1_cost
        best_d2_cost = get_loaded_cost_cents(best_d2_price) + h2_cost
        edge_1 = 100.0 - best_d1_cost if best_d1_cost < 199.0 else None
        edge_2 = 100.0 - best_d2_cost if best_d2_cost < 199.0 else None
        d1_mark = "" if d1_improved else "="
        d2_mark = "" if d2_improved else "="
        e1_str = (f"Long {team_suffix}-map{d1_mark}: Best={best_d1_price:5.0f} + "
                  f"Hedge={h1_cost:5.1f} = {best_d1_cost:5.1f} Edge={edge_1:>+5.1f}"
                  if edge_1 is not None else "")
        e2_str = (f"Long {opp_label}-map{d2_mark}: Best={best_d2_price:5.0f} + "
                  f"Hedge={h2_cost:5.1f} = {best_d2_cost:5.1f} Edge={edge_2:>+5.1f}"
                  if edge_2 is not None else "")
        threshold = max(self.config.min_edge, self.config.min_absolute_edge, 0.0)
        ctx = (f"[MAP-LEAD] {series_ticker} active={active_map_ticker} (S{state}) | "
               f"Pos:{net_map_pos:>+5d}/{self.config.max_position} | "
               f"{e1_str} | {e2_str} | Thresh:{threshold:.0f}")
        for q in desired_quotes:
            q["_arb_context"] = ctx
        log.info(ctx)

        return cap_filter(desired_quotes)
