"""
Delta Hedge Bot

When the quoter gets filled on resting series orders (maker fills),
this bot immediately fires an IOC hedge on the active map to lock in edge.

The hedge ratio and price come from the arber's hedge math.
State detection is inherited from the arber's full logic.
Triggered by maker fills only — taker fills (arber IOCs) are already hedged.
"""
import logging
from typing import Any, List, Dict

from framework_config import QuoterConfig
from models import QuoteSide
import position_store
from hedge_engine import (
    get_loaded_cost_cents,
    detect_bo3_state,
    compute_hedge_profiles,
    parse_series_ticker,
    load_probabilities,
)

log = logging.getLogger(__name__)

# Hard ceiling on hedge fire price (cents). When our model is badly mispriced,
# the hedge-cost check can rationalize taking 95–99c asks, locking in losses.
# Stop walking the book once asks exceed this.
MAX_FIRE_PRICE_C = 90


class HedgeBot:
    def __init__(self, config: QuoterConfig):
        self.config = config
        self.active = True
        probs = load_probabilities().get(config.ticker)
        if probs:
            self.p1_start = probs["p1"]
            self.p2_start = probs["p2"]
            self.series_start = probs["series"]
            self.g2_momentum = probs.get("g2_momentum", 0.0)
            self.g3_momentum = probs.get("g3_momentum", 0.0)
            self.is_bo5 = probs.get("bo5", False)
            self.p3_start = probs.get("p3", (probs["p1"] + probs["p2"]) / 2.0)
        else:
            self.p1_start = self.p2_start = self.series_start = 0.5
            self.g2_momentum = self.g3_momentum = 0.0
            self.is_bo5 = False
            self.p3_start = 0.5
            self.active = False

    def refresh_probabilities(self, new_probs: dict = None) -> bool:
        """Update cached probs in-place. Safe on transient miss."""
        if new_probs is None:
            new_probs = load_probabilities()
        probs = new_probs.get(self.config.ticker)
        if not probs:
            return False
        self.p1_start = probs["p1"]
        self.p2_start = probs["p2"]
        self.series_start = probs["series"]
        self.g2_momentum = probs.get("g2_momentum", 0.0)
        self.g3_momentum = probs.get("g3_momentum", 0.0)
        self.is_bo5 = probs.get("bo5", False)
        self.p3_start = probs.get("p3", (probs["p1"] + probs["p2"]) / 2.0)
        if not self.active:
            self.active = True
            log.info(f"[HEDGER] {self.config.ticker} | RELOAD activated: "
                     f"p1={probs['p1']*100:.1f}% p2={probs['p2']*100:.1f}% "
                     f"series={probs['series']*100:.1f}%")
        else:
            log.info(f"[HEDGER] {self.config.ticker} | RELOAD: "
                     f"p1={probs['p1']*100:.1f}% p2={probs['p2']*100:.1f}% "
                     f"series={probs['series']*100:.1f}%")
        return True

    def evaluate(self, market_state: Any, top_level_bid: float, top_level_offer: float,
                 bid_theo: float, offer_theo: float,
                 full_market_state: Dict[str, Any] = None,
                 full_top_bids: Dict[str, float] = None,
                 full_top_offers: Dict[str, float] = None,
                 series_eval: Dict[str, Any] = None,
                 **kwargs) -> List[Dict[str, Any]]:

        ticker = self.config.ticker

        if not self.active:
            log.debug(f"[HEDGER] {ticker}: inactive (no baseline probs)")
            return []
        if not full_top_bids or not full_top_offers:
            log.debug(f"[HEDGER] {ticker}: no book data")
            return []

        parsed = parse_series_ticker(ticker)
        if not parsed:
            return []
        series_base, team_suffix, _ = parsed

        # Find opponent
        opp_suffix = None
        for k in full_top_bids:
            if k.startswith(series_base + "-") and k != ticker:
                opp_suffix = k.split("-")[-1]
                break
        if not opp_suffix:
            # Check if fills exist but opponent not found
            pending_own = position_store.pop_maker_fills(ticker)
            if pending_own:
                log.warning(f"[HEDGER] {ticker}: {len(pending_own)} fills LOST — no opponent found in book. "
                            f"fills={[(q,p,s) for q,p,s,_ in pending_own]}")
            return []

        opp_ticker = f"{series_base}-{opp_suffix}"

        # Only the alphabetically-first team runs (same dedup as arber)
        if team_suffix > opp_suffix:
            # Still need to check for fills on both tickers — if we're the second team,
            # the first team's evaluate() will pop them
            return []

        # Collect maker fills from BOTH our ticker and opponent's
        our_fills = position_store.pop_maker_fills(ticker)
        opp_fills = position_store.pop_maker_fills(opp_ticker)

        if not our_fills and not opp_fills:
            return []

        # Log raw fills, then normalized (opp NO→our YES, opp YES→our NO, same price)
        log.info(f"[HEDGER] {ticker}: FILLS DETECTED — our={len(our_fills)} opp={len(opp_fills)} "
                 f"| raw_our={[(q,p,s) for q,p,s,_ in our_fills]} "
                 f"| raw_opp={[(q,p,s) for q,p,s,_ in opp_fills]}")


        # Net the fills into our team's perspective
        net_delta = 0
        long_fills = []
        short_fills = []

        # Normalize opp fills into our team's terms before netting.
        # trade_logger pushes `yes_cost` = the YES-side price for the FILL'S TICKER
        # (so for opp_fills it's opp's YES price, not ours). Flipping the side
        # label without transforming the scalar leaves a NO-side number tagged
        # as a YES-equivalent. Required transform: our_yes_equiv = 100 - opp_yes_equiv,
        # since opp_YES_price == our_NO_price == 100 − our_YES_price.
        all_fills = list(our_fills)
        for qty, yes_cost, side, ts in opp_fills:
            flipped_side = "yes" if side == "no" else "no"
            all_fills.append((qty, 100 - yes_cost, flipped_side, ts))

        # Now net: YES = LONG our team, NO = SHORT our team. Price = yes_cost of our ticker.
        for qty, yes_cost, side, ts in all_fills:
            if side == "yes":
                net_delta += qty
                long_fills.append((qty, yes_cost))
            else:
                net_delta -= qty
                short_fills.append((qty, yes_cost))

        min_fill = self.config.volumes[0] if self.config.volumes else 200
        if abs(net_delta) < min_fill:
            log.info(f"[HEDGER] {ticker}: net_delta={net_delta} below min_fill={min_fill} — SKIPPING")
            return []

        # Most aggressive fill price (deepest level swept).
        # For longs: the cheapest price we bought at (most edge).
        # For shorts: the highest price we sold at (most edge).
        if net_delta > 0:
            worst_price = min(p for _, p in long_fills) if long_fills else 50
        else:
            worst_price = max(p for _, p in short_fills) if short_fills else 50

        # State detection — prefer series_eval (Phase 5c), fall back to inline.
        _, _, map_base = parse_series_ticker(ticker)
        if series_eval is not None and "state" in series_eval:
            state = series_eval["state"]
            active_map = series_eval["active_map"]
            active_map_opp = series_eval["active_map_opp"]
            if active_map is None:
                log.info(f"[HEDGER] {ticker}: series_eval has no active_map (state={state}) — skipping hedge")
                return []
            wa = int(series_eval.get("m1_won", False)) + int(series_eval.get("m2_won", False))
            wb = int(series_eval.get("m1_lost", False)) + int(series_eval.get("m2_lost", False))
        else:
            state_info = detect_bo3_state(
                map_base, team_suffix, full_market_state,
                full_top_bids, full_top_offers, series_base=series_base,
                data_source=self.config.data_source,
                disable_forfeit_check=self.config.disable_forfeit_check)
            if state_info is None or state_info["active_map"] is None:
                log.info(f"[HEDGER] {ticker}: state_info={state_info} — skipping hedge")
                return []
            state = state_info["state"]
            active_map = state_info["active_map"]
            active_map_opp = state_info["active_map_opp"]
            wa = int(state_info["m1_won"]) + int(state_info["m2_won"])
            wb = int(state_info["m1_lost"]) + int(state_info["m2_lost"])

        # Check can_trade
        series_status = full_market_state.get(ticker, {}).get("status", "") if full_market_state else ""
        active_map_status = full_market_state.get(active_map, {}).get("status", "") if full_market_state else ""
        if series_status != "active" or active_map_status != "active":
            log.info(f"[HEDGER] {ticker}: can't trade — series={series_status} map={active_map_status}")
            return []

        p1 = self.p1_start
        p2 = self.p2_start
        series_prob = self.series_start
        g2 = self.g2_momentum
        g3 = self.g3_momentum
        is_bo5 = self.is_bo5
        p3 = self.p3_start
        min_edge = max(self.config.min_edge, 0)

        # For LONG fills, we paid YES cost (worst_price is in YES terms).
        # For SHORT fills, we paid NO cost = 100 - worst_price (worst_price is YES terms).
        if net_delta > 0:
            actual_cost = float(worst_price)
        else:
            actual_cost = 100.0 - float(worst_price)
        series_loaded = get_loaded_cost_cents(actual_cost)
        hedge_threshold = 100.0 - min_edge

        direction = "LONG" if net_delta > 0 else "SHORT"
        log.info(f"[HEDGER] {ticker}: Maker fill {direction} {abs(net_delta)} lots @ yes={worst_price}c (actual_cost={actual_cost:.0f}c) (S{state}) | "
                 f"series_loaded={series_loaded:.1f}c threshold={hedge_threshold:.1f}c | "
                 f"map={active_map} map_opp={active_map_opp} | "
                 f"normalized={[(q,p,s) for q,p,s,_ in all_fills]}")

        # Get map orderbooks for level-by-level sweep.
        # raw_ob has the full book (from WS). If missing (EU hours / Poly path),
        # build a single-level book from full_top_bids/full_top_offers.
        active_ob = full_market_state.get(active_map, {}).get("raw_ob", {}).get("orderbook_fp", {}) if full_market_state else {}
        opp_ob = full_market_state.get(active_map_opp, {}).get("raw_ob", {}).get("orderbook_fp", {}) if full_market_state else {}

        if not active_ob.get("yes_dollars") and not active_ob.get("no_dollars"):
            # Fallback: build synthetic single-level book from top-of-book prices
            bid = full_top_bids.get(active_map, 0)
            ask = full_top_offers.get(active_map, 100)
            if bid > 0:
                active_ob = {"yes_dollars": [[str(bid / 100.0), "9999"]], "no_dollars": []}
            if ask < 100:
                no_bid = 100 - ask
                active_ob.setdefault("no_dollars", [])
                active_ob["no_dollars"] = [[str(no_bid / 100.0), "9999"]]
                active_ob.setdefault("yes_dollars", [])

        if not opp_ob.get("yes_dollars") and not opp_ob.get("no_dollars"):
            opp_bid = full_top_bids.get(active_map_opp, 0)
            opp_ask = full_top_offers.get(active_map_opp, 100)
            if opp_bid > 0:
                opp_ob = {"yes_dollars": [[str(opp_bid / 100.0), "9999"]], "no_dollars": []}
            if opp_ask < 100:
                opp_no_bid = 100 - opp_ask
                opp_ob.setdefault("no_dollars", [])
                opp_ob["no_dollars"] = [[str(opp_no_bid / 100.0), "9999"]]
                opp_ob.setdefault("yes_dollars", [])

        if net_delta > 0:
            # Long our team — hedge by buying map NO
            levels = []
            for pt in active_ob.get("yes_dollars", []):
                no_ask = 100 - int(round(float(pt[0]) * 100))
                vol = int(float(pt[1]))
                if vol > 0 and no_ask > 0:
                    levels.append((no_ask, vol, active_map, "no"))
            for pt in opp_ob.get("no_dollars", []):
                opp_yes_ask = 100 - int(round(float(pt[0]) * 100))
                vol = int(float(pt[1]))
                if vol > 0 and opp_yes_ask > 0:
                    levels.append((opp_yes_ask, vol, active_map_opp, "yes"))
            levels.sort(key=lambda x: x[0])
            kalshi_side = "no"
            # hedge_profile_1 = long direction hedge
            hedge_dir = 1
        else:
            # Short our team — hedge by buying map YES
            levels = []
            for pt in active_ob.get("no_dollars", []):
                yes_ask = 100 - int(round(float(pt[0]) * 100))
                vol = int(float(pt[1]))
                if vol > 0 and yes_ask > 0:
                    levels.append((yes_ask, vol, active_map, "yes"))
            for pt in opp_ob.get("yes_dollars", []):
                opp_no_ask = 100 - int(round(float(pt[0]) * 100))
                vol = int(float(pt[1]))
                if vol > 0 and opp_no_ask > 0:
                    levels.append((opp_no_ask, vol, active_map_opp, "no"))
            levels.sort(key=lambda x: x[0])
            kalshi_side = "yes"
            # hedge_profile_2 = short direction hedge
            hedge_dir = 2

        if not levels:
            active_y = len(active_ob.get("yes_dollars", []))
            active_n = len(active_ob.get("no_dollars", []))
            opp_y = len(opp_ob.get("yes_dollars", []))
            opp_n = len(opp_ob.get("no_dollars", []))
            has_raw = "raw_ob" in (full_market_state.get(active_map, {}) if full_market_state else {})
            log.info(f"[HEDGER] {ticker}: no map levels — active_map={active_map} has_raw_ob={has_raw} "
                     f"ob={active_y}y/{active_n}n | opp={active_map_opp} ob={opp_y}y/{opp_n}n")
            return []

        log.info(f"[HEDGER] {ticker}: {len(levels)} map levels available: {[(p,v,t,s) for p,v,t,s in levels[:8]]}")

        # Compute hedge ratio from the cheapest level using centralized engine
        # For LONG: hp1 is the relevant profile (map_no_ask = cheapest level)
        # For SHORT: hp2 is the relevant profile (map_yes_ask = cheapest level)
        cheapest = levels[0][0]
        if hedge_dir == 1:
            hp1, _ = compute_hedge_profiles(state, p1, p2, series_prob, g2, g3, cheapest, 50,
                                            is_bo5=is_bo5, p3=p3, wins_a=wa, wins_b=wb)
            hedge_ratio = hp1["shares_b"]
        else:
            _, hp2 = compute_hedge_profiles(state, p1, p2, series_prob, g2, g3, 50, cheapest,
                                            is_bo5=is_bo5, p3=p3, wins_a=wa, wins_b=wb)
            hedge_ratio = hp2["shares_b"]

        target_qty = round(abs(net_delta) * hedge_ratio)
        max_fire = getattr(self.config, "max_fire_size", 1000)
        target_qty = min(target_qty, max_fire)

        log.info(f"[HEDGER] {ticker}: hedge_ratio={hedge_ratio:.3f} target_qty={target_qty} "
                 f"(net_delta={abs(net_delta)} × {hedge_ratio:.3f}, max_fire={max_fire})")

        if target_qty <= 0:
            log.info(f"[HEDGER] {ticker}: target_qty=0 — SKIPPING")
            return []

        # Walk the book level by level, taking volume where we can afford it
        orders = {}  # (ticker, side) -> {"vol": 0, "worst_price": 0}
        filled = 0

        for price, vol_avail, lvl_ticker, lvl_side in levels:
            if price > MAX_FIRE_PRICE_C:
                log.info(f"[HEDGER] {ticker}: level {price}c > MAX_FIRE_PRICE={MAX_FIRE_PRICE_C}c — STOP")
                break
            # Compute hedge cost at this map level via centralized engine
            if hedge_dir == 1:
                hp1, _ = compute_hedge_profiles(state, p1, p2, series_prob, g2, g3, price, 50,
                                                is_bo5=is_bo5, p3=p3, wins_a=wa, wins_b=wb)
                h_cost = hp1["synthetic_cost"]
            else:
                _, hp2 = compute_hedge_profiles(state, p1, p2, series_prob, g2, g3, 50, price,
                                                is_bo5=is_bo5, p3=p3, wins_a=wa, wins_b=wb)
                h_cost = hp2["synthetic_cost"]
            total_cost = series_loaded + h_cost
            if total_cost > hedge_threshold:
                log.info(f"[HEDGER] {ticker}: level {price}c too expensive "
                         f"(loaded={series_loaded:.1f} + synth={h_cost:.1f} = {total_cost:.1f} > {hedge_threshold:.1f})")
                break  # too expensive, stop

            take = min(vol_avail, target_qty - filled)
            if take <= 0:
                break

            key = (lvl_ticker, lvl_side)
            if key not in orders:
                orders[key] = {"vol": 0, "worst_price": 0}
            orders[key]["vol"] += take
            orders[key]["worst_price"] = max(orders[key]["worst_price"], price)
            filled += take

            if filled >= target_qty:
                break

        if filled <= 0:
            log.info(f"[HEDGER] {ticker}: no affordable levels on map book (first level cost check failed)")
            return []

        log.info(f"[HEDGER] {ticker}: book walk allocated {filled}/{target_qty} lots across {len(orders)} IOC order(s) — submitting to Kalshi")

        result = []
        for (ord_ticker, ord_side), r in orders.items():
            if r["vol"] > 0:
                result.append({
                    "side": QuoteSide.BID,
                    "ticker": ord_ticker,
                    "kalshi_side": ord_side,
                    "size": r["vol"],
                    "limit_cents": r["worst_price"],
                    "time_in_force": "immediate_or_cancel",
                })

        if result:
            for o in result:
                log.info(f"[HEDGER] → {o['kalshi_side'].upper()} {o['size']}x @ {o['limit_cents']}c on {o['ticker']}")
        else:
            log.info(f"[HEDGER] {ticker}: no affordable levels on map book")

        return result
