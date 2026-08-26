import logging
import os
from collections import deque
from typing import Callable, Any, List, Dict
import math
import time
import uuid

from framework_config import QuoterConfig, logit_scaled_edge
import hedge_staleness
from models import QuoteSide
import position_store
from position_adjuster import compute_position_skew_shift_cents, variance_scale_at_theo
from hedge_engine import (
    solve_set_probability,
    get_loaded_cost_cents,
    get_raw_cost_cents,
    get_actual_cost_per_share,
    solve_game_3_probability,
    build_profile,
    calculate_state_0_hedge,
    calculate_state_1_hedge,
    calculate_state_2_hedge,
    calculate_state_3_hedge,
    _bo5_game_probs,
    _bo5_series_prob,
    calculate_bo5_hedge,
    detect_bo3_state,
    evaluate_forfeit,
    bo5_disabled,
)

log = logging.getLogger(__name__)

DEBUG_MODE = False  # Toggle this to True to print every cycle regardless of snapshot changes

# ── MAP-WIDTH TAKER BLOCK (26AUG23, OSGSPE) ────────────────────────────────
# Do not TAKE when the map book that produces the hedge is too wide to price.
#
# Distinct from the FAKE-MOVE check further down, which is a *transition* test
# (was-wide AND exactly one side moved). That XOR catches 31/95 -> 31/32, but
# OSGSPE went 31/95 -> 5/32 — both sides moved, so `bid_moved != ask_moved` was
# False and a 64c-wide book sailed through. This gate is a *level* test: it does
# not care how the book got wide, only that it is.
#
# What it would have stopped: KXVALORANTGAME-26AUG230400OSGSPE 08:36:39.745,
# 9,139 lots swept on a +29.6c "edge" derived from a Poly map book quoting
# 31/95 (64c wide, 1-2 orders a side) that had been frozen 121s. The hedge read
# 41.0c, reverted to 75.3c within 2.5s, and Kalshi's own OSG ask never moved
# (28c before, 26c after) — there was no move to trade.
#
# Width is two-sided and venue-agnostic: yes_ask + no_ask - 100. The empty-book
# sentinel (both sides 100) yields 100 and blocks, which is correct.
#
# Kept separate from the FAKE-MOVE threshold so the two can be tuned apart —
# they answer different questions. 15c is the inherited value; it is NOT yet
# validated against the width-vs-markout study, so expect to move it.
MAP_WIDE_BLOCK_C: float = 15.0

# Taker-only. The quoter prices wide books deliberately and is left alone.
MAP_WIDE_BLOCK_LOG_SEC: float = 60.0  # throttle: eval runs ~3x/s per event

# Require more edge on series-triggered taker fires than map-triggered ones.
# Series-triggered = series book moved more than the hedge book on this cycle,
# which historically PnLs worse than hedge/map-driven fires. Applied ONLY when
# series_edge_mult is now a per-row config field on QuoterConfig (2026-06-03).
# d{1,2}_trigger == "series" uses self.config.series_edge_mult; map/init/empty
# triggers keep self.config.min_edge unchanged.

# G3 edge buffer multiplier. Buffer = mult * P(G3) * min_edge during BO3
# G2 (state 1/2). 2026-06-08: per-config-prefix override. Baseline lowered
# 1.0 → 0.5 across all esports as a pilot; CS2 Tier 1 further dropped to
# 0.25 since those markets move fast enough that the full audit-derived
# buffer was too punitive. Tier-1 detection via "_T1" in the config
# market_id (e.g. CS2_ARB_L1_T1 → 0.25; CS2_ARB_L1 → 0.5; LOL_ARB_L1 →
# 0.5). Revert to a flat constant or per-row CSV column if validated.
def _g3_edge_mult_for(config) -> float:
    # Read from config.g3_edge_mult (per-row knob in market_parameters.csv).
    # Falls back to the legacy per-prefix table when the attribute is missing
    # (defensive guard for tests / configs built without the new column).
    v = getattr(config, "g3_edge_mult", None)
    if v is not None:
        return float(v)
    mid = (getattr(config, "market_id", "") or "").upper()
    if "CS2" in mid and "_T1" in mid:
        return 0.25
    return 0.5


def calculate_sweep_payload(
    ob_pts: list,
    hedge_cost: float,
    min_edge: float,
    scale_step_c: float,
    base_vol: int,
    max_fire_size: int,
    max_position: int,
    current_pos: int,
    direction_sign: int,
    min_absolute_edge: float = 0.0,
    position_skew_cents: float = 0.0,
    fav_edge_k: float = 0.0,
    retreat_shift_cents: float = 0.0,
) -> tuple[int, int]:
    """
    Iterates over the local 'raw_ob' ticks dynamically parsing asymmetric expanding/reducing rules
    and perfectly isolating the strict IOC payload magnitude according to the mathematical scaling rules.
    Edge gates use variance-scaled min_edge per price level (NARROWEST at the extremes,
    widest at 50c); min_absolute_edge floors it. fav_edge_k > 0 layers the favourite-side
    adverse-selection premium on top — series_ask_p is already the price we PAY for the leg
    we go long, so it is the correct orientation for the asymmetric multiplier.
    ob_pts: array of [opponent_price_dollars, size] which natively unpacks the ask side of Kalshi
    """
    accumulated_vol = 0
    worst_price_taken = 0

    # Sort from best price to worst: descending opponent_bids means ascending asks
    sorted_pts = sorted((pt for pt in ob_pts if len(pt) >= 2), key=lambda x: float(x[0]), reverse=True)

    for pt in sorted_pts:
        no_bid_p = float(pt[0])
        size_avail = int(float(pt[1]))
        if size_avail <= 0: continue

        series_ask_p = int(round(100.0 - (no_bid_p * 100.0)))
        series_loaded = get_loaded_cost_cents(float(series_ask_p))
        cost_synth = series_loaded + hedge_cost
        edge_c = 100.0 - cost_synth
        eff_min_edge = logit_scaled_edge(series_ask_p, min_edge, min_absolute_edge, fav_edge_k=fav_edge_k)
        # Per-price-level variance-scaled position skew. With position_skew_cents=0
        # adj_skew=0.0 and ALL gates reduce to the legacy (eff_min_edge ± 0) values.
        # retreat_shift_cents (TRADE_RETREAT_SCOPE.md taker knob, 26AUG18) is a
        # SIGNED theo-shift equivalent: + when this fire direction is the
        # recently-accumulated side (edge overstated by the shift), - on the
        # cold side. Unlike the skew it moves BOTH gates together (a theo
        # shift is add/reduce-agnostic). 0.0 = exact legacy behavior.
        adj_skew = position_skew_cents * variance_scale_at_theo(series_ask_p)
        add_gate = eff_min_edge + adj_skew + retreat_shift_cents
        reduce_gate = eff_min_edge - adj_skew + retreat_shift_cents

        taken_here = 0

        while size_avail > 0:
            projected_pos = current_pos + (direction_sign * accumulated_vol)
            # Short condition check
            is_reducing = (direction_sign > 0 and projected_pos < 0) or (direction_sign < 0 and projected_pos > 0)

            if is_reducing:
                if edge_c < (reduce_gate - 0.05):  # Require gate (with float tolerance) for reducing fires
                    break
                chunk_cap = min(size_avail, abs(projected_pos))
                if accumulated_vol >= max_fire_size:
                    break
                take = min(chunk_cap, max_fire_size - accumulated_vol)
                if take <= 0: break

                accumulated_vol += take
                size_avail -= take
                taken_here += take

            else:
                if edge_c < (add_gate - 0.05):
                    break
                # Linear Multiplier: Expand strictly natively
                scaling_tiers = max(0, int((edge_c - add_gate) / max(1.0, scale_step_c)))
                authorized = base_vol + (scaling_tiers * base_vol)
                max_allowed = min(max_fire_size, authorized)

                if accumulated_vol >= max_allowed:
                    break

                global_room = max_position - abs(projected_pos)
                if global_room <= 0:
                    # Hedge zone: allow up to 2x max_position if edge >= 2.5x add_gate
                    hedge_zone_room = int(1.5 * max_position) - abs(projected_pos)
                    if hedge_zone_room > 0 and edge_c >= (add_gate * 2.5):
                        global_room = hedge_zone_room
                    else:
                        break

                take = min(size_avail, max_allowed - accumulated_vol, global_room)
                if take <= 0: break

                accumulated_vol += take
                size_avail -= take
                taken_here += take

        if taken_here > 0:
            worst_price_taken = max(worst_price_taken, series_ask_p)
        else:
            # Structurally hit our mathematical bounds threshold, stop analyzing book
            break

    return accumulated_vol, worst_price_taken


# ── Surplus over-fire gate ──────────────────────────────────────────────────
# Enabled by env ARBER_SURPLUS_FIRE=1 (boot-time) OR the presence of
# `enable_arber_surplus.flag` (live-toggleable — touch/rm with no restart,
# matching enable_ws_rest_shadow.flag). The file check is cached ~1s so the
# sweep hot-path never stats the filesystem more than once a second.
_SURPLUS_FLAG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enable_arber_surplus.flag")
_surplus_flag_cache = {"ts": -1.0, "on": False}


def _surplus_fire_enabled() -> bool:
    if os.environ.get("ARBER_SURPLUS_FIRE") == "1":
        return True
    now = time.time()
    if now - _surplus_flag_cache["ts"] > 1.0:
        _surplus_flag_cache["ts"] = now
        try:
            _surplus_flag_cache["on"] = os.path.exists(_SURPLUS_FLAG_PATH)
        except Exception:
            _surplus_flag_cache["on"] = False
    return _surplus_flag_cache["on"]


# How many cents past the last-taken price the SURPLUS order may reach.
# Every intervening level must still clear the edge gate independently.
SURPLUS_REACH_CENTS = 2


def calculate_cross_book_sweep(
    books: list,
    hedge_cost: float,
    min_edge: float,
    scale_step_c: float,
    base_vol: int,
    max_fire_size: int,
    max_position: int,
    current_pos: int,
    direction_sign: int,
    min_absolute_edge: float = 0.0,
    position_skew_cents: float = 0.0,
    fav_edge_k: float = 0.0,
    retreat_shift_cents: float = 0.0,
) -> list:
    """
    Merges economically equivalent order books (e.g., our YES book + opponent NO book)
    and sweeps the combined liquidity optimally, routing each fill to the cheapest source.

    books: [(ob_pts, ticker, kalshi_side), ...] where ob_pts are [price_dollars, size] arrays.
           All books must represent the same economic position (e.g., all ways to go "long team A").

    Returns list of order dicts, one per (ticker, side) that received fills.
    """
    tagged_pts = []
    for ob_pts, src_ticker, src_side in books:
        for pt in ob_pts:
            if len(pt) >= 2:
                p = float(pt[0])
                s = int(float(pt[1]))
                if s > 0:
                    tagged_pts.append((p, s, src_ticker, src_side))

    # Sort descending by price (= ascending by effective cost — cheapest first)
    tagged_pts.sort(key=lambda x: x[0], reverse=True)

    results = {}  # (ticker, side) -> {"vol": 0, "worst_price": 0}
    accumulated_vol = 0
    # PTA-only: was max_fire_size the binding constraint that stopped the sweep?
    # True iff we broke out at a cap-check where max_fire_size <= authorized.
    # Distinguishes "we'd take more if our cap were higher" from "edge ran out"
    # / "book ran out" / "position cap hit." Used downstream by trade_logger to
    # tag is_full_fill on taker fills only when raising the cap would have helped.
    hit_size_cap = False

    for no_bid_p, size_avail, src_ticker, src_side in tagged_pts:
        series_ask_p = int(round(100.0 - (no_bid_p * 100.0)))
        series_loaded = get_loaded_cost_cents(float(series_ask_p))
        cost_synth = series_loaded + hedge_cost
        edge_c = 100.0 - cost_synth
        eff_min_edge = logit_scaled_edge(series_ask_p, min_edge, min_absolute_edge, fav_edge_k=fav_edge_k)
        # Per-price-level variance-scaled position skew. With position_skew_cents=0
        # adj_skew=0.0 and ALL gates reduce to the legacy (eff_min_edge ± 0) values.
        # retreat_shift_cents: signed theo-shift equivalent (see
        # calculate_sweep_payload) — moves BOTH gates together; 0.0 = legacy.
        adj_skew = position_skew_cents * variance_scale_at_theo(series_ask_p)
        add_gate = eff_min_edge + adj_skew + retreat_shift_cents
        reduce_gate = eff_min_edge - adj_skew + retreat_shift_cents

        key = (src_ticker, src_side)
        taken_here = 0
        # Telemetry-only: track Σ(signed_skew * take) at this price level so we
        # can surface a volume-weighted skew_shift_cents on the resulting order.
        # Sign convention: + = skew favored the take (reduce_gate was used),
        # - = skew opposed (add_gate). Decision math unchanged.
        skew_x_take_here = 0.0

        while size_avail > 0:
            projected_pos = current_pos + (direction_sign * accumulated_vol)
            is_reducing = (direction_sign > 0 and projected_pos < 0) or (direction_sign < 0 and projected_pos > 0)

            if is_reducing:
                if edge_c < (reduce_gate - 0.05):  # Require gate (with float tolerance) for reducing fires
                    break
                chunk_cap = min(size_avail, abs(projected_pos))
                if accumulated_vol >= max_fire_size:
                    hit_size_cap = True
                    break
                take = min(chunk_cap, max_fire_size - accumulated_vol)
                if take <= 0:
                    break
                accumulated_vol += take
                size_avail -= take
                taken_here += take
                skew_x_take_here += (+adj_skew) * take
            else:
                if edge_c < (add_gate - 0.05):
                    break
                scaling_tiers = max(0, int((edge_c - add_gate) / max(1.0, scale_step_c)))
                authorized = base_vol + (scaling_tiers * base_vol)
                max_allowed = min(max_fire_size, authorized)
                if accumulated_vol >= max_allowed:
                    # Flag whenever the sweep stops because our internal sizing
                    # config bound us — whether the binding sub-constraint was
                    # max_fire_size (raise cap → more fills) or `authorized`
                    # (raise base_vol / lower scale_step_c → more fills). Both
                    # are actionable PTA signals: "raising our sizing config
                    # would have given us more fills here." Updated 2026-05-26.
                    hit_size_cap = True
                    break
                global_room = max_position - abs(projected_pos)
                if global_room <= 0:
                    # Hedge zone: allow up to 2x max_position if edge >= 2.5x add_gate
                    hedge_zone_room = int(1.5 * max_position) - abs(projected_pos)
                    if hedge_zone_room > 0 and edge_c >= (add_gate * 2.5):
                        global_room = hedge_zone_room
                    else:
                        break
                take = min(size_avail, max_allowed - accumulated_vol, global_room)
                if take <= 0:
                    break
                accumulated_vol += take
                size_avail -= take
                taken_here += take
                skew_x_take_here += (-adj_skew) * take

        if taken_here > 0:
            if key not in results:
                results[key] = {"vol": 0, "worst_price": 0, "worst_edge": None,
                                "skew_x_vol": 0.0}
            results[key]["vol"] += taken_here
            results[key]["worst_price"] = max(results[key]["worst_price"], series_ask_p)
            results[key]["worst_edge"] = edge_c
            results[key]["skew_x_vol"] += skew_x_take_here
        else:
            break

    # Cross-book sweep generates multiple orders (one per leg) but the
    # sizing cap (max_fire_size / authorized) binds on the COMBINED sweep
    # accumulated_vol, not per-leg. Tag every order in this sweep with a
    # shared sweep_id + total combined vol so trade_logger can aggregate
    # cumulative taker fills across both legs when deciding is_full_fill.
    # Without this, a sweep that orders 393+1233 lots is checked per-leg
    # (e.g. 333/393 → tagged full) instead of vs the actual cap target.
    # Added 2026-05-27.
    sweep_id = str(uuid.uuid4()) if results else ""
    sweep_total = sum(r["vol"] for r in results.values())
    orders = []
    for (ord_ticker, ord_side), r in results.items():
        if r["vol"] > 0:
            # Store theo in YES terms so trade logger's NO-flip produces correct edge.
            # For YES orders: theo = 100 - hedge_cost (fair YES value)
            # For NO orders:  theo = hedge_cost (fair YES value, since NO value = 100 - hedge_cost)
            # Trade logger flips NO theos: 100 - hedge_cost = fair NO value → correct edge.
            if ord_side == "no":
                arb_theo_yes = hedge_cost
            else:
                arb_theo_yes = 100.0 - hedge_cost
            # Volume-weighted signed skew shift in cents. + = skew favored,
            # - = skew opposed. Display-only — surfaced by watch_fills.
            mean_skew = r.get("skew_x_vol", 0.0) / r["vol"] if r["vol"] else 0.0
            orders.append({
                "side": QuoteSide.BID,
                "ticker": ord_ticker,
                "kalshi_side": ord_side,
                "size": r["vol"],
                "limit_cents": r["worst_price"],
                "time_in_force": "immediate_or_cancel",
                "_raw_theo": arb_theo_yes,
                "_adj_theo": arb_theo_yes,
                "_skew_shift_cents": mean_skew,
                "_was_size_capped": hit_size_cap,
                "_sweep_id": sweep_id,
                "_sweep_total_vol": sweep_total,
            })

    # ── Surplus over-fire (flag-gated, PURELY ADDITIVE) ─────────────────────
    # WS-visible depth caps our SIZE exactly like a stale ask caps our price:
    # if the real book is deeper than WS showed, a book-limited sweep silently
    # under-fills (looks like a full fill). When ARBER_SURPLUS_FIRE=1, fire
    # EXTRA IOC size split EQUALLY across the equivalent books, at the worst
    # price we ALREADY accepted, so genuinely under-reported depth still fills.
    # IOC cancels any excess → no resting exposure. The visible orders above are
    # untouched, so normal firing is unaffected even if this whole block no-ops
    # or throws. Gated ON only after shadow data confirms edges are mostly real
    # (over-firing scales up whatever the edge truly is). (2026-07-26)
    try:
        if (_surplus_fire_enabled()
                and results and not hit_size_cap and accumulated_vol > 0 and books):
            allowable = min(max_fire_size, max_position - abs(current_pos))
            try:
                mult = float(os.environ.get("ARBER_SURPLUS_MULT", "2.0"))
            except ValueError:
                mult = 2.0
            # Conservative surplus: at most `mult`x the visible size, and never
            # past the sizing cap. `mult` bounds tail risk on a phantom fire.
            surplus_total = max(0, min(allowable - accumulated_vol, int(mult * accumulated_vol)))
            per_leg = surplus_total // max(1, len(books))
            if per_leg > 0:
                # Fire at the worst economic ask we already accepted — captures
                # under-reported depth at prices we've already vetted as ≥min_edge.
                base_surplus_limit = max(r["worst_price"] for r in results.values())
                # REACH (2026-07-31): extend the surplus limit up to
                # SURPLUS_REACH_CENTS past the last price we actually took, but
                # ONLY across levels that independently clear every gate the main
                # walk applies. Rationale from 4.5d of phantom probes: when a fire
                # zero-fills, the real liquidity is +1c away in 933 cases and +2c
                # in 444; 17.1% of those misses (137,840 lots, ~$5.2k of edge)
                # sat at a price still profitable by our own gate. Beyond +2c the
                # edge is genuinely gone, so reaching further only buys tail risk.
                #
                # An IOC limit is a WORST-acceptable price, not a target: if the
                # displayed level is real we still fill there first and this
                # changes nothing. It only bites when the top level was phantom.
                #
                # Uses the ADD gate (eff_min_edge + skew) even though the surplus
                # may end up reducing — the reducing gate is looser, and we would
                # rather under-reach than take a lot that fails the strict test.
                surplus_limit = base_surplus_limit
                for _extra in range(1, SURPLUS_REACH_CENTS + 1):
                    _p = base_surplus_limit + _extra
                    if _p > 99:
                        break
                    _edge_c = 100.0 - (get_loaded_cost_cents(float(_p)) + hedge_cost)
                    _gate = (logit_scaled_edge(_p, min_edge, min_absolute_edge,
                                               fav_edge_k=fav_edge_k)
                             + position_skew_cents * variance_scale_at_theo(_p)
                             + retreat_shift_cents)
                    if _edge_c < (_gate - 0.05):
                        break          # edge exhausted — do not reach further
                    # Position room must also still exist at the extended reach.
                    if max_position - abs(current_pos) <= 0:
                        break
                    surplus_limit = _p
                if surplus_limit != base_surplus_limit:
                    log.info(f"[ARBER SURPLUS REACH] extended limit "
                             f"{base_surplus_limit}c -> {surplus_limit}c "
                             f"(+{surplus_limit - base_surplus_limit}c, hedge={hedge_cost:.1f}c) "
                             f"— all levels clear gate")
                visible_by_book = {(t, s): r["vol"] for (t, s), r in results.items()}
                surplus_sweep_id = str(uuid.uuid4())      # own id — do NOT pollute visible is_full_fill accounting
                for _ob, bk_ticker, bk_side in books:
                    arb_theo_yes = hedge_cost if bk_side == "no" else (100.0 - hedge_cost)
                    orders.append({
                        "side": QuoteSide.BID,
                        "ticker": bk_ticker,
                        "kalshi_side": bk_side,
                        "size": per_leg,
                        "limit_cents": surplus_limit,
                        "time_in_force": "immediate_or_cancel",
                        "_raw_theo": arb_theo_yes,
                        "_adj_theo": arb_theo_yes,
                        "_skew_shift_cents": 0.0,
                        "_was_size_capped": False,
                        "_sweep_id": surplus_sweep_id,
                        "_sweep_total_vol": per_leg * len(books),
                        "_is_surplus": True,
                        "_visible_vol": visible_by_book.get((bk_ticker, bk_side), 0),
                    })
                log.info("[ARBER SURPLUS] %s dir | visible=%d +surplus=%d (%d/leg x%d books) "
                         "@<=%dc | allowable=%d mult=%.1f",
                         books[0][1], accumulated_vol, per_leg * len(books), per_leg,
                         len(books), surplus_limit, allowable, mult)
    except Exception as _e:
        log.warning("[ARBER SURPLUS] skipped (%s: %s); visible orders unaffected.",
                    type(_e).__name__, _e)

    return orders


_BLACKOUT_PRICE = 100
"""Empty-book sentinel. With no offers, series_ask defaults to 100.0 and the
opponent bid to 0.0, so best_d*_price computes to exactly 100 — a synthetic
'price' that no real book prints as a tradeable ask."""


def _quoter_retreat_conf(ticker):
    """The armed QUOTER config for this ticker, so the LEAD-LAG retreat readout
    reflects what is actually applied to quotes. Returns None if unavailable."""
    try:
        import position_adjuster
        reg = getattr(position_adjuster, "_LIVE_CONFIGS", None)
        if reg:
            c = reg.get(ticker)
            if c is not None and float(getattr(c, "retreat_cap_cents", 0) or 0) > 0:
                return c
    except Exception:
        pass
    return None


def improvement_gate(prev, new):
    """Decide whether a direction's edge IMPROVED vs the previous cycle.

    prev/new are (best_price_cents, hedge_tenths) snapshots; prev is None on
    the first cycle. Returns (improved, trigger, blocked_reason).

    POST-BLACKOUT BASELINE RESET (26AUG19, USEBIG 18:53:41): when a book goes
    empty it prints the 100c sentinel. On refill the raw comparison reads the
    return-to-normal as an enormous price improvement (USEBIG: 100c -> 79c =
    "21c better"), which sized a 9,609-lot sweep off a book that had been blank
    400ms earlier — into a hedge that was simultaneously printing a 1-second
    blip. A refill is a return to normal, NOT an edge. So a transition OUT of
    the sentinel never counts as an improvement: it re-baselines silently and
    the direction can fire on the NEXT genuine improvement measured against
    real prices. Costs at most one cycle (~0.5s) of latency on the rare refill
    that is a true opportunity; the fire it blocks has no valid baseline at all.
    """
    if prev == new:
        return False, "", None          # unchanged — nothing to evaluate
    if prev is None:
        return True, "init", None       # first cycle — always allow
    old_price, old_hedge = prev
    new_price, new_hedge = new
    if old_price >= _BLACKOUT_PRICE and new_price < _BLACKOUT_PRICE:
        return False, "", "post_blackout"
    price_delta = old_price - new_price   # positive = price dropped = better
    hedge_delta = old_hedge - new_hedge   # positive = hedge cheaper = better
    if price_delta > 0 or hedge_delta > 0:
        if price_delta > 0 and hedge_delta <= 0:
            trigger = "series"
        elif hedge_delta > 0 and price_delta <= 0:
            trigger = "map"
        else:
            trigger = "series" if price_delta >= hedge_delta else "map"
        return True, trigger, None
    return False, "", None


class EsportsArberBot:
    """
    Arbitrage bot for Esports series. It evaluates dual-leg IOC orders 
    across the Series and Map markets to capture synthetic discrepancies.
    """
    def __init__(self, config: QuoterConfig):
        self.config = config
        # One-time INIT log so operator can grep `[ARBER INIT]` to confirm the
        # per-row config (esp. series_edge_mult) actually loaded from template.
        # Fires once per ticker bind — low noise, definitive diagnostic.
        log.info(f"[ARBER INIT] {config.ticker} min_edge={config.min_edge} "
                 f"min_abs_edge={config.min_absolute_edge} "
                 f"series_edge_mult={config.series_edge_mult} "
                 f"max_pos={config.max_position} data_source={config.data_source}")
        self.active = True
        self.g2_momentum = 0.05
        self.g3_momentum = 0.05
        self.p1_start = 0.5
        self.p2_start = 0.5
        self.p3_start = 0.5
        self.p4_start = None  # populated from Polymarket Game 4 Winner when available
        self.p5_start = None  # populated from Polymarket Game 5 Winner when available
        self.series_start = 0.5
        self.is_bo5 = False
        self.last_log_time = 0.0
        self._last_prob_retry = 0.0
        self._prev_map_bid = None   # Track previous map bid for fake-move detection
        self._prev_map_ask = None   # Track previous map ask for fake-move detection
        self._reduce_only_active = False  # 1.5× enter / 1.4× release hysteresis

        # Edge-bucketed fire throttle: max 5 fires per bucket per 60 seconds.
        # Bucket = int(100 - edge). Bucket <= 93 (7c+ edge) = unlimited.
        self._fire_timestamps: dict[int, list[float]] = {}  # bucket -> [timestamps]

        # Snapshot cache: only log/fire when inputs change
        self._last_d1 = None  # (series_ask, hedge_1_cost) rounded to 0.1c
        self._last_d2 = None  # (series_no_ask, hedge_2_cost) rounded to 0.1c
        self._last_state = None

        # Pregame-taker gate (see config.pregame_taker_block). Latched per-event:
        # pregame until the series mid OR the map hedge moves >= config.pregame_block_cents
        # from the FIRST-SEEN init, then in-game for the rest of the event. Init is
        # captured on the first evaluate() cycle. A mid-event restart re-latches to
        # pregame (conservative: only ever over-blocks, never fires a bad take).
        self._in_pregame = True
        self._pregame_init_mid = None    # first-seen series mid (cents)
        self._pregame_init_hedge = None  # first-seen hedge cost (cents)

        # ── Series-dominance rolling window (added 2026-06-14) ──
        # Stores recent (ts, d1_price, d1_hedge_tenths, d2_price, d2_hedge_tenths)
        # snapshots so we can compare price moves over a 5s window. The
        # instantaneous d1/d2_trigger classifier looks at the LAST tick only;
        # this lets us catch the case where the last tick was a tiny map blip
        # but the dominant move over the past 5s was series-driven (e.g.
        # 9ZTS-9Z at 21:03 on 2026-06-13: series dropped 10c monotonically
        # while hedge bounced 6c — instantaneous classified as "map" → no
        # series_edge_mult applied → 6000-lot taker fired at +0.1c edge →
        # hedge moved 32c against us in 1.1s).
        self._window_history: deque = deque(maxlen=200)

        # Cluster-fire suppression: tracks the snapshot at the LAST ACTUAL FIRE
        # (separate from _last_d1/d2 which update every cycle). Additive on top
        # of the improvement gate — only blocks fires that the existing gate
        # would have allowed, never unblocks. Driven by
        # config.maker_refill_cooldown_sec (shared with QuoterBot).
        self._last_fire_d1 = None       # (price, hedge_tenths) at last D1 fire
        self._last_fire_d2 = None       # (price, hedge_tenths) at last D2 fire
        self._last_fire_d1_ts = 0.0
        self._last_fire_d2_ts = 0.0
        # Conditional re-fire (2026-08-01). The cluster gate's "price same +
        # hedge same" row is usually flash-fade, but it also suppresses the
        # genuine-refill case: our fire FILLED, new liquidity re-posted at the
        # same price at the same edge, and the 15s blanket cooldown let others
        # eat it (measured: ASTPAIN/MIBREG echo clusters). Within the cooldown
        # we now allow a capped number of re-fires when the change is provably
        # MATERIAL: position moved in the fire direction since the last fire
        # (fill evidence — a zero-fill/phantom never qualifies) AND real
        # displayed depth refilled at the fire price AND the hedge has not
        # worsened (whipsaw rows stay suppressed). See _material_refire.
        self._last_fire_d1_pos = None   # net series position at last D1 fire
        self._last_fire_d2_pos = None
        self._refires_d1 = 0            # conditional re-fires used this window
        self._refires_d2 = 0
        # Decay veto (2026-08-02, BLGJDG). Comparing only against the LAST fire
        # lets a fading signal re-arm the gate on every wobble: hedge 65.4 →
        # 65.8 (suppressed) → 64.8 reads as "improved" against 65.4 even though
        # it is 2.5c off the burst's best of 62.3 — BLGJDG chased a fading
        # spike for 47k lots through exactly that hole, while the ask refilled
        # at the same level ~20 times. Track the BEST (lowest) fire ask and
        # hedge seen in the current cooldown window; a fire at a no-better ask
        # whose hedge has faded past the tolerance is suppressed outright, with
        # NO conditional-refire override. An ask strictly below the window's
        # best still fires (a real price dislocation remains tradeable).
        # Contrast TESLGD (same trigger, +14c markout): its hedge improved on
        # every fire, so each fire sets a new window best and the veto never
        # engages — the discriminator is trajectory, not edge size.
        self._best_fire_d1 = None       # (min ask, min hedge_tenths) this window
        self._best_fire_d2 = None
        # Per-window gate statistics (2026-08-02): every suppression path
        # counts here (including the previously-silent improved=False table
        # suppressions), and a one-line [ARBER GATE WINDOW] summary is emitted
        # when a burst's window closes — one greppable line per burst for EOD
        # review, alongside the per-event [ARBER DECAY SUPPRESS] /
        # [ARBER CLUSTER SUPPRESS] / [ARBER REFIRE-OK] lines.
        self._win_stats_d1 = None       # {"fires","veto","table","dust","refire"}
        self._win_stats_d2 = None
        # Knobs — per-row overridable via market_parameters columns of the
        # same name; these are only the fallback defaults.
        self._refire_max = int(getattr(self.config, "refire_max_per_window", 2))
        self._refire_min_fill = int(getattr(self.config, "refire_min_fill_lots", 10))
        self._refire_min_depth = int(getattr(self.config, "refire_min_depth_lots", 100))
        # hedge fade tolerance for the decay veto, in TENTHS of a cent
        # (snap hedge units): 5 = 0.5c of wobble allowed under the window best.
        self._fade_tenths = int(getattr(self.config, "cluster_hedge_fade_tenths", 5))

        self._load_pre_game_probabilities()

    def _check_fire_throttle(self, edge_c: float) -> bool:
        """Check if this fire is allowed by the edge-bucketed throttle.
        Returns True if allowed, False if throttled.
        Max 5 fires per edge bucket per 60 seconds. 7c+ edge = unlimited."""
        bucket = int(100 - edge_c)
        if bucket <= 93:
            return True  # 7c+ edge, unlimited

        now = time.time()

        # Prune old timestamps
        if bucket in self._fire_timestamps:
            self._fire_timestamps[bucket] = [
                t for t in self._fire_timestamps[bucket] if now - t < 60.0
            ]
        else:
            self._fire_timestamps[bucket] = []

        if len(self._fire_timestamps[bucket]) >= 5:
            return False  # Throttled

        return True

    def _record_fire(self, edge_c: float):
        """Record a fire for throttle tracking."""
        bucket = int(100 - edge_c)
        if bucket > 93:
            if bucket not in self._fire_timestamps:
                self._fire_timestamps[bucket] = []
            self._fire_timestamps[bucket].append(time.time())

    # ── FIRE-MOVE instrumentation (log-only, added 2026-08-08) ──
    # Records the map-implied hedge cost ~1/s per direction so each fire can
    # log how far the hedge moved BEFORE the trigger. Knife-catch study
    # 26AUG08: winning bursts rode ≥10c favorable hedge moves, fake triggers
    # fired on ≤9c. No gate here — offline threshold sweep (5/10/15c) against
    # markouts first: grep '\[FIRE-MOVE\]' run.log.
    _FIRE_MOVE_LOOKBACKS_S = (30, 60, 120, 300)
    _FIRE_MOVE_RETENTION_S = 360.0

    def _note_fire_move_hist(self, now_ts, h1_tenths, h2_tenths, state):
        """Sample (ts, hedge_d1_tenths, hedge_d2_tenths, state), ≥1s apart.
        A side is None when its hedge profile is missing this cycle."""
        hist = getattr(self, "_fire_move_hist", None)
        if hist is None:
            hist = self._fire_move_hist = deque()
        if hist and now_ts - hist[-1][0] < 1.0:
            return
        hist.append((now_ts, h1_tenths, h2_tenths, state))
        while hist and now_ts - hist[0][0] > self._FIRE_MOVE_RETENTION_S:
            hist.popleft()

    # Enforcement thresholds (shipped 2026-08-08 after day-1 backtest):
    # map opens need a 5c favorable pre-move when same-direction exposure is
    # under 20% of max_position, 10c once loaded past that; series opens need
    # 5c flat. Reduces, init/other triggers, state-flip windows, and warmup
    # (no samples yet) are exempt. Kill switch: `touch fire_gate_off.flag`.
    _FIRE_GATE_FRESH_FRAC = 0.20
    _FIRE_GATE_FRESH_MOVE_C = 5.0
    _FIRE_GATE_LOADED_MOVE_C = 10.0
    _FIRE_GATE_SERIES_MOVE_C = 5.0
    _FIRE_GATE_FLAG_OFF = "fire_gate_off.flag"

    def _fire_move_gate(self, direction: str, trigger: str, now_ts: float,
                        hedge_now_c: float, net_series_pos: float,
                        reducing: bool, state) -> tuple:
        """(allow, reason) for a would-be taker fire. Blocks knife-catching
        opens: small-move map/series fires get denied, scaled by how loaded
        the book already is in the fire's direction. Backtest 26AUG08:
        blocked bucket mk900 +1.5c (loser after spread+fees, incl. the
        NAVISK -39k knife), allowed bucket +15.1c."""
        if trigger not in ("map", "series"):
            return True, "trigger-exempt"
        if reducing:
            return True, "reduce-exempt"
        if os.path.exists(self._FIRE_GATE_FLAG_OFF):
            return True, "flag-off"
        hist = getattr(self, "_fire_move_hist", None) or ()
        idx = 1 if direction == "d1" else 2
        mv = None
        for lb in self._FIRE_MOVE_LOOKBACKS_S:
            then = None
            for entry in hist:
                if now_ts - entry[0] <= lb:
                    then = entry[idx]
                    break
            if then is not None:
                d = then / 10.0 - hedge_now_c
                mv = d if mv is None else max(mv, d)
        for entry in hist:
            if now_ts - entry[0] <= 300 and entry[3] != state:
                # Deltas span a map/state boundary — unreliable either way.
                return True, "stateflip-exempt"
        if mv is None:
            return True, "warmup"
        if trigger == "series":
            need = self._FIRE_GATE_SERIES_MOVE_C
        else:
            frac = abs(net_series_pos) / max(self.config.max_position, 1)
            need = (self._FIRE_GATE_FRESH_MOVE_C
                    if frac < self._FIRE_GATE_FRESH_FRAC
                    else self._FIRE_GATE_LOADED_MOVE_C)
        if mv >= need:
            return True, f"mv={mv:+.1f}c>=need{need:.0f}c"
        return False, f"mv={mv:+.1f}c<need{need:.0f}c"

    def _fire_move_str(self, direction: str, now_ts: float,
                       hedge_now_c: float, state) -> str:
        """Pre-fire hedge deltas per lookback; positive = hedge FELL toward
        the fire (favorable). peak300 = drop from the 300s hedge peak.
        stateflip300=1 → a map/state boundary sits inside the window, so the
        move is suspect (transition artifact, not a comeback)."""
        hist = getattr(self, "_fire_move_hist", None) or ()
        idx = 1 if direction == "d1" else 2
        parts = []
        for lb in self._FIRE_MOVE_LOOKBACKS_S:
            then = None
            for entry in hist:  # oldest→newest: first sample inside window
                if now_ts - entry[0] <= lb:
                    then = entry[idx]
                    break
            parts.append(
                f"d{lb}=" + (f"{then / 10.0 - hedge_now_c:+.1f}"
                             if then is not None else "na"))
        peak = None
        flip = 0
        for entry in hist:
            if now_ts - entry[0] <= 300:
                if entry[idx] is not None:
                    peak = entry[idx] if peak is None else max(peak, entry[idx])
                if entry[3] != state:
                    flip = 1
        parts.append("peak300=" + (f"{peak / 10.0 - hedge_now_c:+.1f}"
                                   if peak is not None else "na"))
        parts.append(f"stateflip300={flip}")
        return " ".join(parts)

    # ── Series-dominance window helpers (added 2026-06-14) ──

    # Tunables. Kept as constants for now; can be lifted into market_parameters
    # columns later if we want per-row control. Defaults chosen to MATCH the
    # filter intent ("filter out catastrophic 6000-lot scenarios, leave
    # current behavior alone otherwise"):
    #   WINDOW_S         5.0 — same horizon momentum_bot uses
    #   MIN_SERIES_MOVE  3.0 — moves smaller than this are spread noise
    #   DOMINANCE_RATIO  2.0 — series must be ≥2× the hedge change to override
    _SERIES_WINDOW_S = 5.0
    _SERIES_MIN_MOVE_C = 3.0
    _SERIES_DOMINANCE_RATIO = 2.0

    def _map_wide_block(self, map_width_c) -> bool:
        """True when the map book is too wide to price a hedge off — suppress
        the taker. See MAP_WIDE_BLOCK_C for the incident this exists for.

        Level test, not a transition test: it does not matter how the book got
        wide. The FAKE-MOVE check below handles the was-wide-and-moved-one-sided
        case and misses the two-sided version; this catches both.

        `map_width_c is None` means neither hedge dispatch path set it, which
        should be impossible. It ALLOWS rather than blocks, loudly: failing
        closed on an assumption I have not seen hold for a full day would
        silently halt the taker across every sport. Flip to fail-closed once
        a day of logs shows zero MAP-WIDE UNKNOWN lines.
        """
        if map_width_c is None:
            now = time.time()
            if now - getattr(self, "_last_mapwide_none_ts", 0.0) >= MAP_WIDE_BLOCK_LOG_SEC:
                self._last_mapwide_none_ts = now
                log.error(
                    f"[ARBER MAP-WIDE UNKNOWN] {self.config.ticker}: map width "
                    f"unavailable on both hedge paths — width gate INACTIVE for "
                    f"this eval (allowing). Investigate: this should not happen.")
            return False
        if map_width_c <= MAP_WIDE_BLOCK_C:
            return False
        now = time.time()
        if now - getattr(self, "_last_mapwide_log_ts", 0.0) >= MAP_WIDE_BLOCK_LOG_SEC:
            self._last_mapwide_log_ts = now
            log.warning(
                f"[ARBER MAP-WIDE BLOCK] {self.config.ticker}: map book "
                f"{map_width_c:.0f}c wide > {MAP_WIDE_BLOCK_C:.0f}c — taker "
                f"suppressed (hedge is not priceable off this book; quoter "
                f"unaffected)")
        return True

    def _find_window_baseline(self, current_ts: float, lookback_s: float):
        """Return the most-recent window history entry that's ≥ lookback_s
        old, or None if no entry is old enough yet (still warming up)."""
        for past in reversed(self._window_history):
            if current_ts - past[0] >= lookback_s:
                return past
        return None

    def _series_dominance_override(self, direction: str, d_trigger: str,
                                    current_ts: float) -> tuple:
        """Override a 'map' trigger to 'series' when the past 5s shows
        series clearly dominated the hedge move. Returns (new_trigger, log_str).

        Rule (per user 2026-06-14):
            Δseries_favorable ≥ MIN_SERIES_MOVE
            AND Δseries_favorable ≥ DOMINANCE_RATIO × |Δhedge_c|
            → reclassify map → series

        Direction-aware: only series moves in the favorable direction for
        the leg count (price DROPS, since both d1=buy-YES and d2=buy-NO
        want their ask to fall). An adverse series move (e.g. d1 ask
        81→88) leaves the original "map" trigger intact even if the
        magnitude would dominate — that's a genuine map signal firing
        into a series-deteriorating window, not a series-led arb.

        Only overrides 'map' → 'series'; never the reverse. 'series', 'init',
        and '' are pass-through. Returns ('', '') as a no-op when not
        overriding so the caller can preserve the original trigger string."""
        if d_trigger != "map":
            return d_trigger, ""
        baseline = self._find_window_baseline(current_ts, self._SERIES_WINDOW_S)
        if baseline is None:
            return d_trigger, ""  # still warming up — fall through to instantaneous
        _ts_base, b_d1_price, b_d1_hedge_t, b_d2_price, b_d2_hedge_t = baseline
        if direction == "d1":
            if self._last_d1 is None:
                return d_trigger, ""
            cur_price, cur_hedge_t = self._last_d1
            base_price, base_hedge_t = b_d1_price, b_d1_hedge_t
        else:
            if self._last_d2 is None:
                return d_trigger, ""
            cur_price, cur_hedge_t = self._last_d2
            base_price, base_hedge_t = b_d2_price, b_d2_hedge_t
        # A baseline sitting on the empty-book sentinel is NOT a series move.
        # When the Kalshi book blacks out, best_d*_price defaults to
        # _BLACKOUT_PRICE; the refill back to a real price then reads here as a
        # huge favorable delta and dominates any hedge move, silently
        # reclassifying a genuine map trigger as 'series'. Observed live on
        # TTEDG 2026-08-21 07:27:42: "Δseries↓=63.0c ≥ 2×|Δhedge|=0.0c" was the
        # sentinel refill (100 -> 39) logged one second after
        # [POST-BLACKOUT RESET] reported that exact transition on the same leg.
        # Same guard improvement_gate already applies — see _BLACKOUT_PRICE.
        if base_price >= _BLACKOUT_PRICE and cur_price < _BLACKOUT_PRICE:
            return d_trigger, ""
        # Signed series delta — favorable = ask dropped (cheaper to buy).
        # Both legs buy at an ask, so favorable = base_price - cur_price > 0.
        # Hedge delta stays magnitude-only since either-direction hedge moves
        # count toward the comparison.
        delta_series_c = base_price - cur_price
        delta_hedge_c = abs(cur_hedge_t - base_hedge_t) / 10.0
        if delta_series_c < self._SERIES_MIN_MOVE_C:
            return d_trigger, ""
        if delta_series_c >= self._SERIES_DOMINANCE_RATIO * delta_hedge_c:
            return ("series",
                    f"[SERIES-DOMINANCE {direction.upper()}] "
                    f"Δseries↓={delta_series_c:.1f}c ≥ "
                    f"{self._SERIES_DOMINANCE_RATIO:.0f}×|Δhedge|={delta_hedge_c:.1f}c "
                    f"over {self._SERIES_WINDOW_S:.0f}s window — reclassifying map→series")
        return d_trigger, ""

    def _is_cluster_fire(self, new_snap, last_fire_snap, last_fire_ts, cooldown_sec) -> bool:
        """Additive suppression on top of the improvement gate.

        Truth table (relative to last fire snapshot, within cooldown window):
          price improved, hedge improved → ALLOW
          price improved, hedge same     → ALLOW
          price improved, hedge worse    → ALLOW
          price same,     hedge improved → ALLOW
          price same,     hedge same     → SUPPRESS  (classic flash-fade)
          price same,     hedge worse    → SUPPRESS  (hedge fade/flash)
          price worse,    hedge improved → ALLOW
          price worse,    hedge same     → SUPPRESS  (series fade/flash)
          price worse,    hedge worse    → SUPPRESS  (paying more on a weaker
                             signal — the BLGJDG chase row. Was ALLOW as "new
                             scenario" until 2026-08-02: the 12:30 burst's
                             30→32c fires with a fading hedge all landed here.)

        Returns True iff fire should be SUPPRESSED.

        For both directions: lower value = better. D1 snap stores
        (series_ask, hedge_1_tenths); D2 stores (series_no_ask, hedge_2_tenths).
        Lower series_ask = we pay less to buy YES. Lower series_no_ask = we pay
        less to buy NO. Hedge tenths: lower = cheaper hedge. So the comparison
        is uniform: less = better.

        cooldown_sec=0 disables the gate entirely (returns False = allow).
        last_fire_snap=None means we've never fired in this direction yet (allow).
        After cooldown_sec elapses since last fire, gate releases (allow).
        """
        if last_fire_snap is None or cooldown_sec <= 0:
            return False
        if time.time() - last_fire_ts >= cooldown_sec:
            return False
        new_price, new_hedge = new_snap
        old_price, old_hedge = last_fire_snap
        if new_price < old_price or new_hedge < old_hedge:
            return False  # at least one dimension improved
        return True  # nothing improved (incl. both-worse) → cluster

    def _material_refire(self, snap, last_snap, last_pos, pos_now,
                         refires_used, avail_lots, want_sign) -> tuple:
        """May a cluster-suppressed fire re-fire anyway? (allow, reason).

        Only the truth table's SUPPRESS rows reach this. Allows exactly the
        (price same, hedge same) genuine-refill case:
          - previous fire verifiably FILLED: net position moved ≥
            _refire_min_fill in the fire direction (want_sign +1 for D1 buys,
            -1 for D2). Zero-fills/phantoms never re-fire.
          - price not worse and hedge not worse than at the fire (whipsaw and
            fade rows stay suppressed).
          - real displayed depth ≥ _refire_min_depth at the fire level.
          - circuit breaker: ≤ _refire_max per cooldown window.
        Knobs come from config (refire_max_per_window / refire_min_fill_lots /
        refire_min_depth_lots) with defaults set in __init__.
        """
        if refires_used >= self._refire_max:
            return False, f"cap({refires_used})"
        if snap is None or last_snap is None or last_pos is None or pos_now is None:
            return False, "no-baseline"
        if snap[0] > last_snap[0]:
            return False, "price-worse"
        if snap[1] > last_snap[1]:
            return False, "hedge-worse"
        moved = (pos_now - last_pos) * want_sign
        if moved < self._refire_min_fill:
            return False, f"no-fill({moved:+.0f})"
        if avail_lots < self._refire_min_depth:
            return False, f"thin({avail_lots})"
        return True, f"filled{moved:+.0f} depth={avail_lots}"

    def _cluster_gate(self, direction: str, snap, improved: bool, avail_lots: int,
                      net_pos, cooldown_sec: float) -> bool:
        """Cluster-cooldown decision for one direction (True = suppress).

        Wraps _is_cluster_fire with the conditional-refire override and owns
        the counter and logging, so evaluate() carries one call per direction
        and the whole path is unit-testable. A re-fire from the budget is only
        consumed when `improved` is set (a non-candidate never spends one)."""
        if direction == "d1":
            last_snap, last_ts, last_pos = (self._last_fire_d1,
                                            self._last_fire_d1_ts,
                                            self._last_fire_d1_pos)
            refires, sign, best = self._refires_d1, +1, self._best_fire_d1
        else:
            last_snap, last_ts, last_pos = (self._last_fire_d2,
                                            self._last_fire_d2_ts,
                                            self._last_fire_d2_pos)
            refires, sign, best = self._refires_d2, -1, self._best_fire_d2
        # DUST GUARD (2026-08-02, operator-caught): price improvement only
        # counts when the improved level shows real size. A 1-lot flash at 32c
        # after fires at 33-36c would otherwise flip the truth table to its
        # price-improved ALLOW row AND bypass the decay veto's price escape —
        # re-arming a full ladder sweep into 33-36c off one lot of
        # "improvement" (which an informed refiller can post on purpose).
        # When the candidate ask beats the last fire but displayed size at it
        # is under _refire_min_depth, the gate evaluates as if the ask were
        # unchanged: dust unlocks nothing. (avail_lots = top-of-book size at
        # the fire level, same feed the refill proof uses.)
        eff = snap
        if (snap is not None and last_snap is not None
                and snap[0] < last_snap[0]
                and avail_lots < self._refire_min_depth):
            eff = (last_snap[0], snap[1])
        # Decay veto: within an active window, a fire at a no-better ask whose
        # hedge has faded > _fade_tenths past the window-BEST fire is dead —
        # truth-table rows and the conditional-refire override never see it.
        # Compared against the window best, not the last fire, so a fading
        # hedge cannot re-arm the gate by wobbling under the previous fire.
        stats = self._win_stats_d1 if direction == "d1" else self._win_stats_d2

        def _count(kind):
            if stats is not None:
                stats[kind] += 1
                if eff is not snap:
                    stats["dust"] += 1

        if (eff is not None and best is not None and cooldown_sec > 0
                and last_snap is not None
                and time.time() - last_ts < cooldown_sec
                and eff[0] >= best[0] and eff[1] > best[1] + self._fade_tenths):
            _count("veto")
            log.warning(f"[ARBER DECAY SUPPRESS] {self.config.ticker} "
                        f"{direction.upper()}: hedge {eff[1] / 10:.1f}c faded vs "
                        f"window-best {best[1] / 10:.1f}c at no-better ask "
                        f"({eff[0]}c >= {best[0]}c"
                        f"{', dust-clamped from ' + str(snap[0]) + 'c' if eff is not snap else ''}) "
                        f"— suppressed, no override "
                        f"(age={time.time() - last_ts:.1f}s"
                        f"{', n_veto=' + str(stats['veto']) if stats else ''})")
            return True
        suppressed = self._is_cluster_fire(eff, last_snap, last_ts, cooldown_sec)
        if not (improved and suppressed):
            if suppressed:
                # improved=False table suppression: kept quiet per-event (the
                # [ARBER GATE BLOCK] line already carries cluster_blocks=T),
                # but it must show up in the window summary.
                _count("table")
            return suppressed
        ok, why = self._material_refire(eff, last_snap, last_pos, net_pos,
                                        refires, avail_lots, sign)
        tag = direction.upper()
        if ok:
            if direction == "d1":
                self._refires_d1 += 1
                used = self._refires_d1
            else:
                self._refires_d2 += 1
                used = self._refires_d2
            if stats is not None:
                stats["refire"] += 1
            log.warning(f"[ARBER REFIRE-OK] {self.config.ticker} {tag}: {why} "
                        f"(refire {used}/{self._refire_max}, "
                        f"age={time.time() - last_ts:.1f}s) — cooldown overridden")
            return False
        _count("table")
        log.warning(f"[ARBER CLUSTER SUPPRESS] {self.config.ticker} {tag}: "
                    f"snap={snap} vs last_fire={last_snap} "
                    f"{'(dust-clamped to ' + str(eff[0]) + 'c) ' if eff is not snap else ''}"
                    f"(age={time.time() - last_ts:.1f}s, cooldown={cooldown_sec}s, "
                    f"refire={why}) — suppressed")
        return True

    def _note_cluster_fire(self, direction: str, snap, net_pos,
                           cooldown_sec: float) -> None:
        """Record a fire for the cluster gate. A fire opening a FRESH cooldown
        window resets that direction's conditional-refire budget."""
        now = time.time()

        def _close_window(tag, st, prev_ts):
            """One greppable summary line per burst, emitted when the NEXT
            burst opens (a window with no suppressions closes silently)."""
            if st and (st["veto"] or st["table"]):
                log.warning(
                    f"[ARBER GATE WINDOW] {self.config.ticker} {tag}: burst "
                    f"closed ({now - prev_ts:.0f}s after last fire) — "
                    f"{st['fires']} fire(s), suppressed {st['veto'] + st['table']} "
                    f"eval(s) [decay-veto {st['veto']}, table {st['table']}, "
                    f"dust-clamped {st['dust']}, refires-allowed {st['refire']}]")

        if direction == "d1":
            if now - self._last_fire_d1_ts >= cooldown_sec:
                _close_window("D1", self._win_stats_d1, self._last_fire_d1_ts)
                self._win_stats_d1 = {"fires": 1, "veto": 0, "table": 0,
                                      "dust": 0, "refire": 0}
                self._refires_d1 = 0
                self._best_fire_d1 = snap        # fresh window: best restarts here
            else:
                if self._win_stats_d1 is None:
                    self._win_stats_d1 = {"fires": 0, "veto": 0, "table": 0,
                                          "dust": 0, "refire": 0}
                self._win_stats_d1["fires"] += 1
                if self._best_fire_d1 is None:
                    self._best_fire_d1 = snap
                else:
                    self._best_fire_d1 = (min(self._best_fire_d1[0], snap[0]),
                                          min(self._best_fire_d1[1], snap[1]))
            self._last_fire_d1, self._last_fire_d1_ts = snap, now
            self._last_fire_d1_pos = net_pos
        else:
            if now - self._last_fire_d2_ts >= cooldown_sec:
                _close_window("D2", self._win_stats_d2, self._last_fire_d2_ts)
                self._win_stats_d2 = {"fires": 1, "veto": 0, "table": 0,
                                      "dust": 0, "refire": 0}
                self._refires_d2 = 0
                self._best_fire_d2 = snap
            else:
                if self._win_stats_d2 is None:
                    self._win_stats_d2 = {"fires": 0, "veto": 0, "table": 0,
                                          "dust": 0, "refire": 0}
                self._win_stats_d2["fires"] += 1
                if self._best_fire_d2 is None:
                    self._best_fire_d2 = snap
                else:
                    self._best_fire_d2 = (min(self._best_fire_d2[0], snap[0]),
                                          min(self._best_fire_d2[1], snap[1]))
            self._last_fire_d2, self._last_fire_d2_ts = snap, now
            self._last_fire_d2_pos = net_pos

    def refresh_probabilities(self, new_probs: dict = None) -> bool:
        """Update cached pre-game probs in-place from a fresh esports_probabilities.csv read.
        Safe on transient miss: if the ticker isn't in the new map, leave existing state untouched.
        Returns True if the bot's cached values were updated."""
        if new_probs is None:
            from hedge_engine import load_probabilities
            new_probs = load_probabilities()
        probs = new_probs.get(self.config.ticker)
        if not probs:
            return False
        self.p1_start = probs["p1"]
        self.p2_start = probs["p2"]
        self.series_start = probs["series"]
        self.g2_momentum = probs["g2_momentum"]
        self.g3_momentum = probs["g3_momentum"]
        self.p3_start = probs["p3"]
        self.p4_start = probs.get("p4")
        self.p5_start = probs.get("p5")
        self.is_bo5 = probs.get("bo5", False)
        self._verified = probs["verified"]
        if not self.baseline_loaded:
            self.baseline_loaded = True
            self.active = True
            if "GAME" in self.config.ticker:
                log.info(f"[ARBER MATH ENGINE] {self.config.ticker} | RELOAD activated: "
                         f"p1={probs['p1']*100:.1f}% p2={probs['p2']*100:.1f}% "
                         f"series={probs['series']*100:.1f}%")
        else:
            if "GAME" in self.config.ticker:
                log.info(f"[ARBER MATH ENGINE] {self.config.ticker} | RELOAD: "
                         f"p1={probs['p1']*100:.1f}% p2={probs['p2']*100:.1f}% "
                         f"series={probs['series']*100:.1f}%")
        return True

    def _load_pre_game_probabilities(self):
        from hedge_engine import load_probabilities

        self.baseline_loaded = False
        probs = load_probabilities().get(self.config.ticker)
        if probs:
            self.p1_start = probs["p1"]
            self.p2_start = probs["p2"]
            self.series_start = probs["series"]
            self.g2_momentum = probs["g2_momentum"]
            self.g3_momentum = probs["g3_momentum"]
            self.p3_start = probs["p3"]
            self.p4_start = probs.get("p4")
            self.p5_start = probs.get("p5")
            self.is_bo5 = probs.get("bo5", False)
            self._verified = probs["verified"]
            self.baseline_loaded = True

        if not self.baseline_loaded:
            if "GAME" in self.config.ticker:
                log.error(f"--- FATAL RISK HALT --- | No pre-game probabilities for {self.config.ticker}. Arber deactivating.")
            self.active = False
        else:
            if "GAME" in self.config.ticker:
                log.info(f"[ARBER MATH ENGINE] {self.config.ticker} | Loaded CSV Map Baseline Bounds: Map 1 ({self.p1_start*100:.1f}%), Map 2 ({self.p2_start*100:.1f}%), Series Anchor ({self.series_start*100:.1f}%)")

    def _update_pregame_latch(self, series_mid: float, hedge_cost: float) -> None:
        """Latch pregame→in-game. Records first-seen (series_mid, hedge) as init;
        flips self._in_pregame False once EITHER moves >= config.pregame_block_cents
        from init. Once in-game, stays in-game (latched). Cheap; called every cycle
        regardless of whether the gate is enabled, so `phase` is always tracked."""
        if not self._in_pregame:
            return
        if self._pregame_init_mid is None:
            self._pregame_init_mid = series_mid
            self._pregame_init_hedge = hedge_cost
            return
        thresh = self.config.pregame_block_cents
        if (abs(series_mid - self._pregame_init_mid) >= thresh or
                abs(hedge_cost - self._pregame_init_hedge) >= thresh):
            self._in_pregame = False
            log.info(f"[PREGAME END] {self.config.ticker} | series/hedge moved "
                     f">={thresh:g}c from init — in-game (takers enabled)")


    def _log_book_diag(self, tag, ticker, opp_ticker, top_level_bid, top_level_offer,
                       full_top_bids, full_top_offers, full_market_state,
                       series_eval, computed, orders):
        """ONE line per FIRE: every book representation this bot can see, side by side.

        Added 2026-07-31 to find why the arber prices fires below the real ask
        (65% of its IOCs zero-fill, vs 53% for momentum, which reads only the
        manager dict). The arber mixes FOUR sources and takes a min() across
        two of them, so whenever they disagree it selects the stale-lower one:
          dict   — full_top_bids/offers (what momentum uses)
          scalar — top_level_bid, captured separately in manager.py:1127
          raw    — raw_ob orderbook_fp ladder (what the sweep actually walks)
          eval   — series_eval synthetic costs (rebuilt from raw_ob upstream)
        `first` vs `best` on the raw ladder tests whether run.py:162's y[0][0]
        assumption (list already sorted desc) holds; if first != best the
        manager's top-of-book is simply reading the wrong level.
        Diagnostic only — must never raise into the trading path.
        """
        try:
            def raw(tk, side):
                ob = ((full_market_state or {}).get(tk, {}) or {}).get("raw_ob", {}) or {}
                pts = (ob.get("orderbook_fp", {}) or {}).get(side, []) or []
                if not pts:
                    return "none"
                try:
                    first = int(round(float(pts[0][0]) * 100))
                    best = int(round(max(float(x[0]) for x in pts) * 100))
                    flag = "" if first == best else " UNSORTED!"
                    return f"first={first} best={best} n={len(pts)}{flag}"
                except Exception as e:
                    return f"parse-err:{type(e).__name__}"
            se_used = bool(series_eval and "hedge_profile_long" in series_eval)
            se = ""
            if se_used:
                se = (f" syn_long={series_eval.get('synthetic_long_cost')} "
                      f"syn_short={series_eval.get('synthetic_short_cost')}")
            od = ",".join(f"{o.get('ticker','?').split('-')[-1]}:{o.get('kalshi_side','?')}@{o.get('limit_cents','?')}"
                          for o in (orders or [])[:6])
            log.info(
                f"[BOOKDIAG {tag}] {ticker} opp={opp_ticker} | "
                f"DICT self_bid={(full_top_bids or {}).get(ticker)} self_ask={(full_top_offers or {}).get(ticker)} "
                f"opp_bid={(full_top_bids or {}).get(opp_ticker)} opp_ask={(full_top_offers or {}).get(opp_ticker)} | "
                f"SCALAR bid={top_level_bid} offer={top_level_offer} | "
                f"RAWself yes[{raw(ticker,'yes_dollars')}] no[{raw(ticker,'no_dollars')}] | "
                f"RAWopp yes[{raw(opp_ticker,'yes_dollars')}] no[{raw(opp_ticker,'no_dollars')}] | "
                f"EVAL used={se_used}{se} | COMPUTED {computed} | ORDERS {od}")
        except Exception as e:
            log.debug(f"[BOOKDIAG] failed: {type(e).__name__}: {e}")

    def evaluate(self, market_state: Any, top_level_bid: float, top_level_offer: float, bid_theo: float, offer_theo: float, full_market_state: Dict[str, Any] = None, full_top_bids: Dict[str, float] = None, full_top_offers: Dict[str, float] = None, series_eval: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        # Retry CSV probability load every 10 minutes for deactivated bots
        if not self.active and not self.baseline_loaded:
            # Note: do NOT add a local `import time` here. Python treats `time`
            # as a function-local variable across all of evaluate() if any
            # import time appears in this scope, which UnboundLocalErrors any
            # time.time() use elsewhere in the function. Module-level
            # `import time` at the top of this file already covers all uses.
            now = time.time()
            if now - self._last_prob_retry >= 600:
                self._last_prob_retry = now
                self._load_pre_game_probabilities()
                if self.baseline_loaded:
                    self.active = True
                    log.info(f"[ARBER REVIVED] {self.config.ticker} | Probabilities found on retry — reactivating!")

        if not self.active or not full_top_bids or not full_top_offers:
            return []

        # ── Theo-sanity guards (added 2026-06-16 after VGYB blowup) ──
        # Mirror of bot.py guards so the arber's taker path doesn't fire on
        # inverted theos either. See bot.py for the full incident write-up:
        # VGYB at 13:33-13:34 had bid_theo > offer_theo for 8 seconds while
        # cons_map_no + cons_map_yes summed to 63 (instead of ~100), causing
        # the bot to self-trade ~700 lots at -1c/round-trip.
        # Sum >100 is fine (wide market). Sum <98 = one ticker is stale.
        try:
            if (isinstance(bid_theo, (int, float)) and isinstance(offer_theo, (int, float))
                    and bid_theo > offer_theo):
                log.error(
                    f"[THEO-INVERTED HALT] {self.config.ticker} "
                    f"bid_theo={bid_theo:.1f} > offer_theo={offer_theo:.1f} — arber refusing"
                )
                return []
            _cmn = (series_eval or {}).get("cons_map_no_ask")
            _cmy = (series_eval or {}).get("cons_map_yes_ask")
            if (isinstance(_cmn, (int, float)) and isinstance(_cmy, (int, float))
                    and (_cmn + _cmy) < 98.0):
                log.error(
                    f"[MAP-INCONSISTENT HALT] {self.config.ticker} "
                    f"cons_map_no={_cmn:.1f} + cons_map_yes={_cmy:.1f} = {_cmn + _cmy:.1f} (<98) — "
                    f"one ticker is stale, arber refusing"
                )
                return []
        except Exception:
            pass  # never let sanity guards break the arber on edge-case types

        # We assume self.config.ticker is the Series ticker for a specific team.
        # Format: KXLOLGAME-YYMMDDTEAMATEAMB-TEAMA
        ticker = self.config.ticker
        if "GAME" not in ticker and "MATCH" not in ticker:
            # We ONLY anchor the Arber on the Series/Match ticker to prevent duplicate
            # inverse firing from the Map/Set tickers.
            return []

        parts = ticker.split("-")
        if len(parts) < 3:
            return []

        # Hard position cap at 1.5× max_position with 1.4× release hysteresis.
        # For non-tennis tickers, switch to reduce-only firing once |pos| ≥
        # 1.5×max_position — only the series direction that shrinks |pos| is
        # allowed (long → direction_sign=-1 only, short → direction_sign=+1 only).
        # Stay in reduce-only mode until |pos| < 1.4×max_position, then resume.
        # The gap prevents flapping right at the boundary. For tennis tickers
        # (MATCH-...) the arb trades SET tickers, not the match, so the match-
        # level cap can't be cleanly translated to a per-set reduce side;
        # preserve the original full halt there (with the same hysteresis).
        # Game-decided / stale-book detection is handled centrally in
        # hedge_engine.detect_bo3_state via the loose-5c rule + live-M2 hedge math.
        reduce_only_direction = 0  # 0 = unrestricted; +1 = only buy YES; -1 = only buy NO
        try:
            cur_pos = position_store.get_position(ticker)
            hard_cap = int(1.5 * self.config.max_position)
            release_cap = int(1.4 * self.config.max_position)
            abs_pos = abs(cur_pos) if cur_pos is not None else 0
            if self._reduce_only_active:
                if abs_pos < release_cap:
                    self._reduce_only_active = False
                    log.warning(f"[STOP-CAP] {ticker}: |pos|={abs_pos} < "
                                f"1.4×max_position={release_cap} — released")
            elif cur_pos is not None and abs_pos >= hard_cap:
                self._reduce_only_active = True
                if "MATCH" in ticker:
                    log.warning(f"[STOP-CAP] {ticker}: |pos|={abs_pos} ≥ "
                                f"1.5×max_position={hard_cap} — halting (tennis)")
                else:
                    entry_dir = -1 if cur_pos > 0 else 1
                    log.warning(f"[STOP-CAP] {ticker}: |pos|={abs_pos} ≥ "
                                f"1.5×max_position={hard_cap} — reduce-only "
                                f"(direction_sign={entry_dir:+d})")
            if self._reduce_only_active and cur_pos is not None:
                if "MATCH" in ticker:
                    return []
                reduce_only_direction = -1 if cur_pos > 0 else 1
        except Exception:
            pass

        series_base = parts[0] + "-" + parts[1]
        team_suffix = parts[2]
        
        # Map/Set tickers: KXLOLMAP-...-1-TEAM, KXATPSETWINNER-...-1-TEAM
        # 2026-07-28: rewrite the PREFIX only. series_base.replace(...) mangles
        # the date+teams half when a TEAM NAME contains the token — "2GAME
        # Esports" turned ...SR2GAME into ...SR2MAP, a ticker that 404s.
        # See memory/project_map_base_game_substring_collision.md
        _mpfx = parts[0]
        if "MATCH" in _mpfx:
            map_base = _mpfx.replace("MATCH", "SETWINNER") + "-" + parts[1]
        else:
            map_base = _mpfx.replace("GAME", "MAP") + "-" + parts[1]
        
        map1_ticker = f"{map_base}-1-{team_suffix}"
        map2_ticker = f"{map_base}-2-{team_suffix}"
        map3_ticker = f"{map_base}-3-{team_suffix}"

        map4_ticker = f"{map_base}-4-{team_suffix}"
        map5_ticker = f"{map_base}-5-{team_suffix}"

        is_bo5 = self.is_bo5  # set from probabilities CSV col 6 (populated for Bo5 events)

        # BO5 runtime kill-switch (belt-and-suspenders with live_series_model).
        # Even if a BO5 row exists in template_quoter_config.csv, the
        # disable_bo5.flag halts all arber firing for it before any state
        # detection, hedge math, or order construction runs.
        if is_bo5 and bo5_disabled():
            if not getattr(self, '_bo5_disabled_logged', False):
                log.info(f"[ARBER BO5-DISABLED] {ticker}: disable_bo5.flag present — refusing to trade")
                self._bo5_disabled_logged = True
            return []

        # Get top offers (cost to BUY YES)
        series_ask = full_top_offers.get(ticker, 100.0)
        m1_ask = full_top_offers.get(map1_ticker, 100.0)
        m2_ask = full_top_offers.get(map2_ticker, 100.0)
        m3_ask = full_top_offers.get(map3_ticker, 100.0)
        m4_ask = full_top_offers.get(map4_ticker, 100.0) if is_bo5 else 100.0
        m5_ask = full_top_offers.get(map5_ticker, 100.0) if is_bo5 else 100.0

        m1_no_ask = 100.0 - full_top_bids.get(map1_ticker, 0.0)
        m2_no_ask = 100.0 - full_top_bids.get(map2_ticker, 0.0)
        m3_no_ask = 100.0 - full_top_bids.get(map3_ticker, 0.0)

        # Series NO ask = 100 - Series BID
        series_no_ask = 100.0 - top_level_bid

        m1_bid = full_top_bids.get(map1_ticker, 0.0)
        m2_bid = full_top_bids.get(map2_ticker, 0.0)

        # Forfeit detection via centralized oracle (general + CS2 Poly).
        # data_source from config drives the CS2 check timing (Tier 1 = 24/7, Tier 2 = EU only).
        forfeited, forfeit_reason = evaluate_forfeit(
            series_base, map1_ticker, map2_ticker,
            full_top_bids, full_top_offers,
            series_ask=series_ask, series_no_ask=series_no_ask,
            data_source=self.config.data_source,
            disable_forfeit_check=self.config.disable_forfeit_check,
        )
        if forfeited:
            if DEBUG_MODE:
                log.info(f"[ARBER FORFEIT] {ticker} reason={forfeit_reason}")
            return []


        # ── State Detection ──
        # FURIA debug — preserved for live debugging
        if "FURIA" in ticker and not getattr(self, '_furia_debug_fired', False):
            print("[DEBUG FURIA STATE INIT] ticker:", ticker)
            self._furia_debug_fired = True

        # State detection — all BO5 and BO3 paths go through live_series_model
        # (which calls hedge_engine.detect_bo5_state or detect_bo3_state). Single
        # source of truth: arber and quoter both consume series_eval, so they
        # cannot diverge on state.
        if series_eval is not None and "state" in series_eval:
            state = series_eval["state"]
            active_map_ticker = series_eval["active_map"]
            if DEBUG_MODE:
                log.info(f"[ARBER STATE] {ticker} from series_eval state={state} active_map={active_map_ticker}")
            # state == 3 only fires for BO3 1-1 deciding game (no separate
            # M3 market). BO5 states are tuples, so this comparison is False
            # for BO5. active_map_ticker is None when state detection refused
            # (series decided, G5 reached for BO5, forfeit, etc.).
            if state == 3 or active_map_ticker is None:
                return []
            active_map_ask = full_top_offers.get(active_map_ticker, 100.0)
            active_map_bid = full_top_bids.get(active_map_ticker, 0.0)
            active_map_no_ask = 100.0 - active_map_bid
        elif is_bo5:
            # BO5 without series_eval — first-cycle race or live_series declined.
            # Don't fall back to detect_bo3_state (which would mis-detect a BO5
            # past G2 as BO3 state 3). Skip until live_series_model populates.
            return []
        else:
            # BO3 fallback (no series_eval): inline detect_bo3_state.
            # Preserves the original first-cycle / no-eval path for BO3.
            state_info = detect_bo3_state(
                map_base, team_suffix, full_market_state,
                full_top_bids, full_top_offers, series_base=series_base,
                data_source=self.config.data_source,
                disable_forfeit_check=self.config.disable_forfeit_check,
            )
            if DEBUG_MODE:
                log.info(f"[ARBER STATE] {ticker} inline state_info={state_info}")
            if state_info is None:
                return []
            state = state_info["state"]
            if state == 3 or state_info["active_map"] is None:
                return []
            active_map_ticker = state_info["active_map"]
            active_map_ask = full_top_offers.get(active_map_ticker, 100.0)
            active_map_bid = full_top_bids.get(active_map_ticker, 0.0)
            active_map_no_ask = 100.0 - active_map_bid

        # No-map-data safety (ALL formats — BO3 and BO5, 2026-07-14): if the active
        # map has no real Kalshi orderbook data, halt. full_top_offers.get(..., 100.0)
        # and full_top_bids.get(..., 0.0) return the defaults when the ticker isn't in
        # the feed — a real listed market always has some bid>0 or ask<100. Without a
        # live map to hedge on, the model falls back to a static/pregame theo with no
        # live grounding and quotes it — the LRS date-divergence incident (Kalshi-only
        # BO3 whose maps were on a different date) traded off a pregame theo after
        # game 1 was already decided. Was BO5-only (assumed BO1-3 error was "small");
        # that assumption was wrong for a fully-missing map book. Fail safe: no map
        # book → refuse to quote. Auto-resumes the cycle Kalshi lists the map data.
        if active_map_ask >= 99 and active_map_bid <= 0:
            log.warning(
                f"[HALT — NO KALSHI MAP] {ticker} state={state} "
                f"active_map={active_map_ticker} ask={active_map_ask:.1f} "
                f"bid={active_map_bid:.1f} — refusing to trade until live map data appears"
            )
            return []

        # Removed silent return on illiquid states to ensure logging still fires
        desired_quotes = []
        
        # Direction 1: Buy Series YES, Hedge with Map NO
        # To calculate exact shares mathematically, we use a simplified proxy. 
        # In a generic balanced scenario, 1 share Series = approx 1 share Map hedge.
        series_loaded = get_loaded_cost_cents(series_ask)
        
        # Determine strict threshold dynamically from CSV variables
        # E.g. min_edge = 3.0 means synthetic must cost less than 97 cents!
        min_edge = max(self.config.min_edge, self.config.min_absolute_edge, 0.0)
        threshold = 100.0 - min_edge
        
        # Find opponent series ticker for cross-book routing
        # Buying our YES and buying opponent NO are economically identical — route through cheapest side
        opp_series_ticker = None
        for k in (full_top_bids or {}):
            if k.startswith(series_base + "-") and k != ticker:
                opp_series_ticker = k
                break

        opp_raw_ob = {}
        if opp_series_ticker and full_market_state:
            opp_raw_ob = full_market_state.get(opp_series_ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})

        # NOTE: alphabetical dedup moved BELOW the hedge_staleness.update_snapshot
        # call so both teams' bots feed the staleness detector. Previously the
        # alphabetically-second bot returned here at line 826, before line 912's
        # update_snapshot — only one team ever contributed snapshots, so
        # _recompute_stale's `len(teams) < 2` guard structurally returned False
        # for every event. Fixed 2026-06-19 (MENGRIND incident).
        opp_suffix = opp_series_ticker.split("-")[-1] if opp_series_ticker else ""

        # position_store.get_position already nets across both tickers for esports
        # (e.g., HOTU pos = raw_HOTU - raw_LGC), so no further netting needed
        current_series_pos = position_store.get_position(ticker)
        net_series_pos = current_series_pos
        
        # ── Hedge Profile Dispatch ──
        # Three paths:
        # 1. series_eval (BO3 and BO5): read hedge profiles + synthetics from
        #    live_series_model — single source of truth for both bot families.
        # 2. Fallback: inline cross-book + calculate_state_X_hedge (BO3 only —
        #    BO5 is hard-gated above to require series_eval).
        hedge_profile_1 = None
        hedge_profile_2 = None
        series_no_loaded = get_loaded_cost_cents(series_no_ask)
        # Two-sided width of the map book THIS hedge was priced off. Set on both
        # dispatch paths below so the MAP_WIDE_BLOCK_C gate cannot become a
        # one-branch guard. None = could not be determined (see the gate).
        map_width_c = None

        if series_eval is not None and "hedge_profile_long" in series_eval:
            hedge_profile_1 = series_eval["hedge_profile_long"]
            hedge_profile_2 = series_eval["hedge_profile_short"]
            synthetic_1 = series_eval["synthetic_long_cost"]
            synthetic_2 = series_eval["synthetic_short_cost"]
            _cy = series_eval.get("cons_map_yes_ask")
            _cn = series_eval.get("cons_map_no_ask")
            if _cy is not None and _cn is not None:
                map_width_c = float(_cy) + float(_cn) - 100.0
            if DEBUG_MODE:
                log.info(f"[ARBER HEDGE] {ticker} from series_eval "
                          f"syn_long={synthetic_1:.2f} syn_short={synthetic_2:.2f}")
        else:
            # Fallback: inline cross-book + calculate_state_X_hedge
            opp_map_ticker = None
            active_map_parts = active_map_ticker.split("-")
            active_map_base = "-".join(active_map_parts[:-1])
            for k in (full_top_bids or {}):
                if k.startswith(active_map_base + "-") and k != active_map_ticker:
                    opp_map_ticker = k
                    break

            best_map_no_ask = active_map_no_ask
            best_map_ask = active_map_ask
            if opp_map_ticker:
                opp_map_ask_val = full_top_offers.get(opp_map_ticker, 100.0)
                opp_map_bid_val = full_top_bids.get(opp_map_ticker, 0.0)
                best_map_no_ask = min(active_map_no_ask, opp_map_ask_val)
                best_map_ask = min(active_map_ask, 100.0 - opp_map_bid_val)

            # Same quantity as the series_eval path: yes_ask + no_ask - 100.
            map_width_c = float(best_map_ask) + float(best_map_no_ask) - 100.0

            if state == 0:
                hedge_profile_1 = calculate_state_0_hedge(self.p1_start, self.p2_start, self.series_start, self.g2_momentum, self.g3_momentum, int(round(best_map_no_ask)))
                hedge_profile_2 = calculate_state_0_hedge(1.0 - self.p1_start, 1.0 - self.p2_start, 1.0 - self.series_start, self.g2_momentum, self.g3_momentum, int(round(best_map_ask)))
            elif state == 1:
                hedge_profile_1 = calculate_state_1_hedge(self.p1_start, self.p2_start, self.series_start, self.g2_momentum, self.g3_momentum, int(round(best_map_no_ask)))
                hedge_profile_2 = calculate_state_2_hedge(1.0 - self.p1_start, 1.0 - self.p2_start, 1.0 - self.series_start, self.g2_momentum, self.g3_momentum, int(round(best_map_ask)))
            elif state == 2:
                hedge_profile_1 = calculate_state_2_hedge(self.p1_start, self.p2_start, self.series_start, self.g2_momentum, self.g3_momentum, int(round(best_map_no_ask)))
                hedge_profile_2 = calculate_state_1_hedge(1.0 - self.p1_start, 1.0 - self.p2_start, 1.0 - self.series_start, self.g2_momentum, self.g3_momentum, int(round(best_map_ask)))
            elif state == 3:
                hedge_profile_1 = calculate_state_3_hedge(int(round(best_map_no_ask)))
                hedge_profile_2 = calculate_state_3_hedge(int(round(best_map_ask)))

            synthetic_1 = series_loaded + hedge_profile_1["synthetic_cost"] if hedge_profile_1 else 100.0
            synthetic_2 = series_no_loaded + hedge_profile_2["synthetic_cost"] if hedge_profile_2 else 100.0

        shares_b_1 = hedge_profile_1["shares_b"] if hedge_profile_1 else 1.0
        shares_b_2 = hedge_profile_2["shares_b"] if hedge_profile_2 else 1.0

        # Snapshot cache: detect which directions changed since last cycle
        # Direction 1 (buy our YES) sweeps our YES book + opponent NO book, hedges on map
        # Direction 2 (buy our NO) sweeps our NO book + opponent YES book, hedges on map
        # Include both sides' top-of-book so opponent book changes trigger re-evaluation
        h1_cost = hedge_profile_1["synthetic_cost"] if hedge_profile_1 else 0.0
        h2_cost = hedge_profile_2["synthetic_cost"] if hedge_profile_2 else 0.0
        opp_bid = full_top_bids.get(opp_series_ticker, 0.0) if opp_series_ticker else 0.0
        opp_ask = full_top_offers.get(opp_series_ticker, 100.0) if opp_series_ticker else 100.0
        # D1 (buy our YES): best price = min(our YES ask, 100 - opp YES bid)
        # D2 (buy our NO): best price = min(our NO ask, opp YES ask)
        # Use string formatting to eliminate floating point: "%.1f" then parse back to get clean values
        best_d1_price = int(min(round(series_ask), round(100.0 - opp_bid)))
        best_d2_price = int(min(round(series_no_ask), round(opp_ask)))
        h1_tenths = int(float(f"{h1_cost:.1f}") * 10)
        h2_tenths = int(float(f"{h2_cost:.1f}") * 10)
        d1_snap = (best_d1_price, h1_tenths)
        d2_snap = (best_d2_price, h2_tenths)

        # Pregame latch: series mid = (our YES ask + (100 - opp YES ask))/2; hedge = h1_cost.
        # Tracked every cycle (drives `phase` + the pregame-taker gate below).
        self._update_pregame_latch((series_ask + (100.0 - opp_ask)) / 2.0, h1_cost)

        # Per-event hedge-freeze detector (see hedge_staleness.py). Each per-team
        # arber feeds its OWN team's snapshot; both teams' bots together populate
        # the event-level state. If flagged, suppress all fires this cycle.
        # We still call update_snapshot so the rolling window stays fresh and
        # the stale state can auto-clear once hedge moves again.
        _now_ts = time.time()
        hedge_staleness.update_snapshot(series_base, team_suffix, _now_ts, best_d1_price, h1_cost)
        if hedge_staleness.is_stale(series_base):
            return []
        # Boot-hole guard (26AUG13): never fire until this event's Poly hedge
        # has been SEEN to tick in this process's lifetime. A restart during a
        # Poly freeze wipes the staleness state; in the re-arm window Kalshi's
        # live book vs an hour-dead hedge is a large fake edge fired into an
        # unhedgeable venue. Healthy hedges prove liveness within seconds.
        if not hedge_staleness.hedge_proven_live(series_base):
            return []

        # Alphabetical dedup — only one team's bot actually fires (see note above
        # the position_store call). MUST come after update_snapshot so both teams
        # feed the staleness detector.
        if opp_suffix and team_suffix > opp_suffix:
            return []

        if state != self._last_state:
            self._last_d1 = None
            self._last_d2 = None
            self._last_state = state
        d1_changed = (d1_snap != self._last_d1)
        d2_changed = (d2_snap != self._last_d2)
        prev_d1 = self._last_d1  # Save for fire logging
        prev_d2 = self._last_d2
        # Store raw components for fire log (not used in comparison)
        d1_detail = (int(round(series_ask)), int(round(100.0 - opp_bid)), h1_tenths)
        d2_detail = (int(round(series_no_ask)), int(round(opp_ask)), h2_tenths)
        prev_d1_detail = self._last_d1_detail if hasattr(self, '_last_d1_detail') else None
        prev_d2_detail = self._last_d2_detail if hasattr(self, '_last_d2_detail') else None
        # NOTE: _last_d1/_last_d2/_last_d*_detail are committed AFTER the
        # early-return guards below (can-trade, fake-move). Committing here
        # made those aborts consume the one-shot "improved" signal: the eval
        # that first saw a better hedge died in a guard, and every later eval
        # compared new==new -> improved=F until the market moved AGAIN
        # (GLYPLAT 26AUG03 07:54:12 -> :14.4, a 1.85s hole on a real edge).

        # Only allow firing when edge IMPROVED (price dropped or hedge got cheaper)
        # Also determine trigger attribution: series_move vs map_move
        d1_improved, d1_trigger, d1_blocked = improvement_gate(self._last_d1, d1_snap)
        d2_improved, d2_trigger, d2_blocked = improvement_gate(self._last_d2, d2_snap)
        for _dir, _blocked, _prev, _new in (("d1", d1_blocked, self._last_d1, d1_snap),
                                            ("d2", d2_blocked, self._last_d2, d2_snap)):
            if _blocked == "post_blackout":
                log.info(f"[POST-BLACKOUT RESET] {self.config.ticker} {_dir}: book "
                         f"refilled from the {_BLACKOUT_PRICE}c empty-book sentinel "
                         f"({_prev} -> {_new}) — re-baselining, NOT treating the "
                         f"refill as an improvement. Next genuine improvement can fire.")

        # (snapshot commit deferred — see note above; happens after the
        # fake-move guard so aborted evals keep the improvement pending)

        # ── Series-dominance window override (added 2026-06-14) ──
        # The two assignments above classify d1/d2_trigger by comparing the
        # CURRENT tick vs the PREVIOUS tick only. That misses the case where a
        # large series move (e.g. 10c over 5s) ends on a tick where the hedge
        # twitched a fraction; instantaneous = "map" but the dominant signal is
        # series. Without this override, the series_edge_mult never fires on
        # those decisions, and a thin +0.1c map-trigger edge gets the full
        # max_fire_size cap. Per the 2026-06-13 21:03 9ZTS-9Z incident
        # post-mortem, that's the single biggest source of catastrophic
        # 6000-lot trades. Only overrides map→series; never the reverse.
        _now_ts = time.time()
        new_d1_trig, d1_log = self._series_dominance_override("d1", d1_trigger, _now_ts)
        if new_d1_trig != d1_trigger:
            log.warning(f"{d1_log} | {self.config.ticker}")
            d1_trigger = new_d1_trig
        new_d2_trig, d2_log = self._series_dominance_override("d2", d2_trigger, _now_ts)
        if new_d2_trig != d2_trigger:
            log.warning(f"{d2_log} | {self.config.ticker}")
            d2_trigger = new_d2_trig
        # Push this cycle's snapshot to the window history for future overrides.
        if d1_snap is not None and d2_snap is not None:
            self._window_history.append(
                (_now_ts, d1_snap[0], d1_snap[1], d2_snap[0], d2_snap[1])
            )
        # FIRE-MOVE sampling (log-only): hedge tenths per direction, None when
        # that side's hedge profile is missing (h_cost would read a fake 0.0).
        self._note_fire_move_hist(
            _now_ts,
            h1_tenths if hedge_profile_1 is not None else None,
            h2_tenths if hedge_profile_2 is not None else None,
            state,
        )

        # ── Pregame-taker gate (config.pregame_taker_block, default OFF) ──
        # Suppress ALL arber taker fires while the event is pregame. Both fire paths
        # (cluster @ d{1,2}_improved and d{1,2}_cluster_fire; main @ d{1,2}_improved and
        # not d{1,2}_cluster_fire) require d{1,2}_improved, so zeroing them blocks every
        # taker this cycle. The maker/quoter is a separate class and is untouched.
        # Markout study 2026-07-14: pregame takes net ~-0.4c/contract (every trigger);
        # in-game +2.5c. See [[project_taker_trigger_toxicity_markout]].
        if self.config.pregame_taker_block and self._in_pregame and (d1_improved or d2_improved):
            log.info(f"[PREGAME BLOCK] {self.config.ticker} | taker fire suppressed "
                     f"(state {state}, d1_trig={d1_trigger or '-'} d2_trig={d2_trigger or '-'})")
            d1_improved = False
            d2_improved = False

        # (cluster-fire suppression moved below get_avail — the conditional
        # re-fire override needs refill depth at the fire level.)
        cooldown_sec = getattr(self.config, "maker_refill_cooldown_sec", 0.0)

        def get_avail(t_ticker: str, side: str) -> int:
            if not full_market_state: return 0
            ob = full_market_state.get(t_ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})
            pts = ob.get(side, [])
            return int(float(pts[0][1])) if pts else 0

        series_yes_avail = get_avail(ticker, "no_dollars")
        map_no_avail = get_avail(active_map_ticker, "yes_dollars")
        series_no_avail = get_avail(ticker, "yes_dollars")
        map_yes_avail = get_avail(active_map_ticker, "no_dollars")

        # Cluster-cooldown gate with conditional re-fire — one call per
        # direction; decision logic + logging live in _cluster_gate.
        d1_cluster_fire = self._cluster_gate("d1", d1_snap, d1_improved,
                                             series_yes_avail, net_series_pos, cooldown_sec)
        d2_cluster_fire = self._cluster_gate("d2", d2_snap, d2_improved,
                                             series_no_avail, net_series_pos, cooldown_sec)

        series_status = full_market_state.get(ticker, {}).get("status", "")
        active_map_status = full_market_state.get(active_map_ticker, {}).get("status", "")
        # Real-Time Orderbooks derived straight from live Kalshi streaming cache
        raw_ob_1 = full_market_state.get(ticker, {}).get("raw_ob", {}).get("orderbook_fp", {}).get("no_dollars", [])
        raw_ob_2 = full_market_state.get(ticker, {}).get("raw_ob", {}).get("orderbook_fp", {}).get("yes_dollars", [])
        
        base_vol = self.config.volumes[0] if self.config.volumes else 100
        fire_cap = getattr(self.config, "max_fire_size", 500)
        scale_step = getattr(self.config, "arb_scale_step_cents", 1.0)
        
        # Legacy intended proxy bounds for fallback momentum / logs
        intended_vol_1 = min(base_vol, max(0, self.config.max_position - current_series_pos))
        intended_vol_2 = min(base_vol, max(0, self.config.max_position + current_series_pos))
        
        can_trade = (series_status == "active" and active_map_status == "active")

        # BO5 G4/G5 fallback: some BO5 events (e.g., Valorant THVIT G4) have map
        # winner markets only on Polymarket — Kalshi doesn't list them, so
        # active_map_status is "" even when the series is active and Poly is
        # providing live orderbook data. Permit trading when state is past G2
        # AND we have valid orderbook data. BO3 path is untouched because
        # is_bo5=False guards this branch.
        if (not can_trade and is_bo5 and isinstance(state, tuple)
                and state[0] + state[1] >= 3
                and series_status == "active"
                and active_map_status == ""
                and (active_map_ask < 100.0 or active_map_bid > 0.0)):
            can_trade = True
            log.info(f"[BO5-POLY-ONLY] {ticker} S{state}: no Kalshi {active_map_ticker}, "
                     f"using Poly data (bid={active_map_bid:.0f} ask={active_map_ask:.0f})")

        # If either leg is suspended, settled, or purged from the live stream, immediately halt structural execution and tracking logs
        if not can_trade:
            if "FURIA" in ticker: print(f"[DEBUG] Dropped can_trade false. series_status={series_status}, active_map_status={active_map_status}")
            if is_bo5:
                log.info(f"[BO5-CANTRADE] {ticker} blocked: state={state} active_map={active_map_ticker} "
                         f"series_status='{series_status}' active_map_status='{active_map_status}' "
                         f"map_bid={active_map_bid:.0f} map_ask={active_map_ask:.0f}")
            else:
                # BO3 path was previously silent — operator had no signal that an active
                # market was sitting unquoted. Most common cause: bulk /markets metadata
                # fetch 429'd and status stayed "" until next manager hot-reload (~30
                # min). Throttled to 1/min/ticker so persistent blocks stay visible
                # without spamming run.log.
                last_ts = self._last_cantrade_log_ts if hasattr(self, "_last_cantrade_log_ts") else 0.0
                now = time.time()
                if now - last_ts >= 60.0:
                    log.warning(f"[ARBER NO-TRADE] {ticker} blocked: state={state} active_map={active_map_ticker} "
                                f"series_status='{series_status}' active_map_status='{active_map_status}' "
                                f"map_bid={active_map_bid:.0f} map_ask={active_map_ask:.0f}")
                    self._last_cantrade_log_ts = now
            return []

        # FAKE-MOVE DETECTION: if the previous map spread was wide (>15c) and only
        # one side moved while the other stayed, it's a rogue order narrowing the
        # spread, not a real market move. Skip to avoid trading on phantom edge.
        # A real move shifts both bid and ask together.
        MAP_WIDE_THRESHOLD = 15  # cents — below this spread we trust the market
        MAP_MOVE_THRESHOLD = 3   # cents — minimum change to count as "moved"
        cur_map_bid = active_map_bid
        cur_map_ask = active_map_ask

        # If the previous spread was wide and only one side moved, it's a rogue order
        if self._prev_map_bid is not None and self._prev_map_ask is not None:
            prev_spread = self._prev_map_ask - self._prev_map_bid
            if prev_spread > MAP_WIDE_THRESHOLD:
                bid_moved = abs(cur_map_bid - self._prev_map_bid) >= MAP_MOVE_THRESHOLD
                ask_moved = abs(cur_map_ask - self._prev_map_ask) >= MAP_MOVE_THRESHOLD
                if bid_moved != ask_moved:
                    # Was a SILENT return — invisible in forensics. WS deltas
                    # land per level, so a real reprice passes through a
                    # one-sided instant; log it so the skip is auditable.
                    log.warning(
                        f"[ARBER FAKE-MOVE SKIP] {self.config.ticker}: wide map "
                        f"({self._prev_map_bid:.0f}@{self._prev_map_ask:.0f} -> "
                        f"{cur_map_bid:.0f}@{cur_map_ask:.0f}) moved one-sided — "
                        f"eval skipped (improvement kept pending)")
                    self._prev_map_bid = cur_map_bid
                    self._prev_map_ask = cur_map_ask
                    return []
        self._prev_map_bid = cur_map_bid
        self._prev_map_ask = cur_map_ask

        # ── MAP-WIDTH LEVEL BLOCK ──────────────────────────────────────────
        # Placed AFTER the _prev_map_* commit above so prev-tracking stays
        # accurate across blocked cycles, and BEFORE the _last_d1/_last_d2
        # commit below so a blocked eval keeps its pending improvement (same
        # contract as FAKE-MOVE SKIP).
        if self._map_wide_block(map_width_c):
            return []

        # Commit the improvement-gate snapshots ONLY now that every early
        # return is behind us: an eval that aborted above keeps _last_d1/_last_d2
        # unchanged, so the next eval still sees the pending improvement and can
        # fire (~0.4s later) instead of waiting for the market to move again.
        self._last_d1 = d1_snap
        self._last_d2 = d2_snap
        self._last_d1_detail = d1_detail
        self._last_d2_detail = d2_detail

        # --- TENNIS SET ARB: trade lagging sets off the leading match price ---
        is_tennis = "MATCH" in ticker
        if is_tennis:
            match_mid = (top_level_bid + series_ask) / 2.0
            if match_mid > 5 and match_mid < 95:
                match_prob = match_mid / 100.0
                implied_set_prob = solve_set_probability(match_prob)
                set_theo_cents = implied_set_prob * 100.0

                for set_ticker in [map1_ticker, map2_ticker]:
                    set_st = full_market_state.get(set_ticker, {}).get("status", "")
                    set_bid = full_top_bids.get(set_ticker, 0.0)
                    set_ask = full_top_offers.get(set_ticker, 100.0)

                    if set_st != "active" or (set_bid == 0 and set_ask == 100):
                        continue

                    # Find opponent set ticker for cross-book routing
                    set_parts = set_ticker.split("-")
                    set_base = "-".join(set_parts[:-1])
                    opp_set_ticker = None
                    for k in (full_top_bids or {}):
                        if k.startswith(set_base + "-") and k != set_ticker:
                            opp_set_ticker = k
                            break

                    opp_set_ob = {}
                    if opp_set_ticker and full_market_state:
                        opp_set_ob = full_market_state.get(opp_set_ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})

                    set_ob = full_market_state.get(set_ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})
                    set_pos = position_store.get_position(set_ticker)

                    # Direction A: Buy SET YES (set is cheap relative to match-implied theo)
                    # hedge_cost = 100 - set_theo so edge = set_theo - loaded_ask
                    books_a = [(set_ob.get("no_dollars", []), set_ticker, "yes")]
                    if opp_set_ticker and opp_set_ob.get("yes_dollars"):
                        books_a.append((opp_set_ob["yes_dollars"], opp_set_ticker, "no"))

                    orders_a = calculate_cross_book_sweep(
                        books=books_a, hedge_cost=100.0 - set_theo_cents,
                        min_edge=self.config.min_edge, scale_step_c=scale_step,
                        base_vol=base_vol, max_fire_size=fire_cap,
                        max_position=self.config.max_position,
                        current_pos=set_pos, direction_sign=1,
                        min_absolute_edge=self.config.min_absolute_edge,
                        fav_edge_k=self.config.fav_edge_k,
                    )
                    for order in orders_a:
                        desired_quotes.append(order)
                        log.info(f"TENNIS SET ARB: {order['ticker']} {order['kalshi_side'].upper()} {order['size']}x at {order['limit_cents']}c "
                                 f"[Set theo: {set_theo_cents:.1f}c from match mid {match_mid:.1f}c]")

                    # Direction B: Buy SET NO (set is expensive relative to match-implied theo)
                    # hedge_cost = set_theo so edge = (100-set_theo) - loaded_NO_cost
                    books_b = [(set_ob.get("yes_dollars", []), set_ticker, "no")]
                    if opp_set_ticker and opp_set_ob.get("no_dollars"):
                        books_b.append((opp_set_ob["no_dollars"], opp_set_ticker, "yes"))

                    orders_b = calculate_cross_book_sweep(
                        books=books_b, hedge_cost=set_theo_cents,
                        min_edge=self.config.min_edge, scale_step_c=scale_step,
                        base_vol=base_vol, max_fire_size=fire_cap,
                        max_position=self.config.max_position,
                        current_pos=set_pos, direction_sign=-1,
                        min_absolute_edge=self.config.min_absolute_edge,
                        fav_edge_k=self.config.fav_edge_k,
                    )
                    for order in orders_b:
                        desired_quotes.append(order)
                        log.info(f"TENNIS SET ARB: {order['ticker']} {order['kalshi_side'].upper()} {order['size']}x at {order['limit_cents']}c "
                                 f"[Set theo: {set_theo_cents:.1f}c from match mid {match_mid:.1f}c]")

                # Log tennis state
                if desired_quotes or DEBUG_MODE:
                    log.info(f"[TENNIS TRACKING] {ticker} (State {state}) | Match mid: {match_mid:.1f}c -> Implied set: {set_theo_cents:.1f}c | "
                             f"Set1: {full_top_bids.get(map1_ticker, 0):.0f}/{full_top_offers.get(map1_ticker, 100):.0f} | "
                             f"Set2: {full_top_bids.get(map2_ticker, 0):.0f}/{full_top_offers.get(map2_ticker, 100):.0f}")

            return desired_quotes

        # Direction 1: Long self via cross-book sweep (our YES book + opponent NO book merged)
        # The opponent's bot handles the inverse direction via its own Direction 1, preventing duplication.
        edge_1_approx = 100.0 - synthetic_1
        d1_is_reducing = net_series_pos < 0
        # Per-condition booleans so the diagnostic log can show which one is blocking.
        d1_can_trade_ok = bool(can_trade)
        d1_map_ok       = (active_map_no_ask < 99)
        d1_hedge_ok     = (hedge_profile_1 is not None)
        d1_throttle_ok  = self._check_fire_throttle(edge_1_approx)
        d1_reduce_ok    = (reduce_only_direction in (0, 1))
        d1_gate = (d1_can_trade_ok and d1_map_ok and d1_hedge_ok
                   and d1_improved and (not d1_cluster_fire)
                   and d1_throttle_ok and d1_reduce_ok)

        # G3 edge buffer: during G2 of a BO3 (state 1 = up 1-0, state 2 =
        # down 0-1; both mean G2 active), widen the required edge by
        # P(G3) * min_edge to protect against systematic G3 mispricing.
        # P(G3) = P(G1 loser wins G2):
        #   state 1 (we won G1): P(G3) = P(opp wins G2) = (100 - m2_mid_self)/100
        #   state 2 (we lost G1): P(G3) = P(we win G2) = m2_mid_self/100
        # The m1_won flag distinguishes the two — its boolean value already
        # encodes which perspective we're on. Source: series_eval (live_series
        # path) or state_info (BO3 fallback path). Applied to both d1 and d2.
        # Added 2026-05-27.
        g3_buf = 0.0
        g3_info = series_eval if (series_eval is not None and "state" in series_eval) else locals().get("state_info")
        if (not is_bo5) and state in (1, 2) and g3_info is not None:
            m2_mid_self = (active_map_bid + active_map_ask) / 2.0
            if g3_info.get("m1_won"):
                p_g3 = max(0.0, (100.0 - m2_mid_self) / 100.0)
            else:
                p_g3 = max(0.0, m2_mid_self / 100.0)
            g3_buf = _g3_edge_mult_for(self.config) * p_g3 * self.config.min_edge

        # Diagnostic — when edge > 0 but gate blocks, show every condition's value
        if edge_1_approx > 0 and not d1_gate:
            log.warning(
                f"[ARBER GATE BLOCK D1] {self.config.ticker} edge=+{edge_1_approx:.1f}c: "
                f"can_trade={'T' if d1_can_trade_ok else 'F'} "
                f"map_ask_ok={'T' if d1_map_ok else f'F(no_ask={active_map_no_ask})'} "
                f"hedge={'T' if d1_hedge_ok else 'F(None)'} "
                f"improved={'T' if d1_improved else 'F'} "
                f"cluster_blocks={'F' if not d1_cluster_fire else 'T'} "
                f"throttle={'T' if d1_throttle_ok else 'F'} "
                f"reduce_dir={reduce_only_direction}({'OK' if d1_reduce_ok else 'BLOCKED'}) "
                f"g3_buf={g3_buf:.2f}c"
            )

        # ── Taker retreat knob (TRADE_RETREAT_SCOPE.md, 26AUG18) ──
        # Per-row arm switch: retreat_cap_cents > 0 on THIS (arber) row makes
        # the taker gates honor the recency-flow theo shift the quoter side
        # applies. retreat_c is SIGNED for direction 1 (long this ticker's
        # team): + when d1 is the recently-accumulated side (its edge is
        # overstated by the shift), - when d1 is the cold side (understated).
        # d2 gets the negation. Rows with retreat_cap_cents=0 (all ARB/MOM
        # rows at ship time): retreat_c=0.0 and every gate is byte-identical.
        retreat_c = 0.0
        _r_cap = getattr(self.config, "retreat_cap_cents", 0.0) or 0.0
        if _r_cap > 0:
            _r_parts = ticker.split("-")
            if len(_r_parts) == 3:
                try:
                    import retreat_shadow
                    _rc, _rf0, _rfcap, _rhl, _rtk = retreat_shadow.params_from_conf(self.config)
                    _r_f = retreat_shadow.get_f(
                        "-".join(_r_parts[:2]), _r_parts[-1], self.config.max_position,
                        half_life=_rhl, include_takers=_rtk)
                    _r_mag = retreat_shadow.shift_cents(
                        abs(_r_f), cap_cents=_rc, f0=_rf0, f_cap=_rfcap)
                    retreat_c = _r_mag if _r_f > 0 else -_r_mag
                    if abs(retreat_c) > 0.05:
                        log.info(f"[RETREAT-TAKER] {ticker} f={_r_f:+.3f} "
                                 f"d1_shift={retreat_c:+.2f}c d2_shift={-retreat_c:+.2f}c")
                except Exception:
                    # Loud by design — never silently demote to no-retreat.
                    log.exception(f"[RETREAT-TAKER] shift computation FAILED for "
                                  f"{ticker} — taker gates running WITHOUT retreat")

        if d1_gate:
            books = [(raw_ob_1, ticker, "yes")]
            opp_yes_pts = opp_raw_ob.get("yes_dollars", [])
            if opp_series_ticker and opp_yes_pts:
                books.append((opp_yes_pts, opp_series_ticker, "no"))

            eff_min_edge_1 = self.config.min_edge * (self.config.series_edge_mult if d1_trigger == "series" else 1.0) + g3_buf
            # Position skew: shifts gate up for adding / down for reducing per side.
            # With skew_max_shift_cents=0 on the row, skew_c=0.0 and gates are unchanged.
            skew_c_d1 = compute_position_skew_shift_cents(net_series_pos, self.config)
            cross_orders = calculate_cross_book_sweep(
                books=books,
                hedge_cost=hedge_profile_1["synthetic_cost"],
                min_edge=eff_min_edge_1,
                scale_step_c=scale_step,
                base_vol=base_vol,
                max_fire_size=fire_cap,
                max_position=self.config.max_position,
                current_pos=net_series_pos,
                direction_sign=1,
                min_absolute_edge=self.config.min_absolute_edge,
                fav_edge_k=self.config.fav_edge_k,
                position_skew_cents=skew_c_d1,
                retreat_shift_cents=retreat_c,
            )

            total_sweep = sum(o["size"] for o in cross_orders)
            # Diagnostic: if the multiplier specifically pushed eff_min_edge above
            # the available edge (and no fire happened), surface it. Compares
            # mult-applied vs mult-stripped thresholds; only fires for the band
            # where the multiplier alone is the blocker.
            if (total_sweep == 0 and d1_trigger == "series"
                    and self.config.series_edge_mult > 1.0):
                base_thresh = self.config.min_edge + g3_buf
                if base_thresh <= edge_1_approx < eff_min_edge_1:
                    log.info(f"[SERIES-MULT BLOCK D1] {self.config.ticker} "
                             f"edge=+{edge_1_approx:.1f}c base_min={base_thresh:.1f}c "
                             f"eff_min={eff_min_edge_1:.1f}c "
                             f"(mult={self.config.series_edge_mult}x) — "
                             f"would have fired without multiplier")
            if total_sweep > 0:
                try:
                    gate_ok, gate_why = self._fire_move_gate(
                        "d1", d1_trigger, _now_ts, h1_cost,
                        net_series_pos, d1_is_reducing, state)
                except Exception as e:
                    gate_ok, gate_why = True, f"gate-error:{e}"
                if not gate_ok:
                    log.warning(f"[FIRE-GATE BLOCK] {self.config.ticker} D1/{d1_trigger} "
                                f"{gate_why} hedge={h1_cost:.1f}c "
                                f"{self._fire_move_str('d1', _now_ts, h1_cost, state)} "
                                f"pos={net_series_pos} reducing={d1_is_reducing} "
                                f"state={state} sweep={total_sweep}")
                    cross_orders = []
                    total_sweep = 0
                else:
                    self._record_fire(edge_1_approx)
                    self._note_cluster_fire("d1", d1_snap, net_series_pos, cooldown_sec)
                    log.info(f"[FIRE-MOVE] {self.config.ticker} D1/{d1_trigger} "
                             f"hedge={h1_cost:.1f}c "
                             f"{self._fire_move_str('d1', _now_ts, h1_cost, state)} "
                             f"pos={net_series_pos} reducing={d1_is_reducing} "
                             f"state={state} sweep={total_sweep} gate={gate_why}")
            elif g3_buf > 0:
                log.warning(f"[ARBER G3-BUF SUPPRESS D1] {self.config.ticker} "
                            f"edge=+{edge_1_approx:.1f}c eff_min_edge={eff_min_edge_1:.2f}c "
                            f"(base={self.config.min_edge:.1f}c, g3_buf=+{g3_buf:.2f}c) "
                            f"State {state}")
            prev_str = ""
            if prev_d1_detail:
                pa, pb, ph = prev_d1_detail  # series_ask, 100-opp_bid, hedge_tenths
                prev_best = min(pa, pb)
                prev_edge = 100.0 - (get_loaded_cost_cents(prev_best) + ph / 10.0)
                prev_str = f" | prev: ask={pa}c xbook={pb}c hedge={ph/10:.1f}c edge={prev_edge:+.1f}"
            cur_str = f" | now: ask={d1_detail[0]}c xbook={d1_detail[1]}c hedge={d1_detail[2]/10:.1f}c"
            self._log_book_diag(
                f"D1/{d1_trigger}", ticker, opp_series_ticker, top_level_bid,
                top_level_offer, full_top_bids, full_top_offers, full_market_state,
                series_eval,
                f"series_ask={series_ask} series_no_ask={series_no_ask} "
                f"opp_bid={opp_bid} opp_ask={opp_ask} "
                f"best_d1={best_d1_price} best_d2={best_d2_price}",
                cross_orders)
            for order in cross_orders:
                order["_trigger_type"] = d1_trigger
                desired_quotes.append(order)
                log.info(f"LEAD-LAG ARBITRAGE [{d1_trigger.upper()}]: {order['ticker']} {order['kalshi_side'].upper()} (State {state}). "
                         f"{order['size']}x at {order['limit_cents']}c [sweep: {total_sweep}]{prev_str}{cur_str}")

        # Direction 2: Long opponent via cross-book sweep (our NO book + opponent YES book merged)
        # Since only one bot runs per series (alphabetical dedup), we handle both directions here.
        edge_2_approx = 100.0 - synthetic_2
        d2_is_reducing = net_series_pos > 0
        d2_can_trade_ok = bool(can_trade)
        d2_map_ok       = (active_map_ask < 99)
        d2_hedge_ok     = (hedge_profile_2 is not None)
        d2_throttle_ok  = self._check_fire_throttle(edge_2_approx)
        d2_reduce_ok    = (reduce_only_direction in (0, -1))
        d2_gate = (d2_can_trade_ok and d2_map_ok and d2_hedge_ok
                   and d2_improved and (not d2_cluster_fire)
                   and d2_throttle_ok and d2_reduce_ok)
        if edge_2_approx > 0 and not d2_gate:
            log.warning(
                f"[ARBER GATE BLOCK D2] {self.config.ticker} edge=+{edge_2_approx:.1f}c: "
                f"can_trade={'T' if d2_can_trade_ok else 'F'} "
                f"map_ask_ok={'T' if d2_map_ok else f'F(ask={active_map_ask})'} "
                f"hedge={'T' if d2_hedge_ok else 'F(None)'} "
                f"improved={'T' if d2_improved else 'F'} "
                f"cluster_blocks={'F' if not d2_cluster_fire else 'T'} "
                f"throttle={'T' if d2_throttle_ok else 'F'} "
                f"reduce_dir={reduce_only_direction}({'OK' if d2_reduce_ok else 'BLOCKED'}) "
                f"g3_buf={g3_buf:.2f}c"
            )
        if d2_gate:
            books = [(raw_ob_2, ticker, "no")]
            opp_no_pts = opp_raw_ob.get("no_dollars", [])
            if opp_series_ticker and opp_no_pts:
                books.append((opp_no_pts, opp_series_ticker, "yes"))

            eff_min_edge_2 = self.config.min_edge * (self.config.series_edge_mult if d2_trigger == "series" else 1.0) + g3_buf
            # Position skew: shifts gate up for adding / down for reducing per side.
            # With skew_max_shift_cents=0 on the row, skew_c=0.0 and gates are unchanged.
            skew_c_d2 = compute_position_skew_shift_cents(net_series_pos, self.config)
            cross_orders_2 = calculate_cross_book_sweep(
                books=books,
                hedge_cost=hedge_profile_2["synthetic_cost"],
                min_edge=eff_min_edge_2,
                scale_step_c=scale_step,
                base_vol=base_vol,
                max_fire_size=fire_cap,
                max_position=self.config.max_position,
                current_pos=net_series_pos,
                direction_sign=-1,
                min_absolute_edge=self.config.min_absolute_edge,
                fav_edge_k=self.config.fav_edge_k,
                position_skew_cents=skew_c_d2,
                retreat_shift_cents=-retreat_c,
            )

            total_sweep_2 = sum(o["size"] for o in cross_orders_2)
            # Same series-mult diagnostic as D1 (see above).
            if (total_sweep_2 == 0 and d2_trigger == "series"
                    and self.config.series_edge_mult > 1.0):
                base_thresh = self.config.min_edge + g3_buf
                if base_thresh <= edge_2_approx < eff_min_edge_2:
                    log.info(f"[SERIES-MULT BLOCK D2] {self.config.ticker} "
                             f"edge=+{edge_2_approx:.1f}c base_min={base_thresh:.1f}c "
                             f"eff_min={eff_min_edge_2:.1f}c "
                             f"(mult={self.config.series_edge_mult}x) — "
                             f"would have fired without multiplier")
            if total_sweep_2 > 0:
                try:
                    gate_ok2, gate_why2 = self._fire_move_gate(
                        "d2", d2_trigger, _now_ts, h2_cost,
                        net_series_pos, d2_is_reducing, state)
                except Exception as e:
                    gate_ok2, gate_why2 = True, f"gate-error:{e}"
                if not gate_ok2:
                    log.warning(f"[FIRE-GATE BLOCK] {self.config.ticker} D2/{d2_trigger} "
                                f"{gate_why2} hedge={h2_cost:.1f}c "
                                f"{self._fire_move_str('d2', _now_ts, h2_cost, state)} "
                                f"pos={net_series_pos} reducing={d2_is_reducing} "
                                f"state={state} sweep={total_sweep_2}")
                    cross_orders_2 = []
                    total_sweep_2 = 0
                else:
                    self._record_fire(edge_2_approx)
                    self._note_cluster_fire("d2", d2_snap, net_series_pos, cooldown_sec)
                    log.info(f"[FIRE-MOVE] {self.config.ticker} D2/{d2_trigger} "
                             f"hedge={h2_cost:.1f}c "
                             f"{self._fire_move_str('d2', _now_ts, h2_cost, state)} "
                             f"pos={net_series_pos} reducing={d2_is_reducing} "
                             f"state={state} sweep={total_sweep_2} gate={gate_why2}")
            elif g3_buf > 0:
                log.warning(f"[ARBER G3-BUF SUPPRESS D2] {self.config.ticker} "
                            f"edge=+{edge_2_approx:.1f}c eff_min_edge={eff_min_edge_2:.2f}c "
                            f"(base={self.config.min_edge:.1f}c, g3_buf=+{g3_buf:.2f}c) "
                            f"State {state}")
            prev_str2 = ""
            if prev_d2_detail:
                pa, pb, ph = prev_d2_detail  # series_no_ask, opp_ask, hedge_tenths
                prev_best2 = min(pa, pb)
                prev_edge2 = 100.0 - (get_loaded_cost_cents(prev_best2) + ph / 10.0)
                prev_str2 = f" | prev: ask={pa}c xbook={pb}c hedge={ph/10:.1f}c edge={prev_edge2:+.1f}"
            cur_str2 = f" | now: ask={d2_detail[0]}c xbook={d2_detail[1]}c hedge={d2_detail[2]/10:.1f}c"
            self._log_book_diag(
                f"D2/{d2_trigger}", ticker, opp_series_ticker, top_level_bid,
                top_level_offer, full_top_bids, full_top_offers, full_market_state,
                series_eval,
                f"series_ask={series_ask} series_no_ask={series_no_ask} "
                f"opp_bid={opp_bid} opp_ask={opp_ask} "
                f"best_d1={best_d1_price} best_d2={best_d2_price}",
                cross_orders_2)
            for order in cross_orders_2:
                order["_trigger_type"] = d2_trigger
                desired_quotes.append(order)
                log.info(f"LEAD-LAG ARBITRAGE [{d2_trigger.upper()}]: {order['ticker']} {order['kalshi_side'].upper()} (State {state}). "
                         f"{order['size']}x at {order['limit_cents']}c [sweep: {total_sweep_2}]{prev_str2}{cur_str2}")

        # GAME MOMENTUM / VWAP sweep removed 2026-07-19. It was dead code:
        # `self.momentum_enabled` was hardcoded False and assigned nowhere else,
        # so the branch never ran (0 `VWAP MOMENTUM TRIGGERED` in run.log since
        # 2026-06-09). It priced the hedge off the map MID rather than the
        # executable bid/ask, bypassed series_edge_mult, and carried the last two
        # `is_bo5` edge widenings (BO5 runs on its own bot, so widening in shared
        # code double-counts — the same 1.5x was already reverted in momentum_bot).
        # Standalone momentum lives in momentum_bot.py / the *_MOM_L1 rows.

        opp_label = opp_series_ticker.split("-")[-1] if opp_series_ticker else "?"
        best_d1_cost = get_loaded_cost_cents(best_d1_price) + h1_cost
        best_d2_cost = get_loaded_cost_cents(best_d2_price) + h2_cost
        edge_1 = 100.0 - best_d1_cost if best_d1_cost < 199.0 else None
        edge_2 = 100.0 - best_d2_cost if best_d2_cost < 199.0 else None
        d1_mark = "" if d1_changed else "="
        d2_mark = "" if d2_changed else "="
        e1_str = f"Long {team_suffix}{d1_mark}: Best={best_d1_price:5.0f} + Hedge={h1_cost:5.1f} = {best_d1_cost:5.1f} Edge={edge_1:>+5.1f}" if edge_1 is not None else ""
        e2_str = f"Long {opp_label}{d2_mark}: Best={best_d2_price:5.0f} + Hedge={h2_cost:5.1f} = {best_d2_cost:5.1f} Edge={edge_2:>+5.1f}" if edge_2 is not None else ""
        # Effective threshold accounts for the G3 buffer (BO3 state 1/2 only).
        # threshold = 100 - base_min_edge; subtract g3_buf to show the actual
        # bar the arber must clear when buffer is active.
        eff_thresh = threshold - g3_buf
        g3_str = f" | G3Buf:+{g3_buf:.2f}c" if g3_buf > 0 else ""
        # Position skew always shown so operators can confirm gate state per cycle.
        # 0.00c = baseline (no skew); positive = adding harder / reducing easier.
        skew_c_ctx = compute_position_skew_shift_cents(net_series_pos, self.config)
        skew_str = f" | Skew:{skew_c_ctx:.2f}c"
        # Recency-flow retreat readout (TRADE_RETREAT_SCOPE.md) — logging
        # only on this (taker) path; the armed shift lives in
        # PositionAdjuster.adjust_theos (makers only). f>0 = one-sided maker
        # accumulation toward team_suffix over the trailing window; s = theo
        # shift vs adds under this row's blueprint retreat params (disarmed
        # rows display the default 10c schedule so the tape stays live).
        try:
            import retreat_shadow
            # Render with the QUOTER row's params, not this arber row's. The
            # arber row is disarmed (retreat_cap_cents=0), so passing it made
            # status_str fall back to the module-default 10c-at-18% schedule
            # while the quoter was actually applying its own 4-5c-at-7.2% one —
            # the line printed s0.0c/s1.2c during a TYLOOAG burst where the
            # applied shift was 0.8-4.0c (capped). Understating by 3-4x exactly
            # when the shift matters is worse than printing nothing.
            _rconf = _quoter_retreat_conf(ticker) or self.config
            retreat_str = " | " + retreat_shadow.status_str(
                "-".join(ticker.split("-")[:2]), team_suffix,
                getattr(_rconf, "max_position", self.config.max_position),
                conf=_rconf)
        except Exception:
            retreat_str = ""
        arb_context = f"[LEAD-LAG] {ticker} (S{state}) | Pos:{net_series_pos:>+5d}/{self.config.max_position} | {e1_str} | {e2_str} | Thresh:{eff_thresh:.1f}{g3_str}{skew_str}{retreat_str}"

        # Attach context to each order so the trade logger can display it on fill
        for q in desired_quotes:
            q["_arb_context"] = arb_context

        log.info(arb_context)

        return desired_quotes
