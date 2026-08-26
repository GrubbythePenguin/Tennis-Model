"""Shared scaffolding for arber-style bots (series arber and map arber).

The two strategies share substantially identical machinery (probability
loading, fire throttling, cross-book sweep, position cap, fake-move
detection, snapshot caching) and differ only in:
  - which leg is the trade leg vs hedge leg
  - which math functions compute the hedge profile
  - whether tennis / BO5 special-paths apply

This module centralizes the shared parts. EsportsArberBot (series) and
EsportsMapArberBot (map) both subclass ArberBotBase and override only
`evaluate()` with their leg-specific logic.
"""
import logging
import time
from typing import Any, Dict, List

import position_store
from framework_config import logit_scaled_edge
from hedge_engine import get_loaded_cost_cents
from models import QuoteSide

log = logging.getLogger(__name__)


# ── Shared sweep primitives ─────────────────────────────────────────────────
# These are leg-agnostic: the caller supplies hedge_cost (cents) and the
# orderbook to sweep; the routine returns IOC orders sized by edge tiers
# and constrained by position cap + hedge zone.

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
    fav_edge_k: float = 0.0,
) -> tuple[int, int]:
    """Sweep a single side of an orderbook. Returns (total_volume, worst_price_taken).

    Edge gates use variance-scaled min_edge per price level (narrowest at the extremes);
    min_absolute_edge floors the requirement.

    Asymmetric expanding/reducing rules:
      - Reducing fires (position would shrink toward 0): require eff_min_edge.
      - Expanding fires: scale by edge tiers, capped at max_position; allow
        breach to 1.5x max_position only when edge >= 2.5x eff_min_edge.
    """
    accumulated_vol = 0
    worst_price_taken = 0

    sorted_pts = sorted(
        (pt for pt in ob_pts if len(pt) >= 2),
        key=lambda x: float(x[0]), reverse=True,
    )

    for pt in sorted_pts:
        no_bid_p = float(pt[0])
        size_avail = int(float(pt[1]))
        if size_avail <= 0:
            continue

        series_ask_p = int(round(100.0 - (no_bid_p * 100.0)))
        series_loaded = get_loaded_cost_cents(float(series_ask_p))
        cost_synth = series_loaded + hedge_cost
        edge_c = 100.0 - cost_synth
        eff_min_edge = logit_scaled_edge(series_ask_p, min_edge, min_absolute_edge, fav_edge_k=fav_edge_k)

        taken_here = 0

        while size_avail > 0:
            projected_pos = current_pos + (direction_sign * accumulated_vol)
            is_reducing = (
                (direction_sign > 0 and projected_pos < 0) or
                (direction_sign < 0 and projected_pos > 0)
            )

            if is_reducing:
                if edge_c < (eff_min_edge - 0.05):
                    break
                chunk_cap = min(size_avail, abs(projected_pos))
                if accumulated_vol >= max_fire_size:
                    break
                take = min(chunk_cap, max_fire_size - accumulated_vol)
                if take <= 0:
                    break
                accumulated_vol += take
                size_avail -= take
                taken_here += take
            else:
                if edge_c < (eff_min_edge - 0.05):
                    break
                scaling_tiers = max(0, int((edge_c - eff_min_edge) / max(1.0, scale_step_c)))
                authorized = base_vol + (scaling_tiers * base_vol)
                max_allowed = min(max_fire_size, authorized)
                if accumulated_vol >= max_allowed:
                    break
                global_room = max_position - abs(projected_pos)
                if global_room <= 0:
                    hedge_zone_room = int(1.5 * max_position) - abs(projected_pos)
                    if hedge_zone_room > 0 and edge_c >= (eff_min_edge * 2.5):
                        global_room = hedge_zone_room
                    else:
                        break
                take = min(size_avail, max_allowed - accumulated_vol, global_room)
                if take <= 0:
                    break
                accumulated_vol += take
                size_avail -= take
                taken_here += take

        if taken_here > 0:
            worst_price_taken = max(worst_price_taken, series_ask_p)
        else:
            break

    return accumulated_vol, worst_price_taken


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
    fav_edge_k: float = 0.0,
) -> list:
    """Merge economically-equivalent books and sweep cheapest-first.

    books: [(ob_pts, ticker, kalshi_side), ...] where all books represent
           the same economic position (e.g., all ways to go long team A:
           our YES book + opponent NO book).

    Returns IOC order dicts, one per (ticker, side) that received fills.
    """
    tagged_pts = []
    for ob_pts, src_ticker, src_side in books:
        for pt in ob_pts:
            if len(pt) >= 2:
                p = float(pt[0])
                s = int(float(pt[1]))
                if s > 0:
                    tagged_pts.append((p, s, src_ticker, src_side))

    tagged_pts.sort(key=lambda x: x[0], reverse=True)

    results = {}
    accumulated_vol = 0

    for no_bid_p, size_avail, src_ticker, src_side in tagged_pts:
        series_ask_p = int(round(100.0 - (no_bid_p * 100.0)))
        series_loaded = get_loaded_cost_cents(float(series_ask_p))
        cost_synth = series_loaded + hedge_cost
        edge_c = 100.0 - cost_synth
        eff_min_edge = logit_scaled_edge(series_ask_p, min_edge, min_absolute_edge, fav_edge_k=fav_edge_k)

        key = (src_ticker, src_side)
        taken_here = 0

        while size_avail > 0:
            projected_pos = current_pos + (direction_sign * accumulated_vol)
            is_reducing = (
                (direction_sign > 0 and projected_pos < 0) or
                (direction_sign < 0 and projected_pos > 0)
            )

            if is_reducing:
                if edge_c < (eff_min_edge - 0.05):
                    break
                chunk_cap = min(size_avail, abs(projected_pos))
                if accumulated_vol >= max_fire_size:
                    break
                take = min(chunk_cap, max_fire_size - accumulated_vol)
                if take <= 0:
                    break
                accumulated_vol += take
                size_avail -= take
                taken_here += take
            else:
                if edge_c < (eff_min_edge - 0.05):
                    break
                scaling_tiers = max(0, int((edge_c - eff_min_edge) / max(1.0, scale_step_c)))
                authorized = base_vol + (scaling_tiers * base_vol)
                max_allowed = min(max_fire_size, authorized)
                if accumulated_vol >= max_allowed:
                    break
                global_room = max_position - abs(projected_pos)
                if global_room <= 0:
                    hedge_zone_room = int(1.5 * max_position) - abs(projected_pos)
                    if hedge_zone_room > 0 and edge_c >= (eff_min_edge * 2.5):
                        global_room = hedge_zone_room
                    else:
                        break
                take = min(size_avail, max_allowed - accumulated_vol, global_room)
                if take <= 0:
                    break
                accumulated_vol += take
                size_avail -= take
                taken_here += take

        if taken_here > 0:
            if key not in results:
                results[key] = {"vol": 0, "worst_price": 0, "worst_edge": None}
            results[key]["vol"] += taken_here
            results[key]["worst_price"] = max(results[key]["worst_price"], series_ask_p)
            results[key]["worst_edge"] = edge_c
        else:
            break

    orders = []
    for (ord_ticker, ord_side), r in results.items():
        if r["vol"] > 0:
            if ord_side == "no":
                arb_theo_yes = hedge_cost
            else:
                arb_theo_yes = 100.0 - hedge_cost
            orders.append({
                "side": QuoteSide.BID,
                "ticker": ord_ticker,
                "kalshi_side": ord_side,
                "size": r["vol"],
                "limit_cents": r["worst_price"],
                "time_in_force": "immediate_or_cancel",
                "_raw_theo": arb_theo_yes,
                "_adj_theo": arb_theo_yes,
            })
    return orders


# ── ArberBotBase ────────────────────────────────────────────────────────────

class ArberBotBase:
    """Shared scaffolding for series-side and map-side arber bots.

    Subclasses implement `evaluate()` and use these inherited helpers:
      - _check_fire_throttle / _record_fire (edge-bucketed throttle)
      - refresh_probabilities / _load_pre_game_probabilities (CSV priors)
      - _check_fake_move (rogue-order detection on watched book)
      - _apply_position_cap_filter (1.5x cap with reducing-only override)

    Subclasses also have access to module-level calculate_sweep_payload and
    calculate_cross_book_sweep.
    """

    # Fake-move detection thresholds (cents). When the watched book's
    # previous spread exceeded WIDE_THRESHOLD and only one side moves by
    # >= MOVE_THRESHOLD this cycle, we treat it as a rogue order narrowing
    # the spread rather than a real market move.
    FAKE_MOVE_WIDE_THRESHOLD = 15
    FAKE_MOVE_MOVE_THRESHOLD = 3

    def __init__(self, config):
        self.config = config
        self.active = True
        self.g2_momentum = 0.05
        self.g3_momentum = 0.05
        self.p1_start = 0.5
        self.p2_start = 0.5
        self.p3_start = 0.5
        self.series_start = 0.5
        self.last_log_time = 0.0
        self.momentum_enabled = False
        self._last_prob_retry = 0.0

        # Watched-book prev prices for fake-move detection. Subclasses set
        # which book they watch (series watches active map; map watches series).
        self._prev_watched_bid = None
        self._prev_watched_ask = None

        # Edge-bucketed fire throttle: max 5 fires per bucket per 60s,
        # 7c+ edge bucket (<=93) is unlimited.
        self._fire_timestamps: Dict[int, List[float]] = {}

        # Snapshot cache: (price, hedge_tenths) per direction; only fire when
        # one of these "improved" (price down or hedge cheaper).
        self._last_d1 = None
        self._last_d2 = None
        self._last_state = None

        self._load_pre_game_probabilities()

    # ── Fire throttle ────────────────────────────────────────────────────────

    def _check_fire_throttle(self, edge_c: float) -> bool:
        bucket = int(100 - edge_c)
        if bucket <= 93:
            return True

        now = time.time()
        if bucket in self._fire_timestamps:
            self._fire_timestamps[bucket] = [
                t for t in self._fire_timestamps[bucket] if now - t < 60.0
            ]
        else:
            self._fire_timestamps[bucket] = []

        if len(self._fire_timestamps[bucket]) >= 5:
            return False
        return True

    def _record_fire(self, edge_c: float):
        bucket = int(100 - edge_c)
        if bucket > 93:
            if bucket not in self._fire_timestamps:
                self._fire_timestamps[bucket] = []
            self._fire_timestamps[bucket].append(time.time())

    # ── Probability loading ─────────────────────────────────────────────────

    def refresh_probabilities(self, new_probs: dict = None) -> bool:
        """Hot-reload priors from esports_probabilities.csv.

        Safe on transient miss: leaves existing state intact if ticker absent.
        """
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
        self._verified = probs["verified"]
        if not self.baseline_loaded:
            self.baseline_loaded = True
            self.active = True
            if "GAME" in self.config.ticker:
                log.info(f"[{self.__class__.__name__} MATH ENGINE] {self.config.ticker} | "
                         f"RELOAD activated: p1={probs['p1']*100:.1f}% "
                         f"p2={probs['p2']*100:.1f}% series={probs['series']*100:.1f}%")
        else:
            if "GAME" in self.config.ticker:
                log.info(f"[{self.__class__.__name__} MATH ENGINE] {self.config.ticker} | "
                         f"RELOAD: p1={probs['p1']*100:.1f}% "
                         f"p2={probs['p2']*100:.1f}% series={probs['series']*100:.1f}%")
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
            self._verified = probs["verified"]
            self.baseline_loaded = True

        if not self.baseline_loaded:
            if "GAME" in self.config.ticker:
                log.error(f"--- FATAL RISK HALT --- | No pre-game probabilities for "
                          f"{self.config.ticker}. {self.__class__.__name__} deactivating.")
            self.active = False
        else:
            if "GAME" in self.config.ticker:
                log.info(f"[{self.__class__.__name__} MATH ENGINE] {self.config.ticker} | "
                         f"Loaded CSV Map Baseline Bounds: Map 1 ({self.p1_start*100:.1f}%), "
                         f"Map 2 ({self.p2_start*100:.1f}%), "
                         f"Series Anchor ({self.series_start*100:.1f}%)")

    # ── Fake-move detection ────────────────────────────────────────────────

    def _is_fake_move(self, cur_bid: float, cur_ask: float) -> bool:
        """Return True if the watched book's move looks rogue (not real).

        Detection rule: previous spread > FAKE_MOVE_WIDE_THRESHOLD AND only
        one side moved by >= FAKE_MOVE_MOVE_THRESHOLD. Real moves shift both
        sides together. Updates _prev_watched_bid/_prev_watched_ask after
        the check so callers don't have to manage state.
        """
        prev_bid = self._prev_watched_bid
        prev_ask = self._prev_watched_ask
        is_fake = False
        if prev_bid is not None and prev_ask is not None:
            prev_spread = prev_ask - prev_bid
            if prev_spread > self.FAKE_MOVE_WIDE_THRESHOLD:
                bid_moved = abs(cur_bid - prev_bid) >= self.FAKE_MOVE_MOVE_THRESHOLD
                ask_moved = abs(cur_ask - prev_ask) >= self.FAKE_MOVE_MOVE_THRESHOLD
                if bid_moved != ask_moved:
                    is_fake = True
        self._prev_watched_bid = cur_bid
        self._prev_watched_ask = cur_ask
        return is_fake

    # ── Position cap filter ────────────────────────────────────────────────

    def _build_position_cap_filter(self, primary_ticker: str):
        """Returns (filter_fn, current_pos, hard_cap, at_hard_cap).

        At cap, drop quotes that would push position further from 0.
        The filter is built lazily so subclasses can early-return without
        paying for it.
        """
        try:
            cur_pos = position_store.get_position(primary_ticker)
            hard_cap = int(1.5 * self.config.max_position)
            at_hard_cap = (cur_pos is not None and abs(cur_pos) >= hard_cap)
        except Exception:
            cur_pos, hard_cap, at_hard_cap = None, None, False

        def _filter(quotes):
            if not at_hard_cap or cur_pos is None or not quotes:
                return quotes
            sign_to_drop = +1 if cur_pos > 0 else -1
            kept = []
            for q in quotes:
                q_ticker = q.get("ticker", "")
                ks = (q.get("kalshi_side") or "").lower()
                # Net position effect:
                #   our ticker + yes  → +1, our + no → -1
                #   opp ticker + yes  → -1, opp + no → +1
                if q_ticker == primary_ticker:
                    sign = +1 if ks == "yes" else -1
                else:
                    sign = +1 if ks == "no" else -1
                if sign != sign_to_drop:
                    kept.append(q)
            dropped = len(quotes) - len(kept)
            if dropped > 0:
                log.warning(f"[STOP-CAP-REDUCE] {primary_ticker}: pos={cur_pos:+d} ≥ "
                            f"1.5×max_position={hard_cap} — dropped {dropped} adding-direction "
                            f"orders, keeping {len(kept)} reducing")
            return kept

        return _filter, cur_pos, hard_cap, at_hard_cap

    # ── Snapshot cache helpers ─────────────────────────────────────────────

    def _check_d_improved(self, dn_snap: tuple, last_attr: str, last_state):
        """Compare current direction snapshot vs the cached previous one.

        Returns (improved: bool, trigger: str). trigger ∈ {"", "init", "price",
        "hedge"} — "price" / "hedge" indicate which leg moved.

        Caller is responsible for setting self.<last_attr> = dn_snap after
        this returns (so the trigger comparison reflects this cycle's input).
        Resets cache when state transitions.
        """
        if last_state != self._last_state:
            # State boundary clears caches
            self._last_d1 = None
            self._last_d2 = None
            self._last_state = last_state

        last = getattr(self, last_attr)
        if last is None:
            return True, "init"
        if last == dn_snap:
            return False, ""
        old_price, old_hedge = last
        new_price, new_hedge = dn_snap
        price_delta = old_price - new_price
        hedge_delta = old_hedge - new_hedge
        if price_delta > 0 or hedge_delta > 0:
            if price_delta > 0 and hedge_delta <= 0:
                return True, "price"
            if hedge_delta > 0 and price_delta <= 0:
                return True, "hedge"
            return True, ("price" if price_delta >= hedge_delta else "hedge")
        return False, ""
