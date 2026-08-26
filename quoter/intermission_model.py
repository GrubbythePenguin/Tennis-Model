"""
Intermission Theo Generator

Computes bid_theo and offer_theo for the inter-game break in a BO3 series.
The theo = the series price at which the arber would start firing (threshold - hedge_cost).

Used by the quoter to passively bid during the 10-minute intermission
instead of paying the spread with IOC orders.
"""
import logging
import os
import csv
import math
from typing import Dict, List, Any
from base_model import BaseTheoGenerator

log = logging.getLogger(__name__)


def _fee_loaded(price_cents: float) -> float:
    p = price_cents / 100.0
    return price_cents + 7.0 * p * (1.0 - p)


def _bo3_series_prob(p1: float, p2: float) -> float:
    """BO3 series prob for team A given per-game probs p1, p2."""
    p3 = (p1 + p2) / 2.0
    return p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3


def _state1_hedge_cost(p1: float, p2: float, map_ask_cents: float) -> float:
    """Hedge cost after team A won game 1 (state 1).
    map_ask_cents = cost to buy 'B wins game 2'."""
    p3 = (p1 + p2) / 2.0
    p3_B_if_B_won_G2 = 1.0 - p3
    cost_g3 = _fee_loaded(p3_B_if_B_won_G2 * 100.0)
    shares_b = cost_g3 / 100.0
    return shares_b * _fee_loaded(map_ask_cents)


def _state2_hedge_cost(p1: float, p2: float, map_ask_cents: float) -> float:
    """Hedge cost after team A lost game 1 (state 2).
    map_ask_cents = cost to buy 'A wins game 2'."""
    p3 = (p1 + p2) / 2.0
    p3_B_if_A_won_G2 = 1.0 - p3
    cost_g3 = _fee_loaded(p3_B_if_A_won_G2 * 100.0)
    shares_b = (100.0 - cost_g3) / 100.0
    return cost_g3 + shares_b * _fee_loaded(map_ask_cents)


class EsportsIntermissionTheoGenerator(BaseTheoGenerator):
    """
    Generates theos for series tickers during the inter-game intermission.

    For each ticker, computes the maximum profitable series price using
    the arber's hedge math. This becomes bid_theo (for buying) and
    offer_theo = 100 - opponent's bid_theo (for the other side).

    Reads probabilities from esports_probabilities.csv.
    """

    def __init__(self, client, configs=None):
        super().__init__(client, configs)
        self._prob_cache = {}  # ticker -> {p1, p2, series, state}
        self._load_probabilities()

    def _load_probabilities(self):
        """Load pre-game probabilities from CSV."""
        prob_file = "esports_probabilities.csv"
        if not os.path.exists(prob_file):
            return
        try:
            with open(prob_file, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header row — matches hedge_engine.load_probabilities
                for row in reader:
                    if len(row) >= 4:
                        try:
                            ticker = row[0].strip()
                            self._prob_cache[ticker] = {
                                "p1": float(row[1]),
                                "p2": float(row[2]),
                                "series": float(row[3]),
                            }
                        except (ValueError, IndexError):
                            # Skip bad rows (column-shift, garbage data) — don't
                            # abort the entire load. Without per-row try/except a
                            # single corrupt row leaves the entire cache empty,
                            # which silently disables intermission predictions
                            # for every ticker. Observed 2026-06-15.
                            continue
        except Exception as e:
            log.error(f"[INTERMISSION] Failed loading probabilities: {e}")

    def _batch_generate(self, tickers: List[str], dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        results = {}

        for ticker in tickers:
            conf = self.configs.get(ticker)
            if not conf:
                continue

            probs = self._prob_cache.get(ticker)
            if not probs:
                continue

            parts = ticker.split("-")
            # 2026-07-28: test/rewrite the PREFIX only — a TEAM NAME can contain
            # the token ("2GAME Esports"), which made both the guard and
            # series_base.replace() wrong (...SR2MAP, 404). See
            # memory/project_map_base_game_substring_collision.md
            _mpfx = parts[0] if parts else ""
            if len(parts) < 3 or ("GAME" not in _mpfx and "MATCH" not in _mpfx):
                continue

            series_base = parts[0] + "-" + parts[1]
            team_suffix = parts[2]

            if "MATCH" in _mpfx:
                map_base = _mpfx.replace("MATCH", "SETWINNER") + "-" + parts[1]
            else:
                map_base = _mpfx.replace("GAME", "MAP") + "-" + parts[1]

            p1 = probs["p1"]
            p2 = probs["p2"]

            # Detect state from map 1
            map1_ticker = f"{map_base}-1-{team_suffix}"
            m1_data = dt_market_state.get(map1_ticker, {})
            m1_status = m1_data.get("status", "")
            m1_result = m1_data.get("result", "")

            m1_won = False
            m1_lost = False
            if m1_status in ["determined", "settled", "closed", "finalized"]:
                if m1_result == "yes":
                    m1_won = True
                elif m1_result == "no":
                    m1_lost = True

            if not m1_won and not m1_lost:
                # Game 1 not settled — no intermission theo
                continue

            state = 1 if m1_won else 2

            # Get map 2 mid for hedge cost estimation
            map2_ticker = f"{map_base}-2-{team_suffix}"
            m2_state = dt_market_state.get(map2_ticker, {})

            # Use map 2 top-of-book if available, else model probability
            # The map 2 price during intermission should be close to pre-game
            m2_bid = 0
            m2_ask = 100
            if "raw_ob" in m2_state:
                ob = m2_state["raw_ob"].get("orderbook_fp", {})
                y = ob.get("yes_dollars", [])
                n = ob.get("no_dollars", [])
                if y:
                    y.sort(key=lambda x: float(x[0]), reverse=True)
                    m2_bid = int(float(y[0][0]) * 100)
                if n:
                    n.sort(key=lambda x: float(x[0]), reverse=True)
                    m2_ask = 100 - int(float(n[0][0]) * 100)

            # If no orderbook, use model probability
            if m2_bid == 0 and m2_ask == 100:
                m2_bid = int(p2 * 100) - 1
                m2_ask = int(p2 * 100) + 1

            # Compute hedge costs
            # map_no_ask = cost to buy "B wins game 2" = 100 - m2_bid
            # map_ask = cost to buy "A wins game 2" = m2_ask
            map_no_ask = 100 - m2_bid
            map_ask = m2_ask

            min_edge = max(conf.min_edge, conf.min_absolute_edge, 0.0)
            threshold = 100.0 - min_edge

            if state == 1:
                # Won game 1 — Long A needs hedge against losing game 2
                hedge_long_a = _state1_hedge_cost(p1, p2, map_no_ask)
                # Long B (opponent) needs state 2 hedge
                hedge_long_b = _state2_hedge_cost(1.0 - p1, 1.0 - p2, map_ask)
            else:
                # Lost game 1 — Long A needs state 2 hedge
                hedge_long_a = _state2_hedge_cost(p1, p2, map_no_ask)
                # Long B needs state 1 hedge
                hedge_long_b = _state1_hedge_cost(1.0 - p1, 1.0 - p2, map_ask)

            # Max profitable series ask = threshold - hedge_cost
            # This is the bid_theo: the most we'd pay for the series
            bid_theo_a = threshold - hedge_long_a
            offer_theo_a = 100.0 - (threshold - hedge_long_b)

            # Clamp to reasonable range
            bid_theo_a = max(1.0, min(99.0, bid_theo_a))
            offer_theo_a = max(1.0, min(99.0, offer_theo_a))

            results[ticker] = {
                "bid_theo": bid_theo_a,
                "offer_theo": offer_theo_a,
                "raw_bid_theo": bid_theo_a,
                "raw_offer_theo": offer_theo_a,
            }

            log.info(f"[INTERMISSION THEO] {ticker} state={state} "
                     f"bid_theo={bid_theo_a:.1f}c offer_theo={offer_theo_a:.1f}c "
                     f"hedge_A={hedge_long_a:.1f} hedge_B={hedge_long_b:.1f}")

        return results
