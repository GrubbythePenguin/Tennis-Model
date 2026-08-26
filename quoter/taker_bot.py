"""
Series Taker Bot

Consumes bid/offer theos from the live_series model and fires IOC sweeps
when the market price diverges beyond min_edge from the theo.

Uses the same cross-book sweep logic as the arber, but all hedge math
lives in the live_series model — this bot just compares theo vs market
and sweeps when there's edge.

Designed to coexist with the quoter (maker) on the same ticker:
- Quoter posts resting orders at 1-5c from theo
- Taker fires IOC when edge > min_edge (typically 5-8c)
Both share position via position_store.
"""
import logging
import time
from typing import Any, Dict, List

import position_store
from arber_bot import calculate_cross_book_sweep
from hedge_engine import get_loaded_cost_cents
from framework_config import QuoterConfig
from models import QuoteSide

log = logging.getLogger(__name__)

FIRE_COOLDOWN_SECS = 1.0  # Min seconds between fires per ticker


class SeriesTakerBot:
    """
    Takes liquidity (IOC sweeps) when the live_series theo shows
    sufficient edge vs the market price.
    """

    def __init__(self, config: QuoterConfig):
        self.config = config
        self.active = True
        self.baseline_loaded = True  # Theos come from model, not CSV

        self._last_fire_ts = 0.0
        self._fire_count = 0

    def _check_fire_throttle(self, edge: float) -> bool:
        """Rate-limit fires to prevent rapid-fire on the same tick."""
        now = time.time()
        if now - self._last_fire_ts < FIRE_COOLDOWN_SECS:
            return False
        return True

    def _record_fire(self, edge: float):
        self._last_fire_ts = time.time()
        self._fire_count += 1

    def evaluate(self, market_state: Any, top_level_bid: float, top_level_offer: float,
                 bid_theo: float, offer_theo: float,
                 full_market_state: Dict[str, Any] = None,
                 full_top_bids: Dict[str, float] = None,
                 full_top_offers: Dict[str, float] = None,
                 **kwargs) -> List[Dict[str, Any]]:

        if not self.active or bid_theo is None or offer_theo is None:
            return []

        ticker = self.config.ticker
        parts = ticker.split("-")
        if len(parts) < 3:
            return []

        series_base = parts[0] + "-" + parts[1]
        team_suffix = parts[2]

        # Find opponent series ticker
        opp_series_ticker = None
        for k in (full_top_bids or {}):
            if k.startswith(series_base + "-") and k != ticker:
                opp_series_ticker = k
                break

        # Only alphabetically-first team runs to prevent duplicate firing
        opp_suffix = opp_series_ticker.split("-")[-1] if opp_series_ticker else ""
        if opp_suffix and team_suffix > opp_suffix:
            return []

        # Current position (netted across both tickers for esports)
        current_pos = position_store.get_position(ticker)

        # Series orderbook
        raw_ob = {}
        if full_market_state and ticker in full_market_state:
            raw_ob = full_market_state[ticker].get("raw_ob", {}).get("orderbook_fp", {})

        opp_raw_ob = {}
        if opp_series_ticker and full_market_state:
            opp_raw_ob = full_market_state.get(opp_series_ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})

        min_edge = max(self.config.min_edge, self.config.min_absolute_edge, 0.0)
        scale_step = self.config.arb_scale_step_cents
        base_vol = self.config.volumes[0] if self.config.volumes else 100
        fire_cap = self.config.max_fire_size

        desired_quotes = []

        # ── Direction 1: Long self (buy series YES) ──
        # hedge_cost for the sweep function = 100 - bid_theo
        # because bid_theo IS the fair value, and the sweep checks:
        #   edge = 100 - (series_loaded + hedge_cost) = 100 - series_loaded - (100 - bid_theo)
        #        = bid_theo - series_loaded
        # Which is exactly "theo minus cost to buy" = real edge
        hedge_cost_1 = 100.0 - bid_theo
        edge_1 = bid_theo - get_loaded_cost_cents(top_level_offer)  # theo - (ask + fee)

        if self._check_fire_throttle(edge_1) and edge_1 >= min_edge:
            raw_ob_yes = raw_ob.get("no_dollars", [])  # NO bids = YES asks
            books = [(raw_ob_yes, ticker, "yes")]

            if opp_series_ticker and opp_raw_ob:
                opp_yes_pts = opp_raw_ob.get("yes_dollars", [])
                if opp_yes_pts:
                    books.append((opp_yes_pts, opp_series_ticker, "no"))

            cross_orders = calculate_cross_book_sweep(
                books=books,
                hedge_cost=hedge_cost_1,
                min_edge=min_edge,
                scale_step_c=scale_step,
                base_vol=base_vol,
                max_fire_size=fire_cap,
                max_position=self.config.max_position,
                current_pos=current_pos,
                direction_sign=1,
            )

            total_sweep = sum(o["size"] for o in cross_orders)
            if total_sweep > 0:
                self._record_fire(edge_1)
                log.info(f"[TAKER] LONG {ticker} | theo={bid_theo:.1f} ask={top_level_offer:.0f} "
                         f"edge={edge_1:.1f}c | sweep={total_sweep} lots")
            for order in cross_orders:
                desired_quotes.append(order)

        # ── Direction 2: Long opponent (buy series NO) ──
        hedge_cost_2 = offer_theo  # 100 - (100 - offer_theo) = offer_theo
        series_no_ask = 100.0 - top_level_bid
        edge_2 = (100.0 - offer_theo) - get_loaded_cost_cents(series_no_ask)  # theo - (no_ask + fee)

        if self._check_fire_throttle(edge_2) and edge_2 >= min_edge:
            raw_ob_no = raw_ob.get("yes_dollars", [])  # YES bids = NO asks
            books = [(raw_ob_no, ticker, "no")]

            if opp_series_ticker and opp_raw_ob:
                opp_no_pts = opp_raw_ob.get("no_dollars", [])
                if opp_no_pts:
                    books.append((opp_no_pts, opp_series_ticker, "yes"))

            cross_orders_2 = calculate_cross_book_sweep(
                books=books,
                hedge_cost=hedge_cost_2,
                min_edge=min_edge,
                scale_step_c=scale_step,
                base_vol=base_vol,
                max_fire_size=fire_cap,
                max_position=self.config.max_position,
                current_pos=current_pos,
                direction_sign=-1,
            )

            total_sweep = sum(o["size"] for o in cross_orders_2)
            if total_sweep > 0:
                self._record_fire(edge_2)
                log.info(f"[TAKER] SHORT {ticker} | theo={offer_theo:.1f} bid={top_level_bid:.0f} "
                         f"edge={edge_2:.1f}c | sweep={total_sweep} lots")
            for order in cross_orders_2:
                desired_quotes.append(order)

        # Log state every ~30 seconds for visibility
        if not desired_quotes and int(time.time()) % 30 == 0:
            log.debug(f"[TAKER] {ticker} | bid_theo={bid_theo:.1f} offer_theo={offer_theo:.1f} "
                      f"| market={top_level_bid}/{top_level_offer} "
                      f"| edge_long={edge_1:.1f}c edge_short={edge_2:.1f}c | pos={current_pos}")

        return desired_quotes
