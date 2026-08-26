import logging
import math
import time
from typing import Any, List, Dict

from framework_config import QuoterConfig
from models import QuoteSide
from hedge_engine import get_loaded_cost_cents
import position_store

log = logging.getLogger(__name__)

# Aggregate "Fight ends in X" markets live under a different Kalshi prefix.
# KXUFCMOV = individual fighter method markets, KXUFCMOF = aggregate fight method markets.
AGGREGATE_PREFIX = "KXUFCMOF"
MOV_PREFIX = "KXUFCMOV"


class MMAArberBot:
    """
    MMA Lead-Lag Arbitrage Bot.

    Monitors all method-of-victory (MOV) sub-markets for a UFC fight event.
    Two strategies:

    1) 7-Way Fade: When one outcome's YES bid >= leader_threshold (e.g. 90c),
       buy NO on the other 6 lagging markets via IOC.

    2) 3-Way Aggregate Arb: When aggregate "Fight ends in KO" markets exist
       (under KXUFCMOF prefix), detect lead-lag between aggregate and its
       two fighter-specific components and buy the lagging side via IOC.
    """

    def __init__(self, config: QuoterConfig):
        self.config = config
        self.active = True
        self.sibling_tickers: List[str] = []
        self.aggregate_tickers: Dict[str, str] = {}  # method_suffix -> aggregate ticker
        self.leader_threshold: float = max(config.min_absolute_edge, 90.0)
        self.last_log_time: float = 0.0

    def get_sibling_tickers(self) -> List[str]:
        """Returns all discovered child tickers for run.py to poll orderbooks."""
        return list(self.sibling_tickers) + list(self.aggregate_tickers.values())

    def _discover_siblings(self, full_market_state: Dict[str, Any]) -> None:
        """
        Dynamically discover all child tickers belonging to this fight event
        by scanning full_market_state keys that share our event ticker prefix.
        Also discover any aggregate KXUFCMOF tickers.
        """
        event_ticker = self.config.ticker  # e.g. KXUFCMOV-26APR11PROULB

        # Extract the date+fighters tag (e.g. 26APR11PROULB)
        parts = event_ticker.split("-", 1)
        if len(parts) < 2:
            return
        fight_tag = parts[1]  # e.g. 26APR11PROULB

        siblings = []
        aggregates = {}

        for ticker in full_market_state.keys():
            # Match individual MOV markets: KXUFCMOV-{fight_tag}-{SUFFIX}
            if ticker.startswith(f"{MOV_PREFIX}-{fight_tag}-"):
                siblings.append(ticker)
            # Match aggregate markets: KXUFCMOF-{fight_tag}-{SUFFIX}
            elif ticker.startswith(f"{AGGREGATE_PREFIX}-{fight_tag}-"):
                suffix = ticker.split("-")[-1]  # e.g. KOTKODQ
                aggregates[suffix] = ticker

        if siblings:
            self.sibling_tickers = sorted(siblings)
            log.info(f"[MMA ARBER] Discovered {len(siblings)} MOV siblings for {event_ticker}: {self.sibling_tickers}")

        if aggregates:
            self.aggregate_tickers = aggregates
            log.info(f"[MMA ARBER] Discovered {len(aggregates)} aggregate markets: {self.aggregate_tickers}")

    def _get_fighter_abbrevs(self) -> tuple:
        """
        Extract the two fighter abbreviations from the event ticker.
        E.g. KXUFCMOV-26APR11PROULB -> event tag is 26APR11PROULB.
        The fighter abbrevs are embedded after the date portion.
        We derive them by checking which sibling suffixes exist.
        """
        suffixes = set()
        for ticker in self.sibling_tickers:
            suffix = ticker.split("-")[-1]  # e.g. PROKOTKODQ, ULBDEC, DRAWDRAW
            suffixes.add(suffix)

        # Known method suffixes
        methods = ["KOTKODQ", "DEC", "SUB"]
        fighters = set()
        for s in suffixes:
            if s == "DRAWDRAW":
                continue
            for m in methods:
                if s.endswith(m):
                    fighter = s[:-len(m)]
                    if fighter:
                        fighters.add(fighter)
                    break

        fighters = sorted(fighters)
        if len(fighters) == 2:
            return (fighters[0], fighters[1])
        return None

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
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not self.active or not full_top_bids or not full_top_offers or not full_market_state:
            return []

        # Discover siblings on first evaluate
        if not self.sibling_tickers:
            self._discover_siblings(full_market_state)
            if not self.sibling_tickers:
                return []

        desired_quotes = []

        # --- Strategy 1: 7-Way Lead-Lag Fade ---
        s1_quotes = self._evaluate_7way_fade(full_market_state, full_top_bids, full_top_offers)
        desired_quotes.extend(s1_quotes)

        # --- Strategy 2: 3-Way Aggregate Arb ---
        s2_quotes = self._evaluate_3way_arb(full_market_state, full_top_bids, full_top_offers)
        desired_quotes.extend(s2_quotes)

        return desired_quotes

    def _evaluate_7way_fade(
        self,
        full_market_state: Dict[str, Any],
        full_top_bids: Dict[str, float],
        full_top_offers: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        When one MOV outcome's YES bid >= leader_threshold, buy NO on the other 6 via IOC.
        """
        # Identify the leader
        leader_ticker = None
        leader_bid = 0.0
        multiple_leaders = False

        for ticker in self.sibling_tickers:
            bid = full_top_bids.get(ticker, 0.0)
            if bid >= self.leader_threshold:
                if leader_ticker is not None:
                    multiple_leaders = True
                    break
                leader_ticker = ticker
                leader_bid = bid

        if multiple_leaders:
            log.warning(f"[MMA 7-WAY] Multiple leaders detected (>={self.leader_threshold}c). Ambiguous signal, skipping.")
            return []

        if leader_ticker is None:
            return []

        # Leader found — sweep the laggers
        desired_quotes = []
        base_vol = self.config.volumes[0] if self.config.volumes else 1
        should_log = (time.time() - self.last_log_time) > 5.0

        if should_log:
            log.info(f"[MMA 7-WAY] LEADER DETECTED: {leader_ticker} YES bid = {leader_bid}c")

        for lagger in self.sibling_tickers:
            if lagger == leader_ticker:
                continue

            lagger_status = full_market_state.get(lagger, {}).get("status", "")
            if lagger_status != "active":
                continue

            lagger_bid = full_top_bids.get(lagger, 0.0)

            # Skip if lagger has already converged to ~0
            if lagger_bid <= 1:
                continue

            # Cost to buy NO = 100 - YES_bid
            no_cost = 100.0 - lagger_bid
            loaded_no_cost = get_loaded_cost_cents(no_cost)

            # Check available volume from the YES side of the orderbook
            # (we are matching against YES bids by buying NO)
            ob = full_market_state.get(lagger, {}).get("raw_ob", {}).get("orderbook_fp", {})
            yes_pts = ob.get("yes_dollars", [])
            avail = int(float(yes_pts[0][1])) if yes_pts else 0

            trade_vol = min(base_vol, avail)
            if trade_vol <= 0:
                continue

            # Position check: buying NO = going short YES = negative position
            current_pos = position_store.get_position(lagger)
            if current_pos - trade_vol < -self.config.max_position:
                trade_vol = max(0, current_pos + self.config.max_position)
                if trade_vol <= 0:
                    continue

            desired_quotes.append({
                "side": QuoteSide.BID,
                "ticker": lagger,
                "kalshi_side": "no",
                "size": trade_vol,
                "limit_cents": int(math.ceil(no_cost)),
                "time_in_force": "immediate_or_cancel",
            })

            if should_log:
                edge = 100.0 - loaded_no_cost
                log.info(
                    f"  [MMA 7-WAY SWEEP] {lagger} | YES bid={lagger_bid:.0f}c | "
                    f"NO cost={no_cost:.0f}c (loaded={loaded_no_cost:.1f}c) | "
                    f"Edge={edge:.1f}c | Vol={trade_vol} | Pos={current_pos}"
                )

        if should_log and desired_quotes:
            self.last_log_time = time.time()

        return desired_quotes

    def _evaluate_3way_arb(
        self,
        full_market_state: Dict[str, Any],
        full_top_bids: Dict[str, float],
        full_top_offers: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        3-Way Aggregate Arb: Fighter A KO + Fighter B KO = Fight ends KO.
        Detect lead-lag between the aggregate and its two components.

        Direction 1: Aggregate leads (e.g. Fight KO bid >= 90) ->
                     Buy Fighter A KO YES + Fighter B KO YES (the cheaper synthetic)
        Direction 2: One fighter KO leads (e.g. Fighter A KO bid >= 90) ->
                     Buy aggregate Fight KO YES
        """
        if not self.aggregate_tickers:
            return []

        fighters = self._get_fighter_abbrevs()
        if not fighters:
            return []
        f1, f2 = fighters

        desired_quotes = []
        base_vol = self.config.volumes[0] if self.config.volumes else 1

        for method_suffix, agg_ticker in self.aggregate_tickers.items():
            # Find the two fighter-specific tickers for this method
            f1_ticker = None
            f2_ticker = None
            for sib in self.sibling_tickers:
                sib_suffix = sib.split("-")[-1]
                if sib_suffix == f"{f1}{method_suffix}":
                    f1_ticker = sib
                elif sib_suffix == f"{f2}{method_suffix}":
                    f2_ticker = sib

            if not f1_ticker or not f2_ticker:
                continue

            # Check all three markets are active
            agg_status = full_market_state.get(agg_ticker, {}).get("status", "")
            f1_status = full_market_state.get(f1_ticker, {}).get("status", "")
            f2_status = full_market_state.get(f2_ticker, {}).get("status", "")
            if not all(s == "active" for s in [agg_status, f1_status, f2_status]):
                continue

            agg_bid = full_top_bids.get(agg_ticker, 0.0)
            agg_offer = full_top_offers.get(agg_ticker, 100.0)
            f1_bid = full_top_bids.get(f1_ticker, 0.0)
            f1_offer = full_top_offers.get(f1_ticker, 100.0)
            f2_bid = full_top_bids.get(f2_ticker, 0.0)
            f2_offer = full_top_offers.get(f2_ticker, 100.0)

            # Synthetic cost to replicate aggregate via components
            component_cost = f1_offer + f2_offer
            component_loaded = get_loaded_cost_cents(f1_offer) + get_loaded_cost_cents(f2_offer)

            # Direction 1: Aggregate leads -> buy components
            if agg_bid >= self.leader_threshold:
                edge = agg_bid - component_loaded
                if edge >= self.config.min_edge:
                    # Buy Fighter 1 method YES
                    f1_quotes = self._build_component_buy(
                        f1_ticker, f1_offer, full_market_state, base_vol
                    )
                    # Buy Fighter 2 method YES
                    f2_quotes = self._build_component_buy(
                        f2_ticker, f2_offer, full_market_state, base_vol
                    )
                    desired_quotes.extend(f1_quotes)
                    desired_quotes.extend(f2_quotes)

                    log.info(
                        f"[MMA 3-WAY] AGG LEADS: {agg_ticker} bid={agg_bid:.0f}c | "
                        f"Components: {f1_ticker} offer={f1_offer:.0f}c + {f2_ticker} offer={f2_offer:.0f}c = {component_cost:.0f}c | "
                        f"Edge={edge:.1f}c"
                    )

            # Direction 2: One fighter leads -> buy aggregate
            for lead_ticker, lead_bid in [(f1_ticker, f1_bid), (f2_ticker, f2_bid)]:
                if lead_bid >= self.leader_threshold:
                    agg_loaded = get_loaded_cost_cents(agg_offer)
                    # The synthetic value of the aggregate = sum of component bids
                    # If lead fighter KO is 95c, the aggregate should be >= 95c
                    # Edge = lead_bid - agg_loaded_offer
                    edge = lead_bid - agg_loaded
                    if edge >= self.config.min_edge:
                        agg_quotes = self._build_component_buy(
                            agg_ticker, agg_offer, full_market_state, base_vol
                        )
                        desired_quotes.extend(agg_quotes)

                        log.info(
                            f"[MMA 3-WAY] FIGHTER LEADS: {lead_ticker} bid={lead_bid:.0f}c | "
                            f"Agg offer: {agg_ticker}={agg_offer:.0f}c (loaded={agg_loaded:.1f}c) | "
                            f"Edge={edge:.1f}c"
                        )

        return desired_quotes

    def _build_component_buy(
        self,
        ticker: str,
        offer_price: float,
        full_market_state: Dict[str, Any],
        base_vol: int,
    ) -> List[Dict[str, Any]]:
        """Build an IOC BUY YES order for a lagging component."""
        if offer_price >= 99 or offer_price <= 1:
            return []

        # Check available volume
        ob = full_market_state.get(ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})
        no_pts = ob.get("no_dollars", [])
        avail = int(float(no_pts[0][1])) if no_pts else 0

        trade_vol = min(base_vol, avail)
        if trade_vol <= 0:
            return []

        # Position check: buying YES = positive position
        current_pos = position_store.get_position(ticker)
        if current_pos + trade_vol > self.config.max_position:
            trade_vol = max(0, self.config.max_position - current_pos)
            if trade_vol <= 0:
                return []

        return [{
            "side": QuoteSide.BID,
            "ticker": ticker,
            "kalshi_side": "yes",
            "size": trade_vol,
            "limit_cents": int(math.floor(offer_price)),
            "time_in_force": "immediate_or_cancel",
        }]
