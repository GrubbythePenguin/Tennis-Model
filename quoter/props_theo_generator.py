"""
PropsTheoGenerator — produces bid/offer theos for esports prop tickers.

Currently supports BO3 over/under 2.5 maps tickers across CS2, LoL, VAL, Dota2:
  - KXCS2TOTALMAPS-{event_base}-3       (CS2 BO3 O/U 2.5)
  - KXLOLTOTALMAPS-{event_base}-3       (LoL)
  - KXVALORANTTOTALMAPS-{event_base}-3  (VAL)
  - KXDOTA2TOTALMAPS-{event_base}-3     (Dota 2)

Pipeline:
  1. Parse prop ticker → derive underlying series + map prefixes + event_base
  2. Discover team suffixes from dt_market_state series-ticker keys
  3. Detect BO3 state via hedge_engine.detect_bo3_state
  4. Respect the disable_props_during_active_map.flag lever
  5. Pull g1 / g2 map book prices, compute theos via props_engine.bo3_prop_theos
  6. Emit {bid_theo, offer_theo} in cents matching the rest of the framework

Data source routing for the underlying map books happens automatically in
run.py via `_route_to_poly` — when the series's config_id has data_source=poly,
the map prices arrive in dt_market_state from Polymarket; when kalshi, from
Kalshi WS/REST. The props generator is source-agnostic — it just reads
whatever was injected.
"""
import logging
from typing import Dict, List, Any, Optional

from base_model import BaseTheoGenerator
from props_engine import bo3_prop_theos, should_emit_prop_theos
from hedge_engine import detect_bo3_state

log = logging.getLogger(__name__)


# Prop ticker prefix → (series prefix, map prefix) used to discover underlying
# tickers in dt_market_state.
_PROP_PREFIX_MAP = {
    "KXCS2TOTALMAPS":       ("KXCS2GAME",       "KXCS2MAP"),
    "KXLOLTOTALMAPS":       ("KXLOLGAME",       "KXLOLMAP"),
    "KXVALORANTTOTALMAPS":  ("KXVALORANTGAME",  "KXVALORANTMAP"),
    "KXDOTA2TOTALMAPS":     ("KXDOTA2GAME",     "KXDOTA2MAP"),
}


def parse_prop_ticker(ticker: str) -> Optional[Dict]:
    """Parse a totalmaps prop ticker.

    Returns dict {prop_prefix, series_prefix, map_prefix, event_base, threshold}
    or None if the ticker isn't a recognized prop shape.
    """
    parts = ticker.split("-")
    if len(parts) != 3:
        return None
    prop_prefix, event_base, threshold_str = parts
    if prop_prefix not in _PROP_PREFIX_MAP:
        return None
    try:
        threshold = int(threshold_str)
    except ValueError:
        return None
    series_prefix, map_prefix = _PROP_PREFIX_MAP[prop_prefix]
    return {
        "prop_prefix": prop_prefix,
        "series_prefix": series_prefix,
        "map_prefix": map_prefix,
        "event_base": event_base,
        "threshold": threshold,
    }


def _build_top_books(dt_market_state: Dict[str, Any]) -> tuple:
    """Build (full_top_bids, full_top_offers) cent-keyed dicts from
    dt_market_state — mirrors live_series_model._batch_generate.
    """
    full_top_bids: Dict[str, int] = {}
    full_top_offers: Dict[str, int] = {}
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
    # Manager-injected top-of-book fallback (covers Poly CLOB path).
    injected_bids = dt_market_state.get("__top_bids__", {})
    injected_offers = dt_market_state.get("__top_offers__", {})
    for t, v in injected_bids.items():
        full_top_bids.setdefault(t, v)
    for t, v in injected_offers.items():
        full_top_offers.setdefault(t, v)
    return full_top_bids, full_top_offers


def _discover_team_suffixes(
    series_prefix: str, event_base: str, full_top_bids: Dict[str, int],
) -> List[str]:
    """Find both team suffixes for an event by scanning series-ticker keys.

    Returns alphabetical-sorted list — caller treats [0] as "team A".
    """
    pattern = f"{series_prefix}-{event_base}-"
    suffixes = set()
    for t in full_top_bids:
        if t.startswith(pattern) and t.count("-") == 2:
            suffixes.add(t.split("-")[-1])
    return sorted(suffixes)


class PropsTheoGenerator(BaseTheoGenerator):
    """BaseTheoGenerator implementation for BO3 prop tickers."""

    def _batch_generate(
        self, tickers: List[str], dt_market_state: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        full_top_bids, full_top_offers = _build_top_books(dt_market_state)
        results: Dict[str, Dict[str, float]] = {}
        for ticker in tickers:
            parsed = parse_prop_ticker(ticker)
            if not parsed:
                continue
            theo = self._generate_one(parsed, full_top_bids, full_top_offers, dt_market_state)
            if theo is not None:
                results[ticker] = theo
        return results

    def _generate_one(
        self,
        parsed: Dict,
        full_top_bids: Dict[str, int],
        full_top_offers: Dict[str, int],
        dt_market_state: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        series_prefix = parsed["series_prefix"]
        map_prefix = parsed["map_prefix"]
        event_base = parsed["event_base"]

        team_suffixes = _discover_team_suffixes(series_prefix, event_base, full_top_bids)
        if len(team_suffixes) < 2:
            log.debug(f"[PROPS] {event_base}: only {len(team_suffixes)} teams found in dt_market_state")
            return None
        team_a, team_b = team_suffixes[0], team_suffixes[1]

        # State detection (from team A's perspective)
        try:
            state_info = detect_bo3_state(
                f"{map_prefix}-{event_base}",
                team_a,
                dt_market_state,
                full_top_bids,
                full_top_offers,
                series_base=f"{series_prefix}-{event_base}",
            )
        except Exception as e:
            log.debug(f"[PROPS] {event_base}: detect_bo3_state raised: {e}")
            return None
        state = state_info["state"] if state_info else None
        if state is None:
            return None

        # Runtime lever (disable_props_during_active_map.flag) — silently
        # skip emission when off, so quoter has nothing to quote.
        if not should_emit_prop_theos(state):
            return None

        # Pull g1 / g2 map prices for team A
        m1_a = f"{map_prefix}-{event_base}-1-{team_a}"
        m2_a = f"{map_prefix}-{event_base}-2-{team_a}"
        p1_a_bid_c = full_top_bids.get(m1_a)
        p1_a_ask_c = full_top_offers.get(m1_a)
        p2_a_bid_c = full_top_bids.get(m2_a)
        p2_a_ask_c = full_top_offers.get(m2_a)

        try:
            if state == 0:
                # Pre-game: need both g1 and g2 books.
                if any(v is None for v in (p1_a_bid_c, p1_a_ask_c, p2_a_bid_c, p2_a_ask_c)):
                    return None
                theos = bo3_prop_theos(
                    state=0,
                    p1_a_bid=p1_a_bid_c / 100.0,
                    p1_a_ask=p1_a_ask_c / 100.0,
                    p2_a=((p2_a_bid_c + p2_a_ask_c) / 2) / 100.0,
                )
            elif state in (1, 2):
                # Active g2: need only g2 books.
                if any(v is None for v in (p2_a_bid_c, p2_a_ask_c)):
                    return None
                theos = bo3_prop_theos(
                    state=state,
                    p1_a_bid=0.0, p1_a_ask=0.0,  # ignored in states 1/2
                    p2_a_bid=p2_a_bid_c / 100.0,
                    p2_a_ask=p2_a_ask_c / 100.0,
                )
            else:
                # state == 3 (decided): never emit (caller-side check too)
                return None
        except ValueError as e:
            log.debug(f"[PROPS] {event_base}: bo3_prop_theos rejected inputs: {e}")
            return None

        bid_c = theos.over_2_5_bid * 100.0
        ask_c = theos.over_2_5_ask * 100.0
        return {
            "bid_theo": bid_c,
            "offer_theo": ask_c,
            "raw_bid_theo": bid_c,
            "raw_offer_theo": ask_c,
        }
