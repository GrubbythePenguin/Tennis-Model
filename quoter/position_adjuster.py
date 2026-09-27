import logging
from typing import Dict, List, Any
from framework_config import QuoterConfig
import requests

log = logging.getLogger(__name__)


def compute_position_skew_shift_cents(pos: int, conf: QuoterConfig) -> float:
    """Linear-fractional position-skew magnitude in cents.

    Three control points:
      - skew_start_fraction × max_position: skew begins ramping (= 0 below)
      - skew_full_fraction  × max_position: skew reaches skew_max_shift_cents
      - skew_max_shift_cents: cap; skew stays here for any |pos| beyond `full`

    Default skew_full_fraction=1.0 preserves legacy "cap at max_position"
    behavior. Setting skew_full_fraction<1 makes the cap happen BEFORE the
    hard position limit, allowing a plateau zone above it.

    Shared by:
      - position_adjuster.adjust_theos (quoter — shifts bid/offer theos)
      - arber_bot sweep functions (arber — shifts min_edge gate)
    """
    my_fills = abs(pos)
    start = conf.skew_start_fraction * conf.max_position
    full_frac = getattr(conf, "skew_full_fraction", 1.0) or 1.0
    full = full_frac * conf.max_position
    if my_fills <= start or full <= start:
        return 0.0
    frac = min(1.0, (my_fills - start) / float(full - start))
    return frac * conf.skew_max_shift_cents


def variance_scale_at_theo(theo_c: float) -> float:
    """4*p*(1-p) variance shaping: peaks at theo=50c, shrinks toward 0 at extremes.

    p is clamped to [0.01, 0.99] to keep the multiplier strictly positive
    even at price edges. No floor — at extreme prices the skew is intentionally
    near-zero because there's little room for the book to move.
    """
    p = max(0.01, min(0.99, theo_c / 100.0))
    return 4.0 * p * (1.0 - p)


_LIVE_CONFIGS = {}


class PositionAdjuster:
    """
    Hook to adjust raw theos based on current portfolio exposure.
    Natively synchronizes Kalshi portfolio states via REST API.
    """
    def __init__(self, kalshi_client: Any, configs: List[QuoterConfig]):
        self.client = kalshi_client
        self.configs = {c.ticker: c for c in configs}
        
        # Sequentially trigger the massive boot REST syphon globally safely once!
        import position_store
        position_store.init_positions(self.client, list(self.configs.keys()))
        
    def _fetch_live_positions(self, active_tickers: List[str]) -> Dict[str, int]:
        """
        Dynamically queries the local lightning-fast position_store RAM matrix directly,
        bypassing all network traffic efficiently.
        """
        import position_store
        if not active_tickers:
            return {}
            
        positions_map = {}
        all_positions = position_store.get_all_positions()
        
        for t in active_tickers:
            if t in all_positions:
                positions_map[t] = all_positions[t]
                
        return positions_map
        
    def _publish(self):
        """Expose the live quoter configs so other modules (the arber's
        LEAD-LAG readout) can render retreat with the params actually applied."""
        global _LIVE_CONFIGS
        _LIVE_CONFIGS = self.configs

    def adjust_theos(self, raw_theos: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Takes in a dictionary mapped by ticker: 
        {"KX-123": {"bid_theo": 40.0, "offer_theo": 60.0}}
        
        Applies linear skew mapping natively converted from legacy quoter logic.
        Outputs the identical dictionary structure dynamically offset by risk loads.
        """
        if not raw_theos:
            return {}
        self._publish()
            
        # 1. Fetch exact live physical balances for actively evaluated tickers
        active_tickers = list(raw_theos.keys())
        raw_positions_map = self._fetch_live_positions(active_tickers)
        
        # 1.5. Dynamically Net Categorical Opposing Legs for Esports (LOL & CS2)
        positions_map = dict(raw_positions_map)
        esports_groups = {}
        
        for t in active_tickers:
            if "LOL" in t or "CS2" in t or "VALORANT" in t or "DOTA2" in t or "COD" in t or "ATP" in t or "WTA" in t or "ITF" in t:
                # Group by base Game ID (stripping the categorical suffix "-DRXC")
                game_id = "-".join(t.split("-")[:-1])
                esports_groups.setdefault(game_id, []).append(t)
                
        for game_id, tickers in esports_groups.items():
            # Only mathematically net if we identically identified exactly 2 opposing sides!
            if len(tickers) == 2:
                t1, t2 = tickers[0], tickers[1]
                pos1 = raw_positions_map.get(t1, 0)
                pos2 = raw_positions_map.get(t2, 0)
                
                # Formula: Net(Team1) = Pos(Team1) - Pos(Team2)
                # Buying YES Team1 (+1) natively equals Buying NO Team2 (-(-1))
                # Holding 5x YES on BOTH natively evaluates into 0 net directional exposure!
                positions_map[t1] = pos1 - pos2
                positions_map[t2] = pos2 - pos1

        adjusted_theos = {}
        
        # 2. Iterate each mapped theo bound to shift aggressively
        for ticker, theos in raw_theos.items():
            # Halt markers carry no fair value (forfeit / decided / no-state — see
            # live_series_model). Pass them through untouched so no phantom 50/50
            # bid/offer gets injected; the manager cancels-all + fires-nothing on them.
            if isinstance(theos, dict) and theos.get("halt"):
                adjusted_theos[ticker] = theos
                continue

            conf = self.configs.get(ticker)

            # If no tracking config, pass clean
            if not conf:
                adjusted_theos[ticker] = theos
                continue
                
            pos = positions_map.get(ticker, 0)
            base_shift = compute_position_skew_shift_cents(pos, conf)

            bid_t = theos.get("bid_theo", 50.0)
            off_t = theos.get("offer_theo", 50.0)

            shift_bid = base_shift * variance_scale_at_theo(bid_t)
            shift_off = base_shift * variance_scale_at_theo(off_t)
            if base_shift > 0:
                log.debug("Found Position Skew | ticker=%s pos=%d base_shift=%.2fc "
                          "shift_bid=%.2fc shift_off=%.2fc",
                          ticker, pos, base_shift, shift_bid, shift_off)

            if pos > 0:
                # We are LONG YES.
                # - Drop Bids downward natively making them conservative (aversive to buying more).
                # - Drop Offers downward natively making them aggressive (seeking desperately to sell exposure).
                bid_t -= shift_bid
                off_t -= shift_off
            elif pos < 0:
                # We are SHORT YES (Long NO).
                # - Raise Offers making them conservative (aversive to selling more).
                # - Raise Bids making them aggressive (seeking desperately to close isolated shorts).
                bid_t += shift_bid
                off_t += shift_off

            # Trade-retreat recency-flow term (TRADE_RETREAT_SCOPE.md,
            # armed 26AUG18). Additive with the static skew above: static
            # prices inventory risk, this prices flow information (decayed
            # signed net fill accumulation, sibling tickers netted). Gated
            # per-row on retreat_cap_cents > 0. Applied in ABSOLUTE cents —
            # the 10c-at-f_cap schedule was calibrated on realized c/lot
            # toxicity, so no variance rescaling. Same sign convention as
            # the skew: accumulating this ticker's team (f > 0) drops both
            # theos; the shift pins at the cap (no quote pull).
            r_cap = getattr(conf, "retreat_cap_cents", 0.0) or 0.0
            if r_cap > 0:
                r_parts = ticker.split("-")
                if len(r_parts) == 3:
                    try:
                        import retreat_shadow
                        r_f = retreat_shadow.get_f(
                            "-".join(r_parts[:2]), r_parts[2], conf.max_position,
                            half_life=getattr(conf, "retreat_half_life_sec", 300.0),
                            include_takers=getattr(conf, "retreat_include_takers", False))
                        r_shift = retreat_shadow.shift_cents(
                            abs(r_f), cap_cents=r_cap,
                            f0=getattr(conf, "retreat_f0", 0.022),
                            f_cap=getattr(conf, "retreat_f_cap", 0.18))
                        if r_shift > 0.05:
                            log.info("[RETREAT] %s f=%+.3f shift=%.2fc "
                                     "(cap=%gc, static_skew=%.2fc)",
                                     ticker, r_f, r_shift, r_cap, base_shift)
                            if r_f > 0:
                                bid_t -= r_shift
                                off_t -= r_shift
                            else:
                                bid_t += r_shift
                                off_t += r_shift
                    except Exception:
                        # Loud by design: a broken retreat term must never
                        # silently demote to "no retreat" (silent fallbacks
                        # are bugs). Theos proceed with the static skew only.
                        log.exception("[RETREAT] shift computation FAILED for %s "
                                      "— quoting with static skew only", ticker)


            # Structurally clamp the adjusted values universally preventing negative/impossible bounds natively!
            bid_t = max(0.0, min(100.0, bid_t))
            off_t = max(0.0, min(100.0, off_t))
                
            adjusted = dict(theos)
            adjusted["bid_theo"] = bid_t
            adjusted["offer_theo"] = off_t
            adjusted["raw_bid_theo"] = theos.get("bid_theo")
            adjusted["raw_offer_theo"] = theos.get("offer_theo")
            adjusted_theos[ticker] = adjusted
            
        return adjusted_theos
