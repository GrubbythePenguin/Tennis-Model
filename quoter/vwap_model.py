import logging
from typing import Dict, List, Any
from base_model import BaseTheoGenerator

log = logging.getLogger(__name__)

class KalshiVwapTheoGenerator(BaseTheoGenerator):
    """
    Computes fair VWAP logic by recursively assessing Top Levels up to the configured limit.
    """

    def _calculate_side_vwap(self, orderbook_levels: List[List[Any]], max_levels: int) -> float:
        total_vol = 0
        total_cost = 0
        log.debug(f"VWAP CALC | Evaluating up to {max_levels} levels. Physical levels found: {len(orderbook_levels)}")
        for i, level in enumerate(orderbook_levels):
            if i >= max_levels:
                break
            price = int(level[0])
            qty = int(level[1])
            total_vol += qty
            total_cost += price * qty
            log.debug(f"   -> Level {i+1}: Price={price}c, Vol={qty}. Cumulative Vol={total_vol}, Cost={total_cost}")
            
        if total_vol == 0:
            log.debug("VWAP CALC | Zero volume found. Returning None.")
            return None
            
        vwap = total_cost / total_vol
        log.debug(f"VWAP CALC | Final Result: {vwap:.2f}c")
        return vwap

    def _batch_generate(self, tickers: List[str], dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        results = {}
        for ticker in tickers:
            conf = self.configs.get(ticker)
            if not conf:
                continue

            try:
                # Natively bypass the network by pulling the cached top-of-book state safely passed from the Orchestrator loop
                # This mathematically deletes 6 redundant API GET requests from the pipeline!
                ob = dt_market_state.get(ticker, {}).get("raw_ob")
                if not ob:
                    continue
                
                # Natively Support Kalshi V2 Float Formatting
                if "orderbook_fp" in ob:
                    y_pts = ob["orderbook_fp"].get("yes_dollars", [])
                    n_pts = ob["orderbook_fp"].get("no_dollars", [])
                    
                    # Sort Descending so Top Level (highest bid) is at index 0 explicitly
                    y_pts.sort(key=lambda x: float(x[0]), reverse=True)
                    n_pts.sort(key=lambda x: float(x[0]), reverse=True)
                    
                    # Map to cleanly parsable grids
                    yes_orders = [[int(float(p) * 100), int(float(v))] for p, v in y_pts]
                    no_orders = [[int(float(p) * 100), int(float(v))] for p, v in n_pts]
                else:
                    yes_orders = ob.get("orderbook", {}).get("yes", [])
                    no_orders = ob.get("orderbook", {}).get("no", [])
                    yes_orders.sort(key=lambda x: int(x[0]), reverse=True)
                    no_orders.sort(key=lambda x: int(x[0]), reverse=True)
                
                # Now explicitly mapped: index 0 is best bid. 
                best_bid = yes_orders[0][0] if yes_orders else 0
                best_offer = 100 - no_orders[0][0] if no_orders else 100
                
                width = best_offer - best_bid
                
                # Hardcoded constraints natively inside specific Math modules exclusively
                vwap_levels = 2 
                max_market_width = 15
                
                log.debug(f"VWAP GENERATOR | {ticker} Matrix: Top Bid {best_bid}c | Best Offer {best_offer}c | Width {width}c")
                
                # Check escape hatch bounds
                if width > max_market_width and width != 100:
                    midpoint = (best_bid + best_offer) / 2.0
                    results[ticker] = {"bid_theo": midpoint, "offer_theo": midpoint}
                    continue
                    
                a_vwap = self._calculate_side_vwap(yes_orders, vwap_levels)
                # Translated inverted side
                b_vwap_no = self._calculate_side_vwap(no_orders, vwap_levels)
                b_vwap = (100 - b_vwap_no) if b_vwap_no is not None else None
                
                final_vwap = 50.0
                if a_vwap is not None and b_vwap is not None:
                    final_vwap = (a_vwap + b_vwap) / 2.0
                elif a_vwap is not None:
                    final_vwap = a_vwap
                elif b_vwap is not None:
                    final_vwap = b_vwap
                
                results[ticker] = {"bid_theo": final_vwap, "offer_theo": final_vwap}
                
            except Exception as e:
                log.error("Failed to map VWAP theo for %s: %s", ticker, e)
                
        return results
