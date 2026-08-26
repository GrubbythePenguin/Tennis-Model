import logging
import sys
from typing import Dict, List, Any
from base_model import BaseTheoGenerator
from framework_config import QuoterConfig

log = logging.getLogger(__name__)

class MlbSpreadTheoGenerator(BaseTheoGenerator):
    """
    Specifically engineered for MLB Spread and Game Result matrices.
    """
    def __init__(self, client: Any, configs: List[QuoterConfig] = None):
        super().__init__(client, configs)
        try:
            if "/Users/bradleyguan/Documents/Coding/kalshi_mlb_tracker" not in sys.path:
                sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_mlb_tracker")
            from spread_theoreticals import get_theoretical_spreads
            self.get_theoretical_spreads = get_theoretical_spreads
        except ImportError:
            log.warning("MLB Spread Theoreticals module could not be loaded into workspace.")
            self.get_theoretical_spreads = None

    def _batch_generate(self, tickers: List[str], dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        results = {}
        for ticker in tickers:
            conf = self.configs.get(ticker)
            if not conf:
                continue

            log.debug(f"MLB SPREAD GEN | Processing ticker {ticker}")
            if conf.stop_quoting:
                log.warning(f"MLB SPREAD GEN | STOP_QUOTING Enabled for {ticker}! Flatlining bounds.")
                results[ticker] = {"bid_theo": 0.0, "offer_theo": 100.0}
                continue

            market_state = dt_market_state.get(ticker, {})
            log.debug(f"MLB GENERATOR | Internal dt_market_state map keys found for {ticker}: {list(market_state.keys())}")
            
            # If state exists and we successfully imported the Old Kalshi MLB matrix
            if market_state and "inning" in market_state and self.get_theoretical_spreads:
                # Assuming the state is populated from identical formatting
                target_win_prob = market_state.get("target_win_prob", 0.5)
                # Parse specific spread line being quoted (e.g. 1.5)
                target_line = market_state.get("target_line", 1.5)
                
                # Rigid logic identically mirroring Kalshi structural boundaries
                if market_state.get("inning", 1) > market_state.get("target_inning_end", 9):
                    log.debug(f"MLB SPREAD GEN | Boundary structurally bypassed for {ticker}. Halting quoting inherently!")
                    results[ticker] = {"bid_theo": 0.0, "offer_theo": 100.0}
                    continue

                if market_state.get("inning", 1) >= 8:
                    log.debug(f"MLB SPREAD GEN | Inning {market_state.get('inning')} >= 8 for {ticker}. Model unreliable, flatlining.")
                    results[ticker] = {"bid_theo": 0.0, "offer_theo": 100.0}
                    continue

                log.debug(f"MLB GENERATOR | Attempting lookup -> target_prob={target_win_prob}, target_line={target_line}, inning={market_state.get('inning')}")
                try:
                    actual_prob_matrix = self.get_theoretical_spreads(
                        inning=market_state.get("inning", 1),
                        is_top=market_state.get("is_top", True),
                        outs=market_state.get("outs", 0),
                        base_map=market_state.get("base_map", 0),
                        target_win_prob=target_win_prob,
                        expected_game_total=market_state.get("expected_game_total", 8.5)
                    )
                    if actual_prob_matrix and "spreads" in actual_prob_matrix:
                        true_theo = actual_prob_matrix["spreads"].get(target_line, 0.5) * 100.0
                        log.debug(f"MLB GENERATOR | Successfully derived probability bounds for {ticker}! Target_Line={target_line} -> Theo {true_theo:.2f}c")
                    else:
                        log.error(f"MLB SPREAD GEN | Prob matrix calculation failed/returned empty tracking for {ticker}!")
                        continue
                except Exception as e:
                    log.error(f"MLB SPREAD GEN | CRITICAL: Failed to execute native MLB matrix for {ticker}: {e}")
                    continue
            else:
                log.debug(f"MLB SPREAD GEN | Missing Market State or MLB matrix hook for {ticker}. Skipping execution.")
                continue
            
            results[ticker] = {
                "bid_theo": float(true_theo), 
                "offer_theo": 100.0
            }
            
        return results

class MlbTotalTheoGenerator(BaseTheoGenerator):
    """
    Specifically engineered for MLB OVER/UNDER (Totals) probabilities.
    """
    def __init__(self, client: Any, configs: List[QuoterConfig] = None):
        super().__init__(client, configs)
        try:
            if "/Users/bradleyguan/Documents/Coding/kalshi_mlb_tracker" not in sys.path:
                sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_mlb_tracker")
            from trading.totals_theoreticals import get_theoretical_totals
            self.get_theoretical_totals = get_theoretical_totals
        except ImportError:
            log.warning("MLB Totals module could not be loaded.")
            self.get_theoretical_totals = None

    def _batch_generate(self, tickers: List[str], dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        results = {}
        for ticker in tickers:
            conf = self.configs.get(ticker)
            if not conf:
                continue
                
            log.debug(f"MLB TOTALS GEN | Processing ticker {ticker}")
            if conf.stop_quoting:
                results[ticker] = {"bid_theo": 0.0, "offer_theo": 100.0}
                continue

            market_state = dt_market_state.get(ticker, {})
            if market_state and "inning" in market_state and self.get_theoretical_totals:
                target_line = market_state.get("target_line", 8.5)
                
                if market_state.get("inning", 1) > market_state.get("target_inning_end", 9):
                    log.debug(f"MLB TOTALS GEN | Boundary structurally bypassed for {ticker}. Flatlining seamlessly!")
                    results[ticker] = {"bid_theo": 0.0, "offer_theo": 100.0}
                    continue

                if market_state.get("inning", 1) >= 8:
                    log.debug(f"MLB TOTALS GEN | Inning {market_state.get('inning')} >= 8 for {ticker}. Model unreliable, flatlining.")
                    results[ticker] = {"bid_theo": 0.0, "offer_theo": 100.0}
                    continue

                try:
                    # Provide exact payload required by get_theoretical_totals engine block
                    res_block = self.get_theoretical_totals(market_state, lines=[target_line])
                    
                    if res_block and "totals" in res_block:
                        # Output probability of hitting the OVER natively
                        true_theo = res_block["totals"].get(target_line, 0.5) * 100.0
                        log.debug(f"MLB TOTALS GEN | Bounded Limit: Line={target_line} -> {true_theo:.2f}c")
                    else:
                        log.error(f"MLB TOTALS GEN | Native totals model crashed for {ticker}!")
                        continue
                except Exception as e:
                    log.error(f"MLB TOTALS GEN | Network/Model execute failure: {e}")
                    continue
            else:
                log.debug(f"MLB TOTALS GEN | Missing State or totals API link for {ticker}.")
                continue
                
            results[ticker] = {
                "bid_theo": float(true_theo), 
                "offer_theo": 100.0
            }
            
        return results
