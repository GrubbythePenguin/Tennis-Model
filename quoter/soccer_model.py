import logging
import math
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, List, Any
from base_model import BaseTheoGenerator
from framework_config import QuoterConfig

log = logging.getLogger(__name__)

class SoccerTheoGenerator(BaseTheoGenerator):
    """
    Specifically engineered for soccer (MLS) Moneyline and Over/Under probabilities.
    Features empirical time-decay logic, red-card state adjustments, and Tri-State
    quoting (Bid/Offer boundaries based on next-goal bounds).
    """
    def __init__(self, client: Any, configs: List[QuoterConfig] = None):
        super().__init__(client, configs)
        self.game_params = {} # Maps game_tag -> {'la': 1.5, 'lb': 1.0, 'rho': 0.0, 'calibrated': False}
        
        # Precompute decay curve
        self.decay_curve = [self._get_decay_multiplier(m) for m in range(1, 91)]
        self.total_decay_area = sum(self.decay_curve)

    def _get_decay_multiplier(self, minute: int) -> float:
        if 1 <= minute <= 15: return 0.82
        elif 16 <= minute <= 30: return 0.98
        elif 31 <= minute <= 45: return 1.30
        elif 46 <= minute <= 60: return 0.90
        elif 61 <= minute <= 75: return 1.05
        else: return 1.60

    def _bivariate_poisson_probability(self, x, y, lambda_a, lambda_b, rho):
        base_prob = poisson.pmf(x, lambda_a) * poisson.pmf(y, lambda_b)
        if x == 0 and y == 0: adjustment = 1 - (lambda_a * lambda_b * rho)
        elif x == 0 and y == 1: adjustment = 1 + (lambda_a * rho)
        elif x == 1 and y == 0: adjustment = 1 + (lambda_b * rho)
        elif x == 1 and y == 1: adjustment = 1 - rho
        else: adjustment = 1.0
        return max(0.0, base_prob * adjustment)

    def _build_probability_matrix(self, lambda_a, lambda_b, rho, max_goals=10):
        matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                matrix[x, y] = self._bivariate_poisson_probability(x, y, lambda_a, lambda_b, rho)
        return matrix

    def _calculate_implied_market_probs(self, matrix, current_home=0, current_away=0):
        prob_home_win, prob_away_win, prob_draw = 0.0, 0.0, 0.0
        prob_o1_5, prob_o2_5, prob_o3_5, prob_o4_5 = 0.0, 0.0, 0.0, 0.0
        
        max_goals = matrix.shape[0]
        for rem_home in range(max_goals):
            for rem_away in range(max_goals):
                final_home = current_home + rem_home
                final_away = current_away + rem_away
                prob = matrix[rem_home, rem_away]
                
                # 3-Way Moneyline
                if final_home > final_away: prob_home_win += prob
                elif final_home < final_away: prob_away_win += prob
                else: prob_draw += prob
                
                # Over probabilities 
                total_goals = final_home + final_away
                if total_goals > 1.5: prob_o1_5 += prob
                if total_goals > 2.5: prob_o2_5 += prob
                if total_goals > 3.5: prob_o3_5 += prob
                if total_goals > 4.5: prob_o4_5 += prob
                    
        return {
            'home_win': prob_home_win,
            'away_win': prob_away_win,
            'draw': prob_draw,
            'over_1_5': prob_o1_5,
            'over_2_5': prob_o2_5,
            'over_3_5': prob_o3_5,
            'over_4_5': prob_o4_5
        }

    def calibrate(self, game_tag: str, target_probs: Dict[str, float]):
        def objective_function(params):
            la, lb, rho = params
            if la <= 0 or lb <= 0 or rho < -0.2 or rho > 0.2: return 1e6
                
            matrix = self._build_probability_matrix(la, lb, rho)
            implied = self._calculate_implied_market_probs(matrix)
            
            error = 0
            for k in target_probs.keys():
                if k in implied:
                    error += (implied[k] - target_probs[k]) ** 2
            return error

        result = minimize(objective_function, [1.5, 1.0, 0.0], method='Nelder-Mead', options={'xatol': 1e-4, 'disp': False})
        self.game_params[game_tag] = {
            'la': result.x[0],
            'lb': result.x[1],
            'rho': result.x[2],
            'calibrated': True
        }
        log.info(f"Soccer Model Calibrated for {game_tag}! Home_Exp={result.x[0]:.3f}, Away_Exp={result.x[1]:.3f}")
        return result.success

    def get_live_probabilities(self, game_tag, minute, current_home, current_away, home_red=0, away_red=0):
        params = self.game_params.get(game_tag, {})
        if not params.get('calibrated', False):
            # Fallback softly so things don't crash before first calibration
            params = {'la': 1.3, 'lb': 1.3, 'rho': 0.0}
            
        if minute >= 90:
            return self._calculate_implied_market_probs(np.zeros((10,10)), current_home, current_away)

        remaining_area = sum(self.decay_curve[max(1, minute)-1:])
        time_factor = remaining_area / self.total_decay_area
        
        rem_la = params['la'] * time_factor
        rem_lb = params['lb'] * time_factor
        
        if home_red > 0:
            rem_la *= 0.35 
            rem_lb *= 1.30 
        if away_red > 0:
            rem_lb *= 0.35
            rem_la *= 1.30
            
        matrix = self._build_probability_matrix(rem_la, rem_lb, params['rho'])
        return self._calculate_implied_market_probs(matrix, current_home, current_away)

    def _batch_generate(self, tickers: List[str], dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        results = {}
        for ticker in tickers:
            conf = self.configs.get(ticker)
            if not conf:
                continue

            if conf.stop_quoting:
                results[ticker] = {"bid_theo": 0.0, "offer_theo": 100.0}
                continue

            market_state = dt_market_state.get(ticker, {})
            if not market_state or "minute" not in market_state:
                continue
                
            minute = market_state.get("minute", 0)
            home_score = market_state.get("home_score", 0)
            away_score = market_state.get("away_score", 0)
            home_red = market_state.get("home_red", 0)
            away_red = market_state.get("away_red", 0)
            target_market = market_state.get("market_type", "") # e.g. 'home_win', 'over_2_5'
            
            if target_market == "":
                continue

            try:
                game_tag = market_state.get("game_tag", "UNKNOWN")
                
                # Tri-State Pricing Bounds!
                # 1. Base Score 
                probs_current = self.get_live_probabilities(game_tag, minute, home_score, away_score, home_red, away_red)
                
                # 2. Home Scores Next
                probs_h_score = self.get_live_probabilities(game_tag, minute, home_score + 1, away_score, home_red, away_red)
                
                # 3. Away Scores Next
                probs_a_score = self.get_live_probabilities(game_tag, minute, home_score, away_score + 1, home_red, away_red)
                
                # Validate the market exists in our enum
                if target_market in probs_current:
                    theo_a = probs_current[target_market] * 100.0
                    theo_b = probs_h_score[target_market] * 100.0
                    theo_c = probs_a_score[target_market] * 100.0
                    
                    bid_theo = min(theo_a, theo_b, theo_c)
                    offer_theo = max(theo_a, theo_b, theo_c)
                    
                    # Hard stop to strictly cap values properly if Game has ended
                    if minute >= 90:
                        bid_theo = theo_a
                        offer_theo = theo_a
                    
                    results[ticker] = {
                        "bid_theo": float(bid_theo),
                        "offer_theo": float(offer_theo)
                    }
                    
                    log.debug(f"SOCCER GEN | {ticker} ({target_market}) [Scores: {home_score}-{away_score}] MinBid={bid_theo:.2f}c, MaxOffer={offer_theo:.2f}c")
                else:
                    log.warning(f"SOCCER GEN | Unknown market type requested for {ticker}: {target_market}")
            except Exception as e:
                log.error(f"SOCCER GEN | Mathematical execution failure for {ticker}: {e}")
                
        return results
