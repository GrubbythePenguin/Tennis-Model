import logging
from typing import Dict, List, Any
from framework_config import QuoterConfig
from base_model import BaseTheoGenerator
from vwap_model import KalshiVwapTheoGenerator
from mlb_model import MlbSpreadTheoGenerator, MlbTotalTheoGenerator
from soccer_model import SoccerTheoGenerator
from intermission_model import EsportsIntermissionTheoGenerator
from live_series_model import LiveSeriesTheoGenerator
from props_theo_generator import PropsTheoGenerator
from tennis_branch_model import TennisBranchTheoGenerator

log = logging.getLogger(__name__)

class HybridTheoGenerator(BaseTheoGenerator):
    """
    Overarching mathematical engine mapping facade. 
    It maintains entirely independent registries of hundreds of different 
    theoretical models dynamically piped strictly based on `model_name`.
    """
    def __init__(self, client: Any, configs: List[QuoterConfig] = None):
        self.models = {
            "vwap": KalshiVwapTheoGenerator(client, configs),
            "mlb_spread": MlbSpreadTheoGenerator(client, configs),
            "mlb_total": MlbTotalTheoGenerator(client, configs),
            "soccer": SoccerTheoGenerator(client, configs),
            "esports_intermission": EsportsIntermissionTheoGenerator(client, configs),
            "live_series": LiveSeriesTheoGenerator(client, configs),
            "props_bo3": PropsTheoGenerator(client, configs),
            # Tennis: bid/offer are the two one-point-ahead branches. See
            # tennis_branch_model.py - the bid/offer gap IS the bracket the
            # feed lag creates, so the model quotes a spread as wide as its
            # own uncertainty about the point it cannot see.
            "tennis_branch": TennisBranchTheoGenerator(client, configs),
        }
        super().__init__(client, configs)

    @property
    def configs(self):
        return getattr(self, "_configs", {})

    @configs.setter
    def configs(self, new_configs):
        self._configs = new_configs
        if hasattr(self, "models"):
            for engine in self.models.values():
                engine.configs = new_configs

    def _batch_generate(self, tickers: List[str], dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        # 1. Segregate processing arrays strictly by their natively assigned model execution target
        model_to_tickers = {}
        for t in tickers:
            conf = self.configs.get(t)
            m_name = conf.model_name if conf else "vwap"
            if m_name not in model_to_tickers:
                model_to_tickers[m_name] = []
            model_to_tickers[m_name].append(t)
            
        merged_raw_theos = {}
        
        # 2. Fire the mathematical bounds generator exactly onto its appropriate component layer!
        for m_name, group in model_to_tickers.items():
            engine = self.models.get(m_name)
            if not engine:
                log.error(f"ROUTER ERROR | No theoretical engine found matching model name: '{m_name}'. Skipping tickers: {group}")
                continue
                
            log.debug(f"HYBRID ROUTER | Delegating {len(group)} tickers -> '{m_name}' Math Pipeline...")
            
            # Ask the specific sub-model to natively calculate its batch
            # We strictly execute `_batch_generate` bypassing the Position Adjuster because Hybrid router
            # natively wraps and assesses the master bundle globally at the very end regardless of the model pipeline!
            group_raw_theos = engine._batch_generate(group, dt_market_state)
            
            merged_raw_theos.update(group_raw_theos)
            
        return merged_raw_theos
