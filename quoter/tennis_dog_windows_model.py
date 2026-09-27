"""Dispatcher hosting BOTH set-break dog makers behind one config row per ticker.

Registered as model_name = "tennis_dog_windows".

WHY A DISPATCHER. The framework routes ONE model per ticker ({ticker: conf} in the
theo registry — last row wins), but the 1-0 dog-leader cell (tennis_set1_dog, all
six BO3 tennis series) and the 1-1 set-3 dog cell (tennis_set3_dog, ITF only) both
want the same ITF tickers. They can safely cohabit: set1 arms only at sets 1-0,
set3 only at 1-1, each with its own one-shot latch, state file, events log, kill
flags and live flag — so a single row routed here lets both watch every match.

Merging: outputs are unioned; a same-ticker collision is structurally impossible
(a match cannot be at 1-0 and 1-1 in one snapshot) and is treated as a bug — the
set3 side wins and an error is logged, never a crossed/blended theo.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List

from base_model import BaseTheoGenerator
from tennis_set1_dog_model import Set1DogTheoGenerator, TAPES
from tennis_set3_dog_model import Set3DogTheoGenerator
from tennis_set1_tau_model import Set1TauTheoGenerator, KILL_FLAG as _TAU_KILL

log = logging.getLogger(__name__)

_SET3_SERIES = ("KXITFMATCH", "KXITFWMATCH")
_ITF_SERIES = _SET3_SERIES


class DogWindowsTheoGenerator(BaseTheoGenerator):
    """set1 (all six series) + set3 (ITF) behind one model_name."""

    def __init__(self, client: Any, configs: List[Any] = None):
        self.set1 = Set1DogTheoGenerator(client, configs)
        self.set3 = Set3DogTheoGenerator(client, configs)
        # tau-gated ITF variant (26SEP17): while enabled it OWNS the ITF 1-0
        # window and the legacy set1 model keeps only the non-ITF series;
        # touching disable_set1_tau.flag restores the legacy routing wholesale.
        self.set1tau = Set1TauTheoGenerator(client, configs)
        super().__init__(client, configs)

    # configs arrive as a {ticker: conf} dict from the manager / Hybrid setter and
    # must reach both sub-models, whose own gating reads self.configs.
    @property
    def configs(self):
        return getattr(self, "_configs", {})

    @configs.setter
    def configs(self, new_configs):
        self._configs = new_configs
        if hasattr(self, "set1"):
            self.set1.configs = new_configs
            self.set3.configs = new_configs
            self.set1tau.configs = new_configs

    def _log_wsbook(self, tickers: List[str], dt_market_state: Dict[str, Any]):
        """1 Hz per-event record of the quoter's WS top-of-book, tapes/wsbook_<ev>.jsonl.

        Same file/format tennis_recenter_model wrote and poll_tennis._wsbook reads
        ({"ts", "me": [bid_c, ask_c], "opp": [...]}, cents). Purpose (26SEP11, GET
        audit): with this present the ITF pollers run --markets-every 0 and take the
        /markets endpoint load to ZERO — the every-cycle GETs were 429ing on nearly
        every poll at ~15 live pollers, stalling tapes past the staleness cutoff and
        killing 3 of the first 4 shadow windows. Books here are the WS feed the
        quoter already holds — no new requests of any kind.
        """
        now = time.time()
        tb = dt_market_state.get("__top_bids__") or {}
        to = dt_market_state.get("__top_offers__") or {}
        if not tb and not to:
            return
        if not hasattr(self, "_last_wslog"):
            self._last_wslog = {}
        for event in {t.rsplit("-", 1)[0] for t in tickers}:
            if now - self._last_wslog.get(event, 0.0) < 1.0:
                continue
            me_tick, opp_tick = self.set1._meta(event)[:2]
            if not me_tick or not opp_tick:
                continue
            tops = (tb.get(me_tick), to.get(me_tick), tb.get(opp_tick), to.get(opp_tick))
            if not any(t is not None for t in tops):
                continue
            self._last_wslog[event] = now
            try:
                with open(os.path.join(TAPES, f"wsbook_{event}.jsonl"), "a") as f:
                    f.write(json.dumps({"ts": round(now, 3), "me": [tops[0], tops[1]],
                                        "opp": [tops[2], tops[3]]}) + "\n")
            except Exception as e:
                log.warning("[DOG-WINDOWS] wsbook write failed for %s: %s", event, e)

    def _batch_generate(self, tickers: List[str],
                        dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        self._log_wsbook(tickers, dt_market_state)
        itf = [t for t in tickers if t.split("-", 1)[0] in _SET3_SERIES]
        # ITF 1-0 routing (26SEP17): the tau model owns it unless its kill flag
        # is up, in which case the legacy set1 model sees ITF again. Non-ITF
        # always goes to the legacy model. Exactly one 1-0 model per ticker, so
        # a same-ticker collision between them is impossible by construction.
        tau_on = not os.path.exists(_TAU_KILL)
        set1_tickers = [t for t in tickers if t not in itf] if tau_on else tickers
        out = self.set1._batch_generate(set1_tickers, dt_market_state)
        if tau_on and itf:
            out.update(self.set1tau._batch_generate(itf, dt_market_state))
        out3 = self.set3._batch_generate(itf, dt_market_state) if itf else {}
        for t, th in out3.items():
            if t in out:
                log.error("[DOG-WINDOWS] %s emitted by BOTH a set1 model and set3 — "
                          "structurally impossible, keeping set3 (%s over %s)",
                          t, th, out[t])
        out.update(out3)
        return out
