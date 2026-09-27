"""TT Elite theo: quote the delayed-feed worst-case band from the screener.

Registered as model_name = "tt_band".

WHY A BAND, NOT A LEVEL. The Kalshi TT score feed runs 2-10 points behind the
table. Any level quoted off the visible score is a level someone watching the
stream can pick off. So this model quotes the two prices the hidden points can
justify:

    bid_theo   = match value if the opponent wins the next N points straight
    offer_theo = match value if our player wins the next N points straight

(N = the screener's --delay-points, recorded in the snapshot). Whichever way
the unseen points actually went, the true value is inside the quoted band, so
a counterparty who already knows them gets fair value, not an edge — the same
philosophy as tennis_recenter, with one inversion: tennis re-anchors to the
market because a live market exists; TT Elite in-play books are empty or
stale (measured 2026-09-07: one two-sided quote in an entire match), so the
LEVEL comes from the model — the screener's pregame-anchored (p, q) evolved
by the score — and only the band edges are quoted.

The band is also exactly what the liquidity-rewards program wants: resting
size that counts toward the top-300-lots share without carrying pickoff risk.

DATA SOURCE. tapes/_tt_state.json, atomically replaced by tt_screener.py every
refresh (~12s). This process does ZERO score GETs — the screener is the single
score consumer. STALENESS IS A HARD STOP, as in every tennis model: a snapshot
(or per-event entry) older than MAX_AGE_S yields no theo, so the quoter pulls
rather than resting on a band nobody is maintaining. A match that is not
status=live — including a suspension, where Kalshi's postponed->$0.50 rule
makes held inventory dangerous — also yields no theo.

This file emits THEOS ONLY. Orders still require a config row naming
model_name=tt_band AND the execution-layer series allowlist to include
KXTTELITEMATCH — neither of which this module touches.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List

from base_model import BaseTheoGenerator

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_P = os.path.join(ROOT, "tapes", "_tt_state.json")
QUOTE_LOG_P = os.path.join(ROOT, "tapes", "tt_band_quotes.jsonl")

# Screener refreshes every ~12s. 60s = several missed cycles is a dead feed,
# not lag; the screener also refreshes BEFORE its slow discovery sweeps, so a
# healthy feed stays well under this.
MAX_AGE_S = 60.0


class TTBandTheoGenerator(BaseTheoGenerator):

    def __init__(self, client: Any, configs: List[Any] = None):
        super().__init__(client, configs)
        self._snap = {}
        self._snap_mtime = 0.0
        self._last_logged = {}        # event -> state list last written to the log

    # ------------------------------------------------------------- snapshot

    def _load(self):
        """Reload tapes/_tt_state.json iff it changed. Never raises."""
        try:
            mt = os.path.getmtime(SNAP_P)
        except OSError:
            self._snap = {}
            return
        if mt == self._snap_mtime:
            return
        try:
            with open(SNAP_P) as f:
                self._snap = json.load(f)
            self._snap_mtime = mt
        except Exception as e:                      # torn reads are impossible
            log.warning(f"[TT BAND] snapshot unreadable: {e}")  # (atomic replace),
            self._snap = {}                                     # but never crash a tick

    # ---------------------------------------------------------------- theos

    def _batch_generate(self, tickers: List[str],
                        dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        self._load()
        out: Dict[str, Dict[str, float]] = {}
        now = time.time()
        if not self._snap or now - float(self._snap.get("ts") or 0) > MAX_AGE_S:
            return out                              # dead feed -> quote nothing
        events = self._snap.get("events") or {}
        for ticker in tickers:
            if ticker in out or not self.configs.get(ticker):
                continue
            event = ticker.rsplit("-", 1)[0]
            e = events.get(event)
            if not e:
                continue
            if now - float(e.get("ts") or 0) > MAX_AGE_S:
                continue                            # this match stopped updating
            if not e.get("live") or e.get("ended"):
                continue                            # includes suspensions: no theo
                                                    # (post-decision 99s tried and
                                                    # reverted 2026-09-09 — Kalshi
                                                    # settles TT too fast to matter)
            qb, qa = e.get("qbid"), e.get("qask")
            # qb == 0.0 is a REAL state (N unseen points can end the match), not
            # a degenerate one: the bid side simply cannot rest — bid_theo=0
            # floors below 1c and calculate_quote_levels drops it — while the
            # offer still quotes. Skip only no-anchor and actually-decided
            # states (best case worthless / worst case certain).
            if qb is None or qa is None or qb > qa or qa <= 0.0 or qb >= 1.0:
                continue
            if ticker == e.get("t1"):
                bid_c, off_c = qb * 100.0, qa * 100.0
            elif ticker == e.get("t2"):
                bid_c, off_c = (1.0 - qa) * 100.0, (1.0 - qb) * 100.0
            else:
                log.warning(f"[TT BAND] {ticker}: not t1/t2 of {event} — skipping")
                continue
            out[ticker] = {"bid_theo": bid_c, "offer_theo": off_c}
            self._log_quote(event, e)
        return out

    # ------------------------------------------------------------------ log

    def _log_quote(self, event, e):
        """One line per score change per event — the audit trail of what the
        quoter was fed, without a line per tick."""
        st = e.get("state")
        if self._last_logged.get(event) == st:
            return
        self._last_logged[event] = st
        try:
            with open(QUOTE_LOG_P, "a") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event,
                    "state": st, "score": e.get("score"),
                    "theo": e.get("theo"), "qbid": e.get("qbid"),
                    "qask": e.get("qask"), "anchor": e.get("anchor"),
                    "pregame_anchor": e.get("pregame_anchor"),
                    "delay_points": self._snap.get("delay_points")}) + "\n")
        except Exception:
            pass
