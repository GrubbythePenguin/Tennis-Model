"""Track bid-zone transitions to distinguish "freshly-decided" markets from
"stale stuck books" for the loose-5c rule.

Rule (per the user's spec):
  - A bid is "fresh" if it was ≥5c and dropped to <5c within the last 60s.
  - Movements WITHIN the loose zone (3→2→3→4) do NOT refresh the timer —
    those are just micro-jitters of stuck orders.
  - When the bid leaves the loose zone (≥5c), the freshness state resets.
  - Next time it drops below 5c, the timer starts again.

State machine per ticker:
                  bid >= 5
                ┌────────────┐
                ▼            │
            [ABOVE] ────────►[BELOW]   (records transition_ts; "fresh")
                ▲   bid <5    │  bid stays <5 (3→2→3→4): no change
                │             │
                └─────────────┘
                  bid >= 5    (resets transition_ts)

Empty bids (bid<=0) are ignored — pre-game / no-data state.
"""
from __future__ import annotations
import time
from threading import Lock
from typing import Optional

LOOSE_THRESHOLD = 5            # cents — boundary for "loose-decided" zone
DEFAULT_FRESH_WINDOW_SEC = 60  # seconds — how recent a transition counts as "fresh"

_lock = Lock()
_zone: dict[str, str] = {}            # ticker -> 'above' | 'below' | 'empty'
_transition_ts: dict[str, float] = {} # ticker -> ts of last 'above'→'below' crossing


def _classify(bid: float) -> str:
    if bid is None or bid <= 0:
        return "empty"
    if bid < LOOSE_THRESHOLD:
        return "below"
    return "above"


def observe(ticker: str, bid: float) -> None:
    """Called per cycle by the main loop with each ticker's top-of-book bid.
    Records transition timestamp only on 'above' → 'below' crossings.
    Resets the timestamp when the bid leaves the below zone.
    """
    cur = _classify(bid)
    with _lock:
        prev = _zone.get(ticker)
        if cur == "below" and prev == "above":
            # Fresh transition into the loose zone.
            _transition_ts[ticker] = time.time()
        elif cur != "below":
            # Left the loose zone (or empty) — clear the timer.
            _transition_ts.pop(ticker, None)
        # 'below' → 'below' / 'above' → 'above' / 'empty' anywhere:
        # no timestamp change. Micro-movements within the zone don't refresh.
        _zone[ticker] = cur


def is_fresh(ticker: str, max_age_sec: float = DEFAULT_FRESH_WINDOW_SEC) -> bool:
    """True iff the ticker's bid is currently in the loose zone AND entered
    that zone within max_age_sec from above (≥5c)."""
    with _lock:
        if _zone.get(ticker) != "below":
            return False
        ts = _transition_ts.get(ticker)
        if ts is None:
            return False
        return (time.time() - ts) < max_age_sec


def seconds_since_transition(ticker: str) -> Optional[float]:
    """Diagnostic: seconds since the last 'above' → 'below' transition."""
    with _lock:
        ts = _transition_ts.get(ticker)
        return (time.time() - ts) if ts is not None else None


def get_zone(ticker: str) -> Optional[str]:
    """Diagnostic: current zone classification."""
    with _lock:
        return _zone.get(ticker)


def reset() -> None:
    """For tests: clear all tracking state."""
    with _lock:
        _zone.clear()
        _transition_ts.clear()
