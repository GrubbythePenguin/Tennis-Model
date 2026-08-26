"""Attribution for WS book misses — WHY a book could not be served.

Context (2026-07-28): `[WS FALLBACK]` fires hundreds of times an hour, but the
log only ever said WHICH tickers fell back, never why. Both books
(`isolated_ws_book`, `ws_delta_book`) return a bare `None` from
`get_orderbook_fp` for four structurally different reasons, and because both
apply the SAME guards with the SAME 60s threshold they fail together — so
book_view's partial-coverage merge recovers nothing and logs nothing
(it only logs `if recovered:`). The result is a silent REST fallback with no
attribution, which is what made the cause un-diagnosable from run.log.

Every site that decides "this ticker cannot be served from WS" now calls
`record()`. Consumers call `describe()` for a one-line per-ticker reason, and
run.py prints a throttled `[WS MISS ROLLUP]` so a session's fallback load can
be attributed to a cause rather than inferred from burst timing.

This module is PURE OBSERVABILITY. `record()` swallows every exception and
nothing here influences routing — a bug in attribution must never be able to
take down the read path.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

# Reason codes. Kept short — they appear inline in every fallback log line.
UNSEEDED = "unseeded"      # no book object at all (never received a snapshot)
NOT_LIVE = "not_live"      # book exists but is flagged not-live
STALE = "stale"            # no message within stale_book_s
EMPTY_LIVE = "empty_live"  # live and fresh, but every price level is gone
ERROR = "error"            # exception while reading the book

ALL_REASONS = (UNSEEDED, NOT_LIVE, STALE, EMPTY_LIVE, ERROR)

# Attribution older than this is not reported — a reason from two minutes ago
# says nothing about why THIS tick missed, and a confidently wrong reason is
# worse than "?".
_FRESH_S = 5.0

_lock = threading.Lock()
_last: dict[tuple[str, str], tuple[str, str, float]] = {}   # (src,tkr) -> (reason, detail, ts)
_counts: dict[tuple[str, str, str], int] = defaultdict(int)  # (src,tkr,reason) -> n
_served: dict[str, int] = defaultdict(int)                   # src -> successful serves
_since = time.time()


def record(source: str, ticker: str, reason: str, detail: str = "") -> None:
    """Note that `source` could not serve `ticker`, and why. Never raises."""
    try:
        now = time.time()
        with _lock:
            _last[(source, ticker)] = (reason, detail, now)
            _counts[(source, ticker, reason)] += 1
    except Exception:
        pass


def record_served(source: str) -> None:
    """Note a successful serve, so the rollup can show a miss RATE."""
    try:
        with _lock:
            _served[source] += 1
    except Exception:
        pass


def last_reason(source: str, ticker: str) -> str:
    """Most recent miss reason for (source, ticker), or '?' if stale/absent."""
    try:
        with _lock:
            v = _last.get((source, ticker))
        if not v:
            return "?"
        reason, detail, ts = v
        if (time.time() - ts) > _FRESH_S:
            return "?"
        return f"{reason}({detail})" if detail else reason
    except Exception:
        return "?"


def describe(ticker: str, sources: tuple[str, ...] = ("isolated", "ws_delta")) -> str:
    """One-line 'isolated=empty_live ws_delta=stale(72s)' for a ticker."""
    return " ".join(f"{s}={last_reason(s, ticker)}" for s in sources)


def rollup_and_reset() -> str | None:
    """Aggregate counts since the last call, then reset. None if nothing to say."""
    global _since
    try:
        with _lock:
            if not _counts:
                # Still reset the window so the next rollup's rate is honest.
                _since = time.time()
                _served.clear()
                return None
            per_ticker: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            per_reason: dict[str, int] = defaultdict(int)
            for (source, ticker, reason), n in _counts.items():
                per_ticker[ticker][f"{source}:{reason}"] += n
                per_reason[f"{source}:{reason}"] += n
            served = dict(_served)
            window = max(1e-6, time.time() - _since)
            _counts.clear()
            _served.clear()
            _since = time.time()

        totals = ", ".join(f"{k}={v}" for k, v in
                           sorted(per_reason.items(), key=lambda kv: -kv[1]))
        serve_str = ", ".join(f"{k}={v}" for k, v in sorted(served.items())) or "none"
        lines = [f"[WS MISS ROLLUP] {window:.0f}s window | served: {serve_str} | misses: {totals}"]
        # Worst offenders first — that is the list you act on.
        ranked = sorted(per_ticker.items(), key=lambda kv: -sum(kv[1].values()))
        for ticker, reasons in ranked[:8]:
            detail = ", ".join(f"{k}={v}" for k, v in
                               sorted(reasons.items(), key=lambda kv: -kv[1]))
            lines.append(f"    {ticker}: {detail}")
        return "\n".join(lines)
    except Exception:
        return None
