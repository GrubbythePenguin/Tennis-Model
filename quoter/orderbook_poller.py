"""Phase 1 of the feed-integrity project.

Background thread that polls Kalshi REST `/markets/{ticker}/orderbook` for
every ticker the bot is currently tracking, storing the result in a
thread-safe in-memory `OrderbookRegistry`. Phase 0 (`feed_integrity_poller.py`)
proved REST is a reliable ground-truth source — 100% of 364 sampled cycles
where the bot's WS view said "empty book," REST showed full books.

This module does NOT alter the bot's decision path. Reads from the registry
are exposed for Phase 2 (where the manager's evaluator will prefer REST when
WS shows empty), but Phase 1 is observation only — confirms the registry
stays warm under live trading load and lets us start emitting in-line
[FEED CHECK] logs alongside [DIFF] events.

Failure containment: the poller thread is a daemon. Any exception inside
the loop is caught and logged; the loop continues after a brief backoff so
a bug here CANNOT take down run.py.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

# Ticker alias layer — translate synthetic → real at the REST URL boundary.
# The registry stays keyed on synthetic so book_view.RestBookView and the
# rest of the bot read consistently. Pass-through for non-aliased tickers.
import ticker_aliases

log = logging.getLogger("orderbook_poller")

KALSHI_BASE_URL = "https://external-api.kalshi.com"


@dataclass
class RestBook:
    """Snapshot of one ticker's orderbook as last seen via REST.

    Holds BOTH summary stats (for cheap reads) AND the full level data
    (`orderbook_fp` dict in Kalshi REST shape) so callers that want the
    same shape as the WS fetcher can use it directly.
    """
    ticker: str
    ts_unix: float
    best_yes_bid: int          # cents; 0 if no yes-bids visible
    best_yes_ask: int          # cents; 0 if no no-bids visible (= no yes-asks)
    depth_yes_count: float     # total qty across all yes_dollars levels
    depth_no_count: float      # total qty across all no_dollars levels
    yes_levels_n: int
    no_levels_n: int
    rtt_ms: float
    # Full level data, exact REST response shape:
    # {"yes_dollars": [["0.5700","12.00"], …], "no_dollars": […]}
    # Stored as default-factory dict so existing tests/callers that build
    # RestBook without the full payload still work.
    orderbook_fp: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.depth_yes_count == 0 and self.depth_no_count == 0

    @property
    def age_sec(self) -> float:
        return time.time() - self.ts_unix

    def to_ws_style(self) -> dict:
        """Return same shape as WSOrderbookFetcher.fetch_all() entries:
            {"orderbook_fp": {"yes_dollars": [...], "no_dollars": [...]}}
        Lets a RestBookView serve drop-in replacement for WS fetches.
        """
        return {"orderbook_fp": self.orderbook_fp}


class OrderbookRegistry:
    """Thread-safe in-memory store of the latest REST snapshot per ticker.

    Phase 1 callers only write. Phase 2 will add a `prefer_rest_if_ws_empty`
    helper on top of `get()` to let the manager fall back to REST when its
    WS-derived view is suspect.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._books: dict[str, RestBook] = {}

    def update(self, book: RestBook) -> None:
        with self._lock:
            self._books[book.ticker] = book

    def get(self, ticker: str) -> Optional[RestBook]:
        with self._lock:
            return self._books.get(ticker)

    def remove(self, ticker: str) -> None:
        with self._lock:
            self._books.pop(ticker, None)

    def snapshot(self) -> dict[str, RestBook]:
        with self._lock:
            return dict(self._books)

    def __len__(self) -> int:
        with self._lock:
            return len(self._books)


class _LocalRateLimit:
    """Rolling 1-second cap, independent of KalshiClient's global cap.

    Reasoning: the poller's cadence is predictable (1 poll/ticker/second),
    so giving it its own bucket prevents it from competing with the bot's
    writes/reads inside the client's 75 RPS cap. Cap is generous enough to
    saturate sensible ticker counts while leaving margin for bursts.
    """

    def __init__(self, cap_rps: float) -> None:
        self.cap = cap_rps
        self.events: deque = deque()
        self.lock = threading.Lock()

    def consume(self) -> None:
        with self.lock:
            now = time.time()
            while self.events and now - self.events[0] > 1.0:
                self.events.popleft()
            if len(self.events) >= self.cap:
                sleep_for = 1.0 - (now - self.events[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.time()
                while self.events and now - self.events[0] > 1.0:
                    self.events.popleft()
            self.events.append(now)


class OrderbookPoller:
    """Background thread maintaining the OrderbookRegistry from REST polls.

    ticker_provider: zero-arg callable returning the current set of tickers
    the bot wants polled. Called every TICKER_REFRESH_SEC. Returning a
    smaller set causes dropped tickers to be removed from the registry.
    """

    POLL_INTERVAL_SEC = 1.0
    TICKER_REFRESH_SEC = 30.0
    CONSECUTIVE_404_DROP = 3
    BACKOFF_AFTER_ERROR_SEC = 5.0
    LOCAL_RPS_CAP = 15.0  # 2026-06-10 03:00 UTC — empirically pinned at 15 RPS.
                          # Smart-sweep probe (_smart_sweep.py) confirmed:
                          #   15 RPS sustained 6 min → 0% bulk-fetch 429s ✓
                          #   16 RPS sustained 6 min → 12.5% bulk-fetch 429s ✗
                          #   25 RPS sustained 6 min → 100% bulk-fetch 429s ✗
                          # The bulk endpoint's bucket is drained by sustained
                          # per-ticker orderbook polling. Adaptive RPS (drop
                          # to 10 during bulk fetch) does NOT help — high
                          # sustained rates poison the bucket for minutes,
                          # brief dips can't recover. 15 is the hard ceiling.
                          # For 16 tickers this gives 16/15 = 1.07s per ticker
                          # — 7% above 1s target, well within eval-cycle noise.

    def __init__(
        self,
        client,
        registry: OrderbookRegistry,
        ticker_provider: Callable[[], set[str]],
        enabled: bool = True,
    ) -> None:
        self.client = client
        self.registry = registry
        self.ticker_provider = ticker_provider
        self.enabled = enabled

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._consecutive_404: dict[str, int] = {}
        self._rate_limit = _LocalRateLimit(self.LOCAL_RPS_CAP)

        # Dedicated session so connection pooling is hot for the polling
        # workload without sharing pool capacity with the client's other
        # request flows.
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self._session.mount("https://", adapter)

        self._cycles = 0
        self._last_heartbeat = 0.0

    def start(self) -> None:
        if not self.enabled:
            log.info("[POLLER] disabled by config; skipping start.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="orderbook-poller",
        )
        self._thread.start()
        log.info(
            f"[POLLER] started (poll_interval={self.POLL_INTERVAL_SEC}s, "
            f"ticker_refresh={self.TICKER_REFRESH_SEC}s, local_rps={self.LOCAL_RPS_CAP})"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run_loop(self) -> None:
        last_discover = 0.0
        tickers: set[str] = set()
        while not self._stop.is_set():
            try:
                now = time.time()
                if now - last_discover > self.TICKER_REFRESH_SEC:
                    try:
                        new_tickers = set(self.ticker_provider() or [])
                    except Exception as e:
                        log.warning(f"[POLLER] ticker_provider error: {e}")
                        new_tickers = tickers
                    if new_tickers != tickers:
                        removed = tickers - new_tickers
                        added = new_tickers - tickers
                        for t in removed:
                            self.registry.remove(t)
                            self._consecutive_404.pop(t, None)
                        if added or removed:
                            log.info(
                                f"[POLLER] tickers added={len(added)} removed={len(removed)} "
                                f"total={len(new_tickers)}"
                            )
                        tickers = new_tickers
                    last_discover = now

                active = sorted(
                    t for t in tickers
                    if self._consecutive_404.get(t, 0) < self.CONSECUTIVE_404_DROP
                )

                cycle_start = time.time()
                for ticker in active:
                    if self._stop.is_set():
                        break
                    self._poll_one(ticker)

                self._cycles += 1
                if now - self._last_heartbeat > 60.0:
                    self._last_heartbeat = now
                    snap = self.registry.snapshot()
                    n_empty = sum(1 for b in snap.values() if b.is_empty)
                    log.info(
                        f"[POLLER] heartbeat cycles={self._cycles} "
                        f"registry={len(snap)} empty_books={n_empty} "
                        f"429s={getattr(self, '_n_429s', 0)} "
                        f"dropped404={sum(1 for n in self._consecutive_404.values() if n >= self.CONSECUTIVE_404_DROP)}"
                    )

                elapsed = time.time() - cycle_start
                if elapsed < self.POLL_INTERVAL_SEC:
                    self._stop.wait(self.POLL_INTERVAL_SEC - elapsed)
            except Exception as e:
                log.exception(f"[POLLER] run-loop exception: {e}; backing off {self.BACKOFF_AFTER_ERROR_SEC}s")
                self._stop.wait(self.BACKOFF_AFTER_ERROR_SEC)
        log.info("[POLLER] stopped.")

    def _poll_one(self, ticker: str) -> None:
        # Translate to real ticker for the wire; registry storage below stays
        # keyed on synthetic so RestBookView reads cleanly. Pass-through for
        # non-aliased.
        wire_ticker = ticker_aliases.resolve(ticker)
        path = f"/trade-api/v2/markets/{wire_ticker}/orderbook"
        self._rate_limit.consume()
        try:
            headers = self.client.auth.get_headers("GET", path)
        except Exception as e:
            log.warning(f"[POLLER] {ticker} auth error: {e}")
            return
        t0 = time.time()
        try:
            r = self._session.get(f"{KALSHI_BASE_URL}{path}", headers=headers, timeout=5.0)
        except Exception as e:
            log.warning(f"[POLLER] {ticker} request error: {e}")
            return
        rtt_ms = (time.time() - t0) * 1000.0

        if r.status_code == 404:
            self._consecutive_404[ticker] = self._consecutive_404.get(ticker, 0) + 1
            return
        if r.status_code == 429:
            # Track 429s separately so we can see the poller's contribution
            # to /markets family pressure. Was previously silent.
            self._n_429s = getattr(self, "_n_429s", 0) + 1
            return
        if r.status_code != 200:
            return  # transient — try again next cycle
        self._consecutive_404.pop(ticker, None)

        try:
            book = r.json().get("orderbook_fp") or {}
        except Exception:
            return

        yes_levels = book.get("yes_dollars") or []
        no_levels = book.get("no_dollars") or []

        def _cents(p) -> int:
            try:
                return int(round(float(p) * 100))
            except (ValueError, TypeError):
                return 0

        def _qty(q) -> float:
            try:
                return float(q)
            except (ValueError, TypeError):
                return 0.0

        best_yes_bid = max((_cents(l[0]) for l in yes_levels if l), default=0)
        best_no_bid = max((_cents(l[0]) for l in no_levels if l), default=0)
        best_yes_ask = (100 - best_no_bid) if best_no_bid > 0 else 0
        depth_yes_count = sum(_qty(l[1]) for l in yes_levels if len(l) > 1)
        depth_no_count = sum(_qty(l[1]) for l in no_levels if len(l) > 1)

        self.registry.update(RestBook(
            ticker=ticker,
            ts_unix=time.time(),
            best_yes_bid=best_yes_bid,
            best_yes_ask=best_yes_ask,
            depth_yes_count=depth_yes_count,
            depth_no_count=depth_no_count,
            yes_levels_n=len(yes_levels),
            no_levels_n=len(no_levels),
            rtt_ms=rtt_ms,
            # Full level data so RestBookView can serve as a WS-fetcher
            # drop-in. Storing the original Kalshi REST shape (string
            # prices/qtys) so consumers that already parse that shape
            # don't need extra translation.
            orderbook_fp={
                "yes_dollars": yes_levels,
                "no_dollars": no_levels,
            },
        ))
