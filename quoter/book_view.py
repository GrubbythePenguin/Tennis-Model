"""BookView abstraction — Phase 2 of feed-integrity migration.

Provides a single interface (`BookView.fetch_all`) that the bot reads
orderbooks from. Two implementations:

- `WSBookView`: wraps the existing `WSOrderbookFetcher`. Preserves the
  current behavior — bit-for-bit identical results to pre-Phase-2 code.
- `RestBookView`: reads from the `OrderbookRegistry` that the REST
  orderbook poller maintains. Up to ~1s stale but immune to WS
  delta-application corruption that caused the VTCGL incident
  (2026-06-09 21:52 UTC).

Both implementations return the same shape as the WS fetcher always has:

    {ticker: {"orderbook_fp": {"yes_dollars": [...], "no_dollars": [...]}}}

so downstream parsing code in `run.py` / `manager.py` / `arber_bot.py`
needs no per-ticker changes — only the construction-site swap.

Migration plan: per-row `book_source=ws|rest` flag in `market_parameters.csv`
defaults to `ws`. Flip individual rows to `rest` once we trust the source.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import ws_miss_stats

log = logging.getLogger(__name__)

# How stale a REST snapshot can be before RestBookView treats it as "missing".
# At 1s poll cadence we expect ~500ms typical age; 3s is a generous ceiling
# that catches an unhealthy poller without false-positives during ordinary
# scheduling jitter.
REST_MAX_AGE_SEC = 3.0


class BookView(ABC):
    """Abstract orderbook source. Implementations return the same dict shape
    so callers can swap sources without other code changes."""

    @abstractmethod
    async def fetch_all(self, tickers: list[str]) -> dict:
        """Fetch orderbooks for `tickers`. Tickers without a fresh book are
        omitted from the result — callers should handle missing entries
        the same way they handle WS snapshot timeouts today.

        Returns:
            {ticker: {"orderbook_fp": {"yes_dollars": [...], "no_dollars": [...]}}}
        """

    async def close(self) -> None:
        """Optional cleanup. Default no-op; WS implementation overrides."""
        return None


class WSBookView(BookView):
    """Reads books via the existing WebSocket fetcher. Pass-through wrapper —
    behavior identical to pre-Phase-2 direct `ws_fetcher.fetch_all()` calls."""

    def __init__(self, ws_fetcher) -> None:
        self._ws = ws_fetcher

    async def fetch_all(self, tickers: list[str]) -> dict:
        return await self._ws.fetch_all(tickers)

    async def close(self) -> None:
        await self._ws.close()


class WSDeltaBookView(BookView):
    """Reads books from a long-lived `WSDeltaManager` (delta-applied WS books).

    Defensive: any exception from the manager OR a None result (book not yet
    seeded, or stale beyond `STALE_BOOK_S`) is silently dropped from the
    result dict. The caller (RouterBookView → run.py) treats omitted tickers
    as "missing" and falls back to REST via the `ws_missed` path at
    `run.py:428-451`.

    This is the safety contract: even total manager failure degrades to the
    existing REST fallback without intervention.
    """

    def __init__(self, manager) -> None:
        self._mgr = manager

    async def fetch_all(self, tickers: list[str]) -> dict:
        result: dict = {}
        if self._mgr is None:
            return result
        for t in tickers:
            try:
                book = self._mgr.get_orderbook_fp(t)
            except Exception as e:
                log.warning(f"[ws_delta] fetch error for {t}: {type(e).__name__}: {e}")
                continue
            if book is not None:
                result[t] = book
        return result

    async def close(self) -> None:
        if self._mgr is not None:
            try:
                await self._mgr.stop()
            except Exception:
                pass


# Phase-2 flip gate. While this file exists, book reads route to the isolated
# WS book; remove it and reads revert to the old delta book within one tick.
# Checked live (cached ~1s) so the flip needs NO run.py restart.
ISOLATED_FLAG_PATH = "use_isolated_ws.flag"
ISOLATED_FLAG_CACHE_S = 1.0
# How often the partial-coverage merge may log (it would otherwise fire per tick).
MERGE_LOG_INTERVAL_S = 60.0
# Throttle for the dual-miss log (both books missed). Separate knob: this
# one names the tickers actually driving REST fallback, so it earns its own
# cadence independent of the merge-success log.
DUAL_MISS_LOG_INTERVAL_S = 30.0


class IsolatedBookView(BookView):
    """Live switch between the isolated WS book (new) and a fallback view (old).

    Phase 2 of WS_MIGRATION_PLAN.md. Routes `fetch_all` to
    `IsolatedWSBook.get_orderbook_fp` while `use_isolated_ws.flag` exists,
    otherwise delegates to `fallback` (the existing `WSDeltaBookView`).

    Safety contract — identical to `WSDeltaBookView`, plus two guards:
      * Flag absent, provider returns None, or the isolated reader is not
        running  → delegate to `fallback`. Never silently blind the arber.
      * Isolated book serves ZERO tickers while some were requested → delegate
        to `fallback` (a dead reader thread degrades to the old book, not to an
        empty result).
      * Any per-ticker exception or None book → that ticker is omitted, which
        run.py already treats as "missing" and covers via REST fallback.

    `close()` deliberately does NOT stop the isolated book: its lifecycle is
    owned by the shadow supervisor in run.py, which starts/stops it on
    `enable_isolated_ws.flag`. Closing it here would fight that owner.
    """

    def __init__(
        self,
        book_provider: Callable[[], object],
        fallback: "BookView | None",
        flag_path: str = ISOLATED_FLAG_PATH,
        cache_s: float = ISOLATED_FLAG_CACHE_S,
    ) -> None:
        self._provider = book_provider   # callable → IsolatedWSBook | None (resolved late)
        self._fallback = fallback
        self._flag_path = flag_path
        self._cache_s = cache_s
        self._flag_ts = 0.0
        self._flag_on = False
        self._last_path_logged: Optional[str] = None
        # Throttle for the partial-coverage merge log (fires every tick otherwise).
        self._last_merge_log: float = 0.0
        self._last_dual_miss_log: float = 0.0

    def _flag_set(self) -> bool:
        now = time.time()
        if now - self._flag_ts >= self._cache_s:
            self._flag_ts = now
            try:
                self._flag_on = os.path.exists(self._flag_path)
            except Exception:
                self._flag_on = False
        return self._flag_on

    def _resolve_book(self):
        """Return the isolated book iff the flag is set AND it is running."""
        if not self._flag_set():
            return None
        try:
            book = self._provider()
        except Exception:
            return None
        if book is None:
            return None
        try:
            if not book.is_running():
                return None
        except Exception:
            return None
        return book

    async def _delegate(self, tickers: list[str]) -> dict:
        if self._fallback is None:
            return {}
        return await self._fallback.fetch_all(tickers)

    async def fetch_all(self, tickers: list[str]) -> dict:
        flag_on = self._flag_set()
        book = self._resolve_book()

        # Log only on transition, so the active read path is greppable without
        # spamming run.log every tick. The middle case matters: the flip needs
        # BOTH flags — `enable_isolated_ws.flag` (supervisor starts the reader)
        # and `use_isolated_ws.flag` (this switch). Flipping only the latter
        # silently keeps the old book, so say so loudly.
        state = "ISOLATED" if book is not None else ("FLAG_NO_READER" if flag_on else "OLD")
        if state != self._last_path_logged:
            self._last_path_logged = state
            if state == "ISOLATED":
                log.warning("[ISOLATED WS] READ PATH -> ISOLATED (new book) — arber is now trading on it")
            elif state == "FLAG_NO_READER":
                log.warning(
                    "[ISOLATED WS] READ PATH -> ws_delta (old book): use_isolated_ws.flag is SET "
                    "but the isolated reader is not running — `touch enable_isolated_ws.flag` "
                    "to start it. NO flip has occurred."
                )
            else:
                log.warning("[ISOLATED WS] READ PATH -> ws_delta (old book)")

        if book is None:
            return await self._delegate(tickers)

        result: dict = {}
        for t in tickers:
            try:
                b = book.get_orderbook_fp(t)
            except Exception as e:
                log.warning(f"[ISOLATED WS] fetch error for {t}: {type(e).__name__}: {e}")
                continue
            if b is not None:
                result[t] = b

        if tickers and not result:
            log.warning(
                "[ISOLATED WS] served 0/%d tickers — falling back to ws_delta this tick",
                len(tickers),
            )
            return await self._delegate(tickers)

        # ── PARTIAL-COVERAGE MERGE (2026-07-27) ───────────────────────────
        # Previously this returned `result` as-is, so the all-or-nothing guard
        # above only engaged when the isolated book served ZERO tickers. Serve
        # 3 of 22 and the other 19 skipped ws_delta entirely and fell out of
        # `ob_results` in run.py → one REST round-trip EACH, every tick.
        #
        # Measured 2026-07-27 over ~14h: "served 0/N" fired 342 times while
        # [WS FALLBACK] fired 49,472 — i.e. 99.3% of the REST fallback load was
        # tickers the isolated book merely MISSED, for which the in-memory
        # ws_delta book was never even consulted.
        #
        # ws_delta is a local dict read: asking it costs ~nothing and can only
        # add coverage. Anything still missing after this falls through to the
        # caller's REST path exactly as before, so this strictly reduces REST.
        missing = [t for t in tickers if t not in result]
        if missing:
            try:
                fb = await self._delegate(missing)
                recovered = 0
                for t, b in (fb or {}).items():
                    if b is not None and t not in result:
                        result[t] = b
                        recovered += 1
                if recovered:
                    now = time.time()
                    if now - self._last_merge_log >= MERGE_LOG_INTERVAL_S:
                        self._last_merge_log = now
                        log.info(
                            "[ISOLATED WS] partial coverage: served %d/%d, "
                            "ws_delta recovered %d more — %d still need REST",
                            len(result) - recovered, len(tickers), recovered,
                            len(tickers) - len(result),
                        )
            except Exception as e:
                # Never let the merge break a working read path.
                log.debug(f"[ISOLATED WS] partial-merge delegate failed: {e}")

            # ── DUAL MISS (2026-07-28) ────────────────────────────────────
            # The `if recovered:` log above is silent in the case that
            # actually drives REST load: BOTH books missed, so recovered==0
            # and nothing was ever logged. Since both apply identical guards
            # with an identical STALE_BOOK_S, they tend to fail together —
            # which is precisely the case worth naming. Log the per-book
            # reason so the fallback is attributable instead of inferred.
            still = [t for t in tickers if t not in result]
            if still:
                now = time.time()
                if now - self._last_dual_miss_log >= DUAL_MISS_LOG_INTERVAL_S:
                    self._last_dual_miss_log = now
                    detail = " | ".join(f"{t} [{ws_miss_stats.describe(t)}]"
                                        for t in sorted(still)[:6])
                    log.warning(
                        "[WS DUAL MISS] %d/%d tickers served by NEITHER isolated "
                        "nor ws_delta → REST: %s",
                        len(still), len(tickers), detail,
                    )
        return result

    async def close(self) -> None:
        # Close the fallback only — the isolated book is owned by the supervisor.
        if self._fallback is not None:
            try:
                await self._fallback.close()
            except Exception:
                pass


class RouterBookView(BookView):
    """Per-ticker dispatch — looks up each ticker's book_source via the
    supplied registry (typically `MarketManager.book_source_by_ticker`)
    and routes to either WS, REST, or (optionally) the delta-maintained WS
    view. Tickers with no explicit routing use `default_source`.

    The `ws_delta_tickers` allowlist is checked BEFORE per-row routing — any
    ticker in that set goes to `ws_delta_view`, regardless of its row config.
    This is the env-var-controlled override that `run.py` uses to opt
    specific tickers (e.g. a single LoL match) into the new path without
    touching `market_parameters.csv`.

    All three backends fire in parallel via asyncio.gather so they can't
    slow each other down.
    """

    def __init__(
        self,
        ws_view: "BookView",
        rest_view: "BookView",
        source_by_ticker: dict,  # ticker → "ws" | "rest"
        default_source: str = "ws",
        ws_delta_view: "BookView | None" = None,
        ws_delta_tickers: set | None = None,
    ) -> None:
        self._ws = ws_view
        self._rest = rest_view
        self._ws_delta = ws_delta_view
        self._ws_delta_tickers = set(ws_delta_tickers or [])
        self._routes = source_by_ticker
        self._default = default_source

    async def fetch_all(self, tickers: list[str]) -> dict:
        import asyncio
        ws_delta_tickers: list[str] = []
        ws_tickers: list[str] = []
        rest_tickers: list[str] = []
        # Dispatch order:
        #   src=="rest" → rest_view (per-row CSV override)
        #   ws_delta_view is set AND ticker not in denylist → ws_delta_view
        #   otherwise → ws_view (snapshot-per-fetch, original behavior)
        # The denylist (_ws_delta_tickers when populated via WS_DELTA_TICKERS
        # env var) lets us SURGICALLY route specific tickers through ws_delta
        # while leaving the rest on ws. When env var is unset, _ws_delta_tickers
        # is empty and we route ALL non-rest tickers through ws_delta — which
        # is the "all games on ws_delta" default after 2026-06-13.
        for t in tickers:
            src = (self._routes.get(t) or self._default).strip().lower()
            if src == "rest":
                rest_tickers.append(t)
            elif self._ws_delta is not None and (
                not self._ws_delta_tickers or t in self._ws_delta_tickers
            ):
                ws_delta_tickers.append(t)
            else:
                ws_tickers.append(t)

        async def _ws_call():
            if not ws_tickers:
                return {}
            return await self._ws.fetch_all(ws_tickers)

        async def _rest_call():
            if not rest_tickers:
                return {}
            return await self._rest.fetch_all(rest_tickers)

        async def _ws_delta_call():
            if not ws_delta_tickers or self._ws_delta is None:
                return {}
            try:
                return await self._ws_delta.fetch_all(ws_delta_tickers)
            except Exception as e:
                log.warning(f"[ws_delta] fetch_all crashed: {type(e).__name__}: {e}; falling through to REST")
                return {}

        ws_result, rest_result, ws_delta_result = await asyncio.gather(
            _ws_call(), _rest_call(), _ws_delta_call()
        )
        result: dict = {}
        result.update(ws_result)
        result.update(rest_result)
        result.update(ws_delta_result)  # ws_delta wins for its allowlisted tickers
        return result

    async def close(self) -> None:
        await self._ws.close()
        await self._rest.close()
        if self._ws_delta is not None:
            await self._ws_delta.close()


class RestBookView(BookView):
    """Reads books from the REST-polled OrderbookRegistry.

    Latency profile:
      - Best case (1s poll cadence, fresh): ~500ms mean staleness
      - Worst case (poller backed up): up to REST_MAX_AGE_SEC, then omit

    Tickers not in the registry, or with snapshots older than
    REST_MAX_AGE_SEC, are omitted from the result. Callers should treat
    omitted entries the same as WS snapshot timeouts (the run.py fallback
    path at lines ~378-403 handles this gracefully).
    """

    def __init__(self, registry, max_age_sec: float = REST_MAX_AGE_SEC) -> None:
        self._registry = registry
        self._max_age = max_age_sec

    async def fetch_all(self, tickers: list[str]) -> dict:
        result: dict = {}
        n_missing = 0
        n_stale = 0
        for t in tickers:
            book = self._registry.get(t)
            if book is None:
                n_missing += 1
                continue
            if book.age_sec > self._max_age:
                n_stale += 1
                continue
            result[t] = book.to_ws_style()
        if n_missing or n_stale:
            log.debug(
                f"[REST BookView] fetch_all({len(tickers)} tickers): "
                f"served={len(result)} missing={n_missing} stale={n_stale}"
            )
        return result

    async def close(self) -> None:
        return None
