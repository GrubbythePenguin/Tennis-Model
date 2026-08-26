"""Dedicated WS delta manager for esports MAP tickers.

Runs as its OWN long-lived connection, isolated from the series
`ws_delta_book.WSDeltaManager` instance, so map-market volume, reconnects, or a
quiet/settled map book can never starve or disturb the series feed (and vice
versa). This is the fix for the map-book starvation: on the single shared
connection, ~50 map tickers seeded once then starved of deltas while series
stayed healthy; a dedicated connection maintains them perfectly — validated
byte-for-byte against REST on VCT C9G2 map 2 (2026-07-25: 58/58 top-of-book
matches, 0 reconnects). Reconstruction is the exact same `WSBook`
(snapshot + deltas) the series book uses; nothing about the protocol path
changes, only the connection it lives on.

Only behavioral difference from the base class: the per-ticker stale-book
watchdog is DISABLED. For a dedicated MAP connection an individual quiet/empty
book (a not-yet-started or already-finished map that gets a snapshot but no
deltas) must NOT trigger a full-connection force_reconnect — that would
blackout the ACTIVE map for the reconnect window. Per-ticker staleness is
already fail-safe WITHOUT a reconnect:
  * `get_orderbook_fp` returns None for a stale/empty book,
  * the run.py map path scrubs it (KALSHI MAP NO-FALLBACK) -> no trade.
True whole-connection death is still caught by the base class's connection-wide
stall watchdog (STALL_THRESHOLD_S of total silence -> close -> reconnect), and
the base reconnect-with-backoff loop still handles genuine drops.
"""
from __future__ import annotations

from ws_delta_book import WSDeltaManager


class MapWSDeltaManager(WSDeltaManager):
    async def _stale_book_watchdog(self) -> None:
        # Intentionally a no-op for the map feed (see module docstring). start()
        # still creates this task; returning immediately makes it inert without
        # changing the base class or the series manager that uses it.
        return
