"""Track per-ticker trade activity via Kalshi WebSocket; classify markets
as active/inactive based on time since last trade.

Rule:
  - Pre-game (now < scheduled_start):       always active
  - Post-game (now >= scheduled_start):
    - last trade < IDLE_THRESHOLD_SEC ago:   active
    - otherwise:                             inactive

The motivation: if no trades for 10+ min on a market that should be live,
the game probably isn't being streamed (insider-info risk) or has been
cancelled/postponed. Either way we don't want to be quoting it.

Architecture:
  - One background thread subscribes to Kalshi WS `trade` channel for
    the tickers passed in via `subscribe(tickers)`.
  - Trade messages update an in-memory `{ticker: last_trade_unix_ts}` dict.
  - `is_market_active(ticker, scheduled_start)` is the single decision point.

Threadsafe: the registry dict is lock-protected; readers can call from any
thread (e.g. populate_configs running in its own daemon).

Wire into populate_configs ONLY after tests pass.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"

# Time since last trade after which we declare a post-game-start ticker inactive.
# 10 min: long enough to absorb between-map gaps in BO3/BO5 series; short enough
# to catch genuinely-dead markets within one populate_configs daemon cycle.
IDLE_THRESHOLD_SEC = 600

# How long after scheduled start before we begin enforcing the activity check.
# Esports tournaments routinely run 5-15 min late; this prevents false-positive
# "going inactive" on games that haven't actually kicked off yet.
GRACE_AFTER_SCHEDULED_START_SEC = 900  # 15 min

# Maximum age of the WS feed since the last received message before we declare
# the tracker unhealthy. When unhealthy, is_market_active fails open to True
# so a broken WS connection cannot cause every ticker to be marked inactive.
TRACKER_HEALTH_THRESHOLD_SEC = 120


# ─── Helpers ────────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Match the timestamp slug in Kalshi tickers. Examples:
#   KXLOLGAME-26JUN101400KCBHRTS-KCB    → 26JUN10 1400 (June 10 14:00 UTC)
#   KXCS2MAP-26JUN081030BETMNTE-2-BET   → 26JUN08 1030 (June 8 10:30 UTC)
#   KXVALORANTGAME-26JUN081000FUTVIT-VIT → 26JUN08 1000
# Pattern: <2-digit year><3-letter month><2-digit day><4-digit HHMM>
_TS_RE = re.compile(r"(\d{2})([A-Z]{3})(\d{2})(\d{4})")


def parse_scheduled_start(ticker: str) -> Optional[float]:
    """Extract scheduled start unix timestamp from a Kalshi ticker.

    Kalshi esports tickers (KXCS2GAME / KXLOLGAME / KXVALORANTGAME / KXDOTA2GAME)
    embed `YYMMMDDhhmm` in US Eastern time, NOT UTC. Confirmed via Poly
    startTime cross-check on 2026-06-14. Parsing as UTC made games appear
    4-5 hours earlier than reality — so `is_market_active` flipped to False
    16 min after the (wrong) parsed start time and suppressed WS subscriptions
    for legitimately-upcoming events (XLGLEV / FUTVIT on 2026-06-16).

    Returns None if the ticker doesn't contain a parseable timestamp.
    """
    m = _TS_RE.search(ticker)
    if not m:
        return None
    yy, mmm, dd, hhmm = m.groups()
    month = _MONTH_MAP.get(mmm)
    if not month:
        return None
    try:
        # Esports ticker time is ET. Use zoneinfo for DST-correct conversion.
        from zoneinfo import ZoneInfo
        dt = datetime.datetime(
            year=2000 + int(yy),
            month=month,
            day=int(dd),
            hour=int(hhmm[:2]),
            minute=int(hhmm[2:]),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


# ─── Core class ─────────────────────────────────────────────────────────────


class TradeActivityTracker:
    """Tracks the last-trade timestamp per ticker via Kalshi WS subscription.

    Background thread maintains the WS connection; trade messages update
    an in-memory registry. Decision method `is_market_active` is read-only
    and threadsafe — call it from any daemon (populate_configs, etc.).
    """

    def __init__(self, auth):
        self._auth = auth
        self._last_trade: dict[str, float] = {}
        self._subscribed: set[str] = set()
        self._lock = threading.Lock()
        self._ws = None
        self._sub_id = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        # WS health: timestamp of the last received message of any kind.
        # If this is stale (> TRACKER_HEALTH_THRESHOLD_SEC ago), is_market_active
        # fails open to True so a broken feed doesn't unload every bot.
        self._last_message_ts: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin background WS listening. No-op if already running."""
        if self._running:
            return
        self._running = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="trade-activity-tracker"
        )
        self._thread.start()
        log.info("[ACTIVE SCANNER] started")

    def stop(self) -> None:
        """Signal background thread to exit and wait briefly."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        log.info("[ACTIVE SCANNER] stopped")

    def subscribe(self, tickers: list[str]) -> None:
        """Mark these tickers for WS trade-channel subscription. Idempotent —
        already-subscribed tickers are not duplicated.

        Note: this only records the intent; the WS thread reconciles its
        actual subscription on next reconnect or via an explicit re-sub.
        """
        with self._lock:
            self._subscribed.update(tickers)

    def record_trade(self, ticker: str, ts_unix: Optional[float] = None) -> None:
        """Update last-trade timestamp for a ticker. Test hook + WS callback.

        ts_unix: epoch seconds. None = "now."
        """
        if ts_unix is None:
            ts_unix = time.time()
        with self._lock:
            self._last_trade[ticker] = ts_unix

    def last_trade_ts(self, ticker: str) -> Optional[float]:
        with self._lock:
            return self._last_trade.get(ticker)

    def is_healthy(self, now: Optional[float] = None) -> bool:
        """True iff the WS feed has produced any message within the last
        TRACKER_HEALTH_THRESHOLD_SEC. Used by is_market_active to fail open
        when the feed is broken — we never want a dead WS to cause every
        ticker to be marked inactive.
        """
        if now is None:
            now = time.time()
        with self._lock:
            last = self._last_message_ts
        if last is None:
            return False
        return (now - last) < TRACKER_HEALTH_THRESHOLD_SEC

    def mark_ws_alive(self, ts: Optional[float] = None) -> None:
        """Test hook + WS callback to record that the feed is producing
        messages. Called for EVERY received message (not just trades) so a
        steady stream of orderbook deltas keeps the tracker healthy even
        on quiet markets.
        """
        if ts is None:
            ts = time.time()
        with self._lock:
            self._last_message_ts = ts

    def is_market_active(
        self,
        ticker: str,
        scheduled_start_unix: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Decide whether `ticker` should remain in active config.

        - scheduled_start_unix: parsed from ticker if None.
        - now: clock override for testing; uses real time if None.

        Rules:
          1. Pre-game (now < scheduled_start):              True
          2. Grace window (within GRACE_AFTER_SCHEDULED_START_SEC of start): True
          3. No trade since tracking began + grace expired: False
          4. Last trade > IDLE_THRESHOLD_SEC ago:           False
          5. Otherwise:                                     True
        """
        if now is None:
            now = time.time()
        if scheduled_start_unix is None:
            scheduled_start_unix = parse_scheduled_start(ticker)
            if scheduled_start_unix is None:
                # Can't determine schedule — conservative: keep active
                return True

        # Rule 1: pre-game
        if now < scheduled_start_unix:
            return True

        # Rule 2: grace window
        if (now - scheduled_start_unix) < GRACE_AFTER_SCHEDULED_START_SEC:
            return True

        # Rule 2.5 (fail-safe): if the WS feed is broken / not yet warmed up,
        # we don't trust our trade-activity data. Keep market active so a
        # silent feed failure can't cascade into unloading every bot.
        if not self.is_healthy(now=now):
            return True

        # Rule 3 / 4 / 5: trade-activity check
        last = self.last_trade_ts(ticker)
        if last is None:
            return False
        return (now - last) < IDLE_THRESHOLD_SEC

    # ── WebSocket loop (skipped in unit tests via dependency injection) ───

    def _run_loop(self) -> None:
        """Maintain WS connection; on failure, reconnect after backoff."""
        import asyncio
        while not self._stop.is_set():
            try:
                asyncio.run(self._listen_async())
            except Exception as e:
                log.warning(f"[ACTIVE SCANNER] WS loop exception: {e} — backoff 5s")
                self._stop.wait(5.0)

    async def _listen_async(self) -> None:
        import websockets
        headers = self._auth.get_headers("GET", WS_PATH)
        async with websockets.connect(
            WS_URL,
            additional_headers=headers,
            ping_interval=30,
            open_timeout=5,
            close_timeout=2,
        ) as ws:
            self._ws = ws
            # Subscribe to all currently-known tickers
            with self._lock:
                tickers = sorted(self._subscribed)
            if tickers:
                self._sub_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "id": self._sub_id,
                            "cmd": "subscribe",
                            "params": {
                                "channels": ["trade"],
                                "market_tickers": tickers,
                            },
                        }
                    )
                )
                log.info(f"[ACTIVE SCANNER] subscribed to {len(tickers)} ticker trade feed(s)")

            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                # Every received message is a health signal — keeps the
                # tracker considered healthy as long as the feed is alive.
                self.mark_ws_alive()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") != "trade":
                    continue
                payload = msg.get("msg") or {}
                ticker = payload.get("market_ticker") or payload.get("ticker") or ""
                if not ticker:
                    continue
                # Kalshi trade msg may carry `created_time` (ISO) or epoch
                ts = time.time()
                self.record_trade(ticker, ts)


import asyncio  # noqa: E402  (used in _run_loop wrapper above)
