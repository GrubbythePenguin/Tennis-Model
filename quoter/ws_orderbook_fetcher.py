"""
WebSocket Orderbook Fetcher

Replaces REST orderbook polling with WS snapshot re-subscription.
Each fetch cycle: subscribe → collect snapshots → unsubscribe.
The initial snapshot is always accurate (confirmed by testing).

Returns data in the same format as the REST API so existing
orderbook processing code works without changes.
"""

import asyncio
import json
import logging
import time

log = logging.getLogger(__name__)

# 2026-06-08: dedicated external WS host (per Kalshi Discord). The blip
# diagnostic showed snapshot alternation on the legacy CloudFront host
# (api.elections.kalshi.com) — almost certainly from fresh connections
# landing on different cached backends each cycle. The dedicated host
# should give us consistent routing.
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"
SNAPSHOT_TIMEOUT = 1.5  # Max seconds to wait for all snapshots (typically ~550ms for 70 tickers)


class WSOrderbookFetcher:
    def __init__(self, auth):
        self.auth = auth
        self._ws = None
        self._sub_id = 0

    async def _fresh_connect(self):
        """Create a fresh WS connection. Always closes any existing one first."""
        if self._ws is not None:
            try:
                await asyncio.wait_for(self._ws.close(), timeout=1.0)
            except Exception:
                pass
            self._ws = None

        import websockets
        headers = self.auth.get_headers("GET", WS_PATH)
        self._ws = await websockets.connect(
            WS_URL, additional_headers=headers, ping_interval=30,
            open_timeout=2, close_timeout=1,
        )
        log.debug("[WS OB] Connected")

    async def fetch_all(self, tickers: list[str]) -> dict:
        """Fetch orderbooks for all tickers via WS snapshots.

        Fresh connection every call to avoid stale data.
        Sends all tickers (up to 100) in a single subscribe.

        Returns dict matching REST format:
        {ticker: {"orderbook_fp": {"yes_dollars": [...], "no_dollars": [...]}}}
        """
        if not tickers:
            return {}

        tickers_set = set(tickers)

        try:
            await self._fresh_connect()
        except Exception as e:
            log.error(f"[WS OB] Connection failed: {e}")
            return {}

        # Subscribe all tickers in one message
        self._sub_id += 1
        sub_msg = {
            "id": self._sub_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }

        try:
            await self._ws.send(json.dumps(sub_msg))
        except Exception as e:
            log.error(f"[WS OB] Subscribe send failed: {e}")
            self._ws = None
            return {}

        # Collect snapshots only — deltas are unreliable
        results = {}
        deadline = time.time() + SNAPSHOT_TIMEOUT
        received = set()

        while time.time() < deadline and len(received) < len(tickers):
            try:
                raw = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=max(0.1, deadline - time.time()),
                )
            except (asyncio.TimeoutError, Exception):
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            payload = msg.get("msg", {})

            if msg_type == "orderbook_snapshot":
                ticker = payload.get("market_ticker", "")
                if ticker and ticker in tickers_set:
                    results[ticker] = {
                        "orderbook_fp": {
                            "yes_dollars": payload.get("yes_dollars_fp", []),
                            "no_dollars": payload.get("no_dollars_fp", []),
                        }
                    }
                    received.add(ticker)

        # Close connection — fresh connect next call to avoid stale data
        try:
            await asyncio.wait_for(self._ws.close(), timeout=1.0)
        except Exception:
            pass
        self._ws = None

        missed = len(tickers) - len(received)
        if missed > 0:
            log.debug(f"[WS OB] Missed {missed}/{len(tickers)} snapshots")

        return results

    async def close(self):
        if self._ws:
            try:
                await asyncio.wait_for(self._ws.close(), timeout=1.0)
            except Exception:
                pass
            self._ws = None
