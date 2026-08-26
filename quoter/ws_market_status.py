"""Probe Kalshi WS to classify market status in one batched call.

Replaces per-event `/markets?event_ticker=…` REST polling in
populate_configs.py. Single auth'd WS connection, one batched `subscribe`,
classify each ticker by Kalshi's response:

  * `orderbook_snapshot` for the ticker → market is **active**
    (we also harvest the orderbook in the same step, so callers get top-of-book
     "for free")
  * `error code:28 "Markets not found"` listing the ticker → market is
    **finalized / settled / unlisted**
  * neither arrives within `timeout_s` → **unknown**

Why this exists
---------------
The `/markets?event_ticker=…` endpoint has a dedicated ~3-5 token/sec bucket
(see `memory/reference_kalshi_markets_shared_bucket.md`). populate_configs
fires hundreds of these per cycle and triggers ~30× 429 retries per run. The
WS protocol gives us the same status information for free over a connection
we already need for other reasons, and the snapshot payload contains the
top-of-book levels that satisfy the bid≥5c gate in `_evaluate_leg`.

Usage
-----
    from ws_market_status import probe_markets
    status_by_ticker = probe_markets(
        ["KXVALORANTGAME-26JUN232100LYONBBB-LYON",
         "KXCS2GAME-26JUN240500FOKECH-FOK"],
        timeout_s=3.0,
    )
    for t, s in status_by_ticker.items():
        if s.active is True:
            print(t, "active", s.yes_top_bid_c, s.no_top_bid_c)
        elif s.active is False:
            print(t, "finalized")
        else:
            print(t, "UNKNOWN (timeout)")

Standalone validation
---------------------
    python ws_market_status.py probe \
        KXVALORANTGAME-26JUN232100LYONBBB-LYON \
        KXCS2GAME-26JUN240500FOKECH-FOK
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import websockets

sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
from dotenv import load_dotenv  # noqa: E402
load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
from kalshi_auth import KalshiAuth  # noqa: E402

import ticker_aliases  # noqa: E402

logger = logging.getLogger("ws_market_status")

WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_SIGNING_PATH = "/trade-api/ws/v2"
SUBSCRIBE_CHUNK = 200  # Kalshi accepts up to this many tickers per subscribe


@dataclass
class MarketStatus:
    """Per-ticker classification + (when active) the seed orderbook.

    active values:
      True   — orderbook_snapshot was received (market is live/listed)
      False  — error code:28 "Markets not found" listed this ticker
      None   — neither response observed within timeout_s (caller should
               treat as "unknown" and fall back to REST)
    """
    active: Optional[bool]
    yes_top_bid_c: Optional[int] = None
    no_top_bid_c: Optional[int] = None
    # Sorted high-to-low; (price_cents, qty_fp_hundredths)
    yes_levels: list = field(default_factory=list)
    no_levels: list = field(default_factory=list)


def _price_to_cents(s) -> int:
    return int(round(float(s) * 100))


def _qty_to_fp(s) -> int:
    return int(round(float(s) * 100))


async def _probe_async(
    tickers: list[str],
    timeout_s: float = 3.0,
    auth: Optional[KalshiAuth] = None,
) -> dict[str, MarketStatus]:
    if not tickers:
        return {}
    if auth is None:
        auth = KalshiAuth(
            os.getenv("KALSHI_API_KEY_ID"),
            os.getenv("KALSHI_PRIVATE_KEY_PATH"),
        )

    # Ticker aliases: bot uses synthetic names internally, Kalshi wire uses
    # real names. Translate for the subscribe, reverse-translate on receipt.
    wire_to_synth: dict[str, str] = {}
    wire_tickers: list[str] = []
    for t in tickers:
        wire_t = ticker_aliases.resolve(t)
        wire_to_synth[wire_t] = t
        wire_tickers.append(wire_t)

    # Initialize all as unknown; classified entries overwrite as messages arrive.
    result: dict[str, MarketStatus] = {t: MarketStatus(active=None) for t in tickers}
    unclassified: set[str] = set(tickers)

    headers = auth.get_headers("GET", WS_SIGNING_PATH)
    async with websockets.connect(
        WS_URL,
        additional_headers=headers,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=10,
        close_timeout=2,
    ) as ws:
        sub_id = 0
        for i in range(0, len(wire_tickers), SUBSCRIBE_CHUNK):
            chunk = wire_tickers[i:i + SUBSCRIBE_CHUNK]
            sub_id += 1
            await ws.send(json.dumps({
                "id": sub_id,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": chunk,
                },
            }))

        deadline = time.time() + timeout_s
        while unclassified and time.time() < deadline:
            remaining = max(deadline - time.time(), 0.1)
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type") or ""
            payload = msg.get("msg") or {}

            if mtype == "orderbook_snapshot":
                wire_t = payload.get("market_ticker") or ""
                synth_t = wire_to_synth.get(wire_t, wire_t)
                if synth_t not in result:
                    continue
                yes_pairs = payload.get("yes_dollars_fp") or []
                no_pairs = payload.get("no_dollars_fp") or []
                yes_levels = []
                for pair in yes_pairs:
                    if not pair or len(pair) < 2:
                        continue
                    try:
                        yes_levels.append((_price_to_cents(pair[0]), _qty_to_fp(pair[1])))
                    except (ValueError, TypeError):
                        continue
                no_levels = []
                for pair in no_pairs:
                    if not pair or len(pair) < 2:
                        continue
                    try:
                        no_levels.append((_price_to_cents(pair[0]), _qty_to_fp(pair[1])))
                    except (ValueError, TypeError):
                        continue
                yes_levels.sort(key=lambda kv: -kv[0])
                no_levels.sort(key=lambda kv: -kv[0])
                result[synth_t] = MarketStatus(
                    active=True,
                    yes_top_bid_c=yes_levels[0][0] if yes_levels else None,
                    no_top_bid_c=no_levels[0][0] if no_levels else None,
                    yes_levels=yes_levels,
                    no_levels=no_levels,
                )
                unclassified.discard(synth_t)
            elif mtype == "error":
                # Kalshi error payload for an unknown/finalized ticker:
                #   {"type":"error","msg":{"code":28,
                #     "msg":"Markets not found",
                #     "market_tickers":["KX..."]}}
                # Code 28 is the definitive finalized/unlisted signal.
                code = payload.get("code")
                if code == 28:
                    for wire_t in (payload.get("market_tickers") or []):
                        synth_t = wire_to_synth.get(wire_t, wire_t)
                        if synth_t in unclassified:
                            result[synth_t] = MarketStatus(active=False)
                            unclassified.discard(synth_t)

    return result


def probe_markets(
    tickers: list[str],
    timeout_s: float = 3.0,
) -> dict[str, MarketStatus]:
    """Sync wrapper around `_probe_async`. Opens a one-shot WS, classifies
    every ticker, returns a dict. For SYNCHRONOUS callers only (populate_configs,
    refresh_tomorrow_markets). Async callers in run.py must use
    `probe_markets_async` instead — `asyncio.run()` raises
    `RuntimeError: cannot be called from a running event loop` when invoked
    from inside an already-running loop."""
    return asyncio.run(_probe_async(tickers, timeout_s))


async def probe_markets_async(
    tickers: list[str],
    timeout_s: float = 3.0,
) -> dict[str, MarketStatus]:
    """Async-native entrypoint for callers already inside an event loop
    (run.py's main loop). Same semantics as `probe_markets` — opens a
    one-shot WS subscribe, classifies every ticker, returns the dict.

    The sync `probe_markets` wraps this with `asyncio.run()`, which fails
    when called from inside a running loop. Use this version from any
    `async def` context."""
    return await _probe_async(tickers, timeout_s)


# ──────────────────────────────────────────────────────────────────────────
# CLI: `python ws_market_status.py probe TICKER1 TICKER2 …`
# ──────────────────────────────────────────────────────────────────────────

def _cli_probe(tickers: list[str], timeout_s: float) -> None:
    t0 = time.time()
    out = probe_markets(tickers, timeout_s=timeout_s)
    elapsed = time.time() - t0
    print(f"\nProbed {len(tickers)} tickers in {elapsed:.2f}s "
          f"(timeout_s={timeout_s})\n")
    name_w = max(len(t) for t in tickers) + 2
    print(f"  {'TICKER':<{name_w}}  STATUS      YES_BID  NO_BID")
    print(f"  {'-'*name_w}  ----------  -------  ------")
    for t in tickers:
        s = out[t]
        if s.active is True:
            label = "ACTIVE"
            yb = f"{s.yes_top_bid_c:>3}c" if s.yes_top_bid_c is not None else "  -"
            nb = f"{s.no_top_bid_c:>3}c" if s.no_top_bid_c is not None else "  -"
        elif s.active is False:
            label = "FINALIZED"
            yb = nb = "    -"
        else:
            label = "UNKNOWN"
            yb = nb = "    -"
        print(f"  {t:<{name_w}}  {label:<10}  {yb:>5}    {nb:>5}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("probe")
    pp.add_argument("tickers", nargs="+")
    pp.add_argument("--timeout", type=float, default=3.0)
    args = p.parse_args()
    if args.cmd == "probe":
        _cli_probe(args.tickers, args.timeout)
