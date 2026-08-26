"""Delta-maintained WS orderbook manager — STANDALONE, no production imports.

This module is NOT YET wired to run.py, manager.py, book_view.py, or any other
production code. It's a candidate replacement for `ws_orderbook_fetcher.py`
(WS-snapshot-per-fetch) AND `orderbook_poller.py` (REST orderbook polling).

The migration is staged in three steps that DO NOT need to happen on the same day:

  1. Build and validate this module standalone (this file + the validate-mode
     CLI at the bottom). Production keeps trading via REST while we test.
  2. Wire a new `WSDeltaBookView` into `book_view.py`. Add `book_source=ws_delta`
     to `framework_config.py` and one row in `market_parameters.csv` to opt in.
  3. Flip rows in `market_parameters.csv` from `ws` (default) or `rest` to
     `ws_delta` once validation is green for that risk class.

Until Step 2 happens, this module has zero effect on production behavior.

---

## Architecture

Single long-lived WebSocket connection subscribed to N tickers. Per-ticker
`WSBook` holds the current orderbook as `{price_cents → qty_fp_hundredths}`
dicts (integer arithmetic — no float drift after thousands of deltas).

### Seed pattern (race-free)

  1. Open WS, send subscribe — start buffering deltas (NOT applied to book).
  2. Kalshi sends `orderbook_snapshot` with a sequence number (`seq`).
  3. Apply snapshot atomically. Drain buffered deltas with `seq > snapshot_seq`
     (those post-date the snapshot). Discard buffered deltas with
     `seq <= snapshot_seq` (already reflected).

This is provably race-free because snapshot and deltas share the same WS wire
and the same `seq` number space (per-WS-connection monotonic, verified via
`_ws_dump.py` on 2026-06-12). REST seeding cannot achieve this because REST
responses have no `seq` field.

### Delta application

WS delta payload (verified shape — see `memory/reference_kalshi_ws_orderbook_delta.md`):

    {"type":"orderbook_delta", "seq":N,
     "msg":{"market_ticker":..., "price_dollars":"0.5200", "delta_fp":"-22.60", "side":"yes"}}

Apply as `new_qty = old_qty + delta_fp` (INCREMENTAL — not absolute). Convert
qty to integer fp-hundredths first so the running sum stays exact across tens
of thousands of deltas. Drop the price level if running sum ≤ 0.

### Failure modes covered

  - `orderbook_snapshot` arrives mid-stream → caught (resync any ticker).
  - Per-connection `seq` skip → CONN_SEQ_GAP event, mark all tickers stale,
    trigger re-snapshot via re-subscribe.
  - WS silent ≥ STALL_THRESHOLD_S → CONN_STALL event, close and reconnect.
  - WS connection drops → CONN_DROP, reconnect with exponential backoff
    (2s → 30s cap), re-subscribe all tickers.

### Output shape (production-compatible)

`get_orderbook_fp(ticker)` returns either `None` (book not ready or stale) or:

    {"orderbook_fp": {
        "yes_dollars": [["0.0100", "12345.00"], ...],   # high price first
        "no_dollars":  [["0.0500", "987.50"], ...],
    }}

Same shape `ws_orderbook_fetcher.py` and `orderbook_poller.RestBook.to_ws_style()`
emit today — so downstream code (`run.py:476-485`, arber_bot, quoter) needs
zero changes when we eventually swap.

---

## Usage (production integration — NOT YET WIRED)

    from ws_delta_book import WSDeltaManager
    from kalshi_auth import KalshiAuth

    auth = KalshiAuth(api_key_id, pem_path)
    mgr = WSDeltaManager(auth)
    await mgr.start()
    await mgr.set_tickers({"KXMLBGAME-26JUN121845SEAWSH-SEA", ...})

    # Quoter calls (from book_view.WSDeltaBookView.fetch_all):
    book = mgr.get_orderbook_fp("KXMLBGAME-26JUN121845SEAWSH-SEA")
    if book is None:
        # Not seeded yet, or stale > STALE_BOOK_S → fall back to REST
        ...

    # Lifecycle:
    await mgr.stop()

## Usage (standalone validation)

    python3 ws_delta_book.py validate \
      --tickers KXMLBGAME-26JUN121845SEAWSH-SEA,KXMLBGAME-26JUN121845SEAWSH-WSH \
      --duration 600 \
      --interval 30

Compares the delta-maintained book to authed REST snapshots (production code
path — same wire bytes as `kalshi_client.get_orderbook`). Same RACE/REST_LAG/
DRIFT_WS classifier as `temp_websocket_rest_comparison_v2.py`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import ws_miss_stats

import requests
import websockets

# Auth lives in the NHL tracker repo — same module the production bot uses.
# Importing from there does NOT touch any production trading code; KalshiAuth
# is a thin signing helper with no side effects.
sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
from dotenv import load_dotenv  # noqa: E402
load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
from kalshi_auth import KalshiAuth  # noqa: E402

# Ticker alias layer — translate synthetic↔real at the API boundary so
# tournament-final markets (listed on Kalshi as tournament-winner contracts,
# not per-series) can be traded as if they were normal series tickers.
# Pass-through for any non-aliased ticker — `resolve`/`reverse` return the
# input unchanged if the ticker isn't in ticker_aliases.csv.
import ticker_aliases  # noqa: E402

logger = logging.getLogger("ws_delta_book")

# ── Endpoints — match production exactly. ─────────────────────────────────────
# Verified 2026-06-13: `external-api.kalshi.com` is the host the bot uses
# (see memory/reference_kalshi_api_hosts_not_aliases.md). The deprecated
# `api.elections.kalshi.com` returns DIFFERENT orderbook data — never mix hosts.
REST_HOST = "https://external-api.kalshi.com"
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"

# ── Lag-proof journal (2026-08-02) — see WSDeltaManager.__init__ ──
_JOURNAL_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_FLAG = os.path.join(_JOURNAL_DIR, "ws_lag_journal.flag")
JOURNAL_PATH = os.path.join(_JOURNAL_DIR, "_ws_lag_journal_prod.csv")
JOURNAL_CAP_BYTES = 300 * 1024 * 1024   # hard stop: never quota-wedge the disk
JOURNAL_FLUSH_LINES = 200
WS_SIGNING_PATH = "/trade-api/ws/v2"
REST_OB_PATH = "/trade-api/v2/markets/{ticker}/orderbook"

# ── Tunables. ────────────────────────────────────────────────────────────────

# A book older than this is reported as None to callers (treated as "missing").
# Callers fall back to REST. Tuned 60s on 2026-06-24:
#
# - 30s (original) was too aggressive — heavy favorites parked at 99¢/1¢ get
#   no deltas for 30-60s naturally (no trading activity), triggering false
#   "stale" → per-tick REST fallback storm. Observed on LYONBBB.
# - 180s (interim) was too slow — when WS genuinely drops a single ticker
#   (rare but happens), 3 minutes of REST fallback per tick.
# - 60s + `_stale_book_watchdog` (below) is the right middle: a 60s gap
#   tolerates routine quiet-market silence, and the watchdog triggers a
#   FORCE-RECONNECT (not perpetual REST) so the book actually recovers
#   within ~5s of detection instead of waiting for periodic_reseed at 600s.
STALE_BOOK_S = 60.0

# ─── Stale-book FORCED-RECONNECT sweep (retuned 2026-07-27) ────────────────
# Deliberately decoupled from STALE_BOOK_S. That 60s value is the SERVE guard
# ("don't hand out a book this old") and must stay tight. These govern the far
# more destructive question: "is the whole connection wedged enough to justify
# dropping every book?" Quiet esports books trip 60s constantly, which is what
# made the sweep fire every cooldown window and storm REST.
STALE_RECONNECT_AGE_S = 600.0     # a book must be silent this long to count
STALE_RECONNECT_MIN_FRAC = 0.6    # ...and this share of live books must be stale
STALE_RECONNECT_MIN_N = 3         # ...with a floor, so tiny slates can't trip it

# WS silent for this long while connected → CONN_STALL, close, reconnect.
#
# Set high (180s) because: at pre-game quiet hours (4am UTC, all esports
# markets pre-game), there can genuinely be 30-60s gaps between deltas on
# our entire subscription basket — no one trades pre-game LCK at 4am.
# 30s threshold caused 5 stall→reconnect cycles in 5 minutes on 2026-06-13
# 04:12-17 UTC, which trapped the bot into ~95% REST-fallback because
# books kept getting marked stale during reconnect cycles.
#
# Real dead-connection detection is owned by `websockets.connect`'s
# ping_interval=20 + ping_timeout=20 — a truly dead conn gets caught
# within 40s at the library level (raises ConnectionClosed → triggers
# our reconnect-with-backoff path). This app-level watchdog is just a
# backstop for the unusual case where pings work but data doesn't.
STALL_THRESHOLD_S = 120.0

# Subscribe in chunks no larger than this. Kalshi caps subscribe arrays.
SUBSCRIBE_CHUNK = 200

# Reconnect backoff: 2s start, *1.5, cap 30s. Same shape as
# temp_websocket_rest_comparison_v2.py:262.
RECONNECT_BACKOFF_START_S = 2.0
RECONNECT_BACKOFF_CAP_S = 30.0
RECONNECT_BACKOFF_MULT = 1.5
# A connection that stayed up at least this long before dropping was healthy, so
# the next drop starts the backoff ladder from the bottom rather than inheriting
# a pinned 30s cap from an unrelated earlier storm.
HEALTHY_CONN_S = 60.0

# Periodic forced re-subscribe (anti-phantom-level hygiene). Setting this to
# >0 closes and reopens the WS every N minutes regardless of health; protects
# against accumulated state if some unknown bug causes phantom levels.
# Default 0 = disabled. Recommend 10 min (600s) for paranoid production use.
PERIODIC_RESEED_S = 0.0


def _price_to_cents(price_str) -> int:
    """'0.5200' → 52. Integer cents — no float drift."""
    return int(round(float(price_str) * 100))


def _qty_to_fp(qty_str) -> int:
    """'638955.90' → 63895590. Integer fp-hundredths so running sums stay exact
    after thousands of incremental ±delta_fp accumulations."""
    return int(round(float(qty_str) * 100))


def _fp_to_qty_str(fp: int) -> str:
    """Reverse of _qty_to_fp. Always 2-decimal string to match Kalshi REST."""
    return f"{fp / 100:.2f}"


def _cents_to_price_str(c: int) -> str:
    """Reverse of _price_to_cents. Always 4-decimal string."""
    return f"{c / 100:.4f}"


# ─── Per-ticker book ─────────────────────────────────────────────────────────


class WSBook:
    """One ticker's delta-maintained orderbook.

    State machine:
      BUFFERING — subscribed, waiting for first snapshot. Deltas accumulate
                  in `_pending` indexed by their `seq`.
      LIVE      — snapshot applied, deltas drained, live deltas apply directly.

    Storage: yes/no dicts of {price_cents: qty_fp_hundredths}. Integer
    arithmetic throughout — no float precision drift.
    """
    STATE_BUFFERING = "buffering"
    STATE_LIVE = "live"

    __slots__ = ("yes", "no", "state", "last_seq", "last_msg_ts",
                 "version", "_pending", "subscribed_ts")

    def __init__(self) -> None:
        self.yes: dict[int, int] = {}
        self.no: dict[int, int] = {}
        self.state: str = self.STATE_BUFFERING
        self.last_seq: Optional[int] = None
        self.last_msg_ts: float = 0.0
        self.version: int = 0
        self._pending: list[tuple[Optional[int], dict]] = []
        self.subscribed_ts: float = time.time()

    @property
    def is_live(self) -> bool:
        return self.state == self.STATE_LIVE

    def buffer_delta(self, payload: dict, seq: Optional[int]) -> None:
        self._pending.append((seq, payload))
        self.last_msg_ts = time.time()

    def apply_snapshot(self, payload: dict, snapshot_seq: Optional[int]) -> int:
        """Atomic re-seed. Wipes current state, writes snapshot, drains pending
        deltas with seq > snapshot_seq. Returns count of replayed deltas."""
        self.yes.clear()
        self.no.clear()
        for src_key, target in (("yes_dollars_fp", self.yes),
                                 ("no_dollars_fp", self.no)):
            for pair in payload.get(src_key, []) or []:
                if not pair or len(pair) < 2:
                    continue
                try:
                    p = _price_to_cents(pair[0])
                    q = _qty_to_fp(pair[1])
                except (ValueError, TypeError):
                    continue
                if q > 0:
                    target[p] = q
        replayed = 0
        last_applied_seq = snapshot_seq
        for s, payload_d in self._pending:
            # Replay only post-snapshot deltas. None seq is conservative-replay
            # (safer to double-apply than miss; incremental += is not
            # idempotent so this CAN cause off-by-N, but missing seq is
            # vanishingly rare in practice).
            if snapshot_seq is None or s is None or s > snapshot_seq:
                self._apply_one(payload_d)
                if s is not None:
                    last_applied_seq = s
                replayed += 1
        self._pending.clear()
        self.last_seq = last_applied_seq
        self.state = self.STATE_LIVE
        self.last_msg_ts = time.time()
        self.version += 1
        return replayed

    def apply_delta_live(self, payload: dict, seq: Optional[int]) -> None:
        """Apply a live delta (post-seed). Caller has already verified state==LIVE."""
        self._apply_one(payload)
        self.last_seq = seq
        self.last_msg_ts = time.time()
        self.version += 1

    def _apply_one(self, payload: dict) -> None:
        side = payload.get("side")
        if side not in ("yes", "no"):
            return
        try:
            p = _price_to_cents(payload["price_dollars"])
            d = _qty_to_fp(payload["delta_fp"])
        except (KeyError, ValueError, TypeError):
            return
        target = self.yes if side == "yes" else self.no
        new_fp = target.get(p, 0) + d
        if new_fp <= 0:
            target.pop(p, None)
        else:
            target[p] = new_fp

    def mark_stale(self) -> None:
        """Drop back to BUFFERING for a re-seed. Keeps existing book contents
        as a best-effort fallback in case callers ask before re-seed completes."""
        self.state = self.STATE_BUFFERING
        self._pending.clear()
        self.last_seq = None
        # Don't clear yes/no — keep the stale book around for STALE_BOOK_S so
        # callers see "stale" via age check rather than "missing". They can
        # decide to use or fall back.

    def to_orderbook_fp(self) -> dict:
        """Production-compatible shape: high price first per side, qty as
        2-decimal string. Matches the wire format `ws_orderbook_fetcher.py`
        emits today."""
        yes_sorted = sorted(self.yes.items(), key=lambda kv: -kv[0])
        no_sorted = sorted(self.no.items(), key=lambda kv: -kv[0])
        return {
            "orderbook_fp": {
                "yes_dollars": [[_cents_to_price_str(p), _fp_to_qty_str(q)]
                                 for p, q in yes_sorted],
                "no_dollars":  [[_cents_to_price_str(p), _fp_to_qty_str(q)]
                                 for p, q in no_sorted],
            }
        }


# ─── Long-lived WS manager ───────────────────────────────────────────────────


class WSDeltaManager:
    """Long-lived WS connection maintaining per-ticker delta books.

    Public surface used by future `WSDeltaBookView` (not yet built):

      await mgr.start()                            # spin up the WS task
      await mgr.set_tickers(set[str])              # subscribe/unsubscribe diff
      book = mgr.get_orderbook_fp(ticker)          # None if not-ready/stale
      books = mgr.get_orderbooks_fp(list[str])     # dict[ticker, book or None]
      stats = mgr.stats                            # event counters
      await mgr.stop()                             # graceful shutdown

    Thread-safety: not thread-safe — assumes asyncio single-thread access for
    `set_tickers` (lock is held during diff). `get_orderbook_fp` may be called
    from anywhere; it only reads dicts (atomic in CPython for int keys/values).
    """

    def __init__(self, auth: KalshiAuth,
                 stale_book_s: float = STALE_BOOK_S,
                 stall_threshold_s: float = STALL_THRESHOLD_S,
                 periodic_reseed_s: float = PERIODIC_RESEED_S) -> None:
        self.auth = auth
        self.stale_book_s = stale_book_s
        self.stall_threshold_s = stall_threshold_s
        self.periodic_reseed_s = periodic_reseed_s

        self.books: dict[str, WSBook] = {}
        self._target_tickers: set[str] = set()
        self._subscribed_tickers: set[str] = set()
        self._ws = None
        self._sub_cmd_id = 0
        self._sub_lock = asyncio.Lock()
        # Subscription-id tracking for correct unsubscribe. Kalshi unsubscribe is
        # BY SID, not market_ticker (`market_tickers` → error code 4). Each
        # subscribe COMMAND yields one sid (in the `subscribed` response, keyed by
        # our command `id`) covering all that command's tickers. We map
        # ticker→sid + sid→ticker-set so we can unsubscribe a sid the moment its
        # last ticker leaves the target set — never dropping a still-wanted
        # ticker, and never sending the failing market_tickers form. All reset on
        # reconnect (a fresh connection re-issues fresh sids).
        # Set by _force_reconnect so _connect_loop can tell a deliberate
        # snapshot refresh from a real connection failure (see _connect_loop).
        self._intentional_close = False
        self._pending_sub: dict[int, list[str]] = {}   # cmd_id → tickers awaiting `subscribed`
        self._sid_by_ticker: dict[str, int] = {}
        self._tickers_by_sid: dict[int, set[str]] = {}
        self._stop = asyncio.Event()
        self._run_task: Optional[asyncio.Task] = None
        self._stall_task: Optional[asyncio.Task] = None
        self._reseed_task: Optional[asyncio.Task] = None
        self._stale_book_task: Optional[asyncio.Task] = None

        # Per-connection state (reset on reconnect).
        self._conn_last_seq: Optional[int] = None
        self._conn_last_msg_ts: Optional[float] = None
        self._conn_open_ts: Optional[float] = None

        # ── Lag-proof journal (2026-08-02) ────────────────────────────────
        # Records (t_recv, t_apply) per delta so production application lag
        # can be PROVEN against the unloaded forensic replica's journal of the
        # same events, matched by (ticker, side, price, delta) sequence. See
        # _ws_lag_verdict.py. Flag-file gated (ws_lag_journal.flag, house
        # style), buffered writes off the hot path's critical dict ops, hard
        # byte cap so it can never quota-wedge the disk (EDQUOT incident).
        self._journal_buf: list = []
        self._journal_on = False
        self._journal_flag_ts = 0.0
        self._journal_bytes = 0
        self._journal_capped = False

        self.stats: dict[str, int] = defaultdict(int)
        # Optional event sink: callback fired on (event_type, dict_details).
        # Useful for production observability — wire into trade-log/Slack/Discord.
        self._on_event: Optional[Callable[[str, dict], None]] = None

    def set_event_callback(self, cb: Callable[[str, dict], None]) -> None:
        self._on_event = cb

    def _emit(self, event_type: str, **details) -> None:
        self.stats[event_type] += 1
        if self._on_event:
            try:
                self._on_event(event_type, details)
            except Exception:
                logger.exception("event callback failed")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._run_task is not None:
            return
        self._stop.clear()
        self._run_task = asyncio.create_task(self._connect_loop(), name="ws_delta_conn")
        self._stall_task = asyncio.create_task(self._stall_watchdog(), name="ws_delta_stall")
        self._stale_book_task = asyncio.create_task(self._stale_book_watchdog(), name="ws_delta_stale_book")
        if self.periodic_reseed_s > 0:
            self._reseed_task = asyncio.create_task(self._periodic_reseed_loop(), name="ws_delta_reseed")
        # Give the connection a beat to come up before set_tickers can run.
        await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._stop.set()
        for t in (self._run_task, self._stall_task, self._reseed_task, self._stale_book_task):
            if t is not None:
                t.cancel()
        for t in (self._run_task, self._stall_task, self._reseed_task, self._stale_book_task):
            if t is not None:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._run_task = self._stall_task = self._reseed_task = self._stale_book_task = None
        self._ws = None

    # ── Subscription management ───────────────────────────────────────────

    async def set_tickers(self, tickers: set[str]) -> None:
        """Diff against current subscriptions. Subscribe new ones, unsubscribe
        gone ones. Tickers are kept on this manager even when the underlying
        WS reconnects — the reconnect path re-subscribes the full target set.

        Idempotent: calling with the same set is a no-op.
        Concurrent-safe: serializes via lock so two callers can't race.
        """
        async with self._sub_lock:
            new_targets = set(tickers)
            to_add = new_targets - self._target_tickers
            to_remove = self._target_tickers - new_targets
            self._target_tickers = new_targets

            for t in to_remove:
                self.books.pop(t, None)

            # If WS is up, dispatch subscribe/unsubscribe. _send_subscribe/
            # _send_unsubscribe handle the case where send() fails (dead conn).
            if self._ws is not None:
                if to_add:
                    await self._send_subscribe(list(to_add))
                if to_remove:
                    await self._send_unsubscribe(list(to_remove))
            # If WS isn't open right now, the next connect cycle will subscribe
            # to the full _target_tickers set.

    async def _send_subscribe(self, tickers: list[str]) -> None:
        # Tickers passed in are the bot's *synthetic* (internal) names.
        # Translate to *real* Kalshi tickers for the wire — pass-through for
        # any non-aliased ticker. Internal book/state stays keyed on synthetic.
        if not tickers:
            return
        for i in range(0, len(tickers), SUBSCRIBE_CHUNK):
            chunk_synth = tickers[i:i + SUBSCRIBE_CHUNK]
            chunk_wire = [ticker_aliases.resolve(t) for t in chunk_synth]
            self._sub_cmd_id += 1
            msg = {
                "id": self._sub_cmd_id,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": chunk_wire,
                },
            }
            for t in chunk_synth:
                # Pre-create a buffering book so deltas arriving before the
                # snapshot have somewhere to go. Keyed on synthetic.
                self.books.setdefault(t, WSBook())
                self._subscribed_tickers.add(t)
            # Remember which tickers this command's sid will cover; resolved when
            # the `subscribed` response arrives (matched by cmd id).
            self._pending_sub[self._sub_cmd_id] = list(chunk_synth)
            try:
                await self._ws.send(json.dumps(msg))
                self._emit("subscribe", n=len(chunk_wire))
            except Exception as e:
                logger.warning(f"subscribe send failed: {e}")
                self._pending_sub.pop(self._sub_cmd_id, None)
                return

    async def _send_unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe BY SID (Kalshi rejects `market_tickers` with code 4).

        A sid is dropped only once its LAST covered ticker leaves — so a sid that
        still has wanted tickers is left intact and no re-subscribe is needed.
        Tickers whose sid isn't known yet (subscribe not yet confirmed) are simply
        dropped from our maps; their now-unwanted deltas are ignored downstream
        (book already popped), and the sid is cleaned up when its remainder goes.
        """
        if not tickers:
            return
        affected_sids: set[int] = set()
        for t in tickers:
            self._subscribed_tickers.discard(t)
            sid = self._sid_by_ticker.pop(t, None)
            if sid is not None:
                affected_sids.add(sid)
                covered = self._tickers_by_sid.get(sid)
                if covered is not None:
                    covered.discard(t)
        sids_to_drop = [sid for sid in affected_sids if not self._tickers_by_sid.get(sid)]
        for sid in sids_to_drop:
            self._tickers_by_sid.pop(sid, None)
        if not sids_to_drop or self._ws is None:
            return
        self._sub_cmd_id += 1
        msg = {"id": self._sub_cmd_id, "cmd": "unsubscribe", "params": {"sids": sids_to_drop}}
        try:
            await self._ws.send(json.dumps(msg))
            self._emit("unsubscribe", n=len(sids_to_drop))
        except Exception as e:
            logger.warning(f"unsubscribe send failed: {e}")

    # ── Read API ──────────────────────────────────────────────────────────

    def get_orderbook_fp(self, ticker: str) -> Optional[dict]:
        """Returns the production-compatible orderbook dict, or None if the
        book is not yet seeded, is stale, OR is empty-but-LIVE (a transient
        state we've observed where deltas have wiped every level but no
        snapshot has yet re-seeded the book).

        Empty-but-LIVE on an active market is ALWAYS a sync corruption — a
        real Kalshi orderbook for an active pre-game has hundreds of resting
        levels. Returning empty here would cause downstream BLIP detection
        to interpret the recovery as a price move; we observed this exact
        failure cause two fake-edge `LEAD-LAG ARBITRAGE [SERIES]` fills on
        KTGEN-GEN at 03:47:01 and 03:50:27 UTC 2026-06-13. The fix is to
        treat empty-but-LIVE as 'missing' so the caller falls back to REST.

        Callers should treat None as 'missing → fall back to REST'."""
        b = self.books.get(ticker)
        # Attribute each distinct miss cause (see ws_miss_stats). The four
        # branches below look alike from the caller — a bare None — which is
        # exactly why REST-fallback load was un-diagnosable. Observability
        # only; the routing decisions are unchanged.
        if b is None:
            ws_miss_stats.record("ws_delta", ticker, ws_miss_stats.UNSEEDED)
            return None
        if not b.is_live:
            ws_miss_stats.record("ws_delta", ticker, ws_miss_stats.NOT_LIVE)
            return None
        age = time.time() - b.last_msg_ts
        if age > self.stale_book_s:
            ws_miss_stats.record("ws_delta", ticker, ws_miss_stats.STALE, f"{age:.0f}s")
            return None
        # Guard against the empty-but-LIVE corruption (see docstring).
        if not b.yes and not b.no:
            ws_miss_stats.record("ws_delta", ticker, ws_miss_stats.EMPTY_LIVE, f"{age:.0f}s")
            return None
        ws_miss_stats.record_served("ws_delta")
        return b.to_orderbook_fp()

    def get_orderbooks_fp(self, tickers: list[str]) -> dict[str, Optional[dict]]:
        return {t: self.get_orderbook_fp(t) for t in tickers}

    # ── Lag-proof journal helpers ─────────────────────────────────────────

    def _journal_active(self) -> bool:
        """Flag-file gate, re-checked at most every 5s (one stat call)."""
        now = time.time()
        if now - self._journal_flag_ts > 5.0:
            self._journal_flag_ts = now
            try:
                self._journal_on = (not self._journal_capped
                                    and os.path.exists(JOURNAL_FLAG))
            except Exception:
                self._journal_on = False
        return self._journal_on

    def _journal(self, t_recv, t_apply, ticker, seq, side, price, delta, etype):
        """Buffer one journal line; flush every JOURNAL_FLUSH_LINES. Never raises."""
        try:
            self._journal_buf.append(
                f"{t_recv:.6f},{t_apply:.6f},{ticker},{seq},{side},{price},{delta},{etype}\n")
            if len(self._journal_buf) >= JOURNAL_FLUSH_LINES:
                self._journal_flush()
        except Exception:
            pass

    def _journal_flush(self) -> None:
        if not self._journal_buf:
            return
        buf, self._journal_buf = self._journal_buf, []
        try:
            data = "".join(buf)
            self._journal_bytes += len(data)
            if self._journal_bytes > JOURNAL_CAP_BYTES:
                if not self._journal_capped:
                    self._journal_capped = True
                    self._journal_on = False
                    logger.warning("[WS-LAG-JOURNAL] byte cap reached — journaling stopped")
                return
            with open(JOURNAL_PATH, "a") as f:
                f.write(data)
        except Exception:
            pass

    # ── Connection loop ───────────────────────────────────────────────────

    async def _connect_loop(self) -> None:
        backoff = RECONNECT_BACKOFF_START_S
        while not self._stop.is_set():
            try:
                await self._connect_and_listen()
                backoff = RECONNECT_BACKOFF_START_S
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._emit("conn_drop", err=str(e)[:200])

                # ── Backoff correctness (2026-07-27) ──────────────────────
                # `_connect_and_listen` never returns normally — it is
                # `async with connect(...)` around `while not self._stop`, so it
                # only exits by raising. That made the `backoff = START` reset
                # after the await unreachable in normal operation: the backoff
                # ratcheted 2→3→4.5→6.8→10→15→23→30 and then stayed pinned at
                # the 30s cap for the entire process lifetime.
                #
                # Measured on the 27 Jul run: 637 of ~650 reconnects logged
                # "backoff 30.0s". Since `_force_reconnect` deliberately closes
                # the socket to refresh snapshots, EVERY intentional reseed also
                # paid that 30s — ~5.3h of a 15h run with no WS books at all,
                # against a docstring that budgeted "~2-3s per reseed".
                #
                # Two corrections:
                #   * an intentional close is not a failure — reconnect at once
                #   * a connection that stayed up a good while and then dropped
                #     is not a failure LOOP — restart the ladder from the bottom
                intentional = self._intentional_close
                self._intentional_close = False
                uptime = (time.time() - self._conn_open_ts) if self._conn_open_ts else 0.0
                if intentional or uptime >= HEALTHY_CONN_S:
                    backoff = RECONNECT_BACKOFF_START_S

                if intentional:
                    logger.info(
                        f"WS closed intentionally (reseed) after {uptime:.0f}s up "
                        f"— reconnecting immediately, no backoff"
                    )
                    continue

                logger.warning(
                    f"WS connection lost: {type(e).__name__}: {e} — "
                    f"up {uptime:.0f}s, backoff {backoff:.1f}s"
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return
                backoff = min(backoff * RECONNECT_BACKOFF_MULT, RECONNECT_BACKOFF_CAP_S)

    async def _connect_and_listen(self) -> None:
        headers = self.auth.get_headers("GET", WS_SIGNING_PATH)
        async with websockets.connect(WS_URL, additional_headers=headers,
                                       ping_interval=20, ping_timeout=20,
                                       open_timeout=10, close_timeout=2) as ws:
            self._ws = ws
            self._conn_open_ts = time.time()
            self._conn_last_msg_ts = time.time()
            self._conn_last_seq = None
            # Re-subscribe to all target tickers. Mark existing books as stale
            # so they reseed cleanly. Sids are per-connection, so drop the old
            # mapping — the re-subscribe below mints fresh ones.
            self._subscribed_tickers.clear()
            self._pending_sub.clear()
            self._sid_by_ticker.clear()
            self._tickers_by_sid.clear()
            for t, book in self.books.items():
                book.mark_stale()
            if self._target_tickers:
                await self._send_subscribe(list(self._target_tickers))
            self._emit("conn_open", n_tickers=len(self._target_tickers))

            try:
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    except asyncio.TimeoutError:
                        self._journal_flush()   # quiet period: drain the buffer
                        continue
                    t_recv = time.time()
                    self._conn_last_msg_ts = t_recv
                    self.stats["msgs_recv"] += 1
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self._handle_msg(msg, t_recv)
            finally:
                self._ws = None

    def _handle_msg(self, msg: dict, t_recv: Optional[float] = None) -> None:
        mtype = msg.get("type") or ""
        payload = msg.get("msg") or {}
        seq = msg.get("seq")  # top-level, NOT inside msg
        if seq is None:
            seq = payload.get("seq")  # defensive

        # ── Per-connection seq gap detection ──
        # Kalshi seq is per-WS-connection monotonic (verified via _ws_dump.py
        # 2026-06-12; per-ticker seqs have constant interleaving artifacts and
        # produce ~11% false positives). Global is the correct gauge.
        if seq is not None:
            try:
                seq_i = int(seq)
            except (ValueError, TypeError):
                seq_i = None
            if seq_i is not None:
                if self._conn_last_seq is not None and seq_i != self._conn_last_seq + 1:
                    self._emit("conn_seq_gap",
                               prev=self._conn_last_seq, new=seq_i,
                               lost=seq_i - self._conn_last_seq - 1)
                    # Mark every subscribed ticker stale; the next snapshot
                    # for each will re-seed it. Force a re-subscribe to get
                    # fresh snapshots.
                    for t in self._subscribed_tickers:
                        b = self.books.get(t)
                        if b is not None:
                            b.mark_stale()
                    # Force a full reconnect to refresh snapshots. Re-subscribing
                    # on the same WS connection does NOT yield a fresh snapshot
                    # (Kalshi just acks; see _force_reconnect docstring). The
                    # reconnect cycle is the only known path. Spawn as task so
                    # we don't block message processing.
                    if self._ws is not None and self._target_tickers:
                        asyncio.create_task(self._force_reconnect())
                self._conn_last_seq = seq_i

        if mtype == "subscribed":
            # {"type":"subscribed","id":<cmd>,"msg":{"channel":...,"sid":N}}
            # Correlate the sid to that subscribe command's tickers (by cmd id)
            # so unsubscribe can target sids (Kalshi rejects market_tickers). Only
            # retain tickers still wanted — a fast add→remove before the ack must
            # not resurrect a dropped ticker into the sid map.
            cmd_id = msg.get("id")
            sid = payload.get("sid")
            tks = self._pending_sub.pop(cmd_id, None) if cmd_id is not None else None
            if sid is not None and tks:
                keep = {t for t in tks if t in self._subscribed_tickers}
                if keep:
                    self._tickers_by_sid[sid] = keep
                    for t in keep:
                        self._sid_by_ticker[t] = sid
            return

        if mtype == "orderbook_snapshot":
            # Inbound payload tags the *real* ticker; translate to *synthetic*
            # so the book lookup matches what set_tickers/set_subscribe stored.
            # Pass-through for non-aliased.
            ticker = ticker_aliases.reverse(payload.get("market_ticker") or "")
            if ticker not in self.books:
                # Snapshot for a ticker we no longer want.
                self.stats["snapshot_ignored"] += 1
                return
            book = self.books[ticker]
            replayed = book.apply_snapshot(payload, seq)
            self._emit("snapshot", ticker=ticker, replayed=replayed,
                       n_yes=len(book.yes), n_no=len(book.no))
            if self._journal_active():
                # Snapshot marker: the aligner resets per-ticker sync here
                # (post-reconnect streams have no event continuity).
                self._journal(t_recv or 0.0, time.time(), ticker, seq,
                              "", "", replayed, "S")
            return

        if mtype == "orderbook_delta":
            # See snapshot handler re: real→synthetic translation.
            ticker = ticker_aliases.reverse(payload.get("market_ticker") or "")
            book = self.books.get(ticker)
            if book is None:
                return
            if not book.is_live:
                book.buffer_delta(payload, seq)
                self.stats["delta_buffered"] += 1
                if self._journal_active():
                    self._journal(t_recv or 0.0, time.time(), ticker, seq,
                                  payload.get("side", ""),
                                  payload.get("price_dollars", ""),
                                  payload.get("delta_fp", ""), "B")
                return
            book.apply_delta_live(payload, seq)
            self.stats["delta_applied"] += 1
            if self._journal_active():
                self._journal(t_recv or 0.0, time.time(), ticker, seq,
                              payload.get("side", ""),
                              payload.get("price_dollars", ""),
                              payload.get("delta_fp", ""), "D")
            return

        if mtype in ("error", "errors"):
            self._emit("ws_error", body=str(payload)[:300])

    async def _force_reconnect(self) -> None:
        """Close the current WS to force `_connect_loop` to re-establish and
        re-subscribe from scratch. This is the only known path to a clean
        snapshot refresh on Kalshi WS.

        Verified 2026-06-13 via direct probe:
          * Re-subscribing on the same connection to a ticker that's already
            subscribed returns {"type":"ok"} and NO snapshot.
          * Unsubscribing with `market_tickers:[...]` returns
            {"type":"error", "code":4, "msg":"Subscription IDs required"}.
            Kalshi requires `sids:[<sub_id>]` for unsubscribe.

        So the only way to force fresh snapshots is to close the underlying
        WS connection and re-subscribe through a fresh subscribe command —
        which our reconnect loop does naturally.

        Cost: ~2-3s window per reseed during which books are stale. At a
        600s reseed cadence that's <0.5% downtime, well within tolerance.
        """
        if self._ws is None:
            return
        async with self._sub_lock:
            if self._ws is None:
                return
            self._emit("forced_reconnect")
            # Tell _connect_loop this close is deliberate so it reconnects
            # immediately instead of charging the failure backoff. Set BEFORE
            # close() — the resulting exception is raised on the reader task and
            # can reach the handler before we return from here.
            self._intentional_close = True
            try:
                await self._ws.close()
            except Exception:
                pass
            # _connect_and_listen's `finally` clears self._ws.
            # _connect_loop's outer while then reconnects with NO backoff (the
            # trailing comment here used to claim "backoff=2s"; in practice the
            # ladder was pinned at its 30s cap — see _connect_loop).

    # ── Watchdogs ─────────────────────────────────────────────────────────

    async def _stall_watchdog(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(5.0)
                if self._ws is None or self._conn_last_msg_ts is None:
                    continue
                # Skip stall check when we haven't yet subscribed to anything.
                # The WS connection sits idle (no message flow) until the host
                # process calls set_tickers(). With ~80 tickers and ~20s of
                # bulk metadata fetch at startup, set_tickers might not run
                # for 30+s — but that's not a stall, just startup latency.
                # Without this guard, we'd false-alarm-close on every restart.
                if not self._target_tickers:
                    continue
                silence = time.time() - self._conn_last_msg_ts
                if silence > self.stall_threshold_s:
                    self._emit("conn_stall", silence_s=round(silence, 1))
                    logger.warning(f"WS stall: {silence:.1f}s with no message — closing for reconnect")
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    # _connect_loop will reconnect with backoff.
                    self._conn_last_msg_ts = None  # avoid retriggering
        except asyncio.CancelledError:
            return

    async def _stale_book_watchdog(self) -> None:
        """Per-ticker staleness → force reconnect (not perpetual REST fallback).

        Without this, a single stale book would trigger REST fallback every
        tick in the host process for the full STALE_BOOK_S window (or until
        Kalshi naturally sent a delta). Reconnecting refreshes every book via
        new snapshots within ~5s and ends the fallback cycle.

        Cooldown (`_STALE_RECONNECT_COOLDOWN_S`) prevents thrashing when a
        legitimately-quiet market (heavy favorite, no trade flow) keeps
        re-flagging stale right after the reconnect's reseed.
        """
        # 2026-07-17: 60s → 300s. At 60s a permanently-quiet book (settled /
        # post-game / pre-game heavy favorite that never trades) re-stales
        # right after each reseed, so the global _force_reconnect fired every
        # ~90s — dropping ALL books for 2-3s and forcing a REST-fallback storm
        # that stretched the trade loop to ~1.3s/cycle and got orders picked
        # off. 300s makes the (rare) genuinely-broken-sub reconnect infrequent
        # enough that quiet books don't storm the connection.
        STALE_RECONNECT_COOLDOWN_S = 300.0
        last_reconnect_ts = 0.0
        try:
            while not self._stop.is_set():
                await asyncio.sleep(10.0)
                if self._ws is None or not self._target_tickers:
                    continue
                now = time.time()
                if (now - last_reconnect_ts) < STALE_RECONNECT_COOLDOWN_S:
                    continue
                # Find books that are LIVE (seeded) but haven't received a
                # delta within stale_book_s. Connection-wide silence is the
                # `_stall_watchdog`'s job — this targets the case where the
                # connection is otherwise healthy but a specific ticker has
                # silently stopped receiving deltas.
                stale_count = 0
                live_count = 0
                worst_age = 0.0
                worst_ticker = ""
                for t, b in self.books.items():
                    if not b.is_live:
                        continue
                    live_count += 1
                    age = now - b.last_msg_ts
                    if age > STALE_RECONNECT_AGE_S:
                        stale_count += 1
                        if age > worst_age:
                            worst_age = age
                            worst_ticker = t
                # 2026-07-27: `stale_count > 0` meant ONE quiet ticker forced a
                # full reconnect of every book. The 2026-07-17 note above raised
                # the COOLDOWN to 300s but left the trigger unchanged, so this
                # simply fired as fast as the cooldown allowed — observed 19:58
                # and 20:03 with "worst" ages of 63.3s and 69.6s across 69 and
                # 52 tickers. With 6 games live, ~70 mostly-illiquid map/series
                # books go >60s without a tick as a matter of course, so the
                # sweep re-armed ~60s after every reseed, forever.
                #
                # Age alone cannot separate "quiet" from "broken" — a pre-game
                # market legitimately sits silent for many minutes. BREADTH can:
                # a wedged connection starves EVERY book at once, whereas quiet
                # markets only ever stale the illiquid subset. So require both a
                # much older age AND most of the connection to be affected.
                #
                # Safety is unchanged: `get_orderbook_fp` still refuses to serve
                # any book older than stale_book_s (60s), so a genuinely broken
                # single subscription degrades to the ws_delta/REST fallback
                # path rather than being served stale — it just no longer takes
                # all 70 books down with it. Connection-wide silence remains the
                # `_stall_watchdog`'s job.
                broad = (live_count > 0
                         and stale_count >= max(STALE_RECONNECT_MIN_N,
                                                STALE_RECONNECT_MIN_FRAC * live_count))
                if broad:
                    self._emit("stale_book_reconnect",
                               n=stale_count, worst_age_s=round(worst_age, 1))
                    logger.warning(
                        f"[STALE-BOOK] {stale_count}/{live_count} live book(s) "
                        f"stale >{STALE_RECONNECT_AGE_S:.0f}s "
                        f"({stale_count / max(live_count, 1) * 100:.0f}% of the "
                        f"connection; worst: {worst_ticker} @ {worst_age:.1f}s) "
                        f"— connection looks wedged, force_reconnect"
                    )
                    last_reconnect_ts = now
                    asyncio.create_task(self._force_reconnect())
        except asyncio.CancelledError:
            return

    async def _periodic_reseed_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.periodic_reseed_s)
                if self._ws is None:
                    continue
                self._emit("periodic_reseed_triggered")
                await self._force_reconnect()
        except asyncio.CancelledError:
            return


# ─── Standalone validation CLI ───────────────────────────────────────────────
#
# `python3 ws_delta_book.py validate --tickers T1,T2 --duration 600 --interval 30`
#
# Runs the same RACE/REST_LAG/DRIFT_WS classifier as
# temp_websocket_rest_comparison_v2.py but against THIS module's
# delta-maintained book and authed REST that matches production exactly.


def _fetch_rest_book_authed(ticker: str, auth: KalshiAuth) -> Optional[dict]:
    """Production-equivalent REST orderbook fetch. Same auth headers, same
    host, same path as `kalshi_client.get_orderbook()`. Verified byte-equivalent
    on 2026-06-13."""
    path = REST_OB_PATH.format(ticker=ticker)
    try:
        r = requests.get(REST_HOST + path, headers=auth.get_headers("GET", path), timeout=5)
        if r.status_code != 200:
            return None
        ob = r.json().get("orderbook_fp") or {}
    except Exception:
        return None
    out_yes: dict[int, int] = {}
    out_no: dict[int, int] = {}
    for pair in ob.get("yes_dollars") or []:
        try:
            out_yes[_price_to_cents(pair[0])] = _qty_to_fp(pair[1])
        except (ValueError, TypeError, IndexError):
            continue
    for pair in ob.get("no_dollars") or []:
        try:
            out_no[_price_to_cents(pair[0])] = _qty_to_fp(pair[1])
        except (ValueError, TypeError, IndexError):
            continue
    return {"yes": out_yes, "no": out_no}


def _book_to_fp_dict(book: WSBook) -> dict:
    """Same shape as _fetch_rest_book_authed return — internal cents/fp form."""
    return {"yes": dict(book.yes), "no": dict(book.no)}


def _diff_count(ws: dict[int, int], rest: dict[int, int]) -> int:
    return sum(1 for p in set(ws) | set(rest)
               if abs(ws.get(p, 0) - rest.get(p, 0)) > 0)  # exact int compare


# ── Meaningful-drift filter ─────────────────────────────────────────────────
# We don't care if the 1¢ tail level is 10k contracts off — that bid won't
# trade against our quotes inside a year. We DO care if the BBO or anything
# within MEANINGFUL_WINDOW_CENTS of mid disagrees, because that's where
# trading happens and that's where bad data causes bad fills.
MEANINGFUL_WINDOW_CENTS = 5


def _bbo_yes_mid(rest_book: dict) -> Optional[int]:
    """Synthetic YES mid in cents. None if either side is empty.

      yes_bid_best = highest YES bid price = max(yes_dollars keys)
      yes_ask_best = 100 − highest NO bid price = 100 − max(no_dollars keys)
      mid = (yes_bid_best + yes_ask_best) // 2
    """
    if not rest_book.get("yes") or not rest_book.get("no"):
        return None
    yes_bid = max(rest_book["yes"])
    no_bid = max(rest_book["no"])
    yes_ask = 100 - no_bid
    return (yes_bid + yes_ask) // 2


def _diff_count_near_bbo(ws: dict[int, int], rest: dict[int, int],
                          mid_yes: Optional[int], side: str,
                          window: int = MEANINGFUL_WINDOW_CENTS) -> int:
    """Count diffs at price levels within `window` cents of the relevant mid.
    side='yes' → keep prices in [mid - window, mid + window]
    side='no'  → keep prices in [(100-mid) - window, (100-mid) + window]
    If mid_yes is None (one side empty), treat every level as 'near' since we
    can't compute the BBO — conservative."""
    if mid_yes is None:
        return _diff_count(ws, rest)
    if side == "yes":
        lo, hi = mid_yes - window, mid_yes + window
    else:
        no_mid = 100 - mid_yes
        lo, hi = no_mid - window, no_mid + window
    return sum(1 for p in set(ws) | set(rest)
               if lo <= p <= hi and abs(ws.get(p, 0) - rest.get(p, 0)) > 0)


def _classify(ws_before: dict, ws_after: dict,
              rest1: dict, rest2: dict
              ) -> tuple[str, int, int, int, int, int, Optional[int]]:
    """Returns (status, db, da, rest_drift, db_near, da_near, mid_yes).

    Status rubric matches temp_websocket_rest_comparison_v2.py — based on ALL
    diffs (every price level). The `*_near` counters report how many of those
    diffs fall within MEANINGFUL_WINDOW_CENTS of the BBO mid. A DRIFT_WS event
    with da_near=0 is uninteresting (deep-tail-only); da_near>0 means the WS
    book disagrees with REST at a price the bot would actually trade against.
    """
    rest_drift = (_diff_count(rest1["yes"], rest2["yes"])
                  + _diff_count(rest1["no"], rest2["no"]))
    db = (_diff_count(ws_before["yes"], rest1["yes"])
          + _diff_count(ws_before["no"], rest1["no"]))
    da = (_diff_count(ws_after["yes"], rest2["yes"])
          + _diff_count(ws_after["no"], rest2["no"]))
    # Meaningful counts use rest2 as the reference (closest to ws_after timing).
    mid_yes = _bbo_yes_mid(rest2)
    db_near = (_diff_count_near_bbo(ws_before["yes"], rest1["yes"], mid_yes, "yes")
               + _diff_count_near_bbo(ws_before["no"], rest1["no"], mid_yes, "no"))
    da_near = (_diff_count_near_bbo(ws_after["yes"], rest2["yes"], mid_yes, "yes")
               + _diff_count_near_bbo(ws_after["no"], rest2["no"], mid_yes, "no"))
    if da == 0:
        status = "OK" if db == 0 else "RACE"
    elif rest_drift > 0:
        status = "REST_LAG"
    else:
        status = "DRIFT_WS"
    return (status, db, da, rest_drift, db_near, da_near, mid_yes)


async def _validate_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    pk = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if pk and not pk.startswith("/"):
        pk = f"/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage/{pk}"
    if not (api_key and pk):
        logger.error("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set in .env")
        return 2
    auth = KalshiAuth(api_key, pk)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        logger.error("no tickers provided")
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"_ws_delta_validate_{stamp}.csv")
    log_fp = open(log_path, "w", buffering=1)
    log_fp.write("ts,ticker,status,db,da,rest_drift,db_near,da_near,mid_yes,ws_top_yes_bid,rest_top_yes_bid\n")

    mgr = WSDeltaManager(auth, periodic_reseed_s=args.periodic_reseed)
    await mgr.start()
    await mgr.set_tickers(set(tickers))
    logger.info(f"validate: {len(tickers)} tickers, duration={args.duration}s, interval={args.interval}s, "
                f"periodic_reseed={args.periodic_reseed}s (0=off)")
    logger.info(f"          log → {log_path}")
    # Warm-up: wait for first seed on all tickers (or 10s, whichever first).
    warm_start = time.time()
    while time.time() - warm_start < 10:
        if all(b.is_live for b in mgr.books.values()):
            break
        await asyncio.sleep(0.5)

    end_at = time.time() + args.duration
    stats: dict[str, int] = defaultdict(int)
    per_ticker_drift_all: dict[str, int] = defaultdict(int)
    per_ticker_drift_near: dict[str, int] = defaultdict(int)
    round_n = 0
    while time.time() < end_at:
        round_n += 1
        logger.info(f"──── round {round_n} ({datetime.now().strftime('%H:%M:%S')}) ────")
        for tkr in tickers:
            book = mgr.books.get(tkr)
            if book is None or not book.is_live:
                stats["ws_unready"] += 1
                continue
            ws_before = _book_to_fp_dict(book)
            rest1 = await asyncio.to_thread(_fetch_rest_book_authed, tkr, auth)
            await asyncio.sleep(0.1)
            rest2 = await asyncio.to_thread(_fetch_rest_book_authed, tkr, auth)
            ws_after = _book_to_fp_dict(book)
            if rest1 is None or rest2 is None:
                stats["rest_err"] += 1
                continue
            status, db, da, rd, db_near, da_near, mid_yes = _classify(ws_before, ws_after, rest1, rest2)
            stats[status] += 1
            # Track the meaningful subset: DRIFT_WS where at least one near-BBO
            # level disagreed. This is the trading-relevant signal.
            if status == "DRIFT_WS":
                per_ticker_drift_all[tkr] += 1
                if da_near > 0:
                    stats["DRIFT_WS_NEAR_BBO"] += 1
                    per_ticker_drift_near[tkr] += 1
            ws_top = max(ws_after["yes"]) if ws_after["yes"] else ""
            rest_top = max(rest2["yes"]) if rest2["yes"] else ""
            log_fp.write(f"{datetime.now().isoformat(timespec='seconds')},{tkr},{status},{db},{da},{rd},{db_near},{da_near},{mid_yes if mid_yes is not None else ''},{ws_top},{rest_top}\n")
            level = logging.WARNING if (status == "DRIFT_WS" and da_near > 0) else logging.DEBUG
            logger.log(level, f"{tkr:<45s}  {status:<8s}  db={db:<3d}/{db_near}  da={da:<3d}/{da_near}  rd={rd:<3d}  mid={mid_yes}  ws_top={ws_top} rest_top={rest_top}")
            await asyncio.sleep(0.2)
        logger.info(f"  round {round_n}: OK={stats['OK']} RACE={stats['RACE']} "
                    f"REST_LAG={stats['REST_LAG']} DRIFT_WS={stats['DRIFT_WS']} "
                    f"(of those, near-BBO={stats.get('DRIFT_WS_NEAR_BBO',0)}) "
                    f"conn={{seq_gap:{mgr.stats.get('conn_seq_gap',0)}, "
                    f"stall:{mgr.stats.get('conn_stall',0)}, "
                    f"drop:{mgr.stats.get('conn_drop',0)}}}")
        await asyncio.sleep(max(0, args.interval - 1.0))

    log_fp.close()
    await mgr.stop()
    total = sum(stats.get(k, 0) for k in ("OK", "RACE", "REST_LAG", "DRIFT_WS"))
    drift_all = stats.get("DRIFT_WS", 0)
    drift_near = stats.get("DRIFT_WS_NEAR_BBO", 0)
    print("\n" + "=" * 75)
    print(f"VALIDATE — {round_n} rounds, {len(tickers)} tickers, {total} comparisons")
    print("=" * 75)
    for k in ("OK", "RACE", "REST_LAG", "DRIFT_WS"):
        v = stats.get(k, 0)
        pct = f"{100*v/max(1,total):>5.1f}%"
        print(f"  {k:<15s} {v:>6d}  {pct}")
    print(f"  {'  ↳ near BBO':<15s} {drift_near:>6d}  {100*drift_near/max(1,total):>5.1f}%  "
          f"({100*drift_near/max(1,drift_all):.1f}% of DRIFT_WS — only these matter for trading)")
    for k in ("ws_unready", "rest_err"):
        v = stats.get(k, 0)
        if v: print(f"  {k:<15s} {v:>6d}")
    print(f"\n  WS event counts:")
    for k, v in sorted(mgr.stats.items()):
        print(f"    {k:<22s} {v}")
    if per_ticker_drift_all:
        print(f"\n  Per-ticker drift (DRIFT_WS_all / DRIFT_WS_near_BBO):")
        for t in sorted(per_ticker_drift_all, key=lambda t: -per_ticker_drift_all[t]):
            print(f"    {t}: {per_ticker_drift_all[t]} / {per_ticker_drift_near.get(t,0)}")
    print(f"\n  Window: ±{MEANINGFUL_WINDOW_CENTS}¢ around BBO mid is 'meaningful'.")
    print(f"  log: {log_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate", help="Run validation against authed REST (production-equivalent path)")
    v.add_argument("--tickers", required=True, help="Comma-separated Kalshi market tickers")
    v.add_argument("--duration", type=int, default=600, help="Seconds to run (default 600)")
    v.add_argument("--interval", type=float, default=30.0, help="Seconds between compare rounds (default 30)")
    v.add_argument("--periodic-reseed", type=float, default=0.0,
                   help="Seconds between forced re-subscribes (anti-phantom hygiene). "
                        "0 = disabled. Recommended 600s for any long live-game run.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "validate":
        try:
            return asyncio.run(_validate_main(args))
        except KeyboardInterrupt:
            return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
