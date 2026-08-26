"""Isolated WS delta orderbook — the black box.

    set_tickers([...])  →  [ own WS connection, OWN THREAD ]  →  get_orderbook_fp(t)

Drop-in replacement for `ws_delta_book.WSDeltaManager`'s READ interface, but the
reader runs on its own thread + own event loop + own connection, so it can never
be starved by the trading loop → no dropped deltas → the incremental base never
diverges → the book never rots. Output format is byte-identical to the current
`get_orderbook_fp`, so every downstream consumer (scrub, sweep, blip, arber) is
UNCHANGED — it just receives the full deep book instead of a truncated one.

Validated design: see `_ws_correct_book.py` (0 gaps / 0 persistent drops vs REST
over a 25-min live soak). This module productionizes it: integer fixed-point qty
(no float drift), dynamic subscribe/unsubscribe, ticker aliases, staleness guards.

Phase 1 use: run in SHADOW (constructed + fed tickers, but not routed to the
arber). See WS_MIGRATION_PLAN.md.
"""
import time
import json
import threading
import asyncio
import logging
from typing import Optional, Callable

import websockets
import ticker_aliases
import ws_miss_stats

log = logging.getLogger("isolated_ws_book")

WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_SIGNING_PATH = "/trade-api/ws/v2"
# Serve guard: a book older than this is reported as None (caller falls back to
# REST). Raised 60 -> 180 on 2026-07-28.
#
# 60s was cutting into a band this book's OWN safety net creates.
# `_periodic_reconcile_loop` queues EVERY live book for an in-place REST
# reconcile every RECONCILE_PERIOD_S (45s), which resets last_msg_ts — so a
# healthy book's age is structurally bounded at ~45s + fetch jitter, and a 60s
# guard trips on the tail of that band rather than on anything wrong.
#
# Measured 2026-07-28 09:38-10:26 (48 min, 93,479 successful serves):
#   isolated stale ages:  60-69s x1545, 70-79s x989, 80-89s x425, 90-99s x3
#   MAX observed age = 90s. Nothing beyond, in either pregame or live books
#   (pregame max 90s / live max 89s — statistically identical, which is why
#   gating on the pregame latch would NOT have helped here; that signal is the
#   right one for the POLY HEDGE freeze, not for Kalshi book silence).
#
# At 180s: 0 of 2,575 staleness misses survive; 2 of 1,344 [WS FALLBACK] lines
# remain (both not_live/empty_live, not staleness) = 99.9% reduction. 120s
# already suffices; 180s per operator for margin.
#
# TRADE-OFF: a genuinely dropped ticker is now served from a stale book for up
# to 3 min instead of 1. Bounded by three things: the 45s reconcile above, the
# BLIP REST-OVERRIDE path which still corrects the book from REST on real
# divergence, and ws_delta's own stale watchdog.
STALE_BOOK_S = 180.0
MAX_QUEUE = 4096                # vs production default 32 — bursts can't backpressure-drop
SUB_RECONCILE_S = 0.5          # how often the reader diffs target vs subscribed
SUB_CHUNK_N = 25               # tickers per subscribe COMMAND (= per sid). One command
                               # for the whole universe put ~245 tickers under ONE sid,
                               # so any single-ticker unsubscribe nuked every book's
                               # data flow (mid-slate silence epidemic, 2026-08-06).
RECONNECT_CAP_S = 10.0
RECONCILE_PERIOD_S = 45.0     # periodic in-place REST-reconcile SAFETY NET: bounds ANY
                              # divergence (seq-gap OR not) to this interval — non-destructive,
                              # no dark window. This is what makes a non-gap drift self-heal.


# ── fixed-point helpers (identical to ws_delta_book.py so the base never drifts) ──
def _price_to_cents(s) -> int:
    return int(round(float(s) * 100))


def _qty_to_fp(s) -> int:
    return int(round(float(s) * 100))


def _fp_to_qty_str(fp: int) -> str:
    return f"{fp / 100:.2f}"


def _cents_to_price_str(c: int) -> str:
    return f"{c / 100:.4f}"


class _Book:
    """One ticker's book. qty stored as integer fp-hundredths (exact sums)."""
    __slots__ = ("yes", "no", "live", "last_msg_ts", "_pending",
                 "reconciling", "_rbuf")

    def __init__(self):
        self.yes = {}   # price_cents -> qty_fp (int)
        self.no = {}
        self.live = False
        self.last_msg_ts = 0.0
        self._pending = []   # (seq, delta_msg) buffered before the snapshot lands
        # REST-reconcile race guard (2026-07-31). While a REST fetch for this
        # book is in flight, live deltas are ALSO buffered here and replayed on
        # top of the snapshot after the overwrite. Without this, every delta
        # applied during the fetch window was silently discarded with the old
        # dicts — and a discarded REMOVAL resurrected its level from the
        # snapshot: a stuck phantom at top-of-book, i.e. the crossed books the
        # XBOOK detector was catching ~11/hr, clustered on busy books (the 45s
        # periodic reconcile + high delta rate made the race near-certain).
        self.reconciling = False
        self._rbuf = []

    def buffer_delta(self, seq, msg):
        self._pending.append((seq, msg))

    def apply_snapshot(self, msg, snapshot_seq=None):
        self.yes.clear(); self.no.clear()
        for key, tgt in (("yes_dollars_fp", self.yes), ("no_dollars_fp", self.no)):
            for pair in msg.get(key) or []:
                if pair and len(pair) >= 2:
                    q = _qty_to_fp(pair[1])
                    if q > 0:
                        tgt[_price_to_cents(pair[0])] = q
        # Replay buffered deltas that post-date the snapshot (seq > snapshot_seq);
        # discard the ones already reflected. Prevents losing an update that
        # arrived in the subscribe→snapshot window (matches production's WSBook).
        for s, dmsg in self._pending:
            if snapshot_seq is None or s is None or s > snapshot_seq:
                self.apply_delta(dmsg)
        self._pending.clear()
        # A wire snapshot supersedes any REST reconcile in flight: cancel its
        # pending replay so the (older) REST payload can't clobber this state.
        self.reconciling = False
        self._rbuf = []
        self.live = True
        self.last_msg_ts = time.time()

    def apply_delta(self, msg):
        side = msg.get("side")
        tgt = self.yes if side == "yes" else self.no if side == "no" else None
        if tgt is None:
            return
        try:
            p = _price_to_cents(msg["price_dollars"]); d = _qty_to_fp(msg["delta_fp"])
        except (KeyError, ValueError, TypeError):
            return
        nv = tgt.get(p, 0) + d
        if nv <= 0:
            tgt.pop(p, None)
        else:
            tgt[p] = nv
        self.last_msg_ts = time.time()

    def rest_reconcile(self, ob):
        """Overwrite in place from a fresh REST snapshot — no blank, no dark.

        Caller MUST NOT pass a failed fetch. A 404/error collapses to `{}`
        upstream, and treating that as a valid empty snapshot marks the book
        live + fresh + empty forever — which is how phantom map tickers
        (maps 3-5 of a BO3) became permanent `empty_live` misses that drew a
        REST fetch every RECONCILE_PERIOD_S. `_reconcile_worker` now drops
        empty payloads instead of calling this.
        """
        ny, nn = {}, {}
        for key, tgt in (("yes_dollars", ny), ("no_dollars", nn)):
            for pair in ob.get(key) or []:
                if pair and len(pair) >= 2:
                    q = _qty_to_fp(pair[1])
                    if q > 0:
                        tgt[_price_to_cents(pair[0])] = q
        self.yes, self.no = ny, nn
        self._pending.clear()
        self.live = True
        self.last_msg_ts = time.time()

    def to_fp(self) -> dict:
        ys = sorted(self.yes.items(), key=lambda kv: -kv[0])
        ns = sorted(self.no.items(), key=lambda kv: -kv[0])
        return {"orderbook_fp": {
            "yes_dollars": [[_cents_to_price_str(p), _fp_to_qty_str(q)] for p, q in ys],
            "no_dollars":  [[_cents_to_price_str(p), _fp_to_qty_str(q)] for p, q in ns],
        }}


class IsolatedWSBook:
    """WS delta book maintained on its OWN thread. Public read/control interface
    matches ws_delta_book.WSDeltaManager so it's a routing-level drop-in."""

    def __init__(self, auth, client, stale_book_s: float = STALE_BOOK_S):
        self.auth = auth
        self.client = client            # KalshiClient — used (via to_thread) for gap REST-reconcile
        self.stale_book_s = stale_book_s
        self.books = {}                 # synthetic ticker -> _Book
        self._target = set()            # synthetic tickers the trading side wants (set from main thread)
        self._subscribed = set()        # synthetic tickers live on the current connection (reader thread)
        self._conn_seq = None
        self._lock = threading.Lock()   # guards books + _target
        self._stop = threading.Event()
        self._loop = None
        self._thread = None
        self._run_task = None
        self._ws = None
        self._reconcile = set()         # tickers to REST-reconcile after a gap
        self._cmd_tickers = {}          # cmd_id -> [synthetic tickers] awaiting `subscribed` ack
        self._ticker_sid = {}           # synthetic ticker -> sid
        self._sid_tickers = {}          # sid -> set(synthetic tickers)
        self._cmd_id = 0
        self._event_cb: Optional[Callable] = None
        self.stats = {"msgs": 0, "gaps": 0, "reconnects": 0, "reconciles": 0}

    # ───────────────── public interface (main / trading thread) ─────────────────
    async def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="isolated_ws", daemon=True)
        self._thread.start()
        log.warning("IsolatedWSBook started (own thread, max_queue=%d).", MAX_QUEUE)

    def stop(self):
        self._stop.set()
        loop = self._loop; task = self._run_task
        if loop is not None and task is not None:
            # Cancel the reader's main task from its own loop → the ws context
            # managers close cleanly and every coroutine unwinds via
            # CancelledError (no orphan thread, no "task destroyed" warnings).
            try:
                loop.call_soon_threadsafe(task.cancel)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if not self._thread.is_alive():   # reset so start() can spin a fresh reader later
                self._thread = None
                self._run_task = None
                self._loop = None
                self._stop = threading.Event()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    async def set_tickers(self, tickers):
        """Trading side declares which games it wants. Thread-safe state update;
        the reader thread reconciles subscriptions on its own loop."""
        with self._lock:
            self._target = set(tickers)
            for t in self._target:
                self.books.setdefault(t, _Book())
            # Drop books for games no longer wanted so `books` can't grow
            # unboundedly over a multi-day run (the reader unsubscribes them on
            # its next reconcile; a stray delta before that just no-ops).
            for t in [x for x in self.books if x not in self._target]:
                self.books.pop(t, None)

    def get_orderbook_fp(self, ticker) -> Optional[dict]:
        """Thread-safe read. Same None-means-fall-back-to-REST semantics as
        ws_delta_book.get_orderbook_fp: None if unseeded, stale, or empty-but-live."""
        # Each miss below is a structurally different failure needing a
        # different fix, so classify it (see ws_miss_stats). Attribution is
        # recorded AFTER releasing self._lock: this is the hot read path and
        # the whole point of this book is that its reader thread is never
        # contended, so we must not nest the stats lock inside the book lock.
        reason = detail = None
        fp = None
        with self._lock:
            b = self.books.get(ticker)
            if b is None:
                reason = ws_miss_stats.UNSEEDED
            elif not b.live:
                reason = ws_miss_stats.NOT_LIVE
            else:
                age = time.time() - b.last_msg_ts
                if age > self.stale_book_s:
                    reason, detail = ws_miss_stats.STALE, f"{age:.0f}s"
                elif not b.yes and not b.no:
                    reason, detail = ws_miss_stats.EMPTY_LIVE, f"{age:.0f}s"
                else:
                    fp = b.to_fp()

        if reason is not None:
            ws_miss_stats.record("isolated", ticker, reason, detail or "")
            return None
        ws_miss_stats.record_served("isolated")
        return fp

    def get_orderbooks_fp(self, tickers) -> dict:
        return {t: self.get_orderbook_fp(t) for t in tickers}

    def request_reconcile(self, tickers) -> int:
        """Thread-safe: queue tickers for an immediate in-place REST reconcile.

        External corruption detectors (XBOOK crossed-book, run.py) call this
        when a SERVED book is provably wrong — the existing _reconcile_worker
        drains the queue within ~0.2s and overwrites from authed REST with no
        dark window, exactly as it does after a seq gap. Returns how many were
        queued (0 if the reader isn't running — caller should treat that as
        a miss, not silently assume a cure is coming).
        """
        if not self.is_running():
            return 0
        with self._lock:
            n = 0
            for t in tickers:
                if t in self.books:
                    self._reconcile.add(t)
                    n += 1
        return n

    def set_event_callback(self, cb):
        self._event_cb = cb

    def _emit(self, et, **d):
        if self._event_cb:
            try:
                self._event_cb(et, d)
            except Exception:
                pass

    # ───────────────── reader thread internals ─────────────────
    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._run_task = self._loop.create_task(self._main())
        try:
            self._loop.run_until_complete(self._run_task)
        except asyncio.CancelledError:
            pass                              # clean stop() cancellation
        except Exception as e:
            if not self._stop.is_set():
                log.error("IsolatedWSBook reader thread died: %s: %s", type(e).__name__, e)
        finally:
            # Cancel + drain any coroutines still pending (e.g. the websockets
            # keepalive) so the loop closes without "task destroyed" warnings.
            try:
                leftovers = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for t in leftovers:
                    t.cancel()
                if leftovers:
                    self._loop.run_until_complete(asyncio.gather(*leftovers, return_exceptions=True))
            except Exception:
                pass
            try:
                self._loop.close()
            except Exception:
                pass

    async def _main(self):
        try:
            await asyncio.gather(self._connect_loop(), self._sub_reconciler(),
                                 self._reconcile_worker(), self._periodic_reconcile_loop())
        except asyncio.CancelledError:
            pass                              # propagated from stop() → unwind cleanly

    async def _periodic_reconcile_loop(self):
        """Safety net: every RECONCILE_PERIOD_S, queue every live book for an
        in-place REST reconcile. Guarantees any divergence — including drift that
        never produced a seq gap — self-heals within the interval, without a dark
        window. Staggered so we don't REST-fetch every ticker at once."""
        while not self._stop.is_set():
            await asyncio.sleep(RECONCILE_PERIOD_S)
            with self._lock:
                live = [t for t, b in self.books.items() if b.live]
            n = max(1, len(live))
            spacing = min(RECONCILE_PERIOD_S / n, 1.0)
            for t in live:
                if self._stop.is_set():
                    break
                with self._lock:
                    self._reconcile.add(t)
                await asyncio.sleep(spacing)

    async def _connect_loop(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_and_listen()
                backoff = 1.0
            except Exception as e:
                self.stats["reconnects"] += 1
                self._emit("conn_drop", err=str(e)[:120])
                log.warning("IsolatedWSBook conn lost: %s: %s — backoff %.1fs",
                            type(e).__name__, str(e)[:80], backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, RECONNECT_CAP_S)

    async def _connect_and_listen(self):
        hdr = self.auth.get_headers("GET", WS_SIGNING_PATH)
        async with websockets.connect(WS_URL, additional_headers=hdr,
                                      ping_interval=10, ping_timeout=10,
                                      max_queue=MAX_QUEUE) as ws:
            self._ws = ws
            self._conn_seq = None
            self._subscribed.clear()
            self._cmd_tickers.clear(); self._ticker_sid.clear(); self._sid_tickers.clear()
            with self._lock:
                for b in self.books.values():
                    b.live = False    # BUFFERING until snapshot; get_orderbook_fp returns None → REST fallback
                    b._pending.clear()   # discard stale pre-reconnect buffered deltas
            self._emit("conn_open")
            epoch_t0 = time.time()
            epoch_msgs0 = self.stats["msgs"]
            try:
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=45.0)
                    except asyncio.TimeoutError:
                        # Necropsy BEFORE teardown — the one line that shows
                        # whether the epoch was flowing and died (sid nuke /
                        # server stopped sending) or was born dead (subscribe
                        # rejected, maintenance window).
                        log.warning(
                            "IsolatedWSBook 45s DATA SILENCE necropsy: "
                            "epoch_age=%.0fs epoch_msgs=%d subscribed=%d "
                            "sids=%d pending_acks=%d — tearing down",
                            time.time() - epoch_t0,
                            self.stats["msgs"] - epoch_msgs0,
                            len(self._subscribed), len(self._sid_tickers),
                            len(self._cmd_tickers))
                        raise
                    self._handle(json.loads(raw))
            finally:
                self._ws = None

    async def _send(self, msg):
        if self._ws is not None:
            await self._ws.send(json.dumps(msg))

    async def _sub_reconciler(self):
        """On the reader loop: keep the live subscription set == target set."""
        while not self._stop.is_set():
            await asyncio.sleep(SUB_RECONCILE_S)
            if self._ws is None:
                continue
            await self._sync_subscriptions()

    async def _sync_subscriptions(self):
        """One target-vs-subscribed reconcile pass (split out so tests can
        drive it directly).

        SUBSCRIBE in chunks of SUB_CHUNK_N: Kalshi returns ONE sid per
        subscribe COMMAND, and unsubscribe works only by sid
        (`market_tickers` unsubscribe is rejected with code 4 — see
        ws_delta_book._send_unsubscribe). Chunking bounds any sid's blast
        radius to SUB_CHUNK_N books.

        UNSUBSCRIBE a sid only when its LAST covered ticker leaves — the
        main feed's rule. The pre-2026-08-06 code unsubscribed every sid a
        dropped ticker touched; with the whole universe under one sid, any
        single removal (settled market, phantom map) silently killed data
        flow for ALL books while pings stayed healthy: total silence, the
        45s recv timeout, teardown — ~100 times/day, and every frozen-book
        window served stale asks into the arber (IOC PHANTOM family). A
        dropped ticker whose sid still covers wanted tickers keeps streaming
        deltas; set_tickers already popped its book, so _handle no-ops them.
        """
        with self._lock:
            target = set(self._target)
        add = target - self._subscribed
        drop = self._subscribed - target
        if add:
            add_list = sorted(add)
            for i in range(0, len(add_list), SUB_CHUNK_N):
                chunk = add_list[i:i + SUB_CHUNK_N]
                self._cmd_id += 1
                self._cmd_tickers[self._cmd_id] = list(chunk)
                self._subscribed |= set(chunk)
                self.stats["sub_cmds"] = self.stats.get("sub_cmds", 0) + 1
                await self._send({"id": self._cmd_id, "cmd": "subscribe",
                                  "params": {"channels": ["orderbook_delta"],
                                             "market_tickers": [ticker_aliases.resolve(t) for t in chunk]}})
        if drop:
            self._subscribed -= drop
            empty_sids = []
            for t in drop:
                sid = self._ticker_sid.pop(t, None)
                if sid is None:
                    continue
                covered = self._sid_tickers.get(sid)
                if covered is not None:
                    covered.discard(t)
                    if not covered:
                        self._sid_tickers.pop(sid, None)
                        empty_sids.append(sid)
            if empty_sids:
                # NEVER actually unsubscribe — observed live 2026-08-06 18:09:
                # Kalshi acks every subscribe after the first with type="ok",
                # SAME sid=1, cumulative ticker list. i.e. the server MERGES
                # all subscribes on a connection into ONE channel subscription.
                # Unsubscribing "our" sid therefore kills EVERY book on the
                # connection (the original nuke, fully explained). There is no
                # safe per-ticker removal primitive here; dropped tickers just
                # go quiet naturally (settled markets stop trading) and the
                # daily maintenance reconnect resets the server-side set.
                self.stats["sid_emptied_not_unsubscribed"] = (
                    self.stats.get("sid_emptied_not_unsubscribed", 0) + len(empty_sids))
                log.info("IsolatedWSBook sid(s) %s fully emptied — NOT unsubscribing "
                         "(server merges all subs into one sid; an unsubscribe would "
                         "kill every book)", empty_sids)

    async def _reconcile_worker(self):
        """After a seq gap: pull fresh REST for affected books and overwrite in
        place (no blanking → no dark window)."""
        while not self._stop.is_set():
            await asyncio.sleep(0.2)
            with self._lock:
                q = list(self._reconcile); self._reconcile.clear()
            for t in q:
                try:
                    await self._reconcile_one(t)
                except Exception as e:
                    # Was `pass` until 2026-08-06 — a silent fallback that hid
                    # every reconcile failure mode. Count always, log sampled.
                    n = self.stats.get("reconcile_worker_error", 0) + 1
                    self.stats["reconcile_worker_error"] = n
                    if n <= 5 or n % 100 == 0:
                        log.warning("IsolatedWSBook reconcile worker error for %s "
                                    "(n=%d): %s: %s", t, n,
                                    type(e).__name__, str(e)[:120])

    async def _reconcile_one(self, t):
        """One in-place REST reconcile, race-protected.

        The reader loop keeps applying live deltas to this book WHILE the REST
        fetch is awaited in a thread. Those deltas are buffered (see _handle)
        and REPLAYED on top of the snapshot after the overwrite. Discarding
        them instead — the pre-2026-07-31 behavior — turned every reconcile of
        a busy book into a coin-flip for a stuck phantom level (a discarded
        removal resurrects from the snapshot), which is what crossed the books.
        Over-application from replay is the safe direction: a double-applied
        removal floors at 0 (pop), a double-applied add inflates qty at a REAL
        price level until the next touch/reconcile — a depth error, never a
        fake price.
        """
        with self._lock:
            b = self.books.get(t)
            if b is None:
                return
            b.reconciling = True
            b._rbuf = []
        fetch_err = None
        try:
            real = ticker_aliases.resolve(t)
            ob = (await asyncio.to_thread(self.client.get_orderbook, real) or {}).get("orderbook_fp") or {}
        except Exception as e:
            ob = None
            fetch_err = e
            # Counted separately from truly-empty books (below) — conflating
            # them made 48% "empty" reconciles undiagnosable. Log sampled;
            # stats appear in the shadow-health beacon automatically.
            n = self.stats.get("reconcile_fetch_error", 0) + 1
            self.stats["reconcile_fetch_error"] = n
            if n <= 5 or n % 100 == 0:
                log.warning("IsolatedWSBook reconcile fetch error for %s (n=%d): "
                            "%s: %s", t, n, type(e).__name__, str(e)[:120])
        with self._lock:
            b = self.books.get(t)
            if b is None:
                return
            try:
                # A failed/404 fetch arrives as {}/None and is NOT a snapshot.
                # Reconciling it would set live=True on an empty book (perma
                # `empty_live` miss). Leave the book untouched — the live
                # deltas were already applied directly, so NO replay either.
                if not ob or (not ob.get("yes_dollars") and not ob.get("no_dollars")):
                    if fetch_err is None:
                        self.stats["reconcile_empty"] = self.stats.get("reconcile_empty", 0) + 1
                    return
                if not b.reconciling:
                    # A wire snapshot landed mid-fetch (reconnect) — it is
                    # fresher than this REST payload; keep it.
                    return
                b.rest_reconcile(ob)
                if b._rbuf:
                    for dmsg in b._rbuf:
                        b.apply_delta(dmsg)
                    self.stats["reconcile_replays"] = (
                        self.stats.get("reconcile_replays", 0) + len(b._rbuf))
                self.stats["reconciles"] += 1
            finally:
                b.reconciling = False
                b._rbuf = []

    def _handle(self, m):
        self.stats["msgs"] += 1
        seq = m.get("seq")
        typ = m.get("type") or ""
        payload = m.get("msg") or {}
        seq_i = None
        if seq is not None:
            try:
                seq_i = int(seq)
            except (ValueError, TypeError):
                seq_i = None
        # ── strict per-connection seq integrity ──
        if seq_i is not None:
            if self._conn_seq is not None and seq_i != self._conn_seq + 1:
                self.stats["gaps"] += 1
                self._emit("seq_gap", prev=self._conn_seq, new=seq_i)
                with self._lock:
                    self._reconcile.update(self._subscribed)   # REST-reconcile, don't blank
            self._conn_seq = seq_i

        if typ == "subscribed":
            cid = m.get("id"); sid = payload.get("sid")
            tks = self._cmd_tickers.pop(cid, None) if cid is not None else None
            if sid is not None and tks:
                keep = {t for t in tks if t in self._subscribed}
                if keep:
                    self._sid_tickers[sid] = keep
                    for t in keep:
                        self._ticker_sid[t] = sid
            return

        if typ == "orderbook_snapshot":
            t = ticker_aliases.reverse(payload.get("market_ticker") or "")
            with self._lock:
                b = self.books.get(t)
                if b is not None:
                    b.apply_snapshot(payload, seq_i)
            return

        if typ == "orderbook_delta":
            t = ticker_aliases.reverse(payload.get("market_ticker") or "")
            with self._lock:
                b = self.books.get(t)
                if b is not None:
                    if b.live:
                        b.apply_delta(payload)
                        if b.reconciling:
                            # REST fetch in flight: this delta will be wiped by
                            # the overwrite — buffer it for replay on top.
                            b._rbuf.append(payload)
                    else:
                        b.buffer_delta(seq_i, payload)   # replayed when the snapshot lands
            return

        # Anything else — most importantly server `error` frames — was
        # silently dropped until 2026-08-06, which hid subscribe failures
        # completely. Errors log verbatim at WARNING; other unknown types
        # (unsubscribe acks etc.) log the first few then just count.
        if typ == "error":
            self.stats["error_frames"] = self.stats.get("error_frames", 0) + 1
            log.warning("IsolatedWSBook ERROR frame: %s", json.dumps(m)[:400])
        else:
            k = f"unhandled_{typ or 'untyped'}"
            n = self.stats.get(k, 0) + 1
            self.stats[k] = n
            if n <= 5:
                log.info("IsolatedWSBook unhandled frame type=%r (n=%d): %s",
                         typ, n, json.dumps(m)[:300])
