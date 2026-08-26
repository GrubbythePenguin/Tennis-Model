"""Write priority gate — see WRITE_PRIORITY_GATE_PLAN.md (scoped 2026-08-02).

All exchange writes serialize through one slot (kalshi_client's inflight lock
+ rate budget). Today that slot is FIFO, so taker fires queue behind quote
housekeeping: measured 2026-08-02, fires ran 95ms with 1 concurrent amend vs
308ms with 8, worst 1161ms — and zero-fills ran 60-100ms slower than fills.

This gate reorders WHO takes the next slot; it never changes HOW MANY writes
happen (the client's rate limiter is untouched downstream).

Classes (operator-decided):
  P0_FIRE   — taker IOCs (arber/momentum fires, hedger — hedger orders are
              IOC by construction) + post-fire crossing-cancels. Delay-safe:
              an IOC can't rest; late = missed capture, never bad exposure.
  P1_CANCEL — all cancels. Delay-safe: a late cancel is merely conservative.
  ADMISSION — resting posts and amends are NEVER QUEUED. `try_slot` grants
              only when the gate is completely idle; declined writes are
              DISCARDED and re-derived by the caller's next tick, so a stale
              price has no code path to the wire (operator invariant).

Flag file `write_gate.flag` (house style): absent → every call degrades to
passthrough (no serialization here; the client's internal lock keeps today's
FIFO behavior exactly). Delete the flag to fall back live.

Maker-dark accounting (operator-decided): a ticker whose posts keep getting
declined logs [MAKER DARK WARN] at 5s and [MAKER DARK ALERT] at 15s, reset by
any admitted post for that ticker.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)

P0_FIRE = 0
P1_CANCEL = 1

_DIR = os.path.dirname(os.path.abspath(__file__))
FLAG_PATH = os.path.join(_DIR, "write_gate.flag")

MAKER_DARK_WARN_S = 5.0
MAKER_DARK_ALERT_S = 15.0
_DARK_LOG_EVERY_S = 5.0
_SUMMARY_EVERY_S = 30.0


class _Slot:
    """Async context manager for one occupancy of the write slot.

    Records t_enqueue/t_grant/t_release so callers can log queue-wait and
    wire-time. `owns` is False in passthrough mode (flag off) — then exit
    must not touch gate state.
    """

    __slots__ = ("gate", "priority", "tag", "owns", "t_enqueue", "t_grant",
                 "t_release")

    def __init__(self, gate: "WriteGate", priority: Optional[int], tag: str,
                 owns: bool, t_enqueue: float, t_grant: Optional[float]):
        self.gate = gate
        self.priority = priority
        self.tag = tag
        self.owns = owns
        self.t_enqueue = t_enqueue
        self.t_grant = t_grant
        self.t_release = None

    @property
    def queue_wait_s(self) -> float:
        return (self.t_grant - self.t_enqueue) if self.t_grant else 0.0

    @property
    def wire_s(self) -> float:
        if self.t_grant is None or self.t_release is None:
            return 0.0
        return self.t_release - self.t_grant

    async def __aenter__(self):
        if self.t_grant is None:                 # blocking classes (P0/P1)
            self.t_grant = await self.gate._acquire(self)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.t_release = time.time()
        if self.owns:
            self.gate._record(self)
            self.gate._release()
        return False


class WriteGate:
    def __init__(self, flag_path: str = FLAG_PATH):
        self._flag_path = flag_path
        self._flag_cached = False
        self._flag_checked_ts = 0.0
        self._busy = False
        self._waiters: list = []                 # heap: (priority, seq, Future, slot)
        self._seq = itertools.count()
        self.stats = {"grant_p0": 0, "grant_p1": 0, "grant_admit": 0,
                      "post_declined": 0, "amend_declined": 0}
        self._wait_ms: dict = {P0_FIRE: [], P1_CANCEL: []}
        self._last_summary = time.time()
        self._dark_since: dict = {}              # synthetic ticker -> first decline ts
        self._dark_last_log: dict = {}

    # ── flag ──
    def enabled(self) -> bool:
        now = time.time()
        if now - self._flag_checked_ts > 2.0:
            self._flag_checked_ts = now
            try:
                self._flag_cached = os.path.exists(self._flag_path)
            except Exception:
                self._flag_cached = False
        return self._flag_cached

    # ── blocking classes ──
    def slot(self, priority: int, tag: str = "") -> _Slot:
        """P0/P1 acquisition: `async with gate.slot(P0_FIRE): ...`"""
        t0 = time.time()
        if not self.enabled():
            return _Slot(self, priority, tag, owns=False, t_enqueue=t0, t_grant=t0)
        return _Slot(self, priority, tag, owns=True, t_enqueue=t0, t_grant=None)

    async def _acquire(self, slot: _Slot) -> float:
        if not self._busy and not self._waiters:
            self._busy = True
            return time.time()
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (slot.priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            # If the slot was already transferred to us, hand it onward —
            # otherwise it leaks busy forever (manager cancels batches).
            if fut.done() and not fut.cancelled():
                self._release()
            raise
        return time.time()

    def _release(self) -> None:
        # transfer to the best waiter, skipping cancelled futures (loop, not
        # recursion — a burst of cancelled waiters must not blow the stack)
        while self._waiters:
            _prio, _seq, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)             # slot stays busy, transferred
                return
        self._busy = False

    # ── admission (posts/amends — never queued) ──
    def try_slot(self, tag: str = "") -> Optional[_Slot]:
        t0 = time.time()
        if not self.enabled():
            return _Slot(self, None, tag, owns=False, t_enqueue=t0, t_grant=t0)
        if self._busy or self._waiters:
            return None
        self._busy = True
        self.stats["grant_admit"] += 1
        return _Slot(self, None, tag, owns=True, t_enqueue=t0, t_grant=t0)

    # ── maker-dark accounting ──
    def note_post_declined(self, ticker: str) -> None:
        self.stats["post_declined"] += 1
        now = time.time()
        first = self._dark_since.setdefault(ticker, now)
        dark = now - first
        if dark >= MAKER_DARK_ALERT_S:
            lvl, tag = log.error, "ALERT"
        elif dark >= MAKER_DARK_WARN_S:
            lvl, tag = log.warning, "WARN"
        else:
            return
        if now - self._dark_last_log.get(ticker, 0.0) >= _DARK_LOG_EVERY_S:
            self._dark_last_log[ticker] = now
            lvl(f"[MAKER DARK {tag}] {ticker}: posts declined for {dark:.1f}s "
                f"(write-gate contention)")

    def note_post_admitted(self, ticker: str) -> None:
        self._dark_since.pop(ticker, None)
        self._dark_last_log.pop(ticker, None)

    # ── stats ──
    def _record(self, slot: _Slot) -> None:
        if slot.priority == P0_FIRE:
            self.stats["grant_p0"] += 1
            self._wait_ms[P0_FIRE].append(slot.queue_wait_s * 1000.0)
        elif slot.priority == P1_CANCEL:
            self.stats["grant_p1"] += 1
            self._wait_ms[P1_CANCEL].append(slot.queue_wait_s * 1000.0)
        now = time.time()
        if now - self._last_summary >= _SUMMARY_EVERY_S:
            self._last_summary = now
            self._emit_summary()

    def _emit_summary(self) -> None:
        def p95(v):
            if not v:
                return 0.0
            s = sorted(v)
            return s[min(len(s) - 1, int(0.95 * len(s)))]
        log.info(f"[WRITE GATE] p0={self.stats['grant_p0']} "
                 f"(p95 wait {p95(self._wait_ms[P0_FIRE]):.0f}ms) "
                 f"p1={self.stats['grant_p1']} "
                 f"(p95 wait {p95(self._wait_ms[P1_CANCEL]):.0f}ms) "
                 f"admits={self.stats['grant_admit']} "
                 f"post_declines={self.stats['post_declined']} "
                 f"amend_declines={self.stats['amend_declined']}")
        self._wait_ms = {P0_FIRE: [], P1_CANCEL: []}


# Module-level singleton — execution.py and tests share one gate.
GATE = WriteGate()
