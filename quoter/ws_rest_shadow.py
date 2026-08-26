"""Phase 0 — WS-vs-REST orderbook divergence shadow logger.

PURPOSE
    Measure how far our WebSocket delta book drifts from REST truth in
    production, WITHOUT changing any trading behavior. This is the baseline /
    before-after yardstick for the WS accuracy work (see WS_ACCURACY_PLAN.md).

ZERO-RISK BY CONSTRUCTION
    * Read-only. Never mutates a book, never places/cancels an order, never
      touches a bot decision.
    * Runs in its OWN daemon thread (like OrderbookPoller / TradeLogger), so it
      adds nothing to the trading event loop.
    * Makes NO new REST calls. It reads the REST book from the registry the
      OrderbookPoller already maintains.
    * INERT unless env WS_REST_SHADOW_ENABLE=1 is set. start() is a no-op
      otherwise.
    * Every read is exception-guarded — a cross-thread race that raises (e.g.
      a book mutated mid-read) just skips that sample; it can never affect
      trading.

OUTPUT
    Appends one row per (active ticker) per tick to ws_rest_divergence.csv, and
    emits a WARNING on material divergence so it's greppable in run.log:
        WS-REST DIVERGENCE | ticker=... ws_yes_ask=.. rest_yes_ask=.. diff=..c dir=WS_OPTIMISTIC
    Directions:
        WS_OPTIMISTIC  — WS ask cheaper than REST  → phantom / false-positive risk
        WS_PESSIMISTIC — WS ask worse than REST     → missed-edge / false-negative risk
        WS_DARK        — WS has no book while REST does (stale/dropped)
"""
import os
import csv
import time
import threading
import logging
from collections import Counter
from typing import Optional, Callable

log = logging.getLogger("ws_rest_shadow")

_REPO = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_REPO, "ws_rest_divergence.csv")

# Live enable gate. The logger thread is ALWAYS armed but idle; it logs only
# while enabled. Enable without a restart by creating this flag file
# (`touch enable_ws_rest_shadow.flag`); disable by removing it. The env var
# WS_REST_SHADOW_ENABLE=1 also enables it (for boot-time-on). Checking a file
# once per interval in a daemon thread is negligible and read-only.
ENABLE_FLAG_PATH = os.path.join(_REPO, "enable_ws_rest_shadow.flag")

DEFAULT_INTERVAL_S = 1.0
DIVERGENCE_TOL_C = 1          # |ws_ask - rest_ask| >= this (cents) → flag
REST_FRESH_MAX_AGE_S = 3.0    # only judge divergence when the REST oracle is fresh


def _cents(price_str) -> int:
    return int(round(float(price_str) * 100))


def _asks_from_inner(inner: dict) -> Optional[dict]:
    """Given the inner {"yes_dollars":[[price,qty]...], "no_dollars":[...]} dict,
    return best asks (cents) + depth. yes_dollars=YES bids, no_dollars=NO bids;
    YES ask = 100 - best NO bid, NO ask = 100 - best YES bid. Returns None on a
    malformed/empty payload."""
    if not inner:
        return None
    yes_levels = inner.get("yes_dollars") or []
    no_levels = inner.get("no_dollars") or []
    best_yes_bid = max((_cents(l[0]) for l in yes_levels if len(l) >= 2), default=0)
    best_no_bid = max((_cents(l[0]) for l in no_levels if len(l) >= 2), default=0)
    return {
        "yes_ask": (100 - best_no_bid) if best_no_bid > 0 else 0,
        "no_ask": (100 - best_yes_bid) if best_yes_bid > 0 else 0,
        "yes_bid_depth": sum(float(l[1]) for l in yes_levels if len(l) >= 2),
        "no_bid_depth": sum(float(l[1]) for l in no_levels if len(l) >= 2),
    }


class WSRestShadowLogger:
    """Compares each active ticker's WS delta book against the REST registry
    snapshot and logs divergence. Daemon-thread, read-only, opt-in."""

    def __init__(
        self,
        delta_managers: list,          # [ws_delta_mgr, map_delta_mgr, ...] (any may be None)
        registry,                      # OrderbookRegistry (REST poller's)
        ticker_provider: Callable[[], set],   # returns active synthetic tickers
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self._mgrs = [m for m in delta_managers if m is not None]
        self._registry = registry
        self._ticker_provider = ticker_provider
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._writer = None
        self._fh = None
        # Lean-logging state: only record actionable verdicts, de-duped per
        # ticker, with a periodic aggregate summary for the denominator.
        self._last_written: dict = {}      # ticker -> last-written signature
        self._summary = Counter()          # verdict-family counts since last summary
        self._last_summary_ts = 0.0
        self._SUMMARY_EVERY_S = 60.0
        self._WARN_MIN_C = 3               # only WARN on divergences this big (cuts REST-lag noise)

    # ── lifecycle ────────────────────────────────────────────────────────
    def _enabled(self) -> bool:
        """Enabled if the env var is set OR the flag file exists. Checked live
        each tick, so logging can be toggled without a restart."""
        if os.environ.get("WS_REST_SHADOW_ENABLE", "").strip():
            return True
        try:
            return os.path.exists(ENABLE_FLAG_PATH)
        except Exception:
            return False

    def start(self) -> None:
        if self._registry is None or not self._mgrs:
            log.warning("ws_rest_shadow: no registry or delta managers — not starting.")
            return
        self._thread = threading.Thread(target=self._run, name="ws_rest_shadow", daemon=True)
        self._thread.start()
        log.warning("ws_rest_shadow ARMED (idle). Enable live WITHOUT a restart: "
                    "`touch %s` (remove it to stop). Read-only, zero new REST calls.",
                    ENABLE_FLAG_PATH)

    def stop(self) -> None:
        self._stop.set()

    def _ensure_csv(self) -> None:
        if self._writer is not None:
            return
        new_file = not os.path.exists(CSV_PATH)
        self._fh = open(CSV_PATH, "a", newline="", buffering=1)
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow([
                "ts_utc", "ticker", "ws_present", "rest_present", "rest_age_s",
                "ws_yes_ask", "rest_yes_ask", "yes_ask_diff",
                "ws_no_ask", "rest_no_ask", "no_ask_diff",
                "ws_yes_depth", "ws_no_depth", "verdict",
            ])
            self._fh.flush()

    # ── worker ───────────────────────────────────────────────────────────
    def _run(self) -> None:
        was_enabled = False
        while not self._stop.is_set():
            try:
                enabled = self._enabled()
                if enabled and not was_enabled:
                    self._ensure_csv()
                    self._last_summary_ts = time.time()   # start the summary window now
                    log.warning("ws_rest_shadow: logging STARTED → %s (interval=%.1fs, tol=%dc).",
                                CSV_PATH, self._interval_s, DIVERGENCE_TOL_C)
                elif not enabled and was_enabled:
                    log.warning("ws_rest_shadow: logging STOPPED (flag cleared).")
                was_enabled = enabled
                if enabled and self._writer is not None:
                    self._tick()
            except Exception as e:
                log.warning("ws_rest_shadow loop error (ignored): %s", e)
            self._stop.wait(self._interval_s)

    def _read_ws(self, ticker: str) -> Optional[dict]:
        """Read the WS delta book fp for a ticker from whichever manager has it.
        Exception-guarded: a mid-read mutation race just yields None (skip)."""
        for m in self._mgrs:
            try:
                fp = m.get_orderbook_fp(ticker)
            except Exception:
                fp = None
            if fp:
                inner = fp.get("orderbook_fp") if "orderbook_fp" in fp else fp
                return _asks_from_inner(inner)
        return None

    def _read_rest(self, ticker: str):
        try:
            rb = self._registry.get(ticker)
        except Exception:
            return None
        if rb is None:
            return None
        try:
            inner = (rb.to_ws_style() or {}).get("orderbook_fp") or {}
            asks = _asks_from_inner(inner)
            age = rb.age_sec
        except Exception:
            return None
        if asks is None:
            return None
        asks["age_s"] = age
        return asks

    def _tick(self) -> None:
        try:
            tickers = set(self._ticker_provider() or set())
        except Exception:
            tickers = set()
        # Also fold in every ticker the delta managers are actively tracking, so
        # a WS-tracked ticker absent from the bot list is still measured.
        for m in self._mgrs:
            try:
                tickers |= set(getattr(m, "books", {}).keys())
            except Exception:
                pass
        ts = _utc_now_iso()
        active_now = set()
        for t in sorted(tickers):
            ws = self._read_ws(t)
            rest = self._read_rest(t)
            ws_present = ws is not None
            rest_present = rest is not None
            verdict = self._verdict(ws, rest, ws_present, rest_present)
            self._summary[verdict.split("(")[0]] += 1

            # Only actionable verdicts are recorded. Empty books (BOTH_MISSING),
            # unjudgeable states (REST_MISSING / REST_STALE) and agreements (OK)
            # are counted for the once-a-minute summary but NOT written per-row —
            # that's the bulk of the volume and none of the signal.
            if not (verdict.startswith("DIVERGE") or verdict == "WS_DARK"):
                continue
            active_now.add(t)

            ws_ya = ws["yes_ask"] if ws_present else ""
            ws_na = ws["no_ask"] if ws_present else ""
            r_ya = rest["yes_ask"] if rest_present else ""
            r_na = rest["no_ask"] if rest_present else ""
            # De-dupe: only write when this ticker's state CHANGES. A dark or
            # steadily-diverging ticker otherwise repeats an identical row every
            # tick — pure noise. State transitions and value changes still log.
            sig = (verdict.split("(")[0], ws_ya, r_ya, ws_na, r_na)
            if self._last_written.get(t) == sig:
                continue
            self._last_written[t] = sig

            rest_age = round(rest["age_s"], 3) if rest_present else ""
            ya_diff = (ws["yes_ask"] - rest["yes_ask"]) if (ws_present and rest_present) else ""
            na_diff = (ws["no_ask"] - rest["no_ask"]) if (ws_present and rest_present) else ""
            self._writer.writerow([
                ts, t, ws_present, rest_present, rest_age,
                ws_ya, r_ya, ya_diff, ws_na, r_na, na_diff,
                round(ws["yes_bid_depth"], 1) if ws_present else "",
                round(ws["no_bid_depth"], 1) if ws_present else "",
                verdict,
            ])
            # WARN only on MATERIAL divergences (>= WARN_MIN_C). WS_DARK and
            # small (likely REST-lag) diffs go to the CSV silently — no run.log spam.
            worst = max(abs(ya_diff) if isinstance(ya_diff, int) else 0,
                        abs(na_diff) if isinstance(na_diff, int) else 0)
            if verdict.startswith("DIVERGE") and worst >= self._WARN_MIN_C:
                log.warning("WS-REST DIVERGENCE | ticker=%s ws_ya=%s rest_ya=%s ydiff=%sc "
                            "ws_na=%s rest_na=%s ndiff=%sc rest_age=%ss | %s",
                            t, ws_ya, r_ya, ya_diff, ws_na, r_na, na_diff, rest_age, verdict)

        # Forget de-dupe state for tickers no longer actionable, so a fresh
        # DIVERGE/WS_DARK on them later writes a new row (transition captured).
        for t in list(self._last_written):
            if t not in active_now:
                del self._last_written[t]

        # Periodic one-line summary → keeps the denominator/coverage cheaply.
        now = time.time()
        if now - self._last_summary_ts >= self._SUMMARY_EVERY_S:
            self._last_summary_ts = now
            counts = dict(self._summary)
            self._summary.clear()
            log.info("ws_rest_shadow SUMMARY (last %ds): %s", int(self._SUMMARY_EVERY_S),
                     " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(no tickers)")

    def _verdict(self, ws, rest, ws_present, rest_present) -> str:
        if not rest_present:
            return "REST_MISSING" if ws_present else "BOTH_MISSING"
        if rest["age_s"] > REST_FRESH_MAX_AGE_S:
            return "REST_STALE"          # oracle too old to judge
        if not ws_present:
            return "WS_DARK"             # WS has no book while REST does
        # Both present + REST fresh → judge the ask on each side.
        worst = 0
        direction = ""
        for side in ("yes_ask", "no_ask"):
            if ws[side] > 0 and rest[side] > 0:
                d = ws[side] - rest[side]
                if abs(d) > abs(worst):
                    worst = d
                    # ws ask < rest ask → WS thinks it's cheaper to buy → optimistic
                    direction = "WS_OPTIMISTIC" if d < 0 else "WS_PESSIMISTIC"
        if abs(worst) >= DIVERGENCE_TOL_C:
            return f"DIVERGE({direction},{worst:+d}c)"
        return "OK"


def _utc_now_iso() -> str:
    # time.gmtime avoids the Date.now-style hazards; good enough for a log ts.
    t = time.time()
    lt = time.gmtime(t)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f".{int((t % 1) * 1000):03d}Z"
