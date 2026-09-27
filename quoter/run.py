import asyncio
import json
import os
import sys
import logging
import time as _time_mod
from dotenv import load_dotenv

sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
from kalshi_auth import KalshiAuth
from kalshi_client import KalshiClient

from manager import MarketManager
from theos import HybridTheoGenerator
from trade_logger import TradeLogger
from poly_price_feed import PolyMapPriceFeed
from ws_orderbook_fetcher import WSOrderbookFetcher
from orderbook_poller import OrderbookPoller, OrderbookRegistry
from book_view import (WSBookView, RestBookView, RouterBookView, WSDeltaBookView,
                       IsolatedBookView)
from map_delta_book import MapWSDeltaManager
# WS market-status probe — replaces the 300s bulk /markets?tickers= status
# sanity check at the per-tick refresh. Same module powering populate_configs
# after the 2026-06-24 migration. run.py uses the async-native entrypoint
# because the per-tick refresh runs inside the bot's asyncio event loop; the
# sync `probe_markets` wrapper calls `asyncio.run()` which raises
# RuntimeError when invoked from a running loop.
from ws_market_status import probe_markets_async as _ws_probe_markets_async

# ─── Persistent run.py metadata cache ─────────────────────────────────────
# Caches the fields the WS orderbook payload does NOT carry: yes_sub_title,
# title, rules_primary (static for a ticker's lifetime) + last_price_dollars
# and result (refreshed on REST fallback events, otherwise served stale —
# same staleness pattern as the prior 300s bulk REST cadence). Lookup is
# per-ticker, so the cache persists naturally across run.py restarts.
_RUN_METADATA_CACHE_FILE = "_run_kalshi_metadata_cache.json"


def _load_run_metadata_cache(path: str = _RUN_METADATA_CACHE_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.loads(f.read())
    except Exception as e:
        logging.warning(f"[METADATA-CACHE] load failed: {type(e).__name__}: {e}")
        return {}


def _save_run_metadata_cache(cache: dict, path: str = _RUN_METADATA_CACHE_FILE) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(cache, sort_keys=True))
        os.replace(tmp, path)
    except Exception as e:
        logging.warning(f"[METADATA-CACHE] save failed: {type(e).__name__}: {e}")

# Ticker alias layer — translate synthetic↔real at REST API boundaries
# for runtime fallback paths (bulk fetch, BLIP truth-oracle, WS-miss
# REST fallback). Pass-through for non-aliased tickers.
import ticker_aliases
import ws_miss_stats

# Safely hook into the older independent MLB scraping framework natively 
sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_mlb_tracker")
try:
    from trading import mlb_game_state
    mlb_tracker = mlb_game_state.get_instance()
    logging.info("MLB Background Parser hydrated successfully!")
except ImportError:
    mlb_tracker = None
    logging.warning("MLB Background Parser could not be imported.")

try:
    import soccer_tracker
    mls_tracker = soccer_tracker.get_instance()
    logging.info("MLS Live Soccer Tracker hydrated successfully!")
except ImportError:
    mls_tracker = None
    logging.warning("MLS Tracker could not be imported.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', force=True)
log = logging.getLogger(__name__)

# ── DRY RUN MASTER TOGGLE ──
# True  = bot evaluates everything but sends NO orders to Kalshi.
#         All [DRY] log lines show what WOULD have been fired.
# False = LIVE TRADING. Orders fire to Kalshi.
# Flip to False when you're confident in the refactor and ready to go live.
DRY_RUN = False  # 26SEP18: LIVE for the set1 taker (500 lots, set1_taker_live.flag). All maker
                 # models remain shadow via their own flags — with no theos they rest nothing.
                 # 26SEP14 capture-only note kept for history: True = WS books only, zero orders.

def _load_force_start_patterns(path: str = "force_start_tickers.txt") -> list:
    """Read substring patterns that bypass the time gate.

    Format per line:
      <pattern>                       # always force-start (legacy)
      <pattern> @ 2026-05-04T03:00    # force-start only on/after this time (ET)

    Only patterns whose activation time has been reached are returned. Lets you
    schedule overrides ahead of time (e.g. before bed) without having probabilities
    captured at stale prices many hours pre-match.
    """
    if not os.path.exists(path):
        return []
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo as _ZI
        et = _ZI("America/New_York")
    except Exception:
        et = None
    now = _dt.now(et) if et else _dt.now()
    out = []
    try:
        with open(path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "@" in ln:
                    pattern, _, ts_str = ln.partition("@")
                    pattern = pattern.strip()
                    ts_str = ts_str.strip()
                    if not pattern:
                        continue
                    try:
                        d = _dt.fromisoformat(ts_str)
                        if d.tzinfo is None and et is not None:
                            d = d.replace(tzinfo=et)
                    except Exception:
                        continue  # bad datetime → fail safe (don't activate)
                    if now >= d:
                        out.append(pattern)
                else:
                    out.append(ln)
    except Exception:
        pass
    return out


def apply_map_books(results: dict, top_level_bids: dict, top_level_offers: dict,
                    dt_market_state: dict) -> set:
    """Write map orderbook snapshots into the shared top-of-book dicts.

    Returns the set of tickers that produced a REAL book (at least one price
    level on either side). A ticker present in `results` but with both sides
    empty is NOT considered populated — an empty book carries no price, and
    treating it as one is how stale/phantom prices used to leak in.

    Shared by the primary (delta-book) and rescue (fresh-WS) map fetches so
    both paths apply identical semantics.
    """
    populated = set()
    for t, ob in (results or {}).items():
        ob_data = (ob or {}).get("orderbook_fp", {}) or {}
        y = list(ob_data.get("yes_dollars", []) or [])
        n = list(ob_data.get("no_dollars", []) or [])
        y.sort(key=lambda x: float(x[0]), reverse=True)
        n.sort(key=lambda x: float(x[0]), reverse=True)
        if not (y or n):
            continue
        top_level_bids[t] = int(float(y[0][0]) * 100) if y else 0
        top_level_offers[t] = 100 - int(float(n[0][0]) * 100) if n else 100
        if t not in dt_market_state:
            dt_market_state[t] = {}
        dt_market_state[t]["raw_ob"] = ob
        # Parity with the Poly path (poly_price_feed.inject_map_prices): ensure a
        # status so the arber doesn't skip a freshly-priced map. setdefault, so a
        # real status already set from Kalshi metadata (e.g. "finalized") wins —
        # mirrors Poly's `if "status" not in ...` guard exactly.
        dt_market_state[t].setdefault("status", "active")
        populated.add(t)
    return populated


def scrub_unpopulated_maps(map_tickers, populated: set,
                           top_level_bids: dict, top_level_offers: dict) -> list:
    """EMPTY DATA MEANS NO TRADE.

    Force-absent every map ticker that no live source populated this cycle, so
    the downstream active-map guard suppresses instead of trading on whatever
    happened to be left in the dicts. Returns the scrubbed tickers for logging.

    This deliberately has NO metadata/cache fallback. The 300s /markets bulk
    fetch is a status sanity check, not trading data; using its yes_bid/yes_ask
    as a price is what kept Kalshi map prices up to 5 minutes stale between
    2026-06-13 and 2026-07-18 without anyone noticing.
    """
    scrubbed = []
    for t in map_tickers:
        if t in populated:
            continue
        if t in top_level_bids or t in top_level_offers:
            top_level_bids.pop(t, None)
            top_level_offers.pop(t, None)
        scrubbed.append(t)
    return scrubbed


# ─── Crossed-book shadow detectors (2026-07-31) ─────────────────────────────
# See CROSSED_BOOK_DETECTION_PLAN.md. SHADOW MODE: log only, no suppression, no
# REST trigger — enforcement decisions come after one slate of shadow data.
#
# Detector A — [XBOOK CROSSED]: yes_bid + no_bid > 100 on a single ticker
#   (equivalently ask < bid). Arithmetically impossible, so every sample is
#   feed corruption. Measured 126/27,696 arber-fire snapshots (0.45%) across
#   26 episodes on 2026-07-31; the BLIP jump-detector saw only 4 of the 26,
#   because a stale level that never gets removed produces no >=5c one-tick
#   jump for BLIP to notice.
# Detector B — [XBOOK ASYM]: complementary tickers' spreads differ > 6c.
#   Catches the mirror corruption (sides summing UNDER 100) that silently
#   hides real edge. Ambiguous (a book can be genuinely wide), so even after
#   shadow this must never hard-suppress — at most trigger a REST refresh.
# ── Enforcement (Detector A only; B never enforces) ──
# XBOOK_ENFORCE=1 in the environment turns detection into action:
#   1. Crossed ticker → mark untrusted, and while untrusted force-absent the
#      whole event pair (top_level_bids/offers AND raw_ob) each tick — "empty
#      data means no trade", the same invariant scrub_unpopulated_maps leans
#      on. Consumers see (0,100)/no-ladder and every fire path goes dead.
#      Known consequence: losing raw_ob can null the synthetic/hedge theo,
#      which cancels resting quotes via the existing theos-disappeared path in
#      manager.py (~L1119). That is accepted conservatism: a book that is
#      provably corrupt should not price ANYTHING, including our makers.
#   2. Kick a per-ticker WS reseed: WSOrderbookFetcher opens a FRESH
#      connection (a re-subscribe on the shared conn does NOT yield a new
#      snapshot — Kalshi just acks) and the snapshot is applied atomically to
#      the shared delta book via WSBook.apply_snapshot. All due tickers ride
#      ONE fetch_all call, so a multi-ticker storm costs one connection.
#   3. Trust restored the first tick the rebuilt book shows uncrossed.
XBOOK_ENFORCE = os.environ.get("XBOOK_ENFORCE", "0").strip() == "1"
# Runtime toggle: `touch xbook_enforce.flag` arms, `rm` disarms — takes effect
# next tick, NO restart. This is the kill-switch that matters when testing
# enforcement against live corruption. Env var = permanent arm (survives flag
# file deletion); either one arms.
XBOOK_ENFORCE_FLAG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "xbook_enforce.flag")


def _xbook_enforce_active() -> bool:
    return XBOOK_ENFORCE or os.path.exists(XBOOK_ENFORCE_FLAG_FILE)


_XBOOK_LOG_COOLDOWN_S = 2.0
_XBOOK_RESEED_COOLDOWN_S = 5.0    # per-ticker floor between reseed kicks
_XBOOK_UNTRUST_TTL_S = 600.0      # stuck-untrusted safety valve (re-arms if still crossed)
_XBOOK_STORM_N = 3                # simultaneous reseeds worth flagging loudly
# Restore debounce — THGX 2026-07-31 19:51 flapped 9 untrust/restore cycles in
# ~25s on a 1c cross that flickered per tick. Restoring on the FIRST clean tick
# turns a flickering book into episode churn; require a sustained clean streak.
_XBOOK_RESTORE_CLEAN_TICKS = 3    # consecutive uncrossed scans required...
_XBOOK_RESTORE_MIN_S = 3.0        # ...and at least this long untrusted
_xbook_last_log: dict = {}       # (kind, key) -> last log ts
_xbook_active: dict = {}         # ticker -> episode start ts (Detector A)
_xbook_untrusted: dict = {}      # ticker -> untrust start ts (enforce mode)
_xbook_clean_streak: dict = {}   # ticker -> consecutive uncrossed scans while untrusted
_xbook_reseed_last: dict = {}    # ticker -> last reseed kick ts
_xbook_reseed_task = None        # single in-flight batch guard
_xbook_enforce_prev = False      # for ARMED/DISARMED transition logging


async def _xbook_reseed_batch(tickers: list, ws_delta_mgr, fetcher) -> None:
    """Fetch fresh WS snapshots for the given (synthetic) tickers on a dedicated
    connection and apply them atomically into the shared delta book."""
    try:
        real_by_syn = {t: ticker_aliases.resolve(t) for t in tickers}
        res = await fetcher.fetch_all(list(real_by_syn.values())) or {}
        for syn, real in real_by_syn.items():
            ob = (res.get(real) or {}).get("orderbook_fp") or None
            book = (getattr(ws_delta_mgr, "books", None) or {}).get(syn)
            if not ob or book is None:
                log.warning(
                    f"[XBOOK RESEED-MISS] {syn} snapshot={'ok' if ob else 'NONE'} "
                    f"delta_book={'ok' if book else 'NONE'} — stays suppressed until uncrossed"
                )
                continue
            book.apply_snapshot(
                {"yes_dollars_fp": ob.get("yes_dollars") or [],
                 "no_dollars_fp": ob.get("no_dollars") or []},
                None,
            )
            nb = max(book.yes) if book.yes else 0
            na = (100 - max(book.no)) if book.no else 100
            log.warning(
                f"[XBOOK RESEED] {syn} fresh WS snapshot applied — "
                f"new top bid={nb} ask={na} (n_yes={len(book.yes)} n_no={len(book.no)})"
            )
    except Exception as _e:
        log.error(f"[XBOOK RESEED] batch error (ignored): {_e}")


def _xbook_ladder(dt_market_state: dict, t: str) -> str:
    """Top-5 levels per side (cents x qty) from raw_ob — the diagnosis data for
    WHY a phantom level exists. Empty string if no raw_ob (e.g. Poly-injected)."""
    try:
        ob = ((dt_market_state.get(t) or {}).get("raw_ob") or {}).get("orderbook_fp") or {}
        def _top(levels):
            lv = sorted(
                ((int(round(float(l[0]) * 100)), float(l[1])) for l in (levels or []) if l),
                reverse=True,
            )[:5]
            return ",".join(f"{p}x{q:g}" for p, q in lv)
        return f" | yes[{_top(ob.get('yes_dollars'))}] no[{_top(ob.get('no_dollars'))}]"
    except Exception:
        return ""


def _spawn_poly_ws_listener() -> None:
    """Start the Poly WS listener if it is not already running.

    The feed lives in its own process on purpose: a hung socket or a bug
    there must never stall the trading loop. Operators should still only
    have to start run.py, so we spawn it here.

    Safe to call unconditionally — the listener claims a pidfile singleton
    and a duplicate exits immediately, so restarting run.py while the old
    listener is alive does not create two processes fighting over the
    snapshot. Best-effort: any failure logs and trading continues on the
    unchanged REST path.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "_poly_ws_listener.py")
    if not os.path.exists(script):
        log.warning("[WS LISTENER] _poly_ws_listener.py not found — Poly "
                    "books will use the REST path")
        return
    try:
        out = open(os.path.join(here, "_poly_ws_listener.out"), "a")
        proc = subprocess.Popen([sys.executable, script], cwd=here,
                                stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        log.info(f"[WS LISTENER] spawned pid={proc.pid} "
                 f"(duplicate instances self-exit via pidfile)")
    except Exception as e:
        log.warning(f"[WS LISTENER] spawn failed ({e}) — Poly books will "
                    f"use the REST path")


_WS_SUPERVISE = {"last": 0.0}
# Recovery budget: the listener refreshes the snapshot ~1/s, so 45s of
# silence is unambiguous death. Worst case detection = DEAD_S + EVERY_S
# = 75s. A false positive is nearly free (the duplicate self-exits via the
# pidfile), so bias toward respawning early — the downside is log noise,
# the upside is not spending a tier-1 game on the REST fallback.
WS_SUPERVISE_EVERY_S = 30.0
WS_SNAPSHOT_DEAD_S = 45.0


def _supervise_poly_ws_listener() -> None:
    """Respawn the Poly WS listener if its snapshot has gone cold.

    26AUG04 19:16: the listener exited on its own byte cap and nothing
    restarted it — production ran on the REST fallback for 93 minutes.
    Spawning only at boot means any listener death needs a human.

    Cheap and safe to call every tick: throttled to once per
    WS_SUPERVISE_EVERY_S, only acts when the snapshot is older than
    WS_SNAPSHOT_DEAD_S, and _spawn_poly_ws_listener is idempotent (a
    duplicate self-exits via the pidfile singleton). Fully swallowed —
    supervision must never be able to break the trading loop.
    """
    try:
        if not os.path.exists(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ws_feed.flag")):
            return
        now = _time_mod.time()
        if now - _WS_SUPERVISE["last"] < WS_SUPERVISE_EVERY_S:
            return
        _WS_SUPERVISE["last"] = now
        snap = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_poly_ws_snapshot.json")
        age = now - os.path.getmtime(snap) if os.path.exists(snap) else 1e9
        if age < WS_SNAPSHOT_DEAD_S:
            return
        log.warning(f"[WS LISTENER] snapshot {age:.0f}s stale — respawning "
                    f"listener (production is on REST until it recovers)")
        _spawn_poly_ws_listener()
    except Exception as e:
        log.warning(f"[WS LISTENER] supervise failed ({e}) — ignored")


def xbook_shadow_scan(top_level_bids: dict, top_level_offers: dict,
                      dt_market_state: dict, tickers, *,
                      enforce: bool = False, ws_delta_mgr=None,
                      reseed_fetcher=None, isolated_book=None,
                      now: float = None) -> None:
    global _xbook_reseed_task
    if now is None:
        now = _time_mod.time()
    spreads = {}
    for t in tickers:
        b = top_level_bids.get(t)
        o = top_level_offers.get(t)
        if b is None or o is None or (b == 0 and o == 100):
            continue  # absent or no-data book — carries no price to corrupt
        spreads[t] = (b, o, o - b)

    # Detector A — per-ticker crossed book, tracked as episodes.
    for t, (b, o, sp) in spreads.items():
        if sp < 0:
            start = _xbook_active.setdefault(t, now)
            if now - _xbook_last_log.get(("A", t), 0.0) >= _XBOOK_LOG_COOLDOWN_S:
                _xbook_last_log[("A", t)] = now
                log.warning(
                    f"[XBOOK CROSSED] {t} bid={b} ask={o} cross={-sp}c "
                    f"dur={now - start:.1f}s{_xbook_ladder(dt_market_state, t)}"
                )
    for t in list(_xbook_active):
        if t not in spreads or spreads[t][2] >= 0:
            start = _xbook_active.pop(t)
            log.warning(f"[XBOOK CLEARED] {t} crossed for {now - start:.1f}s")

    # Detector B — complementary spread asymmetry, per event pair.
    by_ev: dict = {}
    for t in spreads:
        by_ev.setdefault(t.rsplit("-", 1)[0], []).append(t)
    for ev, pair in by_ev.items():
        if len(pair) != 2:
            continue
        t1, t2 = pair
        d = abs(spreads[t1][2] - spreads[t2][2])
        if d > 6 and now - _xbook_last_log.get(("B", ev), 0.0) >= _XBOOK_LOG_COOLDOWN_S:
            _xbook_last_log[("B", ev)] = now
            log.warning(
                f"[XBOOK ASYM] {ev} d={d}c "
                f"{t1.rsplit('-', 1)[1]}=({spreads[t1][0]},{spreads[t1][1]}) "
                f"{t2.rsplit('-', 1)[1]}=({spreads[t2][0]},{spreads[t2][1]})"
            )

    # ── Enforcement (Detector A only — see block comment above) ──
    global _xbook_enforce_prev
    if enforce != _xbook_enforce_prev:
        log.warning(f"[XBOOK ENFORCE] {'ARMED' if enforce else 'DISARMED'} "
                    f"(flag file: {XBOOK_ENFORCE_FLAG_FILE})")
        _xbook_enforce_prev = enforce
    if not enforce:
        if _xbook_untrusted:
            # Disarmed mid-episode: stop suppressing NOW and drop state so a
            # later re-arm can't act on stale untrust.
            log.warning(f"[XBOOK ENFORCE] clearing {len(_xbook_untrusted)} "
                        f"untrusted on disarm: {sorted(_xbook_untrusted)}")
            _xbook_untrusted.clear()
            _xbook_clean_streak.clear()
        return

    # Trust transitions, evaluated on the FRESHLY rebuilt book (the loop
    # repopulates the dicts from the WS sources every tick, so a book we
    # force-absented last tick reappears here in its current true state).
    for t, (b, o, sp) in spreads.items():
        if sp < 0 and t not in _xbook_untrusted:
            _xbook_untrusted[t] = now
            log.warning(
                f"[XBOOK UNTRUSTED] {t} bid={b} ask={o} — suppressing event "
                f"pair and reseeding until the book uncrosses"
            )
    for t in list(_xbook_untrusted):
        started = _xbook_untrusted[t]
        if t in spreads and spreads[t][2] >= 0:
            # Debounced restore: a flickering book (crossed on-and-off per
            # tick) must ride out ONE episode, not churn through many.
            streak = _xbook_clean_streak.get(t, 0) + 1
            _xbook_clean_streak[t] = streak
            if streak >= _XBOOK_RESTORE_CLEAN_TICKS and now - started >= _XBOOK_RESTORE_MIN_S:
                del _xbook_untrusted[t]
                _xbook_clean_streak.pop(t, None)
                log.warning(f"[XBOOK TRUST-RESTORED] {t} after {now - started:.1f}s")
        else:
            _xbook_clean_streak.pop(t, None)   # crossed again → streak resets
            if now - started > _XBOOK_UNTRUST_TTL_S:
                # Book never came back uncrossed (event ended / source gone).
                # Expire so state can't leak; still-crossed books re-arm next tick.
                del _xbook_untrusted[t]
                log.warning(f"[XBOOK UNTRUST-EXPIRED] {t} after {now - started:.0f}s")

    if not _xbook_untrusted:
        return

    # Force-absent every ticker of every untrusted event: top-of-book AND the
    # raw ladder, since the arber sweep and momentum walk raw_ob directly.
    evs = {t.rsplit("-", 1)[0] for t in _xbook_untrusted}
    popped = {}
    for tt in set(top_level_bids) | set(top_level_offers):
        ev = tt.rsplit("-", 1)[0]
        if ev in evs:
            top_level_bids.pop(tt, None)
            top_level_offers.pop(tt, None)
            (dt_market_state.get(tt) or {}).pop("raw_ob", None)
            popped.setdefault(ev, []).append(tt)
    for ev, tts in popped.items():
        if now - _xbook_last_log.get(("S", ev), 0.0) >= _XBOOK_RESEED_COOLDOWN_S:
            _xbook_last_log[("S", ev)] = now
            log.warning(f"[XBOOK SUPPRESS] {ev} force-absented {sorted(tts)}")

    # Kick the WS reseed for every untrusted ticker past its cooldown. One
    # batch task at a time; all due tickers share a single fresh connection.
    if ws_delta_mgr is None or reseed_fetcher is None:
        return
    if _xbook_reseed_task is not None and not _xbook_reseed_task.done():
        return
    due = [t for t in _xbook_untrusted
           if now - _xbook_reseed_last.get(t, 0.0) >= _XBOOK_RESEED_COOLDOWN_S]
    if not due:
        return
    for t in due:
        _xbook_reseed_last[t] = now
    if len(due) >= _XBOOK_STORM_N:
        log.warning(
            f"[XBOOK STORM] {len(due)} tickers crossed simultaneously "
            f"({sorted(due)}) — connection-level corruption suspected"
        )
    # CRITICAL — reseed the book that is actually SERVED. With
    # use_isolated_ws.flag present, production reads come from IsolatedWSBook,
    # and reseeding only ws_delta_mgr fixes the FALLBACK book while the served
    # corruption persists (observed live on ASTPAIN 2026-07-31 19:41: 13 clean
    # reseeds, zero effect, 67s crossed). The isolated book's own
    # _reconcile_worker cures it in-place from authed REST within ~0.2s.
    if isolated_book is not None:
        try:
            n = isolated_book.request_reconcile(due)
            log.warning(f"[XBOOK RECONCILE] queued {n}/{len(due)} on isolated book")
        except Exception as _ib_e:
            log.error(f"[XBOOK RECONCILE] isolated queue error (ignored): {_ib_e}")
    # Also reseed the shared (fallback) delta book via fresh-connection WS
    # snapshot, so a later flag flip can't resurface the same corruption.
    try:
        _xbook_reseed_task = asyncio.create_task(
            _xbook_reseed_batch(due, ws_delta_mgr, reseed_fetcher))
    except RuntimeError:
        pass  # no running loop (tests) — suppression still applied above


# ─── BLIP → forced-WS-reconnect governor (retuned 2026-07-27) ───────────────
# A single ticker hitting OVERRIDE_TRIGGER_COUNT REST-overrides in 10s forces a
# reconnect of the SHARED ws_delta connection, which empties EVERY book for
# ~35-45s (30s backoff at cap + re-subscribe + snapshot). Measured over 2.6h on
# 27 Jul: 168 forced reconnects (~60/hr, 43% of gaps sitting on the old 30s
# cooldown floor), 49,472 REST fallbacks, and 453 real resting orders pulled in
# cancel-all-no-post cycles against healthy REST books.
#
# The old 30s cooldown expired at almost exactly the moment books finished
# re-seeding, so each reconnect armed the next one. RESEED_GRACE_S stops the
# artifacts of our own reconnect from being counted; the cooldown is raised well
# clear of the measured recovery time as defence in depth.
# A series ticker showing NO yes bids at all (bid=0) while still quoting a real
# ask (offer<100) has lost — nobody will buy that side at any price. Held
# continuously for this long, the event is over and stops being fetched, without
# waiting for Kalshi to flip status to settled (observed lagging 5.5h+).
DECIDED_NO_YES_SECS = 900.0    # 15 min, per operator 2026-07-27

OVERRIDE_TRIGGER_COUNT = 3
RECONNECT_COOLDOWN_S = 120.0   # was 30.0 — shorter than book re-seed time
RESEED_GRACE_S = 45.0          # measured re-seed ~30s (2.87 ovr/s → 0.10/s); + margin

# How long a map ticker confirmed ABSENT from /markets stays unsubscribed before
# we re-probe. Not permanent: some formats list the decider map late, so a map
# that does not exist pre-game can appear once the series progresses. 15 min
# costs one bulk-fetch slot per phantom per quarter hour, versus a WS
# subscription plus a REST reconcile every 45s forever.
MAP_ABSENT_RECHECK_SEC = 900.0

# Consecutive WS probes with an EMPTY top-of-book on BOTH sides before we stop
# trusting the probe's "active" verdict and force a REST status read. A settled
# market remains WS-subscribable for a while, so the probe keeps saying active
# and the finalized set never fills. One empty probe is normal for an illiquid
# pregame book; 3 in a row is not. Cost when it fires: one bulk-fetch slot.
EMPTY_PROBE_REST_VERIFY = 3


async def main():
    load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
    api_key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    
    if not private_key_path.startswith('/'):
        private_key_path = f"/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage/{private_key_path}"

    auth = KalshiAuth(api_key_id, private_key_path)
    client = KalshiClient(auth)

    import time as _time
    _run_mtime = os.path.getmtime(__file__)
    _run_modified = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(_run_mtime))
    log.info(f"Starting VWAP Esports Evaluator Engine... (run.py last modified: {_run_modified})")

    if DRY_RUN:
        # Bright magenta banner so it's impossible to miss
        log.warning("\033[1;95m" + "=" * 70 + "\033[0m")
        log.warning("\033[1;95m  DRY RUN MODE — NO TRADES WILL BE EXECUTED\033[0m")
        log.warning("\033[1;95m  Set DRY_RUN = False at top of run.py to go live.\033[0m")
        log.warning("\033[1;95m" + "=" * 70 + "\033[0m")
    else:
        log.warning("\033[1;91m  LIVE TRADING — orders will fire to Kalshi.\033[0m")

    # Shadow taker signals (VAL cluster + LoL pack-eat) — flag-gated
    # (taker_shadow.flag), single daemon thread, no orders, no trading-state
    # interaction. init() never raises.
    try:
        import taker_shadow
        if taker_shadow.init():
            log.info("taker_shadow: in-process shadow signals ARMED")
    except Exception as _ts_e:
        log.warning(f"taker_shadow unavailable: {_ts_e!r}")

    # Init the Dynamic Mathematical Layer
    generator = HybridTheoGenerator(client=client)

    # Phase 1 feed-integrity: spin up the REST orderbook poller. Maintains a
    # thread-safe registry of last-seen book state per ticker as a parallel
    # ground-truth source to the WS-derived books. Phase 2 will let the
    # manager fall back to this when WS shows empty. For now: observation +
    # heartbeat logging only — bot decisions are unchanged.
    orderbook_registry = OrderbookRegistry()

    # Spin up the Market Manager pipeline
    quoter = MarketManager(kalshi_client=client, theo_generator=generator, trading_enabled=not DRY_RUN)

    # Ticker provider: poll whatever tickers the manager currently has bots
    # bound to. Refreshed every 30s inside the poller loop.
    # PLUS (26AUG13): map tickers for currently-stale events, so the
    # poly-stale Kalshi fallback can actually CERTIFY — see
    # hedge_staleness.fallback_poll_tickers for the provider=declined
    # root cause. Empty set when nothing is stale (steady-state unchanged).
    def _active_tickers_for_poller() -> set:
        try:
            base = {b.config.ticker for b in quoter.bots if b.config.ticker}
            try:
                import hedge_staleness as _hs_pt
                base |= _hs_pt.fallback_poll_tickers(base)
            except Exception:
                pass          # certification helper must never break polling
            return base
        except Exception:
            return set()

    # RE-ENABLED 2026-06-09 22:30 UTC for diagnostic re-run with caller trace
    # (KALSHI_GET_CALLER_TRACE=1) to identify what's consuming /markets budget.
    # Pairs with the 50 RPS /markets/* solo cap empirically measured by
    # _get_rate_limit_probe.py on 2026-06-09 ~01:30 UTC. If 429s recur, the
    # get_callers.log will tell us who's calling /markets concurrently with
    # the poller. DRY_RUN must remain True for this test.
    orderbook_poller = OrderbookPoller(
        client=client,
        registry=orderbook_registry,
        ticker_provider=_active_tickers_for_poller,
        # OFF for tennis (26AUG27). Tennis rows quote off the ws_delta WebSocket book;
        # this REST poller was the esports stack's fallback and here it was pure load:
        # 6 tickers x 1 GET/s on the key the tennis pollers share, which starved them
        # into 429 backoff, made their tapes stale, and cancelled every resting order.
        # The registry stays constructed (empty) so the guarded readers are unaffected.
        enabled=False,
    )
    orderbook_poller.start()
    # Make the registry available on the manager for Phase 2 reads.
    quoter.orderbook_registry = orderbook_registry

    # ── Poly-stale → Kalshi map-book fallback provider (26AUG13, see
    # POLY_STALE_KALSHI_FALLBACK_SCOPE.md). Certifies, per event, that
    # Kalshi's OWN map books (REST registry — same books [FEED CHECK] reads)
    # are fresh/two-sided/tight/deep enough to carry quoting while Poly is
    # frozen. Consumed by hedge_staleness.mode(): SHADOW-logs by default,
    # only changes behavior once enable_stale_kalshi_fallback.flag exists.
    import hedge_staleness as _hs_fb

    def _kalshi_map_fallback_provider(event_base, _reg=orderbook_registry):
        try:
            _prefix, _code = event_base.split("-", 1)
        except ValueError:
            return None
        _map_prefix = _prefix.replace("GAME", "MAP")
        if _map_prefix == _prefix:
            return None                      # no map universe for this sport
        _want = f"{_map_prefix}-{_code}-"
        out = {}
        for _t, _b in _reg.snapshot().items():
            if not _t.startswith(_want):
                continue
            # Age bar is CERTIFICATION freshness, not pricing freshness — the
            # fallback theos ride the live Kalshi WS books; these registry
            # books only certify carrying capacity + feed the 10c dev guard.
            # 3.0s was unreachable: the poller is capped at 15 RPS across all
            # bot tickers (~60+), so per-ticker age is structurally 4-7s —
            # the provider declined EVERYTHING on 26AUG13 while NSEAKTC sat
            # 75/77 with 26k lots on Kalshi. 10s covers the poll cycle.
            if (_b.age_sec > 10.0 or _b.best_yes_bid <= 0
                    or _b.best_yes_ask <= 0
                    or _b.best_yes_ask - _b.best_yes_bid > 6
                    or min(_b.depth_yes_count, _b.depth_no_count) < 25):
                continue                     # this map leg can't carry us
            out[_t] = (_b.best_yes_bid, _b.best_yes_ask)
        return out or None

    _hs_fb.set_fallback_provider(_kalshi_map_fallback_provider)
    
    # Init the entirely autonomous Trade Monitor subsystem 
    watcher = TradeLogger(kalshi_client=client)
    watcher.start() # Deploy fully synchronous background threading!
    
    # Init Polymarket CLOB feed for esports map prices.
    # Pass a Kalshi fetcher so the feed can deterministically derive team
    # suffixes when not present in market state (replaces the brittle
    # _split_kalshi_teams TEAM_ALIASES heuristic that fails on missing aliases).
    def _kalshi_event_markets(event_base):
        try:
            r = client._get("/trade-api/v2/markets",
                            {"event_ticker": event_base, "limit": 10})
            return r.get("markets", []) if isinstance(r, dict) else []
        except Exception:
            return []
    poly_feed = PolyMapPriceFeed(kalshi_fetcher=_kalshi_event_markets)
    _spawn_poly_ws_listener()

    # Init WebSocket orderbook fetcher (replaces REST polling)
    ws_fetcher = WSOrderbookFetcher(auth)
    # Dedicated fetcher for crossed-book reseeds — fetch_all stomps self._ws on
    # every call, so sharing the rescue-path instance would race it.
    xbook_fetcher = WSOrderbookFetcher(auth)

    # Phase 2 feed-integrity BookView routing. RouterBookView dispatches
    # per-ticker based on each bot's `book_source` config:
    #   "ws"   → reads via ws_fetcher (default — preserves prior behavior)
    #   "rest" → reads from orderbook_registry (REST poller's snapshots)
    # The registry is the same one Phase 1 already populates. Default
    # source is "ws" so unconfigured tickers behave exactly as before.
    ws_book_view = WSBookView(ws_fetcher)
    rest_book_view = RestBookView(orderbook_registry, max_age_sec=3.0)

    # ─ Delta-maintained WS book (ws_delta) — DEFAULT ON ────────────────────
    # As of 2026-06-13 the orderbook read path defaults to the long-lived
    # delta-maintained WS book (`ws_delta_book.WSDeltaManager`). Active
    # esports tickers are subscribed dynamically each main-loop iteration
    # via `set_tickers(obs_to_fetch)`. The `WSDeltaBookView` is wrapped
    # behind `RouterBookView`, which still honors per-row `book_source=rest`
    # overrides from `market_parameters.csv`.
    #
    # ── Knobs (env vars) ──
    #
    # WS_DELTA_DISABLE=1
    #   Emergency kill switch. When set, NO WSDeltaManager is constructed
    #   and every orderbook read falls through to the old ws/rest path —
    #   byte-identical to pre-2026-06-13 behavior. Use this if anything
    #   about the new path looks wrong in production: `kill <pid>;
    #   WS_DELTA_DISABLE=1 .venv/bin/python3 -u run.py >> run.log 2>&1 &`.
    #
    # WS_DELTA_TICKERS=A,B,C
    #   Optional surgical-routing allowlist. If non-empty, ONLY these
    #   tickers route through ws_delta; everything else uses the old
    #   ws/rest path. Useful for staged rollouts (e.g. test on 1 game
    #   before opening up to the whole slate). When unset (default), ALL
    #   non-"rest" tickers go through ws_delta.
    #
    # ── Failure behavior (defense in depth) ──
    #
    # 1. WSDeltaManager.start() throws → log ERROR, set view to None,
    #    bot continues with the original ws/rest path. No-op restart.
    # 2. WSDeltaManager crashes mid-game → WSDeltaBookView catches the
    #    exception per-ticker and returns nothing for that ticker →
    #    RouterBookView omits it → run.py:~470 ws_missed fallback fetches
    #    a REST orderbook. Trading uninterrupted.
    # 3. WS connection drops → WSDeltaManager.connect_loop reconnects
    #    with exponential backoff (2-30s) and re-subscribes all targets.
    # 4. Periodic re-snapshot every 600s catches phantom levels via clean
    #    re-subscribe through forced reconnect.
    #
    # ── Full revert (one command) ──
    #
    # kill <run.py pid>; WS_DELTA_DISABLE=1 .venv/bin/python3 -u run.py >> run.log 2>&1 &
    #
    # No code edits, no CSV edits, no data migrations.
    ws_delta_disabled = bool(os.environ.get("WS_DELTA_DISABLE", "").strip())
    ws_delta_tickers_raw = os.environ.get("WS_DELTA_TICKERS", "").strip()
    ws_delta_allowlist = {t.strip() for t in ws_delta_tickers_raw.split(",") if t.strip()}
    ws_delta_view = None
    ws_delta_mgr = None
    if ws_delta_disabled:
        log.warning("[WS_DELTA] disabled via WS_DELTA_DISABLE env var — using ws+rest only")
    else:
        try:
            from ws_delta_book import WSDeltaManager
            ws_delta_mgr = WSDeltaManager(client.auth, periodic_reseed_s=600.0)
            await ws_delta_mgr.start()
            ws_delta_view = WSDeltaBookView(ws_delta_mgr)
            if ws_delta_allowlist:
                # Surgical mode: subscribe only to the named tickers. Static.
                await ws_delta_mgr.set_tickers(ws_delta_allowlist)
                log.warning(f"[WS_DELTA] surgical allowlist mode — {len(ws_delta_allowlist)} ticker(s): "
                            f"{sorted(ws_delta_allowlist)}  (periodic_reseed=600s)")
            else:
                log.warning("[WS_DELTA] default-on mode — all active esports tickers subscribed dynamically "
                            "each loop iteration (periodic_reseed=600s)")
        except Exception as e:
            log.error(f"[WS_DELTA] failed to start ({type(e).__name__}: {e}) — disabling. "
                      f"Bot will continue with ws+rest routing only.")
            ws_delta_view = None
            ws_delta_mgr = None

    # ── Phase 2 flip gate — INERT unless `use_isolated_ws.flag` exists ───────
    # WS_MIGRATION_PLAN.md Phase 2. IsolatedBookView is a live switch:
    #
    #   flag ABSENT  → delegates to ws_delta_view  = TODAY's behavior, exactly.
    #                  Restarting without the flag changes NOTHING the arber reads.
    #   flag PRESENT → arber reads the isolated book (own thread/conn/max_queue).
    #                  `rm use_isolated_ws.flag` reverts within one tick.
    #
    # Both directions are live — the flag is polled (~1s cache) on every fetch,
    # so flipping and reverting need NO run.py restart.
    #
    # `isolated_book` is constructed ~80 lines below (it costs nothing until
    # started), so the view resolves it LAZILY through a closure; by the time
    # any fetch_all runs (inside the main loop) the name is bound. Declared
    # None here so the closure can never hit an unbound local.
    #
    # Wrapped ONLY when ws_delta_view exists: with WS_DELTA_DISABLE=1 (or a
    # failed WSDeltaManager start) ws_delta_view is None and RouterBookView must
    # keep routing to the plain ws_view — preserving that kill switch verbatim.
    isolated_book = None
    _isolated_switch_view = (
        IsolatedBookView(book_provider=lambda: isolated_book, fallback=ws_delta_view)
        if ws_delta_view is not None else None
    )

    book_view = RouterBookView(
        ws_view=ws_book_view,
        rest_view=rest_book_view,
        source_by_ticker=quoter.book_source_by_ticker,
        default_source="ws",
        ws_delta_view=_isolated_switch_view,
        ws_delta_tickers=ws_delta_allowlist,  # empty set → ALL non-rest go to ws_delta
    )

    # ─ Dedicated MAP orderbook feed (isolated from the series ws_delta) ────────
    # Maps get their OWN long-lived WSDeltaManager (own WS connection) so map
    # volume / reconnects / a quiet map book can't starve or disturb the series
    # feed — the exact failure that left Kalshi map books empty on the shared
    # connection. Same WSBook reconstruction the series uses; validated == REST
    # on VCT C9G2 map 2 (2026-07-25).
    #
    # DORMANT until a map routes to the Kalshi path (data_source != poly): its
    # subscription set is `kalshi_map_tickers`, which is empty while every map is
    # data_source=poly, so this is a no-op until you flip a source. REVERT = set
    # the map's source back to poly (no restart needed — routing reads
    # data_source live each loop). Kill switch: MAP_WS_DEDICATED_DISABLE=1 makes
    # map reads fall back to the shared book_view path (today's behavior).
    #
    # Failure containment mirrors the series ws_delta: start() throws → view is
    # None → the map read below falls back to book_view + fresh-WS rescue; the
    # KALSHI MAP NO-FALLBACK scrub still force-absents any map with no fresh book
    # so we never trade a stale/phantom price.
    map_delta_mgr = None
    if not os.environ.get("MAP_WS_DEDICATED_DISABLE", "").strip():
        try:
            map_delta_mgr = MapWSDeltaManager(client.auth, periodic_reseed_s=0.0)
            await map_delta_mgr.start()

            # Heartbeat is POLLED at the set_tickers site below (see _map_ws_hb),
            # not event-driven: the manager increments stats["delta_applied"] as a
            # counter but never _emit()s it, so an event-callback heartbeat only
            # ever fired on the (rare, post-seed) snapshot and looked frozen.
            _map_ws_hb = {"ts": 0.0}

            def _map_ws_on_event(et, details):
                # Tagged, low-noise observability so map seeding/starvation is
                # never silent again (the shared-feed failure hid for a month
                # because the manager's callback was never wired). Only the rare
                # connection-level events are logged here; liveness is the polled
                # heartbeat at the set_tickers site.
                if et in ("conn_drop", "conn_stall", "forced_reconnect",
                          "ws_error", "conn_seq_gap"):
                    log.warning(f"[MAP WS] {et} {details}")

            map_delta_mgr.set_event_callback(_map_ws_on_event)
            log.warning("[MAP WS] dedicated map feed started (isolated from series; "
                        "dormant until a map routes to Kalshi)")

            # FULL-WS FALLBACK Phase 0 (26AUG19): shadow provider handing the
            # dedicated feed's RAW map TOBs (+age/depth) to hedge_staleness's
            # observation-only comparator. Bars are applied THERE so the WS
            # and REST paths are judged identically. No routing/trading use.
            def _kalshi_map_ws_shadow_provider(event_base, _mgr=map_delta_mgr):
                try:
                    _prefix, _code = event_base.split("-", 1)
                except ValueError:
                    return None
                _map_prefix = _prefix.replace("GAME", "MAP")
                if _map_prefix == _prefix:
                    return None
                _want = f"{_map_prefix}-{_code}-"
                _now = _time_mod.time()
                out = {}
                for _t, _b in list(_mgr.books.items()):
                    if not _t.startswith(_want) or not _b.is_live:
                        continue
                    _bid = max(_b.yes) if _b.yes else 0
                    _no_bid = max(_b.no) if _b.no else 0
                    _ask = (100 - _no_bid) if _no_bid else 0
                    # Depth = WHOLE-BOOK lots per side, matching the REST
                    # registry's depth_yes_count/depth_no_count (sum across
                    # levels, feed_integrity_poller.py:140) — top-level-only
                    # made the WS bar spuriously stricter (T1AKTC map-2 T1A
                    # 26AUG19: ws depth=7 vs rest PASS on the same book).
                    out[_t] = (_bid, _ask, round(_now - _b.last_msg_ts, 1),
                               sum(_b.yes.values()) // 100,
                               sum(_b.no.values()) // 100)
                return out or None

            _hs_fb.set_ws_shadow_provider(_kalshi_map_ws_shadow_provider)
        except Exception as e:
            log.error(f"[MAP WS] failed to start ({type(e).__name__}: {e}) — maps fall back "
                      f"to the shared book_view path. Bot continues.")
            map_delta_mgr = None

    # ── Phase 0: WS-vs-REST divergence shadow logger (measure-only) ─────────
    # Read-only, own daemon thread, ZERO new REST calls (reuses the REST
    # poller's `orderbook_registry`), INERT unless WS_REST_SHADOW_ENABLE=1.
    # Cannot affect trading — construction and start are fully guarded, and the
    # logger never mutates a book or a decision. See WS_ACCURACY_PLAN.md.
    ws_rest_shadow = None
    try:
        from ws_rest_shadow import WSRestShadowLogger
        ws_rest_shadow = WSRestShadowLogger(
            delta_managers=[ws_delta_mgr, map_delta_mgr],
            registry=orderbook_registry,
            ticker_provider=_active_tickers_for_poller,
        )
        ws_rest_shadow.start()
    except Exception as e:
        log.warning(f"[WS-REST SHADOW] init skipped ({type(e).__name__}: {e}); trading unaffected.")
        ws_rest_shadow = None

    # ── Isolated WS book (Phase 1 SHADOW) — constructed always (cheap: no thread
    # until started). Runs its own isolated reader ONLY while
    # enable_isolated_ws.flag exists, and is NEVER routed to the arber in Phase 1.
    # Single toggle: touch the flag to run it in shadow at full scale alongside
    # the old book; rm to stop. Fully additive — cannot affect trading.
    isolated_book = None
    try:
        from isolated_ws_book import IsolatedWSBook
        isolated_book = IsolatedWSBook(client.auth, client)
    except Exception as e:
        log.warning(f"[ISOLATED WS] construct skipped ({type(e).__name__}: {e}); trading unaffected.")
        isolated_book = None

    # Load the configured constraints!
    # QUOTER_CONFIG_CSV (2026-09-10, set3 dog maker): a second framework instance
    # (own cwd, own subaccount via QUOTER_SUBACCOUNT) needs its own config file —
    # two instances hot-reloading one template would fight over it. Default
    # unchanged: cwd-relative template_quoter_config.csv.
    CSV_PATH = os.environ.get("QUOTER_CONFIG_CSV", "template_quoter_config.csv")
    quoter.load_bots_from_csv(CSV_PATH)
    
    log.info(f"VWAP Engine armed with configuration from {CSV_PATH}. Engaging loop...")

    last_mod_time = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0
    PROBS_PATH = "esports_probabilities.csv"
    last_probs_mtime = os.path.getmtime(PROBS_PATH) if os.path.exists(PROBS_PATH) else 0

    try:
        while True:
            # 0. Hot Reload Hook: Visually check physical file delta without blocking!
            try:
                if os.path.exists(CSV_PATH):
                    current_mod_time = os.path.getmtime(CSV_PATH)
                    if current_mod_time > last_mod_time:
                        log.warning("\n>>> CSV MATRIX FRAGMENTATION DETECTED! Triggering native hot-reload protocol! <<<")
                        quoter.reload_bots_from_csv(CSV_PATH)
                        last_mod_time = current_mod_time
            except Exception as e:
                log.error(f"Failed to stat config file for hot reload: {e}")

            # 0b. esports_probabilities.csv watcher — refresh prob caches on populate_configs/picker-detector writes
            try:
                if os.path.exists(PROBS_PATH):
                    cur_probs_mtime = os.path.getmtime(PROBS_PATH)
                    if cur_probs_mtime > last_probs_mtime:
                        log.warning("\n>>> ESPORTS_PROBABILITIES.CSV CHANGED — refreshing prob caches <<<")
                        quoter.reload_probabilities()
                        last_probs_mtime = cur_probs_mtime
            except Exception as e:
                log.error(f"Failed to stat esports_probabilities.csv for hot reload: {e}")
                
            try:
                top_level_bids = {}
                top_level_offers = {}
                dt_market_state = {}

                force_patterns = _load_force_start_patterns()

                active_tickers = set()
                for bot in quoter.bots:
                    if bot.active:
                        ticker = bot.config.ticker

                        # Ticker-time culling REMOVED 2026-06-16.
                        # This used to skip bots whose Kalshi-listed start was >120 min
                        # away, but populate_configs.py already filters using the
                        # accurate Poly start time (poly_parsed_markets.csv
                        # poly_start_time_et column). Anything in
                        # template_quoter_config.csv is by definition within its
                        # capture window — re-deriving from the Kalshi ticker name
                        # (which can be hours wrong, e.g. XLGLEV 2026-06-16 listed
                        # 13:00 ET, real 10:00 ET) silently culled live events the
                        # user needed quoted. Trust populate_configs.
                        parts = ticker.split("-")

                        active_tickers.add(ticker)
                        from arber_bot import EsportsArberBot
                        from hedge_bot import HedgeBot
                        if isinstance(bot, (EsportsArberBot, HedgeBot)):
                            # ── SUBSTRING-COLLISION FIX (2026-07-28) ──────────
                            # Both the guard and the rewrite must look at the
                            # SERIES PREFIX only (parts[0]), never the whole
                            # ticker or the date+teams component (parts[1]).
                            #
                            # A TEAM NAME can contain the token. "2GAME Esports"
                            # made KXVALORANTGAME-26JUL281500SR2GAME rewrite to
                            # ...SR2*MAP* — so we subscribed
                            # KXVALORANTMAP-26JUL281500SR2MAP-1-SR, which does
                            # not exist. No map book -> detect_bo3_state returns
                            # None -> [LIVE SERIES] cannot detect state ->
                            # HALT SIGNAL -> the event never quoted at all
                            # (logged ONCE at boot, then silenced by
                            # _skipped_tickers, so it failed invisibly).
                            #
                            # The old `"GAME" in ticker` guard had the same bug
                            # in reverse: a MAP-ticker bot
                            # (KXVALORANTMAP-...-SR2GAME-1-SR) matched on the
                            # team name and got treated as a series, emitting
                            # garbage subscriptions with team_suffix="1".
                            _pfx = parts[0]
                            if ("GAME" in _pfx or "MATCH" in _pfx) and len(parts) >= 3:
                                series_base = parts[0] + "-" + parts[1]
                                if "MATCH" in _pfx:
                                    map_base = _pfx.replace("MATCH", "SETWINNER") + "-" + parts[1]
                                else:
                                    map_base = _pfx.replace("GAME", "MAP") + "-" + parts[1]
                                team_suffix = parts[2]
                                # Subscribe maps 1-5 optimistically — BO5 state
                                # detection past G2 needs M3-M5 orderbook data,
                                # and gating on is_bo5 here left BO5 events
                                # un-subscribed when probs hadn't loaded yet.
                                #
                                # BUT skip any map CONFIRMED absent from
                                # /markets. The old comment claimed nonexistent
                                # maps "silently no-op"; they did not. Each one
                                # held a WS subscription slot and drew a REST
                                # orderbook fetch every RECONCILE_PERIOD_S
                                # forever, because a 404 reconciled into a
                                # live-but-empty book (isolated_ws_book:377).
                                # Measured 2026-07-28: DKCKTC-4/-5 and BROHLE-3
                                # — all BO3s, so maps 3-5 never existed at any
                                # date — accounted for every `empty_live` miss.
                                #
                                # Absence is existence-driven, not format-driven
                                # (BO5s vary too), and expires after
                                # MAP_ABSENT_RECHECK_SEC so a late-listed
                                # decider still gets picked up.
                                _absent_maps = globals().get("_known_absent_map_tickers", {})
                                _t_now = _time_mod.time()
                                # ── STRUCTURAL MAP RANGE (2026-08-06) ─────────
                                # The decider map is NEVER listed — Kalshi only
                                # has the series market at that point. BO3 ⇒
                                # maps 1-2; BO5 ⇒ maps 1-4; NO format lists a
                                # map 5. Blind 1-5 subscription kept every
                                # impossible decider cycling through the
                                # /markets bulk fetch via the 900s absence
                                # re-probe, forever. Trust the format only once
                                # probs are loaded (baseline_loaded); before
                                # that stay optimistic at 1-4 so a BO5 whose
                                # probs load late is never left un-subscribed
                                # (the regression the comment above warns
                                # about). The absence-TTL check below stays as
                                # belt-and-braces for maps still in range.
                                # Probs-loaded indicator differs by class:
                                # EsportsArberBot → baseline_loaded; HedgeBot →
                                # active (False in __init__ iff probs missing).
                                _probs_known = getattr(
                                    bot, "baseline_loaded",
                                    getattr(bot, "active", False))
                                if _probs_known and not getattr(bot, "is_bo5", False):
                                    _map_range = (1, 2)
                                else:
                                    _map_range = (1, 2, 3, 4)
                                for _n in _map_range:
                                    _mt = f"{map_base}-{_n}-{team_suffix}"
                                    _abs_ts = _absent_maps.get(_mt)
                                    if _abs_ts is not None and (_t_now - _abs_ts) < MAP_ABSENT_RECHECK_SEC:
                                        continue        # confirmed phantom, still cooling down
                                    active_tickers.add(_mt)
                                # Also subscribe to the opponent series ticker for cross-book routing
                                # Check ALL bots (including inactive) since opponent may have deactivated
                                for opp_bot in quoter.bots:
                                    opp_t = opp_bot.config.ticker
                                    if opp_t.startswith(series_base + "-") and opp_t != ticker:
                                        active_tickers.add(opp_t)
                                        break
                        from mma_arber_bot import MMAArberBot
                        if isinstance(bot, MMAArberBot):
                            siblings = bot.get_sibling_tickers()
                            if siblings:
                                active_tickers.update(siblings)
                            else:
                                # First loop: discover children from Kalshi API
                                event_ticker = bot.config.ticker
                                try:
                                    resp = client._get(f"/trade-api/v2/markets?event_ticker={event_ticker}&status=active&limit=20")
                                    children = [m["ticker"] for m in resp.get("markets", [])]
                                    # Also check for aggregate markets under KXUFCMOF prefix
                                    fight_tag = event_ticker.split("-", 1)[1] if "-" in event_ticker else ""
                                    if fight_tag:
                                        agg_resp = client._get(f"/trade-api/v2/markets?event_ticker=KXUFCMOF-{fight_tag}&status=active&limit=20")
                                        agg_children = [m["ticker"] for m in agg_resp.get("markets", [])]
                                        children.extend(agg_children)
                                    bot.sibling_tickers = [t for t in children if t.startswith("KXUFCMOV")]
                                    for t in children:
                                        if t.startswith("KXUFCMOF"):
                                            suffix = t.split("-")[-1]
                                            bot.aggregate_tickers[suffix] = t
                                    active_tickers.update(children)
                                    log.info(f"[MMA DISCOVERY] Found {len(children)} markets for {event_ticker}")
                                except Exception as e:
                                    log.error(f"Failed to discover MMA sibling tickers for {event_ticker}: {e}")
                active_tickers = list(active_tickers)

                # ── Isolated WS book SHADOW supervisor (Phase 1) ─────────────
                # Single toggle: `touch enable_isolated_ws.flag` runs the new
                # isolated-reader book at full scale ALONGSIDE the old book (its
                # own thread/connection); `rm` stops it. NEVER read by the arber
                # here — pure shadow. Guarded so it can't affect trading.
                if isolated_book is not None:
                    try:
                        if os.path.exists("enable_isolated_ws.flag"):
                            if not isolated_book.is_running():
                                await isolated_book.start()
                                log.warning("[ISOLATED WS] SHADOW started (own thread; NOT routed to trading).")
                            # Drop settled tickers from the subscription set
                            # (2026-07-28). `obs_to_fetch` already excludes them
                            # from trading, so this changes nothing we quote —
                            # it only stops MAINTAINING books nobody reads.
                            # Measured: NEMIGAPRAN maps 1-2 finalized at 11:0x
                            # kept reporting isolated:stale forever (68 misses /
                            # 17 min) because a settled market receives no
                            # deltas by definition.
                            #
                            # Safe because map results are authoritative from
                            # /markets METADATA (status/result/last_price), not
                            # from the book — see hedge_engine._recover_map_wl,
                            # whose docstring already notes "settled maps drop
                            # out of full_market_state -> empty book" and
                            # recovers from cache. Authoritative results
                            # explicitly outrank any live-book signal.
                            #
                            # Uses the PERSISTENT _known_finalized_tickers set
                            # (accumulated in prior iterations — it is populated
                            # later in this loop body, so this reads one cycle
                            # behind, which is harmless: a ticker that just
                            # settled simply unsubscribes next tick).
                            _fin = globals().get("_known_finalized_tickers", set())
                            await isolated_book.set_tickers(set(active_tickers) - _fin)
                            _now = _time_mod.time()
                            if _now - globals().get("_iso_ws_log_ts", 0.0) >= 60.0:
                                globals()["_iso_ws_log_ts"] = _now
                                _m = _t = 0   # free correctness peek vs the REST poller's registry (no new REST calls)
                                for _tk in active_tickers[:8]:
                                    _fp = isolated_book.get_orderbook_fp(_tk)
                                    _rb = orderbook_registry.get(_tk)
                                    if _fp and _rb is not None:
                                        _in = len((_fp.get("orderbook_fp") or {}).get("no_dollars") or [])
                                        _rn = len((_rb.to_ws_style().get("orderbook_fp") or {}).get("no_dollars") or [])
                                        _t += 1
                                        _m += 1 if abs(_in - _rn) <= 1 else 0
                                log.warning("[ISOLATED WS] shadow health | stats=%s books=%d | depth≈REST-registry %d/%d",
                                            isolated_book.stats, len(isolated_book.books), _m, _t)
                        elif isolated_book.is_running():
                            isolated_book.stop()
                            log.warning("[ISOLATED WS] SHADOW stopped (flag cleared).")
                    except Exception as _e:
                        log.warning("[ISOLATED WS] supervisor error (ignored): %s", _e)

                # ── WS miss rollup (2026-07-28) ──────────────────────────────
                # Per-tick fallback lines are too noisy to count by hand and
                # too sparse to reason about. This aggregates every miss in the
                # window by (book, reason, ticker) so the question "what is
                # actually causing REST fallback" is answered by one grep:
                #   grep '\[WS MISS ROLLUP\]' -A8 run.log
                # A reason that dominates here is the thing to fix; a reason
                # that never appears can be ruled out.
                try:
                    _nowr = _time_mod.time()
                    if _nowr - globals().get("_ws_miss_rollup_ts", 0.0) >= 60.0:
                        globals()["_ws_miss_rollup_ts"] = _nowr
                        _ro = ws_miss_stats.rollup_and_reset()
                        if _ro:
                            log.warning(_ro)
                except Exception as _e:
                    log.debug("[WS MISS ROLLUP] error (ignored): %s", _e)

                # Split: series + map tickers fetched at low cadence (status
                # changes are coarse — game start, suspension, settlement —
                # nothing happens faster than a few seconds). Live prices come
                # from WS orderbook + Poly (data_source=poly for LoL/CS2), so
                # /markets is only needed for the status string. Was hammering
                # /markets at 6+ RPS for the same 2 series tickers, eating
                # Kalshi's read budget for no benefit. 2026-06-08.
                # /markets polling is purely a status sanity check (active/closed/
                # settled). Real-time data — best bid/ask, fills, settlement —
                # comes from WS orderbook + price-collapse detection in
                # live_series_model. The bot doesn't need /markets for any
                # trading decision; we keep a slow background poll just so the
                # status string in logs stays roughly current.
                # 2026-06-10: bumped from 120/60 → 300/300 after the bulk
                # endpoint cap was empirically isolated. `/markets?tickers=`
                # has a ~35-token bucket with ~3-5/sec refill — SEPARATE
                # from the per-ticker /markets/{T}/orderbook 50 RPS cap.
                # Bot's 8-chunk refresh round drained the bucket faster than
                # it refilled, causing the persistent 429 storm. 5-min
                # cadence keeps the bot's burst under refill rate.
                # Caveat: stale `status` field for up to 5 min after a map
                # finalizes — covered partially by live_series_model's
                # price-collapse detection from orderbook polls.
                MAP_METADATA_REFRESH_SEC = 300.0
                SERIES_METADATA_REFRESH_SEC = 300.0
                _map_meta_cache = globals().setdefault("_map_meta_cache", {})
                _last_map_refresh_ts = globals().get("_last_map_refresh_ts", 0.0)
                _series_meta_cache = globals().setdefault("_series_meta_cache", {})
                _last_series_refresh_ts = globals().get("_last_series_refresh_ts", 0.0)
                # Persistent per-ticker static-metadata cache (loaded once
                # per run.py boot, written back when REST fallback fills a
                # missing entry). Holds yes_sub_title/title/rules_primary
                # plus last-known last_price_dollars/result so the WS-probe
                # fast path can synthesize full_markets[t] dicts without
                # touching /markets?tickers=.
                _run_meta_cache = globals().setdefault(
                    "_run_meta_cache", _load_run_metadata_cache(),
                )
                if globals().get("_run_meta_cache_first_log") is None:
                    globals()["_run_meta_cache_first_log"] = True
                    if _run_meta_cache:
                        log.info(f"[METADATA-CACHE] loaded {len(_run_meta_cache)} cached ticker(s)")

                # Distinguish per-map tickers (KXCS2MAP, KXATPSETWINNER) from
                # event-level tickers including prop tickers (KXCS2TOTALMAPS).
                # The substring "MAP" alone matches both KXCS2MAP and KXCS2TOTALMAPS,
                # so use endswith("MAP") on the first dash-separated component.
                def _is_per_map(t: str) -> bool:
                    p = t.split("-", 1)[0]
                    return p.endswith("MAP") or "SETWINNER" in p
                series_tickers = [t for t in active_tickers if not _is_per_map(t)]
                map_tickers    = [t for t in active_tickers if _is_per_map(t)]

                _now = __import__("time").time()
                refresh_maps = (_now - _last_map_refresh_ts) >= MAP_METADATA_REFRESH_SEC
                refresh_series = (_now - _last_series_refresh_ts) >= SERIES_METADATA_REFRESH_SEC

                tickers_to_fetch = ((series_tickers if refresh_series else []) +
                                    (map_tickers if refresh_maps else []))

                full_markets = {}
                # Cached metadata is still valid; seed before any GET so a
                # partial refresh doesn't drop the un-refreshed half.
                if not refresh_maps:
                    full_markets.update(_map_meta_cache)
                if not refresh_series:
                    full_markets.update(_series_meta_cache)

                # ─── WS-probe-first pass (2026-06-24) ───────────────────
                # Replaces the bulk REST /markets?tickers= sanity check with
                # a single WS subscribe that classifies every ticker (active
                # vs finalized) AND seeds top-of-book. Combined with the
                # persistent _run_meta_cache for static fields, this lets
                # the per-cycle status check skip REST entirely for any
                # ticker we've previously seen and that WS classifies cleanly.
                #
                # Only tickers that need REST fall through to the loop below:
                #   * missing from cache (new ticker — one-shot REST)
                #   * unclassified by WS (timeout edge case — REST fallback)
                _ws_resolved: set[str] = set()
                # Drop tickers we've already classified as finalized in a
                # prior cycle — they'd return code:28 again (or nothing,
                # holding the probe loop until timeout) and burn wall-clock
                # in the main bot loop. The probe blocks the per-tick path
                # for its full duration; shrinking the set is the biggest
                # win we can ship.
                _pre_known_count = len(tickers_to_fetch)
                tickers_to_fetch = [
                    t for t in tickers_to_fetch
                    if t not in globals().get("_known_finalized_tickers", set())
                ]
                _ws_skip_count = _pre_known_count - len(tickers_to_fetch)
                if _ws_skip_count:
                    log.info(f"[WS-PROBE] skipping {_ws_skip_count} known-finalized ticker(s) — "
                             f"probing {len(tickers_to_fetch)} of {_pre_known_count}")
                if tickers_to_fetch:
                    try:
                        _ws_t0 = _time_mod.time()
                        _ws_probe_results = await _ws_probe_markets_async(
                            tickers_to_fetch, timeout_s=2.0,
                        )
                        for _t in tickers_to_fetch:
                            _ms = _ws_probe_results.get(_t)
                            if _ms is None or _ms.active is None:
                                continue  # WS didn't classify → REST fallback
                            _cached = _run_meta_cache.get(_t)
                            if _cached is None:
                                continue  # No static metadata yet → REST fallback
                            # Synthesize the same dict shape the REST loop builds.
                            _mk = {
                                "ticker": _t,
                                "yes_sub_title": _cached.get("yes_sub_title", ""),
                                "title": _cached.get("title", ""),
                                "rules_primary": _cached.get("rules_primary", ""),
                                # last_price_dollars + result: stale from last
                                # REST refresh, matching prior 300s cadence.
                                # When status transitions to finalized, the
                                # transition watchdog below will REST-refresh
                                # to capture the settlement `result`.
                                "last_price_dollars": _cached.get(
                                    "last_price_dollars", "0",
                                ),
                                "result": _cached.get("result", ""),
                            }
                            if _ms.active is False:
                                _mk["status"] = "finalized"
                                # Permanent skip-set: once WS reports a
                                # ticker as finalized (code:28 in the WS
                                # probe), we'll never need to fetch its
                                # book again. This survives stale
                                # _series_meta_cache entries that might
                                # otherwise re-mark it "active" after a
                                # later partial refresh.
                                globals().setdefault(
                                    "_known_finalized_tickers", set(),
                                ).add(_t)
                            else:
                                # ── EMPTY-BOOK REST VERIFY (2026-07-28) ──────
                                # A settled market stays WS-SUBSCRIBABLE for a
                                # while after Kalshi finalizes it, so the probe
                                # reports active=True and we mark it "active"
                                # forever — it never enters
                                # _known_finalized_tickers, and because it is
                                # _ws_resolved the REST path that WOULD read
                                # status=finalized never runs. Measured
                                # 2026-07-28 19:49-20:59: GXPBAR was
                                # status=finalized on REST with a genuinely
                                # empty book (0 levels on REST too) yet produced
                                # 818 [WS FALLBACK] lines, and [FINALIZED-SKIP]
                                # fired ZERO times in 70 minutes.
                                #
                                # An empty top-of-book on BOTH sides is the
                                # signature. One empty probe is normal (illiquid
                                # pregame); N consecutive is not. Fall through to
                                # REST so the authoritative status is read —
                                # do NOT mark _ws_resolved.
                                _empty_streak = globals().setdefault("_ws_empty_streak", {})
                                if _ms.yes_top_bid_c is None and _ms.no_top_bid_c is None:
                                    _empty_streak[_t] = _empty_streak.get(_t, 0) + 1
                                else:
                                    _empty_streak.pop(_t, None)
                                if _empty_streak.get(_t, 0) >= EMPTY_PROBE_REST_VERIFY:
                                    if _empty_streak[_t] == EMPTY_PROBE_REST_VERIFY:
                                        log.info(
                                            f"[EMPTY-BOOK VERIFY] {_t} empty on "
                                            f"{EMPTY_PROBE_REST_VERIFY} consecutive WS probes "
                                            f"while WS reports active — forcing REST status check"
                                        )
                                    continue  # not _ws_resolved -> REST reads real status
                                _mk["status"] = "active"
                                if _ms.yes_top_bid_c is not None:
                                    _mk["yes_bid_dollars"] = f"{_ms.yes_top_bid_c / 100:.4f}"
                                if _ms.no_top_bid_c is not None:
                                    _mk["yes_ask_dollars"] = f"{(100 - _ms.no_top_bid_c) / 100:.4f}"
                            full_markets[_t] = _mk
                            _pp_ws = _t.split("-", 1)[0]
                            _is_map_ws = (
                                _pp_ws.endswith("MAP") or "SETWINNER" in _pp_ws
                            )
                            if refresh_maps and _is_map_ws:
                                _map_meta_cache[_t] = _mk
                            if refresh_series and not _is_map_ws:
                                _series_meta_cache[_t] = _mk
                            _ws_resolved.add(_t)
                            # ── Status-transition watchdog ──
                            # When WS reports finalized but our cache lacks
                            # a `result`, the market just settled. Mark the
                            # ticker for a one-shot REST refresh below to
                            # capture the settlement winner (`result` field
                            # is the only thing REST gives that WS can't).
                            if _ms.active is False and not _cached.get("result"):
                                _ws_resolved.discard(_t)
                        log.info(
                            f"[WS-PROBE] resolved {len(_ws_resolved)}/"
                            f"{len(tickers_to_fetch)} via WS in "
                            f"{_time_mod.time() - _ws_t0:.2f}s"
                        )
                    except Exception as _ws_e:
                        log.error(
                            f"[WS-PROBE] failed: {type(_ws_e).__name__}: {_ws_e} "
                            f"— falling back to REST for all tickers"
                        )
                # Only REST-fetch tickers the WS path couldn't fully resolve.
                tickers_to_fetch = [t for t in tickers_to_fetch if t not in _ws_resolved]

                # 2026-06-10: Chunked from batch_size=50 to 10 + 0.5s sleep
                # between chunks. The previous 50-ticker bulk fired as one
                # synchronous call which Kalshi appears to meter per-ticker
                # (1 call = 50 budget units in a sub-100ms burst). Combined
                # with the orderbook poller's sustained ~20 RPS, this caused
                # a per-minute 429 on the bot's bulk fetch (the family bucket
                # rolling window briefly exceeded cap). Smaller batches +
                # inter-batch sleep spread the load across ~5s instead of
                # spiking instantaneously. /markets metadata is just a status
                # sanity check (active/closed/finalized), NOT trading data —
                # so the extra 4-5s of fetch wall-time is irrelevant. Each
                # batch is still a single GET, just smaller.
                BULK_CHUNK_SIZE = 10
                # 2026-06-10 bumped from 0.5s → 2.0s. With 0.5s, multiple
                # chunks landed within the same rolling-window second alongside
                # the poller's 20 RPS, pushing total to ~30 RPS — every chunk
                # got 429'd. 2s ensures each chunk lives in its own second so
                # combined load stays under 30 RPS. Fetch wall-time goes from
                # ~4s → ~16s for 80 tickers, but /markets is a status-only
                # sanity check so the extra latency is irrelevant.
                INTER_CHUNK_SLEEP_SEC = 2.0
                for i in range(0, len(tickers_to_fetch), BULK_CHUNK_SIZE):
                    batch = tickers_to_fetch[i:i+BULK_CHUNK_SIZE]
                    # Wire batch uses real tickers (synthetic→real, pass-through
                    # for non-aliased). Internal keys (full_markets, _*_meta_cache)
                    # stay on synthetic so the bot can look up by its internal name.
                    wire_batch = [ticker_aliases.resolve(t) for t in batch]
                    tickers_str = ",".join(wire_batch)
                    try:
                        req_url = f"/trade-api/v2/markets?tickers={tickers_str}"
                        markets_resp = client._get(req_url)
                        if markets_resp and "markets" in markets_resp:
                            # ── PHANTOM MAP DETECTION (2026-07-28) ──────────
                            # `?tickers=` returns ONLY tickers that exist, so a
                            # requested ticker missing from a SUCCESSFUL response
                            # does not exist. That is authoritative existence
                            # data we were already paying for and discarding.
                            #
                            # We subscribe maps 1-5 blind (see the subscribe site
                            # below), and a BO3 has only maps 1-2 — verified
                            # 2026-07-28 that DKCKTC/BROHLE/KTDRX expose exactly
                            # maps 1-2 at ANY date. The surplus tickers were not
                            # "silently no-op" as the old comment claimed: the
                            # isolated book's reconcile worker turned each 404
                            # into a live-but-empty book and re-fetched it via
                            # REST every RECONCILE_PERIOD_S, forever.
                            _returned = {
                                ticker_aliases.reverse(_m.get("ticker", ""))
                                for _m in markets_resp["markets"]
                            }
                            _absent_seen = globals().setdefault("_map_absent_seen", {})
                            _absent = globals().setdefault("_known_absent_map_tickers", {})
                            for _bt in batch:
                                if _bt in _returned or not _is_per_map(_bt):
                                    # Present (or not a map) — clear any partial
                                    # strike so a transient blip can't accumulate,
                                    # and lift a prior absence immediately rather
                                    # than waiting out the remaining TTL (a
                                    # late-listed decider must go live at once).
                                    _absent_seen.pop(_bt, None)
                                    if _absent.pop(_bt, None) is not None:
                                        log.warning(f"[PHANTOM MAP] {_bt} now LISTED — resubscribing")
                                    continue
                                _absent_seen[_bt] = _absent_seen.get(_bt, 0) + 1
                                # Require 2 independent confirmations before we
                                # stop subscribing: one truncated/odd response
                                # must not be able to blind us to a real map.
                                if _absent_seen[_bt] >= 2:
                                    _was_known = _bt in _absent
                                    # Re-arm the TTL on every re-confirmation so
                                    # a still-absent map goes back to sleep
                                    # instead of re-probing every tick.
                                    _absent[_bt] = _now
                                    if not _was_known:
                                        log.warning(
                                            f"[PHANTOM MAP] {_bt} absent from /markets "
                                            f"(confirmed {_absent_seen[_bt]}x) — unsubscribing "
                                            f"and skipping REST until re-probe in "
                                            f"{int(MAP_ABSENT_RECHECK_SEC)}s"
                                        )
                            for mk in markets_resp["markets"]:
                                # Reverse-alias the response ticker so internal
                                # state is keyed on synthetic. Pass-through for
                                # non-aliased.
                                internal_ticker = ticker_aliases.reverse(mk["ticker"])
                                # CRITICAL: also rewrite the embedded `ticker`
                                # field so downstream consumers that read
                                # `mk["ticker"]` (poly_price_feed._build_entry
                                # parses team suffix from this field) see the
                                # synthetic name. Otherwise FURIA → "FURIA"
                                # suffix mismatches the bot's "-FUR" key, the
                                # Poly inject writes to the wrong map ticker,
                                # and BO5 HALT fires on the FUR side.
                                mk["ticker"] = internal_ticker
                                full_markets[internal_ticker] = mk
                                _pp = internal_ticker.split("-", 1)[0]
                                is_map = (_pp.endswith("MAP") or
                                          "SETWINNER" in _pp)
                                if refresh_maps and is_map:
                                    _map_meta_cache[internal_ticker] = mk
                                if refresh_series and not is_map:
                                    _series_meta_cache[internal_ticker] = mk
                                # Persist static + last-known fields so the
                                # next cycle's WS-probe path can synthesize
                                # full_markets without REST. Updates whenever
                                # the REST fallback runs (new ticker, WS
                                # timeout, or status-transition refetch).
                                _run_meta_cache[internal_ticker] = {
                                    "yes_sub_title": (mk.get("yes_sub_title") or ""),
                                    "title": (mk.get("title") or ""),
                                    "rules_primary": (mk.get("rules_primary") or ""),
                                    "last_price_dollars": (mk.get("last_price_dollars") or "0"),
                                    "result": (mk.get("result") or ""),
                                }
                    except Exception as e:
                        log.error(f"Failed pulling full market state batch: {e}")
                    # Pace between batches so the per-second peak load
                    # stays bounded. Skip the sleep after the last chunk.
                    if i + BULK_CHUNK_SIZE < len(tickers_to_fetch):
                        await asyncio.sleep(INTER_CHUNK_SLEEP_SEC)
                if refresh_maps:
                    globals()["_last_map_refresh_ts"] = _now
                    log.info(f"[BULK FETCH] refreshed {len(map_tickers)} map ticker(s) "
                             f"(next refresh in {int(MAP_METADATA_REFRESH_SEC)}s)")
                if refresh_series:
                    globals()["_last_series_refresh_ts"] = _now
                # Persist the metadata cache so a run.py restart can hit the
                # WS-probe fast path immediately. Skipped if neither half
                # refreshed this tick (nothing changed).
                if refresh_maps or refresh_series:
                    _save_run_metadata_cache(_run_meta_cache)
                
                # Pre-filter explicitly for Orderbook-dependent derivations
                # KXTTELITEMATCH added 2026-09-09: this list feeds obs_to_fetch
                # (WS book subscriptions + fetch_all), so a series missing here
                # NEVER GETS BOOKS and its bots silently cannot quote.
                all_esports_tickers = [t for t in active_tickers if any(prefix in t for prefix in ["KXLOL", "KXCS2", "KXVALORANT", "KXUFCMOV", "KXDOTA2", "KXCOD", "KXATP", "KXWTA", "KXTTELITEMATCH"])]

                # Split: Poly CLOB for esports maps, WS for series + everything else
                # endswith("MAP") on the first dash-separated component avoids
                # misclassifying KXCS2TOTALMAPS (prop ticker) as a map ticker —
                # props live on Kalshi binary and don't route through Poly CLOB.
                esports_map_tickers = [
                    t for t in all_esports_tickers
                    if t.split("-", 1)[0].endswith("MAP") or "SETWINNER" in t.split("-", 1)[0]
                ]
                # Filter out finalized/settled tickers — Kalshi WS returns
                # `code:28 "Markets not found"` for these (event unsubscribable
                # post-settle), so every loop tick a finalized ticker drops
                # into ws_missed and fires a wasted REST fallback. Observed
                # 4IKIL1GA / ACESASHI / SOSARE / EFAM each bleeding ~30 REST
                # calls/min for hours after settle on 2026-06-24.
                #
                # Multi-source detection (any one triggers permanent skip):
                #   1. full_markets[t]["status"] from the WS-probe / bulk
                #      REST pass above
                #   2. _run_meta_cache[t]["result"] is non-empty (REST has
                #      seen the market settle and persisted the winner)
                #   3. Permanent _known_finalized_tickers set in globals(),
                #      accumulated below — once we identify a finalized
                #      ticker, we remember it forever this run so a stale
                #      cache entry can't un-finalize it.
                _FINALIZED_STATUSES = ("finalized", "settled", "determined", "closed")
                _known_finalized = globals().setdefault(
                    "_known_finalized_tickers", set(),
                )
                for _t in all_esports_tickers:
                    _fm = full_markets.get(_t) or {}
                    if _fm.get("status") in _FINALIZED_STATUSES:
                        _known_finalized.add(_t)
                    elif (_run_meta_cache.get(_t) or {}).get("result"):
                        _known_finalized.add(_t)
                esports_finalized = {t for t in all_esports_tickers if t in _known_finalized}
                # Log ONLY when the set changes — this runs every loop iteration
                # and the set is permanent for the process, so an unconditional
                # log emits ~4-5 lines/sec forever. Measured 2026-07-29: 14,888
                # of 40,000 run.log lines (37%) were this one message. It never
                # fired before the empty-book REST verify started populating
                # _known_finalized_tickers, which is why it went unnoticed.
                # Mirrors the [DECIDED-SKIP] throttle immediately below.
                _fin_logged = globals().setdefault("_finalized_logged", set())
                if esports_finalized and esports_finalized != _fin_logged:
                    _new = sorted(esports_finalized - _fin_logged)
                    globals()["_finalized_logged"] = set(esports_finalized)
                    log.info(f"[FINALIZED-SKIP] now excluding {len(esports_finalized)} settled "
                             f"ticker(s) from per-tick fetch (+{len(_new)} new: {_new[:6]})")
                # Events retired by the DECIDED-EVENT detector (see below, near
                # the staleness_tracker feed). Kalshi-status-independent: catches
                # finished games that Kalshi still reports as `active`.
                _decided_events = globals().get("_decided_events", {})
                esports_decided = {
                    t for t in all_esports_tickers
                    if t.rsplit("-", 1)[0] in _decided_events
                }
                if esports_decided and not os.path.exists("disable_decided_event_skip.flag"):
                    if _decided_events.get("_logged_n") != len(esports_decided):
                        _decided_events["_logged_n"] = len(esports_decided)
                        log.info(f"[DECIDED-SKIP] excluding {len(esports_decided)} decided "
                                 f"ticker(s) from per-tick fetch: {sorted(esports_decided)[:6]}")
                else:
                    esports_decided = set()

                obs_to_fetch = [
                    t for t in all_esports_tickers
                    if t not in esports_map_tickers and t not in esports_finalized
                    and t not in esports_decided
                ]
                # ── ITF LIVE BOOK (26SEP27) ──────────────────────────────────────
                # KXITF is absent from the all_esports_tickers prefix list at ~1539,
                # which is the ONLY input to set_tickers() below. So KXITFMATCH /
                # KXITFWMATCH were never subscribed to ws_delta and never fetched —
                # while run.py:647 had already disabled the REST OrderbookPoller for
                # tennis on the belief that "tennis rows quote off the ws_delta
                # WebSocket book" (true for KXATP/KXWTA, false for ITF). ITF top of
                # book therefore fell through to the full_markets metadata snapshot
                # at ~1714. Measured cost: 294 set-1 taker fires carried both the
                # displayed ask and a REST book, and REST was dearer by a mean of
                # 11.5c, in 286 of 294 cases.
                #
                # Added HERE and deliberately NOT to all_esports_tickers: that list
                # also feeds esports_map_tickers (which matches "SETWINNER" in the
                # prefix and would route ITF set markets into Poly CLOB map logic),
                # finalization, and the Poly fallback. obs_to_fetch is exactly the
                # two things ITF needs — set_tickers() and book_view.fetch_all() —
                # and the application loop at ~1692 already iterates active_tickers,
                # so b_bid/b_offer land in top_level_bids/offers with no other change.
                #
                # DOUBLES added 26SEP27 for BOOK CAPTURE ONLY. Their config rows exist
                # so the tickers reach active_tickers and get a live book; the recenter
                # model (the only ungated quoter in the tennis_dog_windows dispatcher)
                # carries a RECENTER_SERIES guard that stops at book-logging for them,
                # and every other model in that dispatcher is already scoped to the two
                # singles series. SETWINNER stays out: it is what the naive fix at ~1539
                # would have misrouted into Poly CLOB map logic.
                _itf_ws = [
                    t for t in active_tickers
                    if t.split("-", 1)[0] in ("KXITFMATCH", "KXITFWMATCH",
                                              "KXITFDOUBLES", "KXITFWDOUBLES")
                    and t not in esports_finalized and t not in esports_decided
                ]
                if _itf_ws:
                    obs_to_fetch = obs_to_fetch + [t for t in _itf_ws if t not in obs_to_fetch]
                    log.warning("[ITF-WS] %d ITF ticker(s) added to the ws_delta target set "
                                "(total obs_to_fetch=%d)", len(_itf_ws), len(obs_to_fetch))
                ob_results = {}

                # Keep ws_delta's subscription set in sync with the live ticker
                # universe. set_tickers() is idempotent + diffs internally — new
                # tickers get subscribed (Kalshi sends a snapshot), removed
                # tickers get books dropped. Skipped entirely in surgical-
                # allowlist mode (static) and when ws_delta is disabled.
                # 2026-07-18: map tickers are now subscribed too. Previously this
                # passed only set(obs_to_fetch) — which EXCLUDES maps — so the
                # delta manager never seeded a map book, and (because set_tickers
                # also pops books outside the target set) never could. From the
                # 2026-06-13 ws_delta migration until 2026-07-18 that made
                # book_view.fetch_all(kalshi_map_tickers) return nothing every
                # cycle, and Kalshi map prices silently came from the 300s
                # /markets metadata cache instead. Subscribing maps restores the
                # fast long-lived-book path without a per-cycle reconnect.
                # Sizing: ~17 series + ~50 map = ~67 tickers, well under the 100
                # per-subscribe that WSOrderbookFetcher already assumes.
                if ws_delta_mgr is not None and not ws_delta_allowlist:
                    try:
                        # `obs_to_fetch` is already finalized-filtered; the MAP
                        # half was not, so settled maps stayed subscribed here
                        # too (2026-07-28). Same rationale as the isolated book
                        # above — results are authoritative from metadata, not
                        # from the book.
                        await ws_delta_mgr.set_tickers(
                            set(obs_to_fetch)
                            | (set(esports_map_tickers) - esports_finalized)
                        )
                    except Exception as e:
                        log.error(f"[WS_DELTA] set_tickers failed: {type(e).__name__}: {e}")

                # Fetch series orderbooks via BookView (per-ticker WS, ws_delta,
                # or REST routing per book_source config; default after
                # 2026-06-13 is ws_delta when available, else ws).
                ob_results = await book_view.fetch_all(obs_to_fetch)

                # Fallback to REST for any tickers that WS missed
                ws_missed = [t for t in obs_to_fetch if t not in ob_results]
                # Publish this tick's fallback set so the BLIP override path can
                # state whether the book it just overrode came from WS or was
                # ALREADY REST. An override on a REST-sourced book is REST-vs-REST
                # and means something other than WS divergence.
                globals()["_ws_fallback_this_tick"] = set(ws_missed)
                if ws_missed:
                    # Carry the per-book miss reason (unseeded / not_live /
                    # stale / empty_live) into the line. Previously this said
                    # only WHICH tickers fell back, so a session's REST load
                    # could not be attributed to a cause — see ws_miss_stats.
                    _why = " | ".join(f"{t} [{ws_miss_stats.describe(t)}]"
                                      for t in sorted(ws_missed))
                    log.info(f"[WS FALLBACK] Fetching {len(ws_missed)} tickers via REST: {_why}")
                    for i in range(0, len(ws_missed), 10):
                        chunk = ws_missed[i:i+10]
                        async def fetch_target(t_val):
                            import requests
                            # Translate synthetic → real for the wire URL;
                            # return t_val (synthetic) as the key so the
                            # caller stores the result under the bot's
                            # internal name. Pass-through for non-aliased.
                            wire_t = ticker_aliases.resolve(t_val)
                            def _get_rest():
                                url = f"https://external-api.kalshi.com/trade-api/v2/markets/{wire_t}/orderbook"
                                return requests.get(url, headers={"Accept": "application/json"}, timeout=1.5).json()
                            try:
                                res = await asyncio.to_thread(_get_rest)
                                return t_val, res
                            except Exception as e:
                                log.error(f"Failed pulling REST fallback for {t_val}: {e}")
                                return t_val, None

                        tasks = [fetch_target(tkr) for tkr in chunk]
                        results = await asyncio.gather(*tasks)
                        for tkr, data in results:
                            if data:
                                ob_results[tkr] = data
                        # 2026-07-17: removed `await asyncio.sleep(0.5)` between
                        # chunks — it blocked the trade loop, adding up to ~1s per
                        # cycle during a WS-fallback storm. Per-ticker orderbook GET
                        # is 50 RPS solo, so ≤20 tickers in chunks of 10 is well
                        # within limits without the pause.

                for t in active_tickers:
                    try:
                        if t not in dt_market_state:
                            dt_market_state[t] = {}
                            
                        if t not in obs_to_fetch:
                            # Still populate top-of-book from full_markets for non-orderbook tickers (e.g., MLB)
                            #
                            # 2026-07-18: esports MAP tickers no longer get a price
                            # seeded here. full_markets is fed by the 300s /markets
                            # metadata bulk fetch (MAP_METADATA_REFRESH_SEC) — a
                            # status sanity check, NOT trading data. Seeding a price
                            # from it put an up-to-5-minute-stale book into
                            # top_level_bids/offers that was indistinguishable from
                            # live to every downstream gate. Map prices now come
                            # ONLY from their live sources: Poly CLOB inject
                            # (poly_map_tickers) or the Kalshi WS book
                            # (kalshi_map_tickers) below. If neither supplies a
                            # book this cycle the ticker stays ABSENT and the
                            # active-map guard suppresses — empty data means no
                            # trade. Status is still carried through, since that
                            # is what the metadata fetch is actually for.
                            if t in full_markets:
                                fm = full_markets[t]
                                dt_market_state[t]["status"] = fm.get("status", "")
                                if t not in esports_map_tickers:
                                    bid_d = fm.get("yes_bid_dollars")
                                    ask_d = fm.get("yes_ask_dollars")
                                    if bid_d is not None:
                                        top_level_bids[t] = int(float(bid_d) * 100)
                                    if ask_d is not None:
                                        top_level_offers[t] = int(float(ask_d) * 100)
                            continue
                            
                        ob = ob_results.get(t)
                        if not ob:
                            raise Exception("Stale or isolated API payload drop")
                            
                        dt_market_state[t]["raw_ob"] = ob
                        
                        if t in full_markets:
                            dt_market_state[t]["status"] = full_markets[t].get("status", "")
                            dt_market_state[t]["result"] = full_markets[t].get("result", "")
                            dt_market_state[t]["last_price_dollars"] = full_markets[t].get("last_price_dollars", "0")
                            
                        ob_data = ob.get("orderbook_fp", {})
                        y_pts = ob_data.get("yes_dollars", [])
                        n_pts = ob_data.get("no_dollars", [])

                        # ── Snapshot raw orderbook BEFORE scrub for blip diagnostics ──
                        # y_pts / n_pts get reassigned below; preserve top 3 so we can
                        # compare raw vs cleaned if a blip fires.
                        try:
                            _raw_y_sorted = sorted(y_pts or [], key=lambda x: -float(x[0]))[:3]
                            _raw_n_sorted = sorted(n_pts or [], key=lambda x: -float(x[0]))[:3]
                            _raw_y_top = [(int(round(float(p[0]) * 100)), float(p[1])) for p in _raw_y_sorted]
                            _raw_n_top = [(int(round(float(p[0]) * 100)), float(p[1])) for p in _raw_n_sorted]
                        except Exception:
                            _raw_y_top = []
                            _raw_n_top = []

                        # CRITICAL FIX: Scrub physical resting volume from our own account exactly out of the arrays
                        # to strictly prevent Kalshi immediately rejecting the crossing IOC series execution (Wash Trading Limits)!
                        # 2026-06-07: skip orders in _cancel_in_flight — Kalshi has
                        # likely already removed them from the book, so subtracting
                        # our size phantom-drops the level (real bug: caused 20c top
                        # oscillation that bot chased via cancel-replace loop).
                        _in_flight = quoter.exec_engine._cancel_in_flight
                        cleaned_y_pts = []
                        for pt in y_pts:
                            cents = int(round(float(pt[0]) * 100))
                            vol = float(pt[1])
                            for cid, q in quoter.exec_engine.active_quotes.items():
                                if q.order_id in _in_flight:
                                    continue
                                if q.ticker == t and q.status.name in ["LIVE", "PARTIALLY_FILLED"] and q.kalshi_side == "yes" and q.side.name == "BID" and q.limit_cents == cents:
                                    vol -= float(q.size - q.filled_count)
                            if vol >= 1.0: cleaned_y_pts.append([pt[0], vol])

                        cleaned_n_pts = []
                        for pt in n_pts:
                            cents = int(round(float(pt[0]) * 100))
                            vol = float(pt[1])
                            for cid, q in quoter.exec_engine.active_quotes.items():
                                if q.order_id in _in_flight:
                                    continue
                                if q.ticker == t and q.status.name in ["LIVE", "PARTIALLY_FILLED"] and q.kalshi_side == "no" and q.side.name == "BID" and q.limit_cents == cents:
                                    vol -= float(q.size - q.filled_count)
                            if vol >= 1.0: cleaned_n_pts.append([pt[0], vol])

                        y_pts = cleaned_y_pts
                        n_pts = cleaned_n_pts
                        ob_data["yes_dollars"] = y_pts
                        ob_data["no_dollars"] = n_pts

                        y_pts.sort(key=lambda x: float(x[0]), reverse=True)
                        n_pts.sort(key=lambda x: float(x[0]), reverse=True)

                        b_bid = int(float(y_pts[0][0]) * 100) if y_pts else 0
                        b_offer = 100 - int(float(n_pts[0][0]) * 100) if n_pts else 100

                        # ── Data-integrity blip detector + REST truth-oracle ──
                        # Compares this tick's top to previous tick. If either side
                        # moved >= BLIP_THRESHOLD cents in a single tick, fetches a
                        # FRESH AUTHED REST orderbook for this ticker (production
                        # `client._get` — same wire as the orderbook poller) and
                        # uses REST as ground truth IF it disagrees with WS by
                        # ≥ REST_OVERRIDE_THRESHOLD cents on either side.
                        #
                        # Rationale (2026-06-13 KTGEN-GEN incident): WS delta book
                        # transiently goes empty for ~300ms; the recovery makes the
                        # arber compare empty-book "prev" to recovered "now" and
                        # fire a fake LEAD-LAG SERIES taker at 90c. REST has the
                        # real book the whole time. Fetching REST ONLY when the
                        # blip detector triggers means we pay zero baseline REST
                        # cost in the 95% case where WS is healthy — REST is the
                        # truth oracle ONLY for suspect events.
                        #
                        # Three outcomes per blip:
                        #   * REST agrees with cleaned WS  → genuine market move,
                        #     proceed normally. Log [BLIP REST-AGREE].
                        #   * REST disagrees with cleaned WS by ≥ override
                        #     threshold → corrupted feed. Replace y_pts/n_pts/
                        #     b_bid/b_offer/ob_data with REST data so downstream
                        #     bots see REST truth this tick. Update _blip_prev_*
                        #     so the next-tick comparison doesn't see a fake
                        #     recovery from the corrupted state. Log
                        #     [BLIP REST-OVERRIDE].
                        #   * REST fetch fails (429, timeout) → fall back to WS
                        #     (best we have). Log [BLIP REST-ERR]. Trades
                        #     downstream proceed at their own risk.
                        try:
                            if not hasattr(quoter, "_blip_prev_bid"):
                                quoter._blip_prev_bid = {}
                                quoter._blip_prev_offer = {}
                            BLIP_THRESHOLD = 5  # cents — triggers the REST check
                            REST_OVERRIDE_THRESHOLD = 3  # cents — REST overrides WS
                            _pb = quoter._blip_prev_bid.get(t)
                            _po = quoter._blip_prev_offer.get(t)
                            if _pb is not None and _po is not None:
                                _db = abs(b_bid - _pb)
                                _do = abs(b_offer - _po)
                                if _db >= BLIP_THRESHOLD or _do >= BLIP_THRESHOLD:
                                    _cln_y = [(int(round(float(p[0]) * 100)), float(p[1])) for p in y_pts[:3]]
                                    _cln_n = [(int(round(float(p[0]) * 100)), float(p[1])) for p in n_pts[:3]]
                                    _ours = []
                                    _if_n = 0
                                    for _cid, _q in quoter.exec_engine.active_quotes.items():
                                        if _q.ticker == t and _q.status.name in ["LIVE", "PARTIALLY_FILLED"]:
                                            _if = _q.order_id in quoter.exec_engine._cancel_in_flight
                                            if _if:
                                                _if_n += 1
                                            _ours.append(
                                                f"{_q.kalshi_side}/{_q.side.name}@{_q.limit_cents}c×{_q.size}"
                                                f"{'(IF)' if _if else ''}"
                                            )
                                    log.warning(
                                        f"[BLIP] {t} bid {_pb}->{b_bid}(d{_db}) "
                                        f"offer {_po}->{b_offer}(d{_do}) | "
                                        f"raw_y={_raw_y_top} raw_n={_raw_n_top} | "
                                        f"cln_y={_cln_y} cln_n={_cln_n} | "
                                        f"in_flight={_if_n} | ours={_ours}"
                                    )
                                    # ── REST truth-oracle pull ──
                                    # Authed via client._get → matches the wire
                                    # bytes orderbook_poller uses. asyncio.to_thread
                                    # so we don't block the main loop.
                                    try:
                                        # Translate synthetic → real for the
                                        # wire URL. The response is consumed
                                        # purely as orderbook levels (no
                                        # ticker field used) so no reverse
                                        # translation needed. Pass-through
                                        # for non-aliased.
                                        _rest_path = f"/trade-api/v2/markets/{ticker_aliases.resolve(t)}/orderbook"
                                        _rest_resp = await asyncio.to_thread(client._get, _rest_path)
                                        _rest_ob = (_rest_resp or {}).get("orderbook_fp") or {}
                                        _rest_y = _rest_ob.get("yes_dollars") or []
                                        _rest_n = _rest_ob.get("no_dollars") or []
                                        # Kalshi REST returns levels UNSORTED
                                        # (typically ascending — deep tail first).
                                        # Earlier this file assumed _rest_y[0] was
                                        # the best bid, but that grabs the 1¢ tail
                                        # instead of the actual top-of-book. That
                                        # parser bug spawned the 07:20 fake-blip
                                        # storm where REST "looked" (1, 99) for
                                        # every ticker. Match orderbook_poller.py:317
                                        # which uses max() across all price levels.
                                        # Apply same dust filter as WS scrubber (>= 1.0 lots)
                                        # so REST and WS agree on top-of-book. Without this,
                                        # a persistent <1-lot bid at a better price makes REST
                                        # and cleaned WS disagree every tick → infinite BLIP.
                                        # Observed 2026-07-02 on 100TRRQ-100T with a 0.35-lot
                                        # NO bid at 33c persistently overriding to offer=67c
                                        # while WS-cleaned showed offer=72c.
                                        _rest_bid = max(
                                            (int(round(float(_l[0]) * 100)) for _l in _rest_y
                                             if _l and len(_l) >= 2 and float(_l[1]) >= 1.0),
                                            default=0,
                                        )
                                        _rest_no_bid = max(
                                            (int(round(float(_l[0]) * 100)) for _l in _rest_n
                                             if _l and len(_l) >= 2 and float(_l[1]) >= 1.0),
                                            default=0,
                                        )
                                        _rest_offer = (100 - _rest_no_bid) if _rest_no_bid > 0 else 100
                                        _ws_vs_rest_db = abs(_rest_bid - b_bid)
                                        _ws_vs_rest_do = abs(_rest_offer - b_offer)
                                        # Wide-spread sanity gate. Empirical
                                        # pattern observed 2026-06-13: corrupted
                                        # feeds (whether WS or REST) ALWAYS show
                                        # the "tail-only" pattern — top yes bid
                                        # at 1¢ and top yes ask at 99¢, i.e. a
                                        # ~98¢ spread. Real markets (even
                                        # pre-game underdogs) have spreads ≤ 5c.
                                        # So: any source with spread > WIDE_SPREAD_C
                                        # is untrustworthy this tick, full stop.
                                        #
                                        # Decision matrix:
                                        #   WS tight + REST tight  → both real,
                                        #     compare; if diff ≥ override
                                        #     threshold, OVERRIDE with REST
                                        #     (real disagreement). Else AGREE.
                                        #   WS wide + REST tight   → WS corrupt
                                        #     (03:50 KTGEN-GEN morning case),
                                        #     OVERRIDE with REST.
                                        #   WS tight + REST wide   → REST corrupt
                                        #     (07:20 KTGEN-KT case during 429
                                        #     storm), REJECT, trust WS.
                                        #   WS wide + REST wide    → both corrupt,
                                        #     SKIP, do nothing — neither source
                                        #     is reliable, wait for next tick.
                                        WIDE_SPREAD_C = 70
                                        _ws_spread = b_offer - b_bid
                                        _rest_spread = _rest_offer - _rest_bid
                                        _ws_wide = _ws_spread > WIDE_SPREAD_C
                                        _rest_wide = _rest_spread > WIDE_SPREAD_C
                                        if _ws_wide and _rest_wide:
                                            log.warning(
                                                f"[BLIP BOTH-WIDE] {t} "
                                                f"ws=({b_bid},{b_offer}) spread={_ws_spread}c, "
                                                f"rest=({_rest_bid},{_rest_offer}) spread={_rest_spread}c → "
                                                f"both sources unreliable, skipping"
                                            )
                                        elif _rest_wide and not _ws_wide:
                                            log.warning(
                                                f"[BLIP REST-REJECT] {t} "
                                                f"ws=({b_bid},{b_offer}) spread={_ws_spread}c, "
                                                f"rest=({_rest_bid},{_rest_offer}) spread={_rest_spread}c → "
                                                f"REST wide, trusting WS"
                                            )
                                        elif (_ws_vs_rest_db >= REST_OVERRIDE_THRESHOLD
                                                or _ws_vs_rest_do >= REST_OVERRIDE_THRESHOLD):
                                            # Provenance: was the "ws" book here
                                            # actually served by WS this tick, or
                                            # was it already a REST fallback? The
                                            # latter is REST-vs-REST and points at
                                            # something other than WS divergence.
                                            _prov = ("REST-fallback"
                                                     if t in globals().get("_ws_fallback_this_tick", ())
                                                     else "WS-served")
                                            log.warning(
                                                f"[BLIP REST-OVERRIDE] {t} "
                                                f"ws=({b_bid},{b_offer}) rest=({_rest_bid},{_rest_offer}) "
                                                f"diff=({_ws_vs_rest_db},{_ws_vs_rest_do}) → trusting REST "
                                                f"| src={_prov} miss=[{ws_miss_stats.describe(t)}]"
                                            )
                                            # Replace the orderbook this tick.
                                            y_pts = list(_rest_y)
                                            n_pts = list(_rest_n)
                                            y_pts.sort(key=lambda x: float(x[0]), reverse=True)
                                            n_pts.sort(key=lambda x: float(x[0]), reverse=True)
                                            ob_data["yes_dollars"] = y_pts
                                            ob_data["no_dollars"] = n_pts
                                            b_bid = _rest_bid
                                            b_offer = _rest_offer
                                            # Prevent next-tick "recovery" blip
                                            # firing when WS catches up to match
                                            # REST. Without this, the WS recovery
                                            # would be compared against the
                                            # overridden prev → another blip → arb.
                                            _pb = _rest_bid
                                            _po = _rest_offer
                                            # ── Auto-recovery: force WS reconnect
                                            # when a single ticker keeps overriding.
                                            # If WS state is persistently corrupted
                                            # for a ticker, REST overrides will fire
                                            # every loop iteration (we observed 854
                                            # in 5 min on KTGEN-KT on 2026-06-13).
                                            # That works but burns REST budget. After
                                            # OVERRIDE_TRIGGER_COUNT overrides for a
                                            # single ticker in 10s, force a full WS
                                            # reconnect to refresh snapshots and
                                            # clear the corrupted book.
                                            # RECONNECT_COOLDOWN prevents reconnect
                                            # storms if multiple tickers are stuck.
                                            try:
                                                import time as _t_mod
                                                _now_ts = _t_mod.time()
                                                if not hasattr(quoter, "_blip_rest_override_history"):
                                                    quoter._blip_rest_override_history = {}
                                                    quoter._last_force_reconnect_ts = 0.0

                                                # ── (1) SETTLED-MARKET EXCLUSION (2026-07-27) ──
                                                # A decided market (REST at 0/1 or
                                                # 99/100) has NO tradeable book. WS
                                                # correctly holds nothing for it, REST
                                                # returns a stub, and the diff reads as
                                                # ~99c — so three ticks of a worthless
                                                # market tore down the SHARED connection
                                                # for every live market.
                                                #
                                                # Measured 2026-07-27: rest=(0,1) 807x +
                                                # rest=(99,100) 620x = 36% of ALL override
                                                # fuel came from already-decided markets.
                                                # The very first forced reconnect of the
                                                # run (26 Jul 22:22:40) was STXJEN — a
                                                # Dota game that had ended hours earlier.
                                                #
                                                # The override itself still happens (the
                                                # book is still corrected from REST); we
                                                # only refuse to let it justify killing
                                                # the feed.
                                                _settled_book = (_rest_offer <= 2 or _rest_bid >= 98)
                                                # ── (3) REST-vs-REST (2026-07-28) ──
                                                # If this ticker fell back to REST
                                                # this tick, `ob_data` IS REST — so
                                                # this "divergence" is one REST
                                                # snapshot vs another, and says
                                                # NOTHING about WS health. Counting
                                                # it toward a forced reconnect tears
                                                # down the shared feed over an
                                                # artifact of our own fallback.
                                                #
                                                # Measured 09:38-10:26: 349 of 353
                                                # overrides were src=REST-fallback
                                                # (4 genuinely WS-served), and ALL 6
                                                # forced reconnects were triggered by
                                                # tickers in REST fallback — two
                                                # quiet pregame Dota books
                                                # (TTNS, PCKCPNH) whose 60s stale
                                                # guard put them on REST in the first
                                                # place. Uptime collapsed 600s→160s.
                                                # The book is still corrected from
                                                # REST; we only refuse to let this
                                                # justify killing the feed.
                                                _rest_sourced = t in globals().get(
                                                    "_ws_fallback_this_tick", ())
                                                # ── (2) RE-SEED GRACE (2026-07-27) ──
                                                # A forced reconnect closes the socket,
                                                # waits out the backoff (pinned at its
                                                # 30s cap), re-subscribes and awaits fresh
                                                # snapshots — ~35-45s during which EVERY
                                                # ticker reads empty and overrides. Those
                                                # overrides are artifacts of our own
                                                # reconnect, so counting them made each
                                                # reconnect trigger the next one.
                                                # Measured: 2.87 overrides/s in the first
                                                # 10s after a reconnect vs 0.10/s at
                                                # 30-60s — a 28x collapse once books
                                                # re-seed.
                                                _in_reseed = (
                                                    _now_ts - getattr(quoter, "_last_force_reconnect_ts", 0.0)
                                                    < RESEED_GRACE_S
                                                )
                                                if _settled_book or _in_reseed or _rest_sourced:
                                                    # Do not count this override toward a
                                                    # reconnect. Log rarely — this fires
                                                    # thousands of times a session.
                                                    if _now_ts - globals().get("_blip_skip_log_ts", 0.0) >= 300.0:
                                                        globals()["_blip_skip_log_ts"] = _now_ts
                                                        log.info(
                                                            f"[BLIP RECONNECT-SKIP] {t} override not counted "
                                                            f"({'settled book ' if _settled_book else ''}"
                                                            f"{'post-reconnect re-seed ' if _in_reseed else ''}"
                                                            f"{'REST-vs-REST (ticker was in REST fallback)' if _rest_sourced else ''}) "
                                                            f"rest=({_rest_bid},{_rest_offer}) — book still "
                                                            f"corrected from REST, feed left alone"
                                                        )
                                                else:
                                                    _hist = quoter._blip_rest_override_history.setdefault(t, [])
                                                    _hist.append(_now_ts)
                                                    # Prune entries older than 10s
                                                    quoter._blip_rest_override_history[t] = [
                                                        _ts for _ts in _hist if _now_ts - _ts < 10.0
                                                    ]
                                                    if (len(quoter._blip_rest_override_history[t]) >= OVERRIDE_TRIGGER_COUNT
                                                            and _now_ts - quoter._last_force_reconnect_ts > RECONNECT_COOLDOWN_S
                                                            and ws_delta_mgr is not None):
                                                        log.warning(
                                                            f"[WS_DELTA AUTO-RECONNECT] {t} triggered "
                                                            f"{len(quoter._blip_rest_override_history[t])} REST overrides "
                                                            f"in 10s — forcing full WS reconnect to refresh all snapshots"
                                                        )
                                                        quoter._last_force_reconnect_ts = _now_ts
                                                        quoter._blip_rest_override_history.clear()
                                                        asyncio.create_task(ws_delta_mgr._force_reconnect())
                                            except Exception as _ar_e:
                                                log.error(f"[WS_DELTA AUTO-RECONNECT] error on {t}: {_ar_e}")
                                        else:
                                            log.info(
                                                f"[BLIP REST-AGREE] {t} "
                                                f"ws=({b_bid},{b_offer}) rest=({_rest_bid},{_rest_offer}) — "
                                                f"genuine market move, proceeding"
                                            )
                                    except Exception as _re:
                                        log.error(f"[BLIP REST-ERR] {t}: {type(_re).__name__}: {_re}")
                            quoter._blip_prev_bid[t] = b_bid
                            quoter._blip_prev_offer[t] = b_offer
                        except Exception as _e:
                            log.debug(f"[BLIP] detector error on {t}: {_e}")
                        
                        top_level_bids[t] = b_bid
                        top_level_offers[t] = b_offer
                    except Exception as e:
                        log.error(f"Failed fetching mapping for {t}: {e}")
                        top_level_bids[t] = 0
                        top_level_offers[t] = 100
                        
                # ── Esports map prices: data_source-driven routing ──
                # Routing policy is per-config (data_source column in market_parameters.csv):
                #   "poly"               → always Poly (Tier 1 CS — sponsored, 1c books 24/7)
                #   "kalshi_na_poly_eu"  → Poly during EU hours, Kalshi WS during NA hours
                #   "kalshi"             → always Kalshi WS
                # Fallback: time-based check via is_na_hours (preserves pre-Tier-1 behavior).
                if esports_map_tickers:
                    from esports_config import is_na_hours as _is_na_hours

                    # Map ticker → series ticker → config's data_source.
                    # Map tickers don't have configs directly; look up the parent series.
                    ds_by_ticker = getattr(quoter, "data_source_by_ticker", {})

                    # BO5 events: ALWAYS Poly-only, regardless of data_source or
                    # time-of-day. Kalshi doesn't list G4/G5 winner markets for
                    # most esports BO5 events, and routing to Kalshi WS leaves
                    # G3/G4/G5 with no orderbook data → can_trade=False forever.
                    # We build the set of BO5 series ticker bases from active
                    # bots' is_bo5 flag (set during _load_pre_game_probabilities
                    # from CSV col 6).
                    bo5_series_bases = set()
                    for b in getattr(quoter, "bots", []):
                        if getattr(b, "is_bo5", False):
                            t = getattr(b.config, "ticker", "")
                            t_parts = t.split("-")
                            if len(t_parts) >= 2:
                                bo5_series_bases.add(f"{t_parts[0]}-{t_parts[1]}")

                    def _route_to_poly(map_ticker: str) -> bool:
                        # Derive the series ticker from a map ticker:
                        # KXLOLMAP-EVENT-1-TEAM → KXLOLGAME-EVENT-TEAM
                        # KXATPSETWINNER-EVENT-1-TEAM → KXATPMATCH-EVENT-TEAM
                        parts = map_ticker.split("-")
                        if len(parts) >= 4:
                            game_prefix = parts[0].replace("MAP", "GAME").replace("SETWINNER", "MATCH")
                            # BO5 override: force Poly for any map of a BO5 event.
                            series_base = f"{game_prefix}-{parts[1]}"
                            if series_base in bo5_series_bases:
                                return True
                            series_t = f"{game_prefix}-{parts[1]}-{parts[3]}"
                            ds = ds_by_ticker.get(series_t) or ds_by_ticker.get(map_ticker)
                        else:
                            ds = ds_by_ticker.get(map_ticker)
                        if ds == "poly":
                            return True
                        if ds == "kalshi":
                            return False
                        # "kalshi_na_poly_eu" or unmapped → time-of-day decides
                        return not _is_na_hours(map_ticker)

                    poly_map_tickers = [t for t in esports_map_tickers if _route_to_poly(t)]
                    # Settled maps excluded (2026-07-28): they receive no deltas,
                    # so they sit stale forever and are scrubbed anyway. Their
                    # result is already authoritative from /markets metadata and
                    # cached by hedge_engine._recover_map_wl, which is built for
                    # exactly this ("settled maps drop out of full_market_state").
                    kalshi_map_tickers = [t for t in esports_map_tickers
                                          if not _route_to_poly(t)
                                          and t not in esports_finalized]

                    # Drive the dedicated map feed with EXACTLY the Kalshi-routed
                    # maps. Empty when every map is data_source=poly → feed idle.
                    # set_tickers diffs internally (subscribe new / drop gone), so
                    # this is cheap to call every loop.
                    #
                    # WARM STANDBY (26AUG19, FULL-WS FALLBACK Phase 0): with
                    # enable_map_ws_warm_standby.flag present, ALSO subscribe the
                    # poly-routed live maps so the fallback WS-shadow comparator
                    # has books to judge. Subscription-only — kalshi_map_tickers
                    # (the ROUTING set) is untouched, so map theo sourcing is
                    # bit-identical. `rm` the flag → set drains next loop.
                    if map_delta_mgr is not None:
                        try:
                            _ws_sub = set(kalshi_map_tickers)
                            if os.path.exists("enable_map_ws_warm_standby.flag"):
                                _ws_sub |= {t for t in esports_map_tickers
                                            if t not in esports_finalized}
                            await map_delta_mgr.set_tickers(_ws_sub)
                            # Polled liveness heartbeat (≤ once/60s) — reads the
                            # stats counters directly, so it reflects real delta
                            # flow even though the manager doesn't emit per-delta
                            # events. Only logs when maps are actually routed here.
                            _hb_now = _time_mod.time()
                            if _ws_sub and _hb_now - _map_ws_hb["ts"] > 60.0:
                                _map_ws_hb["ts"] = _hb_now
                                _s = map_delta_mgr.stats
                                _n_live = sum(1 for b in map_delta_mgr.books.values() if b.is_live)
                                log.info(f"[MAP WS] heartbeat live={_n_live}/{len(map_delta_mgr.books)} "
                                         f"routed={len(kalshi_map_tickers)} "
                                         f"warm={len(_ws_sub) - len(kalshi_map_tickers)} "
                                         f"snapshots={_s.get('snapshot', 0)} "
                                         f"deltas_applied={_s.get('delta_applied', 0)}")
                        except Exception as _mws_e:
                            log.error(f"[MAP WS] set_tickers failed: "
                                      f"{type(_mws_e).__name__}: {_mws_e}")

                    poly_events = []
                    seen_bases = set()
                    for t in poly_map_tickers:
                        parts = t.split("-")
                        if len(parts) >= 4:
                            game_prefix = parts[0].replace("MAP", "GAME").replace("SETWINNER", "MATCH")
                            event_base = game_prefix + "-" + parts[1]
                            team = parts[3]
                            if event_base not in seen_bases:
                                seen_bases.add(event_base)
                                poly_events.append((event_base, team))

                    # ── KALSHI (WS or REST per book_source) for NA-hours tickers ──
                    if kalshi_map_tickers:
                        # 2026-07-18 REGRESSION FIX: use ws_fetcher, NOT book_view.
                        #
                        # Until the ws_delta migration (2026-06-13) this line read
                        # `ws_fetcher.fetch_all(kalshi_map_tickers)` (see git
                        # de0afd0:run.py L572) and worked for weeks. The migration
                        # swapped it to book_view, which routes to WSDeltaBookView —
                        # a reader over the LONG-LIVED WSDeltaManager book cache.
                        # That cache only holds tickers passed to set_tickers(), and
                        # run.py passes set(obs_to_fetch), which EXCLUDES map tickers
                        # (they're filtered out at the obs_to_fetch comprehension).
                        # set_tickers also actively pops books for any ticker not in
                        # the target set, so a map ticker can never be seeded.
                        #
                        # Net effect from 2026-06-13: get_orderbook_fp(map) returned
                        # None every cycle → WSDeltaBookView silently dropped it →
                        # the ws_missed REST safety net didn't cover it either (that
                        # is also scoped to obs_to_fetch) → the ONLY thing left was
                        # the 300s /markets metadata fallback below. Kalshi map
                        # prices were therefore up to 5 minutes stale for over a
                        # month, undetected, because the fallback made it look live.
                        #
                        # WSOrderbookFetcher.fetch_all opens a FRESH connection per
                        # call and subscribes the tickers it is handed, so it needs
                        # no pre-registration and cannot serve a stale cache.
                        # TIER 1 — dedicated map delta feed (own connection,
                        # isolated from series). Maps are subscribed via
                        # map_delta_mgr.set_tickers() above, so this is the fast
                        # path: no per-cycle reconnect, same WSBook reconstruction
                        # series tickers use, but on a connection that can't be
                        # starved by series volume. get_orderbook_fp returns None
                        # for any unseeded/stale/empty-but-live book → that ticker
                        # drops to the TIER 2 fresh-WS rescue below. Falls back to
                        # the shared book_view only if the dedicated feed failed to
                        # start (MAP_WS_DEDICATED_DISABLE or a start() exception).
                        if map_delta_mgr is not None:
                            ws_map_results = {}
                            for _mt in kalshi_map_tickers:
                                _ob = map_delta_mgr.get_orderbook_fp(_mt)
                                if _ob is not None:
                                    ws_map_results[_mt] = _ob
                        else:
                            ws_map_results = await book_view.fetch_all(kalshi_map_tickers)
                        ws_populated = apply_map_books(
                            ws_map_results, top_level_bids, top_level_offers, dt_market_state
                        )

                        # TIER 2 — LIVE rescue. WSDeltaBookView returns nothing for
                        # a ticker that is unseeded or stale beyond STALE_BOOK_S
                        # (60s), e.g. right after a subscribe or a reconnect. Fetch
                        # those with a fresh WS snapshot. This is a fallback to
                        # LIVE data, never to a cache — the distinction that
                        # matters. Only the missed subset is fetched, so the
                        # ~550ms connect+snapshot cost is paid rarely, not every
                        # cycle.
                        _map_missed = [t for t in kalshi_map_tickers if t not in ws_populated]
                        if _map_missed:
                            try:
                                _rescue = await ws_fetcher.fetch_all(_map_missed)
                            except Exception as e:
                                _rescue = {}
                                log.error(f"[KALSHI MAP RESCUE] fresh-WS fetch failed for "
                                          f"{len(_map_missed)} ticker(s): {type(e).__name__}: {e}")
                            _rescued = apply_map_books(
                                _rescue, top_level_bids, top_level_offers, dt_market_state
                            )
                            ws_populated |= _rescued
                            if _rescued:
                                log.info(f"[KALSHI MAP RESCUE] delta book missed "
                                         f"{len(_map_missed)} map ticker(s); fresh WS "
                                         f"recovered {len(_rescued)}: {sorted(_rescued)[:6]}")

                        # ── KALSHI MAP NO-FALLBACK (2026-07-18) ──
                        # EMPTY DATA MEANS NO TRADE. A kalshi-source map ticker
                        # with no fresh WS book this cycle is force-absented so
                        # the active-map guard suppresses, exactly like the
                        # PREGAME MAP SCRUB does on the poly side.
                        #
                        # This previously fell back to full_markets — the 300s
                        # /markets *metadata* cache (MAP_METADATA_REFRESH_SEC),
                        # which is a status sanity check, NOT trading data. That
                        # served a map price up to 5 minutes stale and was
                        # indistinguishable from live to every downstream gate,
                        # so the theo froze between bulk fetches and phantom edge
                        # accumulated as the real book moved away from it.
                        #
                        # 100TG2 2026-07-18: map-1 collapsed 68→75→90 while the
                        # G2 series theo sat pinned at 46.4 for 4 minutes (theo
                        # stepped only on the 300s fetch boundaries 21:21:29 /
                        # 21:26:29 / 21:31:29). A taker fired 1,000 lots into a
                        # 9.4c phantom edge at t+241s into the stale window; the
                        # theo then corrected 7.1c downward 59s later.
                        _kalshi_map_stale = scrub_unpopulated_maps(
                            kalshi_map_tickers, ws_populated,
                            top_level_bids, top_level_offers,
                        )
                        if _kalshi_map_stale:
                            log.warning(
                                f"[KALSHI MAP NO-FALLBACK] forced {len(_kalshi_map_stale)} "
                                f"kalshi-source map ticker(s) absent (no fresh WS book this "
                                f"cycle): {sorted(_kalshi_map_stale)[:6]}"
                            )

                    # ── POLY CLOB for EU-hours tickers ──
                    if poly_map_tickers:
                        injected_map_tickers = set()
                        # to_thread: inject_all does blocking HTTP. Called
                        # inline it stalled the event loop for the full
                        # duration, so nothing else — the Kalshi read path,
                        # quoting, cancels — could run while Poly was in
                        # flight. It mutates the dicts passed in, which is safe
                        # because we await it: no other coroutine touches them
                        # concurrently.
                        failed_events = await asyncio.to_thread(
                            poly_feed.inject_all,
                            poly_events, top_level_bids, top_level_offers, dt_market_state,
                            full_market_state=full_markets,
                            injected_out=injected_map_tickers,
                        )

                        # ── PREGAME MAP SCRUB (2026-07-17, EFKC phantom-hedge fix) ──
                        # A poly-source map ticker is seeded ~L787 with a Kalshi default
                        # (yes_bid/ask from _map_meta_cache). For date-divergent VAL/Dota
                        # maps that Kalshi book is FROZEN at the pregame ~50/50 snapshot —
                        # a value Polymarket cannot produce. When Poly's overwrite skips a
                        # leg this cycle (feed churn / mid re-discovery), the pregame
                        # default STANDS, the live-series guard (empty-only) waves the
                        # non-empty book through, and we fire into a phantom ~20c edge
                        # (EFKC 2026-07-17: ~101 reverts, -$7,898). Fix: force-absent any
                        # poly-source map ticker not FRESHLY injected this cycle so the
                        # active-map guard suppresses. Default-on; kill via flag.
                        if not os.path.exists("disable_pregame_map_scrub.flag"):
                            _scrubbed = []
                            for _mt in poly_map_tickers:
                                if _mt not in injected_map_tickers and (
                                        _mt in top_level_bids or _mt in top_level_offers):
                                    top_level_bids.pop(_mt, None)
                                    top_level_offers.pop(_mt, None)
                                    _scrubbed.append(_mt)
                            if _scrubbed:
                                log.info(
                                    f"[PREGAME MAP SCRUB] forced {len(_scrubbed)} poly-source "
                                    f"map ticker(s) absent (no fresh Poly inject this cycle): "
                                    f"{sorted(_scrubbed)[:6]}"
                                )

                        # data_source=poly is AUTHORITATIVE — no Kalshi fallback.
                        # (2026-07-15, EFTL falling-knife post-mortem) These are all
                        # poly-source events. Previously, when Poly was unavailable
                        # (beyond the feed's <10s stale cache) we fetched the Kalshi
                        # map book here. For date-divergent games that Kalshi ticker
                        # is a PHANTOM empty book (series date ≠ map date), and an
                        # empty/partial book gets synthesized into a pregame-priced
                        # "56 at 57" market → phantom edge → we caught a falling knife
                        # all game. poly→Poly, kalshi→Kalshi, never cross. On Poly
                        # failure we leave the map tickers ABSENT so the live-series
                        # active-map guard suppresses theos for this cycle.
                        if failed_events:
                            log.warning(
                                f"[POLY FEED] Poly unavailable for {len(failed_events)} "
                                f"poly-source event(s) → suppressing this cycle "
                                f"(NO Kalshi fallback): {failed_events[:5]}"
                            )

                    # Populate status/result from Kalshi metadata for ALL map tickers
                    for t in esports_map_tickers:
                        if t not in dt_market_state:
                            dt_market_state[t] = {}
                        if t in full_markets:
                            dt_market_state[t]["status"] = full_markets[t].get("status", "")
                            dt_market_state[t]["result"] = full_markets[t].get("result", "")
                            dt_market_state[t]["last_price_dollars"] = full_markets[t].get("last_price_dollars", "0")

                    # SAME for series tickers (obs_to_fetch). Without this, the
                    # series ticker's `status` stays "" forever because the
                    # orderbook-polled path at ~line 463 has `continue` for
                    # tickers IN obs_to_fetch. BO5 `can_trade` requires
                    # `series_status == "active"` (arber_bot.py:912) and so
                    # blocked every BO5 event with `[BO5-CANTRADE] series_status=''`.
                    # Mirrors the map-ticker block above — same fields, same source.
                    # 2026-06-12 fix (T1HLE BO5-CANTRADE incident).
                    for t in obs_to_fetch:
                        if t not in dt_market_state:
                            dt_market_state[t] = {}
                        if t in full_markets:
                            dt_market_state[t]["status"] = full_markets[t].get("status", "")
                            dt_market_state[t]["result"] = full_markets[t].get("result", "")
                            dt_market_state[t]["last_price_dollars"] = full_markets[t].get("last_price_dollars", "0")

                # --- Compile hybrid dynamic market state seamlessly ---
                
                # 1. Dynamically Auto-Detect MLB Pre-Game Expected Totals from active Orderbooks!
                game_expected_totals = {}
                for t in active_tickers:
                    if "MLB" in t and "TOT" in t:
                        parts = t.split("-")
                        if len(parts) >= 2:
                            game_tag = parts[1]
                            t_bid = top_level_bids.get(t, 0)
                            t_offer = top_level_offers.get(t, 100)
                            
                            # Valid competitive spread exists on this quoting line:
                            if t_bid > 0 and t_offer < 100 and (t_offer - t_bid) <= 35:
                                try:
                                    target_line = float(parts[-1]) - 0.5
                                    mid_pct = (t_bid + t_offer) / 2.0
                                    inferred_total = target_line + ((mid_pct - 50.0) / 30.0)
                                    if game_tag not in game_expected_totals:
                                        game_expected_totals[game_tag] = inferred_total
                                    else:
                                        # Aggregate linearly via exponential smoothing (average all lines tracked)
                                        game_expected_totals[game_tag] = (game_expected_totals[game_tag] + inferred_total) / 2.0
                                except ValueError:
                                    pass

                # 2. Support user fallback static definitions from CSV if provided!
                static_totals = {}
                try:
                    import csv
                    if os.path.exists("mlb_expected_totals.csv"):
                        with open("mlb_expected_totals.csv", "r") as f:
                            for row in csv.reader(f):
                                if len(row) >= 2:
                                    # Expected Struct = TeamHint (e.g. CHC), TotalValue (e.g. 6.5)
                                    static_totals[row[0].upper().strip()] = float(row[1])
                except Exception:
                    pass

                if mlb_tracker:
                    for ticker in active_tickers:
                        if "MLB" not in ticker: continue
                        
                        parts = ticker.split("-")
                        if len(parts) >= 2:
                            game_tag = parts[1]
                            
                            team_hint = None
                            tail = game_tag[-6:]
                            for abbr in mlb_game_state.KALSHI_TO_MLB.keys():
                                if abbr in tail:
                                    team_hint = abbr
                                    break
                            
                            if team_hint:
                                gs = mlb_tracker.get(team_hint)
                            else:
                                gs = None
                                
                            if gs and gs.get("status") == "Live":
                                s_str = gs.get("state_str", "")
                                if "Top" in s_str or "Bot" in s_str:
                                    is_top = "Top" in s_str
                                    try:
                                        # Clean ordinal suffixes like 'st', 'nd', 'rd', 'th'
                                        raw_val = s_str.replace("Top ", "").replace("Bot ", "").strip()
                                        clean_digits = "".join(c for c in raw_val if c.isdigit())
                                        inning = int(clean_digits) if clean_digits else 1
                                    except ValueError:
                                        inning = 1
                                        
                                    # Fast-forward 1 half-inning (e.g., Top 1 -> Bot 1, Bot 1 -> Top 2)
                                    if is_top:
                                        proj_inning = inning
                                        proj_is_top = False
                                    else:
                                        proj_inning = inning + 1
                                        proj_is_top = True
                                        
                                    target_line = 1.5
                                    if "TOT" in ticker:
                                        try:
                                            # Kalshi's total tickers end in the ceiling integer (e.g., -16 for Over 15.5)
                                            # We must shift -0.5 to cleanly represent the true float boundary
                                            target_line = float(parts[-1]) - 0.5
                                        except ValueError:
                                            pass
                                    
                                    target_end = 5 if "F5" in ticker else 9
                                    
                                    # Hierarchical Total Anchoring!
                                    # 1. CSV static map (Matches BOTH Away and Home teams interchangeably!)
                                    # 2. Automated Active Orderbook Detection
                                    # 3. Last Resort League Average Backup (8.5)
                                    base_total = 8.5
                                    csv_total = None
                                    for t_key, t_val in static_totals.items():
                                        if t_key in game_tag:
                                            csv_total = t_val
                                            break
                                            
                                    if csv_total is not None:
                                        base_total = min(max(csv_total, 5.0), 14.0)
                                    elif game_tag in game_expected_totals:
                                        base_total = min(max(game_expected_totals[game_tag], 5.0), 14.0)
                                    
                                    dt_market_state[ticker] = {
                                        "inning": proj_inning,
                                        "is_top": proj_is_top,
                                        "outs": 0,
                                        "base_map": 0,
                                        "away_score": gs.get("away_score", 0),
                                        "home_score": gs.get("home_score", 0),
                                        "target_line": target_line,
                                        "expected_game_total": base_total,
                                        "target_win_prob": 0.5,
                                        "target_inning_end": target_end
                                    }

                if mls_tracker:
                    soccer_generator = generator.models.get("soccer")
                    for ticker in active_tickers:
                        if "MLS" not in ticker and "SOCCER" not in ticker: continue
                        
                        parts = ticker.split("-")
                        if len(parts) >= 3:
                            game_tag = parts[2]
                            gs = mls_tracker.get(game_tag)
                            
                            # Parse Kalshi's Sub-Title to discover exactly which leg we are quoting
                            market_type = ""
                            if ticker in full_markets:
                                yes_sub = full_markets[ticker].get("yes_sub_title", "").lower()
                                title = full_markets[ticker].get("title", "").lower()
                                
                                # Protect against Exotic / Spread Markets!
                                if "spread" in title or "handicap" in title or "+" in yes_sub or "-" in yes_sub:
                                    pass # Strictly ignore spread math!
                                elif "tie" in yes_sub or "draw" in yes_sub:
                                    market_type = "draw"
                                elif "over 1.5" in yes_sub: market_type = "over_1_5"
                                elif "over 2.5" in yes_sub: market_type = "over_2_5"
                                elif "over 3.5" in yes_sub: market_type = "over_3_5"
                                elif "over 4.5" in yes_sub: market_type = "over_4_5"
                                elif "over 5.5" in yes_sub: market_type = "over_5_5"
                                else:
                                    # It's a team win market. Match to tracked team names!
                                    # Kalshi uses exactly "[Team] wins" for moneyline!
                                    if gs: 
                                        home_name = gs.get("home_name", "").lower()
                                        away_name = gs.get("away_name", "").lower()
                                        if home_name and home_name in yes_sub and "wins" in yes_sub:
                                            market_type = "home_win"
                                        elif away_name and away_name in yes_sub and "wins" in yes_sub:
                                            market_type = "away_win"
                                            
                            if gs and market_type:
                                dt_market_state[ticker] = {
                                    "game_tag": game_tag,
                                    "minute": gs.get("minute", 0),
                                    "home_score": gs.get("home_score", 0),
                                    "away_score": gs.get("away_score", 0),
                                    "home_red": gs.get("home_red", 0),
                                    "away_red": gs.get("away_red", 0),
                                    "market_type": market_type
                                }
                                
                            # Pre-Game Auto-Calibration Routine (Fires Exactly ONCE per game_tag)
                            if soccer_generator and not soccer_generator.game_params.get(game_tag, {}).get("calibrated", False):
                                # Gather all active trackers for this game to build a target list
                                targets = {}
                                for sub_tkr in active_tickers:
                                    if game_tag in sub_tkr and sub_tkr in full_markets:
                                        t_bid = top_level_bids.get(sub_tkr, 0)
                                        t_offer = top_level_offers.get(sub_tkr, 100)
                                        
                                        # Parse sub-market 
                                        s_yes = full_markets[sub_tkr].get("yes_sub_title", "").lower()
                                        s_type = ""
                                        if "tie" in s_yes or "draw" in s_yes: s_type = "draw"
                                        elif "over 1.5" in s_yes: s_type = "over_1_5"
                                        elif "over 2.5" in s_yes: s_type = "over_2_5"
                                        elif "over 3.5" in s_yes: s_type = "over_3_5"
                                        else:
                                            if gs:
                                                h_n = gs.get("home_name", "").lower()
                                                a_n = gs.get("away_name", "").lower()
                                                if h_n and h_n in s_yes: s_type = "home_win"
                                                if a_n and a_n in s_yes: s_type = "away_win"
                                        
                                        if s_type and t_bid > 0 and t_offer < 100 and (t_offer - t_bid) <= 35:
                                            mid_pct = (t_bid + t_offer) / 200.0 # Kalshi quotes in cents!
                                            targets[s_type] = mid_pct
                                            
                                # Calibrate model utilizing the Kalshi market mid-spread inputs
                                if len(targets) >= 2:
                                    log.info(f"Auto-Calibrating Soccer Model for {game_tag} against targets: {targets}")
                                    soccer_generator.calibrate(game_tag, targets)

                # Feed staleness tracker so detect_bo3_state can distinguish
                # "freshly-moved" markets from "stuck stale" books for the
                # loose-5c rule.
                try:
                    import staleness_tracker
                    for _t, _b in top_level_bids.items():
                        staleness_tracker.observe(_t, _b)
                except Exception:
                    pass

                # ── DECIDED-EVENT DETECTOR (2026-07-27) ───────────────────
                # [FINALIZED-SKIP] above drops tickers once KALSHI says they are
                # settled — but Kalshi's status lags reality by hours.
                # KXDOTA2GAME-26JUL271400AVIVDS-VDS still reported status=active
                # 5.5h after its game ended, so it was fetched every tick: the
                # WS legitimately held nothing (no two-sided market left), the
                # book fell back to REST, and it logged 362 fallbacks in 18 min.
                #
                # A market with NO YES BIDS AT ALL, while still showing a real
                # ask, is decided — that side lost. AVIVDS-VDS sat at bid=0 /
                # ask=4 (winner at 96, not 99), which is why the price-threshold
                # test in the BLIP guard (bid>=98) missed it.
                #
                # Requiring `offer < 100` is what separates "decided" from "no
                # data": a pregame or unquoted market reads 0/100 and must NEVER
                # be retired on that basis. The dwell must be CONTINUOUS — any
                # tick with a YES bid clears it.
                #
                # Retirement is per EVENT: if either side is decided, the event
                # is over, so both tickers stop being fetched.
                try:
                    _now_dec = _time_mod.time()
                    _no_yes_since = globals().setdefault("_no_yes_since", {})
                    _decided = globals().setdefault("_decided_events", {})
                    for _t in all_esports_tickers:
                        if _t in esports_map_tickers:
                            continue
                        _b = top_level_bids.get(_t)
                        _o = top_level_offers.get(_t)
                        if _b is None or _o is None:
                            continue
                        if _b == 0 and _o < 100:
                            _first = _no_yes_since.setdefault(_t, _now_dec)
                            _held = _now_dec - _first
                            _ev = _t.rsplit("-", 1)[0]
                            if _held >= DECIDED_NO_YES_SECS and _ev not in _decided:
                                _decided[_ev] = _now_dec
                                log.warning(
                                    f"[DECIDED-EVENT] {_ev}: {_t} has had NO yes bids "
                                    f"(bid=0, ask={_o}) for {_held/60:.0f} min — event is "
                                    f"over regardless of Kalshi status; dropping its "
                                    f"series tickers from the per-tick fetch set"
                                )
                        else:
                            _no_yes_since.pop(_t, None)
                except Exception as _dec_e:
                    log.error(f"[DECIDED-EVENT] detector error (ignored): {_dec_e}")

                # ── Crossed-book scan (see detector block ~L200) ──
                # After ALL book writes (series WS loop, map books, Poly inject,
                # scrubs), before any bot evaluates. Detection always logs;
                # suppression + WS reseed only when XBOOK_ENFORCE=1.
                try:
                    xbook_shadow_scan(
                        top_level_bids, top_level_offers,
                        dt_market_state, all_esports_tickers,
                        enforce=_xbook_enforce_active(),
                        ws_delta_mgr=ws_delta_mgr,
                        reseed_fetcher=xbook_fetcher,
                        isolated_book=isolated_book,
                    )
                except Exception as _xb_e:
                    log.error(f"[XBOOK] scan error (ignored): {_xb_e}")

                await quoter.run_tick(
                    dt_market_state=dt_market_state,
                    top_level_bids=top_level_bids,
                    top_level_offers=top_level_offers
                )

            except Exception as e:
                log.error(f"Loop runtime error: {e}")
                
            # 2026-07-27: TRIED 0.1 → 0.02, REVERTED before going live. The
            # [WS FALLBACK] path at ~L950 REST-fetches every ticker the WS
            # missed, ungated, EVERY tick — measured up to 189 bursts/min in
            # run.log (26 Jul), 1-4 tickers each. Cutting the sleep 5x multiplies
            # that REST load by the same factor, and kalshi_client._wait_for_token
            # sleeps *while holding a global lock*, so hitting the 75 RPS bucket
            # would serialize every caller — including order placement. Making
            # the loop faster could make fires slower. Also unquantified: 429s
            # (see the suspected Basic-vs-Advanced tier mismatch).
            #
            # Not worth it: the Poly cache_only fix already removed ~1000ms/tick,
            # so this is ~80ms more against a real stall risk — and shipping both
            # at once would make neither attributable in the next measurement.
            # Revisit once fire→ack latency is measured (kalshi_api_trace.flag).
            _supervise_poly_ws_listener()
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        log.warning("Received Cancel/Interrupt Flag. Breaking loop.")
    except BaseException as e:
        log.warning(f"Process intercepted by {type(e).__name__}!")
    finally:
        # Guarantee physical synchronous wipe BEFORE event loop termination completes!
        log.info("Triggering SYNC Kalshi shutdown sequence...")
        quoter.shutdown_sync()
        watcher.shutdown_sync()
        asyncio.get_event_loop().run_until_complete(book_view.close())

if __name__ == "__main__":
    # SIGTERM handler — converts a kill -TERM into a KeyboardInterrupt so the
    # finally block in main() runs (shutdown_sync → cancel-all). Without this,
    # SIGTERM hard-kills Python before the finally fires, leaving resting
    # orders orphaned on Kalshi (observed 5+ times 2026-06-25..06-27, with
    # the same 16 orphan IDs persisting across reconciliation cycles because
    # the cancel endpoint was also broken; that's the companion fix in
    # manager._execute_sync_cancel). SIGINT (Ctrl-C) already maps to
    # KeyboardInterrupt via Python defaults.
    import signal
    def _sigterm_to_kbi(signum, frame):
        log.warning(f"Received signal {signum} — raising KeyboardInterrupt for clean shutdown")
        raise KeyboardInterrupt()
    try:
        signal.signal(signal.SIGTERM, _sigterm_to_kbi)
    except (ValueError, OSError):
        # signal.signal raises if not main thread; safe to ignore.
        pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
