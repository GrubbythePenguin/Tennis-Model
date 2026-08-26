"""Per-event Poly hedge staleness detector. Standalone module shared by:
  - arber_bot (producer + consumer)
  - map_arber_bot (producer + consumer)
  - bot.py / QuoterBot (consumer only)

Trigger condition (recomputed on every snapshot update), over a rolling
WINDOW_SEC window with ≥ MIN_SAMPLES per team:

    ENTER : min(hedge_range_a, hedge_range_b) <= HEDGE_TOLERANCE
            AND max(series_range_a, series_range_b) >= SERIES_RANGE_THRESHOLD
       OR : hedge frozen >= FREEZE_MIN_SEC AND the series moved
            >= FREEZE_SERIES_DANGER_C cents SINCE THE FREEZE BEGAN
            (longer lookback than the rolling window — see below)
    CLEAR : min(hedge_range_a, hedge_range_b) >  HEDGE_TOLERANCE

ENTER and CLEAR are DELIBERATELY ASYMMETRIC (see `_recompute_stale`). The
series-chop term only guards against false-positive entries on quiet markets;
it is not evidence the hedge recovered, so it cannot clear the gate. Before
2026-07-28 the clear was the plain negation of the enter condition, which let
a merely-quiet series release a suppression while the hedge was still frozen —
the BROHLE 08:51:09 incident (15,000 lots fired into a dead hedge).

While flagged: arber fires and quoter quotes are suppressed. Clears only once
a hedge actually moves.

Design rationale (validated 2026-06-01):
  Real Poly staleness affects both teams' mids (same broken feed); whichever
  side ends up bit-exact frozen trips the detector. Healthy events typically
  have at least small jitter on both sides within the window, so the
  "EITHER strict-frozen" condition rarely false-positives.

Validation results (per the detector parameters below):
  - FAZE9Z 12:39-13:28 freeze: 90.7% coverage of the 50-min window
  - TNCPHA 09:01:46-09:04:36 chop: 100% coverage, +0s first-trigger latency
  - LOSRED / GXVIT / SRTL / NRGLOUD / GENGNS (healthy events): ≤1.1% per event
"""
import logging
import os
import time as _time
from collections import deque
from typing import Dict, Deque

log = logging.getLogger(__name__)

# Tunables. Validated against the 2026-05-28 TNCPHA and 2026-05-29 FAZE9Z
# incidents on 2026-06-01 — don't change without re-running the test harness.
WINDOW_SEC: float = 90.0
MIN_SAMPLES: int = 10
SERIES_RANGE_THRESHOLD: float = 3.0   # cents (Kalshi series ask range across window)
HEDGE_TOLERANCE: float = 0.0          # cents (strict — any movement clears the side)

# ── HARD-FREEZE duration-only override: TRIED AND REJECTED (2026-07-28) ─────
# `None` disables it. Do NOT re-enable without re-running the measurement below.
#
# The idea was: a hedge bit-exact frozen for N seconds suppresses on its own,
# no series movement required. It is unusable, because A FROZEN HEDGE IS THE
# NORMAL PREGAME STATE. Measured over 2026-07-28 03:00-09:10 (89,301 LEAD-LAG
# samples), natural hedge-freeze episodes with Poly perfectly healthy:
#
#     phase     episodes   median      p90      p99      max
#     PREGAME         70      90s    2011s    5779s    5779s
#     LIVE          1259       0s      16s     122s    1642s
#
# A pregame hedge routinely sits still for 33 minutes at p90. Share of healthy
# time a duration-only rule would suppress:
#
#     threshold    LIVE/healthy   PREGAME/healthy
#          300s          16.1%             88.3%
#          600s          16.1%             80.6%
#         1200s           0.0%             62.0%
#         3600s           0.0%             33.5%
#
# i.e. it would have shut off ~88% of normal pregame trading. No threshold
# rescues it: the signal it keys on is indistinguishable from healthy pregame.
#
# The danger was never "hedge frozen" — it is "hedge frozen WHILE THE SERIES
# MOVES". A frozen hedge against a frozen series generates no new edge, which
# is why NEMIGAPRAN / DKCKTC / BRODNF sat through the whole outage with a dead
# hedge and produced ZERO fires. Leaving them untraded is correct, not a miss.
# See FREEZE_SERIES_DANGER_C below for the rule that actually works.
HARD_FREEZE_SEC = None

# ── LIVE-GAME fast path (added 2026-07-28) ─────────────────────────────────
# Measured over the FREEZE EPISODE (since the hedge last actually changed),
# not the 90s rolling window. That longer lookback is what separates a live
# game from a pregame book while the hedge is dead:
#
#     event        state      freeze-episode range   max 90s window
#     NEMIGAPRAN   pregame            1c                   1c
#     DKCKTC       pregame            0c                   0c
#     BRODNF       pregame            0c                   0c
#     BROHLE       LIVE              28c                   3c
#     KTDRX        LIVE              32c                   4c
#
# (2026-07-28 Poly outage.) The rolling window cannot tell these apart — 3-4c
# vs 0-1c — which is exactly why BROHLE's gate kept clearing on a dead hedge.
# Cumulative movement since the freeze began separates them by ~28x.
#
# A live market that has moved this far against a frozen hedge is the maximum-
# danger state: every cent of that move is fake edge. Suppress at once rather
# than waiting out HARD_FREEZE_SEC (BROHLE moved 7-15c in the first 5 minutes
# of the outage — 300s of exposure would have been far too slow).
# Pregame books never reach this, so they fall through to HARD_FREEZE_SEC.
# Threshold sweep over the same 2026-07-28 dataset, as SHARE OF TIME suppressed
# (freeze sustained >= FREEZE_MIN_SEC and series moved >= this many cents):
#
#     move_c   LIVE/healthy   PREGAME/healthy   LIVE/outage
#        2c           55.4%            17.4%         ~93%
#        3c            1.7%             5.3%         93.3%
#        4c            0.9%             2.2%         93.3%
#        5c            0.9%             1.1%         90.0%
#        8c            0.0%             0.0%         90.0%
#
# 3c chosen to match `min_edge` on the arber rows (LOL_ARB_L1 min_edge=3.0):
# suppress exactly when the fake edge could be large enough to fire. Costs
# ~5% of healthy pregame time and ~2% of healthy live time, versus catching
# 93% of the outage. Raise to 4-5c if that pregame cost is too high.
#
# RESIDUAL RISK: `min_absolute_edge` is 1.5c on those rows, so a fake edge
# between 1.5c and 3c can still fire un-suppressed. Dropping to 2c to cover it
# costs 17% of pregame and 55% of live — not worth it. Accepted, not solved.
FREEZE_SERIES_DANGER_C: float = 3.0   # cents of series movement during the freeze
FREEZE_MIN_SEC: float = 60.0
# NOTE on FREEZE_MIN_SEC: in practice it never binds, and that is fine. Any
# freeze shorter than WINDOW_SEC is fully visible to the rolling window, so a
# brief plateau with >=5c of movement already trips the ORIGINAL ENTER
# condition (frozen hedge + chop) before this path is reached. The fast path
# only adds coverage when the movement has aged OUT of the 90s window — which
# by definition means the freeze is longer than the window. Kept as a cheap
# defensive floor so the constant can be raised if that ever changes.

# Cadence for "still stale" reminder log lines (seconds). Set to 0 to disable.
STILL_STALE_REMINDER_SEC: float = 60.0

# event_base → state dict:
#   "teams":              {team_suffix: deque[(ts, series_ask_cents, hedge_cents)]}
#   "stale":              bool — current flagged state
#   "last_transition_ts": float — wall-clock when state last toggled (for logging only)
#   "last_reminder_ts":   float — wall-clock when we last emitted the "still stale" reminder
_event_state: Dict[str, Dict] = {}


def _prune_old(snaps: Deque, now_ts: float) -> None:
    while snaps and (now_ts - snaps[0][0]) > WINDOW_SEC:
        snaps.popleft()


def _hard_freeze_secs(event: Dict, now_ts: float) -> Dict[str, float]:
    """Per-team seconds since the hedge last actually CHANGED value.

    Tracked outside the rolling deque so it can measure freezes longer than
    WINDOW_SEC — the whole point is catching multi-minute feed outages.
    """
    out = {}
    for team, tr in (event.get("hedge_track") or {}).items():
        out[team] = now_ts - tr["since"]
    return out


def _freeze_series_ranges(event: Dict) -> Dict[str, int]:
    """Per-team series-ask range accumulated SINCE the hedge last changed.

    This is the "longer lookback" that distinguishes a live game from a
    pregame book: over a multi-minute freeze a live series racks up tens of
    cents while a pregame one stays at 0-1c. The 90s rolling window flattens
    that distinction and cannot be used for it.
    """
    out = {}
    for team, tr in (event.get("hedge_track") or {}).items():
        if tr.get("smin") is None:
            continue
        out[team] = tr["smax"] - tr["smin"]
    return out


def _recompute_stale(event: Dict, currently_stale: bool = False,
                     now_ts: float = None) -> bool:
    """Asymmetric hysteresis — ENTER and CLEAR do NOT test the same thing.

    ENTER  : hedge frozen AND series chopping
    STAY   : hedge still frozen (series chop is IRRELEVANT once suppressed)
    CLEAR  : only when a hedge has actually MOVED

    Why asymmetric (BROHLE incident, 2026-07-28 08:51:09):
      The old code returned the ENTER conjunction and used its negation to
      clear, so EITHER conjunct falling released the gate. BROHLE's Poly hedge
      was frozen for 631s/2135s; the moment the Kalshi SERIES book went quiet
      (series_ranges {BRO:1, HLE:2}, under the 3.0 threshold) the gate cleared
      — while `hedge_ranges={'BRO':0.0,'HLE':0.0}`, i.e. the hedge was still
      completely dead. The arber unsuppressed, fired 15,000 lots against that
      frozen hedge, and POLY-STALE ENTER re-fired in the SAME second. A ~1s
      hole, entered and exited on a condition unrelated to the actual danger.

      The series-chop term exists only to keep ENTER from false-positiving on
      quiet markets. It is not evidence the hedge recovered, so it must not be
      able to clear. Detection sensitivity is unchanged — this only makes exit
      stricter, so the 2026-06-01 validation numbers above still hold.
    """
    # HARD-FREEZE OVERRIDE — checked FIRST, before the sample-count guards, so
    # a feed that has gone quiet (few samples) can still trip it. One team
    # bit-exact frozen for HARD_FREEZE_SEC is sufficient on its own: no series
    # chop required, no second team required.
    if now_ts is not None:
        _frozen = _hard_freeze_secs(event, now_ts)
        _fsr = _freeze_series_ranges(event)
        for team, secs in _frozen.items():
            # LIVE-GAME fast path: the series has moved materially while THIS
            # team's hedge sat dead. Every cent of that move is fake edge.
            if (secs >= FREEZE_MIN_SEC
                    and _fsr.get(team, 0) >= FREEZE_SERIES_DANGER_C):
                return True
            # Duration-only fallback — DISABLED (HARD_FREEZE_SEC is None).
            # See the constant's comment: a frozen hedge is the normal pregame
            # state, so any duration-only rule shuts off pregame wholesale.
            if HARD_FREEZE_SEC is not None and secs >= HARD_FREEZE_SEC:
                return True

    teams = event["teams"]
    # Require BOTH teams to have at least MIN_SAMPLES in window. Single-sided
    # data isn't enough to make a confident either-side-frozen call.
    #
    # Insufficient data must never CLEAR an active suppression: a feed that
    # stops producing snapshots entirely is the *most* broken state, and the
    # old `return False` treated it as "recovered". Holding suppression only
    # costs us trades on an event we cannot price.
    if len(teams) < 2:
        return currently_stale
    hedge_ranges = []
    series_ranges = []
    for snaps in teams.values():
        if len(snaps) < MIN_SAMPLES:
            return currently_stale
        hedges = [s[2] for s in snaps]
        bests = [s[1] for s in snaps]
        hedge_ranges.append(max(hedges) - min(hedges))
        series_ranges.append(max(bests) - min(bests))

    hedge_frozen = min(hedge_ranges) <= HEDGE_TOLERANCE
    if currently_stale:
        # Already suppressed: hold until a hedge genuinely moves.
        return hedge_frozen
    return hedge_frozen and max(series_ranges) >= SERIES_RANGE_THRESHOLD


def update_snapshot(event_base: str, team: str, ts: float,
                    series_ask_cents: int, hedge_cents: float) -> None:
    """Record one snapshot and recompute the event's stale state.

    Producers (arber_bot, map_arber_bot) call this once per cycle per direction.
    Idempotent against duplicate-ts updates from sibling bots on the same event.
    """
    ev = _event_state.get(event_base)
    if ev is None:
        ev = {"teams": {}, "stale": False, "last_transition_ts": 0.0,
              "last_reminder_ts": 0.0, "hedge_track": {}}
        _event_state[event_base] = ev

    snaps = ev["teams"].get(team)
    if snaps is None:
        snaps = deque()
        ev["teams"][team] = snaps

    # Track when this team's hedge last actually CHANGED, for the hard-freeze
    # override. Kept outside the rolling deque so it can measure freezes far
    # longer than WINDOW_SEC.
    _track = ev.setdefault("hedge_track", {})
    _sa = int(series_ask_cents)
    _tr = _track.get(team)
    if _tr is None:
        _track[team] = {"val": float(hedge_cents), "since": ts,
                        "smin": _sa, "smax": _sa, "moved": False}
    elif float(hedge_cents) != _tr["val"]:
        # Hedge moved -> the freeze episode ends; restart the series accumulator.
        _tr["val"] = float(hedge_cents)
        _tr["since"] = ts
        _tr["smin"] = _tr["smax"] = _sa
        _tr["moved"] = True   # liveness proof for hedge_proven_live()
    else:
        _tr["smin"] = min(_tr["smin"], _sa)
        _tr["smax"] = max(_tr["smax"], _sa)

    snaps.append((ts, int(series_ask_cents), float(hedge_cents)))
    _prune_old(snaps, ts)
    # Also prune sibling teams' deques against the latest ts so a quiet team's
    # old entries don't pollute the window.
    for other_team, other_snaps in ev["teams"].items():
        if other_team != team:
            _prune_old(other_snaps, ts)

    new_stale = _recompute_stale(ev, ev["stale"], ts)
    if new_stale != ev["stale"]:
        # Compute summary stats for the log line — caller wants visibility.
        hr_per_team = {t: (max(s[2] for s in snaps) - min(s[2] for s in snaps))
                       for t, snaps in ev["teams"].items() if snaps}
        sr_per_team = {t: (max(s[1] for s in snaps) - min(s[1] for s in snaps))
                       for t, snaps in ev["teams"].items() if snaps}
        samples_per_team = {t: len(s) for t, s in ev["teams"].items()}
        if new_stale:
            msg = (
                f"[POLY-STALE ENTER] {event_base} — hedge frozen on at least one side. "
                f"hedge_ranges={hr_per_team} series_ranges={sr_per_team} "
                f"frozen_for={ {k: round(v) for k, v in _hard_freeze_secs(ev, ts).items()} }s "
                f"series_move_during_freeze={_freeze_series_ranges(ev)}c "
                f"samples={samples_per_team}. Suppressing arber + quoter."
            )
            log.warning(msg)
            print(msg, flush=True)
            ev["last_reminder_ts"] = ts
        else:
            dur = ts - ev["last_transition_ts"] if ev["last_transition_ts"] else 0
            msg = (f"[POLY-STALE CLEAR] {event_base} — resumed after {dur:.0f}s. "
                   f"hedge_ranges={hr_per_team} series_ranges={sr_per_team}.")
            log.warning(msg)
            print(msg, flush=True)
        ev["stale"] = new_stale
        ev["last_transition_ts"] = ts
    elif new_stale and STILL_STALE_REMINDER_SEC > 0:
        # Periodic "still stale" reminder so the operator sees ongoing suppression.
        last_r = ev["last_reminder_ts"]
        if (ts - last_r) >= STILL_STALE_REMINDER_SEC:
            ongoing = ts - ev["last_transition_ts"]
            hr_per_team = {t: (max(s[2] for s in snaps) - min(s[2] for s in snaps))
                           for t, snaps in ev["teams"].items() if snaps}
            sr_per_team = {t: (max(s[1] for s in snaps) - min(s[1] for s in snaps))
                           for t, snaps in ev["teams"].items() if snaps}
            msg = (f"[POLY-STALE ONGOING] {event_base} — still suppressed ({ongoing:.0f}s). "
                   f"hedge_ranges={hr_per_team} series_ranges={sr_per_team} "
                   f"frozen_for={ {k: round(v) for k, v in _hard_freeze_secs(ev, ts).items()} }s")
            log.warning(msg)
            print(msg, flush=True)
            ev["last_reminder_ts"] = ts


def is_stale(event_base: str) -> bool:
    """Fast read for consumers. Returns False if event has never been seen."""
    ev = _event_state.get(event_base)
    if ev is None:
        return False
    return ev["stale"]


_proven_logged: set = set()


def hedge_proven_live(event_base: str) -> bool:
    """True once BOTH teams' Poly hedges have TICKED in this process's life.

    Boot-hole guard (26AUG13): state here is in-memory, so a restart during a
    Poly freeze wipes the suppression and the gate needs samples to re-enter.
    In that window the taker prices Kalshi's live book against a hedge that
    may have been dead for an hour — if Kalshi moved during the freeze, that
    is a large FAKE edge fired into an unhedgeable venue (BROHLE-shaped, at
    boot). A hedge that has never moved since boot is indistinguishable from
    a frozen one, so the taker must treat it as frozen. Healthy events prove
    liveness within seconds; a quiet book generates no real edge to miss.
    """
    ev = _event_state.get(event_base)
    if ev is None:
        return False
    track = ev.get("hedge_track") or {}
    if len(track) < 2:
        return False
    ok = all(tr.get("moved") for tr in track.values())
    if ok and event_base not in _proven_logged:
        _proven_logged.add(event_base)
        log.info("[HEDGE-PROVEN] %s — both hedge legs ticked since boot; "
                 "taker enabled", event_base)
    return ok


def get_stale_events() -> set:
    """All currently-flagged event_bases. For diagnostics / dashboards."""
    return {eb for eb, ev in _event_state.items() if ev["stale"]}


# ── Kalshi map-book fallback mode (26AUG13, POLY_STALE_KALSHI_FALLBACK_SCOPE) ──
# A third answer between "quote" and "suppress": when Poly is frozen but
# Kalshi's OWN map books are fresh/two-sided/tight, the event can keep making
# markets off Kalshi-derived inputs instead of going fully dark (NSDNF 26AUG12:
# 25 min suppressed spanning a whole intermission on the night's biggest
# market, 0.9% share vs 20.1%; repeated 26AUG13 on both LCK CL matches).
#
# PHASE 0 (default): SHADOW — mode() computes and LOGS the would-be fallback
#   but returns "SUPPRESSED", so every consumer behaves exactly as today.
# PHASE 1: `enable_stale_kalshi_fallback.flag` present → mode() returns
#   "FALLBACK" and consumers may quote makers-only with reduced clips.
#   `rm` the flag to revert within one tick. `disable_stale_kalshi_fallback.flag`
#   force-suppresses regardless (panic switch, checked first).
#
# The provider callback is installed by the manager (it owns the Kalshi REST
# registry): fn(event_base) -> {map_ticker: (bid_c, ask_c)} for the event's
# map tickers, ALREADY health-checked (fresh, two-sided, spread, depth) — or
# None/{} when Kalshi cannot carry the event. This module stays book-agnostic.
FB_ARM_FLAG = "enable_stale_kalshi_fallback.flag"
FB_KILL_FLAG = "disable_stale_kalshi_fallback.flag"
_FB_FLAG_CACHE_S = 1.0

_fb_provider = None
_fb_last_mode: Dict[str, str] = {}          # event_base -> last logged mode
_fb_flag_cache = {"ts": 0.0, "armed": False}


def set_fallback_provider(fn) -> None:
    """Install the Kalshi map-book provider (manager, at boot)."""
    global _fb_provider
    _fb_provider = fn
    log.info("[POLY-STALE FALLBACK] provider installed (%s)",
             getattr(fn, "__name__", "closure"))


def _fb_armed() -> bool:
    now = _time.time()
    if now - _fb_flag_cache["ts"] > _FB_FLAG_CACHE_S:
        _fb_flag_cache["ts"] = now
        _fb_flag_cache["armed"] = (os.path.exists(FB_ARM_FLAG)
                                   and not os.path.exists(FB_KILL_FLAG))
    return _fb_flag_cache["armed"]


# ── FULL-WS FALLBACK, PHASE 0: shadow evaluator (26AUG19) ──────────────────
# Parallel evaluation of the dedicated map WS books against the SAME bars the
# REST provider applies, logged side-by-side with the REST verdict. STRICTLY
# observation-only: called from mode() behind a bare try/except, never touches
# _fb_cert, never changes what mode() returns. Purpose: measure, over real
# stale episodes, whether the WS books would certify earlier/more often than
# the REST poll path before rewiring the operative provider onto them.
# Provider contract: fn(event_base) -> {map_ticker: (bid_c, ask_c, age_s,
# bid_depth_lots, ask_depth_lots)} for every live map book it has, UNFILTERED
# (bars are applied here so both paths are judged identically), or None.
# AGE MEANS DIFFERENT THINGS ON THE TWO PATHS — do not "align" these (26AUG19).
#   REST age = time since we last POLLED. Old => the data may be outdated, so
#     the REST provider's 10s bar is a real freshness test.
#   WS  age = time since a DELTA ARRIVED. Old => the book has not CHANGED. A
#     delta-maintained book is still accurate after a quiet spell; that is the
#     whole point of maintaining it. Quiet pregame esports books go minutes
#     between prints (ws_delta_book.py:176 says as much).
# Borrowing the 10s REST bar made the WS path spuriously stricter: WBLNG
# 26AUG19 06:33/06:34 logged ws=FAIL/rest=PASS twice where the ONLY failures
# were age=19s/44s on books quoting a 1c spread (31/32, 68/69).
# Use production's own WS SERVE guard instead: ws_delta_book.STALE_BOOK_S, the
# bar get_orderbook_fp enforces before handing a WS book to the map pricing
# path. If production will price off a book that age, so should certification.
# Connection death is caught separately (conn-stall watchdog + [MAP WS] events),
# which is the failure this age check must NOT be relied on to detect.
FB_WS_AGE_BAR_S = 60.0
FB_WS_SPREAD_BAR_C = 6
FB_WS_DEPTH_BAR = 25
_FB_WS_ROLLUP_S = 60.0

_fb_ws_provider = None
_fb_ws_last: Dict[str, str] = {}      # event_base -> last logged ws tag
_fb_ws_stats = {"evals": 0, "both": 0, "ws_only": 0, "rest_only": 0,
                "neither": 0, "ws_err": 0, "last_log": 0.0}


def set_ws_shadow_provider(fn) -> None:
    """Install the map-WS shadow provider (manager, at boot). Optional."""
    global _fb_ws_provider
    _fb_ws_provider = fn
    log.info("[FB-WS-SHADOW] provider installed (%s)",
             getattr(fn, "__name__", "closure"))


def _fb_ws_shadow(event_base: str, rest_certified: bool) -> None:
    """Shadow-compare WS map books vs the REST verdict. Never raises to the
    caller (mode() wraps this in try/except as well, belt-and-braces)."""
    if _fb_ws_provider is None:
        return
    now = _time.time()
    try:
        raw = _fb_ws_provider(event_base)
    except Exception as e:
        _fb_ws_stats["ws_err"] += 1
        log.warning("[FB-WS-SHADOW] provider error for %s: %r", event_base, e)
        return
    passing, failing = {}, {}
    for t, tup in (raw or {}).items():
        try:
            bid, ask, age, bd, ad = tup
        except (TypeError, ValueError):
            failing[t] = ("malformed", tup)
            continue
        why = []
        if bid <= 0 or ask <= 0 or ask >= 100:
            why.append("one_sided")
        if ask - bid > FB_WS_SPREAD_BAR_C:
            why.append(f"spread={ask - bid}")
        if age > FB_WS_AGE_BAR_S:
            why.append(f"age={age:.0f}s")
        if min(bd, ad) < FB_WS_DEPTH_BAR:
            why.append(f"depth={min(bd, ad)}")
        if why:
            failing[t] = (bid, ask, ",".join(why))
        else:
            passing[t] = (bid, ask, age)
    # ANY-pass semantics, matching the REST provider exactly: it FILTERS
    # failing legs and certifies on `out or None`. A settled map's dead book
    # (one-sided, no deltas — normal for BO3 game 1 during game 2) must not
    # veto the event, or every post-map-1 episode reports ws=FAIL spuriously
    # (seen live on T1AKTC 26AUG19 05:47, first real shadow event).
    ws_certified = bool(passing)
    _fb_ws_stats["evals"] += 1
    key = ("both" if ws_certified and rest_certified else
           "ws_only" if ws_certified else
           "rest_only" if rest_certified else "neither")
    _fb_ws_stats[key] += 1
    tag = f"ws={'PASS' if ws_certified else 'FAIL'}/rest={'PASS' if rest_certified else 'FAIL'}"
    if _fb_ws_last.get(event_base) != tag:
        _fb_ws_last[event_base] = tag
        msg = (f"[FB-WS-SHADOW] {event_base} {tag} | "
               f"pass={ {t: b for t, b in sorted(passing.items())} } "
               f"fail={ {t: b for t, b in sorted(failing.items())} }"
               if (passing or failing) else
               f"[FB-WS-SHADOW] {event_base} {tag} | no ws map books")
        log.warning(msg)
        print(msg, flush=True)
    if now - _fb_ws_stats["last_log"] >= _FB_WS_ROLLUP_S:
        _fb_ws_stats["last_log"] = now
        s = _fb_ws_stats
        msg = (f"[FB-WS-SHADOW ROLLUP] evals={s['evals']} both={s['both']} "
               f"ws_only={s['ws_only']} rest_only={s['rest_only']} "
               f"neither={s['neither']} ws_err={s['ws_err']}")
        log.warning(msg)
        print(msg, flush=True)


# ── Certification hysteresis (26AUG13 KEYDYAW flap) ────────────────────────
# A borderline book sitting exactly at the spread bar flips raw certification
# tick-to-tick (KEYDYAW: 2 certs vs 10 declines in one evening; one observed
# SHADOW -> UNAVAILABLE flap inside 2.5s). Armed, every flip would pull and
# repost the maker quotes. So: ENTRY needs >=FB_ENTRY_CONFIRM_N passing
# evaluations spanning >=FB_ENTRY_MIN_SPAN_S (a single borderline tick cannot
# flip us in); EXIT holds certification through provider flickers for
# FB_STICKY_S, SERVING THE LAST CERTIFIED BOOKS — critical, because a
# fallback_books() that goes empty mid-FALLBACK would silently skip the
# 10c dev-guard in bot.py. A decline persisting past the sticky window
# de-certifies and requires fresh entry confirmation.
FB_ENTRY_CONFIRM_N = 2
FB_ENTRY_MIN_SPAN_S = 2.0
FB_STICKY_S = 20.0
_fb_cert: Dict[str, dict] = {}


def _fb_certified(event_base: str):
    """(certified, books) with entry/exit hysteresis around the provider."""
    now = _time.time()
    books = None
    if _fb_provider is not None:
        try:
            books = _fb_provider(event_base)
        except Exception as e:  # provider must never break the gate
            log.warning("[POLY-STALE FALLBACK] provider error for %s: %r",
                        event_base, e)
            books = None
    st = _fb_cert.setdefault(event_base, {"passes": 0, "first_pass_ts": 0.0,
                                          "certified": False, "books": None,
                                          "ts": 0.0})
    if books:
        if st["passes"] == 0:
            st["first_pass_ts"] = now
        st["passes"] += 1
        st["books"] = books
        st["ts"] = now
        if (not st["certified"] and st["passes"] >= FB_ENTRY_CONFIRM_N
                and now - st["first_pass_ts"] >= FB_ENTRY_MIN_SPAN_S):
            st["certified"] = True
    elif not (st["certified"] and now - st["ts"] <= FB_STICKY_S):
        # Not a held flicker: reset. (Certified + within sticky -> hold, keep
        # serving the cached books.)
        st["passes"] = 0
        st["certified"] = False
        st["books"] = None
    return st["certified"], st["books"]


def mode(event_base: str) -> str:
    """'OK' | 'FALLBACK' | 'SUPPRESSED' for consumers.

    Not stale -> OK (fast path, no provider call). Stale -> FALLBACK only when
    the provider certifies Kalshi map books can carry the event AND the arm
    flag is present; otherwise SUPPRESSED (exactly today's behavior). Shadow
    (unarmed) logs the would-be decision so Phase 1 can be armed on evidence.
    """
    if not is_stale(event_base):
        if _fb_last_mode.pop(event_base, None):
            pass  # stale cleared; next episode logs fresh
        _fb_cert.pop(event_base, None)
        _fb_ws_last.pop(event_base, None)
        return "OK"
    can_fb, books = _fb_certified(event_base)
    try:
        _fb_ws_shadow(event_base, can_fb)   # observation-only, never raises
    except Exception:
        pass
    armed = _fb_armed()
    m = "FALLBACK" if (can_fb and armed) else "SUPPRESSED"
    # Log once per state change per event, both armed and shadow.
    tag = m if armed or not can_fb else "FALLBACK-SHADOW"
    if _fb_last_mode.get(event_base) != tag:
        _fb_last_mode[event_base] = tag
        if can_fb:
            summary = {t: b for t, b in sorted(books.items())}
            msg = (f"[POLY-STALE {'FALLBACK' if armed else 'FALLBACK-SHADOW'}] "
                   f"{event_base} — kalshi map books can carry the event: "
                   f"{summary}"
                   + ("" if armed else " (unarmed: suppressing as today; "
                      f"touch {FB_ARM_FLAG} to arm)"))
        else:
            msg = (f"[POLY-STALE FALLBACK-UNAVAILABLE] {event_base} — stale and "
                   f"kalshi map books cannot carry it (provider="
                   f"{'missing' if _fb_provider is None else 'declined'}) — "
                   f"SUPPRESSED")
        log.warning(msg)
        print(msg, flush=True)
    return m


def fallback_books(event_base: str):
    """Health-checked Kalshi map TOBs for a FALLBACK event, else None.

    Served from the hysteresis state written by mode() (the consumer always
    calls mode() first in the same cycle), so mode and books can never
    disagree — including during a held flicker, where the last certified
    books keep feeding the dev-guard.
    """
    if not is_stale(event_base):
        return None
    st = _fb_cert.get(event_base)
    if st and st["certified"] and _time.time() - st["ts"] <= FB_STICKY_S:
        return st["books"]
    return None


def fallback_poll_tickers(bot_tickers) -> set:
    """Kalshi MAP tickers the REST poller should ADD for currently-stale events.

    Root cause of the 26AUG13 all-day 'provider=declined': the certification
    provider scans the REST orderbook registry for KX*MAP books, but the
    poller only polls the manager's bot (SERIES) tickers — the registry never
    contains a map book, so the fallback could never certify. Fix: while an
    event is stale, ask the poller to also poll its map-1/map-2 legs (both
    teams). Transient by construction — an event that clears staleness drops
    its map tickers on the next 30s provider refresh, so the steady-state
    poller load is unchanged.

    Names are CONSTRUCTED (GAME->MAP swap), which the map-ticker memory warns
    about (date skew): a skewed name simply 404s, the poller drops it after 3
    tries, and certification fails closed — same outcome as today, never a
    wrong book. Decider (map 3) is never listed on Kalshi and is not built.
    """
    out = set()
    stale = [eb for eb, ev in _event_state.items() if ev.get("stale")]
    if not stale:
        return out
    bt = list(bot_tickers)
    for eb in stale:
        try:
            prefix, code = eb.split("-", 1)
        except ValueError:
            continue
        mp = prefix.replace("GAME", "MAP")
        if mp == prefix:
            continue                      # no map universe for this sport
        suffixes = {t.rsplit("-", 1)[1] for t in bt if t.startswith(eb + "-")}
        for n in (1, 2):
            for s in suffixes:
                out.add(f"{mp}-{code}-{n}-{s}")
    return out


def reset() -> None:
    """Clear all state. For tests."""
    _event_state.clear()


def _debug_state(event_base: str) -> Dict:
    """Inspector for tests / diagnostics. Returns shallow copy of event state."""
    ev = _event_state.get(event_base)
    if ev is None:
        return {}
    return {
        "stale": ev["stale"],
        "last_transition_ts": ev["last_transition_ts"],
        "team_samples": {t: len(s) for t, s in ev["teams"].items()},
    }
