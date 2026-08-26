"""Kalshi's own live game state — round scores for CS2 / VAL toxicity gating.

Replaces bo3.gg as the score source for the maker-suppress gate. Read-only.

WHY: 26AUG20 G2 vs M80 (VCT Americas). bo3 listed the fixture `current` while
publishing `live_updates: None`, so the gate failed OPEN and the quoter ran
through an 11-11 map that finished 14-12 in overtime. Kalshi had the round
score the whole time, 2-12s fresh.

WHY IT IS BETTER: milestones are discovered BY KALSHI EVENT TICKER, which
deletes the entire bo3 name-binding layer (`_dtokens`/`_match_to_kalshi`,
`bo3_team_aliases.csv`, `bo3_parsed_markets.csv`) that
project_bo3_maker_suppress_gate calls "the recurring failure mode — it fails
SILENTLY".

SCOPE: CS2 and VAL only. Dota uses its own feed, LoL uses the Riot API, and
`bo3_score_feed.event_sport()` returns None for both — they were never gated.

    1. GET /trade-api/v2/milestones?limit=200&related_event_ticker=<eb>
         -> milestones[0].id        (`limit` is MANDATORY; omitting it 400s)
    2. GET /trade-api/v2/live_data/batch?milestone_ids=A&milestone_ids=B
         ^^ REPEATED params. Comma-separated silently returns
            {"live_datas": null} — no error, just nothing.

Payload (type=esports_match):
    home_periods {period_1: 14, period_2: 3}   <- ROUND score per map
    away_periods {period_1: 12, period_2: 0}
    home_score / away_score                     <- MAP score. UNRELIABLE at
        close (MOUZGL 26AUG19 read maps 1-1 with three won periods). Never
        used here; the gate reads rounds only.
    home_stats/away_stats [{period, stats:{winner, map_forfeit, ...}}]
    is_live, status, last_updated_ts
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

import requests

log = logging.getLogger(__name__)

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
TIMEOUT = 8.0

# Data older than this on a LIVE match is not trustworthy for gating.
MAX_DATA_AGE_S = 180.0
# Milestone ids are stable per event; cache so the only per-cycle cost is one
# batched live_data call.
_mid_cache: Dict[str, Optional[str]] = {}
# (event_base, period) -> True once that map has been seen toxic. STICKY for
# the rest of the map: a map that reached 10-10 is in the pickoff endgame even
# if it briefly widens to 11-7, and without this a band-boundary oscillation
# tears down and respawns quoter bots every populate cycle.
_sticky: Dict[Tuple[str, str], bool] = {}

KILL_FLAG = "disable_kalshi_game_state.flag"

_TERMINAL_STATUS = {"finished", "completed", "closed", "ended", "cancelled",
                    "canceled", "postponed", "forfeit"}


def disabled() -> bool:
    return os.path.exists(KILL_FLAG)


def game_over(det: dict) -> Optional[str]:
    """Reason if the MATCH is over, else None.

    Operator rule 26AUG20: a game-over state is NON-TOXIC — a concluded match
    must never suppress on its residual round score. UNKNOWN (missing/None)
    is NOT game-over; it falls through to the period logic, because reading
    absence as "finished" would silently un-suppress a live toxic map.
    """
    if det.get("is_live") is False:
        return "is_live=False"
    st = det.get("status")
    if isinstance(st, str) and st.strip().lower() in _TERMINAL_STATUS:
        return f"status={st}"
    return None


def _finished_periods(det: dict) -> set:
    """Periods with a winner stamped — maps that are OVER.

    Without this, a map that ENDED 14-12 scores toxic forever under VAL
    (10,3), suppressing through the whole between-maps break and into the next
    map's pistol round. Caught in shadow on G2/M80 before it shipped.
    """
    done = set()
    for side in ("home_stats", "away_stats"):
        rows = det.get(side)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            stats = row.get("stats")
            period = row.get("period")
            if period and isinstance(stats, dict) and stats.get("winner") is not None:
                done.add(period)
    return done


def active_map_score(det: dict):
    """(home_rounds, away_rounds, period_key) for the LIVE map, else Nones.

    Live map = highest period_N with NO winner declared.
    """
    hp = det.get("home_periods") or {}
    ap = det.get("away_periods") or {}
    if not isinstance(hp, dict) or not isinstance(ap, dict):
        return None, None, None
    keys = (set(hp) | set(ap)) - _finished_periods(det)
    if not keys:
        return None, None, None

    def _n(k):
        try:
            return int(str(k).rsplit("_", 1)[-1])
        except ValueError:
            return -1

    k = max(keys, key=_n)
    return hp.get(k), ap.get(k), k


def milestone_id(event_base: str) -> Optional[str]:
    """Cached milestone id for a Kalshi event base. None when untracked."""
    if event_base in _mid_cache:
        return _mid_cache[event_base]
    mid = None
    try:
        r = requests.get(f"{KALSHI}/milestones",
                         params={"limit": 200,
                                 "related_event_ticker": event_base},
                         timeout=TIMEOUT)
        if r.status_code == 200:
            ms = (r.json() or {}).get("milestones") or []
            if ms:
                mid = ms[0].get("id")
        else:
            log.warning("[KGS] milestones HTTP %s for %s", r.status_code, event_base)
            return None            # transient: do NOT cache a failure
    except Exception as e:
        log.warning("[KGS] milestones error for %s: %r", event_base, e)
        return None                # transient: do NOT cache a failure
    _mid_cache[event_base] = mid   # cache real answers only (incl. "no milestone")
    return mid


def _fetch_batch(mids) -> Dict[str, dict]:
    """milestone_id -> details. REPEATED milestone_ids params (see module doc)."""
    ids = [m for m in mids if m]
    if not ids:
        return {}
    out: Dict[str, dict] = {}
    # Chunk so one huge slate cannot build an over-long URL.
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        try:
            r = requests.get(f"{KALSHI}/live_data/batch",
                             params=[("milestone_ids", m) for m in chunk],
                             timeout=TIMEOUT)
            if r.status_code != 200:
                log.warning("[KGS] live_data HTTP %s", r.status_code)
                continue
            body = r.json() or {}
            if body.get("error"):
                log.warning("[KGS] live_data error: %s", body["error"])
                continue
            for ld in (body.get("live_datas") or []):
                mid = ld.get("milestone_id")
                if mid:
                    out[mid] = ld.get("details") or {}
        except Exception as e:
            log.warning("[KGS] live_data error: %r", e)
    return out


def evaluate(event_bases, thresholds_for, now: Optional[float] = None) -> Dict[str, dict]:
    """event_base -> verdict dict, for CS2/VAL events only.

    verdict = {toxic: bool, reason: str, period: str|None, score: (h,a)|None,
               age_s: float|None, sticky: bool}

    `thresholds_for` is injected (bo3_score_feed.thresholds_for) so the BANDS
    stay defined in exactly one place and this module cannot drift from them.

    FAILURE POLICY — deliberately split, because the two cases mean different
    things and conflating them is what burned us:
      * no milestone at all  -> untracked event, verdict OMITTED (fail open).
        Nothing else can be inferred and suppressing every untracked event
        would kill unrelated volume.
      * milestone EXISTS but state is missing/stale on a LIVE match -> FAIL
        CLOSED (toxic=True). This is the bo3 failure shape — the fixture looks
        healthy while scores are absent — and it is the one that cost money.
    """
    now = now or time.time()
    out: Dict[str, dict] = {}
    if disabled():
        return out

    ebs = [eb for eb in event_bases if eb]
    mids = {eb: milestone_id(eb) for eb in ebs}
    details = _fetch_batch(set(mids.values()))

    for eb in ebs:
        mid = mids.get(eb)
        if not mid:
            continue                                   # untracked -> fail open
        det = details.get(mid)
        if not det:
            out[eb] = {"toxic": True, "reason": "no live_data for known milestone",
                       "period": None, "score": None, "age_s": None,
                       "sticky": False}
            continue

        over = game_over(det)
        if over:
            for k in [k for k in _sticky if k[0] == eb]:
                _sticky.pop(k, None)                   # match done: clear state
            out[eb] = {"toxic": False, "reason": f"game over ({over})",
                       "period": None, "score": None, "age_s": None,
                       "sticky": False}
            continue

        lu = det.get("last_updated_ts")
        age = (now - lu) if isinstance(lu, (int, float)) else None
        if age is None or age > MAX_DATA_AGE_S:
            out[eb] = {"toxic": True,
                       "reason": f"stale state on live match (age={age})",
                       "period": None, "score": None, "age_s": age,
                       "sticky": False}
            continue

        h, a, pk = active_map_score(det)
        if h is None or a is None:
            out[eb] = {"toxic": False, "reason": "between maps / no live period",
                       "period": None, "score": None, "age_s": age,
                       "sticky": False}
            continue

        sport = "CS2" if "KXCS2" in eb else ("VAL" if "KXVALORANT" in eb else None)
        raw = any(max(h, a) >= lead and abs(h - a) <= diff
                  for lead, diff in thresholds_for(sport))
        key = (eb, pk)
        if raw:
            _sticky[key] = True
        # Drop stale sticky entries for maps that have since finished.
        for k in [k for k in _sticky if k[0] == eb and k[1] != pk]:
            _sticky.pop(k, None)
        sticky = _sticky.get(key, False)
        out[eb] = {"toxic": bool(raw or sticky),
                   "reason": ("toxic" if raw else
                              "sticky (map was toxic earlier)" if sticky
                              else "below bands"),
                   "period": pk, "score": (h, a), "age_s": age,
                   "sticky": bool(sticky and not raw)}
    return out
