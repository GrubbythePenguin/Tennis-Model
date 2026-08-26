"""Riot lolesports game-state detection — the authoritative "has this map started,
and exactly when" source for LoL.

WHY THIS EXISTS
---------------
`arber_bot._update_pregame_latch` has no clock. It flips pregame->in-game when the
series mid or the hedge cost moves >=`pregame_block_cents` from the first value it
ever saw, which means:
  * it fires on sharp PREGAME repricing (DRX/DNS 26AUG10: fired 09:00:00, real
    game-1 start was 09:07:33), and
  * it fires on stale-feed artifacts (BROAP 26AUG10: fired 18:10:15, 9.5 minutes
    BEFORE the real 18:19:42 start, driven by a Poly map book frozen 6,230s).
Either way the pregame probability anchor is never revised, so a market that
reprices between populate time and kickoff leaves a standing one-sided basis.

WHAT RIOT GIVES US (measured 26AUG10)
-------------------------------------
  getLeagues / getSchedule / getEventDetails -> 200 (the 26AUG05 403 has resolved)
  feed/livestats/v1/window/{gameId}:
      unstarted game -> HTTP 204, no frames
      started game   -> HTTP 200, frames[0].rfc460Timestamp == EXACT game start
  gameIds are listed by getEventDetails BEFORE a game starts, so the window
  endpoint can be polled as a start detector.

Two delays, do not confuse them:
  * event/game `state` field  -> ~26 min behind (BROAP: start 18:19:42, getLive
    flipped 18:45:26). USELESS as a trigger.
  * frame publication          -> ~5 min (299s measured). This is the detector.
The ANCHOR is delay-immune either way: frames[0] is the true start timestamp, so
the caller picks its cached pregame solve from strictly before that instant. A
slow feed lengthens the trading gate, it never corrupts the anchor.

Scheduled times are worthless for rolling brackets — DRX/DNS was scheduled
08:15Z and started 09:07:33 (52 min late) because it followed another match.

USAGE
-----
    import riot_game_state as rgs
    m = rgs.find_match("DN SOOPers", "Kiwoom DRX", "2026-08-10")
    st = rgs.game_start(m, 1)          # datetime | None  (None => not started)
    ph = rgs.match_phase(m)            # {'started':bool,'active_game':int,'starts':{...}}

Read-only, unauthenticated, cached. Never raises into the caller: every public
function returns None / a safe default and logs the reason.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger("riot_game_state")

API = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"
# The static key the lolesports web client ships. Public, not a secret.
KEY = {"x-api-key": "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"}

_R = os.path.dirname(os.path.abspath(__file__))
INDEX_CACHE = os.path.join(_R, "_riot_schedule_index.json")
INDEX_TTL_S = 30 * 60          # schedule changes slowly; 30 min is plenty
HTTP_TIMEOUT = 12

# in-process caches
_index: Optional[dict] = None
_index_ts: float = 0.0
_games_cache: Dict[str, Tuple[float, list]] = {}     # match_id -> (ts, [game dicts])
_start_cache: Dict[str, Tuple[float, Optional[datetime]]] = {}   # gameId -> (ts, start|None)
# gameIds never change once listed, but the per-game `state` on the same payload
# DOES, and pregame_anchor.map1_decided reads it as its primary "map 1 is over"
# signal. Keep this well under populate's ~150s cycle or that signal goes stale
# by up to half an hour and the intermission window is missed.
GAMES_TTL_S = 90
# A resolved start time never changes -> cache forever. A *negative* result
# (not started yet) must expire, or we would never notice the start — but there
# is no point polling faster than the ~5 min frame publication delay.
START_NEG_TTL_S = 60
# Concurrency for prefetch(). These are small independent GETs against a public
# CDN-fronted feed; serial resolution of a 30-event slate measured 18.9s, which
# would be added to EVERY populate cycle.
PREFETCH_WORKERS = 8


# ── availability / 403 handling ──────────────────────────────────────────────
# The discovery endpoints were 403 for everything between 26AUG05 and 26AUG10
# (static key, no key, spoofed UA/Origin — all refused) and then recovered on
# their own. That WILL happen again, and the failure mode must be: behave
# exactly as the system did before this module existed — keep the frozen
# populate anchor, change nothing, do not raise, and do not hammer the endpoint
# on every populate cycle.
#
# `unavailable()` is the single flag callers can read. It latches on the first
# auth-shaped rejection (401/403) and clears after DISABLED_RETRY_S so a
# recovery is picked up without a restart.
DISABLED_RETRY_S = 1800
_disabled_until: float = 0.0
_disabled_reason: str = ""


def unavailable() -> bool:
    """True while Riot is refusing us — callers must fall back to today's behavior."""
    return time.time() < _disabled_until


def disabled_reason() -> str:
    return _disabled_reason


def _disable(reason: str) -> None:
    global _disabled_until, _disabled_reason
    first = not unavailable()
    _disabled_until = time.time() + DISABLED_RETRY_S
    _disabled_reason = reason
    if first:
        log.error("[RIOT] DISABLED for %ds — %s. Pregame/intermission re-anchoring "
                  "is OFF; probabilities keep their populate-time values (pre-26AUG11 "
                  "behavior). Will retry automatically.", DISABLED_RETRY_S, reason)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _get(url: str, params: dict = None) -> Tuple[int, Optional[dict]]:
    if unavailable():
        return 0, None
    try:
        r = requests.get(url, params=params or {}, headers=KEY, timeout=HTTP_TIMEOUT)
    except Exception as e:                                    # network, DNS, TLS
        log.warning("[RIOT] %s -> %s: %s", url, type(e).__name__, e)
        return 0, None
    if r.status_code == 204:
        return 204, None                                      # game not started
    if r.status_code in (401, 403):
        _disable(f"HTTP {r.status_code} on {url.rsplit('/', 1)[-1]}")
        return r.status_code, None
    if r.status_code == 429:
        _disable("HTTP 429 (rate limited)")
        return 429, None
    if r.status_code != 200:
        log.warning("[RIOT] %s -> HTTP %s", url, r.status_code)
        return r.status_code, None
    try:
        return 200, r.json()
    except Exception:
        log.warning("[RIOT] %s -> 200 but unparseable body", url)
        return 200, None


# ── team-name normalisation ───────────────────────────────────────────────────
# Riot ships full org names ("KIWOOM DRX", "DN SOOPers"); our CSVs carry Poly's
# ("Kiwoom DRX", "DN SOOPers") and Kalshi's short codes. Strip accents, case,
# punctuation and the usual org suffixes so the two sides meet.
_SUFFIXES = ("esports", "esport", "sports", "gaming", "team", "club",
             "gg", "org", "the", "e")
# Tier markers. Stripped for matching (they are noise), but the resulting
# academy/main flag is then required to AGREE between the two sides — matching
# "Dplus KIA Challengers" to "Dplus KIA" would silently resolve to a different
# match. Measured on 185 events: the rule removes 2 such joins and costs 2
# legitimate ones, which riot_team_aliases.csv gets back explicitly.
_TIER_TOKENS = ("challengers", "challenger", "academy", "youth", "global",
                "cl", "jr", "junior")
_ACCENT = re.compile(r"[̀-ͯ]")

_aliases: Optional[Dict[str, str]] = None
_aliases_mtime: float = 0.0
ALIAS_FILE = os.path.join(_R, "riot_team_aliases.csv")


def aliases_mtime() -> float:
    """mtime of riot_team_aliases.csv, 0.0 if absent.

    Public because pregame_anchor compares it against its per-event lookup
    backoff stamp: an alias edit NEWER than the last failed lookup means the
    operator just fixed the join, so the backoff must not be honoured.
    """
    try:
        return os.path.getmtime(ALIAS_FILE)
    except OSError:
        return 0.0


def _load_aliases() -> Dict[str, str]:
    # mtime-watched, same pattern as populate's _maybe_reload_team_map: an
    # alias pasted mid-slate must take effect on the next pass, not the next
    # restart. Aliases are the operator's ONLY lever on this matcher.
    global _aliases, _aliases_mtime
    mt = aliases_mtime()
    if _aliases is not None and mt == _aliases_mtime:
        return _aliases
    out: Dict[str, str] = {}
    try:
        with open(ALIAS_FILE) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("our_name"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    out[_bare(parts[0])] = parts[1]
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("[RIOT] could not read %s: %s", ALIAS_FILE, e)
    _aliases, _aliases_mtime = out, mt
    return out


def _bare(name: str) -> str:
    s = unicodedata.normalize("NFD", str(name or ""))
    s = _ACCENT.sub("", s).lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())


def norm_team(name: str) -> str:
    """Lower/de-accented core tokens, org suffixes and tier markers removed."""
    toks = _bare(name).split()
    return " ".join(t for t in toks
                    if t not in _SUFFIXES and t not in _TIER_TOKENS)


def _parse_team(name: str) -> Tuple[set, bool]:
    """(match keys, is_academy) for one team name, alias-substituted first."""
    alias = _load_aliases().get(_bare(name))
    if alias:
        name = alias
    toks = _bare(name).split()
    is_academy = any(t in _TIER_TOKENS for t in toks)
    core = [t for t in toks if t not in _SUFFIXES and t not in _TIER_TOKENS]
    if not core:
        return set(), is_academy
    keys = {" ".join(core), "".join(core)}
    keys.update(t for t in core if len(t) >= 3)
    # FULL-NAME concatenation, built BEFORE suffix/tier stripping (26AUG11).
    # Riot writes some orgs without spaces, e.g. 'TeamOrangeGaming'. That is a
    # single token, so it survives suffix removal intact, while our spaced
    # 'Team Orange Gaming' loses 'team' and 'gaming' and reduces to {'orange'} —
    # no overlap, no match, and because find_match needs BOTH sides to resolve,
    # that one miss also sank its opponent VfB (which matched fine alone).
    #
    # Safe against the academy/main collapse this file exists to prevent: the
    # tier token is RETAINED here, so 'dpluskiachallengers' != 'dpluskia'. The
    # key only ever unifies names that are already identical modulo whitespace,
    # case and accents — it cannot bridge two genuinely different names.
    if len(toks) > 1:
        keys.add("".join(toks))
    return {k for k in keys if k}, is_academy


def _team_keys(name: str) -> set:
    return _parse_team(name)[0]


# ── schedule index ────────────────────────────────────────────────────────────
def load_index(force: bool = False) -> dict:
    """{league_name: {id, slug, events:[...]}} across every Riot league.

    Cached on disk (INDEX_CACHE) and in-process for INDEX_TTL_S.
    """
    global _index, _index_ts
    now = time.time()
    if _index is not None and not force and now - _index_ts < INDEX_TTL_S:
        return _index
    if not force and os.path.exists(INDEX_CACHE):
        try:
            age = now - os.path.getmtime(INDEX_CACHE)
            if age < INDEX_TTL_S:
                with open(INDEX_CACHE) as fh:
                    _index = json.load(fh)
                _index_ts = now
                return _index
        except Exception as e:
            log.warning("[RIOT] index cache unreadable (%s); refetching", e)

    sc, d = _get(f"{API}/getLeagues", {"hl": "en-US"})
    if sc != 200 or not d:
        if _index is not None:
            log.warning("[RIOT] getLeagues failed; serving stale in-process index")
            return _index
        log.error("[RIOT] getLeagues failed and no cached index — detection unavailable")
        return {}
    out = {}
    for lg in (d.get("data") or {}).get("leagues") or []:
        sc, s = _get(f"{API}/getSchedule", {"hl": "en-US", "leagueId": lg["id"]})
        if sc != 200 or not s:
            continue
        out[lg["name"]] = {
            "id": lg["id"],
            "slug": lg.get("slug", ""),
            "events": ((s.get("data") or {}).get("schedule") or {}).get("events", []),
        }
    if not out:
        log.error("[RIOT] schedule index came back empty — detection unavailable")
        return _index or {}
    _index, _index_ts = out, now
    try:
        with open(INDEX_CACHE, "w") as fh:
            json.dump(out, fh)
    except Exception as e:
        log.warning("[RIOT] could not persist index cache: %s", e)
    return out


def find_match(team_a: str, team_b: str, date_utc: str = None,
               league_hint: str = None) -> Optional[str]:
    """Riot match id for the event featuring both teams, or None.

    date_utc: 'YYYY-MM-DD'. Riot's startTime is the SCHEDULED time and can be an
    hour or more off on rolling brackets, so the date is a filter, not a key —
    neighbouring days are accepted.
    """
    idx = load_index()
    if not idx:
        return None
    ka, acad_a = _parse_team(team_a)
    kb, acad_b = _parse_team(team_b)
    if not ka or not kb:
        return None
    best = None
    for lname, lg in idx.items():
        if league_hint and norm_team(league_hint) not in norm_team(lname) \
           and norm_team(lname) not in norm_team(league_hint):
            continue
        for e in lg.get("events") or []:
            teams = (e.get("match") or {}).get("teams") or []
            if len(teams) != 2:
                continue
            k0, acad_0 = _parse_team(teams[0].get("name"))
            k1, acad_1 = _parse_team(teams[1].get("name"))
            fwd = bool((ka & k0) and (kb & k1))
            rev = bool((ka & k1) and (kb & k0))
            # An academy roster must never match its main roster.
            if fwd and (acad_a != acad_0 or acad_b != acad_1):
                fwd = False
            if rev and (acad_a != acad_1 or acad_b != acad_0):
                rev = False
            if not (fwd or rev):
                continue
            st = (e.get("startTime") or "")[:10]
            if date_utc:
                try:
                    d0 = datetime.strptime(date_utc, "%Y-%m-%d")
                    d1 = datetime.strptime(st, "%Y-%m-%d")
                    if abs((d1 - d0).days) > 1:
                        continue
                except Exception:
                    pass
            mid = (e.get("match") or {}).get("id")
            if mid:
                # exact-date beats neighbour-date
                score = 2 if st == date_utc else 1
                if best is None or score > best[0]:
                    best = (score, mid)
    return best[1] if best else None


def suggest_aliases(team_a: str, team_b: str,
                    date_utc: str = None) -> List[Tuple[str, str]]:
    """Paste-ready riot_team_aliases.csv rows for a FAILED find_match.

    Mirrors refresh_tomorrow_markets._suggest_bo3_aliases: when one of our
    teams resolves against a Riot fixture but its opponent does not, that
    opponent pair is almost always a missing alias — every one of the 8
    aliases added 26AUG11 (DV1, TOG, VfB, JDG, SU, Volda, LEO, Ole Miss) had
    exactly this shape and was diagnosed by hand. Returns
    [(our_name, riot_name)] so the caller can print the row verbatim.

    Evidence bar: the fixture must be on the requested date (±1, same rule as
    find_match), the MATCHED side's academy flag must agree, and the pair is
    only a suggestion — nothing here is adopted automatically. The unmatched
    pair's academy flags are deliberately NOT compared: tier-marker gaps
    ('NSEA' vs 'Nongshim Esports Academy') are precisely what aliases fix.
    """
    idx = load_index()
    if not idx:
        return []
    ka, acad_a = _parse_team(team_a)
    kb, acad_b = _parse_team(team_b)
    out: List[Tuple[str, str]] = []
    seen = set()
    for lg in idx.values():
        for e in lg.get("events") or []:
            teams = (e.get("match") or {}).get("teams") or []
            if len(teams) != 2:
                continue
            st = (e.get("startTime") or "")[:10]
            if date_utc:
                try:
                    d0 = datetime.strptime(date_utc, "%Y-%m-%d")
                    d1 = datetime.strptime(st, "%Y-%m-%d")
                    if abs((d1 - d0).days) > 1:
                        continue
                except Exception:
                    continue
            n0, n1 = teams[0].get("name"), teams[1].get("name")
            k0, acad_0 = _parse_team(n0)
            k1, acad_1 = _parse_team(n1)
            # (our matched name/flag, riot side keys/flag, our missing raw
            # name, riot missing raw name) — both orientations of both sides.
            for m_keys, m_acad, r_keys, r_acad, miss_ours, miss_ours_keys, miss_riot in (
                    (ka, acad_a, k0, acad_0, team_b, kb, n1),
                    (ka, acad_a, k1, acad_1, team_b, kb, n0),
                    (kb, acad_b, k0, acad_0, team_a, ka, n1),
                    (kb, acad_b, k1, acad_1, team_a, ka, n0)):
                if not (m_keys & r_keys) or m_acad != r_acad:
                    continue
                miss_riot_keys = _team_keys(miss_riot)
                if miss_ours_keys & miss_riot_keys:
                    continue        # opponent matched too — not a near-miss
                pair = (str(miss_ours or "").strip(), str(miss_riot or "").strip())
                if all(pair) and pair not in seen:
                    seen.add(pair)
                    out.append(pair)
    return out[:4]


def match_games(match_id: str) -> List[dict]:
    """[{number, id, state}] for a match. gameIds are present BEFORE games start."""
    if not match_id:
        return []
    hit = _games_cache.get(match_id)
    now = time.time()
    if hit and now - hit[0] < GAMES_TTL_S:
        return hit[1]
    sc, d = _get(f"{API}/getEventDetails", {"hl": "en-US", "id": match_id})
    if sc != 200 or not d:
        return hit[1] if hit else []
    games = ((((d.get("data") or {}).get("event") or {}).get("match") or {})
             .get("games") or [])
    out = [{"number": g.get("number"), "id": g.get("id"), "state": g.get("state")}
           for g in games if g.get("id")]
    _games_cache[match_id] = (now, out)
    return out


def game_start_by_id(game_id: str) -> Optional[datetime]:
    """EXACT start of a game (UTC), or None if it has not started.

    HTTP 204 == not started. HTTP 200 == started; frames[0].rfc460Timestamp is
    the first in_game frame and is the ground-truth start instant.
    """
    if not game_id:
        return None
    hit = _start_cache.get(game_id)
    now = time.time()
    if hit:
        ts, val = hit
        if val is not None:                      # resolved starts never change
            return val
        if now - ts < START_NEG_TTL_S:
            return None
    sc, d = _get(f"{FEED}/window/{game_id}")
    if sc == 204 or not d:
        _start_cache[game_id] = (now, None)
        return None
    frames = d.get("frames") or []
    if not frames:
        _start_cache[game_id] = (now, None)
        return None
    # SANITY: a genuine first frame is all zeros — the game instance has just
    # initialised and champions have not spawned (measured 26AUG10: gold 0/0,
    # kills 0/0, level 1, hp 0, on both REDLEV games and BROAP). If Riot ever
    # drops the opening frames and serves a mid-game frame as frames[0], the
    # timestamp would be LATER than the true start and callers would adopt
    # snapshots taken during the game — the one contamination this design exists
    # to prevent. Refuse it; "not started" is the safe failure direction.
    f0 = frames[0]
    for side in ("blueTeam", "redTeam"):
        t = f0.get(side) or {}
        if _f(t.get("totalGold")) > 0 or _f(t.get("totalKills")) > 0:
            log.warning("[RIOT] game %s: frames[0] is NOT a true first frame "
                        "(gold %s/%s kills %s/%s) — refusing it as a start time",
                        game_id, (f0.get('blueTeam') or {}).get('totalGold'),
                        (f0.get('redTeam') or {}).get('totalGold'),
                        (f0.get('blueTeam') or {}).get('totalKills'),
                        (f0.get('redTeam') or {}).get('totalKills'))
            _start_cache[game_id] = (now, None)
            return None
    raw = frames[0].get("rfc460Timestamp")
    try:
        val = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
    except Exception:
        log.warning("[RIOT] unparseable rfc460Timestamp %r for game %s", raw, game_id)
        _start_cache[game_id] = (now, None)
        return None
    _start_cache[game_id] = (now, val)
    return val


def game_start(match_id: str, game_number: int) -> Optional[datetime]:
    for g in match_games(match_id):
        if g["number"] == game_number:
            return game_start_by_id(g["id"])
    return None


def prefetch(match_ids: List[str], game_numbers=(1, 2)) -> None:
    """Warm the games/start caches for many matches concurrently.

    Resolving a slate serially costs ~0.6s per match (getEventDetails + one
    window poll per game), which on a 30-event slate is ~19s added to every
    populate cycle. Callers should invoke this once per pass before iterating;
    every subsequent match_games()/game_start() is then a cache hit. Failures
    are swallowed — the per-event path re-resolves and logs on its own.
    """
    ids = [m for m in dict.fromkeys(match_ids) if m]
    if not ids:
        return
    try:
        from concurrent.futures import ThreadPoolExecutor
    except Exception:
        return
    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
        list(pool.map(match_games, ids))
        gids = []
        for mid in ids:
            for g in _games_cache.get(mid, (0, []))[1]:
                if g["number"] in game_numbers:
                    gids.append(g["id"])
        list(pool.map(game_start_by_id, gids))


def match_phase(match_id: str) -> dict:
    """Everything a caller needs, in one call.

    {
      'resolved'   : bool  — Riot knew about this match at all
      'started'    : bool  — game 1 has started
      'starts'     : {game_number: datetime}   only games that HAVE started
      'active_game': int|None — highest started game number
      'anchor_ts'  : datetime|None — start of the active game; the instant a
                     caller's cached pregame solve must predate
    }
    """
    games = match_games(match_id)
    if not games:
        return {"resolved": False, "started": False, "starts": {},
                "active_game": None, "anchor_ts": None}
    starts = {}
    for g in games:
        t = game_start_by_id(g["id"])
        if t is not None:
            starts[g["number"]] = t
    active = max(starts) if starts else None
    return {
        "resolved": True,
        "started": bool(starts),
        "starts": starts,
        "active_game": active,
        "anchor_ts": starts.get(active) if active else None,
    }
