"""Dota game-1-end detector, bo3.gg-backed — The International intermission.

Mirrors val_game_state's observe-and-persist shape (bo3 publishes NO per-game
timestamps: every instant must be OBSERVED as a transition and persisted —
miss the poll, lose the timestamp), but unlike that module this one also owns
its fetch and its bo3->Kalshi binding, so the populate wiring stays thin:

    dota_game_state.poll(cands)          # 1 list request, discipline 4
    dota_game_state.g1_ended_at(eb)      # observed ts | None

`cands` is the TI-gated candidate list from populate's anchor assembly:
[{event_base, team_a, team_b}] with team names from the Poly mapping's
team_alignments (Poly spellings — "BoomBoys", "Iron Wing", "TEAM VISION").

## What bo3 exposes for Dota (discipline 4; probed 2026-08-02, capture 26AUG02-08)

live_updates: per-team net_worth / game_score (KILLS) / match_score (maps),
game_number, game_ended. `live_coverage` was pre-known and NEVER flipped
mid-match in 33 observed Dota fixtures (coverage=false => zero live data,
22/22). Whether it flips at the upcoming->current transition is UNKNOWN as of
26AUG12 — all TI matches list false pre-match. This module therefore only
ACCELERATES the anchor's map-1-end detection; the book-pin + settlement +
timeout path in pregame_anchor stays load-bearing when bo3 stays dark.

## Game-1-end signals (any one stamps ended_at; first observation wins)

  1. game_number==1 and game_ended==true          (bo3's own flag)
  2. game_number==1 and match_score sum >= 1      (maps-won already credited)
  3. any live payload with game_number >= 2       (g1 is necessarily over)

All are bo3 STATE, never prices — same trust class as Riot's g1 'completed'.

## Binding — exact-canon only, no fuzzy

Team names are normalised (deaccent, lower, drop generic tokens: team/esports/
gaming/dota2/...) then canonicalised through bo3_team_aliases.csv (whole-name,
after the same normalisation; file reloads on mtime like the score feed). A
fixture binds ONLY when BOTH canon names equal the candidate's BOTH canon
names. An unbound TI fixture is printed loudly once — the fix is an alias row,
which takes effect within one poll, no restart (see the 26AUG12 TI block in
bo3_team_aliases.csv: betboom/boomboys, 1win/iron wing, parivision/vision,
l1ga/huligani).

Undocumented internal API — no ToS grant. A failed/empty fetch is "no data"
(the anchor falls back to the book pin), never an error.
"""
import csv
import json
import logging
import os
import re
import time
import unicodedata
from typing import Optional

import requests

log = logging.getLogger(__name__)

BASE = "https://api.bo3.gg/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DISCIPLINE_DOTA = 4
CACHE = "_dota_game_state.json"
CACHE_TTL_S = 36 * 3600
# bo3 is an undocumented internal API with no rate-limit headers. Self-throttle;
# populate's ~3.3min cycle is the real cadence, this floor only guards re-entry.
POLL_MIN_S = 30.0
_ALIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bo3_team_aliases.csv")

# Tokens that carry no identity — dropped before comparison so 'Team Falcons'
# == 'team-falcons-dota2' without an alias row.
_GENERIC_TOKENS = {"team", "esports", "esport", "gaming", "club", "org",
                   "dota", "dota2", "e"}

_state = None
_state_mtime = None
_last_poll_ts = 0.0
_alias_cache: dict = {}
_alias_mtime: float = -1.0
_warned_unbound: set = set()
_logged_coverage: set = set()


def _load() -> dict:
    global _state, _state_mtime
    try:
        mt = os.path.getmtime(CACHE)
    except OSError:
        if _state is None:
            _state = {}
        return _state
    if _state_mtime != mt or _state is None:
        try:
            with open(CACHE) as f:
                _state = json.load(f)
            _state_mtime = mt
        except Exception as e:
            log.warning("[DOTA] cache unreadable (%r) — starting empty", e)
            _state = {}
    return _state


def _save(d: dict) -> None:
    global _state_mtime
    now = time.time()
    for k in [k for k, v in d.items()
              if now - float(v.get("seen_at") or 0) > CACHE_TTL_S]:
        del d[k]
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, CACHE)
    try:
        _state_mtime = os.path.getmtime(CACHE)
    except OSError:
        pass


def _deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _aliases() -> dict:
    """Whole-name alias map from bo3_team_aliases.csv, both sides normalised
    with THIS module's _norm_name; reloaded on mtime change (data edits take
    effect within one poll, no restart — same contract as the score feed)."""
    global _alias_cache, _alias_mtime
    try:
        mt = os.path.getmtime(_ALIAS_FILE)
    except OSError:
        mt = 0.0
    if mt != _alias_mtime:
        merged: dict = {}
        if mt:
            try:
                with open(_ALIAS_FILE, newline="") as f:
                    for row in csv.reader(f):
                        if (not row or not row[0].strip()
                                or row[0].lstrip().startswith("#")):
                            continue
                        v = _norm_tokens(row[0])
                        c = _norm_tokens(row[1]) if len(row) > 1 else ""
                        if v == "variant" or not v or not c:
                            continue
                        merged[v] = c
            except Exception as e:
                log.warning("[DOTA] alias file unreadable (%s) — identity only", e)
                merged = {}
        _alias_cache, _alias_mtime = merged, mt
    return _alias_cache


def _norm_tokens(s: str) -> str:
    """Normalise WITHOUT alias lookup (used while loading the alias file)."""
    s = _deaccent(str(s or "")).lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in _GENERIC_TOKENS]
    return " ".join(toks)


def canon_name(s: str) -> str:
    """Normalised + alias-canonicalised team name ('' when nothing survives)."""
    n = _norm_tokens(s)
    return _aliases().get(n, n)


def _slug_teams(slug: str):
    """('betboom team dota2','og dota2') from a bo3 slug, date stripped."""
    s = re.sub(r"-\d{2}-\d{2}-\d{4}$", "", slug or "")
    parts = s.split("-vs-")
    if len(parts) != 2:
        return "", ""
    return parts[0].replace("-", " "), parts[1].replace("-", " ")


def _fetch_current() -> Optional[list]:
    """Live Dota matches with live_updates inline; None = fetch failed
    (distinct from an empty board, which is a real answer)."""
    try:
        r = requests.get(f"{BASE}/matches",
                         params={"filter[matches.status][eq]": "current",
                                 "filter[matches.discipline_id][eq]": DISCIPLINE_DOTA,
                                 "page[limit]": 30},
                         headers=HEADERS, timeout=8)
        if r.status_code != 200:
            log.warning("[DOTA] bo3 HTTP %s", r.status_code)
            return None
        return r.json().get("results", [])
    except Exception as e:
        log.warning("[DOTA] bo3 fetch failed: %r", e)
        return None


def _observe(eb: str, m: dict, now: float) -> None:
    """Fold one bound bo3 match payload into persisted state. Idempotent:
    only FIRST transitions are stamped (bo3 republishes every poll)."""
    d = _load()
    rec = d.setdefault(eb, {"match_id": str(m.get("id") or ""),
                            "slug": m.get("slug") or "",
                            "seen_at": now, "live_coverage": None})
    rec["seen_at"] = now
    if m.get("live_coverage") is not None:
        cov = bool(m.get("live_coverage"))
        # The 26AUG12 open question — does coverage flip at upcoming->current?
        # Say so loudly either way, once per event.
        if rec.get("live_coverage") != cov and eb not in _logged_coverage:
            _logged_coverage.add(eb)
            print(f"  [DOTA-BO3] {eb}: live_coverage={cov} "
                  f"({'bo3 g1-end detection ACTIVE' if cov else 'bo3 dark — book-pin path only'})")
        rec["live_coverage"] = cov

    lu = m.get("live_updates")
    if isinstance(lu, dict):
        gn = 0
        try:
            gn = int(lu.get("game_number") or 0)
        except (TypeError, ValueError):
            pass
        t1, t2 = (lu.get("team_1") or {}), (lu.get("team_2") or {})

        def _i(v):
            try:
                return int(v or 0)
            except (TypeError, ValueError):
                return 0
        maps_won = _i(t1.get("match_score")) + _i(t2.get("match_score"))
        g1_over = ((gn == 1 and bool(lu.get("game_ended")))
                   or (gn == 1 and maps_won >= 1)
                   or gn >= 2)
        if gn >= 1 and not rec.get("g1_first_seen_at"):
            rec["g1_first_seen_at"] = now
        if g1_over and not rec.get("g1_ended_at"):
            rec["g1_ended_at"] = now
            rec["g1_ended_via"] = ("gn>=2" if gn >= 2 else
                                   "game_ended" if lu.get("game_ended") else
                                   "match_score")
            print(f"  [DOTA-BO3] {eb}: game 1 ENDED (via {rec['g1_ended_via']}, "
                  f"game_number={gn}) — observed now")
        if gn >= 2 and not rec.get("g2_seen_at"):
            rec["g2_seen_at"] = now
    _save(d)


def poll(cands: list) -> None:
    """One bo3 list fetch; bind matches to `cands` and fold transitions.

    cands: [{event_base, team_a, team_b}] — TI-gated by the caller. Exact
    canon binding only; an unbindable live Dota fixture that LOOKS like a
    candidate's teams is impossible to detect here, so every unbound live
    fixture is printed once — operator adds the alias row, next poll binds.
    """
    global _last_poll_ts
    if not cands:
        return
    now = time.time()
    if now - _last_poll_ts < POLL_MIN_S:
        return
    _last_poll_ts = now
    matches = _fetch_current()
    if matches is None:
        return
    by_teams = {}
    for c in cands:
        ka, kb = canon_name(c.get("team_a")), canon_name(c.get("team_b"))
        if ka and kb and ka != kb:
            by_teams[frozenset((ka, kb))] = c["event_base"]
    for m in matches if isinstance(matches, list) else []:
        if not isinstance(m, dict):
            continue
        t1 = ((m.get("team1") or {}).get("name")
              if isinstance(m.get("team1"), dict) else "") or ""
        t2 = ((m.get("team2") or {}).get("name")
              if isinstance(m.get("team2"), dict) else "") or ""
        if not t1 or not t2:
            t1, t2 = _slug_teams(m.get("slug") or "")
        key = frozenset((canon_name(t1), canon_name(t2)))
        eb = by_teams.get(key)
        if eb is None:
            slug = m.get("slug") or "?"
            if slug not in _warned_unbound:
                _warned_unbound.add(slug)
                print(f"  [DOTA-BO3] unbound live fixture {slug!r} "
                      f"(canon {sorted(key)}) — if this is a TI candidate, add "
                      f"the alias row to bo3_team_aliases.csv (no restart)")
            continue
        _observe(eb, m, now)


def g1_ended_at(event_base: str) -> Optional[float]:
    """Observed ts of game 1 ending, else None. Late by up to one poll
    interval relative to the true throne fall — fine against Dota's measured
    21.5-29.5 min inter-game breaks."""
    return (_load().get(event_base) or {}).get("g1_ended_at")


def live_coverage(event_base: str) -> Optional[bool]:
    """bo3's coverage flag for a bound event; None = never seen it live."""
    return (_load().get(event_base) or {}).get("live_coverage")
