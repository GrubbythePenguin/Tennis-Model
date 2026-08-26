"""bo3.gg live-frames observer for LoL — the pregame-exit OR-branch.

Operator directive 26AUG15 (~09:20 UTC): pregame release = Riot game-start OR
bo3 frames, both captured in populate. Riot's livestats first-frame lags the
true start by 5-20 min (BLGTT, LNGNIP); bo3.gg's live_updates.net_worth flows
within the first minutes of the game (validated live on KTHLE 26AUG15: frames
08:08:49Z vs Riot flip ~08:09; and on LNGNIP where Riot was 15 min late).

NOT a draft/champ-select signal — bo3's LoL payload has no pick/ban phase
field, so this releases at GAME start, not draft start. Price-based draft
proxies were REJECTED by the operator (lineup-change confound, 26AUG15).

Mirrors dota_game_state.py: one list fetch per populate cycle (self-throttled),
exact canon binding via bo3_team_aliases.csv, first-transition-only stamps in
a persisted cache (STICKY — net_worth going quiet between games must never
re-latch a released event; see the DKCT1A re-latch incident).

Consumed by pregame_anchor.run_pass via ev["ext_started_at"] (injected in
populate_configs) -> pg source "bo3:frames". Kickoff anchor adoption still
keys off Riot's start exclusively — this module releases the GATE only.
"""
import json
import logging
import os
import re
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

BASE = "https://api.bo3.gg/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DISCIPLINE_LOL = 3
CACHE = "_lol_game_state.json"
CACHE_TTL_S = 36 * 3600
POLL_MIN_S = 30.0
FRAMES_MIN_NET_WORTH = 1000   # KTHLE fired at 2450 ~1-2 min in; 0/None = pre-game

_ALIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bo3_team_aliases.csv")
_GENERIC_TOKENS = {"team", "esports", "esport", "gaming", "club", "org",
                   "lol", "e"}

_last_poll_ts = 0.0
_warned_unbound: set = set()
_aliases: Optional[dict] = None
_aliases_mtime: float = 0.0


def _load_aliases() -> dict:
    global _aliases, _aliases_mtime
    try:
        mt = os.path.getmtime(_ALIAS_FILE)
    except OSError:
        return _aliases or {}
    if _aliases is None or mt != _aliases_mtime:
        _aliases_mtime = mt
        out = {}
        try:
            with open(_ALIAS_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        out[parts[0].strip().lower()] = parts[1].strip().lower()
        except OSError:
            pass
        _aliases = out
    return _aliases


def canon_name(name: Optional[str]) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    s = _load_aliases().get(s, s)
    toks = [t for t in s.split() if t not in _GENERIC_TOKENS]
    s2 = " ".join(toks)
    return _load_aliases().get(s2, s2)


def _slug_teams(slug: str):
    s = re.sub(r"-\d{2}-\d{2}-\d{4}$", "", slug or "")
    parts = s.split("-vs-")
    if len(parts) != 2:
        return "", ""
    return parts[0].replace("-", " "), parts[1].replace("-", " ")


def _load() -> dict:
    try:
        d = json.load(open(CACHE))
    except Exception:
        return {}
    now = time.time()
    return {k: v for k, v in d.items()
            if now - (v.get("seen_at") or now) < CACHE_TTL_S}


def _save(d: dict) -> None:
    try:
        json.dump(d, open(CACHE, "w"))
    except OSError as e:
        log.warning("[LOL-BO3] cache write failed: %r", e)


def _fetch_current() -> Optional[list]:
    try:
        r = requests.get(f"{BASE}/matches",
                         params={"filter[matches.status][eq]": "current",
                                 "filter[matches.discipline_id][eq]": DISCIPLINE_LOL,
                                 "page[limit]": 30},
                         headers=HEADERS, timeout=8)
        if r.status_code != 200:
            log.warning("[LOL-BO3] bo3 HTTP %s", r.status_code)
            return None
        return r.json().get("results", [])
    except Exception as e:
        log.warning("[LOL-BO3] bo3 fetch failed: %r", e)
        return None


def poll(cands: list) -> None:
    """One bo3 list fetch; stamp frames_live_at for bound candidates.

    First transition only — the stamp NEVER clears while cached (sticky:
    between-games net_worth silence must not re-latch a released event).
    Unbound live fixtures print once; operator adds the alias row (no restart).
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
    d = _load()
    dirty = False
    for m in matches if isinstance(matches, list) else []:
        if not isinstance(m, dict):
            continue
        t1, t2 = _slug_teams(m.get("slug") or "")
        key = frozenset((canon_name(t1), canon_name(t2)))
        eb = by_teams.get(key)
        if eb is None:
            slug = m.get("slug") or "?"
            if slug not in _warned_unbound:
                _warned_unbound.add(slug)
                print(f"  [LOL-BO3] unbound live LoL fixture {slug!r} "
                      f"(canon {sorted(key)}) — if this is a slate candidate, "
                      f"add the alias row to bo3_team_aliases.csv (no restart)")
            continue
        rec = d.setdefault(eb, {"slug": m.get("slug") or "", "seen_at": now})
        rec["seen_at"] = now
        if rec.get("frames_live_at") is None:
            lu = m.get("live_updates") or {}
            nw = (lu.get("team_1") or {}).get("net_worth")
            if isinstance(nw, (int, float)) and nw >= FRAMES_MIN_NET_WORTH:
                rec["frames_live_at"] = now
                dirty = True
                print(f"  [LOL-BO3] {eb}: FRAMES LIVE (net_worth={nw:.0f}) — "
                      f"pregame releases via bo3:frames this cycle")
        dirty = True
    if dirty:
        _save(d)


def frames_live_at(event_base: str) -> Optional[float]:
    """Sticky ts of first observed live frames, else None."""
    return (_load().get(event_base) or {}).get("frames_live_at")
