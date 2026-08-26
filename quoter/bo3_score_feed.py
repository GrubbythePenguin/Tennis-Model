"""bo3.gg round-score feed — tells us when a map is in the pickoff regime.

Polls bo3.gg (unauthenticated, free) for live CS2 + VALORANT round scores and
exposes them keyed by Kalshi event_base. The point is NOT to trade the score —
it is to know when a map has entered the coinflip / pickoff zone (past ~10-10,
e.g. FNC-KC game 1 at 13-13 OT), where our thin-edge quotes get run over no
matter how fast we are. A gate reads is_toxic()/get_round_score() and widens or
halts.

Design (mirrors bovada_price_feed): a background thread polls every
BO3_POLL_SEC, parses `live_updates`, binds each fixture to a Kalshi event, and
caches. Deliberately NOT low-latency — a few-minute-old round score still tells
you the regime (a lopsided map can't un-lopside in a few rounds; see
map_score_model).

Binding is FILE-DRIVEN, with name matching as the fallback (2026-07-30):
  bo3_parsed_markets.csv — written each cycle by refresh_tomorrow_markets, which
    already owns every other mapping artifact. Keyed on bo3's numeric match id
    (and slug). `source=pinned` rows are operator source of truth and survive
    rewrites. Read here on mtime change, so repairing a binding is a one-row
    edit that takes effect within one poll — no run.py restart, no interruption
    to quoting.
  bo3_team_aliases.csv — org-name aliases (bo3 'mousesports' == Kalshi 'MOUZ'),
    same live-reload story.
A fixture bo3 lists after the last discovery cycle still binds via name
matching, so a late listing is gated rather than silently dark.

bo3 specifics learned live (2026-07):
  - /api/v1/matches?filter[matches.status][eq]=current  (defaults discipline 1
    = CS2; VALORANT needs filter[matches.discipline_id][eq]=2).
  - live_updates: {"team_1":{"game_score":13,...},"team_2":{"game_score":13,...},
    "map_name":..,"game_number":1,"round_phase":"IN_PROGRESS","game_ended":false}
    game_score = current map round score; match_score = maps won.
  - live_updates is null when there's no coverage → treat as "unknown", NOT safe.
  - Slugs are NOT guessable (carry disambiguators like fnatic-1-vs-...); always
    find matches via the list endpoint, never by constructing a slug.

Undocumented internal API — no ToS grant; can change or lock without notice.
Interpret an empty/failed fetch as "no data" (fail toward caution), never as
"map is early / safe".
"""
import csv
import logging
import os
import re
import threading
import time
import unicodedata
from typing import Optional, Tuple

import requests

log = logging.getLogger(__name__)

BO3_BASE = "https://api.bo3.gg/api/v1"
DISCIPLINES = {"CS2": 1, "VAL": 2}
BO3_POLL_SEC = 150.0          # a few minutes is plenty for regime detection
BO3_TIMEOUT = 30.0            # bo3 can be slow; be patient
STALE_AFTER_SEC = 600.0       # a score older than this is treated as unknown
# Toxicity: a map is in the pickoff/coinflip regime when one team has reached
# TOXIC_LEAD_SCORE rounds AND the map is still within TOXIC_MAX_DIFF rounds of
# level — i.e. close AND late. Thin map markets get run over here no matter how
# fast we are; a lopsided-but-late map (13-4) is NOT toxic (it can't flip).
TOXIC_LEAD_SCORE = 10
TOXIC_MAX_DIFF = 3
# Per-sport override BANDS: toxic if ANY (lead, diff) band matches. CS2 gets
# an early-but-tight band (8,2) — PAINNIP 26JUL26 spent the whole map at
# 8-9 / 9-9 / 10-10 with every maker fill ~-10c markout, and the single >=10
# rule slept through all but the 10-10 tail — plus the original late band
# (10,3). VAL keeps the single 10/3 rule its live-verified track record was
# built on (EDGFPX 07-23).
TOXIC_THRESHOLDS = {"CS2": ((8, 2), (10, 3)), "VAL": ((10, 3),)}


def thresholds_for(sport) -> Tuple[Tuple[int, int], ...]:
    """Toxicity bands ((lead, diff), ...) for a sport key ('CS2'/'VAL')."""
    return TOXIC_THRESHOLDS.get(sport or "",
                                ((TOXIC_LEAD_SCORE, TOXIC_MAX_DIFF),))


def describe_thresholds(sport) -> str:
    """Human-readable band list for log lines, e.g. 'lead>=8&diff<=2 or
    lead>=10&diff<=3'."""
    return " or ".join(f"lead>={lead}&diff<={diff}"
                       for lead, diff in thresholds_for(sport))


def event_sport(event_base) -> Optional[str]:
    """Sport key from a Kalshi event base, or None if unrecognized."""
    eb = event_base or ""
    if "KXCS2" in eb:
        return "CS2"
    if "KXVALORANT" in eb:
        return "VAL"
    return None


def _int_or_none(v):
    """Coerce to int, or None if not a finite integer value. Guards against bo3
    ever handing us a string / null / float('nan') / object for a score."""
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if i == i else None   # reject NaN (int(nan) raises, but belt+braces)


def _score_is_toxic(s_a, s_b, sport=None) -> bool:
    try:
        return any(max(s_a, s_b) >= lead and abs(s_a - s_b) <= diff
                   for lead, diff in thresholds_for(sport))
    except TypeError:
        return False
_POLY_PARSED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "poly_parsed_markets.csv")


def _deaccent(s: str) -> str:
    """Strip diacritics so 'Leviatán' == 'leviatan'. Without this, the [^a-z0-9]
    tokenizer treats the accented char as a delimiter and truncates the token
    (Leviatán -> 'leviat') while bo3's plain 'leviatan' stays whole, so they
    never intersect and the game goes ungated. Fail-safe: returns input on error."""
    if not isinstance(s, str):
        return ""
    try:
        return "".join(c for c in unicodedata.normalize("NFKD", s)
                       if not unicodedata.combining(c))
    except (TypeError, ValueError):
        return s


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _deaccent(s).lower())


def _teams_from_slug(slug: str) -> Tuple[str, str]:
    """bo3 list rows carry team names only in the slug (team1 objects are id-only).

    'fnatic-1-vs-karmine-corp-22-07-2026' -> ('fnatic', 'karmine corp'). Strips
    the trailing DD-MM-YYYY date and any '-N' disambiguator suffix per side.
    """
    if not isinstance(slug, str) or "-vs-" not in slug:
        return "", ""
    left, right = slug.split("-vs-", 1)
    right = re.sub(r"-\d{2}-\d{2}-\d{4}$", "", right)   # drop date

    def clean(part):
        part = re.sub(r"-\d+$", "", part)               # drop disambiguator
        return part.replace("-", " ").strip()
    return clean(left), clean(right)


def _fetch_current(discipline_id: int, status: str = "current") -> list:
    """Return matches of one status for one discipline, or [] on any failure."""
    try:
        r = requests.get(
            f"{BO3_BASE}/matches",
            params={"filter[matches.status][eq]": status,
                    "filter[matches.discipline_id][eq]": discipline_id,
                    "page[limit]": 100},
            timeout=BO3_TIMEOUT)
        if r.status_code != 200:
            log.warning("[BO3] discipline %d HTTP %d", discipline_id, r.status_code)
            return []
        body = r.json()
        res = body.get("results", []) if isinstance(body, dict) else []
        return res if isinstance(res, list) else []
    except Exception as e:
        log.warning("[BO3] fetch discipline %d failed: %s", discipline_id, e)
        return []


def fetch_fixtures(statuses=("current", "upcoming")) -> list:
    """Parsed bo3 fixtures across both disciplines and the given statuses.

    Shared with refresh_tomorrow_markets so the mapping it writes and the
    binding this module does at poll time come from ONE implementation — two
    copies of the name-matching rules is how they drift apart.
    """
    out = []
    for sport, did in DISCIPLINES.items():
        for status in statuses:
            for m in _fetch_current(did, status):
                if not isinstance(m, dict):
                    continue
                try:
                    rec = _parse_match(m, sport)
                except Exception as e:
                    log.warning("[BO3] skipped malformed fixture: %s", e)
                    continue
                if rec and rec.get("team1") and rec.get("team2"):
                    rec["status"] = status
                    out.append(rec)
    return out


def _d(x):
    """A dict or {} — every bo3 nested field is defensively coerced through this,
    since the API can hand back null / string / list where we expect an object."""
    return x if isinstance(x, dict) else {}


def _parse_match(m: dict, sport: str) -> Optional[dict]:
    m = _d(m)
    lu = m.get("live_updates")
    if not isinstance(lu, dict):
        lu = None                         # null / string / list => no live feed
    slug = m.get("slug") if isinstance(m.get("slug"), str) else ""
    t1 = _d(m.get("team1")).get("name") or ""
    t2 = _d(m.get("team2")).get("name") or ""
    if not t1 or not t2:   # list endpoint: names only in the slug
        s1, s2 = _teams_from_slug(slug)
        t1, t2 = t1 or s1, t2 or s2
    if not lu:
        # covered=False or no live feed — record teams but no score (unknown).
        return {"sport": sport, "team1": t1, "team2": t2, "score1": None,
                "score2": None, "game_number": None, "map_name": None,
                "game_ended": None, "covered": bool(m.get("live_coverage")),
                "slug": slug, "match_id": m.get("id"),
                "start_date": m.get("start_date") or "",
                "bo_type": _int_or_none(m.get("bo_type"))}
    a = _d(lu.get("team_1"))
    b = _d(lu.get("team_2"))
    return {"sport": sport,
            "team1": t1 or a.get("name") or "", "team2": t2 or b.get("name") or "",
            "score1": a.get("game_score"), "score2": b.get("game_score"),
            "game_number": lu.get("game_number"), "map_name": lu.get("map_name"),
            "game_ended": bool(lu.get("game_ended")),
            "covered": True, "slug": slug, "match_id": m.get("id"),
            "start_date": m.get("start_date") or "",
            "bo_type": _int_or_none(m.get("bo_type"))}


def _load_kalshi_events() -> list:
    """[(event_base, [team_a_name, team_b_name])] from poly_parsed_markets.csv.

    team names are the Poly-outcome names (col 5, 'A; B'), used to align bo3.
    """
    out = []
    if not os.path.exists(_POLY_PARSED):
        return out
    try:
        with open(_POLY_PARSED, newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].startswith("KX"):
                    continue
                if not (row[0].startswith("KXVALORANT") or row[0].startswith("KXCS2")):
                    continue
                teams = [t.strip() for t in (row[4] if len(row) > 4 else "").split(";") if t.strip()]
                if len(teams) == 2:
                    out.append((row[0], teams))
    except Exception as e:
        log.warning("[BO3] kalshi event load failed: %s", e)
    return out


# "esport" (singular) added 2026-07-30: it was missing while "esports" was
# present, so 'Sashi Esport' vs 'Esport Academy Copenhagen' shared the token
# 'esport'. Both bo3 sides then mapped to the SAME Kalshi team (match_idx takes
# the first intersecting index), i1 == i2 failed the distinctness check, and the
# fixture went unmatched — the same inert-gate outcome as a missing alias.
_GENERIC_TOKENS = {"esports", "esport", "gaming", "team", "club", "academy",
                   "the", "gg", "org", "cs", "csgo", "valorant",
                   # Roster markers are consulted separately, via _roster_tier.
                   # Leaving them in the DISTINCTIVE set breaks academy-vs-
                   # academy fixtures: both teams then share 'challengers', both
                   # map to the same Kalshi index, and the distinctness check
                   # rejects a fixture that should have bound (the 'esport'
                   # collapse again, one family down).
                   "challengers", "prospects", "youth", "rising",
                   "junior", "juniors", "changers"}

# Token aliases: orgs whose bo3.gg spelling shares NO distinctive token with the
# Poly-outcome spelling in poly_parsed_markets.csv. Without an alias the two
# sides never intersect, _match_to_kalshi returns None, and the toxicity gate is
# silently INERT for that entire series — no error, just suppress=False forever
# (MOUZ vs 3DMAX, 2026-07-30 BLAST Bounty: bo3 read 11-11 on de_nuke g2 while
# the quoter ran ungated).
#
# Both sides are canonicalized through this map, so either spelling matches.
# Keys are post-tokenizer forms (lower-case, de-accented, punctuation stripped).
#
# Add entries ONLY for confirmed same-org spellings. Name similarity is NOT
# sufficient evidence — the same board carries 'Sinners'/'Sentinels' and
# 'paiN'/'regain', which are different orgs and must never be aliased.
_TOKEN_ALIASES = {
    # bo3 'mousesports' vs Kalshi/Poly 'MOUZ' — no shared token at all.
    "mousesports": "mouz",
    # Poly 'TheMongolz' tokenizes whole ('themongolz'); bo3 writes 'the mongolz'
    # and the tokenizer drops the generic 'the', leaving 'mongolz'.
    "themongolz": "mongolz",
    # Poly 'Gen.G Esports' -> the dot splits it, 'g' is too short and 'esports'
    # generic, leaving {'gen'}; bo3 'geng esports' leaves {'geng'}. Canonicalize
    # to the more distinctive 'geng' so a bare 'gen' elsewhere can't collide.
    "gen": "geng",
}


# Operator-editable alias file. Same shape as the dict above (variant,canonical),
# '#' comments allowed. Merged OVER the built-ins, so the code defaults keep
# working when the file is absent and the file can also override one of them.
# Re-read on mtime change by every consumer, so adding an alias needs no restart
# of anything — see _ALIAS_FILE / _MAPPING_FILE notes in the module docstring.
_ALIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bo3_team_aliases.csv")
_alias_cache: dict = {}      # merged variant -> canonical
_alias_mtime: float = -1.0


def _load_aliases() -> dict:
    """Built-in aliases merged with _ALIAS_FILE, reloaded when the file changes."""
    global _alias_cache, _alias_mtime
    try:
        mt = os.path.getmtime(_ALIAS_FILE)
    except OSError:
        mt = 0.0                      # absent file: built-ins only
    if mt != _alias_mtime:
        merged = dict(_TOKEN_ALIASES)
        if mt:
            try:
                with open(_ALIAS_FILE, newline="") as f:
                    for row in csv.reader(f):
                        if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                            continue
                        variant = _norm(row[0])
                        canon = _norm(row[1]) if len(row) > 1 else ""
                        if variant == "variant":      # header
                            continue
                        if variant and canon:
                            merged[variant] = canon
            except Exception as e:
                # A malformed alias file must never break score polling — fall
                # back to whatever we had rather than losing the gate entirely.
                log.warning("[BO3] alias file unreadable (%s); using built-ins", e)
                merged = dict(_TOKEN_ALIASES)
        _alias_cache, _alias_mtime = merged, mt
    return _alias_cache


def _canon_token(t: str) -> str:
    """Map a token to its canonical spelling (identity when not aliased)."""
    return _load_aliases().get(t, t)


def _dtokens(s: str) -> set:
    """Distinctive tokens: 'Falcons Esports' -> {'falcons'}, '100 Thieves' ->
    {'100','thieves'}. Falls back to the whole normalized string if nothing
    survives (e.g. an all-generic name). Names and tokens are canonicalized
    through _TOKEN_ALIASES so bo3 and Poly spellings of the same org intersect.

    Aliases apply at BOTH levels, whole-name first:
      whole-name  'largadosepelados' -> 'largadosypelados'   (bo3 writes
                  'largados_e_pelados', which tokenizes to {largados, pelados}
                  and shares nothing with Kalshi's single 'largadosypelados'
                  token, so a per-token alias cannot express the fix)
      per-token   'mousesports' -> 'mouz'
    Without the whole-name level, the alias row that the discovery-time
    suggester prints (it emits normalized whole names) would be silently inert.
    """
    whole = _norm(s)
    canon = _load_aliases().get(whole)
    if canon:
        s, whole = canon, _norm(canon)
    ts = {_canon_token(t) for t in re.split(r"[^a-z0-9]+", _deaccent(s).lower())
          if len(t) >= 3 and t not in _GENERIC_TOKENS}
    return ts or ({_canon_token(whole)} if whole else set())


MAPPING_COLS = ["kalshi_event", "bo3_match_id", "bo3_slug", "bo3_team1",
                "bo3_team2", "bo3_start_date", "map_no", "source"]
# map_no: blank for a SERIES row (the gate wants whatever map is live now), or
# the map number for a KX*MAP-...-N row. A single bo3 fixture therefore fans out
# to several targets — the series event plus one per map — which is why the
# mapping is a list per fixture rather than one event_base.
# Kalshi-event -> bo3-fixture mapping, written each discovery cycle by
# refresh_tomorrow_markets (which already owns every other mapping artifact) and
# READ here at poll time. Two reasons it lives in a file rather than in this
# process:
#   * an operator can repair a binding by editing one row — it takes effect on
#     the next poll (<=BO3_POLL_SEC), with no run.py restart and no interruption
#     to quoting. `source=pinned` marks such a row and refresh preserves it.
#   * the binding is then auditable after the fact: what the gate believed at
#     the time is on disk, not reconstructed from token-matching logic.
# Name-matching remains as the FALLBACK for fixtures bo3 listed after the last
# discovery cycle, so a late listing is still gated rather than silently dark.
_MAPPING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bo3_parsed_markets.csv")
# Kalshi event bases are uppercase, dash-separated, whitespace-free
# (KXCS2GAME-26JUL310830MOUZ3DMAX). Anything else in the file is corruption.
_EB_RE = re.compile(r"KX[A-Z0-9]+-[A-Z0-9-]+")
_map_cache: dict = {"by_id": {}, "by_slug": {}}
_map_stamp = None       # (mtime, size) — size too, so a rewrite that lands in
                        # the same mtime tick can't be served from a stale cache


def load_mapping() -> dict:
    """{'by_id': {bo3_match_id: event_base}, 'by_slug': {slug: event_base}}.

    Re-read whenever the file changes. Missing file => empty mapping, which
    degrades to pure name-matching (the pre-2026-07-30 behaviour). Any read
    error degrades the same way: a broken mapping file must never be worse than
    no mapping file.

    utf-8-sig: a hand-edit saved from Excel carries a BOM, which would otherwise
    corrupt the first header name and silently void every binding in the file.
    """
    global _map_cache, _map_stamp
    try:
        st = os.stat(_MAPPING_FILE)
        stamp = (st.st_mtime, st.st_size)
    except OSError:
        stamp = None
    if stamp != _map_stamp:
        by_id, by_slug = {}, {}
        if stamp:
            try:
                with open(_MAPPING_FILE, newline="", encoding="utf-8-sig") as f:
                    bad = 0
                    for r in csv.DictReader(f):
                        eb = (r.get("kalshi_event") or "").strip().upper()
                        if not eb or eb.startswith("#"):
                            continue
                        # map_no absent (old file) or blank => series row.
                        mn = _int_or_none((r.get("map_no") or "").strip())
                        # Shape-check the event base. A corrupted row (embedded
                        # newline from a bad hand-edit, a stray quote splicing
                        # two rows together) would otherwise create a phantom
                        # binding to an event that cannot exist. Dropping it
                        # sends the fixture to name matching instead.
                        if not _EB_RE.fullmatch(eb):
                            bad += 1
                            continue
                        mid = (r.get("bo3_match_id") or "").strip()
                        slug = (r.get("bo3_slug") or "").strip().lower()
                        if mid:
                            by_id.setdefault(mid, []).append((eb, mn))
                        if slug:
                            by_slug.setdefault(slug, []).append((eb, mn))
                    if bad:
                        log.warning("[BO3] mapping file: skipped %d malformed "
                                    "row(s); those fixtures fall back to name "
                                    "matching", bad)
            except Exception as e:
                log.warning("[BO3] mapping file unreadable (%s); "
                            "falling back to name matching", e)
                by_id, by_slug = {}, {}
        _map_cache, _map_stamp = {"by_id": by_id, "by_slug": by_slug}, stamp
    return _map_cache


_conflict_warned: set = set()


def _bind_all(rec: dict, kalshi_events: list) -> list:
    """[(event_base, map_no, how)] — every Kalshi target this fixture feeds.

    One bo3 fixture legitimately drives several markets: the SERIES event
    (map_no None — "whatever map is live now") plus one target per per-map
    market (KX*MAP-...-N, map_no N). Name matching can only ever produce the
    series binding, so map targets come exclusively from the mapping file.

    `how` is 'map-id' / 'map-slug' / 'name' / '' — recorded on the record so the
    snapshot and the monitor can show whether a binding came from the file or
    from fuzzy name matching.

    The file WINS over name matching: that is the whole point of a pin (the
    operator pins precisely the fixtures whose names don't match). But a file
    binding that contradicts a confident name match is logged loudly — that
    combination means either a stale row or a typo, and it would silently feed
    another match's round score into this event's gate.
    """
    mp = load_mapping()
    mid = str(rec.get("match_id") or "").strip()
    slug = (rec.get("slug") or "").strip().lower()
    targets = mp["by_id"].get(mid) if mid else None
    how = "map-id"
    if not targets and slug:
        targets, how = mp["by_slug"].get(slug), "map-slug"
    name_eb = _match_to_kalshi(rec.get("team1", ""), rec.get("team2", ""),
                               kalshi_events, rec.get("sport"))
    if targets:
        # Compare the SERIES target against the name match; map targets have no
        # name-matching counterpart, so they cannot conflict by construction.
        series_ebs = [eb for eb, mn in targets if mn is None]
        if name_eb and series_ebs and name_eb not in series_ebs:
            key = (slug or mid, series_ebs[0], name_eb)
            if key not in _conflict_warned:
                _conflict_warned.add(key)
                log.warning("[BO3] MAPPING CONFLICT for bo3 '%s' (%s vs %s): the "
                            "mapping file says %s, the team names say %s. Using "
                            "the file. Check bo3_parsed_markets.csv — a stale or "
                            "mistyped row feeds the WRONG match's score to this "
                            "event's gate.", slug, rec.get("team1"),
                            rec.get("team2"), series_ebs[0], name_eb)
        return [(eb, mn, how) for eb, mn in targets]
    return [(name_eb, None, "name")] if name_eb else []


def _bind(rec: dict, kalshi_events: list) -> Tuple[Optional[str], str]:
    """Back-compat single binding: the SERIES target (or the first available).

    Kept because callers and tests that predate per-map targets ask "which event
    does this fixture belong to", which is still a well-formed question.
    """
    got = _bind_all(rec, kalshi_events)
    if not got:
        return None, ""
    for eb, mn, how in got:
        if mn is None:
            return eb, how
    eb, _mn, how = got[0]
    return eb, how


# Secondary-roster markers. An org's Game Changers / academy / challengers side
# is a DIFFERENT team playing a DIFFERENT match, but the tokenizer cannot see the
# difference: "GC" is two characters so it is dropped for being too short, and
# "academy"/"challengers" are on the generic list. So 'GIANTX GC' and 'GIANTX'
# both reduce to {'giantx'}, and a Game Changers fixture would bind to the main
# VCT market — feeding one match's round score into another match's gate.
# Requiring the marker sets to AGREE makes that a non-match instead. When the two
# sources disagree about the marker (Kalshi's 'Los Heretics' is the academy but
# doesn't say so), the fixture goes unmatched and the gate stays inert — the
# right failure direction, and an operator pin fixes it without a restart.
_ROSTER_MARKERS = ("gc", "game changers", "academy", "challengers", "prospects",
                   "youth", "rising", "junior", "juniors")


def _roster_tier(name: str) -> frozenset:
    """Secondary-roster markers present in a team name ('' for a main roster)."""
    low = f" {_deaccent(str(name or '')).lower()} "
    low = re.sub(r"[^a-z0-9]+", " ", low)
    return frozenset(m for m in _ROSTER_MARKERS if f" {m} " in low)


def _match_to_kalshi(team1: str, team2: str, kalshi_events: list,
                     sport: Optional[str] = None) -> Optional[str]:
    """Align a bo3 match (team1,team2) to a Kalshi event_base.

    Both bo3 teams must map to the two DISTINCT Kalshi teams via a shared
    distinctive token ('Falcons Esports' <-> 'Team Falcons' share 'falcons').
    Requiring two different targets prevents a single generic-token collision
    from half-mapping to a wrong event. Secondary-roster markers (GC / academy /
    challengers) must additionally agree — see _ROSTER_MARKERS.
    """
    b1, b2 = _dtokens(team1), _dtokens(team2)
    if not b1 or not b2:
        return None
    bt1, bt2 = _roster_tier(team1), _roster_tier(team2)
    half = []          # (event_base, unmatched_bo3_name, unmatched_kalshi_name)
    for eb, teams in kalshi_events:
        # Sport must agree. Most big orgs field BOTH a CS2 and a VALORANT roster
        # (fnatic, Team Heretics, Liquid, NAVI, G2, FURIA, MIBR...), so the same
        # two team names legitimately describe two different matches in two
        # different games. Without this, a VALORANT map score can drive a CS2
        # series' toxicity gate. Callers that don't know the sport pass None and
        # get the old cross-sport behaviour.
        if sport and event_sport(eb) and event_sport(eb) != sport:
            continue
        kt = [_dtokens(t) for t in teams]
        ktier = [_roster_tier(t) for t in teams]

        def match_idx(bt, tier):
            for i, k in enumerate(kt):
                if bt & k and tier == ktier[i]:
                    return i
            return -1
        i1, i2 = match_idx(b1, bt1), match_idx(b2, bt2)
        if i1 >= 0 and i2 >= 0 and i1 != i2:
            return eb
        # len(teams) == 2 guard: a malformed row (one team, or three) must not
        # IndexError here — this is diagnostics only and must never be able to
        # take down a poll. _load_kalshi_events only emits 2-team rows, but this
        # function is also called with hand-built lists.
        if len(teams) == 2 and (i1 >= 0) != (i2 >= 0):
            hit = i1 if i1 >= 0 else i2
            half.append((eb, team2 if i1 >= 0 else team1, teams[1 - hit]))
    _warn_probable_alias_miss(half)
    return None


_alias_warned: set = set()


def _warn_probable_alias_miss(half) -> None:
    """Log when a half-match looks like a MISSING _TOKEN_ALIASES entry.

    A half-match alone is normal — an org we trade in one fixture appears in
    other fixtures we don't. What is NOT normal is the two unmatched names
    being near-identical strings (bo3 'the mongolz' vs Poly 'TheMongolz'), which
    means the same fixture failed to bind and the gate is inert for it. Only
    that case is logged, so the signal stays rare enough to act on.
    """
    for eb, bo3_name, kalshi_name in half:
        bn, kn = _norm(bo3_name), _norm(kalshi_name)
        if not bn or not kn:
            continue
        near = (bn in kn or kn in bn
                or len(os.path.commonprefix([bn, kn])) >= 4)
        if not near:
            continue
        key = (bn, kn)
        if key in _alias_warned:
            continue
        _alias_warned.add(key)
        if _roster_tier(bo3_name) != _roster_tier(kalshi_name):
            # NOT an alias miss — these are different rosters of one org, and
            # aliasing them would bind an academy/GC match's score to the main
            # market. Refusing to match is correct; say so, or the next operator
            # "fixes" it into a silent mis-map.
            log.warning("[BO3] roster-tier mismatch: %s bo3 '%s' (%s) vs kalshi "
                        "'%s' (%s) — treated as DIFFERENT teams and left "
                        "unmatched. Do NOT add an alias. If the Kalshi name is "
                        "the one that's mislabelled, pin the row in "
                        "bo3_parsed_markets.csv instead.",
                        eb, bo3_name, sorted(_roster_tier(bo3_name)) or "main",
                        kalshi_name, sorted(_roster_tier(kalshi_name)) or "main")
            continue
        log.warning("[BO3] probable alias miss: %s bo3 '%s' vs kalshi '%s' "
                    "— gate stays INERT for this event; add a row to "
                    "bo3_team_aliases.csv (no restart needed)",
                    eb, bo3_name, kalshi_name)


class Bo3ScoreFeed:
    def __init__(self):
        self._cache: dict = {}      # event_base -> parsed dict + updated_ts
        self._by_slug: dict = {}    # slug -> parsed dict (for unmatched display)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── polling ──
    def poll_once(self) -> dict:
        """Blocking single poll of both disciplines. Returns event_base -> record."""
        kalshi = _load_kalshi_events()
        now = time.time()
        matched, by_slug = {}, {}
        for sport, did in DISCIPLINES.items():
            for m in _fetch_current(did):
                try:
                    if not isinstance(m, dict):
                        continue        # bo3 handed us a non-object in results
                    base_rec = _parse_match(m, sport)
                    if not base_rec:
                        continue
                    base_rec["updated_ts"] = now
                    by_slug[base_rec.get("slug", "")] = base_rec
                    for eb, map_no, how in _bind_all(base_rec, kalshi):
                        if not eb:
                            continue
                        # One record object per target: the map targets differ
                        # only in bound_map_no, and sharing a dict would let the
                        # last one written decide every target's map check.
                        rec = dict(base_rec)
                        rec["bind_src"] = how
                        rec["bound_map_no"] = map_no
                        rec["event_base"] = eb
                        prev = matched.get(eb)
                        # Same fixture reaching one target twice (a duplicated
                        # mapping row) is a no-op, not a conflict — only a
                        # DIFFERENT fixture claiming the target is worth warning.
                        if prev is not None and prev.get("slug") != rec.get("slug"):
                            # Two live fixtures claiming one event: a stale
                            # mapping row alongside a fresh name match, or a
                            # rescheduled/duplicated bo3 listing. Dict-order
                            # last-wins could let a FINISHED map overwrite the
                            # live one and silently un-suppress a toxic series,
                            # so prefer the record that actually has a live
                            # score, and warn either way.
                            prev_live = (_int_or_none(prev.get("score1")) is not None
                                         and not prev.get("game_ended"))
                            rec_live = (_int_or_none(rec.get("score1")) is not None
                                        and not rec.get("game_ended"))
                            keep = rec if (rec_live and not prev_live) else prev
                            log.warning("[BO3] DUPLICATE BINDING for %s: '%s' and "
                                        "'%s' both map here; keeping '%s'. Check "
                                        "bo3_parsed_markets.csv for a stale row.",
                                        eb, prev.get("slug"), rec.get("slug"),
                                        keep.get("slug"))
                            matched[eb] = keep
                        else:
                            matched[eb] = rec
                except Exception as e:   # one bad match must not sink the poll
                    log.warning("[BO3] skipped malformed match: %s", e)
                    continue
        with self._lock:
            self._cache = matched
            self._by_slug = by_slug
        return matched

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as e:
                log.error("[BO3] poll loop error: %s", e)
            self._stop.wait(BO3_POLL_SEC)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bo3-score-feed",
                                        daemon=True)
        self._thread.start()
        log.info("[BO3] score feed started (poll %.0fs)", BO3_POLL_SEC)

    def stop(self):
        self._stop.set()

    # ── consumers ──
    def _map_ok(self, rec) -> bool:
        """False when this target is a per-map market and bo3 is showing a
        DIFFERENT map. A KX*MAP-...-1 quoter must never be driven by map 2's
        round score. Series targets (bound_map_no None) always pass — for them
        'whatever map is live' is exactly the right question."""
        want = rec.get("bound_map_no")
        if want is None:
            return True
        return _int_or_none(rec.get("game_number")) == want

    def get_round_score(self, event_base: str) -> Optional[Tuple[int, int, int, float]]:
        """Return (score_team_a, score_team_b, game_number, age_sec) or None.

        a/b are the Kalshi event's teams in poly_parsed order (col 5), so a=
        first-listed team. None when: no match, no live score, or stale.
        """
        with self._lock:
            rec = self._cache.get(event_base)
        if not isinstance(rec, dict):
            return None
        if not self._map_ok(rec):
            return None
        s1, s2 = _int_or_none(rec.get("score1")), _int_or_none(rec.get("score2"))
        ts = rec.get("updated_ts")
        if s1 is None or s2 is None or not isinstance(ts, (int, float)):
            return None
        age = time.time() - ts
        if age > STALE_AFTER_SEC:
            return None
        return s1, s2, _int_or_none(rec.get("game_number")) or 0, age

    def is_toxic(self, event_base: str) -> Optional[bool]:
        """True if the active map is in the pickoff regime (see _score_is_toxic).

        None when the score is unknown/stale OR the map just ended (between-maps
        frame: game_ended with a still-populated final score) — caller must NOT
        read None as safe.
        """
        with self._lock:
            rec = self._cache.get(event_base)
        if not isinstance(rec, dict) or rec.get("game_ended"):
            return None
        if not self._map_ok(rec):
            return None
        s1, s2 = _int_or_none(rec.get("score1")), _int_or_none(rec.get("score2"))
        ts = rec.get("updated_ts")
        if s1 is None or s2 is None or not isinstance(ts, (int, float)):
            return None
        if time.time() - ts > STALE_AFTER_SEC:
            return None
        return _score_is_toxic(s1, s2, rec.get("sport"))


_FEED = Bo3ScoreFeed()


def get_default_feed() -> Bo3ScoreFeed:
    return _FEED


# ── toxicity gate (consumed by live_series_model; OFF by default) ──
_GATE_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "enable_bo3_toxicity_gate.flag")

# Debug heartbeat: log what the gate sees, but only for an ON + in-scope event,
# and only when the (score/game/ended/toxic) picture CHANGES or every
# DEBUG_HEARTBEAT_SEC (proves liveness without spamming — bo3 only refreshes
# every BO3_POLL_SEC anyway).
DEBUG_HEARTBEAT_SEC = 120.0
_dbg_last: dict = {}    # event_base -> (last_log_ts, last_signature)


def gate_enabled() -> bool:
    """True only when enable_bo3_toxicity_gate.flag is present on disk."""
    return os.path.exists(_GATE_FLAG)


def _gate_scope():
    """Event-base substrings the gate is scoped to — one per non-comment line of
    the flag file. Empty file (or only comments) => None => the gate applies to
    ALL matched events. Lets the operator arm the gate for a single event (e.g.
    'KCFNC' for a live test) without touching any other market."""
    try:
        with open(_GATE_FLAG) as f:
            subs = [ln.strip() for ln in f
                    if ln.strip() and not ln.lstrip().startswith("#")]
        return subs or None
    except OSError:
        return None


def toxicity_active(event_base: str) -> bool:
    """True iff the gate is ON, this event is in scope, and bo3 says its active
    map is in the pickoff regime (see _score_is_toxic; live & fresh).

    Reports STATE only — the consumer decides the action. live_series_model
    turns a True here into a maker-only suppression (pull our resting quotes;
    takers keep firing, since near-toxic map takers are the informed, profitable
    side). Sport-agnostic: applies to any CS2/VAL event the feed covers.

    Contract, by design:
      - Flag ABSENT  -> return False immediately. The feed thread is never
        started (no bo3 network I/O), so an off gate is a total no-op.
      - Flag PRESENT but event NOT in scope (flag lists substrings, none match)
        -> return False without touching the feed.
      - In scope -> lazily start the poll thread (idempotent, non-blocking;
        first poll runs on the thread, hot path never waits on network) and read
        the cached score.
      - Score UNKNOWN/STALE/ended (is_toxic -> None) -> return False. bo3 is
        ADDITIVE safety; we never suppress a market bo3 simply can't see.
      - ANY internal error -> return False (fail OPEN: keep quoting). This runs
        in the live quoting loop; it must never raise.
    """
    try:
        if not gate_enabled():
            return False
        scope = _gate_scope()
        if scope is not None and not any(s in (event_base or "") for s in scope):
            return False
        _FEED.start()                   # idempotent; no-op if already running
        return _FEED.is_toxic(event_base) is True
    except Exception as e:
        log.warning("[BO3] toxicity_active(%r) errored (%s) — failing OPEN "
                    "(no suppression)", event_base, e)
        return False


def gate_debug(event_base: str) -> Optional[str]:
    """Throttled one-liner of what the gate currently sees for an ON, in-scope
    event — for run.log visibility during a live test. Returns None (nothing to
    log) when the gate is off, the event is out of scope, or the picture is
    unchanged and within DEBUG_HEARTBEAT_SEC of the last line. Read-only: never
    starts the feed or changes the halt decision.
    """
    try:
        if not gate_enabled():
            return None
        scope = _gate_scope()
        if scope is not None and not any(s in (event_base or "") for s in scope):
            return None
        now = time.time()
        with _FEED._lock:
            raw = _FEED._cache.get(event_base)
            rec = dict(raw) if isinstance(raw, dict) else {}
        tox = _FEED.is_toxic(event_base)
        sig = (rec.get("score1"), rec.get("score2"), rec.get("game_number"),
               bool(rec.get("game_ended")), tox)
        last_ts, last_sig = _dbg_last.get(event_base, (0.0, None))
        if sig == last_sig and now - last_ts < DEBUG_HEARTBEAT_SEC:
            return None
        _dbg_last[event_base] = (now, sig)
        if not rec:
            return (f"{event_base}: NO bo3 record yet (unmatched, or feed not "
                    f"polled) toxic={tox} -> suppress={tox is True}")
        ts = rec.get("updated_ts")
        age = now - ts if isinstance(ts, (int, float)) else float("nan")
        return (f"{event_base}: {rec.get('score1')}-{rec.get('score2')} "
                f"(g{rec.get('game_number')} {rec.get('map_name')}) "
                f"ended={rec.get('game_ended')} age={age:.0f}s "
                f"toxic={tox} -> suppress={tox is True}")
    except Exception:
        return None    # debug logging must never break the quoting loop


def _print_snapshot(feed):
    import datetime
    matched = feed.poll_once()
    with feed._lock:
        recs = list(feed._by_slug.values())
    scored = [r for r in recs if r.get("score1") is not None]
    stamp = time.strftime("%H:%M:%S")
    print(f"\n[{stamp}] bo3 CS2+VAL scores  (matched={len(matched)} scored={len(scored)})")
    if not scored:
        print("  (no scored matches)")
    for r in sorted(scored, key=lambda r: -(min(r['score1'], r['score2']))):
        eb = r.get("event_base", "")
        between = r.get("game_ended")
        toxic = (not between) and _score_is_toxic(r["score1"], r["score2"],
                                                  r.get("sport"))
        tag = ("  *** TOXIC — quotes terrible ***" if toxic
               else "  [between maps]" if between else "")
        src = r.get("bind_src") or ""
        km = f"  kalshi={eb} [{src}]" if eb else "  (unmatched)"
        print(f"  [{r['sport']}] {r['team1']} {r['score1']}-{r['score2']} {r['team2']}"
              f"  (g{r['game_number']} {r['map_name']}){km}{tag}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    feed = get_default_feed()
    if "--watch" in sys.argv:
        print("bo3 score watch — polling every %.0fs (Ctrl-C to stop)" % BO3_POLL_SEC,
              flush=True)
        while True:
            try:
                _print_snapshot(feed)
                sys.stdout.flush()
            except Exception as e:
                print("poll error:", e, flush=True)
            time.sleep(BO3_POLL_SEC)
    else:
        _print_snapshot(feed)
