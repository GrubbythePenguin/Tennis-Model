"""Is this event still PREGAME? Shared state, written by populate, read by bots.

Why a file and not a lookup: `QuoterBot.evaluate` runs in the hot quoting loop.
A `riot_game_state` cache miss is a blocking HTTP round-trip, and putting one on
the quote path would dwarf every latency fix on the docket. populate already
resolves `game_start()` for every LoL event on each anchor pass, so it writes the
answer here and the bots read it in microseconds (mtime-cached).

The 26AUG11 DNFHLE loss is the shape this exists for: the quoter has no pregame
concept at all. `pregame_taker_block` gates the ARBER's taker path
(arber_bot.py:1587) and always has; the maker channel was never covered, so we
rested full-size clips into a 35k cap before the rosters were even known and ate
a 60c inversion on ~24k lots.

## Failure policy — deliberately asymmetric

  file missing / stale / unparseable  -> None, NO gating, logged LOUDLY.
      populate dying must not silently shrink every quote on every sport. The
      blast radius of fail-closed here is the whole book.

  event present, Riot says UNSTARTED  -> True (PREGAME), i.e. fail CLOSED.
      This is the roster-substitution case. On DNFHLE, Riot reported all five
      games `unstarted` for the whole event because it was tracking the
      main-roster fixture that never played, so the event would have stayed
      gated to reduced size throughout — the correct answer.

  event present, join UNRESOLVED      -> pregame null -> None, no gating.
      Riot never joined the fixture at all (LPLOL is outside the index).
      Zero information must not gate the book — fail-closed here would mean
      never trading the game at size (operator decision 26AUG11). The record
      stays in the file with source "riot:unresolved" so the event is visibly
      tracked, but is_pregame returns None and the gate never applies.

  event absent from the file          -> None, no gating.
      Non-LoL sports and anything populate does not track. Riot carries zero
      Dota2, so this must never be read as "pregame".

Never let a lookup here raise into the quote path. Every public function returns
a value; callers treat None as "no opinion".
"""
import json
import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)

STATE_PATH = "_pregame_state.json"
# populate cycles ~3 min. Past this the file is not describing the current slate
# and we stop trusting it rather than gate off a frozen snapshot.
STALE_AFTER_S = 15 * 60

_cache = {"mtime": None, "data": None}
_warned = {"stale": 0.0, "missing": 0.0}
_WARN_EVERY_S = 300.0


def _warn(key: str, msg: str) -> None:
    """Loud, but not once per quote tick."""
    now = time.time()
    if now - _warned.get(key, 0.0) >= _WARN_EVERY_S:
        _warned[key] = now
        log.warning(msg)


def _load() -> Optional[dict]:
    try:
        mt = os.path.getmtime(STATE_PATH)
    except OSError:
        _warn("missing", f"[PREGAME] {STATE_PATH} absent — pregame sizing INACTIVE "
                         f"(quotes at full size). Is populate_configs running?")
        return None
    if _cache["mtime"] != mt:
        try:
            with open(STATE_PATH) as f:
                _cache["data"] = json.load(f)
            _cache["mtime"] = mt
        except Exception as e:
            _warn("missing", f"[PREGAME] {STATE_PATH} unreadable ({e!r}) — pregame sizing INACTIVE")
            return None
    d = _cache["data"] or {}
    written = d.get("written_ts")
    if not isinstance(written, (int, float)) or time.time() - written > STALE_AFTER_S:
        age = "unknown" if not isinstance(written, (int, float)) else f"{(time.time()-written)/60:.1f}min"
        _warn("stale", f"[PREGAME] state file is STALE (age={age}) — pregame sizing "
                       f"INACTIVE (quotes at full size)")
        return None
    return d


# ── Operator force-live override (26AUG14, DKCT1A) ──────────────────────────
# Riot's start signal can lag or regress ('riot:unstarted' while the game is
# plainly live on stream — DKCT1A sat latched ~30min, then RE-latched after a
# populate restart when Riot regressed). One event_base per line in this file
# forces is_pregame() -> False for that event: the pregame config latch
# (min_edge=999, 0.1x caps/volumes) releases on the next populate cycle and
# stays released regardless of Riot. Remove the line (or the file) to revert.
# Lines starting with # are comments. Mtime-cached; edits take effect <=5s.
FORCE_LIVE_FILE = "_pregame_force_live.txt"
_force_cache = {"mtime": None, "checked": 0.0, "events": frozenset()}


def _force_live() -> frozenset:
    now = time.time()
    if now - _force_cache["checked"] < 5.0:
        return _force_cache["events"]
    _force_cache["checked"] = now
    try:
        mt = os.path.getmtime(FORCE_LIVE_FILE)
        if mt != _force_cache["mtime"]:
            with open(FORCE_LIVE_FILE) as f:
                _force_cache["events"] = frozenset(
                    l.strip() for l in f if l.strip() and not l.startswith("#"))
            _force_cache["mtime"] = mt
    except OSError:
        _force_cache["events"] = frozenset()
        _force_cache["mtime"] = None
    return _force_cache["events"]


def is_pregame(event_base: str) -> Optional[bool]:
    """True = pregame, False = live, None = no opinion (do not gate).

    None and False are NOT interchangeable: None means we have nothing to say
    (wrong sport, file down), False means we affirmatively saw a start.
    An event listed in _pregame_force_live.txt is ALWAYS False (operator
    override for a lagging/regressed Riot start signal).
    """
    if event_base in _force_live():
        return False
    d = _load()
    if not d:
        return None
    rec = (d.get("events") or {}).get(event_base)
    if rec is None:
        return None
    v = rec.get("pregame")
    return v if isinstance(v, bool) else None


def detail(event_base: str) -> Optional[dict]:
    """The full record — {pregame, source, start_ts} — for logging."""
    d = _load()
    if not d:
        return None
    return (d.get("events") or {}).get(event_base)


def write_state(events: dict, path: str = None) -> None:
    """Called by populate. `events` is {event_base: {pregame, source, start_ts}}.

    Atomic replace so a bot never reads a half-written file. `path` defaults to
    the MODULE-LEVEL constant at call time (not def time) so test harnesses can
    redirect STATE_PATH — a def-time default let run_pass-driven tests clobber
    the production file.
    """
    path = path or STATE_PATH
    payload = {"written_ts": time.time(), "events": events}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
