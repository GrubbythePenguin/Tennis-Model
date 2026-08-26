"""Operator-input round-score plausibility gate — TEMPORARY (2026-07-21).

Interim stand-in for an automated round-score feed. While watching streams
(2-3 min delayed) the operator types each live game's map score into
manual_scores.csv; this turns that into a plausibility band (map_score_model)
and halts an event when the map-implied probability the bots are about to trade
is impossible for the entered score.

Motivation: KCGX / WALJUS-class phantom-hedge blips move the map consensus to a
level the game cannot produce (KCGX: cons_map implied 31% for a team sitting at
11-4, true ceiling ~5%). A human-entered score — even minutes stale — bounds
this, because the LEVEL check tolerates a few minutes of staleness by design
(a lopsided score cannot swing much in a few rounds; see map_score_model).

TOGGLE  : enable_manual_score_gate.flag must exist. Absent → total no-op.
SCOPE   : CS2 and VAL only (the sports map_score_model models). Others fail open.
FAIL-OPEN when: flag absent, no CSV row for the event, unknown sport, malformed
  row, or a team_suffix that doesn't match the row. The gate ONLY ever acts on
  events the operator has explicitly entered — everything else behaves exactly
  as today.
FAIL-SAFE: a stale operator score makes the band tighter around an OLD state, so
  the error is toward over-blocking (not trading), never toward trading a phantom.

manual_scores.csv (hot-reloaded on mtime, no restart):
    event_base,suffix_a,score_a,suffix_b,score_b,mode
    KXCS2GAME-26JUL200400WALJUS,WAL,4,JUS,11,live
  - score_a / score_b : current MAP round score for each team's Kalshi suffix.
  - mode : "live" (default; band from the score) or "pregame" (map not started;
           rejects only a near-decided price >=95% / <=5%).
  - Remove a row to stop gating that event.
"""
import csv
import os
import time
from typing import Optional, Tuple

import map_score_model as msm

_FLAG = "enable_manual_score_gate.flag"
_CSV = "manual_scores.csv"

# The market can legitimately be ahead of the operator's delayed score by the
# stream+entry lag; map_score_model projects the score forward by this many
# seconds when building the band. Kept modest on purpose — a lopsided score
# cannot reach parity in a few rounds, which is exactly why the check works.
STREAM_DELAY_SEC = 150.0
PAD_LIVE = 10.0       # extra pp of slack (iid model is overconfident at lopsided scores)
PREGAME_DECIDED_PCT = 95.0  # pregame: reject only near-decided prices (>=95 or <=5)

_cache: dict = {}
_mtime: float = 0.0


def _sport(event_base: str) -> Optional[str]:
    if event_base.startswith("KXCS2"):
        return "CS2"
    if event_base.startswith("KXVALORANT"):
        return "VAL"
    return None


def enabled() -> bool:
    return os.path.exists(_FLAG)


def _reload() -> None:
    global _cache, _mtime
    try:
        mt = os.path.getmtime(_CSV)
    except OSError:
        _cache, _mtime = {}, 0.0
        return
    if mt == _mtime and _cache:
        return
    new: dict = {}
    try:
        with open(_CSV, newline="") as f:
            for r in csv.DictReader(f):
                eb = (r.get("event_base") or "").strip()
                if not eb or eb.startswith("#"):
                    continue
                try:
                    new[eb] = {
                        "sa": (r.get("suffix_a") or "").strip(),
                        "pa": int(r.get("score_a")),
                        "sb": (r.get("suffix_b") or "").strip(),
                        "pb": int(r.get("score_b")),
                        "mode": (r.get("mode") or "live").strip().lower(),
                    }
                except (TypeError, ValueError):
                    continue  # malformed → skip; that event just isn't gated
        _cache, _mtime = new, mt
    except Exception:
        pass  # never break trading on a bad CSV read


def check(event_base: str, team_suffix: str, implied_prob_pct: float,
          now: Optional[float] = None) -> Tuple[bool, Optional[str]]:
    """Return (ok, reason). ok=False → caller should halt the event.

    implied_prob_pct : map-implied probability (0-100) that team_suffix's team
                       wins the active map — i.e. cons_map_yes_ask.
    """
    if not enabled():
        return True, None
    _reload()
    row = _cache.get(event_base)
    if not row:
        return True, None
    sport = _sport(event_base)
    if sport is None:
        return True, None
    if team_suffix == row["sa"]:
        my, opp = row["pa"], row["pb"]
    elif team_suffix == row["sb"]:
        my, opp = row["pb"], row["pa"]
    else:
        return True, None  # suffix not in row — cannot align, fail open

    if row["mode"] == "pregame":
        # A round-score model cannot bound a pre-start map — its fair value is
        # a team-strength prior, not derivable from 0-0. So pregame only rejects
        # the one thing that IS impossible before a map begins: a near-decided
        # price. Everything else (incl. heavy favorites) passes; the operator
        # switches the row to "live" with a real score once the map starts.
        if implied_prob_pct >= PREGAME_DECIDED_PCT or implied_prob_pct <= (100 - PREGAME_DECIDED_PCT):
            return False, (f"score_implausible(suffix={team_suffix} PREGAME "
                           f"implied={implied_prob_pct:.0f}% — map not started, "
                           f"cannot be near-decided)")
        return True, None

    now = time.time() if now is None else now
    age = max(0.0, now - _mtime) + STREAM_DELAY_SEC
    bad, lo, hi = msm.level_implausible(my, opp, implied_prob_pct,
                                        sport=sport, feed_age_sec=age, pad_pp=PAD_LIVE)
    if bad:
        return False, (f"score_implausible(suffix={team_suffix} "
                       f"score={my}-{opp} implied={implied_prob_pct:.0f}% "
                       f"plausible[{lo:.0f},{hi:.0f}] age={age:.0f}s)")
    return True, None
