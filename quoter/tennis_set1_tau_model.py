"""TAU-GATED 1-0 LEADER MAKER (ITF only): rest a long-leader maker bid 6c below the
anchor midpoint during the set break after set 1, capped at the tau=0.55 model fair.
Registered inside the "tennis_dog_windows" dispatcher; model_name stays the same.

Scope (operator direction 26SEP17, built from the tau study in this repo's session
scratchpads; sibling of tennis_set1_dog_model, which keeps the non-ITF series):
  * Series: KXITFMATCH / KXITFWMATCH ONLY — the tau fit found the market prices
    ITF sets as independent (market tau = 0) while outcomes imply tau = 0.55
    (n=739, LR p=0.034); Challenger/Tour markets already price the update.
  * Anchor mechanics IDENTICAL to the set1 dog model: first consistent 1-0 row,
    late-detection guard, 30s entry window, 90s timer, falling-knife, staleness.
  * PRICE: bid_theo = min(anchor leader mid - MAKER_OFF_MID_C, tau fair) frozen at
    the trigger. The 6c-off-mid is the operator's edge demand; the tau-fair cap
    means we never rest above model fair even when the market mid is far above it
    (blowout leaders). NO <=70c gate — the tau model prices every anchor.
  * PREGAME REQUIRED: the tau fair needs the leader's pregame prob (clean 0-0 tick,
    fallback seed_<ev>.json). No pregame reference -> skip (unlike the dog model,
    where pregame is telemetry).

MAKER/TAKER MUTUAL EXCLUSION. tennis_set1_taker_bot.py (execution_type=set1_taker
rows, run through run.py like the esports taker) fires a 2000-lot IOC cross-book
sweep when the tau edge after fee-loaded cost clears its min_edge. Exactly one
of the two may act per match: both race for an O_CREAT|O_EXCL claim file in
tapes/set1_claims/<event>.claim. The maker additionally DEFERS arming for
TAKER_GRACE_S after the anchor whenever the anchor book is taker-eligible and the
taker's live flag is up, so the taker wins the claim when its condition holds at
the anchor (a resting bid 6c under mid is worth less than a guaranteed fill at a
>=6c-edge ask). A claim owned by "taker" is a terminal skip; the maker writes its
own claim at ARM time ONLY when live — a shadow maker must never block a live
taker (and a shadow taker never claims either, mirror rule in the taker script).

Shadow by default: theos only while ROOT/set1_tau_live.flag exists; every trigger,
tick, cancel and skip goes to tapes/set1_tau_events.jsonl either way. One-shot
latch per match in tapes/_set1_tau_state.json. Fills ride to settlement.
Kill flags: disable_set1_tau.flag (all), disable_set1_tau_itf/_itfw.flag.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from base_model import BaseTheoGenerator
from tennis_set1_dog_model import Set1DogTheoGenerator
import tau_fair

log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAPES = os.path.join(ROOT, "tapes")

_STATE_PATH = os.path.join(TAPES, "_set1_tau_state.json")
_EVENTS_PATH = os.path.join(TAPES, "set1_tau_events.jsonl")
CLAIMS_DIR = os.path.join(TAPES, "set1_claims")
LIVE_FLAG = os.path.join(ROOT, "set1_tau_live.flag")
KILL_FLAG = os.path.join(ROOT, "disable_set1_tau.flag")
TAKER_LIVE_FLAG = os.path.join(ROOT, "set1_taker_live.flag")
SERIES_FLAGS = {"KXITFMATCH": os.path.join(ROOT, "disable_set1_tau_itf.flag"),
                "KXITFWMATCH": os.path.join(ROOT, "disable_set1_tau_itfw.flag")}
SERIES = tuple(SERIES_FLAGS)

TAU = tau_fair.TAU_ITF                 # 0.55, the ITF fit
MAKER_OFF_MID_C = 6.0                  # rest this far under the anchor leader mid
TAKER_EDGE_C = 6.0                     # mirror of the taker's fire threshold, for
                                       # the grace check only — the taker owns it
TAKER_GRACE_S = 3.0                    # defer arming this long when taker-eligible
MAX_LEADER_SPREAD_C = 8.0
KNIFE_C = 5.0
STALE_S = 20.0
ENTRY_WINDOW_S = 30.0
MAX_CONCURRENT = 4
STATUS_EVERY_S = 5.0

_NOT_LIVE = {"not_started", "closed", "ended", "cancelled", "canceled",
             "finished", "completed", "postponed", ""}


def _timer_s() -> float:
    try:
        return min(120.0, max(60.0, float(os.environ.get("SET1_TAU_TIMER_S", "90"))))
    except ValueError:
        return 90.0


def in_maintenance_window(ts: Optional[float] = None) -> bool:
    """Kalshi weekly maintenance hard blackout: Thursdays 03:00-05:10
    America/New_York — same constants and rationale as populate_configs'
    WEEKLY_BLACKOUT (26AUG20: the status endpoint reported trading_active while
    books were 20c wide; a frozen-but-readable book paints phantom edge — the
    26SEP17 GRECHO shadow trigger, the only in-window taker-eligible that day,
    was also the only one whose "edge" reversed). NO trigger may arm and NO
    taker may fire inside it. disable_weekly_blackout.flag lifts it, mirroring
    the populate script's override."""
    if os.path.exists(os.path.join(ROOT, "disable_weekly_blackout.flag")):
        return False
    from datetime import datetime
    from zoneinfo import ZoneInfo
    d = datetime.fromtimestamp(ts if ts is not None else time.time(),
                               ZoneInfo("America/New_York"))
    return d.weekday() == 3 and (3, 0) <= (d.hour, d.minute) < (5, 10)


def claim(event: str, who: str) -> Optional[str]:
    """Atomically claim an event for `who` ("maker"/"taker"). Returns the owner
    after the call — `who` on success, the existing owner on a lost race, or
    None when the claims dir is unwritable (fail open, log loudly)."""
    try:
        os.makedirs(CLAIMS_DIR, exist_ok=True)
        fd = os.open(os.path.join(CLAIMS_DIR, f"{event}.claim"),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump({"by": who, "ts": time.time()}, f)
        return who
    except FileExistsError:
        return claim_owner(event)
    except Exception:
        log.exception("[SET1-TAU] claim write failed for %s — exclusion is "
                      "NOT guaranteed this match", event)
        return None


def claim_owner(event: str) -> Optional[str]:
    try:
        with open(os.path.join(CLAIMS_DIR, f"{event}.claim")) as f:
            return (json.load(f) or {}).get("by")
    except Exception:
        return None


class Set1TauTheoGenerator(Set1DogTheoGenerator):
    """Set1 dog-model mechanics with the tau price rule, ITF scope and claims.

    Subclasses the dog model for its tape access, pregame, consistency and
    wsbook plumbing; overrides state paths, flags, the trigger price logic and
    the series scope. The parent's _batch_generate/_armed_disarm/_theos_for/
    _tick_log run unchanged against this class's paths via the _PATHS hooks.
    """

    # ---- path/flag hooks (parent reads module globals; we override the methods
    # that touch them instead of the globals themselves)
    def _load_state(self) -> Dict[str, dict]:
        try:
            with open(_STATE_PATH) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _save_state(self):
        tmp = _STATE_PATH + ".tmp"
        try:
            clean = {ev: {k: v for k, v in e.items() if not k.startswith("_")}
                     for ev, e in self._events.items()}
            with open(tmp, "w") as f:
                json.dump(clean, f, indent=1)
            os.replace(tmp, _STATE_PATH)
        except Exception:
            log.exception("[SET1-TAU] could not persist state — one-shot latch "
                          "is memory-only until this succeeds")

    def _emit_event(self, kind: str, event: str, **kw):
        rec = {"ts": time.time(), "type": kind, "event": event}
        rec.update(kw)
        try:
            with open(_EVENTS_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            log.exception("[SET1-TAU] events jsonl write failed")

    def _killed(self, series: str) -> Optional[str]:
        if os.path.exists(KILL_FLAG):
            return "kill_flag"
        f = SERIES_FLAGS.get(series)
        if f and os.path.exists(f):
            return f"kill_flag_{series}"
        return None

    @staticmethod
    def _live_armed() -> bool:
        return os.path.exists(LIVE_FLAG)

    # ------------------------------------------------------------- the trigger
    def _try_trigger(self, event: str, row: dict, age: float,
                     tickers_present: set) -> None:
        sets = self._sets(row)
        status = (row.get("status") or "").strip().lower()
        if status in _NOT_LIVE and status:
            return
        if None in sets:
            return
        if sets[0] is not None and sets[1] is not None and sets[0] + sets[1] >= 2:
            self._to_done(event, "missed_window_sets_moved_on", was_armed=False)
            return
        if sets not in ((1, 0), (0, 1)):
            return
        if age is None or age > STALE_S:
            return

        e = self._events.setdefault(event, {"state": "watch"})
        mt = (self._tape_cache.get(event) or (None,))[0]
        if e.get("_scan_mtime") == mt and "anchor_ts" in e:
            pass                                  # still re-evaluate the grace path
        else:
            e["_scan_mtime"] = mt
        if "anchor_ts" not in e:
            rows = self._tail_rows(event)
            anchor = next((r for r in rows if self._row_consistent_10(r)), None)
            if anchor is None:
                return
            if not self._set2_untouched(anchor) or not (anchor.get("state") or {}).get("_points_known", True):
                self._to_done(event, "late_detection_guard", was_armed=False)
                return
            e["anchor_ts"] = anchor.get("ts") or time.time()
            e["anchor_sets"] = list(self._sets(anchor))
            self._save_state()
            self._anchor_rows[event] = anchor

        now = time.time()
        if now - e["anchor_ts"] > ENTRY_WINDOW_S:
            self._to_done(event, "entry_window_expired", was_armed=False)
            return
        if list(sets) != e.get("anchor_sets") or not self._set2_untouched(row):
            self._to_done(event, "set2_started_before_entry", was_armed=False)
            return

        anchor = self._anchor_rows.get(event)
        if anchor is None:
            rows = self._tail_rows(event)
            anchor = next((r for r in rows if self._row_consistent_10(r)), None)
            if anchor is None:
                return
            self._anchor_rows[event] = anchor

        # ---- structural gates on the ANCHOR ---------------------------------
        me_tick, opp_tick, best_of, tournament = self._meta(event)
        if not me_tick or not opp_tick:
            self._warn_once(f"meta:{event}", f"{event}: no me/opp ticker in log.json — cannot arm")
            return
        if best_of != 3:
            self._to_done(event, f"best_of_{best_of}", was_armed=False)
            return
        if not {me_tick, opp_tick} <= tickers_present:
            self._warn_once(f"legs:{event}", f"{event}: both legs must be configured "
                            f"({me_tick}, {opp_tick}) — not arming")
            return

        if in_maintenance_window(e["anchor_ts"]):
            self._to_done(event, "maintenance_window", was_armed=False)
            return

        owner = claim_owner(event)
        if owner == "taker":
            self._to_done(event, "claimed_by_taker", was_armed=False)
            return

        leader_is_me = self._sets(anchor) == (1, 0)
        pre_me = self._pregame_vf_me(event)
        if pre_me is None:
            self._to_done(event, "no_pregame_ref", was_armed=False)
            return
        pre_leader = pre_me if leader_is_me else 1.0 - pre_me

        bk = self._book_c(anchor)
        if bk["bid_me"] is None or bk["bid_opp"] is None:
            bk = self._wsbook_c(event)
        lb, la = (bk["bid_me"], bk["ask_me"]) if leader_is_me else (bk["bid_opp"], bk["ask_opp"])
        ob, oa = (bk["bid_opp"], bk["ask_opp"]) if leader_is_me else (bk["bid_me"], bk["ask_me"])
        if lb is None or ob is None:
            return                                 # live BID both legs; retry in window
        if la is not None and (la - lb) > MAX_LEADER_SPREAD_C:
            self._to_done(event, f"leader_spread_{la - lb:.1f}c", was_armed=False)
            return

        fair_c = round(tau_fair.fair_1_0(pre_leader, TAU) * 100.0, 2)
        leader_mid_c = (lb + la) / 2.0 if la is not None else lb
        theo_c = round(min(leader_mid_c - MAKER_OFF_MID_C, fair_c), 2)
        if theo_c < 1.0:
            self._to_done(event, f"theo_below_1c_{theo_c}", was_armed=False)
            return

        # ---- taker grace: when the anchor ask is taker-eligible and the taker
        # is live, hold back so the taker wins the claim race.
        taker_eligible = False
        if la is not None:
            fee_c = tau_fair.taker_fee(la / 100.0) * 100.0
            taker_eligible = (fair_c - la - fee_c) >= TAKER_EDGE_C
        if taker_eligible and os.path.exists(TAKER_LIVE_FLAG) \
                and now - e["anchor_ts"] < TAKER_GRACE_S:
            return                                 # re-checked next tick

        n_armed = sum(1 for v in self._events.values() if v.get("state") == "armed")
        if n_armed >= MAX_CONCURRENT:
            self._to_done(event, "concurrency_cap", was_armed=False)
            return

        if self._live_armed() and claim(event, "maker") == "taker":
            self._to_done(event, "claimed_by_taker", was_armed=False)
            return

        leader_tick = me_tick if leader_is_me else opp_tick
        other_tick = opp_tick if leader_is_me else me_tick
        e.update({"state": "armed", "trigger_ts": now, "leader": leader_tick,
                  "other": other_tick, "leader_is_me": leader_is_me,
                  "theo_c": theo_c, "anchor_leader_mid_c": round(leader_mid_c, 2),
                  "fair_c": fair_c, "pregame_vf": round(pre_leader, 4),
                  "series": event.split("-", 1)[0]})
        self._save_state()
        self._emit_event("trigger", event, leader=leader_tick, other=other_tick,
                         theo_c=theo_c, fair_c=fair_c, tau=TAU,
                         pregame_vf=round(pre_leader, 4),
                         anchor_leader_mid_c=round(leader_mid_c, 2),
                         taker_eligible=taker_eligible, book=bk,
                         anchor_ts=e["anchor_ts"], live=self._live_armed(),
                         timer_s=_timer_s())
        log.warning("[SET1-TAU] %s TRIGGERED %s | leader=%s theo %.1fc "
                    "(mid %.1fc - %.0fc, tau-fair %.1fc, pregame %.3f) | timer %.0fs",
                    event, "LIVE" if self._live_armed() else "SHADOW",
                    leader_tick.rsplit("-", 1)[-1], theo_c, leader_mid_c,
                    MAKER_OFF_MID_C, fair_c, pre_leader, _timer_s())

    def _armed_disarm_reason(self, e: dict, row: Optional[dict],
                             age: Optional[float]) -> Optional[str]:
        now = time.time()
        if now - e["trigger_ts"] >= _timer_s():
            return "timer"
        if row is None or age is None or age > STALE_S:
            return "tape_stale"
        status = (row.get("status") or "").strip().lower()
        if status in _NOT_LIVE:
            return f"status_{status or 'unknown'}"
        sets = self._sets(row)
        if list(sets) != e.get("anchor_sets", [None, None]):
            return f"sets_left_10_{sets[0]}-{sets[1]}"
        if not self._set2_untouched(row):
            return "set2_activity"
        b, a = self._leader_book(e, row)
        if b is None:
            wb = self._wsbook_c((e.get("leader") or "").rsplit("-", 1)[0])
            b, a = (wb["bid_me"], wb["ask_me"]) if e.get("leader_is_me") else (wb["bid_opp"], wb["ask_opp"])
        mid = (b + a) / 2.0 if (b is not None and a is not None) else b
        if mid is not None and mid < e["anchor_leader_mid_c"] - KNIFE_C:
            return "falling_knife"
        return None
