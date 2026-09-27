"""1-0 LEADER MAKER: rest long-leader maker bids during the set break after ANY
set-1 winner priced <= 70c vig-free at the anchor (REV 26SEP11-c, operator backtest:
"all 1-0 anchors at 70c or less, pregame dog or not" is the profitable cell — the
pregame-dog tag no longer gates and is kept only as logged telemetry). Registered as
model_name = "tennis_set1_dog"; in production it runs inside the
"tennis_dog_windows" dispatcher next to the set-3 model.

Scope: general_level_based_quoting/TENNIS_SET1_DOG_LEADER_SCOPE.md (sibling of the
set-3 dog maker — same quoter-framework mechanics, one set earlier), as amended by
REV 26SEP11-c. Cell: BO3, either player wins set 1 and is priced <= MAX_LEADER_VF
vig-free at the first-1-0 anchor. The original scoped cell (pregame dog only) is a
subset; the operator backtest found the <=70c-anchor superset profitable outright.

DIFFERENCES FROM THE SET-3 MODEL, all load-bearing:
  * Anchor = first CONSISTENT 1-0 row (round_winners len 1, exactly one completed
    round_scores set per side), either orientation. Late-detection guard: set-2
    games == 0 and points == 0 at that row.
  * PREGAME REFERENCE (telemetry only since REV -c — it no longer gates entry):
    first clean 0-0/0-0/0-0 tape tick with a finite vig_free, fallback
    tapes/seed_<ev>.json (the arm loop's discovery-mids price). Logged as
    pregame_vf on every trigger so the pregame-dog/pregame-fav split stays
    measurable; None when neither source exists.
  * THEOS ARE FROZEN AT THE TRIGGER, not a constant 49.5: leader ticker
    (bid_theo = leader vig-free at anchor in cents, offer_theo = 100); other ticker
    (bid_theo = 0, offer_theo = 100 - that). Same 1-99-bounds trick drops the
    unwanted sides => only leader-YES bid + fav NO bid, joined/improved up to the
    frozen anchor fair. No upward chase; the falling-knife disarm handles down moves.
  * THE price gate: leader anchor vig-free <= 0.70 (the backtested cell edge).
    NO lower band.
  * Book gates: live BID on both legs; an EMPTY leader OFFER is explicitly ALLOWED
    (32/134 in-sample dog-leader anchors had one and they carried outsized edge) —
    leader spread <= 8c applies only when an offer exists. With no leader ask the
    anchor vig-free falls back to 1 - other-side mid.
  * Hard timer 90s (SET1_TIMER_S, clamped 60-120): the 1-0 break is SHORTER than
    the 1-1 break (median 212s vs 336s, p5 132s; breaks under 90s: 0.7%).
  * Series: all six BO3 tennis series, each behind its own kill flag. ITF men was
    the weakest in-sample (+2.1pts) — cut by flag, not by code.

Shadow by default: theos only while ROOT/set1_live.flag exists; every trigger, tick,
cancel and skip goes to tapes/set1_dog_events.jsonl either way (replay input).
One-shot latch per match in tapes/_set1_dog_state.json. Fills ride to settlement.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from base_model import BaseTheoGenerator

log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAPES = os.path.join(ROOT, "tapes")

_STATE_PATH = os.path.join(TAPES, "_set1_dog_state.json")
_EVENTS_PATH = os.path.join(TAPES, "set1_dog_events.jsonl")
LIVE_FLAG = os.path.join(ROOT, "set1_live.flag")
KILL_FLAG = os.path.join(ROOT, "disable_set1_maker.flag")
SERIES_FLAGS = {s: os.path.join(ROOT, f"disable_set1_{tag}.flag") for s, tag in {
    "KXATPMATCH": "atp", "KXWTAMATCH": "wta",
    "KXATPCHALLENGERMATCH": "atpch", "KXWTACHALLENGERMATCH": "wtach",
    "KXITFMATCH": "itf", "KXITFWMATCH": "itfw"}.items()}
SERIES = tuple(SERIES_FLAGS)

MAX_LEADER_VF = 0.70                   # THE cell edge (REV 26SEP11-c): any set-1
                                       # winner at or under this vig-free at the
                                       # anchor is in; no lower band, no pregame tag
MAX_LEADER_SPREAD_C = 8.0              # only when a leader offer exists
KNIFE_C = 5.0
STALE_S = 20.0
ENTRY_WINDOW_S = 30.0
MAX_CONCURRENT = 4
TAIL_BYTES = 400_000
HEAD_BYTES = 3_000_000                 # pregame tick lives at the tape head
STATUS_EVERY_S = 5.0

_NOT_LIVE = {"not_started", "closed", "ended", "cancelled", "canceled",
             "finished", "completed", "postponed", ""}
# Grand Slam tournament tokens (mirrors kalshi_tennis._SLAMS)
_SLAMS = ("us open", "wimbledon", "roland garros", "french open", "australian open")


def _timer_s() -> float:
    try:
        return min(120.0, max(60.0, float(os.environ.get("SET1_TIMER_S", "90"))))
    except ValueError:
        return 90.0


def _c(v) -> Optional[float]:
    return None if v is None else v * 100.0


class Set1DogTheoGenerator(BaseTheoGenerator):
    """Frozen sided long-dog-leader theos for the 1-0 set break; nothing otherwise."""

    def __init__(self, client: Any, configs: List[Any] = None):
        self._events: Dict[str, dict] = self._load_state()
        self._anchor_rows: Dict[str, dict] = {}
        self._tape_cache: Dict[str, tuple] = {}
        self._meta_cache: Dict[str, tuple] = {}
        self._last_status: Dict[str, float] = {}
        self._warned: set = set()
        super().__init__(client, configs)

    # ------------------------------------------------------------- persistence
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
            log.exception("[SET1] could not persist state — one-shot latch is "
                          "memory-only until this succeeds")

    def _emit_event(self, kind: str, event: str, **kw):
        rec = {"ts": time.time(), "type": kind, "event": event}
        rec.update(kw)
        try:
            with open(_EVENTS_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            log.exception("[SET1] events jsonl write failed")

    # ------------------------------------------------------------- tape access
    def _warn_once(self, key: str, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            log.warning("[SET1] %s", msg)

    def _meta(self, event: str):
        if event in self._meta_cache:
            return self._meta_cache[event]
        try:
            d = json.load(open(os.path.join(TAPES, event + ".log.json")))
            m = d.get("meta") or {}
            got = (m.get("me_ticker"), m.get("opp_ticker"), d.get("best_of"),
                   m.get("tournament") or "")
        except Exception:
            return (None, None, None, "")
        self._meta_cache[event] = got
        return got

    def _latest(self, event: str) -> Tuple[Optional[dict], Optional[float]]:
        p = os.path.join(TAPES, event + ".jsonl")
        try:
            st = os.stat(p)
        except OSError:
            return None, None
        age = time.time() - st.st_mtime
        cached = self._tape_cache.get(event)
        if cached and cached[0] == st.st_mtime:
            return cached[1], age
        row = None
        try:
            with open(p, "rb") as f:
                f.seek(max(0, st.st_size - 8192))
                lines = f.read().decode("utf-8", "replace").splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                    break
                except Exception:
                    continue
        except Exception as e:
            self._warn_once(f"tape:{event}", f"{event} tape unreadable: {e}")
            return None, age
        self._tape_cache[event] = (st.st_mtime, row)
        return row, age

    def _tail_rows(self, event: str) -> List[dict]:
        p = os.path.join(TAPES, event + ".jsonl")
        try:
            with open(p, "rb") as f:
                f.seek(max(0, os.stat(p).st_size - TAIL_BYTES))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except Exception:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    # -------------------------------------------------------- pregame reference
    def _pregame_vf_me(self, event: str) -> Optional[float]:
        """Pre-match vig-free prob of the 'me' side, or None.

        Primary: first tape row at a completely untouched 0-0 (sets, games AND
        points all zero) carrying a finite vig_free — the poller's pregame capture.
        Fallback: the arm loop's seed_<ev>.json, whose single observation is the
        discovery-mids vig-free written before attach. Never derived from any row
        after play has started.
        """
        p = os.path.join(TAPES, event + ".jsonl")
        try:
            with open(p, "rb") as f:
                head = f.read(HEAD_BYTES).decode("utf-8", "replace").splitlines()
            for line in head:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                st = r.get("state") or {}
                if (st.get("sets_me"), st.get("sets_opp"), st.get("games_me"),
                        st.get("games_opp"), st.get("points_me"),
                        st.get("points_opp")) != (0, 0, 0, 0, 0, 0):
                    # play (or a mid-match attach) reached before any clean tick
                    if st.get("sets_me") is not None:
                        break
                    continue
                vf = r.get("vig_free")
                if isinstance(vf, (int, float)) and 0.0 < vf < 1.0:
                    return float(vf)
        except Exception:
            pass
        try:
            seed = json.load(open(os.path.join(TAPES, f"seed_{event}.json")))
            px = (seed[0] or {}).get("price")
            if isinstance(px, (int, float)) and 0.0 < px < 1.0:
                return float(px)
        except Exception:
            pass
        return None


    def _wsbook_c(self, event: str) -> dict:
        """Quoter's own WS top-of-book from tapes/wsbook_<ev>.jsonl (cents), same
        shape as _book_c, or all-None if absent/stale (>10s). Added 26SEP11 after
        BERKSH: pre-fix pollers (--markets-every 1) take the GET path every cycle,
        the GETs 429, and the first live set3 candidate expired bookless while the
        quoter's fresh WS book sat unused in the wsbook file. The wsbook is the
        better book anyway — sub-second WS vs a 3-4s polled REST snapshot."""
        empty = {"bid_me": None, "ask_me": None, "bid_opp": None, "ask_opp": None}
        p = os.path.join(TAPES, f"wsbook_{event}.jsonl")
        try:
            with open(p, "rb") as f:
                f.seek(max(0, os.stat(p).st_size - 4096))
                r = json.loads(f.read().decode("utf-8", "replace").strip().splitlines()[-1])
            if time.time() - r.get("ts", 0) > 10.0:
                return empty
            (bm, am), (bo, ao) = r.get("me") or (None, None), r.get("opp") or (None, None)
            ok = lambda v: v if isinstance(v, (int, float)) and 0 < v < 100 else None
            return {"bid_me": ok(bm), "ask_me": ok(am), "bid_opp": ok(bo), "ask_opp": ok(ao)}
        except Exception:
            return empty

    # ------------------------------------------------------------ row decoding
    @staticmethod
    def _sets(row) -> Tuple[Optional[int], Optional[int]]:
        st = row.get("state") or {}
        return st.get("sets_me"), st.get("sets_opp")

    @staticmethod
    def _row_consistent_10(row) -> bool:
        """A genuine, settled 1-0 payload in either orientation."""
        st = row.get("state") or {}
        det = row.get("details") or {}
        if (st.get("sets_me"), st.get("sets_opp")) not in ((1, 0), (0, 1)):
            return False
        if len(det.get("round_winners") or []) != 1:
            return False
        for n in (1, 2):
            rs = det.get(f"competitor{n}_round_scores") or []
            done = [x for x in rs if isinstance(x, dict) and x.get("outcome") != "ongoing"]
            if len(done) != 1:
                return False
        return True

    @staticmethod
    def _set2_untouched(row) -> bool:
        st = row.get("state") or {}
        return (st.get("games_me") == 0 and st.get("games_opp") == 0
                and st.get("points_me") == 0 and st.get("points_opp") == 0)

    @staticmethod
    def _book_c(row) -> Dict[str, Optional[float]]:
        bk = row.get("book") or {}
        return {k: _c(bk.get(k)) for k in ("bid_me", "ask_me", "bid_opp", "ask_opp")}

    # ---------------------------------------------------------------- flags
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

    # ---------------------------------------------------------------- states
    def _to_done(self, event: str, reason: str, was_armed: bool):
        e = self._events.setdefault(event, {})
        e["state"] = "done"
        e["reason"] = reason
        e["done_ts"] = time.time()
        self._save_state()
        self._emit_event("cancel" if was_armed else "skip", event, reason=reason)
        log.warning("[SET1] %s -> DONE (%s)%s", event, reason,
                    " — cancelling both legs" if was_armed else "")

    def _leader_book(self, e: dict, row: dict) -> Tuple[Optional[float], Optional[float]]:
        bk = self._book_c(row)
        if e.get("leader_is_me"):
            return bk["bid_me"], bk["ask_me"]
        return bk["bid_opp"], bk["ask_opp"]

    def _try_trigger(self, event: str, row: dict, age: float,
                     tickers_present: set) -> None:
        """WATCH: never / not-yet / ARM. Gates evaluate at the ANCHOR — the first
        consistent 1-0 row — for the same reason as the set-3 model: the market
        reprices during set point, and the cell was measured at the anchor."""
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
        if e.get("_scan_mtime") == mt:
            return
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

        # ---- gates, on the ANCHOR row ---------------------------------------
        me_tick, opp_tick, best_of, tournament = self._meta(event)
        if not me_tick or not opp_tick:
            self._warn_once(f"meta:{event}", f"{event}: no me/opp ticker in log.json — cannot arm")
            return
        if best_of != 3:
            self._to_done(event, f"best_of_{best_of}", was_armed=False)
            return
        # Grand Slams excluded outright (operator direction 26SEP11): best_of==3
        # keeps men's Slam mains out, but women's Slam matches (and Slam quallies)
        # are BO3 — this catches those by tournament name.
        if any(s in tournament.lower() for s in _SLAMS):
            self._to_done(event, "grand_slam_excluded", was_armed=False)
            return
        if not {me_tick, opp_tick} <= tickers_present:
            self._warn_once(f"legs:{event}", f"{event}: both legs must be configured "
                            f"({me_tick}, {opp_tick}) — not arming")
            return

        leader_is_me = self._sets(anchor) == (1, 0)
        # Pregame tag is TELEMETRY ONLY since REV 26SEP11-c — computed best-effort
        # and logged, never gating. The <=MAX_LEADER_VF anchor price below is the
        # whole entry condition beyond the structural gates.
        pre_me = self._pregame_vf_me(event)
        pre_leader = None if pre_me is None else (pre_me if leader_is_me else 1.0 - pre_me)

        bk = self._book_c(anchor)
        if bk["bid_me"] is None or bk["bid_opp"] is None:
            bk = self._wsbook_c(event)                 # quoter's live WS book (BERKSH fix)
        lb, la = (bk["bid_me"], bk["ask_me"]) if leader_is_me else (bk["bid_opp"], bk["ask_opp"])
        ob, oa = (bk["bid_opp"], bk["ask_opp"]) if leader_is_me else (bk["bid_me"], bk["ask_me"])
        if lb is None or ob is None:
            return                                     # live BID required both legs; retry in window
        if la is not None and (la - lb) > MAX_LEADER_SPREAD_C:
            self._to_done(event, f"leader_spread_{la - lb:.1f}c", was_armed=False)
            return
        if la is not None and oa is not None:
            ml, mo = (lb + la) / 2.0, (ob + oa) / 2.0
            leader_vf = ml / (ml + mo) if (ml + mo) > 0 else None
        elif oa is not None:
            # empty leader offer (allowed — the best in-sample subsample): price the
            # leader off the other side's mid, the only two-sided book available
            leader_vf = 1.0 - ((ob + oa) / 2.0) / 100.0
        else:
            return                                     # no way to price yet; retry in window
        if leader_vf is None or not (0.0 < leader_vf <= MAX_LEADER_VF):
            self._to_done(event, f"leader_vf_out_of_range_{-1 if leader_vf is None else round(leader_vf, 3)}",
                          was_armed=False)
            return

        n_armed = sum(1 for v in self._events.values() if v.get("state") == "armed")
        if n_armed >= MAX_CONCURRENT:
            self._to_done(event, "concurrency_cap", was_armed=False)
            return

        theo_c = round(leader_vf * 100.0, 2)
        leader_tick = me_tick if leader_is_me else opp_tick
        other_tick = opp_tick if leader_is_me else me_tick
        leader_mid_c = (lb + la) / 2.0 if la is not None else lb
        pre_log = None if pre_leader is None else round(pre_leader, 4)
        e.update({"state": "armed", "trigger_ts": now, "leader": leader_tick,
                  "other": other_tick, "leader_is_me": leader_is_me,
                  "theo_c": theo_c, "anchor_leader_mid_c": round(leader_mid_c, 2),
                  "pregame_vf": pre_log,
                  "series": event.split("-", 1)[0]})
        self._save_state()
        self._emit_event("trigger", event, leader=leader_tick, other=other_tick,
                         theo_c=theo_c, pregame_vf=pre_log,
                         leader_vf=round(leader_vf, 4), book=bk,
                         anchor_ts=e["anchor_ts"], live=self._live_armed(),
                         timer_s=_timer_s())
        log.warning("[SET1] %s TRIGGERED %s | leader=%s theo %.1fc (pregame %s, "
                    "anchor vf %.3f) | timer %.0fs | %s", event,
                    "LIVE" if self._live_armed() else "SHADOW",
                    leader_tick.rsplit("-", 1)[-1], theo_c,
                    "?" if pre_log is None else f"{pre_log:.3f}", leader_vf,
                    _timer_s(),
                    "resting maker bids both legs" if self._live_armed()
                    else "logging only (set1_live.flag absent)")

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

    def _theos_for(self, e: dict) -> Dict[str, Dict[str, float]]:
        t = e["theo_c"]
        return {e["leader"]: {"bid_theo": t, "offer_theo": 100.0},
                e["other"]: {"bid_theo": 0.0, "offer_theo": round(100.0 - t, 2)}}

    # ---------------------------------------------------------------- generate
    def _batch_generate(self, tickers: List[str],
                        dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        by_event: Dict[str, set] = {}
        for t in tickers:
            if t in self.configs:
                by_event.setdefault(t.rsplit("-", 1)[0], set()).add(t)

        for event, present in by_event.items():
            series = event.split("-", 1)[0]
            if series not in SERIES:
                self._warn_once(f"series:{event}", f"{event}: series {series} not in the "
                                f"set1 cell — config row is misrouted, ignoring")
                continue
            e = self._events.get(event)
            if e and e.get("state") == "done":
                continue

            kill = self._killed(series)
            row, age = self._latest(event)

            if e and e.get("state") == "armed":
                reason = kill or self._armed_disarm_reason(e, row, age)
                if reason:
                    self._to_done(event, reason, was_armed=True)
                    continue
                live = self._live_armed()
                if live:
                    out.update(self._theos_for(e))
                self._tick_log(event, e, row, live)
                continue

            if kill:
                continue
            if row is None:
                continue
            self._try_trigger(event, row, age, present)
            e = self._events.get(event)
            if e and e.get("state") == "armed" and self._live_armed():
                out.update(self._theos_for(e))
        return out

    def _tick_log(self, event: str, e: dict, row: Optional[dict], live: bool):
        mt = (self._tape_cache.get(event) or (None,))[0]
        if e.get("_logged_mtime") != mt and row is not None:
            e["_logged_mtime"] = mt
            self._emit_event("tick", event, leader=e["leader"], other=e["other"],
                             live=live, book=self._book_c(row),
                             armed_for_s=round(time.time() - e["trigger_ts"], 1),
                             theos=self._theos_for(e))
        now = time.time()
        if now - self._last_status.get(event, 0.0) >= STATUS_EVERY_S:
            self._last_status[event] = now
            left = _timer_s() - (now - e["trigger_ts"])
            b, a = self._leader_book(e, row) if row else (None, None)
            log.info("[SET1] %-18s ARMED %s | %4.0fs left | leader %s book %s/%s "
                     "(theo %.1fc)", event.rsplit("-", 1)[-1],
                     "LIVE" if live else "SHADOW", left,
                     e["leader"].rsplit("-", 1)[-1],
                     "-" if b is None else f"{b:.0f}",
                     "-" if a is None else f"{a:.0f}", e["theo_c"])
