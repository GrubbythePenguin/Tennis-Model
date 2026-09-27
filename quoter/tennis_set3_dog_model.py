"""SET-3 DOG MAKER: rest long-dog maker bids during the set break after a momentum dog
levels an ITF match at 1-1. Registered as model_name = "tennis_set3_dog".

Scope: general_level_based_quoting/TENNIS_SET3_DOG_MAKER_SCOPE.md (REV 26SEP10-b —
implemented as a QuoterBot row through this framework, not a bespoke module). Cell:
ITF, dog priced 30-50c vig-free at 1-1, dog won set 2. In-sample wr 59.0% vs 42.1%
implied (n=61, p=0.004).

SIDED THEOS ARE THE WHOLE MECHANISM. This model never prices the match — during the
armed window it wants exactly two resting orders, both long the dog, and nothing else:

    dog ticker   bid_theo = 49.5   offer_theo = 100.0   -> YES bid only, capped 49c
    fav ticker   bid_theo =  0.0   offer_theo =  49.5   -> NO  bid only, floored 50c YES

quoter.py's 1-99 bounds drop the two unwanted sides (a flat 49.5/49.5 pair would ALSO
emit a dog offer at 50 and a fav knife-catch bid at 49 — measured before this REV, all
four levels land 39-61c so no order-price floor removes them). With min_edge=0 and
min_distance_from_top_level=-1 the surviving side joins/improves top of book, bounded
by the 49.5 theo. Outside the armed window the model returns NOTHING, which is the
framework's proven no-theo cancel-all invariant. Orders are post_only by construction
in execution.py, so the strategy is pure maker — a would-be-marketable improve is
cancelled by Kalshi, never crossed.

STATE MACHINE, one shot per match, persisted across restarts in _STATE_PATH:
    WATCH  -> ARMED   all trigger gates hold at one tape snapshot (below)
    WATCH  -> DONE    1-1 seen too late / missed / match moved on (never entered)
    ARMED  -> DONE    first of: timer, set-3 activity, falling knife, staleness,
                      kill flag, match state left 1-1. No re-entry ever.

TRIGGER GATES (all at the same tape row):
    * series KXITFMATCH / KXITFWMATCH (per-series kill flags), best_of == 3 (log.json)
    * first 1-1 sighting in the tape has set-3 games == 0 AND points == 0 — the
      LATE-DETECTION GUARD: a poller that was dark and woke mid-set-3 must not enter.
      The first-1-1 row is found by tape scan, so a quoter restart mid-break anchors
      to the real transition time, not to when this process first looked.
    * trigger row is self-consistent: round_winners has exactly 2 entries and both
      competitors show exactly 2 completed round_scores sets. The feed emits transient
      reset rows mid-match (sets snap to 0-0, round_winners []) — those must neither
      trigger nor pass as a first sighting.
    * momentum: round_winners[1] (set-2 winner UUID) maps to the DOG ticker. UUIDs are
      bound to me/opp via the server field: details.server is a competitor UUID and
      state.server says whether that competitor is "me" — no dependence on having
      watched the 1-0/0-1 transition live.
    * dog (cheaper side) vig-free prob in [0.30, 0.50); both books two-sided;
      dog spread <= MAX_DOG_SPREAD_C.
    * the anchor dog is ALSO the PREGAME dog (vig-free < 0.50 before the match;
      reference = first clean 0-0 tape tick, fallback seed_<ev>.json — added
      26SEP11: a pregame favourite slid under 50c by 1-1 is not the measured cell).
    * fewer than MAX_CONCURRENT other matches currently armed.

DISARM (first wins; reason is persisted and logged):
    * hard timer: trigger + SET3_TIMER_S (default 120s, clamped 90-150; p<2% of
      observed set-3 first points once the guard removes late-detection artifacts)
    * set-3 activity: any game/point appears, or sets leave 1-1, or status not live
    * falling knife: dog mid drops > KNIFE_C below trigger mid (retirement tail risk)
    * staleness fail-closed: tape older than STALE_S — never rest blind quotes

SHADOW BY DEFAULT. Without ROOT/set3_live.flag the model logs every trigger, intended
quote, per-tick book and cancel reason to tapes/set3_dog_events.jsonl but emits no
theos (nothing can be ordered). The jsonl is the input replay_window.py needs to
simulate break-window fills. Touch the flag to go live; rm it to fall back — the next
tick returns no theos and the framework cancels everything resting.

Fills are never exited in set 3; there is no take-profit. The daily loss cap in the
scope is operational (settlements, not intraday marks) and stays a manual kill-flag
decision for v1.
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

_STATE_PATH = os.path.join(TAPES, "_set3_dog_state.json")
_EVENTS_PATH = os.path.join(TAPES, "set3_dog_events.jsonl")
LIVE_FLAG = os.path.join(ROOT, "set3_live.flag")
KILL_FLAG = os.path.join(ROOT, "disable_set3_maker.flag")
SERIES_FLAGS = {"KXITFMATCH": os.path.join(ROOT, "disable_set3_itf.flag"),
                "KXITFWMATCH": os.path.join(ROOT, "disable_set3_itfw.flag")}

SERIES = ("KXITFMATCH", "KXITFWMATCH")
BAND_LO, BAND_HI = 0.30, 0.50          # vig-free dog prob at trigger, [lo, hi)
MAX_DOG_SPREAD_C = 8.0
KNIFE_C = 5.0                          # dog mid drop below trigger mid -> disarm
STALE_S = 20.0                         # tape age fail-closed. 10s until 26SEP11:
                                       # under /markets 429 pressure a SINGLE 5s
                                       # backoff makes an 11-16s gap right at the
                                       # set boundary (= our trigger), which killed
                                       # 3 of the first 4 shadow windows at ~13s.
                                       # 20s survives one backoff; a dead poller
                                       # still cancels 20s in, ~2 min before any
                                       # set-2/3 first point (p5 132s).
ENTRY_WINDOW_S = 30.0                  # gates must all hold within this many seconds
                                       # of the FIRST 1-1 sighting; the cell was
                                       # measured at the 1-1 anchor, so a dog that
                                       # drifts into band later in the break is a
                                       # different (unmeasured) trade — skip it.
MAX_CONCURRENT = 4
TAIL_BYTES = 400_000                   # ~11 min of 2s rows: covers any real set break
HEAD_BYTES = 3_000_000                 # pregame tick lives at the tape head
STATUS_EVERY_S = 5.0

# theos, in cents (see module doc for why each unwanted side is out of bounds)
DOG_THEOS = {"bid_theo": 49.5, "offer_theo": 100.0}
FAV_THEOS = {"bid_theo": 0.0, "offer_theo": 49.5}

# not-currently-being-played statuses; ITF reads "started" while live (kalshi_tennis
# _NOT_LIVE, vendored so this module needs no cross-repo import at quoter runtime)
_NOT_LIVE = {"not_started", "closed", "ended", "cancelled", "canceled",
             "finished", "completed", "postponed", ""}


def _timer_s() -> float:
    try:
        return min(150.0, max(90.0, float(os.environ.get("SET3_TIMER_S", "120"))))
    except ValueError:
        return 120.0


def _c(v) -> Optional[float]:
    """Tape book value (dollars) -> cents, None-safe."""
    return None if v is None else v * 100.0


class Set3DogTheoGenerator(BaseTheoGenerator):
    """Sided long-dog theos for the set-break window; nothing at any other time."""

    def __init__(self, client: Any, configs: List[Any] = None):
        self._events: Dict[str, dict] = self._load_state()
        self._anchor_rows: Dict[str, dict] = {}       # event -> the first-1-1 tape row
        self._tape_cache: Dict[str, tuple] = {}       # event -> (mtime, row)
        self._meta_cache: Dict[str, tuple] = {}       # event -> (me_ticker, opp_ticker, best_of)
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
            log.exception("[SET3] could not persist state — one-shot latch is "
                          "memory-only until this succeeds")

    def _emit_event(self, kind: str, event: str, **kw):
        rec = {"ts": time.time(), "type": kind, "event": event}
        rec.update(kw)
        try:
            with open(_EVENTS_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            log.exception("[SET3] events jsonl write failed")

    # ------------------------------------------------------------- tape access
    def _warn_once(self, key: str, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            log.warning("[SET3] %s", msg)

    def _meta(self, event: str):
        """(me_ticker, opp_ticker, best_of) from poll_tennis's log.json, cached."""
        if event in self._meta_cache:
            return self._meta_cache[event]
        try:
            d = json.load(open(os.path.join(TAPES, event + ".log.json")))
            m = d.get("meta") or {}
            got = (m.get("me_ticker"), m.get("opp_ticker"), d.get("best_of"))
        except Exception:
            return (None, None, None)                  # transient — do not cache
        self._meta_cache[event] = got
        return got

    def _latest(self, event: str) -> Tuple[Optional[dict], Optional[float]]:
        """(most recent tape row, age of the tape file in seconds)."""
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
    def _pregame_vf_me(self, event: str):
        """Pre-match vig-free prob of the 'me' side, or None (mirrors the set1
        model): first tape row at a fully untouched 0-0 with finite vig_free,
        fallback the arm loop's seed_<ev>.json discovery price."""
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
    def _row_consistent_11(row) -> bool:
        """True only for a genuine, settled 1-1 payload (feed reset rows fail this)."""
        st = row.get("state") or {}
        det = row.get("details") or {}
        if (st.get("sets_me"), st.get("sets_opp")) != (1, 1):
            return False
        rw = det.get("round_winners") or []
        if len(rw) != 2:
            return False
        for n in (1, 2):
            rs = det.get(f"competitor{n}_round_scores") or []
            done = [x for x in rs if isinstance(x, dict) and x.get("outcome") != "ongoing"]
            if len(done) != 2:
                return False
        return True

    @staticmethod
    def _set3_untouched(row) -> bool:
        st = row.get("state") or {}
        return (st.get("games_me") == 0 and st.get("games_opp") == 0
                and st.get("points_me") == 0 and st.get("points_opp") == 0)

    def _me_id(self, event: str, rows: List[dict]) -> Optional[str]:
        """Competitor UUID of the 'me' side, from any row where details.server (a UUID)
        and state.server ('me'/'opp') are both present. Exact by construction: poll's
        serving() produced state.server FROM that UUID."""
        for row in reversed(rows):
            st = row.get("state") or {}
            det = row.get("details") or {}
            sv, side = det.get("server"), st.get("server")
            c1, c2 = det.get("competitor1_id"), det.get("competitor2_id")
            if side not in ("me", "opp") or not sv or sv not in (c1, c2):
                continue
            other = c2 if sv == c1 else c1
            return sv if side == "me" else other
        return None

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
        log.warning("[SET3] %s -> DONE (%s)%s", event, reason,
                    " — cancelling both legs" if was_armed else "")

    def _try_trigger(self, event: str, row: dict, age: float,
                     tickers_present: set) -> None:
        """WATCH: decide never / not-yet / ARM, latest tape row in hand.

        Every gate is evaluated AT THE ANCHOR — the first consistent 1-1 row in the
        tape — because that is the snapshot the cell was measured on. The distinction
        is not pedantry: the market reprices during set point, so one poll later the
        set-2 winner can already be the new favourite (LEEOCH 26SEP02: anchor book
        me 52.5/LEE 47.5, next row me 46.5/LEE 52.5). Gating on the latest row would
        silently trade the complement of the measured cell. The latest row's only
        jobs are freshness and confirming set 3 has not started while we decided.
        """
        sets = self._sets(row)
        status = (row.get("status") or "").strip().lower()
        if status in _NOT_LIVE and status:
            return                                     # pre-match or over; rows vanish with the poller
        if None in sets:
            return
        if sets in ((2, 0), (0, 2), (2, 1), (1, 2)):
            self._to_done(event, "missed_window_sets_moved_on", was_armed=False)
            return
        if sets != (1, 1):
            return
        if age is None or age > STALE_S:
            return

        # Anchor to the FIRST genuine 1-1 row in the tape, not to when we looked
        # (a quoter restart mid-break must not restart the clock). Feed reset /
        # mid-transition rows (round_winners not yet 2, sets snapping around) fail
        # _row_consistent_11 and cannot become the anchor.
        e = self._events.setdefault(event, {"state": "watch"})
        # Everything below may read the tape tail, and the quoter ticks far faster
        # than the 2-3s poll cadence — run the heavy path only when the tape
        # actually advanced. (Underscore key: never persisted.)
        mt = (self._tape_cache.get(event) or (None,))[0]
        if e.get("_scan_mtime") == mt:
            return
        e["_scan_mtime"] = mt
        if "anchor_ts" not in e:
            rows = self._tail_rows(event)
            anchor = next((r for r in rows if self._row_consistent_11(r)), None)
            if anchor is None:
                return                                 # transition not visible yet — wait
            if not self._set3_untouched(anchor) or not (anchor.get("state") or {}).get("_points_known", True):
                self._to_done(event, "late_detection_guard", was_armed=False)
                return
            e["anchor_ts"] = anchor.get("ts") or time.time()
            e["me_id"] = self._me_id(event, rows)
            self._save_state()
            self._anchor_rows[event] = anchor

        now = time.time()
        if now - e["anchor_ts"] > ENTRY_WINDOW_S:
            self._to_done(event, "entry_window_expired", was_armed=False)
            return
        if not self._set3_untouched(row):
            self._to_done(event, "set3_started_before_entry", was_armed=False)
            return

        anchor = self._anchor_rows.get(event)
        if anchor is None:                             # restart between sighting and arming
            rows = self._tail_rows(event)
            anchor = next((r for r in rows if self._row_consistent_11(r)), None)
            if anchor is None:
                return
            self._anchor_rows[event] = anchor

        # ---- gates, all on the ANCHOR row -----------------------------------
        me_tick, opp_tick, best_of = self._meta(event)
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

        bk = self._book_c(anchor)
        if any(v is None for v in bk.values()):
            # The anchor row itself has no full book (WS log gap). Fall forward to
            # the first fully-booked row after it — still inside the entry window,
            # still the earliest price the cell could have seen.
            for r in self._tail_rows(event):
                if (r.get("ts") or 0) < e["anchor_ts"] or not self._row_consistent_11(r):
                    continue
                bk2 = self._book_c(r)
                if all(v is not None for v in bk2.values()):
                    bk = bk2
                    break
            if any(v is None for v in bk.values()):
                bk = self._wsbook_c(event)             # quoter's live WS book (BERKSH fix)
            if any(v is None for v in bk.values()):
                return                                 # no priceable book yet; retry within window
        mid_me = (bk["bid_me"] + bk["ask_me"]) / 2.0
        mid_opp = (bk["bid_opp"] + bk["ask_opp"]) / 2.0
        if mid_me + mid_opp <= 0:
            return
        p_me = mid_me / (mid_me + mid_opp)
        dog_is_me = p_me < 0.5
        p_dog = p_me if dog_is_me else 1.0 - p_me
        if not (BAND_LO <= p_dog < BAND_HI):
            self._to_done(event, f"dog_out_of_band_{p_dog:.3f}", was_armed=False)
            return
        dog_spread = (bk["ask_me"] - bk["bid_me"]) if dog_is_me else (bk["ask_opp"] - bk["bid_opp"])
        if dog_spread > MAX_DOG_SPREAD_C:
            self._to_done(event, f"dog_spread_{dog_spread:.1f}c", was_armed=False)
            return

        # PREGAME-DOG GATE (operator direction 26SEP11): the 1-1 anchor dog must
        # ALSO have been the pregame dog. A pregame favourite who slid under 50c
        # by 1-1 is a different (untraded) animal — the price move against them is
        # information, not the measured momentum cell. Fail closed when no pregame
        # reference exists (rare on ITF: 21/21 recent tapes carry the tick).
        pre_me = self._pregame_vf_me(event)
        if pre_me is None:
            self._to_done(event, "no_pregame_reference", was_armed=False)
            return
        pre_dog = pre_me if dog_is_me else 1.0 - pre_me
        if pre_dog >= 0.50:
            self._to_done(event, f"dog_not_pregame_dog_{pre_dog:.3f}", was_armed=False)
            return

        me_id = e.get("me_id") or self._me_id(event, self._tail_rows(event))
        if not me_id:
            self._warn_once(f"meid:{event}", f"{event}: cannot bind competitor UUIDs "
                            f"to me/opp (no server field seen) — not arming")
            return
        e["me_id"] = me_id
        det = anchor.get("details") or {}
        rw = det.get("round_winners") or []
        if rw[1] not in (det.get("competitor1_id"), det.get("competitor2_id")):
            self._to_done(event, "round_winners_unbindable", was_armed=False)
            return
        set2_winner_is_me = rw[1] == me_id
        if set2_winner_is_me != dog_is_me:
            self._to_done(event, "momentum_gate_fav_won_set2", was_armed=False)
            return

        n_armed = sum(1 for v in self._events.values() if v.get("state") == "armed")
        if n_armed >= MAX_CONCURRENT:
            self._to_done(event, "concurrency_cap", was_armed=False)
            return

        series = event.split("-", 1)[0]
        dog_tick = me_tick if dog_is_me else opp_tick
        fav_tick = opp_tick if dog_is_me else me_tick
        dog_mid_c = mid_me if dog_is_me else mid_opp
        e.update({"state": "armed", "trigger_ts": now, "dog": dog_tick, "fav": fav_tick,
                  "dog_is_me": dog_is_me, "trigger_dog_mid_c": round(dog_mid_c, 2),
                  "p_dog": round(p_dog, 4), "pregame_vf": round(pre_dog, 4),
                  "series": series})
        self._save_state()
        self._emit_event("trigger", event, dog=dog_tick, fav=fav_tick,
                         p_dog=round(p_dog, 4), pregame_vf=round(pre_dog, 4),
                         dog_mid_c=round(dog_mid_c, 2),
                         dog_spread_c=round(dog_spread, 2), book=bk,
                         anchor_ts=e["anchor_ts"], live=self._live_armed(),
                         timer_s=_timer_s())
        log.warning("[SET3] %s TRIGGERED %s | dog=%s @ %.1fc (vig-free %.3f, spread %.1fc) "
                    "fav=%s | timer %.0fs | %s", event,
                    "LIVE" if self._live_armed() else "SHADOW",
                    dog_tick.rsplit("-", 1)[-1], dog_mid_c, p_dog, dog_spread,
                    fav_tick.rsplit("-", 1)[-1], _timer_s(),
                    "resting maker bids both legs" if self._live_armed()
                    else "logging only (set3_live.flag absent)")

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
        # A feed reset row (sets snap to 0-0) also lands here: cancelling early on a
        # glitch is the safe direction, and one-shot means we simply stay out.
        if sets != (1, 1):
            return f"sets_left_11_{sets[0]}-{sets[1]}"
        if not self._set3_untouched(row):
            return "set3_activity"
        b, a = self._dog_book_c(e, row)
        if b is None or a is None:
            wb = self._wsbook_c(self._event_key(e))
            b, a = (wb["bid_me"], wb["ask_me"]) if e.get("dog_is_me") else (wb["bid_opp"], wb["ask_opp"])
        if b is not None and a is not None:
            if (b + a) / 2.0 < e["trigger_dog_mid_c"] - KNIFE_C:
                return "falling_knife"
        return None

    @staticmethod
    def _event_key(e: dict) -> str:
        return (e.get("dog") or "").rsplit("-", 1)[0]

    def _dog_book_c(self, e: dict, row: dict) -> Tuple[Optional[float], Optional[float]]:
        bk = self._book_c(row)
        if e.get("dog_is_me"):
            return bk["bid_me"], bk["ask_me"]
        return bk["bid_opp"], bk["ask_opp"]

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
                self._warn_once(f"series:{event}", f"{event}: series {series} is not a "
                                f"set3 cell — config row is misrouted, ignoring")
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
                    out[e["dog"]] = dict(DOG_THEOS)
                    out[e["fav"]] = dict(FAV_THEOS)
                self._tick_log(event, e, row, live)
                continue

            if kill:
                continue                               # watching costs nothing; just never arm
            if row is None:
                continue
            self._try_trigger(event, row, age, present)
            e = self._events.get(event)
            if e and e.get("state") == "armed" and self._live_armed():
                out[e["dog"]] = dict(DOG_THEOS)
                out[e["fav"]] = dict(FAV_THEOS)
        return out

    def _tick_log(self, event: str, e: dict, row: Optional[dict], live: bool):
        """Per-tick shadow/audit row (throttled to tape updates) + 5s status line."""
        mt = (self._tape_cache.get(event) or (None,))[0]
        if e.get("_logged_mtime") != mt and row is not None:
            e["_logged_mtime"] = mt                    # in-memory only; not persisted
            self._emit_event("tick", event, dog=e["dog"], fav=e["fav"], live=live,
                             book=self._book_c(row),
                             armed_for_s=round(time.time() - e["trigger_ts"], 1),
                             theos={e["dog"]: DOG_THEOS, e["fav"]: FAV_THEOS})
        now = time.time()
        if now - self._last_status.get(event, 0.0) >= STATUS_EVERY_S:
            self._last_status[event] = now
            left = _timer_s() - (now - e["trigger_ts"])
            db, da = self._dog_book_c(e, row) if row else (None, None)
            log.info("[SET3] %-18s ARMED %s | %4.0fs left | dog %s book %s/%s (trig %.1fc)",
                     event.rsplit("-", 1)[-1], "LIVE" if live else "SHADOW", left,
                     e["dog"].rsplit("-", 1)[-1],
                     "-" if db is None else f"{db:.0f}",
                     "-" if da is None else f"{da:.0f}", e["trigger_dog_mid_c"])
