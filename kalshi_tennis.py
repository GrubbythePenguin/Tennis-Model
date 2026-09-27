"""Kalshi tennis feed — live game state + vig-free match midpoint. READ-ONLY.

No order endpoints are touched. Nothing here can trade.

RATE BUDGET — READ THIS FIRST
    These GETs share one account bucket with the live esports trading system in
    general_level_based_quoting. A 429 here is a 429 there. Probing to write this
    module already tripped one. The poller is configured aggressively (~2s inside a
    live match) by operator decision 26AUG24 to get point-level resolution; the
    protections are (a) polling ONLY while a tracked match is actually live,
    (b) a hard RPS cap, (c) exponential backoff that surrenders the bucket on 429
    rather than retrying into it, and (d) a kill flag.

ENDPOINTS (all public, no auth)

    1. GET /trade-api/v2/events?series_ticker=KXATPMATCH&status=open&limit=200
         -> today's matches. Series are KXATPMATCH (men) and KXWTAMATCH (women).
         Event ticker shape: KXATPMATCH-<YYMMDD><P1ABBR><P2ABBR>, e.g.
         KXATPMATCH-26AUG24ALTCOM = "Altmaier vs Comesana".

    2. GET /trade-api/v2/milestones?limit=200&related_event_ticker=<ev>
         ^^ `limit` is MANDATORY; omitting it 400s (same as the esports path).
         details: best_of, tour, gender, first/second_competitor_id, round,
                  tournament_name, status
         `best_of` and `tour`/`gender` feed ImpliedModel(best_of=…, split_prior=…)
         directly, so neither is a manual flag any more.
         source_id is `sr:sport_event:…` — Sportradar underneath.

    3. GET /trade-api/v2/live_data/batch?milestone_ids=A&milestone_ids=B
         ^^ REPEATED params. Comma-separated silently returns
            {"live_datas": null} — no error, just nothing.

    4. GET /trade-api/v2/markets?event_ticker=<ev>
         -> BOTH player markets in one call.

PAYLOAD (type=tennis_tournament_singles). Verified against a finished match that
ended 6-2, 6-3:
    competitorN_overall_score       SETS won                        (0 / 2)
    competitorN_round_scores        GAMES per set, one entry per set, each
                                    {"outcome": winner|loser|ongoing, "score": n}
                                    ([{loser,2},{loser,3}] vs [{winner,6},{winner,6}])
    competitorN_current_round_score games in the IN-PROGRESS set (0 once finished)
    server                          competitor id of the server — this is why the
                                    manual set-first-server bookkeeping is gone
    advantage                       populated at deuce/advantage
    round_winners                   competitor id per completed set
    status                          not_started | live | closed | ended | cancelled

    UNVERIFIED: whether an explicit point score (0/15/30/40) is published. Every
    match reachable when this was written was not_started or closed. `advantage`
    exists, which implies point-level tracking, but the field carrying it is unknown.
    points_from_details() returns (0, 0) and flags `points_known=False` until a live
    match settles the question — game-boundary fitting is unaffected either way,
    since the model only fits boundaries anyway.

ORIENTATION — the dangerous part. live_data names players by UUID; markets name
them by string. The link is inferred, and inverting it silently inverts every price
the model produces. resolve_orientation() therefore refuses to guess: it requires
exactly two markets, each binding to exactly one distinct half of the "A vs B"
title, both halves claimed, and it cross-checks the milestone's competitor ids
against live_data's. Anything else raises.
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
TIMEOUT = 12.0
# Series carrying singles matches. The CHALLENGER series is NOT discoverable by
# guessing (KXATPCHALLENGER / KXCHALLENGER / KXATPCH all return 0 events); the real
# ticker is KXATPCHALLENGERMATCH. Do not infer that a tour is absent from a handful
# of negative ticker guesses — enumerate instead.
# Doubles series (26SEP18, operator: extend data collection to doubles — the tau
# hypothesis should hold harder where nobody models the teams). OPT-IN via
# TENNIS_DOUBLES=1 so the arm loop's poller fleet does not silently double: the
# operator gates when that GET load starts. Kalshi carries full live_data for
# doubles milestones (round_scores/points/server, verified 26SEP18 on
# KXITFWDOUBLES-26SEP17ISHSAWTIAMAT) — type "tennis_tournament_doubles".
# Caveats: the deciding "set" is a 10-point MATCH TIEBREAK (last_set_scoring_type
# = "MatchTiebreak"), which tennis_model's final_set_tb=10 only approximates
# (it models a set with a TB at 6-6), and games are no-ad — fine for capture and
# the tau analysis (model-free), wrong for model-theo quoting. Do not quote
# doubles off tennis_branch without fixing both.
DOUBLES_SERIES = {
    "ATPDBL": "KXATPDOUBLES",
    "WTADBL": "KXWTADOUBLES",
    "ATPCHDBL": "KXATPCHALLENGERDOUBLES",
    "ITFDBL": "KXITFDOUBLES",
    "ITFWDBL": "KXITFWDOUBLES",
    "MXDBL": "KXMIXEDDOUBLESMATCH",   # seasonal (Slams/exhibitions); 0 open most weeks
}

SERIES = {
    "ATP": "KXATPMATCH",
    "WTA": "KXWTAMATCH",
    "ATPCH": "KXATPCHALLENGERMATCH",
    "WTACH": "KXWTACHALLENGERMATCH",
    # ITF (26AUG28): same shard 3, same milestone/live_data shape (status, server,
    # round_scores, current_round_score), tour='ITF', gender populated. match_status
    # is EMPTY for ITF where ATP shows '1st_set' - nothing here reads it. Series
    # fee_type is plain 'quadratic': no maker fee, same as the Challengers.
    "ITF": "KXITFMATCH",
    "ITFW": "KXITFWMATCH",
}
if os.environ.get("TENNIS_DOUBLES") == "1":
    SERIES = {**SERIES, **DOUBLES_SERIES}
KILL_FLAG = "disable_tennis_poller.flag"

# status values that mean "not currently being played"
_NOT_LIVE = {"not_started", "closed", "ended", "cancelled", "canceled",
             "finished", "completed", "postponed", ""}


class RateLimited(RuntimeError):
    """The shared GET bucket pushed back. Surrender it; do not retry into it."""


class OrientationError(RuntimeError):
    """Could not bind competitor UUIDs to markets with certainty."""


def disabled() -> bool:
    return os.path.exists(KILL_FLAG)


# ----------------------------------------------------------------- transport

BACKOFF_S = 5.0              # flat sleep after a 429
CEILING_CONSEC_429 = 60      # give up after this many consecutive 429s (~5 min)


class Feed:
    """Rate-limited read-only Kalshi client.

    rps is a HARD ceiling on this process. On 429 the backoff is a flat BACKOFF_S and
    every retry is announced — a quiet 429 would look like a stalled match. The process
    gives up only after CEILING_CONSEC_429 consecutive rejections.
    """

    def __init__(self, rps: float = 2.0, max_backoff: float = 120.0, verbose: bool = True):
        self.min_gap = 1.0 / max(rps, 0.05)
        self.max_backoff = max_backoff
        self.verbose = verbose
        self.s = requests.Session()
        self._last = 0.0
        self._backoff = 0.0
        self.n_get = 0
        self.n_429 = 0
        self._consec_429 = 0
        self._mid_cache: Dict[str, Optional[dict]] = {}

    def _sleep_to_slot(self):
        gap = self.min_gap - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)

    def get(self, path: str, params=None) -> Optional[dict]:
        """GET or None. Raises RateLimited once the backoff ceiling is reached."""
        if disabled():
            raise RuntimeError(f"kill flag present ({KILL_FLAG})")
        if self._backoff:
            if self.verbose:
                print(f"[feed] backing off {self._backoff:.0f}s after 429", file=sys.stderr)
            time.sleep(self._backoff)
        self._sleep_to_slot()
        try:
            r = self.s.get(f"{KALSHI}{path}", params=params, timeout=TIMEOUT)
        except Exception as e:
            self._last = time.monotonic()
            if self.verbose:
                print(f"[feed] {path} error {e!r}", file=sys.stderr)
            return None
        self._last = time.monotonic()
        self.n_get += 1
        if r.status_code == 429:
            self.n_429 += 1
            self._consec_429 += 1
            # FLAT 5s, not doubling to 120s (changed 26AUG27). The doubling assumed this
            # process was the bad actor; with a handful of pollers it is not, and one
            # escalation past 30s made the quoter's tape stale, which cancels every
            # resting order on that match. A 429 now costs one poll, not the point.
            # The ceiling is CONSECUTIVE rejections (~5 min solid), not backoff size.
            self._backoff = BACKOFF_S
            print(f"[feed] *** HTTP 429 on {path} — shared bucket with the live "
                  f"trading system. backoff {self._backoff:.0f}s "
                  f"({self.n_429} total, {self._consec_429} consecutive)", file=sys.stderr)
            if self._consec_429 >= CEILING_CONSEC_429:
                raise RateLimited(f"429 ceiling reached after {self._consec_429} consecutive rejections")
            return None
        self._backoff = 0.0
        self._consec_429 = 0
        if r.status_code != 200:
            if self.verbose:
                print(f"[feed] {path} HTTP {r.status_code}", file=sys.stderr)
            return None
        try:
            return r.json() or {}
        except ValueError:
            return None

    # ------------------------------------------------------------- discovery

    def matches(self, tours=tuple(SERIES), day: Optional[str] = None) -> List[dict]:
        """Open singles-match events. `day` filters the YYMMDD in the ticker."""
        out = []
        for tour in tours:
            body = self.get("/events", {"series_ticker": SERIES[tour],
                                        "status": "open", "limit": 200}) or {}
            for e in body.get("events") or []:
                t = e.get("event_ticker") or ""
                if day:
                    m = re.match(r"^KX[A-Z]+MATCH-(\d{2}[A-Z]{3}\d{2})", t)
                    if not m or m.group(1) != day.upper():
                        continue
                out.append({"event_ticker": t, "title": e.get("title") or "", "tour": tour})
        return out

    def milestone(self, event_ticker: str) -> Optional[dict]:
        """Milestone for an event, cached. Failures are NOT cached."""
        if event_ticker in self._mid_cache:
            return self._mid_cache[event_ticker]
        body = self.get("/milestones", {"limit": 200,
                                        "related_event_ticker": event_ticker})
        if body is None:
            return None                              # transient — do not cache
        ms = body.get("milestones") or []
        # singles / doubles / mixed all carry the same live_data schema; match by
        # prefix so a new tennis_tournament_* subtype cannot silently drop a feed
        ms = [m for m in ms if (m.get("type") or "").startswith("tennis_tournament")]
        got = ms[0] if ms else None
        self._mid_cache[event_ticker] = got
        return got

    def live_data(self, milestone_ids) -> Dict[str, dict]:
        """milestone_id -> details. REPEATED milestone_ids params (see module doc)."""
        ids = [m for m in milestone_ids if m]
        out: Dict[str, dict] = {}
        for i in range(0, len(ids), 40):
            body = self.get("/live_data/batch",
                            [("milestone_ids", m) for m in ids[i:i + 40]])
            if not body:
                continue
            for ld in body.get("live_datas") or []:
                mid = ld.get("milestone_id")
                if mid:
                    out[mid] = ld.get("details") or {}
        return out

    def markets(self, event_ticker: str) -> List[dict]:
        body = self.get("/markets", {"event_ticker": event_ticker, "limit": 20}) or {}
        return body.get("markets") or []


# --------------------------------------------------------------- book parsing
# Vendored from general_level_based_quoting/kalshi_book_read.py rather than
# imported, to keep this repo standalone. The three invariants that module was
# written to stop people re-deriving wrongly:
#   1. v2 field names are *_dollars and the values are DOLLAR STRINGS, not cents.
#   2. yes_bid == 1 - no_ask and yes_ask == 1 - no_bid, so a one-sided book can be
#      resolved from the other side rather than collapsing to None.
#   3. 0.00 / 1.00 are EMPTY-BOOK SENTINELS, not quotes. Letting them through
#      synthesises a meaningless 0.500 mid on a market that has already resolved.

def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def top_of_book(m: dict) -> Tuple[Optional[float], Optional[float]]:
    """(yes_bid, yes_ask) in dollars from a market object; either may be None."""
    yes_bid, yes_ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    if yes_bid is None:
        na = _f(m.get("no_ask_dollars"))
        yes_bid = None if na is None else 1.0 - na
    if yes_ask is None:
        nb = _f(m.get("no_bid_dollars"))
        yes_ask = None if nb is None else 1.0 - nb
    if yes_bid is not None and yes_bid <= 0.0:
        yes_bid = None
    if yes_ask is not None and yes_ask >= 1.0:
        yes_ask = None
    return yes_bid, yes_ask


def mid(m: dict) -> Optional[float]:
    b, a = top_of_book(m)
    if b is None or a is None:
        return None
    return (b + a) / 2.0


def vig_free(mid_me: Optional[float], mid_opp: Optional[float]) -> Optional[float]:
    """Strip the overround using BOTH player markets (proportional method).

    The two markets are independent quotes on the same event, so their mids sum to
    1 + overround. Normalising is strictly better than halving a single market's
    spread: it uses the other side's information instead of discarding it.
    Falls back to the raw mid if only one side is quoted.
    """
    if mid_me is None:
        return None
    if mid_opp is None:
        return mid_me
    s = mid_me + mid_opp
    if s <= 0:
        return None
    return mid_me / s


# ------------------------------------------------------------ state decoding

_warned = set()


def _warn_once(key: str, msg: str):
    if key not in _warned:
        _warned.add(key)
        print(f"[state] {msg}", file=sys.stderr)


def is_live(det: dict) -> bool:
    st = (det.get("status") or "").strip().lower()
    return st not in _NOT_LIVE


def games_in_current_set(det: dict, n: int) -> int:
    """Games won in the in-progress set by competitor n (1 or 2).

    GAMES come from the trailing `ongoing` entry of round_scores. They are NOT in
    current_round_score — that field is the POINT score (see points_from_details).
    """
    rs = det.get(f"competitor{n}_round_scores") or []
    ongoing = [r for r in rs if isinstance(r, dict) and r.get("outcome") == "ongoing"]
    return int(ongoing[-1].get("score") or 0) if ongoing else 0


def serving(det: dict, me_id: str, opp_id: str) -> Optional[bool]:
    """True if `me` serves the current game, False if opponent, None if unknown.

    `server` is a competitor UUID in every sample seen. Other encodings are
    tolerated defensively but flagged, because guessing here inverts the model's
    view of who holds.
    """
    sv = det.get("server")
    if not sv or not isinstance(sv, str):
        return None
    sv = sv.strip()
    if sv == me_id:
        return True
    if sv == opp_id:
        return False
    low = sv.lower()
    if low in ("1", "competitor1", "home"):
        _warn_once("srv_enum", f"server={sv!r} is not a UUID; assuming competitor1")
        return me_id == det.get("competitor1_id")
    if low in ("2", "competitor2", "away"):
        _warn_once("srv_enum", f"server={sv!r} is not a UUID; assuming competitor2")
        return me_id == det.get("competitor2_id")
    _warn_once("srv_unk", f"unrecognised server value {sv!r}; serving unknown")
    return None


# tennis point score -> the model's integer point index.
# 50 is ADVANTAGE. Observed live 26AUG24: at advantage the pair reads (50, 40) or
# (40, 50) — the ad is carried in current_round_score itself, not only in the
# `advantage` field. Mapping 50 -> 4 yields exactly the (4, 3) / (3, 4) the model
# wants. Omitting it silently dropped every advantage point as "unknown".
_PTS = {0: 0, 15: 1, 30: 2, 40: 3, 50: 4}


def points_from_details(det: dict, me_id: str, opp_id: str,
                        in_tiebreak: bool = False) -> Tuple[int, int, bool]:
    """(points_me, points_opp, points_known) for the game in progress.

    THE POINT SCORE IS `competitorN_current_round_score`. Confirmed live 26AUG24 on
    Poljicak vs Schoenhaus: at deuce in game 1 both competitors read 40 while
    round_scores was still {ongoing, 0} — so `current_round_score` is POINTS and
    `round_scores` is GAMES. (An earlier reading of this module had the two
    swapped; games came out right by luck because round_scores was already the
    source, but the point score was thrown away.)

    Encoding:
      normal game : 0 / 15 / 30 / 40, with `advantage` naming the competitor holding
                    the ad. At deuce both read 40 and `advantage` is "".
      tiebreak    : raw point counts (0, 1, 2, ...), so no 15/30/40 mapping applies.

    Returns points_known=False rather than guessing when a value is unrecognised —
    a wrong point score silently corrupts every mid-game price.
    """
    n_me = 1 if det.get("competitor1_id") == me_id else 2
    raw_me = det.get(f"competitor{n_me}_current_round_score")
    raw_opp = det.get(f"competitor{3 - n_me}_current_round_score")
    if not isinstance(raw_me, int) or not isinstance(raw_opp, int):
        return 0, 0, False

    if in_tiebreak:
        return raw_me, raw_opp, True

    if raw_me not in _PTS or raw_opp not in _PTS:
        _warn_once("pts", f"unrecognised point score {raw_me}/{raw_opp} outside a "
                          f"tiebreak; treating the game as 0-0")
        return 0, 0, False

    pm, po = _PTS[raw_me], _PTS[raw_opp]
    # The `advantage` field, when populated, is the authority; it and the 50 code
    # agree in every sample seen, so a disagreement means one of them is stale.
    adv = (det.get("advantage") or "").strip()
    if adv == me_id:
        if (pm, po) != (4, 3):
            _warn_once("adv_conflict", f"advantage says me but scores read "
                                       f"{raw_me}/{raw_opp}; trusting advantage")
        pm, po = 4, 3
    elif adv == opp_id:
        if (pm, po) != (3, 4):
            _warn_once("adv_conflict", f"advantage says opp but scores read "
                                       f"{raw_me}/{raw_opp}; trusting advantage")
        pm, po = 3, 4
    elif adv:
        _warn_once("adv", f"advantage={adv!r} matches neither competitor; ignoring")
    return pm, po, True


def resolve_orientation(milestone: dict, markets: List[dict], title: str) -> Dict[str, str]:
    """{competitor_id: market_ticker}. Raises OrientationError rather than guess.

    Binding is by surname: the event title is "A vs B", milestone details give
    first/second_competitor_id in that order, and each market's yes_sub_title is the
    player's full name. Every step is asserted because a silent inversion here would
    flip every price the model emits and look like a badly calibrated model.
    """
    det = milestone.get("details") or {}
    c1, c2 = det.get("first_competitor_id"), det.get("second_competitor_id")
    if not c1 or not c2 or c1 == c2:
        raise OrientationError(f"milestone competitor ids unusable: {c1!r}, {c2!r}")

    halves = [h.strip() for h in re.split(r"\s+vs\.?\s+", title, flags=re.I)]
    if len(halves) != 2 or not all(halves):
        raise OrientationError(f"cannot split title into two players: {title!r}")

    active = [m for m in markets if m.get("ticker")]
    if len(active) != 2:
        raise OrientationError(f"expected exactly 2 markets, got {len(active)}")

    def binds(mkt, half) -> bool:
        name = (mkt.get("yes_sub_title") or mkt.get("title") or "").lower()
        # surname is the most selective token in the title half
        surname = max(half.lower().split(), key=len)
        return surname in name

    mapping: Dict[str, str] = {}
    for cid, half in ((c1, halves[0]), (c2, halves[1])):
        hits = [m for m in active if binds(m, half)]
        if len(hits) != 1:
            raise OrientationError(
                f"{half!r} matched {len(hits)} of 2 markets "
                f"({[m.get('yes_sub_title') for m in active]}) — refusing to guess")
        mapping[cid] = hits[0]["ticker"]
    if len(set(mapping.values())) != 2:
        raise OrientationError("both competitors bound to the same market")
    return mapping


def model_state(det: dict, me_id: str, opp_id: str) -> Optional[dict]:
    """live_data details -> ImpliedModel.observe(**state) kwargs, from `me`'s side.

    Returns None if the competitor ids are not the ones this payload describes,
    which would otherwise produce a confidently wrong state.
    """
    ids = {det.get("competitor1_id"), det.get("competitor2_id")}
    if me_id not in ids or opp_id not in ids:
        return None
    n_me = 1 if det.get("competitor1_id") == me_id else 2
    n_opp = 3 - n_me
    srv = serving(det, me_id, opp_id)
    gm, go = games_in_current_set(det, n_me), games_in_current_set(det, n_opp)
    pm, po, known = points_from_details(det, me_id, opp_id, in_tiebreak=(gm == 6 and go == 6))
    st = {
        "sets_me": int(det.get(f"competitor{n_me}_overall_score") or 0),
        "sets_opp": int(det.get(f"competitor{n_opp}_overall_score") or 0),
        "games_me": gm,
        "games_opp": go,
        "points_me": pm,
        "points_opp": po,
    }
    if srv is not None:
        st["server"] = "me" if srv else "opp"
    st["_points_known"] = known
    return st


_SLAMS = ("us open", "wimbledon", "roland garros", "french open", "australian open")
_SETS_RE = re.compile(r"\b(\d)\s*-\s*(\d)\b")


def best_of_from_exact_market(feed, event_ticker: str) -> Optional[int]:
    """best_of read off the EXACT-SCORE market — ground truth, not inference.

    The …EXACTMATCH… event enumerates every possible set score, so the largest set
    count settles the format outright: only 2-0/2-1 outcomes means best-of-3, a 3-x
    outcome means best-of-5. Verified 26AUG24 on KXATPEXACTMATCH-26AUG24CECBRO,
    which listed exactly {CEC 2-0, CEC 2-1, BRO 2-0, BRO 2-1} — confirming that the
    "US Open Men Singles / Round Of 128" rows are QUALIFYING (best-of-3) and that
    Kalshi's own best_of="3" was right all along.

    Returns None when the market is absent or unparseable; callers fall back.
    """
    base = event_ticker.split("-", 1)[-1]
    for prefix in ("KXATPEXACTMATCH", "KXWTAEXACTMATCH",
                   "KXATPCHALLENGEREXACTMATCH", "KXWTACHALLENGEREXACTMATCH"):
        body = feed.get("/markets", {"event_ticker": f"{prefix}-{base}", "limit": 40})
        mkts = (body or {}).get("markets") or []
        best = 0
        for m in mkts:
            txt = m.get("yes_sub_title") or m.get("title") or ""
            for a, b in _SETS_RE.findall(txt):
                best = max(best, int(a), int(b))
        if best in (2, 3):
            return 3 if best == 2 else 5
    return None


def resolve_best_of(mil_details: dict, override: Optional[int] = None,
                    verified: Optional[int] = None) -> int:
    """best_of for the model. Refuses to guess — see below.

    Kalshi's `best_of` is NOT trustworthy:
      * absent (None) on all 200 historical milestones sampled 26AUG24;
      * present as "3" on today's board INCLUDING matches whose tournament_name is
        "US Open Men Singles" at Round Of 128, which is best-of-FIVE.
    Defaulting a missing or suspicious value to 3 would misprice an entire bo5 match
    while looking like nothing worse than a badly calibrated model, so this raises
    and makes the operator pass --best-of instead.
    """
    if override in (3, 5):
        return override
    if verified in (3, 5):
        return verified          # read off the exact-score market: ground truth
    raw = mil_details.get("best_of")
    tour = (mil_details.get("tour") or "").upper()
    gender = (mil_details.get("gender") or "").lower()
    tname = (mil_details.get("tournament_name") or "").lower()
    slam_mens = any(s in tname for s in _SLAMS) and gender.startswith("m") and tour == "ATP"

    if raw in (None, ""):
        raise ValueError(
            f"Kalshi reports no best_of for {mil_details.get('tournament_name')!r} — "
            f"it is absent on most milestones. Pass --best-of 3 or --best-of 5.")
    n = int(raw)
    if slam_mens and n == 3:
        raise ValueError(
            f"Kalshi reports best_of=3 for {mil_details.get('tournament_name')!r} "
            f"({mil_details.get('round')}), but men's Grand Slam singles is best-of-5. "
            f"The field is unreliable. Confirm the format and pass --best-of "
            f"explicitly (--best-of 5 for main draw, --best-of 3 for qualifying).")
    if n not in (3, 5):
        raise ValueError(f"unusable best_of={raw!r}; pass --best-of explicitly")
    return n


def resolve_final_set_tb(mil_details: dict, override: Optional[int] = None) -> int:
    """Deciding-set tiebreak length: 10 at the Grand Slams, 7 elsewhere.

    Since the 2022 unified rule all four majors settle a deciding set with a
    10-point breaker at 6-6; the regular tour uses 7 in every set. Kalshi publishes
    no field for this, so it is inferred from tournament_name and must be
    overridable — a wrong value misprices exactly the highest-leverage state a match
    can reach. A 10-point breaker HELPS the favourite (more points, less variance):
    at p=0.65/q=0.40 it is worth 0.5954 against 0.5816 for a 7-point one.
    """
    if override in (7, 10):
        return override
    tname = (mil_details.get("tournament_name") or "").lower()
    return 10 if any(s in tname for s in _SLAMS) else 7


def resolve_split_prior(mil_details: dict, override: Optional[float] = None) -> float:
    """Prior mean for p-q: 0.28 men, 0.14 women. Falls back to 0.20 only out loud."""
    if override is not None:
        return override
    g = (mil_details.get("gender") or "").lower()
    if g.startswith("m"):
        return 0.28
    if g.startswith("w") or g.startswith("f"):
        return 0.14
    _warn_once("gender", f"milestone gender={g!r} unrecognised — using the "
                         f"unsure-prior 0.20; pass --split-prior to override.")
    return 0.20


def _legal_set(a: int, b: int) -> bool:
    """Is (a, b) a legal COMPLETED set score?"""
    hi, lo = max(a, b), min(a, b)
    return (hi == 6 and lo <= 4) or (hi == 7 and lo in (5, 6))


def state_is_consistent(det: dict) -> Optional[str]:
    """None if the payload is self-consistent, else why it isn't.

    Kalshi's set transitions are NOT atomic. Caught live 26AUG24 on Wong/Moller:
    a new set had already begun (round_scores [5,0] / [6,0]) while the previous
    set's final game count was still un-finalised — 6-5, not a legal set — and
    overall_score had not incremented off 0-0. Decoded naively that reads as a
    FRESH MATCH at 0-0, which priced 0.210 against a market of 0.419 and logged a
    +20.9c error that is entirely an artefact of the feed mid-update.

    A boundary built on such a state is worse than a missing one: it enters the fit
    as a real observation at a state that never existed.
    """
    r1 = det.get("competitor1_round_scores") or []
    r2 = det.get("competitor2_round_scores") or []
    if len(r1) != len(r2):
        return f"round_scores length mismatch ({len(r1)} vs {len(r2)})"

    # AT MOST ONE SET MAY BE 'ongoing'. This is the invariant the Wong/Moller
    # transition actually broke: BOTH entries read ongoing — set 1 stuck at 5-6 and
    # set 2 already at 0-0 — for over a minute, not a momentary flicker. Reading the
    # LAST ongoing entry then yields 0-0 games, and with overall_score also still
    # 0-0 the whole payload decodes as a fresh match. The illegal-score and
    # sets-count checks below both pass in that state, because nothing is marked
    # complete for them to inspect.
    ongoing = sum(1 for x in r1 if isinstance(x, dict) and x.get("outcome") == "ongoing")
    if ongoing > 1:
        return (f"{ongoing} sets marked 'ongoing' at once — only one set can be in "
                f"progress; feed is mid-transition")
    done = [(a, b) for a, b in zip(r1, r2)
            if isinstance(a, dict) and a.get("outcome") != "ongoing"]
    for a, b in done:
        if not _legal_set(int(a.get("score") or 0), int(b.get("score") or 0)):
            return f"illegal completed set {a.get('score')}-{b.get('score')}"
    s1 = det.get("competitor1_overall_score")
    s2 = det.get("competitor2_overall_score")
    if isinstance(s1, int) and isinstance(s2, int) and (s1 + s2) != len(done):
        return (f"sets won ({s1}+{s2}) disagrees with completed sets in "
                f"round_scores ({len(done)}) — mid-transition")
    return None


def boundary_key(st: dict) -> Tuple[int, int, int, int]:
    """The tuple whose change means a game finished — i.e. a fittable price."""
    return (st["sets_me"], st["sets_opp"], st["games_me"], st["games_opp"])
