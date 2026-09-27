"""Tennis theo: re-anchor to the market after EVERY point, quote the bracket around it.

Registered as model_name = "tennis_recenter".

THE DIFFERENCE FROM tennis_branch. That model used the fit's own LEVEL as the theo, so
every error in p+q became a directional position: on 26AUG25CHWPAR the fit sat below the
market for 98% of rows after the second set and 73 of 75 fills went the same way, ending
677 contracts long one side.

This model takes NO view on the level. At every point change it sets the anchor to the
current market mid and quotes:

    bid = anchor + delta_lose        offer = anchor + delta_win

The model supplies only the DELTAS - how far one unseen point can move the price - never
the level. That matters because the deltas are the part the fit actually knows: across the
(p,q) that reproduce a given price, the delta varies ~1c while p itself ranges 0.48-0.80.
Taking win-minus-now and lose-minus-now also cancels the level bias exactly, so a lagging
or mis-split fit still yields usable deltas.

Whichever way the unseen point went, the true price is inside the quote, so a counterparty
who already knows the outcome gets fair value rather than picking us off.

THE BRACKET GUARD IS NOT OPTIONAL. delta width depends on p-q, which match prices barely
identify (2.91c of price per 0.01 on p+q, 0.22c on p-q). When the fit's split collapses
the bracket blows out - 7.57c on CHWPAR at p-q=0.039, and 14c/35c on thin mid-attached
fits. A bracket wider than MAX_BRACKET_C means the fit is degenerate, not that the point
is genuinely pivotal, so this quotes nothing rather than posting a garbage-wide market.

STALENESS IS A HARD STOP, as in tennis_branch: no tape inside --max-age means no theo, so
the quoter pulls rather than resting on an anchor nobody is maintaining.

SETWINNER MARKETS (26AUG31). KX*SETWINNER-<ev>-<N>-<PLAYER> tickers are priced here
too, from the SAME match tape: every row already records the fitted (p, q) per variant,
so the set-in-progress price and its one-point branches are two extra evaluations of
tennis_model - no poller change, no new GETs. Same recenter philosophy: the set
market's own devigged book is the anchor, the model supplies only the deltas. Only the
set IN PROGRESS is quoted - a future set's two sides are NOT complements (they sum to
P(the set is played), and how Kalshi resolves an unplayed set is unverified) - and
nothing is EMITTED until tapes/setwinner_quote.flag exists; without the flag every
computed theo and the set book still land in tapes/setwinner_quotes.jsonl, so the edge
is measured before the first order rests.
"""
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

from base_model import BaseTheoGenerator

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAPES = os.path.join(ROOT, "tapes")

# Series this model is allowed to QUOTE. Doubles are deliberately absent: they are
# configured for book capture only (see the QUOTE SCOPE GUARD below).
RECENTER_SERIES = ("KXITFMATCH", "KXITFWMATCH")

# tennis_model is imported for SETWINNER pricing. Cap its lru caches BEFORE the
# import: this is one long-lived process on the box it shares with everything else,
# and the caches are keyed on continuous fitted (p, q) - every refit mints keys that
# are never looked up again (see the tennis_model.py header). 25k = ~56 MB worst
# case, the live-poller setting. An explicit TENNIS_CACHE in the environment wins.
os.environ.setdefault("TENNIS_CACHE", "25000")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tennis_model import set_number_win_prob
DEFAULT_VARIANT = "ewma2"
MAX_TAPE_AGE_SEC = 30.0
# Was 8, inherited from tennis_branch where the LEVEL had to be right. This model uses
# only the one-point deltas, and their quality does not depend on the boundary count
# (26AUG27, 14,596 points, pre-match-attached pollers):
#
#     boundaries   MAE    past +2c   cost/pt   brackets >20c
#     1            1.11    10.9%      0.17c       0.0%
#     3-4          1.28     9.3%      0.16c       0.7%
#     8-11         1.13     9.3%      0.18c       0.3%
#
# A fit seeded from the pre-match price prices the bracket as well as one with eight
# games behind it. MAX_BRACKET_C remains the backstop for a degenerate fit.
MIN_BOUNDARIES = 1
# Was 8.0. Measured 26AUG27 on 9,368 points / 122 matches: break points carry 10-13c
# brackets LEGITIMATELY and are the cheapest class to quote (2.4% overshoot past the
# proportional margin, 0.07c/point) - an 8c gate refused every one of them. 20c still
# catches the degenerate split-collapse fits (those reached 14c at 3-3 deuce on a
# 0.48 market, which is a 3x overstatement, not a real bracket).
MAX_BRACKET_C = 20.0
STATUS_EVERY_SEC = 5.0
MIN_THEO_C, MAX_THEO_C = 2.0, 98.0
# Mirrors kalshi_tennis._NOT_LIVE. Anything else counts as live: ATP/WTA report "live",
# ITF reports "started" (26AUG29 - a literal == "live" check silently refused every
# ITF match while its tape was updating at 3s).
NOT_LIVE_STATUS = {"not_started", "closed", "ended", "cancelled", "canceled", "finished",
                   "completed", "postponed", "",
                   # not in kalshi_tennis._NOT_LIVE (the poller keeps polling these at 3s
                   # so it sees the resumption) but NOT quotable: the last live anchor is
                   # from before the stoppage.
                   "suspended", "interrupted", "delayed", "walkover", "retired"}
# MARGIN. The quoter's CSV min_edge (2c) is the base; this adds to it. Measured 26AUG27
# on the settled post-point price vs the quote edge (one-sided: only the market going
# PAST the quote costs anything):
#
#     margin scheme      overshoot rate   cost/point   by set 1/2/3
#     flat +2c               10.9%          0.23c      0.28 / 0.19 / 0.34
#     flat +3c                5.7%          0.15c      0.18 / 0.12 / 0.23
#     +2c + 0.25 x width      5.1%          0.13c      0.14 / 0.11 / 0.16
#
# Proportional beats flat at the same average margin and removes the set effect, which
# is not a set effect at all: deciders overshoot at the same RATE (17-18% at 35-65c in
# every set) but by more, because their brackets are bigger.
#
# DEFAULT 0 - NO WIDENING. Every cent of margin removes the transient fills (the edge:
# 89% of real CHWPAR fills, +1.5c at one point) along with the overshoots, and only
# the overshoot side is measurable on a 3.4s tape. The trade-off has to be priced on
# live fills, so the model starts at the bracket and the CSV min_edge is the only
# margin. Knobs kept for the A/B.
MARGIN_FRAC = 0.0
# Class extras on top. Server-game-point states (40-30, AD-40, 40-0, 40-15) overshoot
# most under the proportional margin (6.0-6.5%, 0.16-0.17c/point) because the market
# moves on the GAME while the bracket is priced on the point; +1c takes them to ~3.3%.
# Tiebreaks were 17.7% past a flat +2c; the proportional margin alone takes them to
# 4.7%, +1c to 3.2%. Break points need nothing (2.4%) - they are the best points to quote.
EXTRA_SERVER_GAME_POINT_C = 0.0     # measured value if widening is ever wanted: 1.0
# THE ONE WIDENING KEPT (decision 26AUG27). Tiebreak points at the bare +2c min_edge
# overshoot 17.7% of the time for 0.56c/point - three times any other class - and their
# brackets are 12c, so +2c more is still proportionally the tightest quote on the board.
# Measured at +4c total: 9.1% past, 0.30c/point.
EXTRA_TIEBREAK_C = 2.0
# ── SETWINNER (per-set) markets ───────────────────────────────────────────────
# Set prices move MORE per point than match prices - the set is the shorter race -
# so legitimate set brackets run past the match's 20c cap. 25c is a PLACEHOLDER
# until setwinner_quotes.jsonl has enough points to measure the real distribution;
# the first deployment is log-only anyway (flag below), so a wrong cap costs data,
# not money.
MAX_SET_BRACKET_C = 25.0
# Quotes for SETWINNER tickers are emitted ONLY while this file exists. Without it
# the generator still computes every theo and appends it (with the set book) to
# tapes/setwinner_quotes.jsonl. touch to go live, rm to pull.
SET_QUOTE_FLAG = os.path.join(TAPES, "setwinner_quote.flag")


class TennisRecenterTheoGenerator(BaseTheoGenerator):
    """bid/offer in CENTS: market anchor at the last point boundary + the model's deltas."""

    def __init__(self, client: Any, configs: List[Any] = None,
                 variant: str = DEFAULT_VARIANT, max_age: float = MAX_TAPE_AGE_SEC,
                 min_boundaries: int = MIN_BOUNDARIES,
                 max_bracket_c: float = MAX_BRACKET_C):
        self.variant = variant
        self.max_age = max_age
        self.min_boundaries = min_boundaries
        self.max_bracket_c = max_bracket_c
        self._anchor: Dict[str, Any] = {}     # event -> (point_key, anchor_me_c, anchor_opp_c)
        self._skipped: Dict[str, bool] = {}   # event -> the feed skipped a point into this state
        self._last_stale: Dict[str, float] = {}
        self._last_key: Dict[str, Any] = {}      # event -> last point key seen (anchored or not)
        self._anchor_src: Dict[str, str] = {}    # event -> "ws" | "tape"
        self._last_wslog: Dict[str, float] = {}  # event -> last WS book log write
        # One record per anchored point, so every fill in trades.csv can be joined to
        # the state, bracket, anchor and margin that produced the quote it hit. This is
        # the "store all the states" half of the 26AUG27 plan; fill_markouts.py reads it.
        self._qlog_path = os.path.join(TAPES, "recenter_quotes.jsonl")
        self._qlog_key: Dict[str, Any] = {}
        self._last_status: Dict[str, float] = {}
        self._cache: Dict[str, Any] = {}
        # SETWINNER: (best_of, final_set_tb) per match event, and the per-set-event
        # dedupe key for tapes/setwinner_quotes.jsonl (one line per point per gate).
        self._bo_cache: Dict[str, Any] = {}
        self._setlog_key: Dict[str, Any] = {}
        self._setlog_path = os.path.join(TAPES, "setwinner_quotes.jsonl")
        super().__init__(client, configs)

    # ------------------------------------------------------------------ tape access
    def _meta(self, event: str):
        try:
            d = json.load(open(os.path.join(TAPES, event + ".log.json")))
            m = d.get("meta") or {}
            return m.get("me_ticker"), m.get("opp_ticker"), len(d.get("obs") or [])
        except Exception:
            return None, None, 0

    def _match_meta(self, event: str):
        """(best_of, final_set_tb) from the poller's log.json - static per match."""
        c = self._bo_cache.get(event)
        if c:
            return c
        try:
            d = json.load(open(os.path.join(TAPES, event + ".log.json")))
            got = (int(d.get("best_of") or 3), int(d.get("final_set_tb") or 7))
        except Exception:
            got = (3, 7)      # right for everything except a slam bo5 decider at 6-6
        self._bo_cache[event] = got
        return got

    @staticmethod
    def _parse_set_ticker(ticker: str):
        """KXATPSETWINNER-26AUG31ABCDEF-2-ABC -> ("KXATPMATCH-26AUG31ABCDEF", 2).
        None for anything that is not a per-set winner ticker."""
        parts = (ticker or "").split("-")
        if len(parts) == 4 and "SETWINNER" in parts[0]:
            try:
                return parts[0].replace("SETWINNER", "MATCH") + "-" + parts[1], int(parts[2])
            except ValueError:
                return None
        return None

    def _latest(self, event: str, need_ahead: bool = True):
        p = os.path.join(TAPES, event + ".jsonl")
        try:
            stt = os.stat(p)
        except OSError:
            return None
        if time.time() - stt.st_mtime > self.max_age:
            # Once per event per STATUS_EVERY_SEC, not per ticker per 100ms cycle:
            # 26AUG27 this line ran at ~20/s per stale event and buried everything.
            now = time.time()
            if now - self._last_stale.get(event, 0.0) >= STATUS_EVERY_SEC:
                self._last_stale[event] = now
                log.warning("RECENTER | %s tape stale (%.0fs > %.0fs) — no theo; the poller is "
                            "down or rate-limited", event, now - stt.st_mtime, self.max_age)
            return None
        # NB the write below keys on (event, need_ahead); reading bare `event` meant
        # the cache NEVER hit and the 300KB tail was re-read and re-parsed every
        # ticker every 100ms cycle. Found 26AUG31 while adding SETWINNER support.
        c = self._cache.get((event, need_ahead))
        if c and c[0] == stt.st_mtime:
            return c[1]
        row = None
        try:
            with open(p, "rb") as f:
                f.seek(max(0, stt.st_size - 300_000))
                tail = f.read().decode("utf-8", "replace").splitlines()
            newest = None
            for line in reversed(tail):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if newest is None:
                    newest = r
                    # INTERRUPTED / SUSPENDED / ENDED: the poller keeps writing rows, so
                    # the mtime stays fresh, but the last LIVE row - and its anchor - is
                    # from before the stoppage. Quoting it means resting on a price from
                    # before a rain delay (26AUG27: US Open qualifying, three matches
                    # interrupted at once). No theo until the feed says live again.
                    if (newest.get("status") or "").strip().lower() in NOT_LIVE_STATUS or not newest.get("status"):
                        now = time.time()
                        if now - self._last_stale.get(event + ":nl", 0.0) >= STATUS_EVERY_SEC * 6:
                            self._last_stale[event + ":nl"] = now
                            log.info("RECENTER | %s status=%s — no theo until live", event, newest.get("status"))
                        self._cache[(event, need_ahead)] = (stt.st_mtime, None)
                        return None
                if (r.get("status") or "").strip().lower() not in NOT_LIVE_STATUS and r.get("status") and (
                        not need_ahead or (r.get("ahead") or {}).get("win")):
                    row = r
                    break
        except Exception as e:
            log.warning("RECENTER | %s tape unreadable: %s", event, e)
            return None
        self._cache[(event, need_ahead)] = (stt.st_mtime, row)
        return row

    # ------------------------------------------------------------------ anchoring
    @staticmethod
    def _point_key(s):
        return (s.get("sets_me"), s.get("sets_opp"), s.get("games_me"),
                s.get("games_opp"), s.get("points_me"), s.get("points_opp"))

    @staticmethod
    def _point_class(s):
        """(label, extra margin in cents) for the point about to be played."""
        pm, po = s.get("points_me"), s.get("points_opp")
        if s.get("games_me") == 6 and s.get("games_opp") == 6:
            return "tiebreak", EXTRA_TIEBREAK_C
        if pm is None or po is None:
            return "unknown", 0.0
        srv = s.get("server")
        ps, pr = (pm, po) if srv == "me" else (po, pm)
        if pr >= 3 and pr > ps:
            return "break point", 0.0
        if ps >= 3 and ps > pr:
            return "server game point", EXTRA_SERVER_GAME_POINT_C
        return "deuce" if (ps == pr and ps >= 2) else "early", 0.0

    @staticmethod
    def _legal_step(prev, key):
        """Was prev -> key ONE point? 66 of 1,130 deuce transitions in the tapes ended
        the game directly, which a single point cannot do: the feed skipped a point, the
        market moved two points' worth, and 65% of those went past a +2c quote (1.95c/
        point). A bracket priced for one point must not be quoted across two."""
        if prev is None or None in prev or None in key:
            return True
        same_game = prev[:4] == key[:4]
        if same_game:
            tb = key[2] == 6 and key[3] == 6
            d = (key[4] + key[5]) - (prev[4] + prev[5])
            if tb:
                return d == 1
            # deuce/advantage is encoded 3-3 -> 4-3 -> 3-3: totals move by +1 or -1
            return d == 1 or (d == -1 and max(prev[4], prev[5]) == 4)
        # new game (or set): the previous state must have been a game point for someone
        tb_prev = prev[2] == 6 and prev[3] == 6
        hi, lo = max(prev[4], prev[5]), min(prev[4], prev[5])
        return (hi >= 6 and hi - lo >= 1) if tb_prev else (hi >= 3 and hi - lo >= 1)

    def _anchors(self, event, row, tops=None):
        """(anchor_me, anchor_opp) in cents, re-set whenever the POINT score changes.

        Held constant between points on purpose: re-anchoring on every poll would chase
        the market tick by tick and the quote would never be crossed.

        ANCHOR SOURCE (26AUG28): the quoter's own WebSocket top-of-book (`tops` =
        (bid_me, ask_me, bid_opp, ask_opp) in cents, from dt_market_state) when all
        four sides are present; the tape's vig-free mid only as fallback. The WS book
        is sub-second where the tape is a 3.4s poll, and it costs no GET - the poller's
        /markets call was 1 of its 2 GETs per cycle and existed for this anchor. Both
        sources include our own resting orders when we are the touch; the two-book
        devig damps that, and it is the same bias either way.
        """
        s, bk = row.get("state") or {}, row.get("book") or {}
        key = self._point_key(s)
        prev = self._anchor.get(event)
        if prev and prev[0] == key:
            return prev[1], prev[2], False
        # Legality is judged against the last point SEEN, not the last point ANCHORED.
        # A point whose transition row carries no vig-free price (rate-limited poller)
        # never gets an anchor, so comparing against the anchor key spanned two points
        # and flagged legal transitions as skips (26AUG27: GAUWEN 15-15 -> 30-15).
        seen = self._last_key.get(event)
        if seen != key:
            self._skipped[event] = not self._legal_step(seen, key)
            self._last_key[event] = key
        # ONE anchor, not two. Anchoring each ticker on its own book let the overround
        # push them apart: bid_me + bid_opp reached 100.8 on 1.5% of rows, and both of
        # those filling pays 100.8 for something worth exactly 100 — a locked loss we
        # inflict on ourselves by quoting both sides. The devigged price sums to 100 by
        # construction and uses BOTH books' information rather than discarding one.
        vf = row.get("vig_free")
        src = "tape"
        # TAPE BOOK FIRST when the poller carries one (26AUG29). Kalshi's orderbook_delta
        # stream delivers NOTHING for ITF markets - the WS book refreshed once per
        # 5-minute snapshot (CASCHE sat at 51/52 while the market was 24/25) - so ITF
        # pollers run --markets-every 1 and their tape is the fresh source. ATP/WTA
        # pollers carry no book of their own (it is copied from this WS log), so for
        # them this branch is the WS book by another name.
        tape_ok = (bk.get("bid_me") is not None and bk.get("ask_me") is not None
                   and time.time() - float(row.get("ts") or 0) < 8.0)
        if tape_ok and vf is not None and 0.0 < vf < 1.0:
            a_me = float(vf) * 100.0
            src = "tape"
        elif tops and all(t is not None and 0 < t < 100 for t in tops) and tops[0] < tops[1] and tops[2] < tops[3]:
            mid_me, mid_opp = (tops[0] + tops[1]) / 2.0, (tops[2] + tops[3]) / 2.0
            a_me = mid_me / (mid_me + mid_opp) * 100.0
            src = "ws"
        elif vf is not None and 0.0 < vf < 1.0:
            a_me = float(vf) * 100.0
        else:
            bm, am = bk.get("bid_me"), bk.get("ask_me")
            if None in (bm, am):
                # NEW POINT, NO PRICE: quote nothing until one arrives. This used to
                # hand back the previous point's anchor, i.e. the new point's deltas
                # around a price one point old — 26AUG27 GAUWEN 15-0 -> 30-0 kept
                # anchor 39.5 through both. Under rate limiting that stale window can
                # be the whole point, and a stale anchor is exactly the level risk
                # this model exists to avoid. The first priced row of the point
                # anchors it.
                return None, None, False
            a_me = (bm + am) / 2.0 * 100.0
        a_opp = 100.0 - a_me
        self._anchor[event] = (key, a_me, a_opp)
        self._anchor_src[event] = src
        return a_me, a_opp, True

    def _status(self, event, row, a_me, dlo, dhi, n_obs, gated):
        now = time.time()
        if now - self._last_status.get(event, 0.0) < STATUS_EVERY_SEC:
            return
        self._last_status[event] = now
        s, bk = row.get("state") or {}, row.get("book") or {}
        lbl = ["0", "15", "30", "40", "AD"]
        pm, po = s.get("points_me"), s.get("points_opp")
        try:
            tb = s.get("games_me") == 6 and s.get("games_opp") == 6
            pts = f"{pm}-{po}" if tb else f"{lbl[pm]}-{lbl[po]}"
        except Exception:
            pts = f"{pm}-{po}"
        log.info("[RECENTER] %-14s %s-%s %s-%s %-7s srv=%-4s | n=%-3s %-28s | "
                 "anchor %5.1f  d %+5.2f/%+5.2f  brkt %5.2fc | quote %5.1f/%5.1f "
                 "mkt %s/%s",
                 event.rsplit("-", 1)[-1], s.get("sets_me"), s.get("sets_opp"),
                 s.get("games_me"), s.get("games_opp"), pts, s.get("server", "?"),
                 n_obs, gated, a_me, dlo, dhi, dhi - dlo,
                 a_me + dlo, a_me + dhi, bk.get("bid_me"), bk.get("ask_me"))

    # ------------------------------------------------------------------ generation
    def _batch_generate(self, tickers: List[str],
                        dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for ticker in tickers:
            if not self.configs.get(ticker):
                continue
            if ticker in out:
                continue          # its pair was emitted when the other side was processed
            parsed = self._parse_set_ticker(ticker)
            if parsed is not None:
                self._gen_set(ticker, parsed, out, dt_market_state)
                continue
            event = ticker.rsplit("-", 1)[0]
            # WS BOOK LOG FIRST, before anything that needs the poller's branches. The
            # poller's price now comes from this log (no /markets GET), and its branches
            # need that price - so logging after the branch check deadlocks: 26AUG29
            # Taipei sat at n=0 for 15 minutes with the quoter waiting on the poller and
            # the poller waiting on the quoter.
            row_any = self._latest(event, need_ahead=False)
            if row_any is not None:
                me0, opp0, _n0 = self._meta(event)
                if me0 and opp0:
                    tb0, to0 = dt_market_state.get("__top_bids__") or {}, dt_market_state.get("__top_offers__") or {}
                    self._log_wsbook(event, me0, opp0, (tb0.get(me0), to0.get(me0), tb0.get(opp0), to0.get(opp0)))
            # ── QUOTE SCOPE GUARD (26SEP27) ──────────────────────────────────
            # This model has no series gate of its own and quotes whatever is in
            # self.configs. DOUBLES rows exist in the config ONLY so their tickers
            # enter active_tickers and get a live ws_delta book — the poller reads
            # its price from the wsbook written ABOVE, which is why the guard sits
            # here and not at the top of the loop. Nothing past this line may run
            # for a series this model is not meant to trade.
            if event.split("-", 1)[0] not in RECENTER_SERIES:
                continue
            row = self._latest(event)
            if row is None:
                continue
            ah = row.get("ahead") or {}
            w = (ah.get("win") or {}).get(self.variant)
            l = (ah.get("lose") or {}).get(self.variant)
            nowp = (ah.get("now") or {}).get(self.variant)
            if None in (w, l, nowp):
                continue
            me_tick, opp_tick, n_obs = self._meta(event)
            if not me_tick or not opp_tick:
                continue

            # DELTAS, never levels — this is the whole point of the model.
            dlo = (min(w, l) - nowp) * 100.0
            dhi = (max(w, l) - nowp) * 100.0
            tb, to = dt_market_state.get("__top_bids__") or {}, dt_market_state.get("__top_offers__") or {}
            tops = (tb.get(me_tick), to.get(me_tick), tb.get(opp_tick), to.get(opp_tick))
            self._log_wsbook(event, me_tick, opp_tick, tops)
            a_me, a_opp, fresh = self._anchors(event, row, tops)
            if a_me is None:
                continue

            cls, extra = self._point_class(row.get("state") or {})
            margin = MARGIN_FRAC * (dhi - dlo) + extra
            gate = "PRICING"
            if n_obs < self.min_boundaries:
                gate = f"gated {n_obs}/{self.min_boundaries}"
            elif (dhi - dlo) > self.max_bracket_c:
                gate = f"wide {dhi-dlo:.1f}c"
            elif self._skipped.get(event):
                gate = "skipped pt"
            self._status(event, row, a_me, dlo, dhi, n_obs, f"{gate} {cls} +{margin:.1f}c")
            if gate != "PRICING":
                continue
            dlo, dhi = dlo - margin, dhi + margin       # widen symmetrically

            # BOTH SIDES OR NEITHER. The two tickers are the same bet, so quoting one
            # and not the other IS a directional position. Before this was atomic the
            # MIN_THEO_C floor clipped only the cheap side near 97/98 and left us quoting
            # the favourite alone - the exact asymmetry that built a 677-contract one-way
            # book on 26AUG25CHWPAR. Compute the pair, emit it only if both pass.
            pair = {me_tick: (a_me + dlo, a_me + dhi),
                    # opponent price is the complement, so the deltas flip and reorder
                    opp_tick: (a_opp - dhi, a_opp - dlo)}
            bad = None
            for tk, (b, o) in pair.items():
                if o < b:
                    bad = f"{tk} inverted {b:.2f}/{o:.2f}"
                elif b < MIN_THEO_C or o > MAX_THEO_C:
                    bad = f"{tk} outside [{MIN_THEO_C},{MAX_THEO_C}] ({b:.2f}/{o:.2f})"
            if bad:
                log.debug("RECENTER | %s no quote: %s", event, bad)
                continue
            for tk, (b, o) in pair.items():
                out[tk] = {"bid_theo": round(b, 2), "offer_theo": round(o, 2)}
            self._log_quote(event, row, me_tick, opp_tick, a_me, dlo, dhi, margin, cls, pair)
        return out

    # ------------------------------------------------------------------ SETWINNER
    def _gen_set(self, ticker, parsed, out, dt_market_state):
        """Theo for one KX*SETWINNER-<ev>-<N>-<PLAYER> ticker (and its pair), priced
        from the MATCH tape's fitted (p, q). Emits into `out` only when
        SET_QUOTE_FLAG is up and every gate passes; always logs what it computed."""
        match_event, set_no = parsed
        set_event = ticker.rsplit("-", 1)[0]
        row = self._latest(match_event)       # same staleness / live / branch gates
        if row is None:
            return
        me_tick, opp_tick, _n = self._meta(match_event)
        if not me_tick or not opp_tick:
            return
        sme = set_event + "-" + me_tick.rsplit("-", 1)[1]
        sopp = set_event + "-" + opp_tick.rsplit("-", 1)[1]
        # BOTH SIDES OR NEITHER, and both must be configured: the anchor is the
        # devig of the pair's books, and a one-sided quote is a position.
        if not (self.configs.get(sme) and self.configs.get(sopp)):
            return
        fit = (row.get("fit") or {}).get(self.variant) or {}
        s = row.get("state") or {}
        p, q, nfit = fit.get("p"), fit.get("q"), int(fit.get("n") or 0)
        srv = s.get("server")
        if p is None or q is None or srv not in ("me", "opp"):
            return
        sm, so = int(s.get("sets_me") or 0), int(s.get("sets_opp") or 0)
        if set_no != sm + so + 1:
            return     # future set: sides are not complements; past set: settled
        best_of, final_tb = self._match_meta(match_event)
        st = dict(games_me=int(s.get("games_me") or 0), games_opp=int(s.get("games_opp") or 0),
                  i_serve=(srv == "me"), points_me=int(s.get("points_me") or 0),
                  points_opp=int(s.get("points_opp") or 0))
        try:
            nowp = set_number_win_prob(p, q, set_no, sm, so, best_of, final_tb, **st)
            w = set_number_win_prob(p, q, set_no, sm, so, best_of, final_tb,
                                    **{**st, "points_me": st["points_me"] + 1})
            l = set_number_win_prob(p, q, set_no, sm, so, best_of, final_tb,
                                    **{**st, "points_opp": st["points_opp"] + 1})
        except Exception as e:
            log.debug("SET | %s model error: %s", set_event, e)
            return
        # DELTAS, never levels - same contract as the match path.
        dlo, dhi = (min(w, l) - nowp) * 100.0, (max(w, l) - nowp) * 100.0

        # The skip detector shares the match path's keys: same tape, same points, so
        # whichever path sees the new point first records the verdict for both.
        key = self._point_key(s)
        seen = self._last_key.get(match_event)
        if seen != key:
            self._skipped[match_event] = not self._legal_step(seen, key)
            self._last_key[match_event] = key

        # ANCHOR: the SET market's own WS book, devigged across the pair, held per
        # point exactly like the match anchor. The tape carries no set-market book,
        # so the WS tops are the only source - no two-sided book, no theo.
        tb_ = dt_market_state.get("__top_bids__") or {}
        to_ = dt_market_state.get("__top_offers__") or {}
        tops = (tb_.get(sme), to_.get(sme), tb_.get(sopp), to_.get(sopp))
        prev = self._anchor.get(set_event)
        if prev and prev[0] == key:
            a_me, a_opp = prev[1], prev[2]
        elif (all(t is not None and 0 < t < 100 for t in tops)
              and tops[0] < tops[1] and tops[2] < tops[3]):
            mid_me, mid_opp = (tops[0] + tops[1]) / 2.0, (tops[2] + tops[3]) / 2.0
            a_me = mid_me / (mid_me + mid_opp) * 100.0
            a_opp = 100.0 - a_me
            self._anchor[set_event] = (key, a_me, a_opp)
        else:
            return     # new point, no priced set book yet: quote nothing until one arrives

        cls, extra = self._point_class(s)
        margin = MARGIN_FRAC * (dhi - dlo) + extra
        gate = "PRICING"
        if nfit < self.min_boundaries:
            gate = f"gated {nfit}/{self.min_boundaries}"
        elif (dhi - dlo) > MAX_SET_BRACKET_C:
            gate = f"wide {dhi - dlo:.1f}c"
        elif self._skipped.get(match_event):
            gate = "skipped pt"
        elif not os.path.exists(SET_QUOTE_FLAG):
            gate = "log-only"
        dql, dqh = dlo - margin, dhi + margin
        pair = {sme: (a_me + dql, a_me + dqh),
                sopp: (a_opp - dqh, a_opp - dql)}
        if gate == "PRICING":
            for tk, (b, o) in pair.items():
                if o < b:
                    gate = f"refused {tk} inverted {b:.2f}/{o:.2f}"
                elif b < MIN_THEO_C or o > MAX_THEO_C:
                    gate = f"refused {tk} outside [{MIN_THEO_C},{MAX_THEO_C}] ({b:.2f}/{o:.2f})"
        now = time.time()
        if now - self._last_status.get(set_event, 0.0) >= STATUS_EVERY_SEC:
            self._last_status[set_event] = now
            log.info("[SET%d] %-14s %s-%s %s-%s %s-%s srv=%-4s | p=%.3f q=%.3f n=%-3d %-16s "
                     "| anchor %5.1f  d %+5.2f/%+5.2f  brkt %5.2fc | %5.1f/%5.1f",
                     set_no, set_event.split("-")[1], sm, so, st["games_me"], st["games_opp"],
                     st["points_me"], st["points_opp"], srv, p, q, nfit, gate,
                     a_me, dlo, dhi, dhi - dlo, a_me + dql, a_me + dqh)
        self._log_set(set_event, set_no, row, s, p, q, nfit, nowp, w, l, a_me,
                      dlo, dhi, margin, cls, tops, gate, pair)
        if gate != "PRICING":
            return
        for tk, (b, o) in pair.items():
            out[tk] = {"bid_theo": round(b, 2), "offer_theo": round(o, 2)}

    def _log_set(self, set_event, set_no, row, s, p, q, nfit, nowp, w, l, a_me,
                 dlo, dhi, margin, cls, tops, gate, pair):
        """One line per anchored point per set event (re-logged if the gate changes),
        quoting or not - this is the capture that prices the edge before any order."""
        key = (self._point_key(s), gate.split(" ")[0])
        if self._setlog_key.get(set_event) == key:
            return
        self._setlog_key[set_event] = key
        rec = {"ts": round(time.time(), 3), "tape_ts": row.get("ts"), "event": set_event,
               "set_no": set_no, "state": {k: v for k, v in s.items() if not k.startswith("_")},
               "class": cls, "variant": self.variant, "fit": {"p": p, "q": q, "n": nfit},
               "set_now": round(nowp, 4), "set_win": round(w, 4), "set_lose": round(l, 4),
               "anchor_me": round(a_me, 3), "dlo": round(dlo, 3), "dhi": round(dhi, 3),
               "margin": round(margin, 3), "book": list(tops), "gate": gate,
               "quote": ({tk: [round(b, 2), round(o, 2)] for tk, (b, o) in pair.items()}
                         if gate == "PRICING" else None),
               "match_vig_free": row.get("vig_free")}
        try:
            with open(self._setlog_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            log.warning("SET | quote log write failed: %s", e)

    def _log_wsbook(self, event, me_tick, opp_tick, tops):
        """1 Hz record of the quoter's WebSocket top-of-book, per event, no GETs.
        Replaces the poller's /markets call as the offline price record (markouts,
        anchor audit); sub-second where the tape was a 3.4s poll."""
        now = time.time()
        if now - self._last_wslog.get(event, 0.0) < 1.0 or not any(t is not None for t in tops):
            return
        self._last_wslog[event] = now
        try:
            with open(os.path.join(TAPES, f"wsbook_{event}.jsonl"), "a") as f:
                f.write(json.dumps({"ts": round(now, 3), "me": [tops[0], tops[1]], "opp": [tops[2], tops[3]]}) + "\n")
        except Exception as e:
            log.warning("RECENTER | wsbook log write failed: %s", e)

    def _log_quote(self, event, row, me_tick, opp_tick, a_me, dlo, dhi, margin, cls, pair):
        """Append one line per anchored point (not per poll)."""
        s = row.get("state") or {}
        key = self._point_key(s)
        if self._qlog_key.get(event) == key:
            return
        self._qlog_key[event] = key
        bk = row.get("book") or {}
        rec = {"ts": time.time(), "tape_ts": row.get("ts"), "event": event, "me": me_tick, "opp": opp_tick,
               "state": {k: v for k, v in s.items() if not k.startswith("_")}, "class": cls,
               "anchor_me": round(a_me, 3), "dlo": round(dlo + margin, 3), "dhi": round(dhi - margin, 3),
               "margin": round(margin, 3), "bracket": round(dhi - dlo - 2 * margin, 3),
               "quote": {tk: [round(b, 2), round(o, 2)] for tk, (b, o) in pair.items()},
               "book": bk, "vig_free": row.get("vig_free"), "variant": self.variant,
               "anchor_src": self._anchor_src.get(event, "?")}
        try:
            with open(self._qlog_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            log.warning("RECENTER | quote log write failed: %s", e)
