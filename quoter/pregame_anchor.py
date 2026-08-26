"""Rolling pregame probability anchor — re-solve the continuation at the TRUE
game start instead of freezing whatever populate captured two hours earlier.

THE PROBLEM
-----------
populate_configs writes p1/p2/series once, when the event first appears, and the
capture loop is gated on `sm_ticker not in existing_probs` — so a row is NEVER
recomputed. Measured over 241 LoL event-legs (26JUL26-26AUG10):

    real game-1 start minus populate anchor time : median 127 min (p90 183)
    real start minus the ticker's SCHEDULED time : median +10 min, p90 +70,
                                                   only 8% within +/-5 min
    |p3 re-solved at kickoff - p3 frozen|        : median 0.013, 34% > 0.02,
                                                   10% > 0.05, max 0.119

`hedge_engine.solve_game_3_probability` derives G3 as (p1+p2)/2 and deliberately
discards the series price, so a stale anchor becomes a standing one-sided basis
that nothing closes. DRX/DNS 26AUG10: p3 frozen at 0.612 while the market's
back-out ran 0.534 — 56,743 lots bought vs 6,515 sold, -$29,605.

THE FIX
-------
Every populate pass, for events that have NOT started, re-solve the continuation
off the live books and append it to a rolling cache. When `riot_game_state`
reports game 1 has started, take the median of the cached solves in the window
ending at the TRUE start instant and rewrite the row — but only if it moved more
than ADOPT_THRESHOLD, because re-anchoring a book that did not move just injects
sampling noise.

BACKTEST (241 event-legs, one-sidedness = max(buy%,sell%)/min at a 3c gate)

    game 1        median one-sided   >=10:1   >=50:1
    production           7.4           45%      22%
    re-anchor always     5.5           39%      12%
    |dp3| > 0.010        4.6           34%      10%     <- shipped rule

    by how far the market actually moved:
       |dp3| < 0.01  (n=94)   4.4 ->   5.5   (slightly WORSE - hence the gate)
       0.01 - 0.02   (n=65)   8.0 ->   3.8
       0.02 - 0.04   (n=48)  39.2 ->  10.2
       >= 0.04       (n=34) 118.5 ->  16.9

NOT HANDLED HERE: the game-1/game-2 intermission re-anchor, which is the larger
win in game 2 (buy/sell 13.2%/1.5% -> 4.4%/5.6%, one-sidedness 7.2 -> 4.5). That
is a mid-match update and needs either an explicit p3 column or a live path;
this module is populate-time only.

Never raises into the caller. Every failure path logs and returns a no-op.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple



import prob_reconcile
import riot_game_state as rgs

_R = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_R, "_pregame_anchor_cache.json")

# Window ending at the true start over which cached solves are medianed. The
# backtest showed the solved p3 is flat for any window >= 8 min (it saturates on
# the number of solvable snapshots), so this is deliberately wider than 10 to
# survive populate's ~2-3 min cadence.
WINDOW_MIN = 15
MIN_SNAPSHOTS = 2
# Adopt only if the continuation moved more than this. Tuned on the backtest:
# 0.010 minimises median one-sidedness (4.6) and halves the >=50:1 tail (22%->10%).
ADOPT_THRESHOLD = 0.010

# ── intermission (G1 winner bias) ────────────────────────────────────────────
# Between game 1 ending and game 2 starting, the market prices the series with
# map 1 decided, so P(win G3) backs straight out of the two live books:
#       p3 = (series_mid - m2_mid) / (1 - m2_mid)
# Our frozen (p1+p2)/2 is HIGHER than that market back-out on 81% of events
# (median +0.034) — i.e. we systematically overprice whoever just went 1-0.
# Measured on the leading leg (state 1), 115 events:
#       production          buy 13.2% / sell 1.6%   61% systematically long
#       |err| > 0.010       buy  4.4% / sell 6.7%   21% systematically long
# Higher thresholds are strictly worse (0.03 -> 28%, 0.05 -> 36%), so this is
# deliberately the same low threshold as the pregame anchor.
INTERMISSION_THRESHOLD = 0.010
# Max tolerated error-transfer coefficient on the intermission back-out: reject
# when a 1c error in `series` would move p3 by more than this many cents. See
# the conditioning guard in run_pass for the derivation and the 26AUG11 data.
INTERMISSION_MAX_AMPLIFICATION = 3.0
# Max p3 uncertainty (cents) we tolerate from the SERIES book's own width,
# after amplification. 26AUG20 HLEDK: a ~20c-wide post-halt series book at
# m2~0.50 (2x) implied +-20pts of p3 — the snapshot should never have been
# taken. 5c keeps a tier-1 1-3c book comfortably, admits a 5c tier-2 book at
# low amplification, and rejects anything reopening wide.
MAX_SNAPSHOT_P3_UNCERTAINTY_C = 5.0
# The map-2 book is the DIVISOR; a wide one corrupts the back-out regardless
# of the series book. Bounded separately and absolutely.
MAX_SNAPSHOT_MAP_WIDTH_C = 8.0
# Tier-2 intermission write cap (operator, 26AUG12). Tier-2 intermissions are
# enabled via enable_t2_intermission.flag — the K-guard handles conditioning —
# but an illiquid tier-2 book can still imply an enormous correction off an
# intermittent signal (CKZMEA wrote 21pts at the K boundary the same night the
# tier-2 EXCLUSION cost $8.6k on CAPYDRM's missing write). Bound the WRITE:
# the value written may differ from the current anchor by at most this much.
# A capped write still collapses most of a CAPYDRM-class phantom edge while a
# fake signal can move us at most 15 points.
T2_INTERMISSION_MAX_DELTA = 0.15
INTERMISSION_WINDOW_MIN = 25      # G1-end -> G2-start ran 17-21 min on 26AUG10
# ONE snapshot is enough here, unlike the pregame phase (MIN_SNAPSHOTS=2).
# A pregame snapshot is a SOLVER OUTPUT — solve_continuation root-finds against
# a spread — so medianing several guards against solver noise. An intermission
# snapshot is a direct algebraic identity, p3 = (series-m2)/(1-m2), read straight
# off two live books; there is no solver noise to average away, and it is already
# fenced by the |series-m2| >= 3c branch discriminator, the 5 < m2 < 95 bound,
# the 0.02 < p3 < 0.98 bound and the 0.010 adoption threshold.
#
# Set to 2 originally and it cost a real correction on REDLEV 26AUG10: d=0.056
# in the expected direction, rejected because the second snapshot landed 15s
# after game 2's first frame. The usable window is narrower than it looks —
# Riot's g1 'completed' lag trimmed a 17-minute intermission to ~11 — and at
# populate's ~2.5 min cadence a 2-snapshot floor keeps eating valid signals.
# 87% of intermissions exceed the threshold, so this gate carries most of the
# value; the two readings we did get sat 0.007 apart.
INTERMISSION_MIN_SNAPSHOTS = 1

# Hard exit if game-2 frames NEVER publish. Frame coverage is demonstrably
# patchy per league — CBLOL served game 1's opening page and 404'd on every
# mid-game window — so an event can otherwise sit in "intermission" forever,
# snapshotting and never resolving. Silent, and it would look identical to a
# feature that simply never fires.
INTERMISSION_TIMEOUT_MIN = 20
# On a TIMEOUT we do not know when game 2 actually began, so the normal
# "strictly before t2" filter has nothing to anchor to. Trust only the opening
# minutes of the intermission — those are furthest from any possible game-2
# start. A LoL break is never shorter than draft (~5 min) plus loading, and the
# one measured end to end (REDLEV 26AUG10) ran 17 min, so the first 8 minutes
# after the intermission began are pre-game-2 with a wide margin.
INTERMISSION_TIMEOUT_ADOPT_MIN = 8
# With ZERO intermission snapshots there is no snapshot to arm the timeout from,
# so it arms at game-1 start + this floor (resolving at floor + TIMEOUT = 65 min
# after kickoff). Set well past a normal game so a long map 1 is never mistaken
# for a stuck intermission: pro LoL games run 25-40 min, and closing the phase
# early would forfeit a real intermission later. A zero-snapshot timeout always
# DECLINES — its only job is to stop the record sitting unresolved forever.
G1_MIN_DURATION_MIN = 45
# Drop cache entries older than this.
CACHE_TTL_S = 36 * 3600
# Backoff on an unresolved Riot join (some matches are listed late).
LOOKUP_RETRY_S = 900

# NOTE ON BOOKS: this module does NO HTTP for market data. The caller injects a
# `book_fn(ticker) -> (yes_bid_c, yes_ask_c) | None`. In populate_configs that is
# wired to the existing per-cycle WS probe cache (`_fetch_map_markets_via_ws`),
# which costs nothing extra. Fetching here over REST was tried and is wrong: the
# bulk `/markets?series_ticker=...` scan needed to enumerate map tickers shares
# the scarce bucket with the live bot and returns 429 immediately
# (reference_kalshi_markets_shared_bucket).


# ── cache ────────────────────────────────────────────────────────────────────
def _load() -> dict:
    try:
        with open(CACHE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save(d: dict) -> None:
    try:
        tmp = CACHE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh)
        os.replace(tmp, CACHE)
    except Exception as e:
        print(f"  [ANCHOR] could not persist cache: {e}")


def _prune(d: dict) -> dict:
    """Age out the cache.

    A record must survive as long as it carries live state — `anchored_at` and
    `inter_anchored_at` are what make the pass idempotent, `lookup_failed_at` is
    what backs off the Riot join — but it must NOT survive forever, or every
    event ever seen accumulates. Once the newest timestamp on a record is older
    than the TTL, the match is long settled and the whole record goes.
    """
    now = time.time()
    cut = now - CACHE_TTL_S
    for ev in list(d):
        rec = d[ev]
        rec["snaps"] = [s for s in rec.get("snaps", []) if s.get("ts", 0) >= cut]
        rec["inter_snaps"] = [s for s in rec.get("inter_snaps", []) if s.get("ts", 0) >= cut]
        newest = max([s.get("ts", 0) for s in rec["snaps"]]
                     + [s.get("ts", 0) for s in rec["inter_snaps"]]
                     + [rec.get("anchored_at") or 0,
                        rec.get("inter_anchored_at") or 0,
                        rec.get("lookup_failed_at") or 0])
        if newest < cut:
            del d[ev]
    return d


# ── the pass ─────────────────────────────────────────────────────────────────
def _solve_leg(ev: dict, book_fn) -> Optional[dict]:
    """One reconciled snapshot for leg A off the live books."""
    is_bo5 = bool(ev.get("is_bo5"))
    sb = book_fn(ev["ticker_a"])
    if not sb:
        return None
    if not ev.get("map1_ticker_a"):
        return None
    mb = book_fn(ev["map1_ticker_a"])
    if not mb:
        return None
    s_bid, s_ask = sb
    m_bid, m_ask = mb
    # book_fn may hand back a one-sided book (None on a side) — that is a valid
    # "map decided" signal for map1_decided, but useless for solving. Reject.
    if None in (s_bid, s_ask, m_bid, m_ask):
        return None
    if not (0 < s_bid < s_ask < 100) or not (0 < m_bid < m_ask < 100):
        return None
    p1 = (m_bid + m_ask) / 200.0
    p2 = prob_reconcile.solve_continuation(s_bid, s_ask, m_bid, m_ask, p1, is_bo5)
    if p2 is None:
        return None
    # `cont` is what solve_continuation actually returns: the probability of
    # winning each REMAINING map. How it lands in the CSV differs by format —
    # BO3 keeps p1 and sets p2=cont (G3 is derived as (p1+p2)/2 downstream),
    # BO5 sets p2=p3=p4=p5=cont, matching what populate's RECONCILE writes at
    # capture time. `p3` here is the comparable "continuation we will use", so
    # the adoption threshold means the same thing for both.
    return {"ts": time.time(), "p1": round(p1, 4), "p2": round(p2, 4),
            "cont": round(p2, 4),
            "p3": round(p2 if is_bo5 else (p1 + p2) / 2, 4),
            "series": round((s_bid + s_ask) / 200.0, 4)}


def map1_decided(ev: dict, book_fn, settled_fn, g1_state: Optional[str]):
    """Is map 1 over? -> (decided: bool, source: str)

    Ordered by trustworthiness, NOT by convenience:

    1. Riot `game 1 state == "completed"`. Same feed we take start times from,
       and not derived from any price. On CBLOL 26AUG10 it arrived ~6-8 min
       after map 1 actually ended.
    2. The map-1 book pinned at >=99c bid or <=1c ask. A fallback purely for the
       case where Riot's state lags longer than the break — it ran 26 min late
       on BROAP game 1, which would miss an entire intermission. A book at 99/-
       is not a guess about the result, it is the market stating the map is over.
    3. Kalshi settlement. Authoritative when it fires, but it USUALLY DOES NOT
       during the intermission: on REDLEV 26AUG10 the map-1 market still read
       `status=active, result='', close=2026-08-12T21:00:00Z` while the book sat
       at bid=99. Kalshi settles on its own batch schedule, not the game's, so
       this can never be the primary signal. Keeping it costs nothing.
    """
    if g1_state == "completed":
        # `g1_src` names the feed that supplied the state when it is not Riot
        # (Dota/TI: populate injects bo3's observed game-1 end as "completed").
        return True, ev.get("g1_src") or "riot:g1-completed"
    m1 = ev.get("map1_ticker_a")
    if m1:
        b = book_fn(m1)
        if b:
            bid, ask = b
            # STRICT: the winning side pinned at >=99 AND the losing side not
            # quoted at all. Both halves are required. A lopsided but still
            # two-sided book (99 bid / 100 ask) is a live market on a nearly
            # decided map, not a finished one — and a false positive here takes
            # an intermission snapshot mid-game-1, where the back-out formula is
            # meaningless. We would rather miss the intermission entirely.
            if bid is not None and bid >= 99 and ask is None:
                return True, "book:map1-pinned-yes"
            if ask is not None and ask <= 1 and bid is None:
                return True, "book:map1-pinned-no"
        if settled_fn(m1) is True:
            return True, "kalshi:settled"
    return False, ""


def _intermission_snapshot(ev: dict, book_fn, decided: bool) -> Optional[dict]:
    """Market-implied P(leg A wins the decider), during the G1->G2 intermission.

    Only valid once map 1 is DECIDED — before that the series price still
    reflects an in-progress game 1 and the back-out is meaningless.
    """
    if bool(ev.get("is_bo5")):
        return None      # BO3 algebra only — a BO5 series after G1 depends on
                         # p3/p4/p5 jointly and does not invert this way.
    if not ev.get("map2_ticker_a") or not ev.get("map1_ticker_a"):
        return None
    if not decided:
        return None                                   # game 1 not decided yet
    sb = book_fn(ev["ticker_a"])
    mb = book_fn(ev["map2_ticker_a"])
    if not sb or not mb:
        return None
    if None in (sb[0], sb[1], mb[0], mb[1]):
        return None      # one-sided series or map-2 book: cannot take a mid
    s_mid = (sb[0] + sb[1]) / 2.0
    m_mid = (mb[0] + mb[1]) / 2.0
    if not (5.0 < m_mid < 95.0):
        return None                                   # map 2 already decided/degenerate
    # ── BOOK-WIDTH GUARD (26AUG20, operator) ──────────────────────────────
    # There was NO width check here. A one-sided book was rejected and a
    # degenerate map-2 mid was rejected, but a two-sided 20c-wide book passed
    # and its midpoint was treated as a price. Post-halt that is exactly what
    # happens: books reopen 20c wide and the "mid" is a fiction.
    #
    # This bounds the thing that actually matters — how wrong p3 can be —
    # rather than the raw spread, because the back-out AMPLIFIES:
    #     leader  p3 = (series - m2)/(1 - m2)   ->  d(p3)/d(series) = 1/(1-m2)
    # so a series book of width W contributes W/2 * amplification of p3
    # uncertainty. HLEDK: W~20c at m2~0.50 (2x) => +-20pts on p3. The existing
    # INTERMISSION_MAX_AMPLIFICATION only bounds the MULTIPLIER; it assumes the
    # input is tight. Both guards are needed: one bounds conditioning, this
    # bounds input quality.
    s_w = abs(sb[1] - sb[0])
    m_w = abs(mb[1] - mb[0])
    _den = max(1e-6, (1.0 - m_mid / 100.0) if m_mid <= 50.0 else (m_mid / 100.0))
    _ampl = 1.0 / _den
    _p3_unc = (s_w / 2.0) * _ampl
    if _p3_unc > MAX_SNAPSHOT_P3_UNCERTAINTY_C:
        return None      # series book too wide for this conditioning
    if m_w > MAX_SNAPSHOT_MAP_WIDTH_C:
        return None      # map-2 book too wide to trust as the divisor
    # WHICH SIDE WON GAME 1? leg A is just whichever row sorted first, so this
    # has to be derived, not assumed. Both branches are exact for a BO3:
    #     A leads 1-0 : series_A = m2 + (1-m2)*p3   =>  series_A > m2
    #     A trails 0-1: series_A = m2 * p3          =>  series_A < m2
    # so the sign of (series - m2) IS the discriminator. Require a real gap so a
    # crossed/noisy book near series==m2 doesn't pick a branch at random.
    if abs(s_mid - m_mid) < 3.0:
        return None
    if s_mid > m_mid:
        p3 = (s_mid - m_mid) / (100.0 - m_mid)        # A won game 1
        leads = True
    else:
        p3 = s_mid / m_mid                            # A lost game 1
        leads = False
    if not (0.02 < p3 < 0.98):
        return None
    return {"ts": time.time(), "p3": round(p3, 4), "leads": leads,
            "series": round(s_mid / 100.0, 4), "m2": round(m_mid / 100.0, 4)}


def run_pass(events: List[dict], book_fn, settled_fn=None, log=print) -> Dict[str, dict]:
    """Record a pregame snapshot per event; return re-anchors to apply.

    `events`: [{event_base, ticker_a, map1_ticker_a, team_a, team_b, date,
                is_bo5, cur_p3}]   — ticker_a / map1_ticker_a / team_a / cur_p3
                must all describe the SAME side.
    `book_fn`: ticker -> (yes_bid_c, yes_ask_c) | None. No HTTP is done here.

    Returns {event_base: {p1, p2, series, source_ts, delta_p3, start}} for the
    events whose anchor should be rewritten. Values are for leg A; the caller
    writes the complement to leg B.
    """
    out: Dict[str, dict] = {}
    if not events:
        # An empty slate is INFORMATION, not nothing-to-write: stamp a fresh
        # (empty) pregame_state so the file's written_ts doesn't age past the
        # 15-min staleness threshold and disarm pregame sizing book-wide.
        # Overnight 26AUG18 the file sat 4.5h stale through every 0-event
        # cycle and the first event of the new slate went out ungated.
        # NOTE: the Riot-index-unavailable return below deliberately does NOT
        # write — a Riot outage must degrade to stale/fail-open, not blank
        # the last known records (see pregame_state's failure policy).
        try:
            import pregame_state
            pregame_state.write_state({})
        except Exception as e:
            log(f"  [ANCHOR] pregame_state empty-slate write failed "
                f"({type(e).__name__}: {e}) — file will go stale (fail-open)")
        return out
    d = _prune(_load())
    try:
        rgs.load_index()
    except Exception as e:
        log(f"  [ANCHOR] Riot index unavailable ({e}) — no re-anchoring this pass")
        return out

    # PRE-PASS: resolve every match id first (pure in-memory index scan), then
    # warm the Riot caches concurrently. Doing this inline in the main loop
    # costs ~0.6s per event serially — 18.9s on a 30-event slate, worn on EVERY
    # populate cycle because a pregame event's start poll is a cache miss by
    # design. With the pre-pass the same slate resolves in ~1.7s.
    for ev in events:
        eb = ev.get("event_base")
        if not eb:
            continue
        # BO5 runs the KICKOFF phase (solve_continuation handles both formats),
        # but never the intermission — a BO5 series after game 1 depends on
        # p3/p4/p5 jointly and the back-out does not invert. Enforced below.
        rec = d.setdefault(eb, {"snaps": [], "anchored_at": None, "match_id": None})
        if ev.get("no_feed"):
            continue        # book-driven sport (VAL/CS2): no Riot join exists
        if rec.get("match_id"):
            continue
        # Retry an unresolved join periodically — Riot lists some matches late —
        # but don't re-log every 2-minute pass. An alias-file edit NEWER than
        # the failure stamp voids the backoff: the whole point of the edit is
        # to make the next lookup succeed, and this stamp persists in the
        # cache ACROSS restarts, so honouring it would turn a 10-second alias
        # fix into a 15-minute wait even on a fresh process (26AUG11 lesson).
        last = rec.get("lookup_failed_at") or 0
        if time.time() - last < LOOKUP_RETRY_S and rgs.aliases_mtime() <= last:
            continue
        rec["match_id"] = rgs.find_match(ev.get("team_a"), ev.get("team_b"),
                                         ev.get("date")) or ""
        if not rec["match_id"]:
            rec["lookup_failed_at"] = time.time()
            # Loud, not silent: an unjoined event keeps the frozen anchor.
            log(f"  [ANCHOR] {eb}: no Riot match for "
                f"{ev.get('team_a')!r} vs {ev.get('team_b')!r} — "
                f"anchor stays frozen (retry in {LOOKUP_RETRY_S//60}min)")
            try:
                for _ours, _riot in rgs.suggest_aliases(
                        ev.get("team_a"), ev.get("team_b"), ev.get("date")):
                    log(f"  [ANCHOR]   ^ probable alias: riot {_riot!r} == ours "
                        f"{_ours!r} — paste into riot_team_aliases.csv: "
                        f"{_ours},{_riot}")
            except Exception as _e:
                log(f"  [ANCHOR]   (alias suggester failed: {_e!r})")
    try:
        rgs.prefetch([d[e["event_base"]].get("match_id") for e in events
                      if e.get("event_base") in d])
    except Exception as e:
        log(f"  [ANCHOR] prefetch failed ({type(e).__name__}: {e}) — continuing serially")

    # Side-channel for the quoter's pregame sizing gate. Built here because this
    # loop already resolves the TRUE game-1 start for every LoL event; the bots
    # must never do a Riot lookup on the quote path. See pregame_state.py.
    pg = {}

    for ev in events:
        eb = ev.get("event_base")
        if not eb or eb not in d:
            continue
        rec = d[eb]
        mid = rec.get("match_id") or None

        # UNRESOLVED vs UNSTARTED are opposites in information content:
        #
        #   unresolved — Riot never joined this fixture (LPLOL and anything
        #   else outside the index). ZERO information, and "we know nothing"
        #   must not gate the book — same philosophy as pregame_state's
        #   file-missing case. pregame=None means "no opinion": the config
        #   gate never applies and the event trades exactly as pre-anchor,
        #   real values from the start (operator decision 26AUG11).
        #
        #   unstarted — Riot AFFIRMATIVELY tracks the fixture as not played.
        #   FAIL-CLOSED. DNFHLE 26AUG11 sat here its whole duration (Riot
        #   tracked the main-roster fixture that never played) — exactly when
        #   we want to be small.
        no_feed = bool(ev.get("no_feed"))
        if no_feed:
            # Book-driven sports (VAL/CS2, 26AUG12): no per-game feed exists,
            # so there is no kickoff phase and no pregame sizing signal — pg
            # stays "no opinion" (the legacy taker latch remains those sports'
            # pregame protection). Intermission only: the book-pinned map-1
            # detector plus the TIMEOUT adoption path below, with per-sport
            # windows passed in by populate (VAL intermissions run ~3.5 min).
            # Deliberately NO pregame signal for book-driven sports: the VAL
            # taker gate was shadow-measured and RETIRED 26AUG12 (bo3 can
            # publish the veto minutes-to-seconds before the pistol on
            # rescheduled matches, and a blocked pistol is the worst miss in
            # VAL). pregame=None -> the config gate never applies.
            pg[eb] = {"pregame": None, "source": "no-feed", "start_ts": None}
            if not rec.get("anchored_at"):
                rec["anchored_at"] = time.time()
                log(f"  [ANCHOR] {eb}: book-driven sport — "
                    f"intermission-only tracking armed")
        elif not mid:
            pg[eb] = {"pregame": None, "source": "riot:unresolved", "start_ts": None}
        else:
            pg[eb] = {"pregame": True, "source": "riot:pending", "start_ts": None}

        # Riot's own game count is an independent BO3/BO5 read. If it disagrees
        # with our CSV, the join is probably wrong (or the format is) — either
        # way, don't touch the anchor on a guess.
        if mid:
            n_games = len(rgs.match_games(mid))
            want = 5 if bool(ev.get("is_bo5")) else 3
            if n_games and n_games != want:
                if not rec.get("format_warned"):
                    rec["format_warned"] = True
                    log(f"  [ANCHOR] {eb}: our CSV says "
                        f"{'BO5' if want == 5 else 'BO3'} but Riot lists {n_games} "
                        f"games — refusing to re-anchor (format or join mismatch)")
                continue

        start = rgs.game_start(mid, 1) if mid else None
        if start is not None:
            pg[eb] = {"pregame": False, "source": "riot:started",
                      "start_ts": start.timestamp()}
        elif mid:
            pg[eb]["source"] = "riot:unstarted"

        # bo3-frames OR-branch (26AUG15): the game is observably LIVE on
        # bo3.gg (net_worth flowing = draft over, rosters locked) while Riot
        # has not posted its first frame yet. Release the GATE only — `start`
        # stays None so PHASE A kickoff adoption still waits for Riot's
        # timestamp (or keeps the captured anchor, as today). Sticky upstream
        # (lol_game_state stamps first transition only), so a between-games
        # net_worth lull cannot re-latch (DKCT1A class).
        if start is None and pg[eb].get("pregame") is not False \
                and ev.get("ext_started_at"):
            pg[eb] = {"pregame": False, "source": "bo3:frames",
                      "start_ts": ev["ext_started_at"]}

        if start is None and not no_feed:
            if rec.get("anchored_at"):
                continue
            snap = _solve_leg(ev, book_fn)
            if snap:
                rec["snaps"].append(snap)
                rec["snaps"] = rec["snaps"][-400:]
            continue

        # ── PHASE A adoption: game 1 has started, take the pre-kickoff solves ──
        # `allow_kickoff` is set by populate (flag + tier policy). Mark the record
        # anchored anyway so a disabled phase does not leave it re-evaluating
        # every cycle for the rest of the event.
        if not rec.get("anchored_at") and not ev.get("allow_kickoff", True):
            rec["anchored_at"] = time.time()
            log(f"  [ANCHOR] {eb}: kickoff phase DISABLED for this event "
                f"(tier/flag) — keeping captured anchor")
        elif not rec.get("anchored_at"):
            t0 = start.timestamp()
            win = [s for s in rec["snaps"] if t0 - WINDOW_MIN * 60 <= s["ts"] < t0]
            rec["anchored_at"] = time.time()
            if len(win) < MIN_SNAPSHOTS:
                log(f"  [ANCHOR] {eb}: started {start:%H:%M:%S}Z but only {len(win)} "
                    f"pregame snapshot(s) in the {WINDOW_MIN}min window — keeping frozen anchor")
            else:
                is_bo5 = bool(ev.get("is_bo5"))
                p1 = statistics.median(s["p1"] for s in win)
                p2 = statistics.median(s["p2"] for s in win)
                p3 = p2 if is_bo5 else (p1 + p2) / 2
                cur_p3 = ev.get("cur_p3")
                delta = abs(p3 - cur_p3) if cur_p3 is not None else None
                if delta is not None and delta <= ADOPT_THRESHOLD:
                    log(f"  [ANCHOR] {eb}: start {start:%H:%M:%S}Z, "
                        f"p3 {cur_p3:.3f} -> {p3:.3f} (d={delta:.4f}) below "
                        f"{ADOPT_THRESHOLD:.3f} — keeping captured anchor")
                else:
                    out[eb] = {
                        "p1": round(p1, 3), "p2": round(p2, 3),
                        "cont": round(p2, 3), "is_bo5": is_bo5,
                        # BO5 carries the continuation into p3/p4/p5 as well —
                        # same shape RECONCILE writes at capture.
                        "p3": round(p2, 3) if is_bo5 else None,
                        "series": round(prob_reconcile.series_prob(p2, is_bo5), 3),
                        "phase": "kickoff",
                        "source_ts": max(s["ts"] for s in win), "n": len(win),
                        "delta_p3": delta, "start": start.isoformat()}
                    log(f"  [ANCHOR] {eb}: RE-ANCHORED at true start {start:%H:%M:%S}Z "
                        f"(n={len(win)} snaps) "
                        f"p3 {cur_p3 if cur_p3 is None else f'{cur_p3:.3f}'} -> {p3:.3f} "
                        f"(d={delta:.4f})")

        # ── PHASE B: intermission G3 re-anchor — the G1-winner-bias fix ──
        # Snapshots are only taken once map 1 has SETTLED (before that the
        # series price still reflects an in-progress game 1 and the back-out is
        # meaningless), and are adopted retrospectively from strictly before
        # game 2's true start, so the ~5 min of frame delay can never bleed
        # game-2 information into the anchor. May overwrite a kickoff anchor
        # emitted above in the same pass — later information wins.
        # Per-event phase permissions (26AUG11). The POLICY lives in populate —
        # which tiers and which phases are enabled — and is passed in per event;
        # this module only enforces it. Default True so an un-migrated caller
        # keeps today's behavior.
        #
        # Why tier matters: the whole anchor premise is that the SERIES PRICE is
        # the level authority and any gap is our staleness. In tier-2 that
        # premise is weak — books run 5c+ wide and "the market moved" is often
        # one participant repricing. The intermission phase is the exposed one,
        # since it DIVIDES by a book price (see the conditioning guard) and its
        # two worst writes today, OTNBS 13.3x and 3BLPIV 3.8x, were both thin
        # tier-2 books.
        if not ev.get("allow_intermission", True):
            continue
        if settled_fn is None or rec.get("inter_anchored_at") or bool(ev.get("is_bo5")):
            continue
        g2 = rgs.game_start(mid, 2) if mid else None
        if g2 is None:
            g1_state = (next((g.get("state") for g in rgs.match_games(mid)
                              if g.get("number") == 1), None) if mid else None)
            if g1_state is None:
                # Feedless sports can carry an external game-state read (TI
                # Dota: bo3's observed g1 end, injected by populate). Riot,
                # when joined, stays authoritative — this is a fallback only.
                g1_state = ev.get("g1_state_ext")
            decided, src = map1_decided(ev, book_fn, settled_fn, g1_state)
            if decided and not rec.get("decided_src"):
                rec["decided_src"] = src
                rec["decided_at"] = time.time()
                log(f"  [ANCHOR] {eb}: map 1 decided via {src} — "
                    f"intermission snapshots start now")
            # POST-HALT PROVENANCE FILTER (26AUG20, HLEDK). A snapshot taken
            # while Kalshi's books are still repricing after a maintenance halt
            # is poison for the back-out: p3=(series-m2)/(1-m2) DIVIDES by
            # (1-m2), so a ~13c stale series price at m2~0.50 doubled into a
            # ~27pt error (true p3 ~0.52, computed 0.250). Only the 0.15 delta
            # cap stopped it landing in full; it still wrote 0.374 vs a market
            # implying ~0.51.
            #
            # Gating the WRITE is not enough — HLEDK's write landed at
            # resume+534s, past any sane write cooldown, using snapshots taken
            # from resume+0s. The fix has to reject the DATA, not the moment.
            # Dropping snapshots here degrades safely: too few snapshots means
            # no re-anchor and the frozen anchor is kept.
            _snap_ok = True
            try:
                import populate_configs as _pcfg
                _snap_ok, _snap_why = _pcfg.kalshi_writes_safe()
            except Exception:
                pass                      # never let the guard break the pass
            if not _snap_ok:
                if not rec.get("_halt_snap_warned"):
                    rec["_halt_snap_warned"] = True
                    log(f"  [ANCHOR] {eb}: intermission snapshots DISCARDED — "
                        f"{_snap_why} (26AUG20 HLEDK post-halt provenance rule)")
                snap = None
            else:
                rec.pop("_halt_snap_warned", None)
                snap = _intermission_snapshot(ev, book_fn, decided)
            if snap:
                rec.setdefault("inter_snaps", []).append(snap)
                rec["inter_snaps"] = rec["inter_snaps"][-200:]
            isn = rec.get("inter_snaps") or []
            # Arm the timeout from the first intermission snapshot when we have
            # one, else from game-1 start plus a floor for the game itself.
            # Arming it off inter_snaps[0] ALONE was the 26AUG10 bug: on PNGLOS
            # map 1 was never detected as decided, so there were no snapshots,
            # so `not isn` short-circuited and the timer never armed — the event
            # sat unresolved through a 34-minute intermission. The case that
            # most needs a timeout is exactly the one with no snapshots.
            # Per-sport windows. LoL keeps the module defaults, armed from the
            # first snapshot (Riot frames are the primary exit; this is only
            # the fallback). No-feed sports (VAL/CS2) arm from DECIDED_AT —
            # our first sight of the map-1 pin, 0..1 cycle after the true
            # end — and adopt a FIXED 5 minutes later (operator, 26AUG12):
            # measured VAL breaks put the next-map skeleton at +9min and the
            # pistol at +11min, while populate's ~3.3min sampling means a
            # skeleton-triggered exit would usually be seen too late. +5min
            # from decided executes at +5..+10 real time — pre-pistol without
            # needing to catch the skeleton. A decided event with ZERO
            # snapshots inside the window stays frozen (books one-sided
            # through the break) — safe, and the log says so.
            _t_out = float(ev.get("inter_timeout_min") or INTERMISSION_TIMEOUT_MIN)
            _t_adopt = float(ev.get("inter_adopt_min") or INTERMISSION_TIMEOUT_ADOPT_MIN)
            if no_feed:
                _base = rec.get("decided_at") or (isn[0]["ts"] if isn else None)
            else:
                _base = isn[0]["ts"] if isn else None
            armed_at = (_base if no_feed
                        else (isn[0]["ts"] if isn
                              else (start.timestamp() + G1_MIN_DURATION_MIN * 60
                                    if start is not None else None)))
            if armed_at is None or time.time() - armed_at < _t_out * 60:
                continue
            # TIMEOUT: game-2 start never confirmed. Resolve off the opening
            # minutes only — everything later could already be inside game 2.
            t2 = None
            win = [s for s in isn
                   if _base is not None and s["ts"] - _base <= _t_adopt * 60]
            log(f"  [ANCHOR] {eb}: no game-2 start signal after "
                f"{_t_out:g}min — TIMEOUT exit, using the first "
                f"{_t_adopt:g}min of the intermission "
                f"({len(win)} of {len(isn)} snapshots)")
        else:
            t2 = g2.timestamp()
            win = [s for s in rec.get("inter_snaps", [])
                   if t2 - INTERMISSION_WINDOW_MIN * 60 <= s["ts"] < t2]
        rec["inter_anchored_at"] = time.time()
        cur = ev.get("cur_p3")
        when = f"game 2 started {g2:%H:%M:%S}Z" if t2 else "intermission timed out"
        if len(win) < INTERMISSION_MIN_SNAPSHOTS:
            log(f"  [ANCHOR] {eb}: {when} but only "
                f"{len(win)} intermission snapshot(s) "
                f"(map1 decided via {rec.get('decided_src') or 'NEVER DETECTED'}) "
                f"— G3 anchor stays frozen")
            continue
        p3 = statistics.median(s["p3"] for s in win)

        # ── Conditioning guard (26AUG11) ────────────────────────────────────
        # The back-out DIVIDES by a book price, so its sensitivity to a 1c error
        # in `series` is 1/denominator:
        #     leader   p3 = (series - m2) / (1 - m2)   ->  denom = 1 - m2
        #     trailer  p3 =  series       /  m2        ->  denom = m2
        # OTNBS 26AUG11: trailing leg with m2 = 0.075, i.e. 13.3x amplification,
        # against a series book quoted 0.01/0.06 — 5c wide on a 3.5c mid. Three
        # consecutive snapshots produced 0.333 / 0.375 / 0.467 and we wrote a
        # 23-POINT correction out of pure noise. It happened to point toward the
        # market, which is luck, not a property.
        #
        # Note the asymmetry: for a LEADER the danger is m2 near 1, for a
        # TRAILER m2 near 0. A symmetric floor on the raw price guards the wrong
        # end half the time, so the denominator is computed per snapshot and
        # medianed — which also handles a window whose `leads` flag is mixed.
        #
        # K=3 (denominator >= 0.333). Started at K=4; 3BLPIV 26AUG11 passed at
        # 3.8x and was WRONG — a 4c series move became a 15-point p3 swing, the
        # median then picked the stalest of 3 drifting snapshots, and we wrote
        # 0.547 against a live back-out of 0.327. Tightened to 3. Against every
        # intermission observed 26AUG10-11: REDLEV 1.3x, DKCNSEA 1.9x,
        # FOXYDRXC 2.7x pass (and those two include the correct G1-bias
        # catches); 3BLPIV 3.8x, G2NKHK 4.9x, OTNBS 13.3x reject. Deliberately NOT a dispersion test:
        # dispersion would have killed FOXYDRXC (0.085) and DKCNSEA (0.043),
        # whose windows were genuinely drifting. Drift is signal; division by a
        # near-zero price is not.
        _dens = [((1.0 - s["m2"]) if s.get("leads") else s["m2"]) for s in win
                 if isinstance(s.get("m2"), (int, float))]
        _den = statistics.median(_dens) if _dens else 0.0
        _ampl = (1.0 / _den) if _den > 0 else float("inf")
        if _ampl > INTERMISSION_MAX_AMPLIFICATION:
            log(f"  [ANCHOR] {eb}: intermission p3 {cur if cur is None else f'{cur:.3f}'}"
                f" -> {p3:.3f} REFUSED — ill-conditioned back-out "
                f"(denominator={_den:.3f}, {_ampl:.1f}x amplification > "
                f"{INTERMISSION_MAX_AMPLIFICATION:.0f}x); a 1c series error moves "
                f"p3 by {_ampl:.1f}c. Keeping the frozen anchor.")
            continue

        delta = abs(p3 - cur) if cur is not None else None
        if delta is not None and delta <= INTERMISSION_THRESHOLD:
            log(f"  [ANCHOR] {eb}: intermission p3 {cur:.3f} -> {p3:.3f} "
                f"(d={delta:.4f}) below {INTERMISSION_THRESHOLD:.3f} — keeping")
            continue
        # ALL TIERS since 26AUG13 (was tier-2 only): FOXDRX wrote p3=0.84 —
        # 0.5 from reality — through this exact spot because the clamp assumed
        # tier-1 books are always trustworthy. A Kalshi-maintenance-frozen
        # tier-1 book carries PRE-decided prices with fresh timestamps and the
        # solve passes every consistency guard, so the write bound is the last
        # line of defense and must not be tier-conditional. Cost: ~$10k maker
        # inventory defended at the phantom level to the position cap.
        if (cur is not None
                and delta is not None and delta > T2_INTERMISSION_MAX_DELTA):
            _p3_raw = p3
            p3 = cur + (T2_INTERMISSION_MAX_DELTA if p3 > cur
                        else -T2_INTERMISSION_MAX_DELTA)
            log(f"  [ANCHOR] {eb}: intermission write CAPPED — back-out "
                f"{_p3_raw:.3f} is {delta:.3f} from anchor {cur:.3f}, writing "
                f"{p3:.3f} (max delta {T2_INTERMISSION_MAX_DELTA:.2f})")
            delta = abs(p3 - cur)
        # p1 and p2 reach the post-G1 math ONLY through
        # solve_game_3_probability -> (p1+p2)/2 (hedge_engine state_1/state_2;
        # series_A is discarded, arber_bot.py:1431-1435), so writing both equal
        # to the target sets p3 exactly and changes nothing else. State 0 cannot
        # recur once game 1 is decided.
        out[eb] = {"p1": round(p3, 3), "p2": round(p3, 3), "series": None,
                   "phase": "intermission" if t2 else "intermission-timeout",
                   "source_ts": max(s["ts"] for s in win),
                   "n": len(win), "delta_p3": delta,
                   "start": g2.isoformat() if t2 else None}
        log(f"  [ANCHOR] {eb}: G3 RE-ANCHORED ({when}, n={len(win)}, "
            f"decided via {rec.get('decided_src') or '?'}) "
            f"p3 {cur if cur is None else f'{cur:.3f}'} -> {p3:.3f} (d={delta:.4f})")

    _save(d)
    try:
        import pregame_state
        pregame_state.write_state(pg)
    except Exception as e:
        # The anchor's own job must not fail because the sizing side-channel did.
        # A missing/stale file makes the quoter gate INERT (full size), which is
        # today's behavior — see pregame_state's failure policy.
        log(f"  [ANCHOR] pregame_state write failed ({type(e).__name__}: {e}) — "
            f"quoter pregame sizing will be inert")
    return out
