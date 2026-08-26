"""poll_tennis.py — watch one Kalshi tennis match; log state + vig-free midpoint.

READ-ONLY. No order endpoints exist in this process.

    python3 poll_tennis.py list                            # today's ATP/WTA matches
    python3 poll_tennis.py list --day 26AUG24
    python3 poll_tennis.py inspect KXATPMATCH-26AUG24ALTCOM
    python3 poll_tennis.py watch  KXATPMATCH-26AUG24ALTCOM --me ALT --interval 2

WHAT IT WRITES (both, per match, under ./tapes/)
    <ev>.jsonl      every poll: raw details + both mids + derived state. This is
                    the point-level tape for testing the 1.5x move-amplification
                    and mean-reversion findings at higher resolution than the
                    hand-logged Sherif match allowed.
    <ev>.log.json   live.py-compatible boundary log — one observation per completed
                    game, vig-free. `cp tapes/<ev>.log.json live_log.json` then
                    `python3 live.py report` works unchanged.

RATE BUDGET. Per cycle: 1 batched live_data + 1 markets call = 2 GETs. At the
default 2s that is ~60 GETs/min drawn from the SAME account bucket as the live
esports trading system in general_level_based_quoting. This is aggressive by
operator decision 26AUG24. Protections: polling only runs while the match status is
live (otherwise it idles at --idle-interval, default 60s), a hard RPS ceiling, 429
backoff that surrenders the bucket instead of retrying into it, and the kill flag
`disable_tennis_poller.flag` in the working directory.

best_of and the men/women split prior are read from the milestone, so the only
thing you choose is which player is "me".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import kalshi_tennis as kt
from implied_model import ImpliedModel
from market_implied import ImpliedSplitError

TAPES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tapes")

# FIT VARIANTS CARRIED LIVE. Until 26AUG25 the poller carried one fit — equal-weight,
# refit on every boundary, i.e. ROLLING — so the static-vs-rolling and EWMA questions
# could only ever be asked afterwards, by replaying a settled tape. Carrying all of
# them live means the tape records what each variant believed AT THE TIME, which is the
# only version of the comparison that is not hindsight. Costs no extra Kalshi GETs:
# same payload, more local arithmetic.
#
#   static    frozen after STATIC_N boundaries — keeps a view the market can diverge from
#   rolling   equal weight over every boundary (the original `model`)
#   ewma<h>   weight 0.5**((n-1-j)/h) — follows a genuine re-rating instead of averaging it
#
# Only the EWMA fits need a full rebuild per boundary (their weights all shift when n
# grows); static and rolling stay incremental. Rebuild happens on GAME BOUNDARIES ONLY
# (~20-30 per match), never on the 2s tick, so it cannot disturb the poll cadence.
STATIC_N = 2


def _ewma_update(m, price, state, halflife):
    """Fold one boundary into an EWMA fit IN PLACE. One _fit(), not n.

    The weights are 0.5**((n-1-j)/h) and every one of them shifts when n grows, which
    is why this looks like it needs a rebuild from scratch on each boundary. It does
    not: decaying every stored weight by 0.5**(1/h) and appending the new one at 1.0
    yields exactly the same weight vector. Verified identical to the rebuild to 6.4e-09
    in p and q across a 30-boundary match, for 15x less work (9.87s -> 0.64s) and well
    under half the memo traffic.

    The weights cannot simply be left un-normalised to dodge this: _residuals appends a
    PRIOR row carrying a fixed weight, so scaling the observation weights by any
    constant silently changes the balance between the data and the split prior.

    The 1e-6 floor keeps a boundary many halflives old from reaching exactly zero,
    which would drop it out of the residual set and change the fit's rank.
    """
    dec = 0.5 ** (1.0 / halflife)
    m.obs = [(st, px, max(w * dec, 1e-6)) for st, px, w in m.obs]
    m.observe(price, weight=1.0, **state)
    return m


def _ahead_states(pstate):
    """The two states one point later: tracked side wins it, or loses it.

    Points are just incremented. The model handles a point score that finishes the
    game on its own - game_win_prob(w, 4, 2) is 1.0 and set_win_prob then advances the
    game count - verified identical to constructing the next game explicitly. This also
    works inside a tiebreak, where tiebreak_win_prob rotates the server itself.
    """
    win = dict(pstate)
    win["points_me"] = pstate.get("points_me", 0) + 1
    lose = dict(pstate)
    lose["points_opp"] = pstate.get("points_opp", 0) + 1
    return win, lose


def _point_ahead_block(models, pstate, server):
    """Price the observed state AND one point ahead, for every variant.

    WHY THIS EXISTS. Our scoreboard runs roughly a point behind the participants with
    faster feeds: when we read 15-30 the point in progress is often already decided, so
    the market we are quoting against is pricing 15-40 or 30-30. Comparing
    model(our score) to market(their score) is then comparing two different information
    sets, which is a systematic bias, not noise - and it biases in the direction we
    would misread as "the model lags the market".

    So the tape records what each variant WOULD say one point later, on BOTH branches.
    Offline, once the next observed score reveals which way the point went, the market
    price at this tick can be compared against the branch that actually happened: if it
    already matched that branch, the market had the point before we did, and by how much
    is measurable. Costs no extra GETs - two more evaluations of a model already fitted.

    DO NOT add a probability-weighted blend of the two branches. p*price(win) +
    (1-p)*price(lose) is IDENTICALLY price(now) - the model is a martingale over points
    by construction, verified to 0.00e+00 across 576 states - so a blend carries no
    information at all. The branches are only informative kept apart.
    """
    win, lose = _ahead_states(pstate)
    out = {"now": {}, "win": {}, "lose": {}, "p_point": {}}
    for tag, m in models:
        if m is None or m.p is None:
            continue
        # per-point win probability for whoever is about to serve
        pp = m.p if server == "me" else m.q
        try:
            n = m.price(**pstate)[0]
            w = m.price(**win)[0]
            l = m.price(**lose)[0]
        except Exception:
            continue
        out["now"][tag] = round(n, 4)
        out["win"][tag] = round(w, 4)
        out["lose"][tag] = round(l, 4)
        out["p_point"][tag] = round(pp, 4)
    return out


def _market_clock(ahead, px, tag="rolling"):
    """Where does the market sit between the lose-branch and the win-branch?

        0.0  market is pricing as if the point was LOST
        1.0  market is pricing as if the point was WON
        p    market is on OUR clock - it has not seen the point either

    This is the lag measurement in one number. Compare it against p_point: a market
    consistently above p on points the tracked side went on to win (and below on ones
    they lost) is a market that already knows the outcome. Returns None when the two
    branches are too close to divide by, which happens when a single point barely moves
    the match price.
    """
    if not ahead or px is None:
        return None
    w, l = ahead.get("win", {}).get(tag), ahead.get("lose", {}).get(tag)
    if w is None or l is None or abs(w - l) < 1e-4:
        return None
    return round((px - l) / (w - l), 4)


def _now() -> float:
    return time.time()


def cmd_list(a):
    feed = kt.Feed(rps=a.rps)
    ms = feed.matches(day=a.day)
    if not ms:
        print("no open ATP/WTA match events" + (f" for {a.day}" if a.day else ""))
        return
    print(f"{len(ms)} match events:")
    for m in ms:
        print(f"  {m['tour']:3s}  {m['event_ticker']:34s}  {m['title']}")
    print(f"\n[{feed.n_get} GETs, {feed.n_429} rate-limited]")


def cmd_scan(a):
    """Which of today's matches are actually live, and what does the feed say?

    /milestones only filters by related_event_ticker — min_start_date, order and
    status are SILENTLY IGNORED (verified 26AUG24: all three returned the same
    Feb-2025 rows), and the collection paginates ascending from 2025. So there is no
    cheap server-side "today's live matches" query; the ids have to be resolved one
    event at a time, which is why this is capped by --limit.
    """
    feed = kt.Feed(rps=a.rps)
    evs = feed.matches(day=a.day)[:a.limit]
    print(f"resolving milestones for {len(evs)} events at {a.rps} rps...", file=sys.stderr)
    mids, meta = [], {}
    for e in evs:
        mil = feed.milestone(e["event_ticker"])
        if not mil:
            continue
        mids.append(mil["id"])
        meta[mil["id"]] = (e, mil)
    det = feed.live_data(mids)

    rows = []
    for mid, (e, mil) in meta.items():
        d = det.get(mid) or {}
        dm = mil.get("details") or {}
        rows.append((mil.get("start_date") or "", (d.get("status") or "?"), e["tour"],
                     e["event_ticker"], e["title"], dm.get("tournament_name") or "",
                     d, mil))
    rows.sort()

    live = [r for r in rows if kt.is_live(r[6])]
    print(f"\n{len(rows)} matches resolved, {len(live)} LIVE\n")
    print(f"{'start (UTC)':<20} {'status':<12} {'tour':<4} {'score':<26} match")
    for start, st, tour, ev, title, tname, d, mil in rows:
        sc = ""
        if d:
            c1 = [r.get("score") for r in (d.get("competitor1_round_scores") or [])]
            c2 = [r.get("score") for r in (d.get("competitor2_round_scores") or [])]
            if c1 or c2:
                sc = " ".join(f"{x}-{y}" for x, y in zip(c1, c2))
                sc = f"[{d.get('competitor1_overall_score')}-{d.get('competitor2_overall_score')}] {sc}"
        print(f"{start[:19]:<20} {st:<12} {tour:<4} {sc:<26} {title}")

    for start, st, tour, ev, title, tname, d, mil in live:
        print(f"\n{'='*70}\nLIVE: {title}   ({tname})\n  event {ev}")
        print(json.dumps(d, indent=1))
    print(f"\n[{feed.n_get} GETs, {feed.n_429} rate-limited]")


def _bind(feed, ev, tries=5):
    """(milestone, mapping, markets) with orientation resolved, or exit loudly.

    Both fetches RETRY. A single 429 returning an empty market list looks identical
    to a market that does not exist, and the orientation guard then refuses to run —
    killing the capture before it starts. That took out three Challenger launches on
    26AUG25. The guard is right to refuse on genuinely-absent markets; it just must
    not be handed an empty list by a transient rate limit.
    """
    mil = None
    for i in range(tries):
        mil = feed.milestone(ev)
        if mil:
            break
        time.sleep(2.0 * (i + 1))
    if not mil:
        sys.exit(f"no tennis milestone for {ev} after {tries} tries — wrong ticker, "
                 f"or Kalshi has no feed for it")
    mkts = []
    for i in range(tries):
        mkts = feed.markets(ev)
        if len(mkts) >= 2:
            break
        time.sleep(2.0 * (i + 1))
    try:
        mapping = kt.resolve_orientation(mil, mkts, mil.get("title") or "")
    except kt.OrientationError as e:
        sys.exit(f"ORIENTATION UNRESOLVED for {ev}: {e}\n"
                 f"Refusing to run — a wrong binding inverts every price.")
    return mil, mapping, mkts


def cmd_inspect(a):
    feed = kt.Feed(rps=a.rps)
    mil, mapping, mkts = _bind(feed, a.event)
    det_m = mil.get("details") or {}
    print(f"{mil.get('title')}   [{det_m.get('tour')}, {det_m.get('gender')}, "
          f"best of {det_m.get('best_of')}, {det_m.get('round')}, {det_m.get('tournament_name')}]")
    print(f"milestone {mil.get('id')}   status={det_m.get('status')}")
    print("\norientation:")
    for cid, tick in mapping.items():
        sub = next((m.get("yes_sub_title") for m in mkts if m.get("ticker") == tick), "?")
        print(f"  {cid}  ->  {tick:38s}  YES = {sub}")
    print("\nmarkets:")
    for m in mkts:
        b, k = kt.top_of_book(m)
        print(f"  {m.get('ticker'):38s} {str(m.get('yes_sub_title')):22s} "
              f"bid={b} ask={k} mid={kt.mid(m)}")
    live = feed.live_data([mil.get("id")]).get(mil.get("id")) or {}
    print(f"\nlive_data status={live.get('status')!r} server={live.get('server')!r} "
          f"advantage={live.get('advantage')!r} is_live={kt.is_live(live)}")
    ids = list(mapping)
    st = kt.model_state(live, ids[0], ids[1])
    print(f"model state (me={ids[0][:8]}…): {st}")
    print(f"\n[{feed.n_get} GETs, {feed.n_429} rate-limited]")


def cmd_watch(a):
    os.makedirs(TAPES, exist_ok=True)
    # REFUSE TO DOUBLE-CAPTURE. Two pollers on one event append to the same tape and
    # interleave their writes, so the file no longer represents a single time-ordered
    # observation stream and every downstream rebuild silently inherits the damage.
    # Hit 26AUG24 by re-running discovery over matches already being watched.
    # Match only PYTHON processes. `pgrep -f` also matches the transient `bash -c`
    # wrapper that nohup/& creates, whose command line contains this same string —
    # so the guard fired against its own launcher and blocked the capture entirely
    # (Parry/Vekic, 26AUG25). Check each pid's actual executable before counting it.
    import subprocess as _sp
    me_pid, me_ppid = str(os.getpid()), str(os.getppid())
    try:
        out = _sp.run(["pgrep", "-f", f"poll_tennis.py watch {a.event}"],
                      capture_output=True, text=True).stdout.split()
    except Exception:
        out = []
    others = []
    for pid in out:
        if pid in (me_pid, me_ppid):
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            exe = (cmd[0] or b"").decode(errors="replace")
        except Exception:
            continue
        if "python" in os.path.basename(exe).lower():
            others.append(pid)
    if others:
        sys.exit(f"ALREADY CAPTURING {a.event} (pid {' '.join(others)}) — refusing to "
                 f"start a second poller; it would interleave writes into the same tape.")
    feed = kt.Feed(rps=a.rps)
    mil, mapping, mkts = _bind(feed, a.event)
    det_m = mil.get("details") or {}
    mid_id = mil.get("id")

    # which competitor is "me"
    by_tick = {t: c for c, t in mapping.items()}
    me_id = None
    for tick, cid in by_tick.items():
        if tick.upper().endswith("-" + a.me.upper()):
            me_id = cid
    if me_id is None:
        sys.exit(f"--me {a.me!r} matched no market suffix; options: "
                 f"{[t.rsplit('-', 1)[-1] for t in by_tick]}")
    opp_id = next(c for c in mapping if c != me_id)
    me_tick, opp_tick = mapping[me_id], mapping[opp_id]

    verified = kt.best_of_from_exact_market(feed, a.event)
    try:
        best_of = kt.resolve_best_of(det_m, a.best_of, verified=verified)
    except ValueError as e:
        sys.exit(f"BEST-OF UNRESOLVED: {e}")
    if verified:
        print(f"best_of={best_of} VERIFIED against the exact-score market")
    prior = kt.resolve_split_prior(det_m, a.split_prior)
    final_tb = kt.resolve_final_set_tb(det_m, a.final_set_tb)

    print(f"watching {mil.get('title')}  [best of {best_of}, {det_m.get('tour')}, "
          f"split_prior {prior}, deciding-set tiebreak to {final_tb}]")
    print(f"  me  = {me_tick}")
    print(f"  opp = {opp_tick}")
    print(f"  interval {a.interval}s live / {a.idle_interval}s idle, rps cap {a.rps}")
    print(f"  kill: touch {kt.KILL_FLAG}\n")

    tape_p = os.path.join(TAPES, f"{a.event}.jsonl")
    log_p = os.path.join(TAPES, f"{a.event}.log.json")
    log = {"best_of": best_of, "first_server": "me", "split_prior": prior,
           "split_prior_sd": 0.30, "strict_split": False, "set_servers": {},
           "final_set_tb": final_tb,
           "obs": [], "meta": {"event": a.event, "title": mil.get("title"),
                               "me_ticker": me_tick, "opp_ticker": opp_tick,
                               "tour": det_m.get("tour"), "round": det_m.get("round"),
                               "tournament": det_m.get("tournament_name")}}
    model = ImpliedModel(best_of=best_of, first_server="me", split_prior=prior,
                         final_set_tb=final_tb)
    # STATIC keeps its view: it stops observing after STATIC_N boundaries, so a genuine
    # mispricing persists and stays tradeable instead of being absorbed into p/q at the
    # next refit. `model` above remains ROLLING and remains what the tape's existing
    # "model"/"edge_c" fields mean — downstream readers of old tapes are unaffected.
    static_model = ImpliedModel(best_of=best_of, first_server="me", split_prior=prior,
                                warn_split=False, final_set_tb=final_tb)
    # getattr, not a.ewma_halflives: cmd_watch is also driven programmatically with a
    # hand-built namespace (test_live_path.py), and a missing attribute must not be able
    # to kill a poller mid-match — a live capture cannot be re-run.
    halflives = list(getattr(a, "ewma_halflives", None) or [])
    ewma = {f"ewma{h}": ImpliedModel(best_of=best_of, first_server="me",
                                     split_prior=prior, warn_split=False,
                                     final_set_tb=final_tb)
            for h in halflives}
    last_key = None
    n_obs = 0
    cycles = 0
    misses = 0
    last_pts = None          # last observed (points_me, points_opp, games, sets)
    pt_seq = 0               # increments on every observed point-score change

    # Prior boundary prices observed before the poller attached (hand-read off the
    # board). Prepended in order so the fit starts from the real pre-match anchor
    # instead of whatever game the poller happened to join at.
    if a.seed:
        for o in json.load(open(a.seed)):
            log["obs"].append(o)
            model.observe(o["price"], **o["state"])
            n_obs += 1
            if n_obs <= STATIC_N:
                static_model.observe(o["price"], **o["state"])
            for h in halflives:
                _ewma_update(ewma[f"ewma{h}"], o["price"], o["state"], float(h))
        with open(log_p, "w") as f:
            json.dump(log, f, indent=1)
        print(f"seeded {n_obs} prior boundary price(s) from {a.seed}: "
              f"p={model.p:.3f} q={model.q:.3f}")
        for o in log["obs"]:
            s = o["state"]
            print(f"    {s['sets_me']}-{s['sets_opp']} {s['games_me']}-{s['games_opp']} "
                  f"srv={s.get('server','?')}  {o['price']}  {o.get('label','')}")

    try:
        while True:
            cycles += 1
            if a.max_cycles and cycles > a.max_cycles:
                print(f"\nreached --max-cycles {a.max_cycles}, stopping."); break
            if kt.disabled():
                print(f"\nkill flag {kt.KILL_FLAG} present — stopping."); break
            try:
                det = feed.live_data([mid_id]).get(mid_id)
            except kt.RateLimited as e:
                print(f"\n*** {e}\n*** stopping to protect the live trading system's "
                      f"GET bucket.", file=sys.stderr)
                break

            # A MISSING payload is a fetch failure, not a finished match. The
            # milestone is known to exist, so {} means the request died (Kalshi
            # ReadTimeouts were seen twice in 30 minutes on 26AUG24). Falling through
            # to the not-live branch would idle 60s and can black out a whole game.
            if det is None:
                misses += 1
                # BACK OFF ON CONSECUTIVE misses. The first retry stays fast, which is
                # the original point of this branch: an isolated ReadTimeout must not
                # black out a game. But a 429 ALSO returns an empty payload, so a fixed
                # fast retry turns every waiting poller into a fast poller exactly when
                # the bucket is busiest, and drives more 429s. Measured 26AUG25: 786 such
                # retries across 61 pollers, 11.5 GET/s against an expected 5.5, and by
                # 18:15 the bucket was so contended that a board scan could not complete
                # at all (4 GETs, 4 rate-limited). Doubling per consecutive miss keeps
                # the isolated-failure protection and stops the feedback loop.
                back = min(a.interval * (2 ** (misses - 1)), 60.0)
                print(f"\n[{time.strftime('%H:%M:%S')}] live_data returned nothing "
                      f"(miss {misses}) — retrying in {back:.0f}s.", file=sys.stderr)
                time.sleep(back)
                continue
            misses = 0

            status = (det.get("status") or "").strip().lower()
            if not kt.is_live(det):
                if det.get("winner"):
                    print(f"match over (status={status}, winner={det['winner'][:8]}…)")
                    break
                # PRE-MATCH PRICE CAPTURE. How far the price wanders before a ball is
                # struck measures how settled the market is on (p, q) — and that turns
                # out to predict whether a fixed-(p, q) fit can track the match at all
                # (26AUG24: 6c pre-match drift -> 6-8c in-play error; 0c -> 2.5c).
                # Without this the idle branch fetches no price and writes no tape row,
                # so attaching early captures NOTHING until Kalshi flips the live flag —
                # which it does at its own whim (38 minutes before play on one match,
                # not at all until first ball on another).
                if a.pregame_interval > 0:
                    mkts = feed.markets(a.event)
                    mm = {m.get("ticker"): m for m in mkts}
                    p_me = kt.mid(mm.get(me_tick) or {})
                    p_opp = kt.mid(mm.get(opp_tick) or {})
                    px_pre = kt.vig_free(p_me, p_opp)
                    pb_me, pa_me = kt.top_of_book(mm.get(me_tick) or {})
                    pb_opp, pa_opp = kt.top_of_book(mm.get(opp_tick) or {})
                    with open(tape_p, "a") as f:
                        # models/edges_c empty rather than absent: there is no fit yet
                        # (no boundaries have happened), but keeping the key present on
                        # every row means tomorrow's analysis can read tape["models"]
                        # unconditionally instead of guarding each row.
                        f.write(json.dumps({"ts": _now(), "status": status,
                                            "state": kt.model_state(det, me_id, opp_id) or {},
                                            "mid_me": p_me, "mid_opp": p_opp,
                                            "vig_free": px_pre, "model": None,
                                            "edge_c": None, "models": {}, "edges_c": {},
                                            "book": {"bid_me": pb_me, "ask_me": pa_me,
                                                     "bid_opp": pb_opp, "ask_opp": pa_opp},
                                            "details": det}) + "\n")
                    print(f"\r[{time.strftime('%H:%M:%S')}] pre-match {status or '?'} "
                          f"mkt={px_pre if px_pre is None else round(px_pre, 3)} "
                          f"({feed.n_get}g/{feed.n_429}x) ", end="", flush=True)
                    time.sleep(a.pregame_interval)
                else:
                    print(f"\r[{time.strftime('%H:%M:%S')}] status={status or 'unknown'} — "
                          f"idling {a.idle_interval}s ", end="", flush=True)
                    time.sleep(a.idle_interval)
                continue

            st = kt.model_state(det, me_id, opp_id)
            if st is None:
                print("\ncompetitor ids in live_data do not match the milestone — "
                      "stopping rather than logging a wrong state.", file=sys.stderr)
                break

            mkts = feed.markets(a.event)
            mm = {m.get("ticker"): m for m in mkts}
            mid_me, mid_opp = kt.mid(mm.get(me_tick) or {}), kt.mid(mm.get(opp_tick) or {})
            px = kt.vig_free(mid_me, mid_opp)
            # RAW BOOK. Until 26AUG26 only the derived mids were kept, so every backtest
            # filled at the vig-free midpoint — a price nobody can actually trade — while
            # still charging the real Kalshi fee. That is optimistic on entry and
            # realistic on cost, and it makes every P&L number an upper bound: the
            # 107-match review could not even size the error, because reconstructing the
            # spread from two mids that sum to 1 by construction is impossible.
            # Storing top of book lets a fill be modelled where it really happens —
            # buy at the ask, sell at the bid.
            bid_me, ask_me = kt.top_of_book(mm.get(me_tick) or {})
            bid_opp, ask_opp = kt.top_of_book(mm.get(opp_tick) or {})

            # Once the fit has something to say, price the FULL state (point score
            # included) at every poll. This is the mid-game comparison the hand-logged
            # match could only sample a few times, and where the handoff's
            # "market runs ~1 point ahead of the scoreboard" claim gets tested.
            # Detect a point-score change. The scoreboard runs behind the fastest
            # participants, so the interesting quantity is WHEN we first see a new score
            # relative to when the market moved - pt_new marks that first sighting.
            pt_new = False
            if st.get("_points_known"):
                sig = (st.get("sets_me"), st.get("sets_opp"), st.get("games_me"),
                       st.get("games_opp"), st.get("points_me"), st.get("points_opp"))
                if last_pts is not None and sig != last_pts:
                    pt_seq += 1
                    pt_new = True
                last_pts = sig

            model_px = None
            var_px = {}
            ahead = {}
            if n_obs >= 2 and st.get("_points_known"):
                pstate = {k: v for k, v in st.items() if not k.startswith("_")}
                # Price every variant on the SAME state at the SAME tick, so the tape
                # carries a like-for-like comparison rather than four series that each
                # sampled the match at slightly different moments. A variant with no
                # fit yet (p is None) is skipped rather than recorded as a guess.
                for tag, m in ([("static", static_model), ("rolling", model)]
                               + sorted(ewma.items())):
                    if m is None or m.p is None:
                        continue
                    try:
                        var_px[tag] = m.price(**pstate)[0]
                    except Exception:
                        pass
                model_px = var_px.get("rolling")
                ahead = _point_ahead_block(
                    [("static", static_model), ("rolling", model)] + sorted(ewma.items()),
                    pstate, st.get("server"))

            rec = {"ts": _now(), "status": status, "state": st, "mid_me": mid_me,
                   "mid_opp": mid_opp, "vig_free": px, "model": model_px,
                   "edge_c": (None if (model_px is None or px is None)
                              else round((px - model_px) * 100, 2)),
                   # "model"/"edge_c" stay ROLLING so every existing tape reader keeps
                   # working unchanged; the per-variant detail is additive.
                   "models": {k: round(v, 4) for k, v in var_px.items()},
                   "edges_c": ({} if px is None else
                               {k: round((px - v) * 100, 2) for k, v in var_px.items()}),
                   # POINT-LEVEL. pt_seq increments on every observed change of the point
                   # score, so transitions are findable without re-deriving them; pt_new
                   # marks the first poll that saw the new score, which is what the lag
                   # measurement needs. `ahead` carries each variant priced one point
                   # forward - see _point_ahead_block for why.
                   # mkt_clock: 0 = market prices the point as LOST, 1 = as WON,
                   # ~p_point = market has not seen it either. The lag in one number.
                   "pt_seq": pt_seq, "pt_new": pt_new, "ahead": ahead,
                   "mkt_clock": _market_clock(ahead, px),
                   # raw top of book, both sides — what a fill would actually pay
                   "book": {"bid_me": bid_me, "ask_me": ask_me,
                            "bid_opp": bid_opp, "ask_opp": ask_opp},
                   "details": det}
            with open(tape_p, "a") as f:
                f.write(json.dumps(rec) + "\n")

            key = kt.boundary_key(st)
            if px is not None and key != last_key:
                # a game (or set) just completed -> a fittable, vig-free boundary price
                if last_key is not None or a.log_first:
                    # Fit on GAME BOUNDARIES only: drop the point score, which is
                    # 0-0 at the instant a game completes anyway.
                    obs_state = {k: v for k, v in st.items()
                                 if k not in ("points_me", "points_opp")
                                 and not k.startswith("_")}
                    log["obs"].append({"state": obs_state, "price": round(px, 4),
                                       "label": f"auto {time.strftime('%H:%M:%S')}"})
                    with open(log_p, "w") as f:
                        json.dump(log, f, indent=1)
                    n_obs += 1
                    # OUT-OF-SAMPLE: every variant prices this boundary BEFORE it is
                    # allowed to observe it. Collected first, for all variants, so the
                    # errors printed below are directly comparable.
                    preds = {}
                    for tag, m in ([("static", static_model), ("rolling", model)]
                                   + sorted(ewma.items())):
                        if m is None or m.p is None:
                            continue
                        try:
                            preds[tag] = m.price(**obs_state)[0]
                        except Exception:
                            pass
                    pred = preds.get("rolling") if n_obs > 1 else None
                    t_fit = time.time()
                    try:
                        model.observe(px, **obs_state)
                    except ImpliedSplitError as e:
                        print(f"\n*** strict split: {e}", file=sys.stderr)
                    if n_obs <= STATIC_N:
                        try:
                            static_model.observe(px, **obs_state)
                        except ImpliedSplitError:
                            pass          # rolling already reported it; don't double-warn
                    for h in halflives:
                        try:
                            _ewma_update(ewma[f"ewma{h}"], px, obs_state, float(h))
                        except ImpliedSplitError:
                            pass          # rolling already reported it; don't double-warn
                    fit_ms = (time.time() - t_fit) * 1000
                    err = f"{(px - pred) * 100:+5.1f}c" if pred is not None else "    -"
                    flag = " !p<=q" if model.split_violation else ""
                    print(f"\n[{time.strftime('%H:%M:%S')}] "
                          f"{st['sets_me']}-{st['sets_opp']} sets  "
                          f"{st['games_me']}-{st['games_opp']} games  "
                          f"srv={st.get('server', '?'):3s}  mkt {px:.3f}  "
                          f"pred {pred if pred is None else round(pred, 3)}  err {err}  "
                          f"| p={model.p:.3f} q={model.q:.3f}{flag}  [refit {fit_ms:.0f}ms]")
                    # Per-variant out-of-sample error at this boundary. p/q shown are
                    # POST-update, matching what the rolling line above reports.
                    cur = {"static": static_model, "rolling": model, **ewma}
                    for tag in ["static", "rolling"] + sorted(ewma):
                        if tag not in preds:
                            continue
                        mv = cur[tag]
                        print(f"      {tag:9s} pred {preds[tag]:.3f}  "
                              f"err {(px - preds[tag]) * 100:+5.1f}c   "
                              f"p={mv.p:.3f} q={mv.q:.3f}")
                last_key = key
            else:
                lbl = ['0', '15', '30', '40', 'AD']
                pmm, poo = st['points_me'], st['points_opp']
                in_tb = st['games_me'] == 6 and st['games_opp'] == 6
                pts = (f"{pmm}-{poo}" if in_tb else
                       (f"{lbl[pmm]}-{lbl[poo]}" if st.get('_points_known')
                        and pmm < 5 and poo < 5 else "?"))
                em = "" if rec["edge_c"] is None else f" edge {rec['edge_c']:+.1f}c"
                print(f"\r[{time.strftime('%H:%M:%S')}] "
                      f"{st['sets_me']}-{st['sets_opp']} {st['games_me']}-{st['games_opp']} "
                      f"{pts:>7s} srv={st.get('server','?'):3s} "
                      f"mkt={px if px is None else round(px, 3)} "
                      f"mdl={model_px if model_px is None else round(model_px, 3)}{em} "
                      f"({feed.n_get}g/{feed.n_429}x) ", end="", flush=True)

            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\ninterrupted.")

    print(f"\n{n_obs} boundary observations -> {log_p}")
    print(f"tape -> {tape_p}")
    print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")
    if n_obs >= 2:
        print(f"\nfinal fit: p={model.p:.3f} q={model.q:.3f} p+q={model.p + model.q:.3f}")
        print(f"report with:  cp {log_p} live_log.json && python3 live.py report")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=2.0, help="hard GET/sec ceiling")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list"); p.add_argument("--day", help="YYMMDD, e.g. 26AUG24")
    p = sub.add_parser("scan", help="resolve today's matches and show which are live")
    p.add_argument("--day", help="YYMMDD, e.g. 26AUG24")
    p.add_argument("--limit", type=int, default=40,
                   help="max events to resolve (1 GET each — milestones has no bulk filter)")
    p = sub.add_parser("inspect"); p.add_argument("event")
    p = sub.add_parser("watch")
    p.add_argument("event")
    p.add_argument("--me", required=True, help="market ticker suffix for the tracked player, e.g. ALT")
    p.add_argument("--best-of", type=int, choices=(3, 5),
                   help="REQUIRED when Kalshi's best_of is absent or contradicts the format "
                        "(it reports 3 for US Open men's, which is best-of-5)")
    p.add_argument("--split-prior", type=float, help="prior mean for p-q (default 0.28 men / 0.14 women)")
    p.add_argument("--final-set-tb", type=int, choices=(7, 10),
                   help="deciding-set tiebreak length (default: 10 at Grand Slams, 7 elsewhere)")
    p.add_argument("--ewma-halflives", nargs="*", default=["2", "4"], metavar="H",
                   help="EWMA halflives (in boundaries) carried live alongside static and "
                        "rolling; pass nothing to disable. Default 2 4 — h4 is top-2 on both "
                        "rms and P&L, h2 is the rms winner; h8 was dropped because it "
                        "duplicated h4 exactly on 4 of 10 matches. See per_match_pnl.py")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--idle-interval", type=float, default=60.0)
    p.add_argument("--pregame-interval", type=float, default=60.0,
                   help="seconds between PRE-MATCH price samples (0 disables). The "
                        "pre-match price path is what pregame_stability.py screens on, "
                        "so attach early — 2h out at 60s costs ~240 GETs per match.")
    p.add_argument("--no-log-first", dest="log_first", action="store_false", default=True,
                   help="skip the state you join at. Default is to log it: joining at "
                        "0-0 before play captures the pre-match anchor, which the fit wants.")
    p.add_argument("--seed", help="JSON list of prior boundary observations "
                                  "[{state:{...}, price: float, label: str}] to prepend")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="stop after N poll cycles (0 = run until the match ends). "
                        "A bounded run is the safe way to smoke-test against the shared bucket.")

    a = ap.parse_args()
    {"list": cmd_list, "scan": cmd_scan, "inspect": cmd_inspect,
     "watch": cmd_watch}[a.cmd](a)
