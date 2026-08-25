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
    last_key = None
    n_obs = 0
    cycles = 0
    misses = 0

    # Prior boundary prices observed before the poller attached (hand-read off the
    # board). Prepended in order so the fit starts from the real pre-match anchor
    # instead of whatever game the poller happened to join at.
    if a.seed:
        for o in json.load(open(a.seed)):
            log["obs"].append(o)
            model.observe(o["price"], **o["state"])
            n_obs += 1
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
                print(f"\n[{time.strftime('%H:%M:%S')}] live_data returned nothing "
                      f"(miss {misses}) — retrying at the live cadence, NOT idling.",
                      file=sys.stderr)
                time.sleep(min(a.interval * 2, 10.0))
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
                    with open(tape_p, "a") as f:
                        f.write(json.dumps({"ts": _now(), "status": status,
                                            "state": kt.model_state(det, me_id, opp_id) or {},
                                            "mid_me": p_me, "mid_opp": p_opp,
                                            "vig_free": px_pre, "model": None,
                                            "edge_c": None, "details": det}) + "\n")
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

            # Once the fit has something to say, price the FULL state (point score
            # included) at every poll. This is the mid-game comparison the hand-logged
            # match could only sample a few times, and where the handoff's
            # "market runs ~1 point ahead of the scoreboard" claim gets tested.
            model_px = None
            if n_obs >= 2 and st.get("_points_known"):
                try:
                    model_px, _ = model.price(**{k: v for k, v in st.items()
                                                 if not k.startswith("_")})
                except Exception:
                    model_px = None

            rec = {"ts": _now(), "status": status, "state": st, "mid_me": mid_me,
                   "mid_opp": mid_opp, "vig_free": px, "model": model_px,
                   "edge_c": (None if (model_px is None or px is None)
                              else round((px - model_px) * 100, 2)),
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
                    pred = None
                    if n_obs > 1:
                        try:
                            pred, _ = model.price(**obs_state)
                        except Exception:
                            pred = None
                    try:
                        model.observe(px, **obs_state)
                    except ImpliedSplitError as e:
                        print(f"\n*** strict split: {e}", file=sys.stderr)
                    err = f"{(px - pred) * 100:+5.1f}c" if pred is not None else "    -"
                    flag = " !p<=q" if model.split_violation else ""
                    print(f"\n[{time.strftime('%H:%M:%S')}] "
                          f"{st['sets_me']}-{st['sets_opp']} sets  "
                          f"{st['games_me']}-{st['games_opp']} games  "
                          f"srv={st.get('server', '?'):3s}  mkt {px:.3f}  "
                          f"pred {pred if pred is None else round(pred, 3)}  err {err}  "
                          f"| p={model.p:.3f} q={model.q:.3f}{flag}")
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
