"""Find upcoming (not-yet-started) matches in named tournaments, ready to capture.

    python3 discover.py --tournament "US Open" "Winston Salem" "Monterrey" [--launch]

Only NOT-STARTED matches are returned: the pre-match price path is what
pregame_stability.py screens on, and a match already in progress can never supply
one. Attaching early is the whole point — Kalshi flips matches live at its own whim
(38 minutes before play on one match today, not until first ball on another), so the
only reliable way to get the window is to be watching before it opens.

Milestone ids must be resolved one event at a time (the /milestones collection
ignores every filter except related_event_ticker), so discovery costs one GET per
event. That is the expensive part; it is done once.
"""
import argparse
import datetime as dt
import subprocess
import sys
import time
import os

import kalshi_tennis as kt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tournament", nargs="+", required=True)
    ap.add_argument("--day", nargs="*", default=None, help="YYMMDD filters, e.g. 26AUG24 26AUG25")
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--within-hours", type=float, default=None,
                    help="only matches scheduled to start within this many hours "
                         "(negative lower bound of -1h allows for late starts)")
    ap.add_argument("--include-started", action="store_true",
                    help="also capture matches ALREADY IN PROGRESS. They cannot supply a "
                         "pre-match screen, but they still yield boundary prices and a "
                         "full static-vs-rolling comparison (Poljicak was attached at 1-1 "
                         "and gave 17 boundaries). Use when the schedule is the binding "
                         "constraint rather than the rate budget.")
    ap.add_argument("--launch", action="store_true", help="start a capture for each")
    ap.add_argument("--best-of", type=int, choices=(3, 5), default=None,
                    help="force best_of. REQUIRED for ATP/WTA Challenger: those events "
                         "carry no related markets at all (no exact-score, no set-winner) "
                         "and no best_of field, so it cannot be verified from Kalshi. "
                         "Challenger singles is best-of-3 (confirmed 26AUG24 on "
                         "Poljicak/Schoenhaus, which finished 6-4 6-3).")
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--pregame-interval", type=float, default=60.0)
    a = ap.parse_args()

    feed = kt.Feed(rps=a.rps, verbose=False)
    # The ticker date is NOT the play date. A match postponed across midnight keeps
    # its original ticker: KXWTAMATCH-26AUG23PARVEK actually started 26AUG25 03:10Z.
    # Filtering on the ticker silently hides every rescheduled match, and discovery
    # reports a clean count while having skipped them. When --within-hours is given,
    # scan ALL open events and let the milestone's start_date do the filtering.
    evs = []
    if a.within_hours is not None and not a.day:
        evs = feed.matches()
    else:
        for d in (a.day or [None]):
            evs += feed.matches(day=d)
    seen, uniq = set(), []
    for e in evs:
        if e["event_ticker"] not in seen:
            seen.add(e["event_ticker"])
            uniq.append(e)
    print(f"{len(uniq)} open match events; resolving milestones at {a.rps} rps...",
          file=sys.stderr)

    want = [t.lower() for t in a.tournament]
    hits = []
    for i, e in enumerate(uniq[:a.limit]):
        mil = feed.milestone(e["event_ticker"])
        if not mil:
            continue
        det = mil.get("details") or {}
        tname = (det.get("tournament_name") or "")
        if not any(w in tname.lower() for w in want):
            continue
        hits.append((mil.get("start_date") or "", e, mil, tname, det))
        if (i + 1) % 40 == 0:
            print(f"  ...{i+1}/{len(uniq)} scanned, {len(hits)} matched", file=sys.stderr)

    # live_data in one batch to learn which have already started
    det_by = feed.live_data([m.get("id") for _, _, m, _, _ in hits])
    rows = []
    for start, e, mil, tname, det in hits:
        d = det_by.get(mil.get("id")) or {}
        status = (d.get("status") or "?")
        ms = d.get("match_status")
        started = kt.is_live(d) and ms not in (None, "", "match_about_to_start")
        rows.append((start, e, mil, tname, det, status, ms, started))
    if a.within_hours is not None:
        now = dt.datetime.now(dt.timezone.utc)
        keep = []
        for r in rows:
            try:
                t = dt.datetime.fromisoformat((r[0] or "").replace("Z", "+00:00"))
            except Exception:
                continue
            hrs = (t - now).total_seconds() / 3600
            # a scheduled start up to an hour past is still capturable — today's
            # matches ran 30+ minutes late and Kalshi does not update start_date
            if -1.0 <= hrs <= a.within_hours:
                keep.append(r)
        rows = keep
    rows.sort(key=lambda r: r[0])

    print(f"\n{len(rows)} matches in {a.tournament}\n")
    print(f"{'start (UTC)':<18} {'status':<12} {'started':<8} {'tournament':<26} match")
    upcoming = []
    for start, e, mil, tname, det, status, ms, started in rows:
        mark = "PLAYING" if started else "not yet"
        print(f"{start[11:19]:<18} {status:<12} {mark:<8} {tname[:26]:<26} {mil.get('title')}")
        if (not started) or a.include_started:
            upcoming.append((e, mil, det))

    n_fresh = sum(1 for r in rows if not r[7])
    print(f"\n{n_fresh} not yet started (full pre-match window); "
          f"{len(rows)-n_fresh} already playing (no screen, still capturable)")
    print(f"{len(upcoming)} will be launched")
    print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")

    if not a.launch or not upcoming:
        if upcoming and not a.launch:
            print("\nre-run with --launch to start captures")
        return

    print()
    for e, mil, det in upcoming:
        ev = e["event_ticker"]
        # track the UNDERDOG for consistency with every capture so far
        mkts = feed.markets(ev)
        mids = {m["ticker"]: kt.mid(m) for m in mkts if m.get("ticker")}
        mids = {k: v for k, v in mids.items() if v is not None}
        if len(mids) != 2:
            print(f"  SKIP {ev}: {len(mids)} priced markets")
            continue
        under = min(mids, key=mids.get)
        suffix = under.rsplit("-", 1)[-1]
        # Resolve best_of HERE, with retries. The poller refuses to guess (correctly —
        # a wrong best_of misprices an entire match), so a transient 429 during its
        # exact-score-market lookup kills the launch outright. That took out all 16
        # Challenger captures on 26AUG25: the milestone carries no best_of for that
        # tour, so the market lookup is the only source and it must not be one-shot.
        bo = a.best_of
        for _ in range(0 if bo else 4):
            bo = kt.best_of_from_exact_market(feed, ev)
            if bo:
                break
            time.sleep(2.0)
        if not bo:
            try:
                bo = kt.resolve_best_of(det, None)
            except ValueError:
                print(f"  SKIP {ev}: best_of unresolved after retries "
                      f"({det.get('tournament_name')}) — pass --best-of to force")
                continue
        log = os.path.join(HERE, "tapes", f"watch_{ev.rsplit('-',1)[-1].lower()}.log")
        cmd = [sys.executable, "-u", os.path.join(HERE, "poll_tennis.py"), "watch", ev,
               "--me", suffix, "--interval", str(a.interval),
               "--pregame-interval", str(a.pregame_interval), "--max-cycles", "4000",
               "--best-of", str(bo)]
        with open(log, "w") as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE,
                             start_new_session=True)
        print(f"  launched {ev}  --me {suffix} (underdog @ {mids[under]:.3f}, bo{bo})")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
