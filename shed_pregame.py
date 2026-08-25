"""Stop pollers sitting in pre-match on matches that are still hours away.

    python3 shed_pregame.py --hours 2 [--apply]     (default: dry run)

WHY. A poller attached hours early spends that time idling, but idling is not free
under rate-limit pressure: poll_tennis treats an empty live_data payload as a fetch
failure and retries at min(interval*2, 10)s instead of the 60s pre-match cadence —
correct for an isolated miss, but a 429 also returns an empty payload, so sustained
rate limiting turns every waiting poller into a 10s poller and drives more 429s.
Measured 26AUG25: 786 such retries across 61 pollers, 11.5 GET/s against an expected
5.5, 23% of requests rejected.

Shedding the far-out ones cuts the amplifier. rediscover.sh re-launches them as their
start time approaches, so the cost is a shorter pre-match window, not a lost capture.

SAFETY. Two independent conditions before anything is killed:
  1. the milestone says the match starts more than --hours from now, AND
  2. the poller's own log tail still shows pre-match, not an in-play boundary line
A match in progress is never touched, even if its scheduled start says otherwise —
today's matches ran 30+ minutes late and Kalshi does not update start_date.
"""
import argparse
import datetime as dt
import os
import re
import signal
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "poll_tennis.py watch "


def running_pollers():
    """event_ticker -> pid, for live poll_tennis processes only."""
    out = {}
    ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for ln in ps.splitlines():
        if MARK not in ln:
            continue
        m = re.match(r"\s*(\d+)\s+\S*python3?\b", ln)
        if not m:
            continue
        rest = ln.split(MARK, 1)[1].split()
        if rest and rest[0].startswith("KX"):
            out[rest[0]] = int(m.group(1))
    return out


def is_pregame(ev):
    """True only if the poller's log tail shows pre-match and no in-play boundary."""
    log = os.path.join(HERE, "tapes", f"watch_{ev.rsplit('-', 1)[-1].lower()}.log")
    if not os.path.exists(log):
        return False                      # unknown state -> do not touch
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 3000))
            tail = f.read().decode("utf-8", "replace").replace("\r", "\n")
    except Exception:
        return False
    lines = [l for l in tail.splitlines() if l.strip()][-6:]
    if any("sets  " in l for l in lines):
        return False                      # an in-play boundary printed -> match is live
    return any("pre-match" in l for l in lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0,
                    help="shed pollers whose match starts more than this many hours out")
    ap.add_argument("--apply", action="store_true", help="actually kill (default: dry run)")
    ap.add_argument("--rps", type=float, default=3.0)
    a = ap.parse_args()

    pollers = running_pollers()
    if not pollers:
        print("no pollers running")
        return
    feed = kt.Feed(rps=a.rps, verbose=False)
    now = dt.datetime.now(dt.timezone.utc)
    shed, kept, unknown = [], 0, 0

    for ev, pid in sorted(pollers.items()):
        mil = feed.milestone(ev)
        if not mil:
            unknown += 1                  # could not confirm -> leave it alone
            continue
        try:
            t = dt.datetime.fromisoformat((mil.get("start_date") or "").replace("Z", "+00:00"))
        except Exception:
            unknown += 1
            continue
        hrs = (t - now).total_seconds() / 3600
        if hrs > a.hours and is_pregame(ev):
            shed.append((hrs, ev, pid))
        else:
            kept += 1

    shed.sort(reverse=True)
    print(f"{len(pollers)} pollers: {len(shed)} shed, {kept} kept, {unknown} unresolved "
          f"(threshold {a.hours:.1f}h)")
    for hrs, ev, pid in shed:
        print(f"  {'KILL' if a.apply else 'would kill'} pid {pid:>7}  +{hrs:5.1f}h  {ev}")
        if a.apply:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")
    if shed and not a.apply:
        print("\nre-run with --apply to actually stop them")


if __name__ == "__main__":
    main()
