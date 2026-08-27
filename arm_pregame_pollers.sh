#!/bin/bash
# Attach a poller to every tennis match BEFORE it starts, on a rolling basis.
#
#   ./arm_pregame_pollers.sh [lookahead_min] [cycle_sec]
#
# WHY PRE-MATCH ATTACHMENT IS NOT OPTIONAL. ImpliedModel fits p and q to observed
# market prices at game boundaries. A poller attached mid-match has 1-3 boundaries
# behind its fit, and the fit is then just the current market price with an
# unconstrained bracket around it. Measured 26AUG27, same code, same night:
#
#     attached mid-match   SACLLA  bracket 24.88c     LIURAD 14.23c
#     attached pre-match   CHWPAR  bracket  1.0-3.0c throughout
#
# The mid-attached ones quoted and traded 30 fills for a realised -$90.92 on prices
# that had no basis. So this exists to make full-coverage the default rather than
# something the operator has to remember at the right moment.
#
# Idempotent: skips any event that already has a poller, so it can run on a loop.
# Honours the kill flag. Only attaches to NOT-YET-STARTED matches - a match already in
# progress is deliberately left alone, because a late attach is the failure mode above.
cd "$(dirname "$0")" || exit 1
LOOKAHEAD=${1:-90}
CYCLE=${2:-600}
LOG=tapes/arm_pregame.log

while true; do
    if [ -f disable_tennis_poller.flag ]; then
        echo "$(date -u '+%F %T')Z kill flag — not attaching" >> "$LOG"
        sleep "$CYCLE"; continue
    fi
    python3 - "$LOOKAHEAD" >> "$LOG" 2>&1 <<'PY'
import sys, os, subprocess, datetime as dt, time
sys.path.insert(0, os.getcwd())
import kalshi_tennis as kt

lookahead = float(sys.argv[1])
running = set()
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    try:
        if open(f"/proc/{p}/comm").read().strip() != "python3":
            continue
        c = open(f"/proc/{p}/cmdline", "rb").read().decode("utf-8", "replace").replace("\0", " ")
    except Exception:
        continue
    if "poll_tennis.py" in c and " watch " in c:
        running.add(c.split(" watch ", 1)[1].split()[0])

feed = kt.Feed(rps=3.0, verbose=False)
now = dt.datetime.now(dt.timezone.utc)
n = 0
for e in feed.matches():
    ev = e["event_ticker"]
    if ev in running:
        continue
    mil = feed.milestone(ev)
    if not mil:
        continue
    det = mil.get("details") or {}
    try:
        t = dt.datetime.fromisoformat((mil.get("start_date") or "").replace("Z", "+00:00"))
        mins = (t - now).total_seconds() / 60
    except Exception:
        continue
    if not (0 < mins <= lookahead):
        continue
    d = feed.live_data([mil.get("id")]).get(mil.get("id")) or {}
    if kt.is_live(d):
        continue                      # already started: a late attach is worse than none
    mkts = feed.markets(ev)
    pr = {m["ticker"]: kt.mid(m) for m in mkts if m.get("ticker") and kt.mid(m) is not None}
    if len(pr) != 2:
        continue
    me = min(pr, key=pr.get)
    bo = None
    try:
        bo = kt.best_of_from_exact_market(feed, ev)
    except Exception:
        pass
    try:
        bo = kt.resolve_best_of(det, None, bo)
    except ValueError:
        # Challenger events carry no exact-score market and no milestone best_of;
        # Challenger and all WTA singles are best-of-3.
        bo = 3
    log = open(f"tapes/deep_{ev.rsplit('-',1)[-1].lower()}.log", "a")
    subprocess.Popen([sys.executable, "-u", "poll_tennis.py", "--rps", "3", "watch", ev,
                      "--me", me.rsplit("-", 1)[-1], "--best-of", str(bo),
                      "--interval", "3", "--pregame-interval", "20",
                      "--max-cycles", "20000"],
                     stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd(),
                     start_new_session=True,
                     env=dict(os.environ, TENNIS_CACHE="25000"))
    n += 1
    print(f"{dt.datetime.now(dt.timezone.utc):%F %T}Z attached {ev} --me "
          f"{me.rsplit('-',1)[-1]} bo{bo} (+{mins:.0f}m)")
    time.sleep(0.3)
if n == 0:
    print(f"{dt.datetime.now(dt.timezone.utc):%F %T}Z nothing new within {lookahead:.0f}m "
          f"({len(running)} already attached)")
PY
    sleep "$CYCLE"
done
