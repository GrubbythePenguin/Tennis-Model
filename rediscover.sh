#!/bin/bash
# Rolling rediscovery, for unattended overnight capture.
#
# discover.py only launches matches inside --within-hours, so a single run covers one
# window and nothing else. On 26AUG25 that meant 39 matches captured and 76 within the
# next 24h missed entirely, the first only 6.6h later. This re-runs discovery on a
# cycle so the horizon rolls forward while nobody is watching.
#
# Safe to re-run: discover.py skips any event that already has a live poller, so this
# only ever adds newly-in-window matches.
#
# --include-started is REQUIRED here. discover.py defaults to not-yet-started matches
# only, so any match that went live between cycles is skipped FOREVER - 6 of 21 live
# matches were uncaptured at 18:15 on 26AUG25 for exactly this reason. An in-progress
# match loses the pre-match window but still yields boundaries, and boundaries are the
# binding constraint on every estimate we care about.
#
# Two runs per cycle because --best-of is applied GLOBALLY. Challenger events carry no
# exact-score market and no best_of on the milestone, so they are skipped outright
# unless it is forced; forcing it on the rest would misprice any best-of-5.
#
# Runs every 90 min rather than hourly-ish because the scan is FRAGILE under load:
# with 39 pollers competing for the bucket, a milestone scan observed 3 rate-limits
# and found 1 of 25 matches instead of erroring. A short cycle is the mitigation —
# a degraded pass costs little and the next one picks up what it missed.
#
#   ./rediscover.sh [interval_seconds]     default 90 min
cd "$(dirname "$0")" || exit 1
LOG=tapes/rediscover.log
MAX_FLEET=45          # memory is fine (~55 MB each); this bounds the shared GET bucket.
                      # Passed to discover.py as --max-launch so it bounds the RESULT:
                      # checking only before a cycle let 39 pollers become 54 on 26AUG25.
INTERVAL=${1:-5400}

stamp() { date -u +%H:%M:%S; }
fleet() { ps -eo args | grep -c "[p]oll_tennis.py watch "; }

echo "$(stamp) rediscover.sh started, interval ${INTERVAL}s, max fleet $MAX_FLEET" >> "$LOG"
while true; do
  sleep "$INTERVAL"
  if [ -f disable_tennis_poller.flag ]; then
    echo "$(stamp) kill flag present - not launching" >> "$LOG"
    continue
  fi
  n=$(fleet)
  if [ "$n" -ge "$MAX_FLEET" ]; then
    echo "$(stamp) fleet=$n at cap $MAX_FLEET - skipping cycle" >> "$LOG"
    continue
  fi
  budget=$(( MAX_FLEET - n ))
  echo "$(stamp) rediscovery starting, fleet=$n budget=$budget" >> "$LOG"
  # trim pollers idling on matches still hours out before adding more
  python3 shed_pregame.py --hours 2 --apply >> "$LOG" 2>&1
  # gentler rps than the interactive default: this competes with the trading system
  # and with every poller already running
  python3 discover.py --tournament "Challenger" "125K" --within-hours 2 --include-started \
          --best-of 3 --interval 10 --rps 3 --max-launch "$budget" --launch >> "$LOG" 2>&1
  # recompute: the challenger run above may have consumed part of the budget, and a
  # budget spent twice does not bound anything
  budget=$(( MAX_FLEET - $(fleet) ))
  [ "$budget" -lt 0 ] && budget=0
  python3 discover.py --tournament "US Open" "Winston Salem" "Monterrey" \
          --within-hours 2 --include-started --interval 10 --rps 3 --max-launch "$budget" --launch >> "$LOG" 2>&1
  echo "$(stamp) rediscovery done, fleet=$(fleet)" >> "$LOG"
done
