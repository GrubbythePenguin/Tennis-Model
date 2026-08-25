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
MAX_FLEET=45          # memory is fine (~55 MB each); this bounds the shared GET bucket
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
  echo "$(stamp) rediscovery starting, fleet=$n" >> "$LOG"
  # gentler rps than the interactive default: this competes with the trading system
  # and with every poller already running
  python3 discover.py --tournament "Challenger" "125K" --within-hours 6 \
          --best-of 3 --interval 10 --rps 3 --launch >> "$LOG" 2>&1
  python3 discover.py --tournament "US Open" "Winston Salem" "Monterrey" \
          --within-hours 6 --interval 10 --rps 3 --launch >> "$LOG" 2>&1
  echo "$(stamp) rediscovery done, fleet=$(fleet)" >> "$LOG"
done
