#!/bin/bash
# Fleet snapshot every 5 minutes, appended to tapes/fleet_health.log.
#
# An overnight capture runs with nobody watching, so without this there is no way to
# tell afterwards whether pollers died, memory climbed, or the GET bucket shared with
# the esports trading system saturated. Cheap enough to leave running all night.
#
# Counts only logs touched in the last 15 minutes, so captures that finished earlier
# in the day are not mistaken for live ones.
cd "$(dirname "$0")" || exit 1
LOG=tapes/fleet_health.log
while true; do
  now=$(date +%s)
  n=$(ps -eo args | grep -c "[p]ython3 -u .*poll_tennis.py watch")
  mb=$(ps -eo rss,args | grep "[p]oll_tennis.py watch" | awk '{s+=$1} END {printf "%.0f", s/1024}')
  tg=0; tx=0; live=0
  for f in tapes/watch_*.log; do
    [ -f "$f" ] || continue
    [ $(( now - $(stat -c %Y "$f") )) -lt 900 ] || continue
    body=$(tail -c 2000 "$f" | tr '\r' '\n')
    l=$(printf '%s' "$body" | grep -oE '\([0-9]+g/[0-9]+x\)' | tail -1)
    if [ -n "$l" ]; then
      tg=$(( tg + $(printf '%s' "$l" | grep -oE '[0-9]+g' | tr -d g) ))
      tx=$(( tx + $(printf '%s' "$l" | grep -oE '[0-9]+x' | tr -d x) ))
    fi
    printf '%s' "$body" | grep -q 'sets  ' && live=$(( live + 1 ))
  done
  printf '%s pollers=%s live=%s mem=%sMB gets=%s 429s=%s free=%sMB swap=%sMB\n' \
    "$(date -u +%H:%M:%S)" "${n:-0}" "$live" "${mb:-0}" "$tg" "$tx" \
    "$(free -m | awk '/^Mem:/{print $7}')" "$(free -m | awk '/^Swap:/{print $3}')" >> "$LOG"
  sleep 300
done
