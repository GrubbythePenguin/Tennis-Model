#!/bin/bash
# Enable the BREAK-DOG taker (ITF, 200-lot pilot): append its two rows to
# set3_market_parameters.csv.
#
# WIRING (verified live 2026-09-27). auto_set3_configs.sh is the ONLY config loop
# running; it reads set3_market_parameters.csv and writes SET3_OUT, which defaults to
# quoter/template_quoter_config.csv -- and that is the file THE tennis run.py instance
# loads (no QUOTER_CONFIG_CSV set, cwd Tennis-Model/quoter, sub 1 / shard 3). So these
# rows reach the live quoter within one auto_set3_configs cycle. There is no separate
# "set3 instance": set3_quoter_config.csv does not exist.
#
# WHY THIS SCRIPT REFUSES TO RUN AGAINST A STALE QUOTER
# A running quoter holds manager.py in memory. Until it restarts on the manager.py
# that knows execution_type=break_dog_taker, an unknown type falls through to
# `else: bot = QuoterBot(config)` (manager.py ~line 563) and the rows would come up
# as a 200-lot QUOTING MAKER, which is not this strategy. auto_set3_configs.sh
# hot-reloads within its cycle, so there is no safe window. The check below compares
# every running run.py's start time against manager.py's mtime and refuses if any
# predates it.
#
# ORDER:
#   1. restart the tennis run.py instance (cwd Tennis-Model/quoter) so it loads the
#      new manager.py and the bot. THE OPERATOR does this by hand, in their terminal.
#   2. ./enable_break_dog.sh              (this script)
#   3. watch tapes/break_dog_events.jsonl for would_fire  -- SHADOW, no orders
#   4. touch break_dog_live.flag          (arm real orders)
# Kill switch at any point: touch disable_break_dog.flag
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd -P)"
P=quoter/set3_market_parameters.csv
MGR=quoter/manager.py

if grep -q "break_dog_taker" "$P"; then
    echo "already enabled — $P carries break_dog_taker rows"; exit 0
fi

# --- staleness guard: any run.py in THIS tree older than manager.py blocks us ---
MGR_EPOCH=$(stat -c %Y "$MGR")
STALE=0
for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    # TOKEN-exact match, not a substring. A shell whose command TEXT merely mentions
    # run.py (this script's own caller, for one) must not be mistaken for a quoter, and
    # `pgrep -f run.py` would match it. Require argv[0] to be a python binary AND some
    # argv token to be exactly run.py (or */run.py).
    mapfile -d '' -t argv < "$d/cmdline" 2>/dev/null || continue
    [ "${#argv[@]}" -ge 2 ] || continue
    case "${argv[0]}" in *python*) ;; *) continue ;; esac
    hit=0
    for t in "${argv[@]}"; do
        case "$t" in run.py|*/run.py) hit=1; break ;; esac
    done
    [ "$hit" -eq 1 ] || continue
    cmd="${argv[*]}"
    cwd=$(readlink -f "$d/cwd" 2>/dev/null) || continue
    case "$cwd" in "$ROOT"|"$ROOT"/*) ;; *) continue ;; esac
    # process start time in epoch seconds
    st=$(awk '{print $22}' "$d/stat" 2>/dev/null) || continue
    btime=$(awk '/^btime/{print $2}' /proc/stat)
    hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
    started=$(( btime + st / hz ))
    if [ "$started" -lt "$MGR_EPOCH" ]; then
        echo "REFUSING: pid $pid ($cwd) started $(date -d @"$started" '+%F %T') but"
        echo "          $MGR was modified $(date -d @"$MGR_EPOCH" '+%F %T')."
        echo "          That quoter does not know execution_type=break_dog_taker; enabling"
        echo "          now would hot-reload these rows as a 200-lot QUOTING MAKER."
        STALE=1
    fi
done
if [ "$STALE" -ne 0 ]; then
    echo
    echo "Restart the tennis run.py instance (cwd Tennis-Model/quoter) first, then re-run."
    exit 1
fi

python3 - "$P" <<'PY'
import csv, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
hdr = list(rows[0].keys())
src = {r["config_id"]: r for r in rows}
new = []
for base, cid in (("SET1TAKER_ITF", "BREAKDOG_ITF"), ("SET1TAKER_ITFW", "BREAKDOG_ITFW")):
    r = dict(src[base])                 # inherit every column from the set-1 taker
    r["config_id"] = cid
    r["execution_type"] = "break_dog_taker"
    r["min_edge"] = "6"                 # 6c NET OF FEES against the ask
    r["min_absolute_edge"] = "6"
    r["volumes"] = "200"                # 200-lot pilot
    r["max_fire_size"] = "200"
    r["max_position"] = "200"
    new.append(r)
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
    for r in rows + new:
        w.writerow(r)
print(f"appended {len(new)} break_dog_taker rows to {p}")
PY
echo
echo "SHADOW until you run: touch break_dog_live.flag"
echo "Kill switch:          touch disable_break_dog.flag"
echo "Telemetry:            tapes/break_dog_events.jsonl   (would_fire / fire / fire_empty / skip)"
