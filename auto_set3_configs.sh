#!/bin/bash
# Keep set3_quoter_config.csv in step with the LIVE ITF pollers, on a loop. ZERO GETs.
#
#   ./auto_set3_configs.sh [cycle_sec]
#
# Twin of auto_quote_configs.sh for the SET-3 DOG MAKER instance (see
# TENNIS_SET3_DOG_MAKER_SCOPE.md REV 26SEP10-b). Same discovery: every event with a
# running poll_tennis.py watcher whose tape says live becomes a config row pair — but
# built from set3_market_parameters.csv, which only carries KXITFMATCH/KXITFWMATCH
# prefixes, so non-ITF matches fall out inside tennis_populate_configs (no prefix row
# -> no config). Rows must exist BEFORE a match reaches 1-1: the trigger lives in the
# model, not in config timing, and an unconfigured leg blocks arming (both-legs gate).
#
# The set3 run.py instance loads this file via QUOTER_CONFIG_CSV and hot-reloads on
# any write; the file is swapped only when rows actually change.
cd "$(dirname "$0")" || exit 1
CYCLE=${1:-300}
LOG=tapes/auto_set3_configs.log
# Default is the framework's default CSV (operator direction 26SEP10: the set3 book
# runs as THE tennis instance off template_quoter_config.csv, sub 1 / shard 3 —
# recenter and TT books are mutually exclusive with it anyway, see execution.py).
# Override with SET3_OUT for a separate-instance file.
OUT=${SET3_OUT:-quoter/template_quoter_config.csv}
TMP=$(mktemp)
while true; do
    EVENTS=$(python3 - <<'PY'
import os, json
out = []
for p in os.listdir("/proc"):
    if not p.isdigit(): continue
    try: c = open(f"/proc/{p}/cmdline", "rb").read().decode("utf-8", "replace").replace("\0", " ")
    except Exception: continue
    if "poll_tennis.py" not in c or " watch " not in c: continue
    ev = c.split(" watch ", 1)[1].split()[0]
    # All six BO3 series since the set1 dog-leader joined (26SEP11); populate
    # drops anything without a matching market_prefix row anyway.
    if not ev.startswith(("KXITFMATCH-", "KXITFWMATCH-", "KXATPMATCH-", "KXWTAMATCH-",
                          "KXATPCHALLENGERMATCH-", "KXWTACHALLENGERMATCH-")): continue
    try:
        with open(f"tapes/{ev}.jsonl", "rb") as f:
            f.seek(max(0, os.stat(f"tapes/{ev}.jsonl").st_size - 8192))
            last = f.read().decode("utf-8", "replace").strip().splitlines()[-1]
        st = (json.loads(last).get("status") or "").strip().lower()
        if st and st not in {"not_started", "closed", "ended", "cancelled", "canceled",
             "finished", "completed", "postponed", ""}: out.append(ev)      # ITF says "started"
    except Exception: pass
print(" ".join(sorted(set(out))))
PY
)
    if [ -z "$EVENTS" ]; then
        NEW=""
    else
        python3 quoter/tennis_populate_configs.py --from-discovery tapes/discovered.json \
            --params set3_market_parameters.csv --event $EVENTS --out "$TMP" >> "$LOG" 2>&1
        NEW=$(tail -n +2 "$TMP" 2>/dev/null | sort)
    fi
    [ -f "$OUT" ] || head -1 quoter/set3_market_parameters.csv | sed 's/market_prefix/ticker/' > "$OUT"
    OLD=$(tail -n +2 "$OUT" | sort)
    if [ "$NEW" != "$OLD" ]; then
        if [ -n "$NEW" ]; then cp "$TMP" "$OUT"; else head -1 "$OUT" > "$TMP.h" && cp "$TMP.h" "$OUT"; fi
        echo "$(date -u '+%F %T')Z set3 config updated: $(echo "$NEW" | grep -c .) rows: $(echo "$NEW" | cut -d, -f2 | sed 's/.*-\([0-9A-Z]*\)-[A-Z0-9]*$/\1/' | sort -u | tr '\n' ' ')" >> "$LOG"
    fi
    sleep "$CYCLE"
done
