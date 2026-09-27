#!/bin/bash
# Keep template_quoter_config.csv in step with the LIVE pollers, on a loop. ZERO GETs.
#
#   ./auto_quote_configs.sh [cycle_sec]
#
# Every cycle: the events with a running poll_tennis.py watcher whose tape's newest
# row says status=live become config rows, built from tapes/discovered.json (the arm
# loop's /events payload: tickers + exchange_index) via
# tennis_populate_configs.py --from-discovery - no milestone, live_data, markets or
# shard GETs. The template is swapped ONLY if the ticker set changed (run.py
# hot-reloads on any write). Finished matches fall out when their poller exits;
# not-yet-live matches are NOT configured, so the quoter never REST-fetches a book
# for a pre-match ticker.
cd "$(dirname "$0")" || exit 1
CYCLE=${1:-300}
LOG=tapes/auto_configs.log
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
    try:
        with open(f"tapes/{ev}.jsonl", "rb") as f:
            f.seek(max(0, os.stat(f"tapes/{ev}.jsonl").st_size - 8192))
            last = f.read().decode("utf-8", "replace").strip().splitlines()[-1]
        st = (json.loads(last).get("status") or "").strip().lower()
        if st and st not in {"not_started", "closed", "ended", "cancelled", "canceled",
             "finished", "completed", "postponed", ""}: out.append(ev)      # mirrors kalshi_tennis._NOT_LIVE; ITF says "started", ATP says "live"
    except Exception: pass
print(" ".join(sorted(set(out))))
PY
)
    if [ -z "$EVENTS" ]; then
        NEW=""
    else
        python3 quoter/tennis_populate_configs.py --from-discovery tapes/discovered.json --event $EVENTS --out "$TMP" \
            --ab-edges "${AB_EDGES:-3.0,6.0}" \
            --ab-prefixes "${AB_PREFIXES:-KXITFMATCH,KXITFWMATCH,KXATPCHALLENGERMATCH,KXWTACHALLENGERMATCH}" >> "$LOG" 2>&1
        NEW=$(tail -n +2 "$TMP" 2>/dev/null | sort)     # full rows: any parameter change (edge arm, max_position, retreat...) must reload
    fi
    OLD=$(tail -n +2 quoter/template_quoter_config.csv | sort)
    if [ "$NEW" != "$OLD" ]; then
        if [ -n "$NEW" ]; then cp "$TMP" quoter/template_quoter_config.csv; else head -1 quoter/template_quoter_config.csv > "$TMP.h" && cp "$TMP.h" quoter/template_quoter_config.csv; fi
        echo "$(date -u '+%F %T')Z template updated: $(echo "$NEW" | grep -c . ) rows: $(echo "$NEW" | cut -d, -f2 | sed 's/.*-\([0-9A-Z]*\)-[A-Z0-9]*$/\1/' | sort -u | tr '\n' ' ')" >> "$LOG"
    fi
    sleep "$CYCLE"
done
