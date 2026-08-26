#!/bin/bash
# One-shot: wait until a target time, then deep-capture a SINGLE match.
#
#   ./run_deep_capture_at.sh <epoch_seconds> [interval]
#
# Detached and self-contained so it survives the session that started it, and leaves
# no crontab entry behind to fire again tomorrow. Honours the kill flag, so
# `touch disable_tennis_poller.flag` cancels it even while it is still sleeping.
#
# Retries selection for a while after the target: at 01:30 Pacific whichever match is
# live may not be quoting a two-sided book yet, and giving up on the first look would
# waste the window.
cd "$(dirname "$0")" || exit 1
TARGET=${1:?need epoch seconds}
INTERVAL=${2:-1}
LOG=tapes/deep_capture_cron.log

echo "$(date -u '+%F %T')Z armed for $(date -u -d "@$TARGET" '+%F %T')Z, interval ${INTERVAL}s" >> "$LOG"
while [ "$(date +%s)" -lt "$TARGET" ]; do
    if [ -f disable_tennis_poller.flag ]; then
        echo "$(date -u '+%F %T')Z kill flag present — cancelled before launch" >> "$LOG"
        exit 0
    fi
    sleep 30
done

# Try for up to 40 minutes to find a live, two-sided match.
for attempt in $(seq 1 20); do
    if [ -f disable_tennis_poller.flag ]; then
        echo "$(date -u '+%F %T')Z kill flag present — cancelled" >> "$LOG"
        exit 0
    fi
    echo "=== $(date -u '+%F %T')Z attempt $attempt ===" >> "$LOG"
    if python3 deep_capture.py --launch --interval "$INTERVAL" >> "$LOG" 2>&1; then
        if grep -q "^launched " "$LOG"; then
            echo "$(date -u '+%F %T')Z capture running" >> "$LOG"
            exit 0
        fi
    fi
    sleep 120
done
echo "$(date -u '+%F %T')Z gave up after 20 attempts — no live two-sided match" >> "$LOG"
