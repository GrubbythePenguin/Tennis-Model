#!/bin/bash
# Cron wrapper for the daily tennis set-1 taker P&L email (replaced the dog-windows
# OOS tracker on 26SEP24). Resolves its own directory rather than hard-coding one, so
# the crontab entry needs no `cd` and the venv python can be referenced relatively.
cd "$(dirname "$0")" || exit 1
exec nice -n 15 ionice -c3 .venv/bin/python3 tennis_pnl_email.py \
     --pull --days 2 --email "${TENNIS_PNL_EMAIL-akdlfjsif@gmail.com}" --once-per-date
# NOTE the single dash in ${VAR-default}: it substitutes only when TENNIS_PNL_EMAIL is
# UNSET, so `TENNIS_PNL_EMAIL= ./tennis_pnl_cron.sh` is a real dry run. With the `:-`
# form it had here originally, an explicitly EMPTY value still fell through to the
# default and a 26SEP24 "dry run" sent a live email and wrote the once-per-date marker,
# which would then have suppressed the scheduled 05:25Z send.
