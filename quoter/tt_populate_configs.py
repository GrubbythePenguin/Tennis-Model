"""Write template_quoter_config.csv rows for TABLE TENNIS from the screener snapshot.

REPLACES the whole file — the TT test runs with zero tennis rows (operator
decision 2026-09-09), and in QUOTER_SPORT=tt mode execution.py refuses tennis
tickers at order construction anyway, so a leftover tennis row could only
burn a refused-order log line. ZERO GETs: events and tickers come from
tapes/_tt_state.json, which tt_screener.py maintains; shard membership for the
KXTTELITEMATCH series was verified 2026-09-09 (exchange_index 0, authed
per-ticker GETs) and is enforced per-order by execution.py's series guard.

Only LIVE matches get rows by default, mirroring auto_quote_configs.sh: a
pre-match ticker in the config makes the quoter fetch books for a match the
model will not quote (tt_band gives no theo until status=live). run.py
hot-reloads the CSV on write, so re-running this as matches churn is the
whole lifecycle. --loop does that every N seconds.

usage:
    python tt_populate_configs.py [--volumes 10] [--max-position 50]
        [--min-edge 1.0] [--include-pregame] [--stop-quoting]
        [--loop 120]
"""
import argparse
import csv
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_P = os.path.join(HERE, "template_quoter_config.csv")
SNAP_P = os.path.join(os.path.dirname(HERE), "tapes", "_tt_state.json")
QUOTABLE_P = os.path.join(HERE, "tt_quotable_series.json")


def _auto_verify_series(snap):
    """Shard-verify any NEW table tennis series the screener is tracking and,
    on exchange_index 0, add it to tt_quotable_series.json — which is what
    execution.py's guard and the screener's theo gate both read, so a newly
    verified series becomes quotable end-to-end with no restarts. A series
    that verifies to any other shard is refused loudly and permanently (until
    a human edits the json). Series with a bo5_violation flag are never
    auto-added."""
    try:
        quotable = json.load(open(QUOTABLE_P))
    except Exception:
        quotable = {}
    events = (snap.get("events") or {})
    by_series = {}
    for ev, e in events.items():
        by_series.setdefault(ev.split("-", 1)[0], []).append(e)
    changed = False
    for series, evs in sorted(by_series.items()):
        if series in quotable:
            continue
        if any(e.get("bo5_violation") for e in evs):
            print(f"[{time.strftime('%H:%M:%S')}] NOT adding {series}: "
                  f"bo5 violation seen", flush=True)
            continue
        tick = next((e.get("t1") for e in evs if e.get("t1")), None)
        if not tick:
            continue
        from tennis_populate_configs import _auth, verify_shard
        shard = verify_shard(_auth(), tick)
        if shard == 0:
            quotable[series] = {"shard": 0, "best_of": 5,
                                "verified": time.strftime("%Y-%m-%d"),
                                "note": f"auto-verified via {tick}"}
            changed = True
            print(f"[{time.strftime('%H:%M:%S')}] AUTO-ADDED series {series} "
                  f"(shard 0 verified via {tick})", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] REFUSING series {series}: "
                  f"shard={shard} != 0 (via {tick})", flush=True)
    if changed:
        tmp = QUOTABLE_P + ".tmp"
        with open(tmp, "w") as f:
            json.dump(quotable, f, indent=1)
        os.replace(tmp, QUOTABLE_P)
    return quotable

# Column defaults, taken from the live tennis rows and adjusted for TT.
# Anything not listed defaults to "" (framework_config row.get fallbacks).
DEFAULTS = {
    "config_id": "TT_ELITE",
    # -1 = improve any EXTERNAL top by 1c. Safe only because of the self-jump
    # guard (quoter.py 2026-09-09): a top level matching one of our own
    # resting prices is joined, never improved, so a competitor bulk-posting
    # ahead of us gets leapfrogged but we can never ratchet against ourselves.
    "min_distance_from_top_level": "-1",
    "tick_step": "1",
    "reprice_buffer": "1.0",
    "stop_quoting": "False",
    "skew_start_fraction": "0",
    "skew_max_shift_cents": "5",
    "quote_side": "both",
    "model_name": "tt_band",
    "execution_type": "quoter",
    "arb_scale_step_cents": "1.0",
    "max_fire_size": "300",
    "disable_forfeit_check": "False",
    "maker_refill_cooldown_sec": "15",
    "data_source": "kalshi",
    "skew_full_fraction": "1",
    "series_edge_mult": "1.0",
    "yes_extra_edge_cents": "0.0",
    "g3_edge_mult": "1.0",
    "pregame_taker_block": "false",
    "pregame_block_cents": "5",
    "refire_max_per_window": "2",
    "refire_min_fill_lots": "10",
    "refire_min_depth_lots": "100",
    "fav_edge_k": "0",
    "pregame_position_frac": "0",
    "retreat_cap_cents": "3.0",
    "retreat_f_cap": "0.02",
    "retreat_f0": "0.005",
    "retreat_half_life_sec": "300",
    "retreat_include_takers": "0",
    "capture_buffer_min": "0",
}


def build_rows(a):
    try:
        snap = json.load(open(SNAP_P))
    except Exception as e:
        print(f"cannot read {SNAP_P}: {e} — is tt_screener running?")
        return None
    age = time.time() - float(snap.get("ts") or 0)
    if age > 60:
        print(f"snapshot is {age:.0f}s old — is tt_screener running? Writing nothing.")
        return None
    quotable = _auto_verify_series(snap)
    rows = []
    for ev, e in sorted((snap.get("events") or {}).items()):
        if ev.split("-", 1)[0] not in quotable or e.get("bo5_violation"):
            continue
        if e.get("ended"):
            continue
        if not e.get("live") and not a.include_pregame:
            continue
        for tick in (e.get("t1"), e.get("t2")):
            if not tick:
                continue
            r = dict(DEFAULTS)
            r.update({"ticker": tick,
                      "min_edge": f"{a.min_edge:g}",
                      "min_absolute_edge": f"{a.min_absolute_edge:g}",
                      "volumes": a.volumes,
                      "max_position": str(a.max_position),
                      "stop_quoting": "True" if a.stop_quoting else "False"})
            rows.append(r)
    return rows


def write_csv(rows):
    with open(CSV_P) as f:
        header = next(csv.reader(f))
    tmp = CSV_P + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in header})
    os.replace(tmp, CSV_P)          # atomic: run.py hot-reload never sees a torn file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", default="150")   # 150 lots/level since 2026-09-09
                                                  # (full liquidity-rewards size)
    # Defaults per operator decision 2026-09-09: min_edge 3 (variance-scaled,
    # widest at 50c) floored by min_absolute_edge 1 at the extremes.
    # max_position >= volumes, else a single full fill exceeds the cap and the
    # order itself is bigger than the whole position budget. 300 since
    # 2026-09-09: absorbs two full 150-lot fills before reduce-only.
    ap.add_argument("--max-position", type=int, default=300)
    ap.add_argument("--min-edge", type=float, default=3.0)
    ap.add_argument("--min-absolute-edge", type=float, default=1.0)
    ap.add_argument("--include-pregame", action="store_true")
    ap.add_argument("--stop-quoting", action="store_true",
                    help="write rows with stop_quoting=True (dry wiring test)")
    ap.add_argument("--loop", type=float, default=0,
                    help="rewrite every N seconds as matches churn (0 = once)")
    a = ap.parse_args()
    last = None
    while True:
        # One bad cycle (auto-verify GET failure, torn snapshot, etc.) must
        # not kill an unattended loop — log and try again next interval.
        try:
            rows = build_rows(a)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] cycle error: "
                  f"{type(e).__name__}: {e} — retrying next interval", flush=True)
            rows = None
        if rows is not None:
            # Key on tickers AND settings: a re-run that only flips
            # --stop-quoting must still rewrite, or the old rows stand.
            key = tuple(sorted((r["ticker"], r["stop_quoting"], r["volumes"],
                                r["max_position"], r["min_edge"],
                                r["min_absolute_edge"]) for r in rows))
            if key != last:                 # rewrite ONLY on real change:
                write_csv(rows)             # every write triggers a hot-reload
                last = key
                print(f"[{time.strftime('%H:%M:%S')}] wrote {len(rows)} TT rows "
                      f"({len(rows) // 2} matches) -> {os.path.basename(CSV_P)}",
                      flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] unchanged ({len(rows)} rows)",
                      flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
