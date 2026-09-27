"""tt_glitch_scan.py — find every feed glitch in the screener tape and tie it
to the fills it caused.

A GLITCH is a row whose (sets_played, total_points) regresses below the
event's running high-water mark. Two classes:
  reset    — state collapses to/near 0-0,0-0 mid-match (the AGRFGR class:
             the model prices a fresh match and rests tight quotes)
  partial  — smaller regression (scorer correction, or dual-writer interleave
             from the periods when two screeners wrote the tape)

For each glitch window: duration, the state before/during, what theo/band the
model showed, and any FILLS from trades.csv inside the window or within
GRACE_S after it ends (the snap-back pickoff second).

usage: python3 tt_glitch_scan.py [--since-hours 12]
"""
import argparse
import csv
import json
import time

ROWS = "tapes/tt_screener_rows.jsonl"
TRADES = "quoter/trades.csv"
GRACE_S = 45.0


def _epoch(ts):
    return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))


def scan(since):
    events = {}
    for line in open(ROWS):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        st = r.get("state")
        if not st or not r.get("event"):
            continue
        t = _epoch(r["ts"])
        if t < since:
            continue
        events.setdefault(r["event"], []).append(
            (t, r["ts"][11:], tuple(st), r.get("theo"), r.get("qbid"),
             r.get("qask"), (r.get("status") or "")[:7]))
    glitches = []
    for ev, rows in events.items():
        rows.sort()
        hwm = (0, 0)
        hwm_state = None
        cur = None
        for (t, ts, st, theo, qb, qa, status) in rows:
            prog = (st[0] + st[1], sum(st))
            if prog[0] < hwm[0] or (prog[0] == hwm[0] and prog[1] < hwm[1] - 2):
                kind = "RESET" if sum(st) <= 2 and hwm[1] >= 8 else "partial"
                if cur is None:
                    cur = {"event": ev, "start": t, "start_ts": ts,
                           "hwm_state": hwm_state, "kind": kind,
                           "glitch_state": st, "theo_during": theo,
                           "band_during": (qb, qa), "n": 1}
                else:
                    cur["n"] += 1
                    if kind == "RESET":
                        cur["kind"] = "RESET"
                cur["end"] = t
            else:
                if prog >= hwm:
                    hwm, hwm_state = prog, st
                if cur is not None:
                    cur.setdefault("end", cur["start"])
                    glitches.append(cur)
                    cur = None
        if cur is not None:
            cur.setdefault("end", cur["start"])
            glitches.append(cur)
    return glitches


def fills_since(since):
    out = []
    with open(TRADES) as f:
        for r in csv.DictReader(f):
            if (r.get("ticker") or "").startswith("KXTT") and \
                    float(r["created_ts"]) >= since:
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=float, default=12.0)
    a = ap.parse_args()
    since = time.time() - a.since_hours * 3600
    glitches = scan(since)
    fills = fills_since(since)

    resets = [g for g in glitches if g["kind"] == "RESET"]
    partials = [g for g in glitches if g["kind"] != "RESET"]
    print(f"glitch windows: {len(glitches)} total — {len(resets)} RESET, "
          f"{len(partials)} partial/interleave\n")

    hit_total = 0.0
    for g in sorted(glitches, key=lambda x: x["start"]):
        dur = g["end"] - g["start"]
        evfills = []
        for f in fills:
            if f["ticker"].rsplit("-", 1)[0] != g["event"]:
                continue
            ft = float(f["created_ts"])
            if g["start"] - 2 <= ft <= g["end"] + GRACE_S:
                evfills.append(f)
        flag = " <-- FILLS IN WINDOW" if evfills else ""
        print(f"{g['start_ts']}  {g['event'][-14:]:<15} {g['kind']:<7} "
              f"dur={dur:4.0f}s x{g['n']:<3} hwm={g['hwm_state']} -> "
              f"{g['glitch_state']} theo_during={g['theo_during']} "
              f"band={g['band_during']}{flag}")
        for f in evfills:
            px, qty = float(f["price"]), float(f["count"])
            print(f"    FILL {time.strftime('%H:%M:%S', time.localtime(float(f['created_ts'])))} "
                  f"{f['ticker'][-3:]} {f['action']} {f['side']} "
                  f"{qty:.0f}@{100*px:.0f}c  raw_theo={f['raw_theo']}")
            hit_total += qty * px
    print(f"\nfills inside glitch/grace windows: notional touched "
          f"${hit_total:.0f} (see tt_fill_lag.py for their settled pnl)")


if __name__ == "__main__":
    main()
