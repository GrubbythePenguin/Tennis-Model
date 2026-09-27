"""tt_view.py — READ-ONLY dashboard for the TT screener.

Renders the same table as tt_screener.py, but from tapes/_tt_state.json —
zero GETs, zero WebSockets, zero writes. Run as many copies as you like;
the daemon screener (tapes/tt_screener.pid) is the only writer.

WHY: running a second tt_screener.py to watch the table gives it a second
writer — duplicate tapes, racing anchor files, and an empty-snapshot boot
gap that pulls every quote for a minute (bit twice on 2026-09-09). This is
the watching half, split out.

usage:  python3 tt_view.py [--interval 3]
"""
import argparse
import json
import os
import sys
import time

SNAP_P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "tapes", "_tt_state.json")


def render(show_all):
    try:
        s = json.load(open(SNAP_P))
    except Exception as e:
        return f"tt_view: cannot read {SNAP_P}: {e}"
    age = time.time() - float(s.get("ts") or 0)
    rows = []
    for ev, v in sorted((s.get("events") or {}).items()):
        if v.get("ended") and not show_all:
            continue
        if not v.get("live") and not show_all:
            continue
        fm = lambda x: "  -  " if x is None else f"{x:.3f}"
        an = "  -  " if v.get("anchor") is None else (
            f"{v['anchor']:.3f}" + ("" if v.get("pregame_anchor") else "*"))
        state = "-".join(map(str, v["state"])) if v.get("state") else "?"
        rows.append(f"{ev[-14:]:<15} {(v.get('status') or '?')[:7]:<8} "
                    f"{(v.get('score') or '')[:29]:<30} {state:<9} {an:<7} "
                    f"{fm(v.get('bid'))}  {fm(v.get('ask'))}  {fm(v.get('theo'))}  "
                    f"{fm(v.get('qbid'))}  {fm(v.get('qask'))}  "
                    f"{v.get('name1','?')[:14]} v {v.get('name2','?')[:14]}")
    stale = "  *** FEED STALE ***" if age > 45 else ""
    hdr = (f"tt_view (read-only)  {time.strftime('%H:%M:%S')}  "
           f"snapshot age {age:.0f}s{stale}  delay_points={s.get('delay_points')}\n"
           f"{'EVENT':<15} {'STATUS':<8} {'SCORE':<30} {'STATE':<9} {'PRE':<7} "
           f"{'BID':<6} {'ASK':<6} {'THEO':<6} {'qBID':<6} {'qASK':<6} MATCH")
    return hdr + "\n" + ("\n".join(rows) if rows else "(no live matches)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--all", action="store_true",
                    help="show pregame and ended matches too")
    a = ap.parse_args()
    while True:
        out = render(a.all)
        if sys.stdout.isatty():
            print("\033[2J\033[H" + out, flush=True)
        else:
            print(out + "\n", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
