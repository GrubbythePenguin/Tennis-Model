#!/usr/bin/env python3
"""ASK CALIBRATION — is the displayed ask the ask we can actually pay?

Every break_dog fire/fire_empty event carries BOTH numbers: the manager's displayed
top-of-book (`top_book.breaker_ask`) and what REST returned at fire time
(`rest_top.breaker_no_bids`). The tradeable ask is 100 - highest NO bid.

This is the only calibration of the tape/manager book against executable prices that
exists, because no historical REST snapshots were ever captured. The backtest priced
every entry off the tape's ask; if that is systematically optimistic, the backtested
edge is overstated by the same amount.

    python3 askcal.py [tapes/break_dog_events.jsonl]
"""
import json, sys, collections, statistics as st

def best_rest_ask(levels):
    """levels = NO bids [[price,size],...]; YES ask = 100 - highest NO bid."""
    try:
        px = [float(p) for p, _ in levels]
    except Exception:
        return None
    return (100.0 - max(px) * 100.0) if px else None

def main(path):
    rows = []
    try:
        lines = open(path).read().strip().splitlines()
    except OSError:
        print(f"no events file at {path}"); return
    for l in lines:
        try: r = json.loads(l)
        except Exception: continue
        if r.get("type") not in ("fire", "fire_empty"): continue
        rt = r.get("rest_top")
        if not rt: continue                      # predates the telemetry
        shown = (r.get("top_book") or {}).get("breaker_ask")
        real = best_rest_ask(rt.get("breaker_no_bids") or [])
        if shown is None or real is None: continue
        rows.append(dict(event=r["event"], set_no=r.get("set_no"), type=r["type"],
                         shown=float(shown), real=real, fair=r.get("fair_c"),
                         lots=r.get("lots", 0), n_books=rt.get("n_books")))
    if not rows:
        print("no events yet carry both a displayed ask and REST depth.")
        print("(events logged before the rest_top patch cannot be calibrated.)")
        return
    d = [x["real"] - x["shown"] for x in rows]
    print(f"paired observations: {len(rows)}  across {len({x['event'] for x in rows})} matches")
    print(f"\n  {'event':42} {'set':>3} {'shown':>6} {'REST':>6} {'diff':>6} {'fair':>6} {'lots':>5}")
    for x in rows:
        print(f"  {x['event']:42} {x['set_no']:>3} {x['shown']:6.1f} {x['real']:6.1f} "
              f"{x['real']-x['shown']:+6.1f} {x['fair']:6.1f} {x['lots']:5d}")
    print(f"\n  REST ask minus displayed ask (positive = displayed was too CHEAP):")
    print(f"    mean {st.mean(d):+.2f}c   median {st.median(d):+.2f}c   "
          f"min {min(d):+.1f}c  max {max(d):+.1f}c")
    worse = sum(1 for x in d if x > 0.5)
    print(f"    displayed ask was optimistic in {worse}/{len(d)} cases "
          f"({100*worse/len(d):.0f}%)")
    if len(rows) >= 10:
        print(f"\n  IMPLICATION: the backtest measured +13.7c/lot using the DISPLAYED ask.")
        print(f"  Paying the REST ask instead costs {st.mean(d):+.2f}c, leaving roughly "
              f"{13.69 - st.mean(d):+.2f}c.")
    else:
        print(f"\n  too few observations to conclude — need ~10+ across several matches.")
    fills = [x for x in rows if x["lots"] > 0]
    print(f"\n  fills: {len(fills)}/{len(rows)}   "
          f"lots filled: {sum(x['lots'] for x in fills)}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tapes/break_dog_events.jsonl")
