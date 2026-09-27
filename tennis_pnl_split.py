"""Three-way P&L decomposition per event, rolled up by edge arm (2026-09-03).

  SCALP : FIFO round trips WITHIN a ticker (buy low/sell high same leg) — the
          market-making spread income. Expected positive.
  BOX   : after FIFO, same-direction residuals across the pair (short both /
          long both legs). Locked at entry: (combined entry px - 100c)/lot for
          short-short (reverse for long-long). Realized at settlement.
  LEAN  : the remaining directional residual, P&L'd against settlement (or
          last trade price, flagged *, if unsettled). Expected negative — the
          adverse-selection bleed the boxes must out-earn.

Usage: tennis_pnl_split.py --tag 26SEP03 [26SEP04 ...] [--split-hhmm 14:54]
Reads tapes/kalshi_fills.csv + kalshi_settlements.csv + ab_edges.json.
(Markouts stay in fill_markouts.py — this tool is the P&L ledger.)
"""
import csv, json, argparse, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--tag", nargs="+", required=True)
ap.add_argument("--fills", default="tapes/kalshi_fills.csv")
ap.add_argument("--settlements", default="tapes/kalshi_settlements.csv")
ap.add_argument("--ab", default="tapes/ab_edges.json")
ap.add_argument("--split-hhmm", default=None, help="also roll up events by first-fill before/after this UTC time")
a = ap.parse_args()

try: ab = json.load(open(a.ab))
except Exception: ab = {}
setl = {}
for r in csv.DictReader(open(a.settlements)):
    res = r.get("market_result") or r.get("result") or ""
    if res in ("yes", "no"): setl[r["ticker"]] = 1.0 if res == "yes" else 0.0

fills = defaultdict(list)   # event -> [(ts, ticker, signed_yes_qty, yes_px)]
for r in csv.DictReader(open(a.fills)):
    t = r["ticker"]
    if not any(tag in t for tag in a.tag): continue
    qty = int(float(r["count"])); px = float(r["price"])
    yes_px = px if r["side"] == "yes" else 1 - px
    d = qty if ((r["action"] == "buy") == (r["side"] == "yes")) else -qty
    fills["-".join(t.split("-")[:-1])].append((float(r["created_ts"]), t, d, yes_px))

def fifo2(seq):
    book = []; pnl = 0.0
    for q, px in seq:
        rem = q
        while rem and book and (book[0][0] > 0) != (rem > 0):
            oq, opx = book[0]
            m = min(abs(rem), abs(oq))
            pnl += m * ((px - opx) if oq > 0 else (opx - px))
            noq = oq - m if oq > 0 else oq + m
            if noq: book[0] = (noq, opx)
            else: book.pop(0)
            rem = rem + m if rem < 0 else rem - m
        if rem: book.append((rem, px))
    return pnl, book

rows = []
for ev, fl in sorted(fills.items()):
    fl.sort()
    per = defaultdict(list)
    for ts, t, d, ypx in fl: per[t].append((d, ypx))
    legs = sorted(per)
    scalp = sum(fifo2(per[t])[0] for t in legs)
    residual = {t: fifo2(per[t])[1] for t in legs}
    rpos = {t: sum(q for q, _ in residual[t]) for t in legs}
    ravg = {t: (sum(q*px for q, px in residual[t]) / rpos[t] if rpos[t] else 0.0) for t in legs}
    box_pnl = lean_pnl = 0.0; boxed = 0; unsettled = False
    if len(legs) == 2 and rpos[legs[0]] * rpos[legs[1]] > 0:      # same sign = box
        A, B = legs
        boxed = min(abs(rpos[A]), abs(rpos[B]))
        sgn = 1 if rpos[A] > 0 else -1                            # long-long or short-short
        combined = ravg[A] + ravg[B]
        # short-short (sgn<0): collected `combined` per pair-lot, pays 100 at
        # settlement -> (combined-1)*lots. Long-long: paid combined, receives 100.
        box_pnl = boxed * ((combined - 1.0) if sgn < 0 else (1.0 - combined))
        for t in legs:                                            # shrink residuals by the box
            q = rpos[t]; take = boxed if q > 0 else -boxed
            rpos[t] = q - take
    for t in legs:                                                # remaining = true lean
        q = rpos[t]
        if not q: continue
        if t in setl: s = setl[t]
        else:
            s = per[t][-1][1]; unsettled = True                   # mark at last fill px
        lean_pnl += q * (s - ravg[t])
    arm = (ab.get(ev) or {}).get("edge")
    arm = f"{arm:g}c" if arm else "base"
    rows.append(dict(ev=ev, arm=arm, first_ts=fl[0][0], scalp=scalp, box=box_pnl,
                     boxed=boxed, lean=lean_pnl, unsettled=unsettled,
                     vol=sum(abs(d) for _, _, d, _ in fl)))

def roll(rs, key):
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0, 0])
    for r in rs:
        g = agg[key(r)]
        g[0] += r["scalp"]; g[1] += r["box"]; g[2] += r["lean"]
        g[3] += r["boxed"]; g[4] += 1; g[5] += r["unsettled"]
    return agg

print(f"{'event':40s} {'arm':5s} {'scalp':>8s} {'box':>8s} {'lean':>8s} {'total':>8s}  boxed vol")
for r in sorted(rows, key=lambda r: r["scalp"] + r["box"] + r["lean"]):
    tot = r["scalp"] + r["box"] + r["lean"]
    u = "*" if r["unsettled"] else " "
    print(f"{r['ev'][-22:]:40s} {r['arm']:5s} {r['scalp']*100:+8.0f} {r['box']*100:+8.0f} "
          f"{r['lean']*100:+8.0f} {tot*100:+8.0f}{u} {r['boxed']:5d} {r['vol']:4d}")
print("\n(amounts in cents = $/100)")
for label, key in [("BY EDGE ARM", lambda r: r["arm"]), ("TOTAL", lambda r: "all")]:
    print(f"\n{label}")
    for k, (s, b, l, bx, n, uns) in sorted(roll(rows, key).items()):
        print(f"  {k:6s} scalp ${s:+8.2f}  box ${b:+8.2f}  lean ${l:+8.2f}  "
              f"TOTAL ${s+b+l:+8.2f}   ({n} events, {bx} boxed lots{', %d unsettled*' % uns if uns else ''})")
if a.split_hhmm:
    hh, mm = a.split_hhmm.split(":")
    import datetime as dt
    for r in rows:
        d = dt.datetime.fromtimestamp(r["first_ts"], dt.timezone.utc)
        r["half"] = "after" if (d.hour, d.minute) >= (int(hh), int(mm)) else "before"
    print(f"\nBY REGIME (first fill before/after {a.split_hhmm}Z)")
    for k, (s, b, l, bx, n, uns) in sorted(roll(rows, lambda r: r["half"]).items()):
        print(f"  {k:6s} scalp ${s:+8.2f}  box ${b:+8.2f}  lean ${l:+8.2f}  "
              f"TOTAL ${s+b+l:+8.2f}   ({n} events, {bx} boxed lots)")
