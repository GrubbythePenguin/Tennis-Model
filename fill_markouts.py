"""Every tennis fill, marked out and bucketed by the state that produced the quote.

    python3 fill_markouts.py [--since 26AUG27] [--event X] [--detail]

The tracking half of the 26AUG27 plan: trade 10 lots on the bare bracket, then find the
adverse-selection buckets worth widening from the fills themselves rather than from the
tape (a 3.4s tape cannot see the transient fills that carry the edge).

Joins quoter/trades.csv -> tapes/<event>.jsonl (vig-free mid, state, book at 3.4s) and
tapes/recenter_quotes.jsonl (the anchored point's class, bracket, margin) when present.
Markouts are signed cents per contract vs the vig-free mid of the TRACKED side, at
+5/15/30/60s and at the next point change + 5s ("1pt", the horizon the edge lives on).
"""
import collections
import argparse, bisect, collections, csv, glob, json, os, statistics as st, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H = (5, 15, 30, 60)


def point_class(s):
    pm, po = s.get("points_me"), s.get("points_opp")
    if s.get("games_me") == 6 and s.get("games_opp") == 6: return "tiebreak"
    if pm is None or po is None: return "unknown"
    ps, pr = (pm, po) if s.get("server") == "me" else (po, pm)
    if pr >= 3 and pr > ps: return "break point"
    if ps >= 3 and ps > pr: return "server game point"
    return "deuce" if (ps == pr and ps >= 2) else "early"


def load_tape(ev):
    rows = []
    p = f"tapes/{ev}.jsonl"
    if not os.path.exists(p): return rows
    for line in open(p, errors="replace"):
        if '"status": "live"' not in line: continue
        try: r = json.loads(line)
        except Exception: continue
        s = r.get("state") or {}
        if r.get("vig_free") is None or s.get("sets_me") is None: continue
        rows.append((r["ts"], r["vig_free"] * 100, s, r.get("book") or {}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="date tag like 26AUG27; fills on events before it are ignored")
    ap.add_argument("--event", nargs="*"); ap.add_argument("--detail", action="store_true")
    ap.add_argument("--fills", default="quoter/trades.csv", help="fill file (trades.csv schema); e.g. one rebuilt from Kalshi /portfolio/fills")
    a = ap.parse_args()
    meta = {}
    for lp in glob.glob("tapes/*.log.json"):
        try: m = json.load(open(lp)).get("meta") or {}
        except Exception: continue
        if m.get("me_ticker"): meta[m["event"]] = m
    qlog = {}
    if os.path.exists("tapes/recenter_quotes.jsonl"):
        for line in open("tapes/recenter_quotes.jsonl"):
            try: q = json.loads(line)
            except Exception: continue
            qlog.setdefault(q["event"], []).append(q)
    fills = [r for r in csv.DictReader(open(a.fills)) if r["ticker"].startswith(("KXATP", "KXWTA"))]
    recs, tapes = [], {}
    for f in fills:
        ev = f["ticker"].rsplit("-", 1)[0]
        if a.event and ev not in a.event: continue
        if a.since and ev.split("-")[1][:7] < a.since: continue
        if ev not in tapes: tapes[ev] = load_tape(ev)
        rows = tapes[ev]; m = meta.get(ev)
        if not rows or not m: continue
        ts = [r[0] for r in rows]; t = int(f["created_ts"])
        i0 = bisect.bisect_right(ts, t) - 1
        if i0 < 0 or t - rows[i0][0] > 30: continue
        is_me = f["ticker"] == m["me_ticker"]; buy = f["action"] == "buy"
        px = float(f["price"]) * 100; me_px = px if is_me else 100 - px
        long_me = (buy and is_me) or ((not buy) and (not is_me)); sg = 1 if long_me else -1
        idx = lambda tt: rows[min(len(rows) - 1, max(0, bisect.bisect_right(ts, tt) - 1))]
        mo = {f"{h}s": sg * (idx(t + h)[1] - me_px) for h in H}
        k0 = tuple(rows[i0][2].get(k) for k in ("sets_me", "sets_opp", "games_me", "games_opp", "points_me", "points_opp"))
        j = i0 + 1
        while j < len(rows) and tuple(rows[j][2].get(k) for k in ("sets_me", "sets_opp", "games_me", "games_opp", "points_me", "points_opp")) == k0: j += 1
        mo["1pt"] = sg * (idx(rows[j][0] + 5)[1] - me_px) if j < len(rows) else None
        # time since the point started (last point change before the fill)
        jj = i0
        while jj > 0 and tuple(rows[jj-1][2].get(k) for k in ("sets_me", "sets_opp", "games_me", "games_opp", "points_me", "points_opp")) == k0: jj -= 1
        s = rows[i0][2]
        q = None
        for qq in qlog.get(ev, []):
            if tuple(qq["state"].get(k) for k in ("sets_me", "sets_opp", "games_me", "games_opp", "points_me", "points_opp")) == k0: q = qq
        recs.append(dict(ev=ev, ser=ev.split("-")[0], t=t, side="BUY" if buy else "SELL", ticker=f["ticker"][-3:], me_px=me_px,
                         long_me=long_me, cls=(q["class"] if q else point_class(s)), bracket=(q["bracket"] if q else None),
                         margin=(q["margin"] if q else None), set_=s["sets_me"] + s["sets_opp"], mkt=rows[i0][1],
                         thru=sg * (rows[i0][1] - me_px), since=t - rows[jj][0], fee=float(f["fee_cost"]) / max(1, float(f["count"])) * 100,
                         cnt=float(f["count"]), mo=mo, taker=f["is_taker"] == "True", edge=float(f["edge_cents"] or 0),
                         game=(s.get("games_me") or 0) + (s.get("games_opp") or 0)))
    if not recs:
        print("no fills matched a tape"); return
    def line(lbl, R):
        if not R: return
        s = f"  {lbl:<26} n {len(R):>4} |"
        for h in [f"{h}s" for h in H] + ["1pt"]:
            v = [r["mo"][h] for r in R if r["mo"][h] is not None]
            s += f" {h:>3} {st.mean(v):+5.2f} ({sum(1 for x in v if x > 0)/len(v)*100:2.0f}%)" if v else ""
        nb = sum(1 for r in R if r["long_me"])
        s += f" | long-me {nb/len(R)*100:3.0f}%  fee {st.mean(r['fee'] for r in R):.2f}c"
        print(s)
    print(f"{len(recs)} fills on {len(set(r['ev'] for r in recs))} events   (markout = cents/contract vs tracked-side vig-free mid, share positive)")
    line("ALL", recs)
    print("\nby point class"); [line(c, [r for r in recs if r["cls"] == c]) for c in ("early", "deuce", "server game point", "break point", "tiebreak", "unknown")]
    print("\nby side of the tracked player"); line("long tracked side", [r for r in recs if r["long_me"]]); line("short tracked side", [r for r in recs if not r["long_me"]])
    print("\nby set"); [line(f"set {k+1}", [r for r in recs if r["set_"] == k]) for k in (0, 1, 2)]
    print("\nby tracked-side price at fill"); [line(f"{lo}-{hi}c", [r for r in recs if lo <= r["mkt"] < hi]) for lo, hi in ((0, 15), (15, 35), (35, 65), (65, 85), (85, 101))]
    print("\nby time since the point started"); [line(f"{lo}-{hi}s", [r for r in recs if lo <= r["since"] < hi]) for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 9999))]
    print("\nby games played in the set at the fill"); [line(f"games {lo}-{hi}", [r for r in recs if lo <= r["game"] <= hi]) for lo, hi in ((0, 2), (3, 5), (6, 8), (9, 12))]
    try:
        ab = json.load(open("tapes/ab_edges.json"))
        qpts = collections.Counter(q["event"] for q in (json.loads(l) for l in open("tapes/recenter_quotes.jsonl")) if q["event"] in ab)
        print("\nby ITF edge arm (A/B) — fills per anchored point is the fill rate")
        for arm in sorted({v["edge"] for v in ab.values()}):
            evs = {e for e, v in ab.items() if v["edge"] == arm}
            R = [r for r in recs if r["ev"] in evs]
            men = sum(1 for e in evs if e.startswith("KXITFMATCH")); women = len(evs) - men
            pts = sum(qpts[e] for e in evs)
            line(f"edge {arm:.0f}c ({men}M/{women}W, {pts} pts, {len(R)/pts if pts else 0:.2f} fills/pt)", R)
    except Exception as e:
        print("\n(no A/B edge file:", e, ")")
    print("\nby series"); [line(s_, [r for r in recs if r["ser"] == s_]) for s_ in sorted(set(r["ser"] for r in recs))]
    if any(r["bracket"] is not None for r in recs):
        print("\nby bracket width (from recenter_quotes.jsonl)")
        [line(f"{lo}-{hi}c", [r for r in recs if r["bracket"] is not None and lo <= r["bracket"] < hi]) for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 10), (10, 99))]
    print("\nby event"); [line(e.rsplit("-", 1)[-1], [r for r in recs if r["ev"] == e]) for e in sorted(set(r["ev"] for r in recs))]
    if a.detail:
        print()
        for r in sorted(recs, key=lambda r: r["t"]):
            print(f"  {time.strftime('%H:%M:%S', time.gmtime(r['t']))} {r['ev'].rsplit('-',1)[-1]:<8} {r['side']:<4} {r['ticker']} @{r['me_px']:5.1f} mkt {r['mkt']:5.1f} {r['cls']:<15} since {r['since']:3.0f}s | "
                  + " ".join(f"{h} {r['mo'][h]:+5.1f}" for h in list(f"{h}s" for h in H) + ["1pt"] if r["mo"][h] is not None))


if __name__ == "__main__":
    main()
