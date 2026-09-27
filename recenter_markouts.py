"""The overshoot-recenter strategy as a maker, scored on ONE-POINT markouts across every tape.

    python3 recenter_markouts.py [--edge 0 1 2] [--procs 4]

Rules (live tennis_recenter equivalents): at every point change re-anchor to the vig-free
mid; deltas from the walk-forward ewma2 BOUNDARY fit (n >= 8, bracket <= 8c, theos in
[2, 98]); bid = anchor + d_lo - edge, offer = anchor + d_hi + edge; a resting quote fills
when the market's touch reaches its price; one fill per side per point.

Markout = signed (mark - fill) in cents, mark = vig-free mid at fill+5s / +15s / +30s /
the next point change +5s ("one point") / the second point change +5s. Tapes without a
raw book (pre-26AUG26) get bid/ask = mid -/+ 0.5c, the median spread; reported apart.
"""
import sys, os, json, glob, bisect, statistics as st, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel, _model
from point_fit import boundary_fit_series

MIN_B, MAX_BRK, LO, HI = 8, 8.0, 2.0, 98.0
EDGES = (0.0, 1.0, 2.0)


def run_match(path):
    lp = path[:-6] + ".log.json"
    lg = json.load(open(lp)); bo, tb = lg["best_of"], lg.get("final_set_tb") or 7
    norm = ImpliedModel(best_of=bo, first_server="me", warn_split=False, final_set_tb=tb)
    rows, synth = [], False
    for line in open(path, errors="replace"):
        if '"status": "live"' not in line: continue
        try: r = json.loads(line)
        except Exception: continue
        s = r.get("state") or {}
        if r.get("vig_free") is None or s.get("sets_me") is None or s.get("server") in (None, ""): continue
        b = r.get("book") or {}
        if b.get("bid_me") is None or b.get("ask_me") is None:
            m = r.get("mid_me")
            if m is None: continue
            b = {"bid_me": m - 0.005, "ask_me": m + 0.005}; synth = True
        key = tuple(s.get(k) for k in ("sets_me", "sets_opp", "games_me", "games_opp", "points_me", "points_opp", "server"))
        rows.append((r["ts"], r["vig_free"], b["bid_me"] * 100, b["ask_me"] * 100, key, {k: v for k, v in s.items() if not k.startswith("_")}))
    if len(rows) < 100: return None
    # windows (one per point state) for the boundary fit, and each row's window index
    win, widx = [], []
    for ts, vf, bm, am, key, s in rows:
        if not win or win[-1][0] != key: win.append((key, s, vf, {}))
        widx.append(len(win) - 1)
    bf = boundary_fit_series(lg, win, norm, 2.0)
    ts_all = [r[0] for r in rows]
    pchg = [i for i in range(1, len(rows)) if rows[i][4] != rows[i-1][4]]     # row index of each point change
    def vf_at(t):
        i = bisect.bisect_right(ts_all, t) - 1
        return rows[i][1] * 100
    out = {e: [] for e in EDGES}
    quote = {e: None for e in EDGES}
    ser = os.path.basename(path).split("-")[0]
    for i, (ts, vf, bm, am, key, s) in enumerate(rows):
        w = widx[i]
        if i == 0 or rows[i-1][4] != key:                       # new point: re-anchor
            f = bf[w]
            for e in EDGES: quote[e] = None
            if not f or f[2] < MIN_B: continue
            p, q = f[0], f[1]
            try:
                n0 = norm._state(**s)
                sw = dict(s); sw["points_me"] = s.get("points_me", 0) + 1
                sl = dict(s); sl["points_opp"] = s.get("points_opp", 0) + 1
                now = _model(p, q, n0, bo, tb) * 100
                pw = _model(p, q, norm._state(**sw), bo, tb) * 100
                pl = _model(p, q, norm._state(**sl), bo, tb) * 100
            except Exception:
                continue
            dlo, dhi = min(pw, pl) - now, max(pw, pl) - now
            if dhi - dlo > MAX_BRK: continue
            a = vf * 100
            for e in EDGES:
                b_, o_ = a + dlo - e, a + dhi + e
                if b_ < LO or o_ > HI or o_ < b_: continue
                quote[e] = [b_, o_, False, False]
        for e in EDGES:
            qd = quote[e]
            if qd is None: continue
            b_, o_, fb, fo = qd
            for side, hit in (("BUY", (not fb) and am <= b_ + 1e-9), ("SELL", (not fo) and bm >= o_ - 1e-9)):
                if not hit: continue
                px = b_ if side == "BUY" else o_; sg = 1 if side == "BUY" else -1
                qd[2 if side == "BUY" else 3] = True
                j = bisect.bisect_right(pchg, i)
                n1 = rows[pchg[j]][0] if j < len(pchg) else None
                n2 = rows[pchg[j+1]][0] if j + 1 < len(pchg) else None
                mo = {"5s": sg * (vf_at(ts + 5) - px), "15s": sg * (vf_at(ts + 15) - px), "30s": sg * (vf_at(ts + 30) - px),
                      "1pt": sg * (vf_at(n1 + 5) - px) if n1 else None, "2pt": sg * (vf_at(n2 + 5) - px) if n2 else None,
                      "thru": sg * (vf - px) if False else sg * ((bm + am) / 2 - px)}   # how far through the mid we were filled
                out[e].append((side, px, mo))
    return dict(ev=os.path.basename(path)[:-6], ser=ser, synth=synth, npts=len(pchg), fills=out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    paths = [p for p in sorted(glob.glob("tapes/*.jsonl")) if os.path.exists(p[:-6] + ".log.json")]
    with Pool(a.procs) as pool:
        res = [r for r in pool.imap_unordered(run_match, paths) if r]
    json.dump(res, open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "recenter_markouts.json"), "w"))
    H = ("5s", "15s", "30s", "1pt", "2pt")
    def table(lbl, R, e):
        F = [f for r in R for f in r["fills"][e]]
        if not F: print(f"  {lbl:<28} no fills"); return
        nb = sum(1 for f in F if f[0] == "BUY"); npts = sum(r["npts"] for r in R)
        ow = [max(sum(1 for f in r["fills"][e] if f[0] == "BUY"), sum(1 for f in r["fills"][e] if f[0] == "SELL")) / len(r["fills"][e]) for r in R if len(r["fills"][e]) >= 10]
        s = f"  {lbl:<28} m {len(R):>3} fills {len(F):>5} ({len(F)/npts*100:4.1f}% of pts) buy {nb/len(F)*100:3.0f}% one-way/match {st.median(ow)*100 if ow else float('nan'):3.0f}% | thru-mid {st.mean(f[2]['thru'] for f in F):+5.2f} |"
        for h in H:
            v = [f[2][h] for f in F if f[2][h] is not None]
            s += f" {h} {st.mean(v):+5.2f} ({sum(1 for x in v if x > 0)/len(v)*100:2.0f}%+)" if v else ""
        print(s)
    for e in EDGES:
        print(f"\n=== quote at bracket edge + {e:.0f}c ===   markout c/contract (share positive)")
        table("all tapes", res, e)
        table("real book (26AUG26)", [r for r in res if not r["synth"]], e)
        table("synthetic +-0.5c book", [r for r in res if r["synth"]], e)
        for ser in sorted(set(r["ser"] for r in res)):
            table(ser, [r for r in res if r["ser"] == ser], e)


if __name__ == "__main__":
    main()
