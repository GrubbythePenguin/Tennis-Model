"""Market-implied (p, q) from each PAIR of consecutive point snapshots.

    python3 point_implied.py EVENT [--window 10]

Two prices at two adjacent states are two equations in two unknowns. The level pins
p+q; the one-point move pins the rest. Per point-state the price is the median vig-free
mid over that state's rows; consecutive states are paired and solved by scanning the
split d = p-q and bisecting the sum s = p+q for the first price at each d.
"""
import sys, os, json, statistics as st, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel, _model

S_LO, S_HI = 0.60, 1.35
D_LO, D_HI = -0.20, 0.60


def price_at(s, d, state, bo, tb):
    return _model((s + d) / 2, (s - d) / 2, state, bo, tb)


def solve_sum(d, target, state, bo, tb):
    lo, hi = S_LO, S_HI
    if price_at(lo, d, state, bo, tb) > target or price_at(hi, d, state, bo, tb) < target:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if price_at(mid, d, state, bo, tb) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def solve_pair(px1, st1, px2, st2, bo, tb, n=21):
    """(p, q) such that model(st1)=px1 and model(st2)=px2, or None."""
    grid = [D_LO + (D_HI - D_LO) * i / (n - 1) for i in range(n)]
    res = []
    for d in grid:
        s = solve_sum(d, px1, st1, bo, tb)
        res.append(None if s is None else (price_at(s, d, st2, bo, tb) - px2, s))
    sols = []
    for i in range(n - 1):
        a, b = res[i], res[i + 1]
        if a is None or b is None or a[0] * b[0] > 0:
            continue
        lo, hi = grid[i], grid[i + 1]
        flo = a[0]
        for _ in range(20):
            m = (lo + hi) / 2
            s = solve_sum(m, px1, st1, bo, tb)
            if s is None: break
            fm = price_at(s, m, st2, bo, tb) - px2
            if fm * flo <= 0: hi = m
            else: lo, flo = m, fm
        d = (lo + hi) / 2
        s = solve_sum(d, px1, st1, bo, tb)
        if s is not None:
            sols.append(((s + d) / 2, (s - d) / 2))
    return sols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    lg = json.load(open(f"tapes/{a.event}.log.json"))
    bo, tb = lg["best_of"], lg.get("final_set_tb") or 7
    norm = ImpliedModel(best_of=bo, first_server="me", warn_split=False, final_set_tb=tb)

    # collapse rows to one (state, median price, fit) per point-state window
    windows = []
    for line in open(f"tapes/{a.event}.jsonl", errors="replace"):
        if '"status": "live"' not in line: continue
        r = json.loads(line)
        s = r.get("state") or {}
        if r.get("vig_free") is None or s.get("server") in (None, ""): continue
        key = tuple(s.get(k) for k in ("sets_me","sets_opp","games_me","games_opp","points_me","points_opp","server"))
        if windows and windows[-1][0] == key:
            windows[-1][2].append(r["vig_free"])
        else:
            windows.append([key, {k: v for k, v in s.items() if not k.startswith("_")}, [r["vig_free"]],
                            (r.get("fit") or {}).get("ewma2") or {}])
    lbl = ["0","15","30","40","AD"]
    def show(s):
        pm, po = s["points_me"], s["points_opp"]
        tbk = s["games_me"] == 6 and s["games_opp"] == 6
        pts = f"{pm}-{po}" if tbk else f"{lbl[pm] if pm < 5 else pm}-{lbl[po] if po < 5 else po}"
        return f"{s['sets_me']}-{s['sets_opp']} {s['games_me']}-{s['games_opp']} {pts:<6}{s['server']:<3}"

    out, nsolv, nmulti, nnone = [], 0, 0, 0
    print(f"{'#':>3} {'state before':<20} {'state after':<20} {'mkt1':>6} {'mkt2':>6} {'dP':>5} | "
          f"{'p':>6} {'q':>6} {'p+q':>6} {'p-q':>6} | {'roll p':>6} {'roll q':>6} | {'ewma2 p':>7} {'q':>6}")
    for i in range(len(windows) - 1):
        k1, s1, px1, fit = windows[i]
        k2, s2, px2, _ = windows[i + 1]
        if len(px1) < 1 or len(px2) < 1: continue
        m1, m2 = st.median(px1), st.median(px2)
        try:
            n1, n2 = norm._state(**s1), norm._state(**s2)
        except Exception:
            continue
        sols = solve_pair(m1, n1, m2, n2, bo, tb)
        if not sols: nnone += 1
        elif len(sols) > 1: nmulti += 1
        pq = sols[0] if len(sols) == 1 else None
        if pq: nsolv += 1; out.append(pq)
        rp = rq = float("nan")
        if out:
            w = out[-a.window:]
            rp, rq = st.median(x[0] for x in w), st.median(x[1] for x in w)
        if not a.quiet:
            pqs = f"{pq[0]:6.3f} {pq[1]:6.3f} {pq[0]+pq[1]:6.3f} {pq[0]-pq[1]:6.3f}" if pq else (
                  f"{'MULTI':>27}" if len(sols) > 1 else f"{'none':>27}")
            print(f"{i:>3} {show(s1):<20} {show(s2):<20} {m1*100:6.1f} {m2*100:6.1f} {(m2-m1)*100:+5.1f} | "
                  f"{pqs} | {rp:6.3f} {rq:6.3f} | {fit.get('p', float('nan')):7.3f} {fit.get('q', float('nan')):6.3f}")
    n = len(windows) - 1
    print(f"\n{a.event}: {n} point pairs, {nsolv} solved uniquely, {nmulti} multiple roots, {nnone} no solution")
    if out:
        ps, qs = [x[0] for x in out], [x[1] for x in out]
        print(f"implied p: median {st.median(ps):.3f}  IQR {st.quantiles(ps, n=4)[0]:.3f}-{st.quantiles(ps, n=4)[2]:.3f}")
        print(f"implied q: median {st.median(qs):.3f}  IQR {st.quantiles(qs, n=4)[0]:.3f}-{st.quantiles(qs, n=4)[2]:.3f}")
        sm = [x[0]+x[1] for x in out]; df = [x[0]-x[1] for x in out]
        print(f"p+q: median {st.median(sm):.3f} sd {st.pstdev(sm):.3f}   p-q: median {st.median(df):.3f} sd {st.pstdev(df):.3f}")


if __name__ == "__main__":
    main()
