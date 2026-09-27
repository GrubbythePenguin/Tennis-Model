"""Can the model predict the market's move on a single point? By (p, q) source.

    LS_ONLY=1 python3 delta_test.py --procs 4

Per point transition i -> i+1 (window = one point state, price = median vig-free mid):
    actual    = price(i+1) - price(i)                         cents
    predicted = M(p,q,state_{i+1}) - M(p,q,state_i)           using the REALISED branch
(p, q) as of window i from: the boundary EWMA fit (halflife 2 / 0.5 games), joint LS on
point windows (halflife 4 / 6 points), the pairwise two-snapshot median (5), and a zero
baseline. Scored: MAE, rms, share within 0.5c / 1c, and sign(|pred| - |actual|) bias
(negative = the bracket is narrower than the market's real move).
"""
import sys, os, json, glob, statistics as st, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel, _model
from point_fit import windows_of, boundary_fit_series
from point_implied import solve_pair

LS_H = (4, 6)


def run_match(path):
    lp = path[:-6] + ".log.json"
    lg = json.load(open(lp)); bo, tb, prior = lg["best_of"], lg.get("final_set_tb") or 7, lg.get("split_prior", 0.2)
    norm = ImpliedModel(best_of=bo, first_server="me", warn_split=False, final_set_tb=tb)
    win = windows_of(path)
    if len(win) < 30: return None
    states = []
    for k, s, px, f in win:
        try: states.append(norm._state(**s))
        except Exception: states.append(None)
    bf2 = boundary_fit_series(lg, win, norm, 2.0); bf05 = boundary_fit_series(lg, win, norm, 0.5)
    ls = {h: ImpliedModel(best_of=bo, first_server="me", split_prior=prior, warn_split=False, final_set_tb=tb) for h in LS_H}
    pairs = []
    out = []
    ser = os.path.basename(path).split("-")[0]
    for i in range(int(os.environ.get('MINWIN', '20')), len(win) - 1):
        if states[i] is None or states[i+1] is None: continue
        actual = (win[i+1][2] - win[i][2]) * 100
        est = {"zero": None}
        MINB = int(os.environ.get("MINB", "8"))
        if bf2[i] and bf2[i][2] >= MINB:
            est["ewma2"] = bf2[i][:2]; est["ewma0.5"] = bf05[i][:2]
        else:
            continue                                   # score every source on the same transitions
        for h, m in ({} if os.environ.get("NO_LS") else ls).items():
            dec = 0.5 ** (1.0 / h)
            m.obs = [(states[j], win[j][2], dec ** (i - 1 - j)) for j in range(max(0, i - 4 * h), i) if states[j] is not None]
            if len(m.obs) >= 6:
                try: m._fit(); est[f"ls{h}"] = (m.p, m.q)
                except Exception: pass
        if not os.environ.get("LS_ONLY"):
            past = [x for x in pairs if x]
            if len(past) >= 5:
                w = past[-5:]; est["pair5"] = (st.median(a for a, b in w), st.median(b for a, b in w))
            sol = solve_pair(win[i-1][2], states[i-1], win[i][2], states[i], bo, tb) if states[i-1] else []
            pairs.append(sol[0] if len(sol) == 1 else None)
        same_game = win[i][0][:4] == win[i+1][0][:4]
        preds, up, dn = {}, {}, {}
        s0 = win[i][1]
        try:
            sw = dict(s0); sw["points_me"] = s0.get("points_me", 0) + 1
            sl = dict(s0); sl["points_opp"] = s0.get("points_opp", 0) + 1
            nw, nl = norm._state(**sw), norm._state(**sl)
        except Exception:
            continue
        for tag, pq in est.items():
            if pq is None: preds[tag] = 0.0; up[tag] = 0.0; dn[tag] = 0.0; continue
            p, q = pq
            now = _model(p, q, states[i], bo, tb) * 100
            preds[tag] = _model(p, q, states[i+1], bo, tb) * 100 - now
            a_, b_ = _model(p, q, nw, bo, tb) * 100 - now, _model(p, q, nl, bo, tb) * 100 - now
            up[tag], dn[tag] = max(a_, b_), min(a_, b_)          # bracket edges relative to the anchor
        out.append(dict(actual=actual, preds=preds, up=up, dn=dn, same_game=same_game, ser=ser, px=win[i][2], key=win[i][0], nkey=win[i+1][0], nb=bf2[i][2]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    paths = [p for p in sorted(glob.glob("tapes/*.jsonl")) if os.path.exists(p[:-6] + ".log.json")]
    with Pool(a.procs) as pool:
        res = [x for r in pool.imap_unordered(run_match, paths) if r for x in r]
    json.dump(res, open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "delta_test.json"), "w"))
    tags = [t for t in ("zero", "ewma2", "ewma0.5", "ls4", "ls6", "pair5") if all(t in r["preds"] for r in res[:50])]
    if os.environ.get("NO_LS"): return
    def table(lbl, R):
        print(f"\n{lbl}: {len(R)} point transitions   (actual move: mean |a| {st.mean(abs(r['actual']) for r in R):.2f}c)")
        print(f"  {'source':<9}{'MAE':>7}{'rms':>7}{'<=0.5c':>8}{'<=1c':>7}{'<=2c':>7}  {'|pred|-|act|':>13}  {'sign ok':>8}")
        for t in tags:
            e = [r["preds"][t] - r["actual"] for r in R if t in r["preds"]]
            if not e: continue
            n = len(e); mag = [abs(r["preds"][t]) - abs(r["actual"]) for r in R if t in r["preds"]]
            sg = [r for r in R if t in r["preds"] and abs(r["preds"][t]) > 0.25 and abs(r["actual"]) > 0.25]
            sok = sum(1 for r in sg if r["preds"][t] * r["actual"] > 0) / len(sg) * 100 if sg else float("nan")
            print(f"  {t:<9}{st.mean(abs(x) for x in e):>7.2f}{(st.mean(x*x for x in e))**0.5:>7.2f}"
                  f"{sum(1 for x in e if abs(x) <= 0.5)/n*100:>7.0f}%{sum(1 for x in e if abs(x) <= 1)/n*100:>6.0f}%{sum(1 for x in e if abs(x) <= 2)/n*100:>6.0f}%"
                  f"  {st.mean(mag):>+13.2f}  {sok:>7.0f}%")
    table("ALL", res)
    table("within a game (point -> point)", [r for r in res if r["same_game"]])
    table("game-ending points", [r for r in res if not r["same_game"]])
    for ser in sorted(set(r["ser"] for r in res)):
        table(ser, [r for r in res if r["ser"] == ser])
    # by size of the actual move: where do the errors live?
    best = "ls4" if "ls4" in tags else "ewma2"
    print(f"\nerror by size of actual move ({best} vs ewma2), cents")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99)):
        B = [r for r in res if lo <= abs(r["actual"]) < hi]
        if not B: continue
        print(f"  |actual| {lo}-{hi}c  n {len(B):>5}  ewma2 MAE {st.mean(abs(r['preds']['ewma2'] - r['actual']) for r in B):5.2f}  {best} MAE {st.mean(abs(r['preds'][best] - r['actual']) for r in B if best in r['preds']):5.2f}   mean |pred| {st.mean(abs(r['preds'][best]) for r in B if best in r['preds']):5.2f}")


if __name__ == "__main__":
    main()
