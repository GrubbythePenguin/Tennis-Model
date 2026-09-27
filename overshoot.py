"""Did the market move MORE on a point than (p, q) says a point can move? And does it revert?

    python3 overshoot.py [--thr 1.0] [--fit ewma2]

Per point transition i -> i+1: model move = M(p,q,s_{i+1}) - M(p,q,s_i) using the fit
as of window i (the tape's live ewma2, or the pair5 point-level estimate); actual move =
median vig-free mid of window i+1 minus window i. Overshoot: same sign, |actual| exceeds
|model| by more than --thr cents. Fade = trade against the move at window i+1's price;
markout k = signed price change over the next k windows, in cents, positive = reversion.
"""
import sys, os, json, glob, statistics as st, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel, _model
from point_fit import windows_of, boundary_fit_series
from point_implied import solve_pair

KS = (1, 2, 3, 5, 10)


def run_match(path):
    lp = path[:-6] + ".log.json"
    lg = json.load(open(lp))
    bo, tb, prior = lg["best_of"], lg.get("final_set_tb") or 7, lg.get("split_prior", 0.2)
    norm = ImpliedModel(best_of=bo, first_server="me", warn_split=False, final_set_tb=tb)
    win = windows_of(path)
    states = []
    for k, s, px, f in win:
        try: states.append(norm._state(**s))
        except Exception: states.append(None)
    bf2 = boundary_fit_series(lg, win, norm, 2.0)
    rows = []
    for i in range(len(win) - 1):
        if states[i] is None or states[i+1] is None: continue
        est = {}
        if bf2[i] and bf2[i][2] >= 4:
            est["ewma2"] = bf2[i][:2]
        actual = (win[i+1][2] - win[i][2]) * 100
        fut = {k: (win[i+1+k][2] - win[i+1][2]) * 100 for k in KS if i + 1 + k < len(win)}
        for tag, (p, q) in est.items():
            mdl = (_model(p, q, states[i+1], bo, tb) - _model(p, q, states[i], bo, tb)) * 100
            rows.append((tag, actual, mdl, fut, win[i+1][2], os.path.basename(path).split("-")[0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    paths = [p for p in sorted(glob.glob("tapes/*.jsonl")) if os.path.exists(p[:-6] + ".log.json")]
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.imap_unordered(run_match, paths) for r in rs]
    for tag in ("ewma2",):
        R = [r[:5] for r in rows if r[0] == tag]
        RS = [r for r in rows if r[0] == tag]
        if not R: continue
        print(f"\n=== fit: {tag}   {len(R)} point transitions ===")
        for thr in a.thr:
            cats = {"overshoot": [], "on-model": [], "undershoot": [], "wrong-sign": [], "flat(mdl~0)": []}
            for _, act, mdl, fut, px in R:
                if abs(mdl) < 0.25: cats["flat(mdl~0)"].append((act, mdl, fut, px))
                elif act * mdl < 0: cats["wrong-sign"].append((act, mdl, fut, px))
                elif abs(act) > abs(mdl) + thr: cats["overshoot"].append((act, mdl, fut, px))
                elif abs(act) < abs(mdl) - thr: cats["undershoot"].append((act, mdl, fut, px))
                else: cats["on-model"].append((act, mdl, fut, px))
            print(f"\n-- threshold {thr:.0f}c --")
            print(f"{'category':<13}{'n':>7}{'share':>7}{'|act|':>7}{'|mdl|':>7} | fade markout, cents (+ = reverted)  "
                  + "".join(f"{'k='+str(k):>8}" for k in KS) + f"{'n(k=5)':>8}")
            for c, v in cats.items():
                if not v: continue
                mo = {}
                for k in KS:
                    x = [(-1 if act > 0 else 1) * fut[k] for act, mdl, fut, px in v if k in fut and act != 0]
                    mo[k] = (st.mean(x), len(x)) if x else (float("nan"), 0)
                print(f"{c:<13}{len(v):>7}{len(v)/len(R)*100:>6.0f}%{st.mean(abs(x[0]) for x in v):>7.2f}"
                      f"{st.mean(abs(x[1]) for x in v):>7.2f} | {'':34}"
                      + "".join(f"{mo[k][0]:>+8.2f}" for k in KS) + f"{mo[5][1]:>8}")
        print("\n-- by series: markout of trading TOWARD the model's move (k=5), thr 1c --")
        for ser in sorted(set(r[5] for r in RS)):
            S = [r for r in RS if r[5] == ser]
            line = f"{ser:<24} n {len(S):>5} |"
            for name, cond in (("overshoot", lambda a, m: a*m > 0 and abs(a) > abs(m)+1),
                               ("undershoot", lambda a, m: a*m > 0 and abs(a) < abs(m)-1),
                               ("wrong-sign", lambda a, m: a*m < 0)):
                x = [(1 if m > 0 else -1) * f[5] for _, a, m, f, px, _ in S if abs(m) >= 0.25 and cond(a, m) and 5 in f]
                line += f"  {name} n={len(x):>4} {st.mean(x) if x else float('nan'):+5.2f}c"
            print(line)
        # overshoot by size of excess, thr 1c
        print(f"\n-- overshoot fade by size of excess (|actual| - |model|), {tag} --")
        v = [(act, mdl, fut) for _, act, mdl, fut, px in R if abs(mdl) >= 0.25 and act * mdl > 0 and abs(act) > abs(mdl) + 1.0]
        for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 99)):
            b = [x for x in v if lo <= abs(x[0]) - abs(x[1]) < hi]
            if not b: continue
            line = f"excess {lo}-{hi}c  n {len(b):>5} | "
            for k in KS:
                x = [(-1 if act > 0 else 1) * fut[k] for act, mdl, fut in b if k in fut]
                line += f" k={k}: {st.mean(x):+5.2f}" if x else ""
            print(line)
        # what happens after a wrong-sign move: continuation or reversal?
        print(f"\n-- price at window i+1 vs the MODEL price at i+1: does the residual revert? --")
        # residual = actual - model move; fade residual regardless of category
        for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 99)):
            b = [(act, mdl, fut) for _, act, mdl, fut, px in R if lo <= abs(act - mdl) < hi]
            if not b: continue
            line = f"|act-mdl| {lo}-{hi}c  n {len(b):>5} | "
            for k in KS:
                x = [(-1 if act - mdl > 0 else 1) * fut[k] for act, mdl, fut in b if k in fut]
                line += f" k={k}: {st.mean(x):+5.2f}" if x else ""
            print(line)


if __name__ == "__main__":
    main()
