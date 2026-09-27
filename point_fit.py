"""Walk-forward: (p, q) from every point snapshot vs the boundary-only EWMA fit.

    python3 point_fit.py [--all] [EVENT ...]

Per match, rows are collapsed to one window per point-state (median vig-free mid).
At window i every method uses only windows < i, prices window i, and is scored on
(model - market) in cents. Methods:

    ewma2        the live boundary-only fit as written in the tape (baseline)
    pairK        (p, q) = medians of the last K pairwise two-snapshot solves
    sumK         p+q = median of the last K pairwise solves, p-q pinned at the tour prior
    lsH          joint least squares on the last windows, halflife H POINTS

RESULT (26AUG27, 122 matches / 9,381 point windows, walk-forward, same windows):

    method            MAE    rms   maxrun  matches >1c off all match
    ewma2 (live)     1.92   2.95     22        28%
    ewma0.5          1.86   3.04     14        15%     <- boundary halflife barely matters
    pair5            1.80   3.09     12.5      13%
    sum5 (pinned)    2.02   3.33     16        28%     <- split carries info at point level
    ls3              1.33   2.59      9         3%
    ls4              1.39   2.69     11         4%
    ls6              1.45   2.46     13         8%
    ls8              1.53   2.55     14        11%

The lag is in WHAT the fit observes (game boundaries), not how fast it forgets. A joint
LS fit on every point window with a 3-6 POINT halflife cuts MAE by a quarter to a third
and the longest one-sided run from 22 points to 9-13. MAE and one-sidedness improve
monotonically toward 3; rms (tail misses) bottoms at 6. Shift test: aligned pairs solve
7,713 vs 3,581 shifted - the scoreboard is NOT systematically a point behind the price.

Also the SHIFT test: pair state i with the price of window i+1. If the market is a
point ahead of the scoreboard, the shifted pairing should solve more often and with
less dispersion than the aligned one.
"""
import sys, os, json, glob, gzip, statistics as st, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel, _model
from point_implied import solve_pair

MIN_WIN = 20                     # score windows from here on (lets every method warm up)
METHODS = ["ewma2", "ewma0.5", "pair5", "pair10", "sum5", "sum10", "ls3", "ls4", "ls6", "ls8", "ls12"]
LS_ONLY = os.environ.get("LS_ONLY") == "1"      # skip the pairwise solves (slow) for a halflife-only pass


def windows_of(path):
    op = gzip.open if path.endswith(".gz") else open
    win = []
    for line in op(path, "rt", errors="replace"):
        if '"status": "live"' not in line: continue
        try: r = json.loads(line)
        except Exception: continue
        s = r.get("state") or {}
        if r.get("vig_free") is None or s.get("server") in (None, "") or s.get("sets_me") is None: continue
        key = tuple(s.get(k) for k in ("sets_me","sets_opp","games_me","games_opp","points_me","points_opp","server"))
        if win and win[-1][0] == key:
            win[-1][2].append(r["vig_free"])
        else:
            win.append([key, {k: v for k, v in s.items() if not k.startswith("_")}, [r["vig_free"]],
                        (r.get("fit") or {}).get("ewma2") or {}])
    return [(k, s, st.median(px), f) for k, s, px, f in win]


def boundary_fit_series(lg, win, norm, halflife=2.0):
    """Per-window (p, q, n) from the BOUNDARY-only EWMA fit, reconstructed walk-forward
    from the log's obs: the price at game G's start is folded in when the window's game
    reaches G, so every window is priced by a fit that has seen boundaries up to and
    including the start of its own game — what the live poller had. Needed because tapes
    from pre-armed pollers carry no `fit` field."""
    bo, tb, prior = lg["best_of"], lg.get("final_set_tb") or 7, lg.get("split_prior", 0.2)
    m = ImpliedModel(best_of=bo, first_server="me", split_prior=prior, warn_split=False, final_set_tb=tb)
    obs = lg.get("obs") or []
    gk = lambda s: (s.get("sets_me"), s.get("sets_opp"), s.get("games_me"), s.get("games_opp"))
    dec = 0.5 ** (1.0 / halflife)
    out, j, n = [], 0, 0
    for k, s, px, f in win:
        while j < len(obs) and not _precedes(gk(s), gk(obs[j]["state"])):
            m.obs = [(a, b, max(w * dec, 1e-6)) for a, b, w in m.obs]
            try:
                m.observe(obs[j]["price"], **obs[j]["state"]); n += 1
            except Exception:
                pass
            j += 1
        out.append((m.p, m.q, n) if m.p is not None else None)
    return out


def _precedes(a, b):
    """game key a strictly before b in match order (sets first, then games)."""
    if None in a or None in b:
        return False
    return (a[0] + a[1], a[2] + a[3]) < (b[0] + b[1], b[2] + b[3])


def maxrun(errs):
    best = run = 0; last = 0
    for e in errs:
        sg = (e > 0) - (e < 0)
        run = run + 1 if sg and sg == last else (1 if sg else 0)
        last = sg; best = max(best, run)
    return best


def run_match(path):
    lp = (path[:-3] if path.endswith(".gz") else path)[:-6] + ".log.json"
    lg = json.load(open(lp))
    bo, tb, prior = lg["best_of"], lg.get("final_set_tb") or 7, lg.get("split_prior", 0.2)
    norm = ImpliedModel(best_of=bo, first_server="me", warn_split=False, final_set_tb=tb)
    win = windows_of(path)
    if len(win) < MIN_WIN + 10: return None
    states = []
    for k, s, px, f in win:
        try: states.append(norm._state(**s))
        except Exception: states.append(None)

    bf2 = boundary_fit_series(lg, win, norm, 2.0)
    bf05 = boundary_fit_series(lg, win, norm, 0.5)
    # pairwise solves, aligned and shifted by one window
    pairs, shifted = [None]*len(win), [None]*len(win)
    for i in range(len(win) - 1):
        if LS_ONLY: break
        if states[i] is None or states[i+1] is None: continue
        sol = solve_pair(win[i][2], states[i], win[i+1][2], states[i+1], bo, tb)
        if len(sol) == 1: pairs[i+1] = sol[0]           # known once window i+1's price is seen
        if i + 2 < len(win) and states[i+2] is not None:
            sol = solve_pair(win[i+1][2], states[i], win[i+2][2], states[i+1], bo, tb)
            if len(sol) == 1: shifted[i+2] = sol[0]

    def ps_stats(lst):
        v = [x for x in lst if x]
        if len(v) < 5: return (len(v), float("nan"), float("nan"))
        return (len(v), st.pstdev([a+b for a, b in v]), st.pstdev([a-b for a, b in v]))

    errs = {m: [] for m in METHODS}
    ls = {h: ImpliedModel(best_of=bo, first_server="me", split_prior=prior, warn_split=False,
                          final_set_tb=tb) for h in ((3, 4, 6, 8) if LS_ONLY else (6, 12))}
    for i in range(MIN_WIN, len(win)):
        if states[i] is None: continue
        mkt = win[i][2]
        est = {}
        if bf2[i] and bf2[i][2] >= 8:
            est["ewma2"] = bf2[i][:2]
            est["ewma0.5"] = bf05[i][:2]
        past = [x for x in pairs[:i] if x]
        for K in (5, 10):
            if len(past) >= K:
                w = past[-K:]
                est[f"pair{K}"] = (st.median(a for a, b in w), st.median(b for a, b in w))
        for K in (5, 10):
            if len(past) >= K:
                s_ = st.median(a + b for a, b in past[-K:])
                est[f"sum{K}"] = ((s_ + prior) / 2, (s_ - prior) / 2)
        for h, m in ls.items():
            dec = 0.5 ** (1.0 / h)
            m.obs = [(states[j], win[j][2], dec ** (i - 1 - j)) for j in range(max(0, i - 4*h), i)
                     if states[j] is not None]
            if len(m.obs) >= 6:
                try:
                    m._fit(); est[f"ls{h}"] = (m.p, m.q)
                except Exception: pass
        if "ewma2" not in est: continue        # score all methods on the same windows
        for mth, (p, q) in est.items():
            errs[mth].append((_model(p, q, states[i], bo, tb) - mkt) * 100)

    out = {"ev": os.path.basename(path).split(".")[0], "n_win": len(win),
           "aligned": ps_stats(pairs), "shifted": ps_stats(shifted), "err": errs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.all:
        paths = sorted(glob.glob("tapes/*.jsonl"))
        paths = [p for p in paths if os.path.exists(p[:-6] + ".log.json")]
    else:
        paths = [f"tapes/{e}.jsonl" for e in a.events]
    with Pool(a.procs) as pool:
        res = [r for r in pool.imap_unordered(run_match, paths) if r]
    print(f"{len(res)} matches\n")
    json.dump(res, open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "point_fit_res.json"), "w"))
    print("SHIFT TEST (pairwise solves): n solved, sd(p+q), sd(p-q)")
    for lab in ("aligned", "shifted") if not LS_ONLY else ():
        n = sum(r[lab][0] for r in res)
        ss = st.median(r[lab][1] for r in res if r[lab][1] == r[lab][1])
        sd = st.median(r[lab][2] for r in res if r[lab][2] == r[lab][2])
        print(f"  {lab:<8} solved {n:5d}   median sd(p+q) {ss:.3f}   median sd(p-q) {sd:.3f}")
    print(f"\nNEXT-WINDOW PRICING ERROR, cents (model - market), same windows for every method")
    print(f"{'method':<8}{'windows':>8}{'MAE':>7}{'bias':>7}{'rms':>7}{'maxrun':>8}{'|bias|>1c':>11}")
    for m in METHODS:
        allv = [e for r in res for e in r["err"][m]]
        if not allv: continue
        per_bias = [st.mean(r["err"][m]) for r in res if r["err"][m]]
        per_run = [maxrun(r["err"][m]) for r in res if r["err"][m]]
        print(f"{m:<8}{len(allv):>8}{st.mean(abs(x) for x in allv):>7.2f}{st.mean(allv):>+7.2f}"
              f"{(st.mean(x*x for x in allv))**0.5:>7.2f}{st.median(per_run):>8.1f}"
              f"{sum(1 for b in per_bias if abs(b) > 1)/len(per_bias)*100:>10.0f}%")
    if len(res) <= 8:
        for r in res:
            print(f"\n{r['ev']}  windows {r['n_win']}")
            for m in METHODS:
                v = r["err"][m]
                if v: print(f"  {m:<8} n {len(v):>4} MAE {st.mean(abs(x) for x in v):5.2f} bias {st.mean(v):+5.2f} maxrun {maxrun(v)}")


if __name__ == "__main__":
    main()
