"""EWMA-weighted rolling fit, with optional outlier rejection — does it help?

    python3 ewma_fit.py [--halflives 2 4 8 16 inf] [--outlier 0 3 5]

WHY. The rolling fit weights every boundary equally, so a bad pre-match anchor keeps
full influence for the whole match (Koevermans: a -7c static bias that never washed
out). Exponentially weighting recent prices should let the fit follow a genuine
re-rating instead of averaging it away.

THE TENSION. EWMA adapts to change; outlier rejection resists it. Parry raising her
level after losing a tiebreak set is indistinguishable, at the moment it happens,
from a bad print — one should be followed and the other ignored, and the fit cannot
tell them apart from price alone. Running both together can cancel out.

    halflife h  ->  observation i gets weight 0.5 ** ((n - i) / h)
    halflife inf = the current equal-weight rolling fit
    outlier k   = skip a boundary whose |market - model| exceeds k * running MAD
                  of past errors (0 disables). Skipped prices are still PREDICTED,
                  so they count in the error stats — they just do not update the fit.

REPORTED. Tracking rms AND backtest P&L, because this session established they are
different criteria: a fit that tracks perfectly has no tradeable disagreement by
construction. A configuration that improves rms while reducing P&L is doing exactly
what the equal-weight rolling fit already does wrong.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel
from backtest_divergence import BOOK, fee


def fit_series(obs, best_of, prior, tb, halflife, outlier_k):
    """Walk-forward. Returns (errors_c, trades) using EWMA weights and outlier skips.

    INCREMENTAL. This used to discard the model and replay every kept boundary on each
    step, which is O(n^2) fits — 465 of them for a 30-boundary match instead of 30, and
    the single largest reason a full grid took 11 minutes. It is unnecessary: the weight
    vector 0.5**((n-1-j)/h) is reproduced exactly by decaying every stored weight by
    0.5**(1/h) and appending the new one at 1.0. Same arithmetic, one _fit() per
    boundary. Mirrors poll_tennis._ewma_update, which does the same thing live.

    The weights cannot just be left un-normalised to sidestep this: _residuals appends
    a PRIOR row at fixed weight, so scaling all observation weights by a constant
    quietly changes how hard the split prior pulls.
    """
    errs, trades, past = [], [], []
    n_kept = 0
    dec = 1.0 if halflife is None else 0.5 ** (1.0 / halflife)
    m = ImpliedModel(best_of=best_of, first_server="me", split_prior=prior,
                     warn_split=False, final_set_tb=tb)
    for i, o in enumerate(obs):
        pred = None
        if n_kept:
            try:
                pred = m.price(**o["state"])[0]
            except Exception:
                pred = None
        if pred is not None:
            e = (o["price"] - pred) * 100
            errs.append(e)
            trades.append((o["price"], pred))
            # outlier test against the running spread of past errors
            drop = False
            if outlier_k and len(past) >= 4:
                mad = st.median([abs(x - st.median(past)) for x in past]) or 1e-9
                if abs(e - st.median(past)) > outlier_k * 1.4826 * mad:
                    drop = True
            past.append(e)
            if drop:
                continue                      # predicted, but does not update the fit
        # fold this boundary in: age every existing weight, then append at 1.0
        if dec != 1.0:
            m.obs = [(s, p, max(w * dec, 1e-6)) for s, p, w in m.obs]
        m.observe(o["price"], weight=1.0, **o["state"])
        n_kept += 1
    return errs, trades


def pnl(trades, settle, size=100, thresh=0.02):
    tot = 0.0
    n = 0
    for mkt, mdl in trades:
        e = mdl - mkt
        if abs(e) < thresh:
            continue
        lng = e > 0
        entry = mkt if lng else 1 - mkt
        pay = settle if lng else 1 - settle
        tot += size * (pay - entry) - fee(size, entry)
        n += 1
    return tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflives", nargs="+", default=["2", "4", "8", "16", "inf"])
    ap.add_argument("--outlier", nargs="+", type=float, default=[0.0, 3.0])
    ap.add_argument("--threshold", type=float, default=0.02)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print(f"{'halflife':>9} {'outlier':>8} | {'rms c':>7} {'|mean| c':>9} | "
          f"{'P&L $':>10} {'trades':>7} {'matches +':>10}")
    for hl_s in a.halflives:
        hl = None if hl_s == "inf" else float(hl_s)
        for k in a.outlier:
            all_err, tot, ntr, per = [], 0.0, 0, []
            for label, f, settle in BOOK:
                p = os.path.join(here, "tapes", f + ".json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                tb = d.get("final_set_tb") or 7
                errs, trades = fit_series(d["obs"], d["best_of"], d["split_prior"],
                                          tb, hl, k)
                all_err += errs[1:]           # drop the first (fit on 1 point)
                m, n = pnl(trades[1:], settle, thresh=a.threshold)
                tot += m
                ntr += n
                per.append(m)
            rms = (sum(x * x for x in all_err) / len(all_err)) ** 0.5 if all_err else 0
            mn = abs(sum(all_err) / len(all_err)) if all_err else 0
            print(f"{hl_s:>9} {k:8.0f} | {rms:7.2f} {mn:9.2f} | "
                  f"{tot:10.2f} {ntr:7d} {sum(1 for x in per if x>0):>6}/{len(per)}")


if __name__ == "__main__":
    main()
