"""Per-match P&L for every fit variant, with a paired test across matches.

    python3 per_match_pnl.py [--size 100] [--threshold 0.02] [--halflives 4 8]

WHY THIS EXISTS. backtest_divergence.py prints per-match P&L for STATIC vs ROLLING and
ewma_fit.py prints grid TOTALS for the EWMA variants, so the two never met: the EWMA
configurations were only ever compared on a total, which is the one number the
docstring of backtest_divergence.py explicitly warns against reading.

THE SAMPLE IS MATCHES, NOT TRADES. Every boundary inside a match settles on the same
outcome, so being long the eventual winner wins every trade in that match. ~100 trades
across 10 matches is 10 independent bets, not 100. This script therefore treats a match
as the unit of observation: one net P&L number per match per variant, then a PAIRED
comparison, because the same match is priced by every variant and match difficulty is
the dominant source of variance. Pairing removes it; comparing totals does not.

The 26AUG25 grid found EWMA improves tracking error monotonically as the halflife
shortens (rms 3.58 -> 2.96 from inf to 2) while P&L peaked at halflife 4 and fell away
either side. Those two facts cannot both be the criterion, and the rms one is the one
with real degrees of freedom behind it — hence the sign test here, which asks the only
question a 10-match sample can answer: does this variant beat the baseline on MORE
MATCHES than not, more often than a coin would?

FUTURE MATCHES. The match list is backtest_divergence.BOOK. Add a settled match there
(rebuilt log stem + settlement of the tracked side) and it is picked up by this script,
by backtest_divergence.py and by ewma_fit.py with no other change.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_divergence import BOOK, run as run_static_rolling
from ewma_fit import fit_series, pnl


def sign_test(diffs):
    """Two-sided exact binomial on the count of positive differences.

    Ties are dropped (standard sign-test convention) rather than split, because a
    match where two variants fire identical trades carries no evidence either way.
    Deliberately NOT a t-test: 10 matches of highly skewed P&L is not normal, and the
    sign test needs no distributional assumption to be honest.
    """
    nz = [d for d in diffs if abs(d) > 1e-9]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for d in nz if d > 0)
    # P(X as or more extreme than k) under Binom(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return k, n, min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="minimum |model-market| in DOLLARS to trade (matches ewma_fit's default)")
    ap.add_argument("--halflives", nargs="+", default=["2", "4", "8"],
                    help="EWMA halflives to include alongside static and rolling")
    ap.add_argument("--outlier", type=float, default=0.0,
                    help="outlier-rejection k (0 = off; the 26AUG25 grid found k=3 hurt rms "
                         "in all 5 halflife pairs)")
    ap.add_argument("--baseline", default="static",
                    help="variant every other variant is paired against")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    cols = ["static", "rolling"] + [f"ewma{h}" for h in a.halflives]
    per = {c: [] for c in cols}          # per-match net P&L, aligned with `names`
    ntr = {c: [] for c in cols}          # per-match trade count
    names, settles = [], []

    for label, stem, settle in BOOK:
        p = os.path.join(here, "tapes", stem + ".json")
        if not os.path.exists(p):
            continue
        names.append(label)
        settles.append(settle)

        # STATIC / ROLLING — reuse backtest_divergence's own accounting verbatim so
        # these numbers are the same ones that script prints, not a reimplementation.
        r = run_static_rolling(p, settle, a.size, a.threshold)
        for tag in ("static", "rolling"):
            per[tag].append(sum(x["net"] for x in r[tag]))
            ntr[tag].append(len(r[tag]))

        # EWMA — reuse ewma_fit's fit_series/pnl for the same reason. trades[1:] drops
        # the first prediction (fit on a single point), which is exactly what
        # backtest_divergence's `if i < 2: continue` drops, so all variants trade the
        # same set of boundaries.
        d = json.load(open(p))
        tb = d.get("final_set_tb") or 7
        for h in a.halflives:
            hl = None if h == "inf" else float(h)
            _, trades = fit_series(d["obs"], d["best_of"], d["split_prior"], tb, hl, a.outlier)
            m, n = pnl(trades[1:], settle, size=a.size, thresh=a.threshold)
            per[f"ewma{h}"].append(m)
            ntr[f"ewma{h}"].append(n)

    if not names:
        print("no settled matches found in tapes/ — nothing to compare")
        return

    print(f"size {a.size} contracts, threshold {a.threshold*100:.1f}c, outlier k={a.outlier:.0f}, "
          f"held to settlement, net of Kalshi entry fees")
    print(f"{len(names)} settled matches = {len(names)} independent bets\n")

    head = f"{'match':24}{'set':>4} |" + "".join(f"{c:>11}" for c in cols)
    print(head)
    print("-" * len(head))
    for i, nm in enumerate(names):
        row = f"{nm[:24]:24}{settles[i]:>4} |"
        for c in cols:
            row += f"{per[c][i]:>11.2f}"
        print(row)
    print("-" * len(head))
    print(f"{'TOTAL':24}{'':>4} |" + "".join(f"{sum(per[c]):>11.2f}" for c in cols))
    print(f"{'mean / match':24}{'':>4} |" + "".join(f"{sum(per[c])/len(names):>11.2f}" for c in cols))
    print(f"{'matches profitable':24}{'':>4} |"
          + "".join(f"{str(sum(1 for x in per[c] if x > 0)) + '/' + str(len(names)):>11}" for c in cols))
    print(f"{'trades':24}{'':>4} |" + "".join(f"{sum(ntr[c]):>11d}" for c in cols))
    # Spread across matches is the whole point: a total is meaningless next to it.
    print(f"{'best match':24}{'':>4} |" + "".join(f"{max(per[c]):>11.2f}" for c in cols))
    print(f"{'worst match':24}{'':>4} |" + "".join(f"{min(per[c]):>11.2f}" for c in cols))

    base = a.baseline
    if base not in per:
        print(f"\nbaseline '{base}' not among {cols}")
        return
    print(f"\nPAIRED vs {base.upper()} — same matches, same boundaries, difference per match")
    print(f"{'variant':12}{'mean diff $':>13}{'median diff $':>15}{'won':>7}{'lost':>6}"
          f"{'tied':>6}{'sign p':>9}")
    for c in cols:
        if c == base:
            continue
        diffs = [per[c][i] - per[base][i] for i in range(len(names))]
        k, n, p = sign_test(diffs)
        srt = sorted(diffs)
        med = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt)//2 - 1] + srt[len(srt)//2]) / 2
        ties = len(diffs) - n
        print(f"{c:12}{sum(diffs)/len(diffs):>13.2f}{med:>15.2f}{k:>7}{n-k:>6}{ties:>6}{p:>9.3f}")

    print(f"\nWith {len(names)} matches the sign test cannot resolve anything below about a")
    print("9-1 split (p=0.021); 8-2 is p=0.109 and 7-3 is p=0.344. Treat every p above")
    print("0.05 here as 'this sample cannot tell these apart', NOT as evidence of no")
    print("difference — the test has almost no power at this n.")


if __name__ == "__main__":
    main()
