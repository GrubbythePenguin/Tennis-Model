"""P&L by edge bucket, weighted so no single match can dominate.

    python3 pnl_buckets.py                       # all variants, match-weighted
    python3 pnl_buckets.py --variant ewma2 --detail
    python3 pnl_buckets.py --edges 1 2 4 8       # custom bucket boundaries in cents

THE QUESTION. Not "did the model make money" - one lucky match answers that either way -
but "does realised P&L RISE with the size of the disagreement". A real edge is monotone:
a 1c divergence should earn less than an 8c divergence. A flat or ragged profile across
buckets means the divergences carry no information no matter what the total says.

WHY WEIGHTING IS THE WHOLE PROBLEM. Every boundary inside a match settles on the SAME
outcome, so a 30-boundary match contributes 30 perfectly correlated trades while a
12-boundary match contributes 12. Trade-weighting therefore hands long matches several
times the influence of short ones, and a single mispriced match can invert the sign of
the total. That is exactly what happened on 26AUG25: the historical book said ewma4
+$429 / static -$1, the live captures said static +$550 / ewma4 -$219, and in both
samples two matches carried the entire spread.

So every bucket is summarised MATCH-WEIGHTED: reduce each match to one number inside
the bucket (its mean P&L per trade there), then average across matches. Each match gets
one vote regardless of length. The trade-weighted column is printed alongside so the
distortion is visible rather than assumed away - where the two disagree sharply, a few
matches are driving the result.

Reported per bucket: match-weighted mean, MEDIAN (immune to one runaway match), how
many matches were positive, an exact sign test, and a cluster-bootstrap CI over matches.
Read the median and the match count first; the mean is the fragile one.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_divergence import fee
import beta_report as BR

HERE = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(HERE, "tapes")
HDR = re.compile(r'^\[\d\d:\d\d:\d\d\].*?mkt\s+([\d.]+)\s')
VAR = re.compile(r'^\s+(static|rolling|ewma[\d.]+)\s+pred\s+([\d.]+)\s+err')


def trade_pnl(mkt, mdl, settle, size):
    """Buy the side the model prefers at the market mid, hold to settlement.

    Entry fee only (no exit - held to settlement). Fees are largest at p=0.5, which is
    where most divergences live, so ignoring them flatters everything.
    """
    edge = mdl - mkt
    long_ = edge > 0
    entry = mkt if long_ else 1 - mkt
    payoff = settle if long_ else 1 - settle
    return size * (payoff - entry) - fee(size, entry)


def collect(lo, hi, halflives, include_book):
    """match -> variant -> [(edge_cents, pnl_per_contract_set)]"""
    settle = BR.load_cache()
    out = {}
    for f in sorted(glob.glob(os.path.join(TAPES, "watch_*.log"))):
        nm = os.path.basename(f)[6:-4]
        s = settle.get(nm)
        if s is None:
            s = BR.settle_from_capture(f)
            if s is None:
                continue
            settle[nm] = s
        mk, per = None, {}
        for ln in open(f, errors="replace").read().replace("\r", "\n").splitlines():
            h = HDR.match(ln)
            if h:
                mk = float(h.group(1))
                continue
            m = VAR.match(ln)
            if not m or mk is None or not (lo < mk < hi):
                continue
            per.setdefault(m.group(1), []).append((mk, float(m.group(2)), s))
        if per:
            out[nm] = per
    BR.save_cache(settle)
    if include_book:
        try:
            from backtest_divergence import BOOK
            from backsim import live_replay
            for label, stem, s in BOOK:
                p = os.path.join(TAPES, stem + ".json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                obs = d["obs"]
                preds = live_replay(obs, d["best_of"], d["split_prior"],
                                    d.get("final_set_tb") or 7, halflives)
                per = {}
                for v, series in preds.items():
                    for i, f2 in enumerate(series):
                        mk = obs[i]["price"]
                        if f2 is None or not (lo < mk < hi):
                            continue
                        per.setdefault(v, []).append((mk, f2, s))
                if per:
                    out["BOOK:" + label] = per
        except Exception as e:
            print(f"(book replay unavailable: {e})", file=sys.stderr)
    return out


def sign_test(vals):
    nz = [v for v in vals if abs(v) > 1e-9]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for v in nz if v > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return k, n, min(1.0, 2 * tail)


def boot_ci(per_match, draws, rng):
    """Cluster bootstrap over matches on the match-weighted mean."""
    if len(per_match) < 3:
        return None
    bs = []
    for _ in range(draws):
        s = [rng.choice(per_match) for _ in per_match]
        bs.append(sum(s) / len(s))
    bs.sort()
    return bs[int(len(bs) * .025)], bs[int(len(bs) * .975)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--edges", nargs="+", type=float, default=[1, 2, 4, 8],
                    help="bucket boundaries in CENTS of |model - market|")
    ap.add_argument("--range", nargs=2, type=float, default=[0.15, 0.85], metavar=("LO", "HI"))
    ap.add_argument("--variant", default=None, help="restrict to one variant")
    ap.add_argument("--halflives", nargs="+", default=["2", "4"])
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--signed", action="store_true",
                    help="split buckets by direction (model above vs below market) "
                         "instead of pooling on |edge|")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260825)
    a = ap.parse_args()
    lo, hi = a.range
    rng = random.Random(a.seed)

    data = collect(lo, hi, a.halflives, not a.live_only)
    if not data:
        print("no settled matches yet")
        return 0
    variants = sorted({v for m in data.values() for v in m},
                      key=lambda v: (v != "static", v != "rolling", v))
    if a.variant:
        variants = [v for v in variants if v == a.variant]

    cuts = sorted(a.edges)
    def bucket(e):
        ae = abs(e)
        for i, c in enumerate(cuts):
            if ae < c:
                return i
        return len(cuts)
    def blabel(i):
        if i == 0:
            return f"<{cuts[0]:.0f}c"
        if i == len(cuts):
            return f"{cuts[-1]:.0f}c+"
        return f"{cuts[i-1]:.0f}-{cuts[i]:.0f}c"

    nlive = sum(1 for m in data if not m.startswith("BOOK:"))
    print(f"P&L by edge bucket, size {a.size}, held to settlement, net of entry fees")
    print(f"{len(data)} settled matches ({nlive} live, {len(data)-nlive} book), "
          f"market in {lo:.2f}-{hi:.2f}\n")

    for v in variants:
        groups = {}          # bucket -> match -> [pnl per trade]
        for nm, per in data.items():
            for mk, mdl, s in per.get(v, []):
                e = (mdl - mk) * 100
                key = (bucket(e), (1 if e > 0 else -1) if a.signed else 0)
                groups.setdefault(key, {}).setdefault(nm, []).append(
                    trade_pnl(mk, mdl, s, a.size))
        print(f"=== {v} ===")
        print(f"  {'edge bucket':>14} {'trades':>7} {'matches':>8} {'trade-wtd $':>12}"
              f" {'MATCH-wtd $':>12} {'median $':>10} {'matches +':>11} {'sign p':>8}"
              f" {'95% CI (match-wtd)':>24}")
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            g = groups[key]
            permatch = [sum(x) / len(x) for x in g.values()]
            alltr = [x for lst in g.values() for x in lst]
            if len(g) < 3:
                continue
            k, n, p = sign_test(permatch)
            ci = boot_ci(permatch, a.boot, rng)
            cis = f"[{ci[0]:8.2f},{ci[1]:8.2f} ]" if ci else ""
            permatch.sort()
            med = (permatch[len(permatch)//2] if len(permatch) % 2
                   else (permatch[len(permatch)//2-1] + permatch[len(permatch)//2]) / 2)
            lab = blabel(key[0]) + (" up" if key[1] > 0 else (" dn" if key[1] < 0 else ""))
            print(f"  {lab:>14} {len(alltr):>7} {len(g):>8} {sum(alltr)/len(alltr):>12.2f}"
                  f" {sum(permatch)/len(permatch):>12.2f} {med:>10.2f}"
                  f" {str(k)+'/'+str(n):>11} {p:>8.3f} {cis:>24}")
        print()

    print("A real edge is MONOTONE: bigger disagreement -> bigger P&L. A flat or ragged")
    print("profile means the divergences carry no information, whatever the total says.")
    print("Read MATCH-wtd and median first. Where trade-wtd and match-wtd disagree")
    print("sharply, a few long matches are driving that bucket.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
