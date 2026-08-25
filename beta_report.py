"""Is the model's disagreement with the market informative? Estimate beta.

    python3 beta_report.py                  # live captures + the historical book
    python3 beta_report.py --live-only
    python3 beta_report.py --range 0.15 0.85 --boot 4000

WHAT BETA IS. At every boundary there are three numbers: the market price m, the model
price f, and the eventual settlement s. The model's DISAGREEMENT is (f - m); the
market's ACTUAL ERROR is (s - m). Regress the second on the first:

    (s - m)  =  alpha  +  beta * (f - m)

    beta = 1   every cent of disagreement is realised - the model is right, market wrong
    beta = 0   disagreement says nothing about the market's error - it is noise
    beta < 0   disagreement points the wrong way

alpha is a sample-wide offset (in 26AUG25's captures the tracked underdogs beat their
price, giving alpha ~ +15c) and is NOT the model doing anything - it shifts every
observation equally. Only the SLOPE is about the model.

WHY NOT RMS, WHY NOT BACKTESTED P&L.
  * RMS measures fidelity to the market. Drive it to zero and you have a market
    replica with zero edge BY CONSTRUCTION. It is a broken-fit alarm, not an
    objective. Measured 26AUG25, the beta ranking is roughly the INVERSE of the rms
    ranking - the fit that tracks worst (static) had the most informative disagreement.
  * Backtested P&L thresholds and discretises, throwing away the magnitude of every
    disagreement and collapsing ~20 correlated boundaries into one bet per match. On
    26AUG25 it could not separate the variants (every sign-test p = 0.774) AND its
    ranking flipped sign against the historical book. beta uses every boundary and
    every cent, so it converges far faster - though see the honest warning below.

HONEST WARNING. beta converges faster than P&L, not fast. On 13 matches the cluster
bootstrap 95% CIs all straddled zero (e.g. ewma2 0.969, CI [-0.55, 2.69]) even though
10 of 11 per-match betas were positive. DIRECTION was supported; MAGNITUDE was not, and
the variants could not be ranked. Watch the CI, not the point estimate, and do not act
on a ranking until the intervals separate.

The bootstrap resamples MATCHES, never boundaries: every boundary inside a match shares
one settlement, so resampling boundaries would manufacture precision that does not
exist. The effective sample size is matches.
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

HERE = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(HERE, "tapes")
SETTLE_CACHE = os.path.join(TAPES, "settlements.json")

HDR = re.compile(r'^\[\d\d:\d\d:\d\d\].*?mkt\s+([\d.]+)\s')
VAR = re.compile(r'^\s+(static|rolling|ewma[\d.]+)\s+pred\s+([\d.]+)\s+err')
WINNER = re.compile(r'match over \(status=\S+, winner=([0-9a-f]+)')
LOGREF = re.compile(r'(KX[A-Z0-9]+-[A-Z0-9]+)\.log\.json')


# ----------------------------------------------------------------- settlement

def settle_from_capture(logpath):
    """Settlement of the TRACKED side, inferred from a finished capture.

    The poller logs `winner=<id-prefix>` when Kalshi closes the match. The tape carries
    competitor1_id / competitor2_id plus per-competitor set scores, and the state is
    written from the tracked side's point of view - so matching sets_me/sets_opp against
    the two overall scores identifies which competitor "me" is, and the winner prefix
    then settles it. Returns None when it cannot be established; never guesses.
    """
    body = open(logpath, errors="replace").read().replace("\r", "\n")
    w = WINNER.search(body)
    ev = LOGREF.search(body)
    if not w or not ev:
        return None
    tape = os.path.join(TAPES, ev.group(1) + ".jsonl")
    if not os.path.exists(tape):
        return None
    for line in open(tape, errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        d, st = r.get("details") or {}, r.get("state") or {}
        if not d.get("competitor1_id") or "sets_me" not in st:
            continue
        try:
            s1, s2 = int(d["competitor1_overall_score"]), int(d["competitor2_overall_score"])
        except Exception:
            continue
        if s1 == s2:
            continue                       # ambiguous, try a later row
        if st["sets_me"] == s1 and st["sets_opp"] == s2:
            me = d["competitor1_id"]
        elif st["sets_me"] == s2 and st["sets_opp"] == s1:
            me = d["competitor2_id"]
        else:
            continue
        return 1 if me.startswith(w.group(1)) else 0
    return None


def load_cache():
    try:
        return json.load(open(SETTLE_CACHE))
    except Exception:
        return {}


def save_cache(c):
    json.dump(c, open(SETTLE_CACHE, "w"), indent=1, sort_keys=True)


# ------------------------------------------------------------------ gathering

def from_live(cache, lo, hi):
    """(match -> {variant: [(divergence_c, market_error_c)]}) from finished captures."""
    out = {}
    for f in sorted(glob.glob(os.path.join(TAPES, "watch_*.log"))):
        name = os.path.basename(f)[6:-4]
        if name in cache:
            s = cache[name]
        else:
            s = settle_from_capture(f)
            if s is None:
                continue
            cache[name] = s
        body = open(f, errors="replace").read().replace("\r", "\n")
        mk, per = None, {}
        for ln in body.splitlines():
            h = HDR.match(ln)
            if h:
                mk = float(h.group(1))
                continue
            m = VAR.match(ln)
            if not m or mk is None or not (lo < mk < hi):
                continue
            per.setdefault(m.group(1), []).append(((float(m.group(2)) - mk) * 100,
                                                   (s - mk) * 100))
        if per:
            out[name] = per
    return out


def from_book(halflives, lo, hi):
    """Same, for the settled historical book, by replaying it through the LIVE logic.

    backsim.py already verified that replay reproduces the offline tools exactly
    (856 predictions, 0.0000c), so these observations are directly comparable to the
    live ones rather than a second, subtly different estimate.
    """
    from backtest_divergence import BOOK
    from backsim import live_replay
    out = {}
    for label, stem, settle in BOOK:
        p = os.path.join(TAPES, stem + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        obs = d["obs"]
        preds = live_replay(obs, d["best_of"], d["split_prior"],
                            d.get("final_set_tb") or 7, halflives)
        per = {}
        for v, series in preds.items():
            for i, f in enumerate(series):
                mk = obs[i]["price"]
                if f is None or not (lo < mk < hi):
                    continue
                per.setdefault(v, []).append(((f - mk) * 100, (settle - mk) * 100))
        if per:
            out["BOOK:" + label] = per
    return out


# ------------------------------------------------------------------ estimation

def beta(pairs, minn=8):
    n = len(pairs)
    if n < minn:
        return None
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    if sxx < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / sxx


def sign_test(diffs):
    nz = [d for d in diffs if abs(d) > 1e-9]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return k, n, min(1.0, 2 * tail)


def bootstrap(by_match, v, draws, rng):
    """Cluster bootstrap over MATCHES. Boundaries share a settlement within a match."""
    names = [m for m in by_match if by_match[m].get(v)]
    if len(names) < 3:
        return None
    bs = []
    for _ in range(draws):
        samp = [rng.choice(names) for _ in names]
        pts = [p for m in samp for p in by_match[m][v]]
        b = beta(pts)
        if b is not None:
            bs.append(b)
    if not bs:
        return None
    bs.sort()
    return (bs[int(len(bs) * .025)], bs[int(len(bs) * .975)],
            sum(1 for x in bs if x > 0) / len(bs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", nargs=2, type=float, default=[0.15, 0.85], metavar=("LO", "HI"),
                    help="only boundaries with the market in this band. Outside it the "
                         "market is converging on settlement, so (s-m) and (f-m) shrink "
                         "together and manufacture correlation")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--live-only", action="store_true", help="skip the historical book")
    ap.add_argument("--book-only", action="store_true")
    ap.add_argument("--halflives", nargs="+", default=["2", "4"],
                    help="EWMA halflives when replaying the book (match the live poller)")
    ap.add_argument("--seed", type=int, default=20260825)
    a = ap.parse_args()
    lo, hi = a.range
    rng = random.Random(a.seed)

    cache = load_cache()
    by = {}
    if not a.book_only:
        by.update(from_live(cache, lo, hi))
        save_cache(cache)
    if not a.live_only:
        try:
            by.update(from_book(a.halflives, lo, hi))
        except Exception as e:
            print(f"(book replay unavailable: {e})", file=sys.stderr)

    if not by:
        print("no settled matches with per-variant predictions yet")
        return 0

    live_n = sum(1 for m in by if not m.startswith("BOOK:"))
    print(f"beta = fraction of the model's disagreement that is realised as market error")
    print(f"{len(by)} settled matches ({live_n} live captures, {len(by)-live_n} from the book), "
          f"market in {lo:.2f}-{hi:.2f}\n")

    variants = sorted({v for m in by.values() for v in m},
                      key=lambda v: (v != "static", v != "rolling", v))
    print(f"{'variant':10}{'n bnd':>7}{'matches':>9}{'beta':>8}{'95% CI (cluster boot)':>26}"
          f"{'P(B>0)':>9}{'per-match +':>13}{'sign p':>9}{'exp edge':>10}")
    print("-" * 101)
    for v in variants:
        pts = [p for m in by.values() for p in m.get(v, [])]
        b = beta(pts)
        if b is None:
            continue
        ms = [beta(m[v]) for m in by.values() if m.get(v)]
        ms = [x for x in ms if x is not None]
        k, n, pv = sign_test(ms)
        bt = bootstrap(by, v, a.boot, rng)
        ci = f"[{bt[0]:7.3f} , {bt[1]:7.3f} ]" if bt else " " * 24
        pg = f"{bt[2]*100:8.1f}%" if bt else " " * 9
        mdiv = sum(abs(x) for x, _ in pts) / len(pts)
        print(f"{v:10}{len(pts):>7}{len(ms):>9}{b:>8.3f}{ci:>26}{pg:>9}"
              f"{str(k)+'/'+str(n):>13}{pv:>9.3f}{b*mdiv:>9.2f}c")

    print("\nexp edge = beta x mean |disagreement|: the part of a typical divergence that")
    print("is actually realised. Kalshi entry fee at p=0.5 is 1.75c per 100 contracts, so")
    print("anything under ~2c is not tradeable after costs.")
    print("\nREAD THE CI, NOT THE POINT ESTIMATE. If it straddles zero the sample cannot")
    print("yet tell you whether the disagreement is informative at all, and overlapping")
    print("CIs mean the variants are NOT ranked no matter how their point estimates sort.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
