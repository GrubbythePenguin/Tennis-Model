"""P&L by edge bucket, priced against the WORST unseen point and filled at the ask.

    python3 bracket_pnl.py
    python3 bracket_pnl.py --variant ewma2 --size 100 --detail

THE PROBLEM THIS SOLVES. Our scoreboard runs a point behind the fastest participants,
so at an observed 4-3 40-15 the market is already pricing whatever the next point did.
Comparing model(4-3 40-15) to that market compares two different information sets.
Measured 26AUG26 on 14,915 in-play observations: 81% of all model-vs-market
disagreements were SMALLER than |model(win) - model(lose)| at that moment, i.e. fully
explained by the one point we could not see. Those 81% were never evidence of anything,
and they are most of what every earlier backtest was trading on.

THE RULE (operator, 26AUG26). Price each side at the branch where THAT SIDE lost the
unseen point, and be willing to trade only there. At 4-3 40-15 with A serving the two
branches are 40-30 (B won it) and 5-3 0-0 (A won it, game over), so:

    buying A  ->  price A at 40-30      the branch that is worse for A
    buying B  ->  price B at 5-3 0-0    the branch that is worse for B

In tracked-side terms, with w = model(win) and l = model(lose):

    conservative price for the tracked side = min(w, l)
    conservative price for the opponent     = 1 - max(w, l)

Neither side can be surprised by the point we did not see: whichever way it went, the
true state is at least as good as the one we priced against. The two numbers do NOT sum
to 1 - the gap between them is exactly the width of the bracket, and that is the width
the lag costs us.

FILLS ARE AT THE ASK, NOT THE MID. Every earlier backtest entered at the vig-free
midpoint, a price nobody can trade, while still charging the real Kalshi fee. poll_tennis
now records raw top of book, so a buy pays ask and the fee is charged on that.

ONE OBSERVATION PER POINT STATE. At a ~1.3s cadence a single point score produces ~20
near-identical rows. Trading each of them would count one decision twenty times and
fake the sample size, so rows are collapsed to the FIRST sighting of each distinct
point state - which is also the moment a real system would act.

WEIGHTING. Buckets are summarised match-weighted: each match is reduced to one number
inside a bucket, then matches are averaged, one vote each. Boundaries inside a match
share a settlement, so trade-weighting hands long matches several times the influence
of short ones and lets a single match invert the sign of a total.
"""
import argparse
import glob
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_divergence import fee
import beta_report as BR

HERE = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(HERE, "tapes")


def settle_for(tape_path, settle_cache):
    """Settlement of the tracked side for the match this tape belongs to."""
    stem = os.path.basename(tape_path)[:-6]
    key = stem.rsplit("-", 1)[-1].lower()
    if key in settle_cache:
        return settle_cache[key]
    log = os.path.join(TAPES, f"watch_{key}.log")
    for cand in (log, os.path.join(TAPES, f"deep_{key}.log")):
        if os.path.exists(cand):
            s = BR.settle_from_capture(cand)
            if s is not None:
                settle_cache[key] = s
                return s
    return None


def opportunities(tape_path, variant, settle):
    """One row per distinct point state: both sides' conservative price vs their ask."""
    seen = set()
    out = []
    for line in open(tape_path, errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "live":
            continue
        a = r.get("ahead") or {}
        bk = r.get("book") or {}
        st = r.get("state") or {}
        if variant not in (a.get("win") or {}) or variant not in (a.get("lose") or {}):
            continue
        if not st.get("_points_known"):
            continue
        sig = (st.get("sets_me"), st.get("sets_opp"), st.get("games_me"),
               st.get("games_opp"), st.get("points_me"), st.get("points_opp"))
        if sig in seen:
            continue
        seen.add(sig)
        w, l = a["win"][variant], a["lose"][variant]
        bracket = abs(w - l)
        # worst-branch price for each side
        cons_me = min(w, l)
        cons_opp = 1.0 - max(w, l)
        for side, cons, ask, bid, pays in (
                ("me", cons_me, bk.get("ask_me"), bk.get("bid_me"), settle),
                ("opp", cons_opp, bk.get("ask_opp"), bk.get("bid_opp"), 1 - settle)):
            if ask is None or bid is None:
                continue
            out.append(dict(side=side, ask=ask, bid=bid, cons=cons,
                            bracket_c=bracket * 100, pays=pays))
    return out


def sign_test(vals):
    nz = [v for v in vals if abs(v) > 1e-9]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for v in nz if v > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return k, n, min(1.0, 2 * tail)


def boot(vals, draws, rng):
    if len(vals) < 3:
        return None
    bs = []
    for _ in range(draws):
        s = [rng.choice(vals) for _ in vals]
        bs.append(sum(s) / len(s))
    bs.sort()
    return bs[int(len(bs) * .025)], bs[int(len(bs) * .975)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ewma2")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--edges", nargs="+", type=float, default=[0, 1, 2, 4],
                    help="bucket cuts on (conservative price - ask), in cents")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--maker", action="store_true",
                    help="model QUOTING instead of taking: you are filled at your own "
                         "quote (the conservative price) and pay NO fee, since Kalshi "
                         "charges makers nothing. This is the OPTIMISTIC bound - it "
                         "assumes you get filled and ignores adverse selection, which "
                         "is the dominant risk for a quoter running a point behind.")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cache = BR.load_cache()

    per = {}      # bucket -> match -> [pnl per trade]
    nmatch = 0
    skipped = []
    for tape in sorted(glob.glob(os.path.join(TAPES, "KX*.jsonl"))):
        s = settle_for(tape, cache)
        if s is None:
            continue
        ops = opportunities(tape, a.variant, s)
        if not ops:
            continue
        nmatch += 1
        name = os.path.basename(tape)[:-6]
        for o in ops:
            if a.maker:
                # QUOTING: join the best bid and wait to be hit. Entry is the BID, and
                # Kalshi charges makers nothing. Worth quoting only while our worst-case
                # value is above the bid we would pay.
                entry = o["bid"]
                e = (o["cons"] - o["bid"]) * 100
                if e <= 0:
                    continue
                net = a.size * (o["pays"] - entry)
            else:
                # TAKING: cross to the ask and pay the taker fee. Only when our
                # worst-case value already clears the offer.
                entry = o["ask"]
                e = (o["cons"] - o["ask"]) * 100
                if e <= 0:
                    continue
                net = a.size * (o["pays"] - entry) - fee(a.size, entry)
            b = 0
            for i, c in enumerate(a.edges):
                if e >= c:
                    b = i
            per.setdefault(b, {}).setdefault(name, []).append(net)
    BR.save_cache(cache)

    if not per:
        print("no settled match yet carries one-point branches + book "
              "(needs a capture from 26AUG26 onward)")
        return 0

    def lab(i):
        if i == len(a.edges) - 1:
            return f"{a.edges[i]:.0f}c+"
        return f"{a.edges[i]:.0f}-{a.edges[i+1]:.0f}c"

    mode = ("QUOTING: filled at our own worst-branch price, no maker fee"
            if a.maker else "TAKING: filled at the ASK, net of entry fees")
    print(f"BRACKET-SAFE P&L — {a.variant}, size {a.size}\n  {mode}")
    print(f"{nmatch} settled matches with point-level branches; each side priced at the "
          f"branch where\nit LOST the unseen point, one observation per distinct point "
          f"state.\n")
    print(f"  {'edge vs worst-case':>20} {'trades':>7} {'matches':>8} {'trade-wtd $':>12}"
          f" {'MATCH-wtd $':>12} {'median $':>10} {'matches +':>11} {'sign p':>8}"
          f" {'95% CI':>22}")
    for b in sorted(per):
        g = per[b]
        pm = [sum(x) / len(x) for x in g.values()]
        allt = [x for v in g.values() for x in v]
        k, n, p = sign_test(pm)
        ci = boot(pm, a.boot, rng)
        cis = f"[{ci[0]:8.2f},{ci[1]:8.2f} ]" if ci else ""
        sp = sorted(pm)
        med = sp[len(sp)//2] if len(sp) % 2 else (sp[len(sp)//2-1] + sp[len(sp)//2]) / 2
        print(f"  {lab(b):>20} {len(allt):>7} {len(g):>8} {sum(allt)/len(allt):>12.2f}"
              f" {sum(pm)/len(pm):>12.2f} {med:>10.2f} {str(k)+'/'+str(n):>11} {p:>8.3f}"
              f" {cis:>22}")
    print("\n  edge here is (worst-branch model price - the ASK you would actually pay).")
    print("  A real edge is MONOTONE across these buckets. Read MATCH-wtd and median")
    print("  first; where trade-wtd diverges from it, a few long matches are driving it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
