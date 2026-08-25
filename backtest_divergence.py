"""If we traded the model-vs-market divergence, which fit makes money?

    python3 backtest_divergence.py [--size 100] [--threshold 0]

THE QUESTION THIS ANSWERS. A ROLLING fit refits to every new price, so a genuine
mispricing is absorbed into the parameters rather than flagged — by construction it
cannot hold a view the market disagrees with for long. A STATIC fit keeps its view,
so divergence persists and is tradeable. Lower tracking error is therefore NOT the
criterion that matters: a model that perfectly predicts the market has zero edge by
definition. The criterion is whether the disagreements are RIGHT.

METHOD. At each game boundary, compare the model's price to the vig-free market mid.
If they differ by more than --threshold, buy `size` contracts of the side the model
prefers, at the market mid, and hold to settlement. Settlement is the real Kalshi
result. P&L per trade = size * (settle - entry) for a long, negated for a short.

FEES. Kalshi charges 0.07 * C * P * (1-P), rounded up to the cent, on entry. Charged
here on entry only (held to settlement, no exit fee). Fees are largest at P=0.5,
which is exactly where most divergences occur, so ignoring them flatters the result.

THE CAVEAT THAT DOMINATES EVERYTHING. Trades within one match are NOT independent —
every boundary in a match settles on the same outcome, so being long the eventual
winner wins every trade in that match. The effective sample size is the number of
MATCHES (5), not the number of trades (~100). Per-match P&L is reported for exactly
this reason; the total is close to five correlated bets, and no total from five bets
distinguishes a real edge from luck.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel

# tracked side, settlement (1 = tracked player won), rebuilt log
BOOK = [
    # (label, rebuilt log stem, settlement of the TRACKED side)
    ("Poljicak vs Schoenhaus", "bt_KXATPCHALLENGERMATCH-26AUG24POLSCH", 0),
    ("Cecchinato vs Broady",   "bt_KXATPMATCH-26AUG24CECBRO",           1),
    ("Koevermans vs Monnet",   "bt_KXWTAMATCH-26AUG24KOEMON",           0),
    ("Guerrieri vs Holmgren",  "bt_KXATPMATCH-26AUG24GUEHOL",           1),
    ("Wendelken vs Travaglia", "bt_KXATPMATCH-26AUG24WENTRA",           0),
    ("Wong vs Moller",         "bt_KXATPMATCH-26AUG24WONMOL",           0),
    ("Balshaw vs Kovacevic",   "bt_KXATPMATCH-26AUG24BALKOV",           0),
    ("PinningtonJones/Svajda", "bt_KXATPMATCH-26AUG24PINSVA",           1),
    ("Vedder vs Andreescu",    "bt_KXWTAMATCH-26AUG24VEDAND",           0),
    ("Parry vs Vekic",          "bt_KXWTAMATCH-26AUG23PARVEK",           1),
]


def fee(contracts, price):
    """Kalshi trading fee, rounded UP to the cent."""
    return math.ceil(0.07 * contracts * price * (1 - price) * 100) / 100


def run(path, settle, size, thresh, tb=None):
    d = json.load(open(path))
    obs, prior = d["obs"], d["split_prior"]
    # Use the match's OWN deciding-set tiebreak. Hardcoding 10 priced every
    # Challenger / Winston Salem / Monterrey match as a Grand Slam. The direct effect
    # is small, but it shifts model prices across the trade threshold and so changes
    # WHICH trades fire — worth several hundred dollars on one match.
    if tb is None:
        tb = d.get("final_set_tb") or 7

    def build(n):
        m = ImpliedModel(best_of=d["best_of"], first_server="me", split_prior=prior,
                         warn_split=False, final_set_tb=tb)
        for o in obs[:n]:
            m.observe(o["price"], **o["state"])
        return m

    static = build(2)
    out = {"static": [], "rolling": []}
    for i, o in enumerate(obs):
        if i < 2:
            continue                       # in-sample for static; no fair comparison
        mkt = o["price"]
        preds = {"static": static.price(**o["state"])[0],
                 "rolling": build(i).price(**o["state"])[0]}
        for tag, mdl in preds.items():
            edge = mdl - mkt
            if abs(edge) < thresh:
                continue
            long_ = edge > 0              # model says the tracked player is cheap
            entry = mkt if long_ else 1 - mkt
            payoff = (settle if long_ else 1 - settle)
            gross = size * (payoff - entry)
            out[tag].append(dict(state=o["state"], mkt=mkt, mdl=mdl, edge=edge * 100,
                                 long=long_, gross=gross,
                                 fee=fee(size, entry), net=gross - fee(size, entry)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="minimum |model-market| in DOLLARS to trade (0 = every divergence)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print(f"size {a.size} contracts, threshold {a.threshold*100:.1f}c, held to settlement, "
          f"net of Kalshi entry fees\n")
    print(f"{'match':26} {'settle':>6} | {'STATIC  n':>10} {'net $':>10} {'win%':>6} | "
          f"{'ROLL  n':>9} {'net $':>10} {'win%':>6}")
    tot = {"static": 0.0, "rolling": 0.0}
    cnt = {"static": 0, "rolling": 0}
    wins = {"static": 0, "rolling": 0}
    per_match = []
    for name, f, settle in BOOK:
        p = os.path.join(here, "tapes", f + ".json")
        if not os.path.exists(p):
            continue
        r = run(p, settle, a.size, a.threshold)
        row = [name, settle]
        for tag in ("static", "rolling"):
            t = r[tag]
            net = sum(x["net"] for x in t)
            w = sum(1 for x in t if x["net"] > 0)
            tot[tag] += net
            cnt[tag] += len(t)
            wins[tag] += w
            row += [len(t), net, (w / len(t) * 100 if t else float("nan"))]
        per_match.append((name, row[2], row[3], row[5], row[6]))
        print(f"{row[0]:26} {row[1]:6d} | {row[2]:10d} {row[3]:10.2f} {row[4]:5.0f}% | "
              f"{row[5]:9d} {row[6]:10.2f} {row[7]:5.0f}%")

    print()
    print(f"{'TOTAL':26} {'':6} | {cnt['static']:10d} {tot['static']:10.2f} "
          f"{wins['static']/max(cnt['static'],1)*100:5.0f}% | "
          f"{cnt['rolling']:9d} {tot['rolling']:10.2f} "
          f"{wins['rolling']/max(cnt['rolling'],1)*100:5.0f}%")
    print()
    sm = sum(1 for _, _, s, _, r in per_match if s > 0)
    rm = sum(1 for _, _, s, _, r in per_match if r > 0)
    print(f"matches profitable: STATIC {sm}/{len(per_match)}   ROLLING {rm}/{len(per_match)}")
    print()
    print("EFFECTIVE SAMPLE = 5 MATCHES, not ~%d trades. Every boundary inside a match"
          % cnt["static"])
    print("settles on the same outcome, so the trades are near-perfectly correlated.")
    print("Read the per-match column, not the total.")


if __name__ == "__main__":
    main()
