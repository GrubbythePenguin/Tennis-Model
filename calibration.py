"""Is the market miscalibrated, and specifically in the up-a-break state?

    python3 calibration.py [--band 0.05]

HYPOTHESIS UNDER TEST (operator, 26AUG25): the market overprices the underdog, so
the player who is UP A BREAK is underpriced — buying them around 0.80 should win
more often than 0.80 of the time.

This is a MODEL-FREE test. It compares the market price at a boundary directly with
what actually happened, so it does not inherit any error from the fit. If the market
is well calibrated, prices in the 0.75-0.85 bucket win ~80% of the time.

BREAK DIFFERENTIAL is reconstructed exactly, not guessed. Between consecutive
boundaries exactly one game is played; the server is recorded at the earlier
boundary and the winner is whoever's game count rose. Winner != server is a break.
"Up a break" = more breaks landed than conceded in the CURRENT set.

THE CAVEAT THAT LIMITS EVERYTHING. Boundaries inside one match share an outcome, so
a bucket holding 40 rows from 9 matches carries roughly 9 independent bets, not 40.
Bucket win-rates are reported with the number of distinct MATCHES contributing, and
that is the number to reason about.
"""
import argparse
import json
import math
import os
import sys

BOOK = [
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


def rows_for(path, settle, label):
    """One row per boundary: price, settlement, and break differential in the set."""
    obs = json.load(open(path))["obs"]
    out, breaks = [], 0
    for prev, cur in zip(obs, obs[1:]):
        ps, cs = prev["state"], cur["state"]
        # new set -> break differential resets
        if (cs["sets_me"], cs["sets_opp"]) != (ps["sets_me"], ps["sets_opp"]):
            breaks = 0
            continue
        server = ps.get("server")
        won_me = cs["games_me"] > ps["games_me"]
        if server in ("me", "opp"):
            if won_me and server == "opp":
                breaks += 1                       # I broke
            elif (not won_me) and server == "me":
                breaks -= 1                       # I was broken
        lead = cs["games_me"] - cs["games_opp"]
        out.append(dict(match=label, price=cur["price"], settle=settle,
                        breaks=breaks, lead=lead,
                        sets_lead=cs["sets_me"] - cs["sets_opp"]))
    return out


def wilson(k, n):
    if not n:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def report(rows, title, edges):
    print(f"\n{title}")
    print(f"  {'price band':>12} {'n':>4} {'matches':>8} {'avg price':>10} "
          f"{'realized':>9} {'diff':>7}  {'95% CI':>16}")
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in rows if lo <= r["price"] < hi]
        if not sel:
            continue
        n = len(sel)
        w = sum(r["settle"] for r in sel)
        ap = sum(r["price"] for r in sel) / n
        rz = w / n
        cl, ch = wilson(w, n)
        m = len({r["match"] for r in sel})
        flag = ""
        if not (cl <= ap <= ch):
            flag = "  <-- outside CI"
        print(f"  {lo:5.2f}-{hi:4.2f} {n:4d} {m:8d} {ap:10.3f} {rz:9.3f} "
              f"{(rz-ap)*100:+6.1f}c  [{cl:.2f},{ch:.2f}]{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.70)
    ap.add_argument("--hi", type=float, default=0.90)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    allrows = []
    for label, f, settle in BOOK:
        p = os.path.join(here, "tapes", f + ".json")
        if os.path.exists(p):
            allrows += rows_for(p, settle, label)
    # mirror every row so both sides of every match are represented; otherwise the
    # sample is only "the tracked player", which is the underdog by construction here
    mirrored = allrows + [dict(match=r["match"] + "~", price=1 - r["price"],
                               settle=1 - r["settle"], breaks=-r["breaks"],
                               lead=-r["lead"], sets_lead=-r["sets_lead"])
                          for r in allrows]

    print(f"{len(allrows)} boundaries from {len({r['match'] for r in allrows})} settled "
          f"matches; mirrored to {len(mirrored)} rows (both sides)")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    report(mirrored, "ALL boundaries — is the market calibrated?", edges)

    up = [r for r in mirrored if r["breaks"] > 0]
    report(up, "UP A BREAK in the current set", edges)

    band = [r for r in up if a.lo <= r["price"] < a.hi]
    if band:
        n = len(band); w = sum(r["settle"] for r in band)
        apx = sum(r["price"] for r in band) / n
        cl, ch = wilson(w, n)
        print(f"\nTHE HYPOTHESIS: up a break, priced {a.lo:.2f}-{a.hi:.2f}")
        print(f"  n={n} rows from {len({r['match'] for r in band})} matches")
        print(f"  avg price {apx:.3f}   realized {w/n:.3f}   edge {(w/n-apx)*100:+.1f}c")
        print(f"  95% CI on realized [{cl:.3f}, {ch:.3f}] — "
              f"{'price OUTSIDE CI (miscalibrated)' if not (cl <= apx <= ch) else 'price inside CI (no detectable bias)'}")
    else:
        print(f"\nno up-a-break rows in {a.lo}-{a.hi}")


if __name__ == "__main__":
    main()
