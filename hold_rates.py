"""Market-IMPLIED hold rate vs REALIZED hold rate, from a captured tape.

    python3 hold_rates.py tapes/<event>.jsonl [tapes/<event2>.jsonl ...]

THE QUESTION. Consensus says women hold serve far less often than men. If the market
OVER-corrects for that, the hold rate implied by its prices sits below the hold rate
the players actually produce on court. That is measurable here without any outside
data, because Kalshi ships both halves:

  implied  — fit (p, q) to the boundary prices; G(p) is the implied hold probability
             and G(q) the implied break probability. Nothing in the model biases this;
             it is whatever explains the observed prices.
  realized — competitorN_statistics.service_games_won / (service games played), taken
             from the final payload of the match.

  service games played is NOT published directly. It is derived: total games won by
  both players, split by who served. games_won and service_games_won ARE published,
  so return games won = games_won - service_games_won, and the opponent's service
  games played = their service_games_won + our return games won.

CAVEAT THAT LIMITS EVERYTHING BELOW. One match is ~10-13 service games per player, so
a realized hold rate has a standard error around 12-15 percentage points. A single
match cannot separate a 60% holder from a 75% holder. This tool is for accumulating
across matches; a per-match gap under ~15pp is noise.

Note also that split_prior itself encodes the consensus (0.28 men / 0.14 women) — so
run with --prior-free to check that the implied number is coming from the prices and
not from the prior being asserted back at you.
"""
import argparse
import json
import math
import os
import sys

from implied_model import ImpliedModel
from tennis_model import G


def realized(det, n):
    """(holds, service_games, breaks, return_games) for competitor n from statistics."""
    me = det.get(f"competitor{n}_statistics") or {}
    opp = det.get(f"competitor{3 - n}_statistics") or {}
    if not me or not opp:
        return None
    gw, sgw = me.get("games_won"), me.get("service_games_won")
    ogw, osgw = opp.get("games_won"), opp.get("service_games_won")
    if None in (gw, sgw, ogw, osgw):
        return None
    my_return_wins = gw - sgw            # games I won while receiving
    opp_return_wins = ogw - osgw         # games opp won while receiving = my breaks against
    my_service_games = sgw + opp_return_wins
    opp_service_games = osgw + my_return_wins
    return sgw, my_service_games, my_return_wins, opp_service_games


def wilson(k, n):
    """95% interval for a proportion — small n here, so normal approx is not enough."""
    if not n:
        return (float('nan'), float('nan'))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def analyse(tape, prior_free):
    rows = [json.loads(l) for l in open(tape)]
    if not rows:
        return None
    log = tape.replace(".jsonl", ".log.json")
    if not os.path.exists(log):
        print(f"  (no boundary log beside {tape}; skipping implied fit)")
        return None
    d = json.load(open(log))
    obs = d.get("obs") or []
    if len(obs) < 3:
        print(f"  (only {len(obs)} boundary prices; skipping)")
        return None

    m = ImpliedModel(best_of=d["best_of"], first_server="me",
                     split_prior=d["split_prior"],
                     split_prior_sd=5.0 if prior_free else 0.30,
                     warn_split=False)
    for o in obs:
        m.observe(o["price"], **o["state"])

    det = rows[-1]["details"]
    # "me" is the competitor whose sets match the tracked state
    s0 = rows[0]["state"]
    n_me = 1 if det.get("competitor1_id") and \
        rows[0]["details"].get("competitor1_overall_score") == s0.get("sets_me") else 1
    # tracked side is competitor1 unless the state mirrors competitor2
    if det.get("competitor2_overall_score") == rows[-1]["state"]["sets_me"] and \
       det.get("competitor1_overall_score") == rows[-1]["state"]["sets_opp"] and \
       det.get("competitor1_overall_score") != det.get("competitor2_overall_score"):
        n_me = 2

    r = realized(det, n_me)
    meta = d.get("meta") or {}
    print(f"\n{meta.get('title', tape)}   [{meta.get('tour','?')}] "
          f"best of {d['best_of']}, {len(obs)} boundary prices")
    print(f"  implied   p={m.p:.3f} -> hold {G(m.p):6.1%}   "
          f"q={m.q:.3f} -> break {G(m.q):6.1%}   p+q={m.p + m.q:.3f}")
    if r:
        h, sg, br, og = r
        lo, hi = wilson(h, sg)
        blo, bhi = wilson(br, og)
        print(f"  realized  held {h}/{sg} = {h / sg if sg else float('nan'):6.1%} "
              f"[{lo:.1%}–{hi:.1%}]   broke {br}/{og} = "
              f"{br / og if og else float('nan'):6.1%} [{blo:.1%}–{bhi:.1%}]")
        if sg:
            gap = (G(m.p) - h / sg) * 100
            verdict = ("implied BELOW realized (market may over-correct)" if gap < -1 else
                       "implied ABOVE realized" if gap > 1 else "implied ~ realized")
            inside = lo <= G(m.p) <= hi
            print(f"  gap       {gap:+.1f} pp  — {verdict}"
                  f"{'  (within realized CI: not distinguishable)' if inside else ''}")
        return dict(title=meta.get("title"), tour=meta.get("tour"), best_of=d["best_of"],
                    implied_hold=G(m.p), implied_break=G(m.q),
                    realized_hold=h / sg if sg else None, service_games=sg,
                    realized_break=br / og if og else None, return_games=og)
    else:
        print("  realized  (statistics absent from the final payload)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tapes", nargs="+")
    ap.add_argument("--prior-free", action="store_true",
                    help="widen split_prior_sd to 5.0 so the consensus prior cannot "
                         "assert the answer back at you")
    a = ap.parse_args()

    print("IMPLIED vs REALIZED HOLD RATE"
          + ("   [prior-free fit]" if a.prior_free else "   [default split prior]"))
    out = [x for x in (analyse(t, a.prior_free) for t in a.tapes) if x]

    if len(out) > 1:
        print("\n" + "=" * 62)
        print(f"{'match':32} {'tour':16} {'impl':>6} {'real':>6} {'gap':>7}")
        for r in out:
            g = ((r["implied_hold"] - r["realized_hold"]) * 100
                 if r["realized_hold"] is not None else float('nan'))
            print(f"{(r['title'] or '')[:32]:32} {(r['tour'] or '')[:16]:16} "
                  f"{r['implied_hold']:6.1%} {r['realized_hold']:6.1%} {g:+6.1f}pp")
        tot_sg = sum(r["service_games"] for r in out)
        print(f"\npooled service games: {tot_sg} — a hold rate needs a few hundred "
              f"before a 5pp effect is visible.")


if __name__ == "__main__":
    main()
