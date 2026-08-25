"""Replay settled matches through the LIVE code path and check it against the offline tools.

    python3 backsim.py                      # every match in BOOK
    python3 backsim.py --halflives 2 4 8    # more variants
    python3 backsim.py --latency            # per-boundary timing, for the production question

WHAT THIS IS FOR. poll_tennis.py now carries four fits live (static / rolling / ewma<h>)
and writes each one's price into the tape on every tick. Tomorrow those tapes get scored
by the OFFLINE tools — rebuild_log.py, backtest_divergence.py, ewma_fit.py,
per_match_pnl.py. If the live path and the offline path disagree even slightly, every
conclusion drawn from a live tape is measuring the disagreement rather than the market.

Nothing checked that they agree. They are genuinely different code: the live path folds
one boundary at a time into a warm-started model (_ewma_update); the offline path builds
a model from scratch over a slice of boundaries. Same maths on paper, different
execution order, different optimiser starting points.

So this replays each settled match through the LIVE update logic exactly as the poller
runs it, and compares every out-of-sample boundary prediction against what the offline
tools produce for the same boundary. Any divergence above optimiser tolerance is a bug
in one of the two, and it is much cheaper to find it here than in tomorrow's numbers.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel
from backtest_divergence import BOOK, run as offline_static_rolling
from ewma_fit import fit_series
import poll_tennis as pt


def live_replay(obs, best_of, prior, tb, halflives):
    """Drive the LIVE variant logic over a boundary sequence, exactly as cmd_watch does.

    Returns {variant: [pred or None per boundary]}, each prediction made BEFORE that
    boundary was observed — same out-of-sample discipline as the poller.
    """
    mk = lambda: ImpliedModel(best_of=best_of, first_server="me", split_prior=prior,
                              warn_split=False, final_set_tb=tb)
    model, static = mk(), mk()
    ewma = {f"ewma{h}": mk() for h in halflives}
    preds = {t: [] for t in (["static", "rolling"] + [f"ewma{h}" for h in halflives])}
    for i, o in enumerate(obs, 1):
        for tag, m in ([("static", static), ("rolling", model)] + sorted(ewma.items())):
            if m.p is None:
                preds[tag].append(None)
                continue
            try:
                preds[tag].append(m.price(**o["state"])[0])
            except Exception:
                preds[tag].append(None)
        model.observe(o["price"], **o["state"])
        if i <= pt.STATIC_N:
            static.observe(o["price"], **o["state"])
        for h in halflives:
            pt._ewma_update(ewma[f"ewma{h}"], o["price"], o["state"], float(h))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflives", nargs="+", default=["2", "4"])
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed live-vs-offline disagreement, in CENTS")
    ap.add_argument("--latency", action="store_true",
                    help="also report per-boundary wall time (the production question)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print(f"replaying {len(BOOK)} settled matches through the live path, "
          f"comparing to the offline tools (tolerance {a.tol:.2f}c)\n")
    hdr = f"{'match':24}{'bnd':>5}" + "".join(f"{('ewma' + h if h.isdigit() else h):>12}"
                                              for h in ["static", "rolling"] + a.halflives)
    print(hdr.replace("static", "static").replace("rolling", "rolling"))
    print("-" * len(hdr))

    worst_overall, checked, failures = 0.0, 0, []
    lat = []
    for label, stem, settle in BOOK:
        p = os.path.join(here, "tapes", stem + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        obs, bo, pr = d["obs"], d["best_of"], d["split_prior"]
        tb = d.get("final_set_tb") or 7

        t0 = time.time()
        live = live_replay(obs, bo, pr, tb, a.halflives)
        if a.latency:
            lat.append(((time.time() - t0) / len(obs) * 1000, len(obs), label))

        # OFFLINE static/rolling: backtest_divergence prices each boundary from i>=2
        off = offline_static_rolling(p, settle, 100, 0.0)
        offline = {"static": {}, "rolling": {}}
        for tag in ("static", "rolling"):
            for rec in off[tag]:
                offline[tag][json.dumps(rec["state"], sort_keys=True)] = rec["mdl"]

        row, worst_row = f"{label[:24]:24}{len(obs):>5}", 0.0
        for tag in ["static", "rolling"]:
            worst = 0.0
            for i, o in enumerate(obs):
                if i < 2 or live[tag][i] is None:
                    continue
                k = json.dumps(o["state"], sort_keys=True)
                if k not in offline[tag]:
                    continue
                worst = max(worst, abs(live[tag][i] - offline[tag][k]) * 100)
                checked += 1
            row += f"{worst:11.4f}c"
            worst_row = max(worst_row, worst)

        # OFFLINE ewma: fit_series returns (market, pred) per boundary from i=1
        for h in a.halflives:
            _, trades = fit_series(obs, bo, pr, tb, float(h), 0.0)
            worst = 0.0
            for i, (_mkt, pr_off) in enumerate(trades, start=1):
                if i >= len(obs) or live[f"ewma{h}"][i] is None:
                    continue
                worst = max(worst, abs(live[f"ewma{h}"][i] - pr_off) * 100)
                checked += 1
            row += f"{worst:11.4f}c"
            worst_row = max(worst_row, worst)

        print(row + ("   <-- OVER TOLERANCE" if worst_row > a.tol else ""))
        if worst_row > a.tol:
            failures.append((label, worst_row))
        worst_overall = max(worst_overall, worst_row)

    print("-" * len(hdr))
    print(f"\n{checked} boundary predictions compared, "
          f"worst live-vs-offline disagreement {worst_overall:.4f}c")
    if failures:
        print("\nFAILED — the live path and the offline tools do not agree:")
        for lbl, w in failures:
            print(f"  {lbl}: {w:.4f}c")
    else:
        print(f"PASS — every variant agrees to better than {a.tol:.2f}c, so a live tape "
              f"can be scored\n       with the offline tools without measuring their disagreement.")

    if lat:
        lat.sort(reverse=True)
        print(f"\nPER-BOUNDARY LATENCY (all {len(a.halflives) + 2} fits, one process)")
        print(f"  {'ms/boundary':>12}  {'boundaries':>11}  match")
        for ms, n, lbl in lat[:5]:
            print(f"  {ms:12.1f}  {n:11d}  {lbl}")
        print(f"  worst {max(l[0] for l in lat):.1f} ms/boundary")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
