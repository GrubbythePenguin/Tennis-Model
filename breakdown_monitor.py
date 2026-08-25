"""When the fit collapses, is the MODEL breaking or is the PLAYER breaking?

    python3 breakdown_monitor.py tapes/<event>.jsonl [--window 4]

THE PROBLEM. A fixed (p, q) assumes the players are the same at the end of the match
as at the start. Cramping, an injury, a medical timeout or a retirement violates that
completely, and the fitter has no way to represent it — so it contorts the split
instead, squeezing p toward q until serving confers no advantage. That contortion is
the only visible symptom, and on its own it is ambiguous: an identical collapse comes
from a bad pre-match anchor.

THE DISCRIMINATOR. Kalshi ships per-player serve production in
competitorN_statistics, cumulative over the match. Differencing it between boundaries
gives the REALIZED serve rate in each window, independent of any price. Then:

    implied hold collapsing  +  realized serve production collapsing
        -> PHYSICAL. Something happened to the player. The model cannot price it and
           the market is repricing a different athlete. Stop trusting the fit.

    implied hold collapsing  +  realized serve production steady
        -> MODEL. The prices cannot be reconciled by any constant (p, q) — most often
           a bad anchor. The fit is wrong, the player is fine.

    implied hold steady      +  realized collapsing
        -> the market has not noticed yet, or the sample is too small to mean anything.

PLAUSIBILITY BAND. Separately from the trend, an implied hold outside roughly 35-97%
is not a tennis player. The p > q guard does not catch this — p can sit just above q
and still imply that serving is worth nothing (Travaglia hit 53.5% hold / 44.0% break
on 26AUG24 while passing the guard). Flagged here as HOLD_IMPLAUSIBLE.

Everything is computed from the tape, so it works on a finished match or a live one.
"""
import argparse
import json

import kalshi_tennis as kt
from implied_model import ImpliedModel
from rebuild_log import boundaries_from_tape, resolve_tracked, PLAYING
from tennis_model import G

HOLD_LO, HOLD_HI = 0.35, 0.97


def _stats(det, n):
    return det.get(f"competitor{n}_statistics") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tape")
    ap.add_argument("--best-of", type=int, default=3)
    ap.add_argument("--split-prior", type=float, default=0.28)
    ap.add_argument("--final-set-tb", type=int, default=7)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.tape)]
    me, opp, _, _ = resolve_tracked(rows)
    n_me = 1 if rows[-1]["details"].get("competitor1_id") == me else 2

    obs, _ = boundaries_from_tape(a.tape)
    if len(obs) < 4:
        raise SystemExit(f"only {len(obs)} boundaries; need >= 4")

    # cumulative serve stats at the moment of each boundary
    play = [r for r in rows if r["details"].get("match_status") in PLAYING
            and r.get("vig_free") is not None]
    stat_at = []
    for o in obs:
        want = (o["state"]["sets_me"], o["state"]["sets_opp"],
                o["state"]["games_me"], o["state"]["games_opp"])
        hit = None
        for r in play:
            st = kt.model_state(r["details"], me, opp)
            if st and kt.boundary_key(st) == want:
                hit = r
                break
        stat_at.append(_stats(hit["details"], n_me) if hit else {})

    print(f"{a.tape}   {len(obs)} boundaries, tracked side = competitor{n_me}\n")
    print(f"{'state':>10} {'mkt':>6} {'impl hold':>10} {'d_hold':>7} | "
          f"{'1st-serve%':>10} {'srv-pts won%':>12} {'DF':>3} | flags")

    prev_hold = prev_stat = None
    verdicts = []
    for i, (o, sv) in enumerate(zip(obs, stat_at)):
        m = ImpliedModel(best_of=a.best_of, first_server="me", split_prior=a.split_prior,
                         warn_split=False, final_set_tb=a.final_set_tb)
        for o2 in obs[:i + 1]:
            m.observe(o2["price"], **o2["state"])
        hold = G(m.p)
        s = o["state"]
        lab = f"{s['sets_me']}-{s['sets_opp']} {s['games_me']}-{s['games_opp']}"

        # realized serve production IN THIS WINDOW (difference the cumulative counters)
        f1 = spw = df = None
        if sv and prev_stat:
            d_first = (sv.get("first_serve_successful", 0) - prev_stat.get("first_serve_successful", 0))
            d_won = (sv.get("service_points_won", 0) - prev_stat.get("service_points_won", 0))
            d_lost = (sv.get("service_points_lost", 0) - prev_stat.get("service_points_lost", 0))
            d_df = (sv.get("double_faults", 0) - prev_stat.get("double_faults", 0))
            served = d_won + d_lost
            if served > 0:
                f1, spw, df = d_first / served, d_won / served, d_df

        flags = []
        if not (HOLD_LO <= hold <= HOLD_HI):
            flags.append("HOLD_IMPLAUSIBLE")
        dh = None if prev_hold is None else (hold - prev_hold) * 100
        if dh is not None and dh <= -8:
            # implied hold falling hard — physical or model?
            if spw is not None and spw < 0.45:
                flags.append("PHYSICAL?(serve production down)")
            elif spw is not None:
                flags.append("MODEL?(serve production intact)")
            else:
                flags.append("HOLD_DROP(no serve sample)")
        print(f"{lab:>10} {o['price']:6.3f} {hold:9.1%} "
              f"{'      -' if dh is None else f'{dh:+6.1f}p'} | "
              f"{'         -' if f1 is None else f'{f1:9.0%}'} "
              f"{'           -' if spw is None else f'{spw:11.0%}'} "
              f"{'  -' if df is None else f'{df:3d}'} | {' '.join(flags)}")
        verdicts += flags
        prev_hold, prev_stat = hold, (sv or prev_stat)

    print()
    phys = sum("PHYSICAL" in v for v in verdicts)
    mod = sum("MODEL?" in v for v in verdicts)
    imp = sum(v == "HOLD_IMPLAUSIBLE" for v in verdicts)
    print(f"summary: {imp} implausible-hold boundaries, "
          f"{phys} physical-looking drops, {mod} model-looking drops")
    if phys > mod and phys:
        print("  -> leans PHYSICAL: serve production fell alongside the implied hold.")
        print("     A constant-(p,q) model cannot price an athlete who changed mid-match.")
    elif mod and mod >= phys:
        print("  -> leans MODEL: the implied hold fell while serve production held up.")
        print("     Suspect the anchor before the player; refit from first ball.")
    elif imp:
        print("  -> implied hold left the plausible band without a clear serve signal.")
    else:
        print("  -> no breakdown signal.")


if __name__ == "__main__":
    main()
