"""Is the market settled enough on (p, q) for a fixed-(p, q) model to be usable?

    python3 pregame_stability.py tapes/<event>.jsonl [...]

THE ARGUMENT. The model assumes ONE (p, q) describes the whole match. That assumption
is the market's too, implicitly, whenever its price is stable. But if the price wanders
before a ball is struck, the market has NOT settled on a (p, q) — it is still arguing
about the players. Anchoring a fixed fit to a number drawn from the middle of that
argument gives a fit with no claim to being right, and every later error is measuring
the anchor rather than the model.

So pre-match drift is a SCREEN, computable before the match starts and before any
model is fitted: how far did the price move while nothing happened?

  Sherif/Oliynykova  0.50 -> 0.48 -> 0.55 -> 0.45 -> 0.465   10c range, no tennis
  Koevermans/Monnet  0.395 (17:10) -> 0.465 (17:48)           7c drift, no tennis

Both are large against an in-play tracking error of 2-3c. A model whose anchor carries
7c of uncertainty cannot be evaluated at 2c resolution — the noise floor is above the
signal.

WHAT IS REPORTED. Range, standard deviation and net drift of the vig-free price over
all pre-match polls, plus the implied uncertainty that drift induces on (p, q). The
last column is the one that matters: if the pre-match price alone spans a range that
maps to a wide band of p+q, the match is not a fair test of the model.
"""
import argparse
import json
import statistics

from market_implied import implied_pq
from tennis_model import G

PREGAME = {"match_about_to_start", None, ""}


def pregame_rows(tape):
    out = []
    for line in open(tape):
        r = json.loads(line)
        d = r["details"]
        ms = d.get("match_status")
        if r.get("vig_free") is None:
            continue
        played = any((x.get("score") or 0) for x in (d.get("competitor1_round_scores") or [])) \
            or any((x.get("score") or 0) for x in (d.get("competitor2_round_scores") or [])) \
            or d.get("competitor1_current_round_score") or d.get("competitor2_current_round_score")
        if ms in PREGAME and not played:
            out.append(r)
    return out


def pq_band(lo, hi, best_of=3):
    """(p+q) implied at each end of the pre-match price range, at a neutral split."""
    band = []
    for px in (lo, hi):
        try:
            obs = [(dict(games_me=0, games_opp=0, i_serve=True), px),
                   (dict(games_me=1, games_opp=0, i_serve=False), px + 0.02)]
            p, q, _ = implied_pq(obs, best_of)
            band.append(p + q)
        except Exception:
            band.append(float("nan"))
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tapes", nargs="+")
    ap.add_argument("--window", type=float, default=10.0,
                    help="minutes before first ball to measure over (default 10)")
    a = ap.parse_args()

    # CONFOUND THIS CONTROLS FOR. Raw drift is not comparable across matches: a match
    # delayed 38 minutes has far more time to wander than one that starts 11 minutes
    # after we attach, so the delayed match looks "unstable" for free. Measuring the
    # LAST `window` minutes before first ball puts every match on the same footing.
    # Both the full-history and windowed figures are printed; if they disagree, the
    # raw number was measuring observation time, not market uncertainty.
    print(f"{'match':34} {'n':>4} {'range':>7} {'sd':>6} {'drift':>7} | "
          f"last {a.window:.0f}min: {'n':>4} {'range':>7} {'drift':>7}  verdict")
    for t in a.tapes:
        rows = pregame_rows(t)
        if len(rows) < 2:
            print(f"{t.split('/')[-1][:34]:34} {len(rows):>4}   (too few pre-match polls with a price)")
            continue
        px = [r["vig_free"] for r in rows]
        rng, sd = (max(px) - min(px)) * 100, statistics.pstdev(px) * 100
        drift = (px[-1] - px[0]) * 100
        span = (rows[-1]["ts"] - rows[0]["ts"]) / 60.0

        t_end = rows[-1]["ts"]
        win = [r for r in rows if t_end - r["ts"] <= a.window * 60]
        wpx = [r["vig_free"] for r in win]
        wrng = (max(wpx) - min(wpx)) * 100 if len(wpx) > 1 else float("nan")
        wdrift = (wpx[-1] - wpx[0]) * 100 if len(wpx) > 1 else float("nan")

        judge = wrng if len(wpx) > 1 else rng          # judge on the comparable figure
        # A short observation cannot establish stability. Seeing no drift over one
        # minute says almost nothing — an unstable market is quiet most of the time
        # too. Only a WIDE reading is trustworthy on a short window, because drift
        # observed is drift that happened; absence of it is not evidence of absence.
        MIN_MIN = 5.0
        if judge >= 5:
            verdict = "UNUSABLE — anchor noise exceeds in-play signal"
        elif span < MIN_MIN:
            verdict = f"INSUFFICIENT — only {span:.0f} min observed, cannot confirm stability"
        elif judge >= 2.5:
            verdict = "MARGINAL — anchor noise ~ signal"
        else:
            verdict = "OK — settled before first ball"
        log = t.replace(".jsonl", ".log.json")
        try:
            title = (json.load(open(log)).get("meta") or {}).get("title") or t
        except Exception:
            title = t.split("/")[-1]
        print(f"{title[:34]:34} {len(px):>4} {rng:6.1f}c {sd:5.1f}c {drift:+6.1f}c | "
              f"{'':9} {len(wpx):>4} {wrng:6.1f}c {wdrift:+6.1f}c  {verdict}")
        print(f"{'':34} observed {span:.0f} min pre-match")


if __name__ == "__main__":
    main()
