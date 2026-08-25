"""Static (frozen) fit vs continuously-refitting fit, on a captured boundary log.

    python3 compare_fits.py tapes/<event>.log.json [--static-on 2]

STATIC  ("clean line" in the handoff): (p, q) fitted ONCE on the first --static-on
        boundary prices, then frozen. Every later price is an out-of-sample
        prediction from a model that has seen nothing since.
RUNNING ("pred"): refitted after every price; each prediction uses all EARLIER
        prices and none of the current one. Also out-of-sample, but adaptive.

Which wins tells you something specific. If STATIC tracks as well as RUNNING, the
two constant-probability parameters really do describe the whole match and the
market's wandering is noise around them. If RUNNING wins steadily and the fitted
(p, q) drifts monotonically, the market is re-rating the players and NO fixed
(p, q) will hold — which the handoff calls the real limit of this model.

Also reports the two claims from the Sherif match:
  * move amplification  — median |market move| / |model move| (was 1.47)
  * reversion slope     — regression of change-in-error on error (was -0.77);
                          negative means overshoots come back.
"""
import argparse
import json
import math
import sys

from implied_model import ImpliedModel


def build(obs, best_of, prior, upto, final_set_tb=7):
    m = ImpliedModel(best_of=best_of, first_server='me', split_prior=prior,
                     warn_split=False, final_set_tb=final_set_tb)
    for o in obs[:upto]:
        m.observe(o['price'], **o['state'])
    return m


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return float('nan')
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def ols_slope(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    return float('nan') if sxx == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--final-set-tb', type=int, choices=(7, 10),
                    help='deciding-set tiebreak length (default: from the log, else inferred)')
    ap.add_argument('--static-on', type=int, default=2,
                    help='how many leading prices the frozen fit is built on (default 2)')
    a = ap.parse_args()

    d = json.load(open(a.log))
    obs, bo, prior = d['obs'], d['best_of'], d['split_prior']
    # Deciding-set tiebreak length. Logs written before this existed lack the field,
    # so fall back to the tournament name; captures launched after it carry it.
    ftb = d.get('final_set_tb')
    if ftb is None:
        import kalshi_tennis as _kt
        tname = ((d.get('meta') or {}).get('tournament') or
                 (d.get('meta') or {}).get('title') or '')
        ev = (d.get('meta') or {}).get('event') or a.log
        import kalshi_tennis as _k
        ftb = _k.resolve_final_set_tb({'tournament_name': tname})
        if ftb == 7 and 'USOPEN' in ev.upper():
            ftb = 10        # ticker fallback when meta predates the tournament field
    if a.final_set_tb:
        ftb = a.final_set_tb
    K = a.static_on
    if len(obs) <= K:
        sys.exit(f"only {len(obs)} observations; need more than --static-on={K}")

    static = build(obs, bo, prior, K, ftb)
    meta = d.get('meta') or {}
    print(f"{meta.get('title', a.log)}   best of {bo}, {len(obs)} boundary prices, "
          f"deciding-set TB to {ftb}")
    print(f"STATIC fitted on the first {K}: p={static.p:.3f} q={static.q:.3f} "
          f"(hold {__import__('tennis_model').G(static.p):.1%}, "
          f"break {__import__('tennis_model').G(static.q):.1%})\n")

    print(f"{'state':>12}  {'market':>7} {'static':>7} {'err':>6}  "
          f"{'running':>7} {'err':>6}  {'run p':>6} {'run q':>6}   note")
    rows = []
    for i, o in enumerate(obs):
        s = o['state']
        lab = f"{s['sets_me']}-{s['sets_opp']} {s['games_me']}-{s['games_opp']}"
        mkt = o['price']
        sp, _ = static.price(**s)
        if i == 0:
            run_p, run = None, build(obs, bo, prior, 1, ftb)
        else:
            run = build(obs, bo, prior, i, ftb)
            run_p, _ = run.price(**s)
        se = (mkt - sp) * 100
        re_ = None if run_p is None else (mkt - run_p) * 100
        note = 'in-sample (static)' if i < K else ''
        print(f"{lab:>12}  {mkt:7.3f} {sp:7.3f} {se:+6.1f}  "
              f"{'      -' if run_p is None else f'{run_p:7.3f}'} "
              f"{'     -' if re_ is None else f'{re_:+6.1f}'}  "
              f"{run.p:6.3f} {run.q:6.3f}   {note}")
        rows.append(dict(i=i, lab=lab, mkt=mkt, static=sp, run=run_p,
                         se=se, re=re_, oos=i >= K))

    oos = [r for r in rows if r['oos'] and r['re'] is not None]
    if not oos:
        print("\nno out-of-sample rows yet")
        return

    def stats(key):
        e = [r[key] for r in oos]
        return (sum(e) / len(e), math.sqrt(sum(x * x for x in e) / len(e)),
                max(abs(x) for x in e))

    print(f"\nOut-of-sample rows: {len(oos)}")
    print(f"{'':10} {'mean':>7} {'rms':>7} {'worst':>7}")
    for name, key in (('STATIC', 'se'), ('RUNNING', 're')):
        m, r, w = stats(key)
        print(f"{name:10} {m:+7.2f} {r:7.2f} {w:7.2f}   cents")
    ms, mr = stats('se')[1], stats('re')[1]
    if not math.isnan(ms) and not math.isnan(mr):
        better = 'STATIC' if ms < mr else 'RUNNING'
        print(f"\nlower rms: {better}  ({min(ms, mr):.2f}c vs {max(ms, mr):.2f}c)")

    # --- move amplification: |market move| / |model move| between consecutive prices
    ratios = []
    for prev, cur in zip(rows, rows[1:]):
        if cur['run'] is None:
            continue
        mkt_mv = cur['mkt'] - prev['mkt']
        mdl_mv = cur['run'] - prev['mkt']       # model's move from the last market print
        if abs(mdl_mv) > 1e-4:
            ratios.append(abs(mkt_mv) / abs(mdl_mv))
    if ratios:
        print(f"\nmove amplification |market|/|model|: median {median(ratios):.2f} "
              f"over {len(ratios)} moves   (Sherif match: 1.47)")

    # --- reversion: does an error come back at the next boundary?
    errs = [r['re'] for r in rows if r['re'] is not None]
    if len(errs) >= 4:
        x, y = errs[:-1], [b - a for a, b in zip(errs, errs[1:])]
        print(f"reversion slope (d_error on error): {ols_slope(x, y):+.2f} "
              f"over {len(x)} points   (Sherif match: -0.77; negative = overshoots revert)")
    else:
        print(f"reversion slope: need >=4 out-of-sample errors, have {len(errs)}")


if __name__ == '__main__':
    main()
