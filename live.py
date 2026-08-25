"""
live.py — log game-boundary prices for one match and check the fixed-(p, q) model as they arrive.

  python3 live.py init --first-server me --pregame 0.50 --best-of 3 --split-prior 0.14
  python3 live.py init ... --strict-split       # raise instead of warning if a fit gives p <= q
  python3 live.py add 1 0 0.55                 # score me-opp, then MY price (vig-free mid)
  python3 live.py add 1 1 0.50 --server me     # server override if the alternation is broken
  python3 live.py add 6 6 0.52 --points 3 5    # mid-tiebreak / mid-game points (me, opp) if wanted
  python3 live.py report

For every price it shows what the model (fitted on all EARLIER prices) predicted for that
state before seeing it, the error, and how the fitted p, q move. Steady p, q and errors of a
cent or so = the fixed-probability assumption is holding; drifting p, q or growing errors = not.

A '!' in the report marks a price after which the fit had p <= q — the player would return
better than they serve, which is outside the physical range, so the prices are inconsistent
with the model and nothing downstream of that fit is tradeable.
"""

import json, sys, os, argparse
from implied_model import ImpliedModel, _model, _next_game
from market_implied import ImpliedSplitError
from tennis_model import G

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live_log.json')


def load():
    with open(LOG) as f:
        return json.load(f)


def save(d):
    with open(LOG, 'w') as f:
        json.dump(d, f, indent=1)


def build(d, upto=None, warn_split=False):
    m = ImpliedModel(best_of=d['best_of'], first_server=d['first_server'],
                     split_prior=d['split_prior'], split_prior_sd=d.get('split_prior_sd', 0.30),
                     strict_split=d.get('strict_split', False), warn_split=warn_split)
    for k, (s, sv) in enumerate(d.get('set_servers', {}).items()):
        a, b = map(int, s.split('-'))
        m.set_first_server(a, b, sv)
    obs = d['obs'] if upto is None else d['obs'][:upto]
    for o in obs:
        m.observe(o['price'], **o['state'])
    return m


def cmd_init(a):
    d = dict(best_of=a.best_of, first_server=a.first_server, split_prior=a.split_prior,
             split_prior_sd=a.split_prior_sd, strict_split=a.strict_split, set_servers={}, obs=[])
    d['obs'].append(dict(state=dict(games_me=0, games_opp=0), price=a.pregame, label='pre-match'))
    save(d)
    print(f"initialised: best of {a.best_of}, 'me' serves game 1 = {a.first_server == 'me'}, pre-match {a.pregame}")


def cmd_add(a):
    d = load()
    state = dict(games_me=a.games_me, games_opp=a.games_opp, sets_me=a.sets_me, sets_opp=a.sets_opp)
    if a.server:
        state['server'] = a.server
    if a.points:
        state['points_me'], state['points_opp'] = a.points
    d['obs'].append(dict(state=state, price=a.price, label=a.label or ''))
    save(d)
    cmd_report(a)


def cmd_set_server(a):
    d = load()
    d.setdefault('set_servers', {})[f"{a.sets_me}-{a.sets_opp}"] = a.server
    save(d)
    print(f"set {a.sets_me}-{a.sets_opp}: game 1 served by {a.server}")


def cmd_report(a):
    d = load()
    obs = d['obs']
    print(f"{'state':>14s}  {'market':>7s}  {'predicted':>9s}  {'error':>6s}     |  fit afterwards:  p (hold)      q (break)     p+q")
    m = None
    bad = []                                     # prices after which the fit had p <= q
    for k, o in enumerate(obs):
        st = o['state']
        lab = f"{st.get('sets_me',0)}-{st.get('sets_opp',0)} {st['games_me']}-{st['games_opp']}"
        if 'points_me' in st:
            lab += f" {st['points_me']}-{st['points_opp']}"
        if m is None:
            pred_txt, err_txt = f"{'-':>9s}", f"{'-':>6s}"
        else:
            pred, _ = m.price(**st)
            pred_txt, err_txt = f"{pred:9.3f}", f"{o['price'] - pred:+6.3f}"
        try:
            m = build(d, upto=k + 1)
        except ImpliedSplitError as e:
            print(f"{lab:>14s}  {o['price']:7.3f}  {pred_txt}  {err_txt}   |  *** p <= q, strict mode: {e}")
            print("\nSTOPPED: strict_split is on and this price drove the fit outside p > q.")
            print("Fix or drop the offending price, or re-init without --strict-split.")
            return
        u = m.uncertainty()
        flag = ' !' if m.split_violation else '  '
        if m.split_violation:
            bad.append(lab)
        print(f"{lab:>14s}  {o['price']:7.3f}  {pred_txt}  {err_txt}  {flag} |                   {m.p:.3f} ({G(m.p):.0%})   {m.q:.3f} ({G(m.q):.0%})   {m.p + m.q:.3f} ±{u['sum']:.3f}")

    if bad:
        print(f"\n  ! = fit had p <= q after that price ({len(bad)} of {len(obs)}): {', '.join(bad)}")

    # Keep `server` if the log carries it. Logs written by poll_tennis.py state the
    # server explicitly on every observation and register no set_servers, so
    # dropping it here made the current state underivable (ValueError) for any set
    # after the first.
    cur = dict(obs[-1]['state'])
    print()
    m.report(current=cur)
    # what to expect at the next boundary
    if not (cur['games_me'] == 6 and cur['games_opp'] == 6):
        nxt = {k: v for k, v in cur.items() if k not in ('points_me', 'points_opp')}
        srv = 'me' if m._state(**nxt)['i_serve'] else 'opp'
        w, l = _next_game(nxt, games_me=cur['games_me'] + 1), _next_game(nxt, games_opp=cur['games_opp'] + 1)
        pw, cw = m.price(**w); pl, cl = m.price(**l)
        print(f"\nNext game served by {srv}: expect {pw:.3f} (±{cw:.3f}) if I win it, {pl:.3f} (±{cl:.3f}) if I lose it.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    i = sub.add_parser('init'); i.add_argument('--first-server', default='me'); i.add_argument('--pregame', type=float, default=0.5)
    i.add_argument('--best-of', type=int, default=3); i.add_argument('--split-prior', type=float, default=0.14); i.add_argument('--split-prior-sd', type=float, default=0.30)
    i.add_argument('--strict-split', action='store_true', help='raise instead of warning if a fit gives p <= q')
    ad = sub.add_parser('add'); ad.add_argument('games_me', type=int); ad.add_argument('games_opp', type=int); ad.add_argument('price', type=float)
    ad.add_argument('--sets-me', type=int, default=0); ad.add_argument('--sets-opp', type=int, default=0)
    ad.add_argument('--server', choices=['me', 'opp']); ad.add_argument('--points', nargs=2); ad.add_argument('--label')
    ss = sub.add_parser('set-server'); ss.add_argument('sets_me', type=int); ss.add_argument('sets_opp', type=int); ss.add_argument('server', choices=['me', 'opp'])
    sub.add_parser('report')
    a = ap.parse_args()
    {'init': cmd_init, 'add': cmd_add, 'set-server': cmd_set_server, 'report': cmd_report}[a.cmd](a)
