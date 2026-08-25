"""
implied_model.py — an incremental market-implied model.

    m = ImpliedModel(best_of=3, first_server='me')   # 'me' served game 1 of set 1
    m.observe(0.70)                                   # pre-match price for 'me'
    m.observe(0.73, games_me=1, games_opp=0)          # after game 1 (server of each game is derived
                                                      #   from the alternation; or pass server='me'/'opp')
    m.report()                                        # p, q, uncertainty, and a grid of match prices
    m.observe(0.695, games_me=1, games_opp=1)         # keep feeding prices; every call refits
    m.price(games_me=3, games_opp=1, points_me='15', points_opp='40')   # (prob, 95% half-width)

Every observation is a vig-free probability that 'me' wins the match, at a state.
The fit is weighted least squares over all observations plus a weak prior on the
serve/return split p - q (which stabilises the fit when only near-parallel prices
have been seen — e.g. pre-match and 1-1 — and fades as informative prices arrive).
"""

import math
import itertools
import sys
from tennis_model import match_win_prob, G, game_win_prob, tiebreak_win_prob
from market_implied import invert_G, check_split, ImpliedSplitError


def _model(p, q, state, best_of, final_set_tb=7):
    st = dict(state)
    return match_win_prob(p, q, st.pop('sets_me', 0), st.pop('sets_opp', 0), best_of,
                          final_set_tb, **st)


def _next_game(state, **bump):
    """State after the current game, with the server handed to the other player.

    If the state names its server explicitly (poll_tennis.py logs do, read straight
    off Kalshi's `server` field) it must be FLIPPED — leaving it would credit the
    next game to the same server. If it doesn't, it is left absent so the
    games-played alternation resolves it as before.
    """
    out = dict(state, **bump)
    sv = out.get('server')
    if sv is not None:
        out['server'] = (not sv) if isinstance(sv, bool) else ('opp' if sv == 'me' else 'me')
    return out


class ImpliedModel:
    def __init__(self, best_of=3, first_server='me', sigma=0.005,
                 split_prior=0.20, split_prior_sd=0.30,
                 strict_split=False, warn_split=True, final_set_tb=7):
        """
        best_of        : 3 or 5
        first_server   : who served game 1 of set 1 ('me' or 'opp')
        sigma          : assumed SD of the market's error per price (0.005 ~ 'within 1 point')
        split_prior    : prior mean for p - q. Roughly 0.28 on the men's tour, 0.14 on the
                         women's tour, 0.20 if unsure.
        split_prior_sd : prior SD for p - q. 0.30 is weak: it barely moves a fit that the
                         prices identify, and only stabilises one they don't (e.g. pre-match
                         plus 1-1). Use a large value (5.0) for a pure, prior-free solve.
        strict_split   : raise ImpliedSplitError if a refit gives p <= q. Default False so a
                         single bad quote can't kill a live logging session; the violation
                         still warns loudly and is recorded on .split_violation.
        warn_split     : print the loud stderr banner on a violation. Set False when refitting
                         history in a loop (live.py does this) and surface it once instead.
        """
        self.best_of = best_of
        # 10 at the Grand Slams (deciding-set breaker), 7 everywhere else.
        self.final_set_tb = final_set_tb
        self.sigma = sigma
        self.split_prior, self.split_prior_sd = split_prior, split_prior_sd
        self.strict_split, self.warn_split = strict_split, warn_split
        self.first_server = {(0, 0): first_server == 'me'}
        # (0,0) starts as a PLACEHOLDER, not an observation. Kalshi leaves `server`
        # empty during warm-up, so a pre-match price carries no server and this
        # default stands in. Overwriting a placeholder from the feed is normal and
        # must not warn — only a clash between two OBSERVED servers is a real
        # alternation break. A detector that cries wolf on every match start is one
        # nobody reads when a genuine retirement or medical timeout breaks it.
        self._provisional = {(0, 0)}
        self.obs = []            # (state, market_prob, weight)
        self.p = self.q = None
        self.cov = None
        self.split_violation = None   # message from the last refit, or None if p > q
        self._warned_split = False

    # ------------------------------------------------------------ state helpers

    def set_first_server(self, sets_me, sets_opp, server):
        """Who serves game 1 of a later set (tennis: the player who did NOT serve the last
        game of the previous set; after a tiebreak, whoever received first in it)."""
        self.first_server[(sets_me, sets_opp)] = server == 'me'
        self._provisional.discard((sets_me, sets_opp))   # stated explicitly, not a guess

    def _serving(self, sets_me, sets_opp, games_me, games_opp):
        key = (sets_me, sets_opp)
        if key not in self.first_server:
            raise ValueError(f"unknown first server for set {key}: call set_first_server(...) or pass server=")
        first_me = self.first_server[key]
        return first_me if (games_me + games_opp) % 2 == 0 else not first_me

    def _state(self, games_me=0, games_opp=0, server=None, points_me=0, points_opp=0, sets_me=0, sets_opp=0):
        if server is None:
            i_serve = self._serving(sets_me, sets_opp, games_me, games_opp)
        else:
            i_serve = server if isinstance(server, bool) else server == 'me'
        return dict(sets_me=sets_me, sets_opp=sets_opp, games_me=games_me, games_opp=games_opp,
                    i_serve=i_serve, points_me=points_me, points_opp=points_opp)

    # ------------------------------------------------------------------ fitting

    def _register_first_server(self, sm, so, gm, go, i_serve):
        """Back-fill who served game 1 of set (sm, so) from an observation that states
        its server explicitly.

        poll_tennis.py reads the server off Kalshi's `server` field on every price and
        never calls set_first_server, so without this the set is unregistered and any
        state NOT carrying an explicit server — the report's price grids, the
        next-game branches — raises. Servers alternate, so game 1's server follows
        from this game's server and the games-played parity.
        """
        key = (sm, so)
        first_me = i_serve if (gm + go) % 2 == 0 else (not i_serve)
        prev = self.first_server.get(key)
        if key in self._provisional:
            prev = None                      # placeholder — replace it silently
            self._provisional.discard(key)
        if prev is not None and prev != first_me:
            print(f"[server] set {key}: feed says game {gm}-{go} is served by "
                  f"{'me' if i_serve else 'opp'}, which implies game 1 went to "
                  f"{'me' if first_me else 'opp'} — contradicting the recorded "
                  f"{'me' if prev else 'opp'}. Strict alternation may be broken "
                  f"(retirement, medical timeout, or a bad feed sample).", file=sys.stderr)
        self.first_server[key] = first_me

    def observe(self, market_prob, weight=1.0, **state):
        """Add one observation and refit. `weight` < 1 down-weights a price you trust less
        (e.g. a mid-game price). Returns self so calls can be chained."""
        st = self._state(**state)
        if state.get('server') is not None:
            self._register_first_server(st['sets_me'], st['sets_opp'],
                                        st['games_me'], st['games_opp'], st['i_serve'])
        self.obs.append((st, float(market_prob), float(weight)))
        self._fit()
        return self

    def _residuals(self, p, q):
        r = [math.sqrt(w) * (_model(p, q, st, self.best_of, self.final_set_tb) - m) for st, m, w in self.obs]
        r.append(self.sigma / self.split_prior_sd * ((p - q) - self.split_prior))   # prior row
        return r

    def _jacobian(self, p, q, h=1e-4):
        rp, rm = self._residuals(p + h, q), self._residuals(p - h, q)
        rq, rn = self._residuals(p, q + h), self._residuals(p, q - h)
        return [[(a - b) / (2 * h), (c - d) / (2 * h)] for a, b, c, d in zip(rp, rm, rq, rn)]

    def _fit(self):
        sse = lambda p, q: sum(x * x for x in self._residuals(p, q))
        if self.p is None:                                  # first fit: coarse grid for a start
            best = None
            for i in range(0, 31):
                for j in range(0, 31):
                    p, q = 0.35 + 0.02 * i, 0.05 + 0.02 * j
                    v = sse(p, q)
                    if best is None or v < best[0]:
                        best = (v, p, q)
            _, p, q = best
        else:
            p, q = self.p, self.q                           # warm start
        cur, lam = sse(p, q), 1e-3
        for _ in range(100):                                # Levenberg-Marquardt
            r, J = self._residuals(p, q), self._jacobian(p, q)
            a = sum(j[0] * j[0] for j in J); b = sum(j[0] * j[1] for j in J); d = sum(j[1] * j[1] for j in J)
            g0 = sum(j[0] * ri for j, ri in zip(J, r)); g1 = sum(j[1] * ri for j, ri in zip(J, r))
            A, D = a * (1 + lam), d * (1 + lam)
            det = A * D - b * b
            if abs(det) < 1e-18:
                break
            dp, dq = (-g0 * D + g1 * b) / det, (-g1 * A + g0 * b) / det
            pn, qn = min(max(p + dp, 0.01), 0.99), min(max(q + dq, 0.01), 0.99)
            new = sse(pn, qn)
            if new < cur:
                p, q, cur, lam = pn, qn, new, max(lam / 3, 1e-9)
                if abs(dp) < 1e-9 and abs(dq) < 1e-9:
                    break
            else:
                lam *= 10
        self.p, self.q = p, q
        J = self._jacobian(p, q)
        a = sum(j[0] * j[0] for j in J); b = sum(j[0] * j[1] for j in J); d = sum(j[1] * j[1] for j in J)
        det = a * d - b * b
        s2 = self.sigma ** 2
        self.cov = [[s2 * d / det, -s2 * b / det], [-s2 * b / det, s2 * a / det]]
        # guardrail: the fit must stay inside the physical range p > q
        self.split_violation = check_split(
            p, q, strict=self.strict_split,
            warn=self.warn_split and not self._warned_split,      # loud once per model
            context=f"{len(self.obs)} price{'s' if len(self.obs) != 1 else ''}")
        if self.split_violation:
            self._warned_split = True

    # ------------------------------------------------------------------ outputs

    def price(self, **state):
        """Model probability that 'me' wins the match at `state`, and its 95% half-width
        (from the uncertainty in p, q under random ±sigma errors in the observed prices)."""
        st = self._state(**state)
        h = 1e-4
        gp = (_model(self.p + h, self.q, st, self.best_of, self.final_set_tb) - _model(self.p - h, self.q, st, self.best_of, self.final_set_tb)) / (2 * h)
        gq = (_model(self.p, self.q + h, st, self.best_of, self.final_set_tb) - _model(self.p, self.q - h, st, self.best_of, self.final_set_tb)) / (2 * h)
        var = gp * gp * self.cov[0][0] + 2 * gp * gq * self.cov[0][1] + gq * gq * self.cov[1][1]
        return _model(self.p, self.q, st, self.best_of, self.final_set_tb), 1.96 * math.sqrt(max(var, 0.0))

    def uncertainty(self):
        """95% half-widths for p, q, p+q, p-q (random errors), plus the worst case if every
        observed price is off by up to 2*sigma in the least favourable combination."""
        c = self.cov
        out = dict(p=1.96 * math.sqrt(c[0][0]), q=1.96 * math.sqrt(c[1][1]),
                   sum=1.96 * math.sqrt(c[0][0] + c[1][1] + 2 * c[0][1]),
                   diff=1.96 * math.sqrt(c[0][0] + c[1][1] - 2 * c[0][1]))
        bound = 2 * self.sigma
        J = self._jacobian(self.p, self.q)[:-1]             # observation rows only
        wc = dict(p=0.0, q=0.0, sum=0.0, diff=0.0)
        for i, k in itertools.combinations(range(len(J)), 2):
            for si in (-1, 1):
                for sk in (-1, 1):
                    dt = J[i][0] * J[k][1] - J[i][1] * J[k][0]
                    if abs(dt) < 1e-12:
                        continue
                    dp = (si * bound * J[k][1] - J[i][1] * sk * bound) / dt
                    dq = (J[i][0] * sk * bound - si * bound * J[k][0]) / dt
                    if all(abs(j[0] * dp + j[1] * dq) <= bound + 1e-9 for j in J):
                        wc['p'] = max(wc['p'], abs(dp)); wc['q'] = max(wc['q'], abs(dq))
                        wc['sum'] = max(wc['sum'], abs(dp + dq)); wc['diff'] = max(wc['diff'], abs(dp - dq))
        out['worst_case'] = wc if len(J) >= 2 else None
        return out

    def residuals(self):
        """(state, market, model, model - market) for every observation. Large or drifting
        residuals mean no single (p, q) explains the prices — the market isn't running a
        pure iid model, or the prices you fed are noisier than sigma."""
        return [(st, m, _model(self.p, self.q, st, self.best_of, self.final_set_tb), _model(self.p, self.q, st, self.best_of, self.final_set_tb) - m)
                for st, m, _ in self.obs]

    def game_grid(self, sets_me=0, sets_opp=0):
        """Match price at the start of every game score in a set (server from the alternation)."""
        rows = []
        for a in range(7):
            row = []
            for b in range(7):
                if (a == 6 and b == 6) or (max(a, b) <= 5) or (max(a, b) == 6 and abs(a - b) <= 1):
                    st = self._state(games_me=a, games_opp=b, sets_me=sets_me, sets_opp=sets_opp)
                    row.append(_model(self.p, self.q, st, self.best_of, self.final_set_tb))
                else:
                    row.append(None)
            rows.append(row)
        return rows

    def point_grid(self, **state):
        """Match price at every point score of the game at `state` (rows = my points)."""
        labels = ['0', '15', '30', '40', 'AD']
        grid = {}
        for i, x in enumerate(labels):
            for j, y in enumerate(labels):
                if (x == 'AD' and y != '40') or (y == 'AD' and x != '40'):
                    continue
                st = dict(state); st['points_me'], st['points_opp'] = x, y
                grid[(x, y)] = _model(self.p, self.q, self._state(**st), self.best_of)
        return labels, grid

    def report(self, current=None):
        """Print the fit, its uncertainty, and price grids. `current` = dict of the current
        state (games_me, games_opp, ...) to mark it and show next-game / point prices."""
        p, q = self.p, self.q
        u = self.uncertainty()
        n = len(self.obs)
        print(f"Fit on {n} price{'s' if n != 1 else ''}:  p = {p:.3f} (hold {G(p):.0%})   q = {q:.3f} (break {G(q):.0%})"
              f"   p+q = {p + q:.3f}   tiebreak {tiebreak_win_prob(p, q):.3f}")
        print(f"  95% half-widths (random ±{2 * self.sigma:.0%} errors):  p ±{u['p']:.3f}  q ±{u['q']:.3f}  p+q ±{u['sum']:.3f}  p−q ±{u['diff']:.3f}")
        if u['worst_case']:
            w = u['worst_case']
            print(f"  worst case (every price off by up to {2 * self.sigma:.0%}):  p ±{w['p']:.3f}  q ±{w['q']:.3f}  p+q ±{w['sum']:.3f}")
        rms = math.sqrt(sum(d * d for *_, d in self.residuals()) / n)
        print(f"  rms residual on the {n} price{'s' if n != 1 else ''}: {rms:.4f}")
        if self.split_violation:
            print(f"  *** p <= q: {self.split_violation}")
            print(f"  *** every price below is unreliable — do not trade off this fit.")

        cur = self._state(**current) if current else None
        sm, so = (cur['sets_me'], cur['sets_opp']) if cur else (0, 0)
        grid = self.game_grid(sm, so)
        print(f"\nMatch price at the start of each game, set {sm + so + 1} (rows = my games, cols = opp games):")
        print("       " + "".join(f"{b:>7d}" for b in range(7)))
        for a in range(7):
            cells = []
            for b in range(7):
                v = grid[a][b]
                mark = '*' if cur and (a, b) == (cur['games_me'], cur['games_opp']) else ' '
                cells.append(f"{v:6.3f}{mark}" if v is not None else "      -")
            print(f"  {a:>4d} " + "".join(cells))

        if cur:
            prob, ci = self.price(**current)
            srv = 'me' if cur['i_serve'] else 'opp'
            print(f"\nNow ({cur['games_me']}-{cur['games_opp']}, {srv} serving, {cur['points_me']}-{cur['points_opp']}):  {prob:.3f} ± {ci:.3f}")
            if not (cur['games_me'] == 6 and cur['games_opp'] == 6):
                nxt = dict(current); nxt.pop('points_me', None); nxt.pop('points_opp', None)
                win = _next_game(nxt, games_me=cur['games_me'] + 1)
                lose = _next_game(nxt, games_opp=cur['games_opp'] + 1)
                pw, cw = self.price(**win); pl, cl = self.price(**lose)
                print(f"  if I win this game → {pw:.3f} ± {cw:.3f}     if I lose it → {pl:.3f} ± {cl:.3f}")
                labels, pg = self.point_grid(**{k: v for k, v in current.items() if k not in ('points_me', 'points_opp')})
                print(f"  this game, by point score (rows = my points, cols = opp points):")
                print("         " + "".join(f"{y:>7s}" for y in labels))
                for x in labels:
                    print(f"    {x:>4s} " + "".join(f"{pg[(x, y)]:7.3f}" if (x, y) in pg else "      -" for y in labels))


# ---------------------------------------------------------------------- demo

if __name__ == "__main__":
    m = ImpliedModel(best_of=3, first_server='me', split_prior=0.28)   # men's-tour prior on the split
    m.observe(0.70)                                                     # pre-match
    m.observe(0.73, games_me=1, games_opp=0)                            # I held game 1
    m.report(current=dict(games_me=1, games_opp=0))

    # keep feeding game-boundary prices (here: what a market running p=0.65, q=0.40 would show,
    # rounded to the nearest half point) and watch the fit refine
    print("\n" + "=" * 90 + "\nfeeding more prices...\n")
    truth = (0.65, 0.40)
    m2 = ImpliedModel(best_of=3, first_server='me', split_prior=0.28)
    path = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (4, 2), (4, 3), (5, 3), (5, 4), (6, 4)]
    for a, b in path:
        st = m2._state(games_me=a, games_opp=b)
        market = round(_model(*truth, st, 3) / 0.005) * 0.005
        m2.observe(market, games_me=a, games_opp=b)
        u = m2.uncertainty()
        print(f"  after {a}-{b} (market {market:.3f}):  p = {m2.p:.3f} ±{u['p']:.3f}   q = {m2.q:.3f} ±{u['q']:.3f}   p+q = {m2.p + m2.q:.3f} ±{u['sum']:.3f}")
    print(f"\n  (true values were p = {truth[0]}, q = {truth[1]})")
    print("\nResiduals (model − market) for the fed prices:")
    for st, mk, md, d in m2.residuals():
        print(f"  {st['games_me']}-{st['games_opp']}: market {mk:.3f}  model {md:.3f}  diff {d:+.4f}")
