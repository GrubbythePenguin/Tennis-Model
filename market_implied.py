"""
market_implied.py — back out the (p, q) a market is implying, from match prices
observed at different points of the match.

Each observation is (state, market_prob) where `state` is a dict of keyword
arguments for match_win_prob (sets_me, sets_opp, games_me, games_opp, i_serve,
points_me, points_opp; all default to 0 / True) and `market_prob` is the
vig-free probability the market assigns to "me" winning the match.

Two observations at two different states give two equations in the two
unknowns p, q, which is enough to solve. More observations are fitted by least
squares, which is what you actually want (see sensitivity() for why).

Everything is dependency-free; a coarse grid search finds the neighbourhood,
then a Levenberg–Marquardt polish finds the solution.
"""

from tennis_model import match_win_prob, G


# ------------------------------------------------------------- model wrapper

def model_prob(p, q, state, best_of=3):
    st = dict(state)
    sets_me, sets_opp = st.pop('sets_me', 0), st.pop('sets_opp', 0)
    return match_win_prob(p, q, sets_me, sets_opp, best_of, **st)


def normalise(*implied):
    """Strip the overround: turn raw implied probabilities (1/odds) that sum to
    more than 1 into probabilities that sum to 1 (proportional method)."""
    s = sum(implied)
    return tuple(x / s for x in implied)


# ------------------------------------------------------------------ fitting

def _residuals(p, q, obs, best_of):
    return [model_prob(p, q, st, best_of) - m for st, m in obs]


def _jacobian(p, q, obs, best_of, h=1e-4):
    """d residual / d(p, q) by central differences. Returns list of rows [dr/dp, dr/dq]."""
    rp = _residuals(p + h, q, obs, best_of); rm = _residuals(p - h, q, obs, best_of)
    rq = _residuals(p, q + h, obs, best_of); rn = _residuals(p, q - h, obs, best_of)
    return [[(a - b) / (2 * h), (c - d) / (2 * h)] for a, b, c, d in zip(rp, rm, rq, rn)]


def implied_pq(obs, best_of=3, box=((0.30, 0.95), (0.05, 0.70)), grid_step=0.02, verbose=False):
    """Least-squares fit of (p, q) to a list of (state, market_prob) observations.

    Returns (p, q, rmse). With exactly two observations rmse should be ~0 (exact
    solve); with more, rmse tells you how well one (p, q) explains all the prices —
    a large rmse is evidence the market is NOT using this model (or has vig/noise).
    """
    if len(obs) < 2:
        raise ValueError("need at least two observations at different states")

    # 1) coarse grid to find the right basin
    (plo, phi), (qlo, qhi) = box
    best = None
    p = plo
    while p <= phi + 1e-12:
        q = qlo
        while q <= qhi + 1e-12:
            r = _residuals(p, q, obs, best_of)
            sse = sum(x * x for x in r)
            if best is None or sse < best[0]:
                best = (sse, p, q)
            q += grid_step
        p += grid_step
    sse, p, q = best

    # 2) Levenberg–Marquardt polish (2 parameters, so the linear algebra is explicit)
    lam = 1e-3
    for it in range(100):
        r = _residuals(p, q, obs, best_of)
        J = _jacobian(p, q, obs, best_of)
        a = sum(j[0] * j[0] for j in J); b = sum(j[0] * j[1] for j in J); d = sum(j[1] * j[1] for j in J)
        g0 = sum(j[0] * ri for j, ri in zip(J, r)); g1 = sum(j[1] * ri for j, ri in zip(J, r))
        # solve (JtJ + lam*diag(JtJ)) delta = -Jt r
        A = a * (1 + lam); Dd = d * (1 + lam)
        det = A * Dd - b * b
        if abs(det) < 1e-18:
            break
        dp = (-g0 * Dd + g1 * b) / det
        dq = (-g1 * A + g0 * b) / det
        p_new = min(max(p + dp, 0.01), 0.99)
        q_new = min(max(q + dq, 0.01), 0.99)
        sse_new = sum(x * x for x in _residuals(p_new, q_new, obs, best_of))
        if sse_new < sse:
            p, q, sse = p_new, q_new, sse_new
            lam = max(lam / 3, 1e-9)
            if abs(dp) < 1e-9 and abs(dq) < 1e-9:
                break
        else:
            lam *= 10
        if verbose:
            print(f"iter {it}: p={p:.5f} q={q:.5f} sse={sse:.3e}")
    rmse = (sse / len(obs)) ** 0.5
    return p, q, rmse


def sensitivity(p, q, obs, best_of=3):
    """How far the implied (p, q) move if one market price moves by 1 percentage point.

    Returns a list, one entry per observation, of (dp, dq) per +0.01 in that price.
    Large numbers mean the fit is fragile: price rounding, vig or a bit of flow
    will swing the implied p and q around. (This is (J^T J)^-1 J^T, i.e. J^-1
    when there are exactly two observations.)"""
    J = _jacobian(p, q, obs, best_of)
    a = sum(j[0] * j[0] for j in J); b = sum(j[0] * j[1] for j in J); d = sum(j[1] * j[1] for j in J)
    det = a * d - b * b
    inv = [[d / det, -b / det], [-b / det, a / det]]
    out = []
    for j in J:
        dp = (inv[0][0] * j[0] + inv[0][1] * j[1]) * 0.01
        dq = (inv[1][0] * j[0] + inv[1][1] * j[1]) * 0.01
        out.append((dp, dq))
    return out


# --------------------------------------------------- uncertainty of the fit

def fit_uncertainty(p, q, obs, sigma=0.005, bound=0.01, best_of=3):
    """Uncertainty of a fitted (p, q) given the states it was fitted on.

    sigma : SD of the (vig-free) market's error per price, if errors are random and
            independent across observations. 'Within 1 point' ~ 2 sigma -> sigma = 0.005.
    bound : worst case — every price may be off by up to `bound`, in whatever
            combination is least favourable (the right lens if the market's errors are
            systematic rather than random, since then more prices don't average out).

    Returns dict with the covariance of (p, q) under random errors, 95% half-widths for
    p, q, p+q, p-q, and the worst-case half-widths for the same four quantities.
    """
    import itertools, math
    J = _jacobian(p, q, obs, best_of)
    a = sum(j[0] * j[0] for j in J); b = sum(j[0] * j[1] for j in J); d = sum(j[1] * j[1] for j in J)
    det = a * d - b * b
    inv = [[d / det, -b / det], [-b / det, a / det]]
    cov = [[sigma ** 2 * inv[0][0], sigma ** 2 * inv[0][1]], [sigma ** 2 * inv[1][0], sigma ** 2 * inv[1][1]]]
    ci = dict(p=1.96 * math.sqrt(cov[0][0]), q=1.96 * math.sqrt(cov[1][1]),
              sum=1.96 * math.sqrt(cov[0][0] + cov[1][1] + 2 * cov[0][1]),
              diff=1.96 * math.sqrt(cov[0][0] + cov[1][1] - 2 * cov[0][1]))
    # worst case: extreme points of the polytope {delta : |J_i . delta| <= bound for all i}
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
    return dict(cov=cov, ci95=ci, worst_case=wc)


def model_price_ci(p, q, cov, state, best_of=3):
    """95% half-width on the model's price at `state`, induced by the uncertainty in
    (p, q) (delta method). This is the resolution of your mid-game model: it can only
    call a mid-game market price 'wrong' if the gap exceeds roughly this."""
    import math
    g = _jacobian(p, q, [(state, 0)], best_of)[0]
    var = g[0] * g[0] * cov[0][0] + 2 * g[0] * g[1] * cov[0][1] + g[1] * g[1] * cov[1][1]
    return 1.96 * math.sqrt(max(var, 0.0))


# ------------------------------------------------- game / set markets, if any

def invert_G(g, lo=0.0, hi=1.0):
    """Per-point probability w such that G(w) = g (bisection; G is increasing).
    Use this if the market quotes a 'winner of game N' price: it gives you p
    (server's game) or q (returner's game) directly, with no conditioning issues."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if G(mid) < g:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------- demo

if __name__ == "__main__":
    best_of = 3

    # The example from the conversation: 70% before the match, 73% after A holds game 1.
    obs = [
        (dict(games_me=0, games_opp=0, i_serve=True),  0.70),   # pre-match
        (dict(games_me=1, games_opp=0, i_serve=False), 0.73),   # A held game 1, B to serve
    ]
    p, q, rmse = implied_pq(obs, best_of)
    print(f"Implied from 0.70 -> 0.73 after a hold (best of {best_of}):  p = {p:.4f}, q = {q:.4f}   (rmse {rmse:.1e})")
    print(f"  implied hold G(p) = {G(p):.3f}, break G(q) = {G(q):.3f}")

    print("\nFragility: move one of the two prices by ±0.5 points and refit")
    for m0, m1 in [(0.70, 0.725), (0.70, 0.735), (0.695, 0.73), (0.705, 0.73)]:
        pp, qq, _ = implied_pq([(obs[0][0], m0), (obs[1][0], m1)], best_of)
        print(f"  prices {m0:.3f} -> {m1:.3f}:  p = {pp:.3f}, q = {qq:.3f}")

    print("\nSensitivity of (p, q) to a 1-point move in each price:")
    for (st, m), (dp, dq) in zip(obs, sensitivity(p, q, obs, best_of)):
        print(f"  price at {st}: dp = {dp:+.3f}, dq = {dq:+.3f}")

    # What would make a better second observation? Compare, at the same (p, q),
    # the sensitivity when the second price is observed at a more serve-dependent state.
    print("\nSame ±1-point noise, different second observation (pre-match price kept):")
    alternatives = {
        "after A holds game 1 (1-0)":                  dict(games_me=1, games_opp=0, i_serve=False),
        "after A breaks in game 2 (2-0)":              dict(games_me=2, games_opp=0, i_serve=True),
        "A serving at 30-40 in game 1":                dict(games_me=0, games_opp=0, i_serve=True, points_me='30', points_opp='40'),
        "B serving at 30-40 in game 2 (A up 1-0)":     dict(games_me=1, games_opp=0, i_serve=False, points_me='40', points_opp='30'),
    }
    for label, st in alternatives.items():
        o = [obs[0], (st, model_prob(p, q, st, best_of))]
        (dp0, dq0), (dp1, dq1) = sensitivity(p, q, o, best_of)
        print(f"  {label:42s} model price {o[1][1]:.3f};  1pt there moves p by {dp1:+.3f}, q by {dq1:+.3f}")

    print("\nIf the market quotes game winners: G = 0.83 on serve ->  p =", round(invert_G(0.83), 4))
    # --- How well is (p, q) pinned down by a sequence of game-boundary prices? ---
    print("\nUncertainty from a sequence of game-boundary prices, each within 1 point (sigma = 0.005):")
    pt, qt = 0.65, 0.40
    seq = [
        ("0-0 pre-match", dict(games_me=0, games_opp=0, i_serve=True)),
        ("1-0, B serves", dict(games_me=1, games_opp=0, i_serve=False)),
        ("1-1, A serves", dict(games_me=1, games_opp=1, i_serve=True)),
        ("2-1, B serves", dict(games_me=2, games_opp=1, i_serve=False)),
        ("3-1, A serves", dict(games_me=3, games_opp=1, i_serve=True)),
        ("3-2, B serves", dict(games_me=3, games_opp=2, i_serve=False)),
    ]
    print("  after            95% half-width: p      q     p+q    p-q   | worst case: p      q     p+q    p-q")
    for k in range(2, len(seq) + 1):
        u = fit_uncertainty(pt, qt, [(st, 0.0) for _, st in seq[:k]])
        c, w = u['ci95'], u['worst_case']
        print(f"  {seq[k-1][0]:15s}                {c['p']:.3f}  {c['q']:.3f}  {c['sum']:.3f}  {c['diff']:.3f}   |            {w['p']:.3f}  {w['q']:.3f}  {w['sum']:.3f}  {w['diff']:.3f}")

    u = fit_uncertainty(pt, qt, [(st, 0.0) for _, st in seq])
    print("\nWhat that means for mid-game model prices at 3-2 (95% half-width from the (p, q) uncertainty):")
    for label, st in [
        ("B serving 30-40 (break point for A)", dict(games_me=3, games_opp=2, i_serve=False, points_me='40', points_opp='30')),
        ("B serving 40-30",                     dict(games_me=3, games_opp=2, i_serve=False, points_me='30', points_opp='40')),
        ("B serving deuce",                     dict(games_me=3, games_opp=2, i_serve=False, points_me=3, points_opp=3)),
        ("A serving 15-40 at 4-2",              dict(games_me=4, games_opp=2, i_serve=True, points_me='15', points_opp='40')),
    ]:
        print(f"  {label:38s} model {model_prob(pt, qt, st):.3f} +/- {model_price_ci(pt, qt, u['cov'], st):.3f}")

