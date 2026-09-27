"""Skill-uncertainty ("tau") fair for the set-1 leader — pure stdlib.

Model (operator study 26SEP17, scratchpad tau_fit.py, ITF fit tau=0.55):
    skill gap d ~ N(mu, tau^2), per-set win prob p = Phi(d), sets iid GIVEN d.
    Pregame match prob pins mu:      m = E[p^2 (3-2p)]
    Fair for the leader up 1-0:      P(match | won set 1) = E[2p^2 - p^3] / E[p]

Everything is a prior moment of p (conditional independence given d turns the
Bayesian update into moment ratios — no explicit posterior). Integrals are
Simpson's rule over d in mu +/- 8*tau (161 points); mu is found by bisection.
Cross-checked against the Gauss-Hermite/scipy implementation in the study to
<0.05c across m in [0.03, 0.97].

tau=0.55 is the ITF fit (match outcomes after 1-0, n=739, LR p=0.034). The
market's own 1-0 prices imply tau=0 there — that gap is the cell's edge.
"""
import math

TAU_ITF = 0.55
_N = 160          # Simpson intervals (even)


def _phi_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _moments(mu: float, tau: float):
    """(E[p], E[p^2], E[p^3]) for p = Phi(d), d ~ N(mu, tau^2)."""
    if tau <= 1e-9:
        p = _phi_cdf(mu)
        return p, p * p, p ** 3
    lo, hi = mu - 8.0 * tau, mu + 8.0 * tau
    h = (hi - lo) / _N
    e1 = e2 = e3 = 0.0
    inv = 1.0 / (tau * math.sqrt(2.0 * math.pi))
    for i in range(_N + 1):
        d = lo + i * h
        w = 1.0 if i in (0, _N) else (4.0 if i % 2 else 2.0)
        g = w * inv * math.exp(-0.5 * ((d - mu) / tau) ** 2)
        p = _phi_cdf(d)
        e1 += g * p; e2 += g * p * p; e3 += g * p ** 3
    k = h / 3.0
    return e1 * k, e2 * k, e3 * k


def _match_prob(mu: float, tau: float) -> float:
    e1, e2, e3 = _moments(mu, tau)
    return 3.0 * e2 - 2.0 * e3


def solve_mu(m: float, tau: float) -> float:
    """mu such that the pregame BO3 match prob equals m (monotone -> bisection)."""
    m = min(max(m, 0.02), 0.98)
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _match_prob(mid, tau) < m:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fair_1_0(m_leader: float, tau: float = TAU_ITF) -> float:
    """P(leader wins match | leader won set 1), from the leader's pregame prob."""
    mu = solve_mu(m_leader, tau)
    e1, e2, e3 = _moments(mu, tau)
    return (2.0 * e2 - e3) / e1 if e1 > 0 else float("nan")


def taker_fee(p: float) -> float:
    """Kalshi general taker fee per contract, in the same [0,1] units as p."""
    return 0.07 * p * (1.0 - p)
