"""Back-solve a single per-map probability that reconciles the SERIES and MAP books.

Why (26AUG05, FLC/LIQUID post-mortem — see PREGAME_LEAN_SCOPE.md):
`best_prob_for_leg` prices each leg independently, so p1/p2/p3 and series_start
come from different books and need not agree. On
`KXDOTA2GAME-26AUG051300LIQUIDFLC` they didn't: p1..p5 = .685/.665/.645/.68/.68
imply a series price of 79.7 against a market of 76.5. That +3.2c internal
disagreement, plus the market's own series-vs-map gap, produced a standing
one-sided edge that never closed — 57% of map-1 ticks showed edge to buy
Falcons and 0% to sell, and the arber ran to its 45,000 hard cap.

Two solves, both one-dimensional and monotone in p:

  solve_from_series()  — p such that P(win the series | every map at p) equals
                         the MARKET series price. Removes OUR disagreement with
                         the market. On FLC/LIQUID: p = 0.6501.

  solve_reconciled()   — p such that edge_long == edge_short at the opening
                         snapshot, using the real series AND map books through
                         the production hedge math. Removes the pregame edge
                         outright. On FLC/LIQUID: p = 0.630.

The second is what was validated. Replaying FLC/LIQUID with flat p=0.63:
map-1 fires go 57%/0% -> 17%/15% (two-sided again), peak position 45,000 ->
11,100, and the series-leg settled P&L -$29,864 -> +$2,228. The first solve
alone leaves +2.41c of lean and only 1% short-side fires, because it anchors to
the series book and ignores that the map book opened 4.5c richer.

SCOPE: LoL and Dota only for now (`applies_to`). CS2/VAL are BO3 with different
book dynamics and were NOT tested — a flat constant applied across all 87
captured events was far WORSE than production (mean |lean| 13.6c vs 2.8c), so
this must be solved per event, never hardcoded.

Read-only helper; no I/O.
"""
from typing import Optional

from hedge_engine import compute_hedge_profiles, get_loaded_cost_cents

P_LO, P_HI = 0.02, 0.98
SUPPORTED_PREFIXES = ("KXLOLGAME", "KXDOTA2GAME")


def applies_to(event_base: str) -> bool:
    """LoL + Dota only. Everything else keeps today's per-leg probabilities."""
    return any(event_base.startswith(p) for p in SUPPORTED_PREFIXES)


def series_prob(p: float, is_bo5: bool) -> float:
    """P(win the series) when every remaining map is an independent coin at p."""
    need = 3 if is_bo5 else 2

    def f(a: int, b: int) -> float:
        if a >= need:
            return 1.0
        if b >= need:
            return 0.0
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)

    return f(0, 0)


def implied_series_bo3(p1: float, p2: float) -> float:
    """Market-implied BO3 series probability from map probs, decider = (p1+p2)/2."""
    p3 = (p1 + p2) / 2.0
    return p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3


def solve_delta_preserving(series_price: float, p1: float, p2: float,
                           max_shift: float = 0.08):
    """(p1+t, p2+t) — delta preserved, level shifted so implied series == market.

    VAL capture semantics (operator decision 26AUG11, backed by the 80-event
    backtest): the SERIES book is the level authority — its Brier is flat from
    T-120 to kickoff and the maps' own triangle fails to close by >5c in half
    of events — while the map gap (veto delta) is small but real. So keep
    p1−p2 exactly, move both together until the implied series matches the
    market. Example: 0.60/0.58 against a cheaper series -> 0.58/0.56.

    Returns None (caller keeps per-leg probs and logs loudly) when inputs are
    out of range or the required shift exceeds `max_shift` — a divergence that
    size means one of the books is junk, and dragging the maps 8c+ to meet it
    would be fitting noise. BO3 only; BO5 carries an explicit p3..p5 chain.
    """
    if not (P_LO <= p1 <= P_HI and P_LO <= p2 <= P_HI):
        return None
    if not (0.02 <= series_price <= 0.98):
        return None
    t_lo = max(P_LO - p1, P_LO - p2, -max_shift)
    t_hi = min(P_HI - p1, P_HI - p2, max_shift)
    if t_lo >= t_hi:
        return None
    # implied_series_bo3 is monotone increasing when both maps shift together
    if not (implied_series_bo3(p1 + t_lo, p2 + t_lo) <= series_price
            <= implied_series_bo3(p1 + t_hi, p2 + t_hi)):
        return None                       # root outside the band — books junk
    lo, hi = t_lo, t_hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if implied_series_bo3(p1 + mid, p2 + mid) < series_price:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    return p1 + t, p2 + t


def solve_from_series(series_price: float, is_bo5: bool) -> float:
    """p such that series_prob(p) == the market series price (0-1)."""
    lo, hi = P_LO, P_HI
    if series_price <= series_prob(lo, is_bo5):
        return lo
    if series_price >= series_prob(hi, is_bo5):
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if series_prob(mid, is_bo5) < series_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _lean(p, series_bid, series_ask, map_bid, map_ask, is_bo5):
    """(edge_long - edge_short)/2 at this snapshot, via the PRODUCTION hedge math.

    Mirrors live_series_model: the active map enters as cons_map_no_ask /
    cons_map_yes_ask, the series leg is fee-loaded on both sides. Momentum is
    zeroed — it is a live-state adjustment, not a pregame one.
    """
    hp1, hp2 = compute_hedge_profiles(
        0, p, p, series_prob(p, is_bo5), 0.0, 0.0,
        100.0 - float(map_bid), float(map_ask),
        is_bo5=is_bo5, p3=p, p4=p, p5=p, wins_a=0, wins_b=0)
    e_long = 100.0 - get_loaded_cost_cents(float(series_ask)) - hp1["synthetic_cost"]
    e_short = 100.0 - get_loaded_cost_cents(100.0 - float(series_bid)) - hp2["synthetic_cost"]
    return (e_long - e_short) / 2.0


def solve_reconciled(series_bid: float, series_ask: float,
                     map_bid: float, map_ask: float,
                     is_bo5: bool,
                     max_shift: float = 0.06,
                     max_map_spread: float = 10.0) -> Optional[float]:
    """p that makes edge_long == edge_short at the opening snapshot.

    Returns None when the books are too inconsistent to reconcile inside
    `max_shift` of the series-implied p — a wide/one-sided map book can demand
    an absurd p, and silently shipping that would be worse than leaving the
    event alone. Caller should fall back to per-leg probabilities and log it.

    `max_shift` caps how far the answer may sit from `solve_from_series`.
    FLC/LIQUID needed 0.6501 -> 0.630, a shift of 0.020.
    """
    # A wide map book makes the reconciliation meaningless: the solve will still
    # find a root, but it is fitting noise in a book nobody is quoting. Reject
    # loudly instead. `best_prob_for_leg` already bounds spreads to <=15c, so
    # this only bites the genuinely untradeable ones.
    if float(map_ask) - float(map_bid) > max_map_spread:
        return None
    if not (0 < float(map_bid) < float(map_ask) < 100):
        return None
    if not (0 < float(series_bid) < float(series_ask) < 100):
        return None
    anchor = solve_from_series((float(series_bid) + float(series_ask)) / 200.0, is_bo5)
    lo, hi = max(P_LO, anchor - max_shift), min(P_HI, anchor + max_shift)
    f_lo = _lean(lo, series_bid, series_ask, map_bid, map_ask, is_bo5)
    f_hi = _lean(hi, series_bid, series_ask, map_bid, map_ask, is_bo5)
    if f_lo * f_hi > 0:          # no root inside the band — don't extrapolate
        return None
    # lean is monotone INCREASING in p: a lower continuation probability makes
    # the favourite's series cheaper to be short and dearer to be long, which
    # rotates the two edges toward each other. Verified on the FLC/LIQUID
    # sweep: p .6501 -> +2.41c, .6400 -> +1.18c, .6300 -> -0.06c.
    for _ in range(60):
        mid = (lo + hi) / 2
        if _lean(mid, series_bid, series_ask, map_bid, map_ask, is_bo5) > 0:
            hi = mid
        else:
            lo = mid
    p_star = (lo + hi) / 2
    # Converged onto the band edge => the true root is outside `max_shift`.
    # Treat as unreconcilable rather than shipping a clipped answer.
    if abs(p_star - anchor) > max_shift - 1e-4:
        return None
    return p_star


def solve_from_probs(series_prob_0_1: float, series_spread_c: float,
                     map1_prob_0_1: float, map1_spread_c: float,
                     is_bo5: bool, **kw) -> Optional[float]:
    """`solve_reconciled` in the shape populate_configs already has.

    `best_prob_for_leg` returns (prob, source, used_spread) per leg, so the
    two-sided book is reconstructed as mid +/- spread/2 rather than re-fetched.
    """
    sb = 100.0 * series_prob_0_1 - series_spread_c / 2.0
    sa = 100.0 * series_prob_0_1 + series_spread_c / 2.0
    mb = 100.0 * map1_prob_0_1 - map1_spread_c / 2.0
    ma = 100.0 * map1_prob_0_1 + map1_spread_c / 2.0
    return solve_reconciled(sb, sa, mb, ma, is_bo5, **kw)


# ── Precomputed series -> per-map probability, 0.01..0.99 step 0.005 ──
# `solve_from_series` is a pure function of (series price, format), so the whole
# curve is cacheable. 197 grid points per format, loaded once from
# `_series_to_map_prob.csv` (regenerate with `_series_to_map_prob_gen.py`).
# Off-grid inputs are linearly interpolated; the curve is smooth and monotone,
# so max interpolation error across the grid is ~1e-4 — see the test.
_GRID = None


def _load_grid():
    global _GRID
    if _GRID is None:
        import csv as _csv
        import os as _os
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "_series_to_map_prob.csv")
        xs, b3, b5 = [], [], []
        with open(path) as fh:
            for r in _csv.DictReader(fh):
                xs.append(float(r["series_prob"]))
                b3.append(float(r["bo3_map_prob"]))
                b5.append(float(r["bo5_map_prob"]))
        _GRID = (xs, b3, b5)
    return _GRID


def map_prob_cached(series_price: float, is_bo5: bool) -> float:
    """Cached/interpolated `solve_from_series`. Same contract, no bisection."""
    xs, b3, b5 = _load_grid()
    ys = b5 if is_bo5 else b3
    if series_price <= xs[0]:
        return ys[0]
    if series_price >= xs[-1]:
        return ys[-1]
    import bisect as _b
    i = _b.bisect_right(xs, series_price) - 1
    if i >= len(xs) - 1:
        return ys[-1]
    t = (series_price - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])


def _lean_p1_fixed(p_rest, p1, series_bid, series_ask, map_bid, map_ask, is_bo5):
    """Lean with map 1 held at its REAL market mid and only p2..p5 varying."""
    hp1, hp2 = compute_hedge_profiles(
        0, p1, p_rest, series_prob(p_rest, is_bo5), 0.0, 0.0,
        100.0 - float(map_bid), float(map_ask),
        is_bo5=is_bo5, p3=p_rest, p4=p_rest, p5=p_rest, wins_a=0, wins_b=0)
    e_long = 100.0 - get_loaded_cost_cents(float(series_ask)) - hp1["synthetic_cost"]
    e_short = 100.0 - get_loaded_cost_cents(100.0 - float(series_bid)) - hp2["synthetic_cost"]
    return (e_long - e_short) / 2.0


def solve_continuation(series_bid: float, series_ask: float,
                       map_bid: float, map_ask: float,
                       p1_market: float, is_bo5: bool,
                       max_shift: float = 0.10,
                       max_map_spread: float = 10.0) -> Optional[float]:
    """Backsolve p2..p5 with p1 PINNED to the real map-1 mid.

    This is the form that ships: map 1 already has a real, liquid two-sided
    book, so replacing p1 with a synthetic value throws away the best
    information we have. Only the CONTINUATION probability — the one we have no
    market for pregame, and which is the sole lever at state (0,0) — gets
    solved.

    For BO5 the answer equals `solve_reconciled`: at (0,0) the active map enters
    the hedge as a live trade price, so p1 never appears and pinning it changes
    nothing. For BO3 it does differ, because `solve_game_3_probability` derives
    the G3 prior as (p1 + p2)/2, so a pinned p1 drags G3 with it.

    Returns None on the same guards as `solve_reconciled`.
    """
    if float(map_ask) - float(map_bid) > max_map_spread:
        return None
    if not (0 < float(map_bid) < float(map_ask) < 100):
        return None
    if not (0 < float(series_bid) < float(series_ask) < 100):
        return None
    if not (P_LO < float(p1_market) < P_HI):
        return None
    anchor = map_prob_cached((float(series_bid) + float(series_ask)) / 200.0, is_bo5)
    lo, hi = max(P_LO, anchor - max_shift), min(P_HI, anchor + max_shift)
    f = lambda x: _lean_p1_fixed(x, p1_market, series_bid, series_ask,
                                 map_bid, map_ask, is_bo5)
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(60):                      # lean is monotone INCREASING in p
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    p_star = (lo + hi) / 2
    if abs(p_star - anchor) > max_shift - 1e-4:
        return None
    return p_star


def solve_continuation_from_probs(series_prob_0_1: float, series_spread_c: float,
                                  map1_prob_0_1: float, map1_spread_c: float,
                                  is_bo5: bool, **kw) -> Optional[float]:
    """`solve_continuation` in the shape populate_configs already has.

    `best_prob_for_leg` returns (prob, source, used_spread) per leg, so the
    two-sided book is reconstructed as mid +/- spread/2 rather than re-fetched.
    p1 is passed through pinned — only p2..p5 are solved.
    """
    sb = 100.0 * series_prob_0_1 - series_spread_c / 2.0
    sa = 100.0 * series_prob_0_1 + series_spread_c / 2.0
    mb = 100.0 * map1_prob_0_1 - map1_spread_c / 2.0
    ma = 100.0 * map1_prob_0_1 + map1_spread_c / 2.0
    return solve_continuation(sb, sa, mb, ma, map1_prob_0_1, is_bo5, **kw)
