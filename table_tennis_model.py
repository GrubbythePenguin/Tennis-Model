"""
table_tennis_model.py — match win probability from any game state, plus the
inverse: back out (p, q) from a market match probability via a skill edge.

Same modelling idea as tennis_model.py: every point is independent, I win a
point on my serve with probability p and on my opponent's serve with
probability q. The structural differences from tennis:

    - a "game" (set) is first to 11, win by 2
    - serve alternates every 2 points; from 10-10 it alternates every point
    - a match is best of 5 games (TT Elite Series), first server alternates
      by game
    - the serve advantage is TINY: elite men win ~53% of service points
      (52.78% men / 53.28% women in published elite-match analyses; 53% at
      the 2016 Olympics), versus ~65% in tennis. So p0 = 0.53, q0 = 0.47
      is the equal-skill baseline and everything lives near 50%.

Layers, mirroring tennis_model.py:

    point score -> game     game_win_prob(p, q, x, y, i_served_first)
    game score  -> match    match_win_prob(p, q, a, b, x, y, i_served_first)

and the inverse entry point pq_from_match_prob(target, ...) which applies a
(possibly asymmetric) skill edge to the baseline (p0, q0) and solves for the
edge size so the model match probability equals the market's.

Serve rotation inside a game, with n = points already played (0-indexed next
point) and "0" meaning the player who served the game's first point:

    n < 20  : server = (n // 2) % 2        (two serves each)
    n >= 20 : server = n % 2               (one serve each from 10-10)

The two formulas agree at the boundary (n=18,19 -> player 1; n=20 -> player 0),
which matches the rule that the rotation order simply continues at deuce.

From deuce (any x == y >= 10) the next two points are always one on each
player's serve, so the deuce value is the same closed form as a tennis
tiebreak: D = pq / (pq + (1-p)(1-q)).
"""

import os
from functools import lru_cache

# Bounded for the same reason as tennis_model.py: the memos are keyed on
# continuous p/q, so every solver step mints fresh keys that are never looked
# up again — an unbounded cache is a pure leak in fit/solve loops. The state
# space here is far smaller than tennis (~150 in-game states per (p, q)), so
# the default is modest. TT_CACHE tunes it per process like TENNIS_CACHE.
_CACHE = int(os.environ.get("TT_CACHE") or 100_000)

# Equal-skill baseline: elite men win ~53% of points on serve. At equal skill
# the return-point win probability is the complement.
P0_MEN = 0.53
Q0_MEN = 1.0 - P0_MEN

BEST_OF = 5   # TT Elite Series: best of 5 games to 11


def _server(n):
    """Which player serves point n (0-indexed points already played), where 0
    is the player who served the first point of this game."""
    if n < 20:
        return (n // 2) % 2
    return n % 2


# ---------------------------------------------------------------- point -> game

@lru_cache(maxsize=_CACHE)
def game_win_prob(p, q, x=0, y=0, i_served_first=True):
    """P(I win the current game) at point score x-y (game to 11, win by 2),
    given whether I served the first point of this game."""
    if x >= 11 and x - y >= 2:
        return 1.0
    if y >= 11 and y - x >= 2:
        return 0.0
    D = p * q / (p * q + (1 - p) * (1 - q))        # P(win from deuce)
    if x >= 10 and y >= 10:
        if x == y:
            return D
        server = _server(x + y)
        w = (p if server == 0 else q) if i_served_first else (q if server == 0 else p)
        if x == y + 1:
            return w + (1 - w) * D                 # game point, me
        return w * D                               # game point, them
    server = _server(x + y)
    w = (p if server == 0 else q) if i_served_first else (q if server == 0 else p)
    return (w * game_win_prob(p, q, x + 1, y, i_served_first)
            + (1 - w) * game_win_prob(p, q, x, y + 1, i_served_first))


# ---------------------------------------------------------------- game -> match

@lru_cache(maxsize=_CACHE)
def _match_from_games(p, q, a, b, i_serve_first_this_game, best_of=BEST_OF):
    """P(I win the match) at game score a-b, fresh game about to start."""
    need = best_of // 2 + 1
    if a >= need:
        return 1.0
    if b >= need:
        return 0.0
    g = game_win_prob(p, q, 0, 0, i_serve_first_this_game)
    return (g * _match_from_games(p, q, a + 1, b, not i_serve_first_this_game, best_of)
            + (1 - g) * _match_from_games(p, q, a, b + 1, not i_serve_first_this_game, best_of))


def match_win_prob(p, q, a=0, b=0, x=0, y=0, i_served_first=None, best_of=BEST_OF):
    """P(I win the match) from any state: game score a-b, point score x-y in
    the current game, and whether I served the first point of the current game.

    i_served_first=None (only meaningful at 0-0, 0-0) averages over the coin
    toss — the effect is a fraction of a basis point anyway.
    """
    if i_served_first is None:
        return 0.5 * (match_win_prob(p, q, a, b, x, y, True, best_of)
                      + match_win_prob(p, q, a, b, x, y, False, best_of))
    need = best_of // 2 + 1
    if a >= need:
        return 1.0
    if b >= need:
        return 0.0
    g = game_win_prob(p, q, x, y, i_served_first)
    return (g * _match_from_games(p, q, a + 1, b, not i_served_first, best_of)
            + (1 - g) * _match_from_games(p, q, a, b + 1, not i_served_first, best_of))


def advance_state(a, b, x, y, n, me_wins, best_of=BEST_OF):
    """The state after one player wins the next `n` points straight from
    (a, b, x, y) — the worst/best case a delayed feed can be hiding.

    Rolls a finished game into the set score before each point (the feed shows
    11-9 before the set counter increments, and an advanced point must land in
    the NEXT game, not on top of a finished one). Stops early once the match
    is decided; match_win_prob returns 0/1 on that state regardless of x, y.
    """
    need = best_of // 2 + 1
    for _ in range(n):
        if x >= 11 and x - y >= 2:
            a, x, y = a + 1, 0, 0
        elif y >= 11 and y - x >= 2:
            b, x, y = b + 1, 0, 0
        if a >= need or b >= need:
            break
        if me_wins:
            x += 1
        else:
            y += 1
    return a, b, x, y


# ------------------------------------------------- market prob -> (p, q) edge

def _logit(v):
    from math import log
    return log(v / (1 - v))


def _sigmoid(z):
    from math import exp
    return 1 / (1 + exp(-z))


def pq_from_match_prob(target, p0=P0_MEN, q0=Q0_MEN, serve_share=0.5,
                       best_of=BEST_OF, tol=1e-10):
    """Back out (p, q) from a pregame match probability.

    A single skill edge k is applied to the equal-skill baseline in logit
    space, split between serve and return by `serve_share` s:

        p = sigmoid(logit(p0) + 2s     * k)
        q = sigmoid(logit(q0) + 2(1-s) * k)

    and k is solved (bisection; match prob is strictly increasing in k) so
    that match_win_prob(p, q) == target.

    serve_share = 0.5 is the symmetric default: the favourite's edge shows up
    equally on serve and return, which is the right prior in table tennis
    where the serve advantage itself is only ~3 points and rallies decide
    most points. s > 0.5 concentrates the edge on the favourite's own serve
    (serve + 3rd-ball dominance), s < 0.5 on return. The logit form keeps
    both probabilities in (0, 1) for any edge, and near 50% the probability-
    space shifts are ≈ s·k/2 and (1-s)·k/2 per side. No public dataset pins
    s for TT Elite; fit it from captured tapes once there are fills/obs,
    the way the tennis boundary fit does.

    Returns (p, q). The opponent's pair is (1 - q, 1 - p).
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1)")
    if not 0.0 <= serve_share <= 1.0:
        raise ValueError("serve_share must be in [0, 1]")
    lp, lq = _logit(p0), _logit(q0)
    ws, wr = 2 * serve_share, 2 * (1 - serve_share)

    def pq(k):
        return _sigmoid(lp + ws * k), _sigmoid(lq + wr * k)

    lo, hi = -12.0, 12.0
    for _ in range(200):
        k = 0.5 * (lo + hi)
        p, q = pq(k)
        m = match_win_prob(p, q, best_of=best_of)
        if abs(m - target) < tol:
            break
        if m < target:
            lo = k
        else:
            hi = k
    return pq(0.5 * (lo + hi))


# ------------------------------------------------------------------------ demo

if __name__ == "__main__":
    # Worked example: KXTTELITEMATCH-26SEP071240RSKWTO
    # Rafal Skotniczny vs Wojciech Tobiasz, TT Elite Series, best of 5.
    # Last trade: Skotniczny 48c. Model Skotniczny as "me", target 0.48.
    target = 0.48
    print(f"baseline (equal skill): p0={P0_MEN:.3f} q0={Q0_MEN:.3f} "
          f"-> match {match_win_prob(P0_MEN, Q0_MEN):.4f}")
    print(f"\ntarget match prob {target:.2f} (Skotniczny vs Tobiasz)")
    for s in (0.5, 0.7, 0.3):
        p, q = pq_from_match_prob(target, serve_share=s)
        print(f"  serve_share={s:.1f}: p={p:.4f} q={q:.4f} "
              f"(opponent p'={1-q:.4f} q'={1-p:.4f}) "
              f"-> match {match_win_prob(p, q):.4f}")

    # theoretical mids through the match at the symmetric fit
    p, q = pq_from_match_prob(target)
    print(f"\ntheoretical mids at p={p:.4f} q={q:.4f} (serve_share=0.5):")
    states = [
        ("start of match",                 0, 0, 0, 0, None),
        ("up 5-3 in game 1 (served 1st)",  0, 0, 5, 3, True),
        ("won game 1, 0-0 in game 2",      1, 0, 0, 0, False),
        ("down a game, 0-0 in game 2",     0, 1, 0, 0, False),
        ("1-1, 10-8 game point (recv'd)",  1, 1, 10, 8, False),
        ("1-1, 10-10 deuce",               1, 1, 10, 10, True),
        ("down 0-2, 0-0 in game 3",        0, 2, 0, 0, True),
        ("up 2-1, 9-9 in game 4",          2, 1, 9, 9, False),
        ("2-2, 0-0 in game 5",             2, 2, 0, 0, True),
        ("2-2, 10-10 in game 5",           2, 2, 10, 10, True),
    ]
    for label, a, b, x, y, srv in states:
        m = match_win_prob(p, q, a, b, x, y, srv)
        print(f"  {label:34s} {m:.4f}")
