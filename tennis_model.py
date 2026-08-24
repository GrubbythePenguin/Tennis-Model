"""
tennis_model.py — set (and match) win probability from any game state.

Model: every point is independent. I win a point on my serve with probability p
and a point on my opponent's serve with probability q. No fatigue, no momentum.

The model is built in layers, each a small recursion that bottoms out in a
closed form for the "deuce" states:

    point score  -> game        game_win_prob(w, x, y)
    point score  -> tiebreak    tiebreak_win_prob(p, q, x, y, i_serve_next)
    game score   -> set         set_from_games(p, q, a, b, i_serve_next_game)

and the entry point set_win_prob(...) glues them together for any state:
current game score, who is serving, and the point score inside the current game.

Points are integers 0, 1, 2, 3, 4, ... (3-3 is deuce, 4-3 is my advantage).
Tennis notation is also accepted as strings: '0', '15', '30', '40', 'AD'.
"""

from functools import lru_cache

# ---------------------------------------------------------------- point -> game

@lru_cache(maxsize=None)
def game_win_prob(w, x=0, y=0):
    """P(I win the current game) at point score x-y, where w is my per-point win
    probability in this game (p if I'm serving, q if I'm returning)."""
    if x >= 4 and x - y >= 2:
        return 1.0
    if y >= 4 and y - x >= 2:
        return 0.0
    if x >= 3 and y >= 3:                          # deuce / advantage territory
        k = w * w / (w * w + (1 - w) ** 2)         # P(win from deuce)
        if x == y:
            return k                               # deuce
        if x == y + 1:
            return w + (1 - w) * k                 # my advantage
        return w * k                               # their advantage
    return w * game_win_prob(w, x + 1, y) + (1 - w) * game_win_prob(w, x, y + 1)


def G(w):
    """Closed form for a game from 0-0 (identical to game_win_prob(w, 0, 0))."""
    return w ** 4 * (15 - 4 * w - 10 * w ** 2 / (1 - 2 * w + 2 * w ** 2))


# ------------------------------------------------------------ point -> tiebreak

@lru_cache(maxsize=None)
def tiebreak_win_prob(p, q, x=0, y=0, i_serve_next=True):
    """P(I win the tiebreak) at point score x-y, given whether I serve the next
    point. Rotation: one point, then two each; i.e. the server changes after
    every odd-numbered point of the tiebreak."""
    if x >= 7 and x - y >= 2:
        return 1.0
    if y >= 7 and y - x >= 2:
        return 0.0
    D = p * q / (p * q + (1 - p) * (1 - q))        # P(win from 6-6, 7-7, ...) — server order irrelevant
    w = p if i_serve_next else q
    if x >= 6 and y >= 6:
        if x == y:
            return D
        if x == y + 1:
            return w + (1 - w) * D
        return w * D
    n = x + y + 1                                  # the point about to be played
    serve_after = (not i_serve_next) if n % 2 == 1 else i_serve_next
    return (w * tiebreak_win_prob(p, q, x + 1, y, serve_after)
            + (1 - w) * tiebreak_win_prob(p, q, x, y + 1, serve_after))


# ---------------------------------------------------------------- game -> set

@lru_cache(maxsize=None)
def set_from_games(p, q, a, b, i_serve_next_game=True):
    """P(I win the set) at game score a-b, standing at the start of a game."""
    if a == 6 and b == 6:
        return tiebreak_win_prob(p, q, 0, 0, i_serve_next_game)
    if a >= 6 and a - b >= 2:
        return 1.0
    if b >= 6 and b - a >= 2:
        return 0.0
    w = G(p) if i_serve_next_game else G(q)
    return (w * set_from_games(p, q, a + 1, b, not i_serve_next_game)
            + (1 - w) * set_from_games(p, q, a, b + 1, not i_serve_next_game))


# ------------------------------------------------------------- entry points

_TENNIS = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'A': 4}


def _pts(v):
    """Integers are raw point counts; strings are tennis notation."""
    if isinstance(v, str):
        return _TENNIS[v.strip().upper()]
    return int(v)


def set_win_prob(p, q, games_me=0, games_opp=0, i_serve=True, points_me=0, points_opp=0):
    """P(I win the set) from any state.

    games_me, games_opp   : game score in the set
    i_serve               : True if I'm serving the current game
                            (at 6-6: True if I serve the next tiebreak point)
    points_me, points_opp : point score in the current game (ints, or '0'/'15'/'30'/'40'/'AD');
                            at 6-6 these are tiebreak points
    """
    x, y = _pts(points_me), _pts(points_opp)
    if games_me == 6 and games_opp == 6:
        return tiebreak_win_prob(p, q, x, y, i_serve)
    if games_me >= 6 and games_me - games_opp >= 2:
        return 1.0
    if games_opp >= 6 and games_opp - games_me >= 2:
        return 0.0
    w = p if i_serve else q
    g = game_win_prob(w, x, y)
    return (g * set_from_games(p, q, games_me + 1, games_opp, not i_serve)
            + (1 - g) * set_from_games(p, q, games_me, games_opp + 1, not i_serve))


def match_win_prob(p, q, sets_me=0, sets_opp=0, best_of=3, **state):
    """P(I win the match). `state` describes the set in progress (same keyword
    arguments as set_win_prob). Future sets start fresh at 0-0, and a fresh set's
    win probability doesn't depend on who serves first, so no server tracking is
    needed beyond the current set. Assumes a tiebreak at 6-6 in every set."""
    need = best_of // 2 + 1
    if sets_me >= need:
        return 1.0
    if sets_opp >= need:
        return 0.0
    s = set_win_prob(p, q, **state)
    return (s * match_win_prob(p, q, sets_me + 1, sets_opp, best_of)
            + (1 - s) * match_win_prob(p, q, sets_me, sets_opp + 1, best_of))


# ------------------------------------------------------------------- demo

if __name__ == "__main__":
    p, q = 0.65, 0.40

    print(f"p = {p}, q = {q}")
    print(f"hold  G(p) = {G(p):.4f}   break G(q) = {G(q):.4f}   tiebreak = {tiebreak_win_prob(p, q):.4f}\n")

    examples = [
        ("0-0, I serve",                          dict(games_me=0, games_opp=0, i_serve=True)),
        ("3-1 up, opponent serving",              dict(games_me=3, games_opp=1, i_serve=False)),
        ("3-1 up, I'm serving",                   dict(games_me=3, games_opp=1, i_serve=True)),
        ("3-1 up, opp serving, 0-30 (I lead)",    dict(games_me=3, games_opp=1, i_serve=False, points_me='30', points_opp='0')),
        ("3-1 up, opp serving, 30-0 (they lead)", dict(games_me=3, games_opp=1, i_serve=False, points_me='0', points_opp='30')),
        ("4-1 up, I'm serving",                   dict(games_me=4, games_opp=1, i_serve=True)),
        ("5-5, I serve, deuce",                   dict(games_me=5, games_opp=5, i_serve=True, points_me=3, points_opp=3)),
        ("6-6, tiebreak 3-5, I serve next",       dict(games_me=6, games_opp=6, i_serve=True, points_me=3, points_opp=5)),
    ]
    for label, state in examples:
        print(f"{label:42s} P(win set) = {set_win_prob(p, q, **state):.4f}")

    print("\nP(win set) by game score, opponent serving the next game (rows = my games, cols = theirs):")
    print("     " + "".join(f"{b:>7d}" for b in range(6)))
    for a in range(6):
        row = "".join(f"{set_from_games(p, q, a, b, i_serve_next_game=False):7.3f}" for b in range(6))
        print(f"{a:>4d} {row}")

    print(f"\nBest-of-3 match: lost the first set, now 3-1 up in the second, opponent serving: "
          f"{match_win_prob(p, q, 0, 1, 3, games_me=3, games_opp=1, i_serve=False):.4f}")
