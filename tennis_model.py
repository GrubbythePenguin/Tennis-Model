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

import os
from functools import lru_cache

# CACHE SIZE — DO NOT SET THIS BACK TO None. The three memos below are keyed on p/q,
# which are CONTINUOUS fit parameters, so every Levenberg-Marquardt step and every one
# of the 961 points in ImpliedModel._fit's cold-start grid mints a fresh key set that
# is never looked up again. The memo only pays off WITHIN a single (p, q) evaluation;
# across evaluations an unbounded cache is a pure leak. Measured 26AUG25 on a 17-point
# match: 11,797 distinct (p, q) pairs, 1.6M live entries, 288 MB — growing ~450 MB per
# match for the life of the process, which is what was OOM-killing the box during
# ewma_fit.py grid runs. Bounding is behaviour-preserving (verified: identical error
# checksums over 5 matches, 31.5s -> 33.9s, 2,263 MB -> 217 MB). Quantising p/q into
# the key would ALSO bound it but changes fit results — don't do that instead.
#
# TENNIS_CACHE tunes it per process, because the two workloads want opposite things:
#   backtests  one process, wants speed, has the whole box  -> 200k, ~235 MB, fastest
#   live       ~20 concurrent pollers sharing the box with the trading system
#              -> 25k, ~56 MB each. Costs ~150 ms more per boundary refit, which is
#              nothing against boundaries that arrive minutes apart, and turns
#              20 x 235 MB = 4.7 GB into 20 x 56 MB = 1.1 GB. discover.py sets this
#              for every poller it launches.
_CACHE = int(os.environ.get("TENNIS_CACHE") or 200_000)

# ---------------------------------------------------------------- point -> game

@lru_cache(maxsize=_CACHE)
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

@lru_cache(maxsize=_CACHE)
def tiebreak_win_prob(p, q, x=0, y=0, i_serve_next=True, target=7):
    """P(I win the tiebreak) at point score x-y, given whether I serve the next
    point. Rotation: one point, then two each; i.e. the server changes after
    every odd-numbered point of the tiebreak.

    `target` is the points needed (win by 2 regardless):
        7   standard set tiebreak at 6-6 — every set on the regular tour
        10  DECIDING-SET tiebreak at the Grand Slams (unified rule since 2022)

    The serving rotation and the deuce value D are identical for both. D is
    pq/(pq+(1-p)(1-q)) at ANY level score from target-1 onward, because from there
    the next two points are always one on each player's serve — at 6-6 of a 7-point
    breaker point 13 is mine and 14 is theirs, at 9-9 of a 10-point breaker point 19
    is theirs and 20 is mine. Order does not matter to the product.
    """
    if x >= target and x - y >= 2:
        return 1.0
    if y >= target and y - x >= 2:
        return 0.0
    D = p * q / (p * q + (1 - p) * (1 - q))        # P(win from target-1 all, and every level score after)
    w = p if i_serve_next else q
    if x >= target - 1 and y >= target - 1:
        if x == y:
            return D
        if x == y + 1:
            return w + (1 - w) * D
        return w * D
    n = x + y + 1                                  # the point about to be played
    serve_after = (not i_serve_next) if n % 2 == 1 else i_serve_next
    return (w * tiebreak_win_prob(p, q, x + 1, y, serve_after, target)
            + (1 - w) * tiebreak_win_prob(p, q, x, y + 1, serve_after, target))


# ---------------------------------------------------------------- game -> set

@lru_cache(maxsize=_CACHE)
def set_from_games(p, q, a, b, i_serve_next_game=True, tb_target=7):
    """P(I win the set) at game score a-b, standing at the start of a game.

    tb_target is the tiebreak length for THIS set — 10 in a Grand Slam decider.
    """
    if a == 6 and b == 6:
        return tiebreak_win_prob(p, q, 0, 0, i_serve_next_game, tb_target)
    if a >= 6 and a - b >= 2:
        return 1.0
    if b >= 6 and b - a >= 2:
        return 0.0
    w = G(p) if i_serve_next_game else G(q)
    return (w * set_from_games(p, q, a + 1, b, not i_serve_next_game, tb_target)
            + (1 - w) * set_from_games(p, q, a, b + 1, not i_serve_next_game, tb_target))


# ------------------------------------------------------------- entry points

_TENNIS = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'A': 4}


def _pts(v):
    """Integers are raw point counts; strings are tennis notation."""
    if isinstance(v, str):
        return _TENNIS[v.strip().upper()]
    return int(v)


def set_win_prob(p, q, games_me=0, games_opp=0, i_serve=True, points_me=0, points_opp=0,
                 tb_target=7):
    """P(I win the set) from any state.

    games_me, games_opp   : game score in the set
    i_serve               : True if I'm serving the current game
                            (at 6-6: True if I serve the next tiebreak point)
    points_me, points_opp : point score in the current game (ints, or '0'/'15'/'30'/'40'/'AD');
                            at 6-6 these are tiebreak points
    """
    x, y = _pts(points_me), _pts(points_opp)
    if games_me == 6 and games_opp == 6:
        return tiebreak_win_prob(p, q, x, y, i_serve, tb_target)
    if games_me >= 6 and games_me - games_opp >= 2:
        return 1.0
    if games_opp >= 6 and games_opp - games_me >= 2:
        return 0.0
    w = p if i_serve else q
    g = game_win_prob(w, x, y)
    return (g * set_from_games(p, q, games_me + 1, games_opp, not i_serve, tb_target)
            + (1 - g) * set_from_games(p, q, games_me, games_opp + 1, not i_serve, tb_target))


def set_number_win_prob(p, q, set_no, sets_me=0, sets_opp=0, best_of=3,
                        final_set_tb=7, **state):
    """P(I win set number `set_no` of the match), from the current state.

    `state` describes the set IN PROGRESS (same keyword arguments as set_win_prob,
    minus tb_target, which is derived here). Three cases:

        set_no == the set in progress   set_win_prob on the live state
        set_no  > the set in progress   integrate over reaching it: future sets start
                                        fresh at 0-0 (server-independent), and a set
                                        that is NEVER PLAYED counts as won by NEITHER
                                        player, so P(me wins set N) + P(opp wins set N)
                                        = P(set N is played), not 1. The two sides of a
                                        future-set market are complements only
                                        conditional on the set happening.
        set_no <= sets already played   ValueError - the set is decided, and this
                                        state does not record WHO won it, so any
                                        answer would be a guess.

    The unplayed-set convention matters to whoever prices a market off this: if the
    venue instead VOIDS an unplayed set's market (refund, not No), the quotable value
    is P(win set N | set N played) = this / (this_me + this_opp). Verify the actual
    resolution rules before quoting set 3 of a best-of-3 (or sets 4-5 of a 5).

    tb_target plumbing mirrors match_win_prob: the deciding set (number `best_of`)
    tiebreaks to final_set_tb, every other set to 7.
    """
    need = best_of // 2 + 1
    played = sets_me + sets_opp
    if not 1 <= set_no <= best_of:
        raise ValueError(f"set_no {set_no} outside 1..{best_of}")
    if sets_me >= need or sets_opp >= need:
        raise ValueError(f"match already decided at {sets_me}-{sets_opp}")
    if set_no <= played:
        raise ValueError(f"set {set_no} already played at {sets_me}-{sets_opp}; "
                         "its winner is not recoverable from set COUNTS")

    def future(sm, so):
        """P(I win set `set_no`) standing at the START of set sm+so+1 (fresh)."""
        if sm >= need or so >= need:
            return 0.0                       # match over first -> set never played
        idx = sm + so + 1
        s0 = set_from_games(p, q, 0, 0, True,
                            final_set_tb if idx == best_of else 7)
        if idx == set_no:
            return s0
        return s0 * future(sm + 1, so) + (1 - s0) * future(sm, so + 1)

    cur = played + 1
    s = set_win_prob(p, q, tb_target=(final_set_tb if cur == best_of else 7), **state)
    if set_no == cur:
        return s
    return s * future(sets_me + 1, sets_opp) + (1 - s) * future(sets_me, sets_opp + 1)


def match_win_prob(p, q, sets_me=0, sets_opp=0, best_of=3, final_set_tb=7, **state):
    """P(I win the match). `state` describes the set in progress (same keyword
    arguments as set_win_prob). Future sets start fresh at 0-0, and a fresh set's
    win probability doesn't depend on who serves first, so no server tracking is
    needed beyond the current set.

    final_set_tb is the tiebreak length in the DECIDING set:
        7   regular tour — every set is a 7-point breaker at 6-6
        10  Grand Slams — since the 2022 unified rule, the deciding set is decided
            by a 10-point breaker at 6-6 (third set in best-of-3, fifth in best-of-5)
    Non-deciding sets are always 7. Getting this wrong misprices any match that can
    reach 6-6 in the final set — note that includes states BEFORE the decider, since
    the match price integrates over reaching one.

    Direction of the effect: a LONGER breaker HELPS the better player. More points
    means less variance, so the stronger player converts their edge more reliably —
    at p=0.65/q=0.40 a 7-point breaker is worth 0.5816 and a 10-point one 0.5954
    (verified by simulation). The intuition that a longer race is "closer to a coin
    flip" is backwards; it is the SHORT race that is the coin flip.
    """
    need = best_of // 2 + 1
    if sets_me >= need:
        return 1.0
    if sets_opp >= need:
        return 0.0
    decider = (sets_me + sets_opp) == best_of - 1
    s = set_win_prob(p, q, tb_target=(final_set_tb if decider else 7), **state)
    return (s * match_win_prob(p, q, sets_me + 1, sets_opp, best_of, final_set_tb)
            + (1 - s) * match_win_prob(p, q, sets_me, sets_opp + 1, best_of, final_set_tb))


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
