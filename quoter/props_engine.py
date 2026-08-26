"""
BO3 prop market probability math.

Computes conservative bid/ask theos for Kalshi esports prop markets:
  - Over/Under 2.5 maps (will the match go to a 3rd game?)
  - Team -1.5 (will team X win 2-0 sweep?)

Convention: probabilities are returned in [0, 1] (callers multiply ×100 for
cents). "bid" = conservative LOW probability (what we'd pay buying YES);
"ask" = conservative HIGH probability (what we'd offer selling YES). The
spread comes from the live market range on the currently-active game's
underlying winner market — game 1 pre-game, game 2 mid-match.

State convention mirrors hedge_engine.detect_bo3_state:
  0 = pre-game (no maps decided)
  1 = team A won game 1, game 2 active
  2 = team B won game 1, game 2 active
  3 = match decided / on game 3 — props are no longer interesting

Worked example (user-provided):
  Game 1 market = 56@60 (YES bid 56c, YES ask 60c on team A)
  Game 2 baseline = 52/48 (team A favored at p2_a = 0.52)
  → P(Over 2.5) mid = 1 - 0.58 × 0.52 - 0.42 × 0.48 = 0.4968
  → bid_theo (conservative low) = 1 - max(p1_a × p2_a) - max(p1_b × p2_b) = 0.4768
  → ask_theo (conservative high) = 1 - min(p1_a × p2_a) - min(p1_b × p2_b) = 0.5168
"""
import os
from dataclasses import dataclass, asdict
from typing import Optional


# ── Runtime lever: quote props during active maps (states 1, 2) ──────
# Default ON: we quote BO3 props pre-game (state 0) AND during game 2
# (states 1, 2). During states 1/2, the O/U 2.5 prop is mathematically
# equivalent to the game-2 winner market — i.e., we're effectively
# quoting g2 through a prop wrapper, exposing us to the same adverse
# selection as quoting maps directly.
#
# To kill state-1/2 prop quoting at runtime (panic switch, no restart):
#   touch disable_props_during_active_map.flag
# To restore:
#   rm disable_props_during_active_map.flag
#
# Pre-game (state 0) quoting is always on — that's where the math
# actually synthesizes fresh price discovery from g1 + g2 priors, not
# just relabels g2.
_DISABLE_FLAG = "disable_props_during_active_map.flag"


def props_quote_during_active_map_enabled() -> bool:
    """True iff we should quote BO3 props during active maps (states 1, 2).

    Default ON. Returns False when `disable_props_during_active_map.flag`
    exists in the working directory.
    """
    return not os.path.exists(_DISABLE_FLAG)


def should_emit_prop_theos(state: int) -> bool:
    """Whether the props theo generator should emit a quote at this state.

    State 0 (pre-game): always quote — synthesizes fresh price discovery.
    State 1 / 2 (active map): controlled by the disable flag.
    State 3 (match decided): never quote.
    """
    if state == 0:
        return True
    if state in (1, 2):
        return props_quote_during_active_map_enabled()
    return False


@dataclass
class Bo3PropTheos:
    """Bid/ask theos for the four BO3 prop markets. All values in [0, 1]."""
    over_2_5_bid: float
    over_2_5_ask: float
    under_2_5_bid: float
    under_2_5_ask: float
    team_a_sweep_bid: float   # team A -1.5 (A wins 2-0)
    team_a_sweep_ask: float
    team_b_sweep_bid: float   # team B -1.5 (B wins 2-0)
    team_b_sweep_ask: float

    def to_cents(self) -> "Bo3PropTheos":
        """Return a copy with all probabilities scaled by 100."""
        return Bo3PropTheos(**{k: v * 100.0 for k, v in asdict(self).items()})


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def bo3_prop_theos(
    state: int,
    p1_a_bid: float,
    p1_a_ask: float,
    p2_a: float = 0.5,
    p2_a_bid: Optional[float] = None,
    p2_a_ask: Optional[float] = None,
) -> Bo3PropTheos:
    """Compute conservative bid/ask theos for BO3 props at the given state.

    Args:
      state:      Match state (0 pre-game, 1 A-won-g1, 2 B-won-g1).
      p1_a_bid:   Game-1 YES bid for team A (lower bound on p_a, in [0, 1]).
      p1_a_ask:   Game-1 YES ask for team A (upper bound on p_a). Ignored in
                  states 1/2 (game 1 is decided).
      p2_a:       Point estimate of p(A wins game 2). Used when game-2
                  market range isn't supplied.
      p2_a_bid:   Optional — game-2 YES bid on team A when game 2 is live.
                  When provided, replaces p2_a as the lower bound for the
                  game-2 contribution to the prop math.
      p2_a_ask:   Optional — game-2 YES ask on team A.

    Returns:
      Bo3PropTheos with bid/ask for over/under 2.5 and team -1.5 sweep.

    Raises:
      ValueError on invalid state or out-of-range probabilities.
    """
    if state not in (0, 1, 2):
        raise ValueError(
            f"Unsupported state {state} — BO3 prop theos only valid in "
            f"states 0, 1, 2 (state 3 = match decided)"
        )
    if not (0.0 <= p1_a_bid <= p1_a_ask <= 1.0):
        raise ValueError(
            f"Invalid game-1 market: bid={p1_a_bid} ask={p1_a_ask} "
            f"(must satisfy 0 ≤ bid ≤ ask ≤ 1)"
        )

    # Default g2 market range from point estimate when caller didn't supply.
    p2_a_bid = p2_a if p2_a_bid is None else p2_a_bid
    p2_a_ask = p2_a if p2_a_ask is None else p2_a_ask
    if not (0.0 <= p2_a_bid <= p2_a_ask <= 1.0):
        raise ValueError(
            f"Invalid game-2 market: bid={p2_a_bid} ask={p2_a_ask}"
        )

    if state == 0:
        # Pre-game: both game-1 outcomes possible. Sweep range comes from
        # cross-product of game-1 and game-2 market ranges.
        p1_b_bid = 1.0 - p1_a_ask  # opp bid = 100 - our ask
        p1_b_ask = 1.0 - p1_a_bid
        p2_b_bid = 1.0 - p2_a_ask
        p2_b_ask = 1.0 - p2_a_bid

        a_sweep_min = p1_a_bid * p2_a_bid
        a_sweep_max = p1_a_ask * p2_a_ask
        b_sweep_min = p1_b_bid * p2_b_bid
        b_sweep_max = p1_b_ask * p2_b_ask

        over_2_5_bid = _clamp01(1.0 - a_sweep_max - b_sweep_max)
        over_2_5_ask = _clamp01(1.0 - a_sweep_min - b_sweep_min)

        return Bo3PropTheos(
            over_2_5_bid=over_2_5_bid,
            over_2_5_ask=over_2_5_ask,
            under_2_5_bid=_clamp01(1.0 - over_2_5_ask),
            under_2_5_ask=_clamp01(1.0 - over_2_5_bid),
            team_a_sweep_bid=_clamp01(a_sweep_min),
            team_a_sweep_ask=_clamp01(a_sweep_max),
            team_b_sweep_bid=_clamp01(b_sweep_min),
            team_b_sweep_ask=_clamp01(b_sweep_max),
        )

    if state == 1:
        # A won g1. Match goes to a 3rd game iff B wins g2 (else A sweeps).
        # P(A sweep) = p(A wins g2) — uses g2 market range.
        # P(B sweep) = 0 (B already lost g1).
        # P(Over 2.5) = 1 - p(A wins g2) = p(B wins g2).
        return Bo3PropTheos(
            over_2_5_bid=_clamp01(1.0 - p2_a_ask),
            over_2_5_ask=_clamp01(1.0 - p2_a_bid),
            under_2_5_bid=_clamp01(p2_a_bid),
            under_2_5_ask=_clamp01(p2_a_ask),
            team_a_sweep_bid=_clamp01(p2_a_bid),
            team_a_sweep_ask=_clamp01(p2_a_ask),
            team_b_sweep_bid=0.0,
            team_b_sweep_ask=0.0,
        )

    # state == 2: B won g1. Mirror of state 1.
    return Bo3PropTheos(
        over_2_5_bid=_clamp01(p2_a_bid),
        over_2_5_ask=_clamp01(p2_a_ask),
        under_2_5_bid=_clamp01(1.0 - p2_a_ask),
        under_2_5_ask=_clamp01(1.0 - p2_a_bid),
        team_a_sweep_bid=0.0,
        team_a_sweep_ask=0.0,
        team_b_sweep_bid=_clamp01(1.0 - p2_a_ask),
        team_b_sweep_ask=_clamp01(1.0 - p2_a_bid),
    )


# ═══════════════════════════════════════════════════════════════════════
# BO5 props — *** UNTESTED — DO NOT USE IN LIVE TRADING WITHOUT VALIDATION ***
# Kalshi has not yet published BO5 prop tickers in the wild, so this code
# has not been verified against actual market state. Shipping for future
# readiness. When BO5 props appear: validate path-walk math against simple
# closed-form cases (50/50/.../50 with no momentum) and a hand-computed
# 3-game scenario before trusting at scale.
# ═══════════════════════════════════════════════════════════════════════
BO5_VALIDATED = False  # flip to True after live verification


@dataclass
class Bo5PropTheos:
    """Bid/ask theos for BO5 prop markets. All values in [0, 1].

    UNTESTED — verify before live use.
    """
    over_3_5_bid: float
    over_3_5_ask: float
    under_3_5_bid: float
    under_3_5_ask: float
    over_4_5_bid: float
    over_4_5_ask: float
    under_4_5_bid: float
    under_4_5_ask: float
    team_a_minus_2_5_bid: float   # team A wins 3-0
    team_a_minus_2_5_ask: float
    team_a_minus_1_5_bid: float   # team A wins 3-0 or 3-1
    team_a_minus_1_5_ask: float
    team_b_minus_2_5_bid: float
    team_b_minus_2_5_ask: float
    team_b_minus_1_5_bid: float
    team_b_minus_1_5_ask: float

    def to_cents(self) -> "Bo5PropTheos":
        return Bo5PropTheos(**{k: v * 100.0 for k, v in asdict(self).items()})


def bo5_terminal_probs(
    p1: float, p2: float, p3: float, p4: float, p5: float,
    g2_mom: float = 0.0, g3_mom: float = 0.0,
) -> dict:
    """Walk all BO5 paths → dict of {(a_wins, b_wins): probability}.

    Convention (matches hedge_engine):
      - p_i = baseline P(team A wins map i)
      - g2_mom = momentum bump added to map-2 prob conditional on map-1 winner.
        If A won map 1 → p2_a_effective = p2 + g2_mom (clipped to [0, 1]).
        If B won map 1 → p2_a_effective = p2 - g2_mom.
      - g3_mom = same, conditional on map-2 winner.
      - Maps 4, 5: baseline p4, p5 used directly (no per-game momentum modeled).

    Sums to 1.0 over all 6 terminal states: (3,0), (3,1), (3,2), (0,3), (1,3), (2,3).
    """
    from collections import defaultdict
    results: dict = defaultdict(float)

    def walk(game_idx: int, a_wins: int, b_wins: int, prev: str, prob: float):
        if a_wins == 3 or b_wins == 3:
            results[(a_wins, b_wins)] += prob
            return
        if game_idx == 1:
            p_a = p1
        elif game_idx == 2:
            mom = g2_mom if prev == "A" else (-g2_mom if prev == "B" else 0.0)
            p_a = _clamp01(p2 + mom)
        elif game_idx == 3:
            mom = g3_mom if prev == "A" else (-g3_mom if prev == "B" else 0.0)
            p_a = _clamp01(p3 + mom)
        elif game_idx == 4:
            p_a = p4
        elif game_idx == 5:
            p_a = p5
        else:
            return
        if p_a > 0.0:
            walk(game_idx + 1, a_wins + 1, b_wins, "A", prob * p_a)
        if p_a < 1.0:
            walk(game_idx + 1, a_wins, b_wins + 1, "B", prob * (1.0 - p_a))

    walk(1, 0, 0, "", 1.0)
    return dict(results)


def _bo5_prop_values(terms: dict) -> dict:
    """Convert terminal-state probs to prop probabilities."""
    p_a_30 = terms.get((3, 0), 0.0)
    p_a_31 = terms.get((3, 1), 0.0)
    p_a_32 = terms.get((3, 2), 0.0)
    p_b_30 = terms.get((0, 3), 0.0)
    p_b_31 = terms.get((1, 3), 0.0)
    p_b_32 = terms.get((2, 3), 0.0)
    return {
        "over_3_5": 1.0 - p_a_30 - p_b_30,  # ≥4 maps played
        "over_4_5": p_a_32 + p_b_32,         # goes to 5 maps
        "a_minus_2_5": p_a_30,               # A wins 3-0
        "a_minus_1_5": p_a_30 + p_a_31,      # A wins 3-0 or 3-1
        "b_minus_2_5": p_b_30,
        "b_minus_1_5": p_b_30 + p_b_31,
    }


def bo5_prop_theos(
    p1_bid: float, p1_ask: float,
    p2_bid: float, p2_ask: float,
    p3_bid: float, p3_ask: float,
    p4_bid: Optional[float] = None, p4_ask: Optional[float] = None,
    p5_bid: Optional[float] = None, p5_ask: Optional[float] = None,
    p4_default: float = 0.5, p5_default: float = 0.5,
    g2_mom: float = 0.0, g3_mom: float = 0.0,
) -> Bo5PropTheos:
    """Conservative bid/ask theos for BO5 props via corner walk.

    *** UNTESTED in live trading. Validate before using with real money. ***

    For each (bid, ask) pair, evaluate the prop math at both extremes; walk
    all 32 corners of the 5-map input space; aggregate min for the bid theo
    and max for the ask theo. This is the BO5 analog of the BO3 worked
    formula `bid = 1 - max(P_A_sweep) - max(P_B_sweep)` — for BO5, the
    multi-path nature of 3-1 / 3-2 outcomes means we walk corners rather
    than write a closed-form expression.

    For maps 4 and 5 (which often aren't listed pre-match), pass None to
    fall back to `p4_default` / `p5_default` as point estimates. To express
    "untradeable", pass bid=0.01, ask=0.99 — produces a near-[0, 1] output
    quote (effectively don't-quote-me).
    """
    from itertools import product

    if p4_bid is None: p4_bid = p4_default
    if p4_ask is None: p4_ask = p4_default
    if p5_bid is None: p5_bid = p5_default
    if p5_ask is None: p5_ask = p5_default

    for label, b, a in [("p1", p1_bid, p1_ask), ("p2", p2_bid, p2_ask),
                        ("p3", p3_bid, p3_ask), ("p4", p4_bid, p4_ask),
                        ("p5", p5_bid, p5_ask)]:
        if not (0.0 <= b <= a <= 1.0):
            raise ValueError(f"Invalid {label}: bid={b} ask={a}")

    ranges = [(p1_bid, p1_ask), (p2_bid, p2_ask), (p3_bid, p3_ask),
              (p4_bid, p4_ask), (p5_bid, p5_ask)]

    prop_vals = []
    for corner in product(*[(lo, hi) for lo, hi in ranges]):
        p1, p2, p3, p4, p5 = corner
        terms = bo5_terminal_probs(p1, p2, p3, p4, p5, g2_mom=g2_mom, g3_mom=g3_mom)
        prop_vals.append(_bo5_prop_values(terms))

    keys = prop_vals[0].keys()
    lo = {k: min(pv[k] for pv in prop_vals) for k in keys}
    hi = {k: max(pv[k] for pv in prop_vals) for k in keys}

    return Bo5PropTheos(
        over_3_5_bid=_clamp01(lo["over_3_5"]),
        over_3_5_ask=_clamp01(hi["over_3_5"]),
        under_3_5_bid=_clamp01(1.0 - hi["over_3_5"]),
        under_3_5_ask=_clamp01(1.0 - lo["over_3_5"]),
        over_4_5_bid=_clamp01(lo["over_4_5"]),
        over_4_5_ask=_clamp01(hi["over_4_5"]),
        under_4_5_bid=_clamp01(1.0 - hi["over_4_5"]),
        under_4_5_ask=_clamp01(1.0 - lo["over_4_5"]),
        team_a_minus_2_5_bid=_clamp01(lo["a_minus_2_5"]),
        team_a_minus_2_5_ask=_clamp01(hi["a_minus_2_5"]),
        team_a_minus_1_5_bid=_clamp01(lo["a_minus_1_5"]),
        team_a_minus_1_5_ask=_clamp01(hi["a_minus_1_5"]),
        team_b_minus_2_5_bid=_clamp01(lo["b_minus_2_5"]),
        team_b_minus_2_5_ask=_clamp01(hi["b_minus_2_5"]),
        team_b_minus_1_5_bid=_clamp01(lo["b_minus_1_5"]),
        team_b_minus_1_5_ask=_clamp01(hi["b_minus_1_5"]),
    )


if __name__ == "__main__":
    # User's worked example.
    r = bo3_prop_theos(state=0, p1_a_bid=0.56, p1_a_ask=0.60, p2_a=0.52)
    print(f"User example  (g1=56@60, p2_a=0.52, pre-game):")
    print(f"  Over 2.5:    bid={r.over_2_5_bid:.4f}  ask={r.over_2_5_ask:.4f}  "
          f"mid={(r.over_2_5_bid + r.over_2_5_ask) / 2:.4f}")
    print(f"  Team A -1.5: bid={r.team_a_sweep_bid:.4f}  ask={r.team_a_sweep_ask:.4f}")
    print(f"  Team B -1.5: bid={r.team_b_sweep_bid:.4f}  ask={r.team_b_sweep_ask:.4f}")
    print()

    # State 1: same event after A wins g1
    r1 = bo3_prop_theos(state=1, p1_a_bid=0.0, p1_a_ask=0.0, p2_a=0.55)
    print(f"State 1 (A won g1, p2_a=0.55):")
    print(f"  Over 2.5:    bid={r1.over_2_5_bid:.4f}  ask={r1.over_2_5_ask:.4f}")
    print(f"  Team A -1.5: bid={r1.team_a_sweep_bid:.4f}  ask={r1.team_a_sweep_ask:.4f}")
    print(f"  Team B -1.5: bid={r1.team_b_sweep_bid}  (impossible after losing g1)")
    print()

    # BO5 sanity: 50/50/50/50/50 with no momentum
    print(f"BO5 sanity (50/50/50/50/50, no momentum) — UNTESTED:")
    r5 = bo5_prop_theos(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    print(f"  Over 3.5: bid={r5.over_3_5_bid:.4f}  ask={r5.over_3_5_ask:.4f}  (expected 0.75)")
    print(f"  Over 4.5: bid={r5.over_4_5_bid:.4f}  ask={r5.over_4_5_ask:.4f}  (expected 0.375)")
    print(f"  A -2.5:   bid={r5.team_a_minus_2_5_bid:.4f}  ask={r5.team_a_minus_2_5_ask:.4f}  (expected 0.125)")
    print(f"  A -1.5:   bid={r5.team_a_minus_1_5_bid:.4f}  ask={r5.team_a_minus_1_5_ask:.4f}  (expected 0.3125)")
