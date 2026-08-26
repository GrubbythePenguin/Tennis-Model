"""Round-score → map-probability baseline, and the implied sanity band.

Purpose (2026-07-20, KCGX post-mortem). At 14:41:15 on
KXVALORANTGAME-26JUL191100KCGX the map consensus moved 96 → 71 (25pp) in one
tick while top-of-book sat unchanged at 15/16. The arber read that as a +10.4c
edge and swept 5,000 lots at 17c; the value round-tripped to 98 within 2.3s.
The map score at the time was 11-4. A 25pp move is not merely unlikely at 11-4
— the game cannot produce it. This module quantifies "cannot".

The key property, and why this beats a fixed book-velocity threshold: the
maximum legitimate move is a function of GAME STATE and spans a ~250x range.
  4-11  → one round is worth ~0.9pp
  11-11 → one round is worth 25pp
Any single threshold is simultaneously too loose at 4-11 and too tight at
11-11. Fading a 25pp move is mandatory in the first case and catastrophic in
the second.

MODEL AND ITS LIMITS
The base P(win map | score) is an iid-rounds binomial race. Rounds are NOT iid
in VALORANT/CS2: economy chains them, so winning a pistol makes you heavy
favourite for the next 1-2 rounds. One round outcome therefore relocates the
EXPECTED score by roughly ±2 (a 4-round spread) — e.g. from 6-6, taking the
pistol trends to 8-6, losing it to 6-8. `swing_rounds=2` models that. This is
deliberately a loose baseline to anchor to, not a calibrated model: it exists
to reject the physically impossible, not to price maps. Bands are widened
further by SAFETY and by the unknown-team-strength sweep below.

Everything here is pure math with no feed dependency, so it is testable
offline against replays before any live score feed is wired in.
"""
from functools import lru_cache
from typing import Optional, Tuple

# Format: first to `target`, win by 2 from (target-1, target-1).
FORMATS = {
    "VAL": {"target": 13, "round_sec": 40.0},
    "CS2": {"target": 13, "round_sec": 45.0},   # MR12
}

# Per-round strength sweep. We rarely know which side is stronger, so bands are
# taken as the WIDEST over this range. Capped at +/-0.10 per round, which is a
# large edge: sweeping to 0.65 assumes the team down 4-11 wins 65% of rounds,
# which is not conservatism but incoherence — the score is itself evidence of
# strength. Combined with SAFETY below this was double-counting uncertainty and
# produced a 29.6pp band at 4-11, wider than the 25pp blip it exists to reject.
P_SWEEP = (0.40, 0.50, 0.60)

SAFETY = 3.0        # multiplier on the modelled max move
MIN_BAND_PP = 5.0   # floor; 12-12 OT is symmetric and models to a 0pp swing,
                    # which would otherwise block every move outright
MAX_BAND_PP = 100.0


@lru_cache(maxsize=None)
def p_win(a: int, b: int, p: float = 0.5, target: int = 13) -> float:
    """P(team A wins the map) from score a-b, iid rounds with per-round p."""
    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0
    if a >= target - 1 and b >= target - 1:
        # Deuce: win-by-2 steady state.
        return p * p / (p * p + (1 - p) * (1 - p))
    return p * p_win(a + 1, b, p, target) + (1 - p) * p_win(a, b + 1, p, target)


def max_legit_move(a: int, b: int, sport: str = "VAL",
                   swing_rounds: int = 2) -> float:
    """Largest |ΔP| (in percentage points) one round outcome can justify.

    Widest over P_SWEEP, since team strength is unknown.
    """
    t = FORMATS[sport]["target"]
    worst = 0.0
    for p in P_SWEEP:
        now = p_win(a, b, p, t)
        hi = p_win(min(a + swing_rounds, t), b, p, t)
        lo = p_win(a, min(b + swing_rounds, t), p, t)
        worst = max(worst, abs(hi - now), abs(now - lo))
    return 100.0 * worst


def band_pp(a: int, b: int, sport: str = "VAL", feed_age_sec: float = 0.0,
            safety: float = SAFETY, swing_rounds: int = 2) -> float:
    """Max plausible |Δ map probability| in pp, given score and feed staleness.

    Staleness ADVANCES THE SCORE rather than scaling the band. Multiplying the
    band by elapsed rounds is wrong: it treats a stale 4-11 as "4-11 but more
    so", when the truth is the score may now be 6-11, 4-13 (map over) or even
    11-11. We take the widest band over every reachable score, which is both
    correct and self-limiting — once enough time passes that 11-11 is
    reachable, the band goes wide and the gate correctly stops firing.

    Never returns below MIN_BAND_PP.
    """
    t = FORMATS[sport]["target"]
    extra = int(feed_age_sec // FORMATS[sport]["round_sec"])
    worst = 0.0
    for da in range(extra + 1):
        for db in range(extra + 1 - da):
            aa, bb = min(a + da, t), min(b + db, t)
            if (aa >= t and aa - bb >= 2) or (bb >= t and bb - aa >= 2):
                continue  # map would be over; no live price to bound
            worst = max(worst, max_legit_move(aa, bb, sport, swing_rounds))
    return min(MAX_BAND_PP, max(MIN_BAND_PP, worst * safety))


def implausible(a: int, b: int, delta_pp: float, sport: str = "VAL",
                feed_age_sec: float = 0.0, **kw) -> Tuple[bool, float]:
    """(should_fade, band_pp) for an observed map-probability move.

    Caller supplies |Δ| in pp. Fade only when the move EXCEEDS what the game
    state can generate — this rejects corrupt inputs, it does not express a
    view on price.
    """
    band = band_pp(a, b, sport, feed_age_sec, **kw)
    return abs(delta_pp) > band, band


def score_implied_range(a: int, b: int, sport: str = "VAL",
                        feed_age_sec: float = 0.0) -> Tuple[float, float]:
    """Plausible P(A wins map) range in pp, widest over strength and staleness.

    Use as an absolute sanity bound on a level (not a move). Returns (lo, hi).
    """
    t = FORMATS[sport]["target"]
    extra = int(feed_age_sec // FORMATS[sport]["round_sec"])
    lo, hi = 100.0, 0.0
    for p in P_SWEEP:
        for da in range(extra + 1):
            for db in range(extra + 1 - da):
                v = 100.0 * p_win(min(a + da, t), min(b + db, t), p, t)
                lo, hi = min(lo, v), max(hi, v)
    return lo, hi


def level_implausible(a: int, b: int, implied_pp: float, sport: str = "VAL",
                      feed_age_sec: float = 0.0,
                      pad_pp: float = 12.0) -> Tuple[bool, float, float]:
    """(should_fade, lo, hi) for an observed map-probability LEVEL.

    PREFER THIS over `implausible` on a delayed feed. Measured on the KCGX
    blip (score 4-11, blip level 31%, blip move 25pp):

        feed age   level ceiling  blocks 31%    move band  blocks 25pp
             0s            5.2%         yes       19.9pp          yes
            60s            7.9%         yes       29.1pp           no
           180s           25.7%         yes       77.4pp           no
           300s           69.2%          no      100.0pp           no

    The level bound survives ~3 minutes of staleness; the move band dies at
    ~40s. Reason: a stale score admits many reachable scores, and the SPREAD of
    their per-round sensitivities blows up much faster than the spread of their
    levels. So the move check is a fresh-feed-only refinement, and the level
    check is what a 1-minute-delayed feed should actually run.

    ⚠ `pad_pp` IS DOING MOST OF THE WORK AND IS NOT YET CALIBRATED.
    The iid model is badly overconfident at lopsided scores: at 4-11 it says
    1.1% while the actual Kalshi book was 15-16c. The market is right — real
    comebacks beat a binomial (half resets, tactical adjustments, economy).
    Gating on the raw model would fade legitimate trading, so the pad exists to
    absorb that miscalibration, and the whole gate's behaviour rides on it.

    pad=12 was chosen because it is the value that passes the real 15% book and
    rejects the 31% blip across 0-120s of staleness. That is fitted to ONE
    observation: real=15, blip=31 — a 16pp separation, which is a thin margin
    for a production gate. Before going live this must be calibrated against
    real (score, book) pairs: collect the score feed alongside cons_map for a
    slate, fit the empirical P(win | score) rather than assuming iid, and set
    the pad from the residual spread. Until then treat this as a shadow-mode
    baseline, not a live gate.
    """
    lo, hi = score_implied_range(a, b, sport, feed_age_sec)
    lo, hi = max(0.0, lo - pad_pp), min(100.0, hi + pad_pp)
    return (implied_pp < lo or implied_pp > hi), lo, hi


def describe(a: int, b: int, sport: str = "VAL", feed_age_sec: float = 0.0) -> str:
    lo, hi = score_implied_range(a, b, sport, feed_age_sec)
    return (f"{sport} {a}-{b} (feed age {feed_age_sec:.0f}s): "
            f"P(A) fair≈{100*p_win(a, b, 0.5, FORMATS[sport]['target']):.1f}% "
            f"plausible[{lo:.1f},{hi:.1f}] "
            f"max move {max_legit_move(a, b, sport):.1f}pp "
            f"band ±{band_pp(a, b, sport, feed_age_sec):.1f}pp")
