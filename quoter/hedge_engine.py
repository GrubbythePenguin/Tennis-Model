"""
Centralized Hedge Engine

Single source of truth for:
- BO3 state detection
- Cross-book map price computation
- Hedge profile dispatch (state → calculate_state_X_hedge)
- Synthetic cost assembly (with loaded cost)
- Series theo derivation
- Shared utilities (ticker parsing, probability loading)

All three bots (arber, live_series quoter, hedge bot) call into this module
so they always see the same math and the same prices.
"""
import csv
import json
import logging
import math
import os
import time
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)


# ── Persistent map-result cache ──────────────────────────────────────────
# BO5/BO3 state detection resolves a map's winner from its live book
# (bid>=99 / ask<=1) or Kalshi settlement status/result. But once a map
# settles, run.py drops it from the WS-probe (known-finalized) and its
# settled status/result stops appearing in full_market_state — the book
# reads empty (0/100). detect_bo{3,5}_state then can't tell the map was
# decided and falls back to state (0,0), pointing at the long-dead G1 map
# and suppressing theos forever (TESG2 + T1BLG 2026-07-03/04).
#
# This cache remembers each map ticker's result the moment it IS confidently
# resolved, and recovers it when the live signal later disappears. It is
# purely additive: written only on high-confidence resolution (the same
# signals detect already trusts), read only when the live data is absent —
# so with an empty cache, behaviour is byte-identical to before. File-backed
# so it survives run.py restarts mid-series. Keyed by the exact map ticker
# (date+teams+map#), which is globally unique, so entries never collide or
# go stale (a settled map's result is immutable).
_MAP_RESULT_CACHE_PATH = os.path.join(os.path.dirname(__file__),
                                      "_bo5_map_result_cache.json")
try:
    with open(_MAP_RESULT_CACHE_PATH) as _f:
        _map_result_cache: Dict[str, str] = json.load(_f)
except Exception:
    _map_result_cache = {}


# ── Provisional vs authoritative results (2026-07-20, IMPFLU) ────────────
# A map result is resolved from one of two very different kinds of evidence:
#
#   AUTHORITATIVE — Kalshi settlement status/result (determined/settled, or
#     closed/finalized with an unambiguous last price). A settled map's result
#     really is immutable, so this is cached permanently and never revoked.
#
#   PROVISIONAL — the live book alone (bid>=99 / ask<=1). We must keep acting
#     on this, because Kalshi/Poly settlement routinely lags the actual map end
#     by hours and we can't sit out G3 waiting for it. But a 99 bid is a PRICE,
#     not a result: KXCS2MAP-26JUL191900IMPFLU-3 hit 99 at 12-3 Fluxo, we
#     latched "FLU won M3" to disk, Fluxo were run down to 12-12, and the latch
#     never released — state stuck at (2,1), active_map advanced to a map 4
#     that did not exist, and the DP priced a phantom +15c edge that swept
#     3,750 lots at 52c.
#
# So provisional results are revocable by a live two-sided book away from both
# extremes — the signature of a map that is being played, not one that ended.
# An empty book (0/100) is NOT a contradiction: that is the settled-and-dropped
# case the cache exists to cover (TESG2/T1BLG), and it must still recover.
#
# Revocation is confirmed over REVERT_CONFIRM_SEC rather than acted on
# instantly, because a single corrupt WS frame can print a bogus mid-book on a
# genuinely finished map (VTCGL 2026-06-09, 73->27c). While a contradiction is
# pending we return None → the caller halts (cancel all resting, fire nothing)
# rather than trading on either the stale or the unconfirmed state.
# Contradiction is judged on the MIDPOINT, not on both sides independently.
# A two-sided test (bid>=15 AND ask<=85) misses a thin-but-live book like
# 20/90 — mid 55, plainly still being played, but the ask leg vetoes it. Since
# a missed revert is the expensive direction (IMPFLU: 3,750 lots), we key on
# mid and carve out only the genuinely empty book.
_REVERT_MID_MIN = 15.0      # mid must be at least this to contradict "over"
_REVERT_MID_MAX = 85.0      # ...and at most this
_REVERT_CONFIRM_SEC = 20.0  # sustained contradiction before we revoke
_revert_pending: Dict[str, float] = {}  # map_ticker -> first-contradiction ts


def _book_contradicts_decided(live_bid: Optional[float],
                              live_ask: Optional[float]) -> bool:
    """True when the live book looks like a map still in play.

    Requires a genuinely TWO-SIDED book at a mid price. Rationale:
      - Empty (0/100) is the settled-and-dropped case the cache exists to
        cover (TESG2/T1BLG) — no information, never a contradiction.
      - One-sided (e.g. a stale 40 bid with no ask) is what a FINISHED map
        awaiting settlement looks like once the makers pull. Treating that as
        "still in play" would falsely revoke a real result.
      - We quote these maps ourselves, so a live map reliably has both sides
        posted. Two-sided + mid is therefore strong evidence of live play,
        and it still catches the thin 20/90 case a per-leg test would miss.
    """
    if live_bid is None or live_ask is None:
        return False
    if live_bid <= 0.0 or live_ask >= 100.0:
        return False  # empty or one-sided — insufficient evidence to revoke
    mid = (live_bid + live_ask) / 2.0
    return _REVERT_MID_MIN <= mid <= _REVERT_MID_MAX


def _cache_map_result(map_ticker: str, result: str,
                      authoritative: bool = False) -> None:
    """Persist a resolved map result ("yes"/"no" for map_ticker's own team).

    Authoritative results are stored with a "!" suffix and can never be
    revoked or downgraded. No-op if unchanged, so the hot path only touches
    disk once per map settlement.
    """
    val = f"{result}!" if authoritative else result
    prev = _map_result_cache.get(map_ticker)
    if prev == val:
        return
    # Never downgrade an authoritative entry back to provisional.
    if prev and prev.endswith("!") and not authoritative:
        return
    _map_result_cache[map_ticker] = val
    try:
        with open(_MAP_RESULT_CACHE_PATH, "w") as _f:
            json.dump(_map_result_cache, _f)
    except Exception:
        pass  # cache is best-effort; never break detection on IO error


def _forget_map_result(map_ticker: str) -> None:
    """Drop a revoked provisional result from cache + disk."""
    if _map_result_cache.pop(map_ticker, None) is None:
        return
    try:
        with open(_MAP_RESULT_CACHE_PATH, "w") as _f:
            json.dump(_map_result_cache, _f)
    except Exception:
        pass


def _recover_map_wl(map_ticker: str, won: bool, lost: bool,
                    authoritative: bool = False,
                    live_bid: Optional[float] = None,
                    live_ask: Optional[float] = None) -> Tuple[bool, bool, bool]:
    """Given a map's live-derived (won, lost), persist it if resolved, or
    recover it from cache if the live signal was absent.

    Returns (won, lost, unstable). `unstable` is True when a cached provisional
    result is being contradicted by the live book but the contradiction has not
    yet persisted long enough to revoke — the caller must halt rather than
    trade on an ambiguous state.
    """
    c = _map_result_cache.get(map_ticker)
    cached_won = bool(c) and c.startswith("yes")
    is_authoritative = bool(c) and c.endswith("!")

    if won or lost:
        # An authoritative (settled) result outranks any live-derived signal.
        # Without this, a single corrupt frame pinning ask<=1 on a map we KNOW
        # settled "yes" would return lost=True and flip the series state — the
        # no-downgrade guard in _cache_map_result protects the cache but not
        # the return value.
        if is_authoritative and not authoritative and cached_won != won:
            log.warning(
                f"[MAP-RESULT OVERRIDE] {map_ticker}: live book says "
                f"{'won' if won else 'lost'} but settled result is '{c}' — "
                f"trusting settlement, ignoring book.")
            return cached_won, not cached_won, False
        _cache_map_result(map_ticker, "yes" if won else "no",
                          authoritative=authoritative)
        _revert_pending.pop(map_ticker, None)
        return won, lost, False

    if not c:
        return won, lost, False

    # Authoritative results are final — no live book can revoke them.
    if is_authoritative:
        return cached_won, not cached_won, False

    # Provisional: does the live book contradict "this map is over"?
    if not _book_contradicts_decided(live_bid, live_ask):
        _revert_pending.pop(map_ticker, None)
        return cached_won, not cached_won, False

    first_seen = _revert_pending.get(map_ticker)
    now = time.time()
    if first_seen is None:
        _revert_pending[map_ticker] = now
        log.warning(
            f"[MAP-RESULT CONTRADICTED] {map_ticker}: cached provisional "
            f"'{c}' but live book is {live_bid:.0f}/{live_ask:.0f} — map "
            f"appears still in play. Halting this series pending "
            f"{_REVERT_CONFIRM_SEC:.0f}s confirmation.")
        return won, lost, True

    if now - first_seen < _REVERT_CONFIRM_SEC:
        return won, lost, True

    # Sustained contradiction → the map is genuinely still being played.
    _forget_map_result(map_ticker)
    _revert_pending.pop(map_ticker, None)
    log.warning(
        f"[MAP-RESULT REVOKED] {map_ticker}: provisional '{c}' revoked after "
        f"{now - first_seen:.0f}s of live book {live_bid:.0f}/{live_ask:.0f}. "
        f"Map is still in play — series state reverts.")
    return False, False, False


# ── Math Primitives ──

def solve_set_probability(match_prob: float) -> float:
    """
    Given P(match win) in a BO3, solve for implied P(set win) assuming equal set probabilities.
    P(match) = p^2 * (3 - 2p) where p = P(set win).
    Uses Newton's method: f(p) = 2p^3 - 3p^2 + M = 0.
    """
    if match_prob <= 0.01:
        return 0.01
    if match_prob >= 0.99:
        return 0.99
    p = match_prob  # initial guess
    for _ in range(20):
        f = 2 * p**3 - 3 * p**2 + match_prob
        fp = 6 * p**2 - 6 * p
        if abs(fp) < 1e-12:
            break
        p = p - f / fp
        p = max(0.01, min(0.99, p))
    return p


def get_loaded_cost_cents(raw_cents: float) -> float:
    if raw_cents <= 0 or raw_cents >= 100:
        return raw_cents
    p_dollars = raw_cents / 100.0
    fee_cents = 7.0 * p_dollars * (1.0 - p_dollars)
    return raw_cents + fee_cents

def get_raw_cost_cents(raw_cents: float) -> float:
    """Identity function — no fee loading. Used for hedge calculations where
    the hedge is a theoretical reference, not an actual trade we pay fees on."""
    return float(raw_cents)

def get_actual_cost_per_share(ask_price_cents: int) -> float:
    return get_loaded_cost_cents(float(ask_price_cents))

def solve_game_3_probability(p1_A: float, p2_base: float, series_A: float, g2_momentum: float, g3_momentum: float) -> float:
    # Game 3 = average of map 1 and map 2 pre-game probabilities.
    # Previously derived from series price, but pre-game series pricing is often
    # inefficient (biased toward underdogs), creating phantom G3 values that are
    # 5-17c off from where G3 actually trades. Empirical analysis of Valorant
    # shows actual G3 prices average -0.3c vs avg(p1,p2) across all observed games.
    return max(0.01, min(0.99, (p1_A + p2_base) / 2.0))


G3_MOMENTUM_CAP = 0.80


def _g3_eff_up(base: float, g3_momentum: float, cap: float = G3_MOMENTUM_CAP) -> float:
    """Per-side capped g3 momentum bump.

    Returns the effective upward g3 bump for a team whose pregame G3 base prob
    is `base`. Rule:
      • base > cap → 0 (no momentum; the team is already above the ceiling)
      • else      → min(g3_momentum, cap − base) so base + g3_eff ≤ cap.

    This is applied per-side independently, so state 0 (G1 active, both
    branches modeled) can produce asymmetric bumps when one team is a heavy
    favorite. States 1/2 use the bump only for the side that has G2 momentum
    in the relevant branch.
    """
    if base > cap or g3_momentum <= 0:
        return 0.0
    return min(g3_momentum, cap - base)

def build_profile(shares_b: float, cash_reserve: float, active_leg_cost_per_share: float) -> dict:
    return {
        "shares_b": shares_b,
        "cash_reserve": cash_reserve,
        "synthetic_cost": cash_reserve + (shares_b * active_leg_cost_per_share)
    }


# ── BO3 Hedge Functions ──

def calculate_state_0_hedge(p1_A: float, p2_base: float, series_A: float, g2_momentum: float, g3_momentum: float, g1_b_ask_cents: int, m2_b_ask_cents: float = None) -> dict:
    p3_base_A = solve_game_3_probability(p1_A, p2_base, series_A, g2_momentum, g3_momentum)
    # Asymmetric per-side g3 caps (G3_MOMENTUM_CAP = 0.80). When base prob is
    # already above the cap, the bump is killed for that side.
    g3_A_up = _g3_eff_up(p3_base_A, g3_momentum)
    g3_B_up = _g3_eff_up(1.0 - p3_base_A, g3_momentum)
    # M2 hedge cost: if a live M2 ask is supplied (e.g., M1 essentially decided
    # but M2 still trading), use it in both branches. Per design, ignore
    # g2_momentum in this case — live market price already incorporates
    # whatever conditional info exists.
    if m2_b_ask_cents is not None:
        m2_b_cost_a = float(m2_b_ask_cents)
        m2_b_cost_b = float(m2_b_ask_cents)
    else:
        p2_B_if_A = 1.0 - (p2_base + g2_momentum)
        p2_B_if_B = 1.0 - (p2_base - g2_momentum)
        m2_b_cost_a = p2_B_if_A * 100.0
        m2_b_cost_b = p2_B_if_B * 100.0
    p3_B_if_B_won_G2 = (1.0 - p3_base_A) + g3_B_up
    cost_to_fund_g3_B_branch_a = get_raw_cost_cents(p3_B_if_B_won_G2 * 100.0)
    shares_b_g2_branch_a = cost_to_fund_g3_B_branch_a / 100.0
    v_a = shares_b_g2_branch_a * get_raw_cost_cents(m2_b_cost_a)
    p3_B_if_A_won_G2 = (1.0 - p3_base_A) - g3_A_up
    cost_to_fund_g3_B_branch_b = get_raw_cost_cents(p3_B_if_A_won_G2 * 100.0)
    shares_b_g2_branch_b = (100.0 - cost_to_fund_g3_B_branch_b) / 100.0
    v_b = cost_to_fund_g3_B_branch_b + (shares_b_g2_branch_b * get_raw_cost_cents(m2_b_cost_b))
    shares_b_g1 = (v_b - v_a) / 100.0
    return build_profile(shares_b_g1, v_a, get_raw_cost_cents(g1_b_ask_cents))

def calculate_state_1_hedge(p1_A: float, p2_base: float, series_A: float, g2_momentum: float, g3_momentum: float, g2_b_ask_cents: int) -> dict:
    p3_base_A = solve_game_3_probability(p1_A, p2_base, series_A, g2_momentum, g3_momentum)
    # Only the "B wins G2 → G3 played, B has momentum" branch is reachable, so
    # cap on B's side.
    g3_B_up = _g3_eff_up(1.0 - p3_base_A, g3_momentum)
    p3_B_if_B_won_G2 = (1.0 - p3_base_A) + g3_B_up
    cost_to_fund_g3_B = get_raw_cost_cents(p3_B_if_B_won_G2 * 100.0)
    shares_b_g2 = cost_to_fund_g3_B / 100.0
    return build_profile(shares_b_g2, 0.0, get_raw_cost_cents(g2_b_ask_cents))

def calculate_state_2_hedge(p1_A: float, p2_base: float, series_A: float, g2_momentum: float, g3_momentum: float, g2_b_ask_cents: int) -> dict:
    p3_base_A = solve_game_3_probability(p1_A, p2_base, series_A, g2_momentum, g3_momentum)
    # Only the "A wins G2 → G3 played, A has momentum" branch is reachable, so
    # cap on A's side.
    g3_A_up = _g3_eff_up(p3_base_A, g3_momentum)
    p3_B_if_A_won_G2 = (1.0 - p3_base_A) - g3_A_up
    cost_to_fund_g3_B = get_raw_cost_cents(p3_B_if_A_won_G2 * 100.0)
    shares_b_g2 = (100.0 - cost_to_fund_g3_B) / 100.0
    return build_profile(shares_b_g2, cost_to_fund_g3_B, get_raw_cost_cents(g2_b_ask_cents))

def calculate_state_3_hedge(g3_b_ask_cents: int) -> dict:
    return build_profile(1.0, 0.0, get_raw_cost_cents(g3_b_ask_cents))


# ── BO5 Hedge Functions ──
# For BO5, we need p1-p5 probabilities. p4 and p5 = avg(p1, p2, p3).
# States track (wins_A, wins_B). Team A needs 3 wins to take the series.
#
# The hedge logic follows the same principle as BO3:
# - At each state, compute the fractional hedge needed for the NEXT game
# - The hedge accounts for all possible future paths to determine
#   how many shares of the opponent's next-game win we need to hold

def _bo5_game_probs(p1: float, p2: float, p3: float, momentum: float):
    """Return (p1, p2, p3, p4, p5) for team A, applying momentum to later games."""
    p4 = max(0.01, min(0.99, (p1 + p2 + p3) / 3.0))
    p5 = p4  # Same as p4
    return p1, p2, p3, p4, p5

def _bo5_series_prob(p1: float, p2: float, p3: float) -> float:
    """Compute BO5 series win probability for team A from game probabilities.
    Team A wins if they get 3 wins before team B gets 3 wins."""
    _, _, _, p4, p5 = _bo5_game_probs(p1, p2, p3, 0)
    ps = [p1, p2, p3, p4, p5]

    # Dynamic programming: prob of winning from state (wins_a, wins_b)
    # starting from the next game index = wins_a + wins_b
    memo = {}
    def win_prob(wa, wb):
        if wa == 3: return 1.0
        if wb == 3: return 0.0
        if (wa, wb) in memo: return memo[(wa, wb)]
        gi = wa + wb  # game index (0-based)
        p = ps[gi] if gi < len(ps) else ps[-1]
        result = p * win_prob(wa + 1, wb) + (1 - p) * win_prob(wa, wb + 1)
        memo[(wa, wb)] = result
        return result

    return win_prob(0, 0)

def calculate_bo5_hedge(wins_a: int, wins_b: int,
                        p1: float, p2: float, p3: float,
                        momentum: float,
                        next_game_b_ask_cents: int,
                        p4: Optional[float] = None,
                        p5: Optional[float] = None) -> dict:
    """Recursive self-financing BO5 hedge — matches dry_run_bo5.bo5_hedge_cost.

    All five per-game probabilities are anchored from the caller (CSV loaded
    at startup). No live-market override on ps. The active map's live ask
    enters only via next_game_b_ask_cents (the actual hedge trade price), not
    as a per-game probability. This prevents the active-map cross-book from
    contaminating future-map probability slots — the THVIT 2026-05-17 bug
    where forward-propagating live G2 into p3/p4/p5 inflated series probs.

    `p4` / `p5`: explicit per-map probabilities for games 4 and 5. When the
    caller has loaded these from Polymarket Game 4/5 Winner markets, pass them
    here so the DP uses market-priced future-game probs instead of the
    `(p1+p2+p3)/3` stand-in. When omitted (None), they fall back to the avg
    stand-in (preserves prior behavior for callers that haven't been updated).
    """
    # p4 / p5 from explicit caller values when available.
    # Fallback policy:
    #   p4 absent  → use avg(p1,p2,p3) (no other info to anchor on)
    #   p5 absent  → use p4_eff (carry latest available signal forward).
    avg_p = (p1 + p2 + p3) / 3.0
    p4_eff = p4 if p4 is not None else avg_p
    p5_eff = p5 if p5 is not None else p4_eff
    p4_eff = max(0.01, min(0.99, p4_eff))
    p5_eff = max(0.01, min(0.99, p5_eff))

    ps = [p1, p2, p3, p4_eff, p5_eff]

    # P_A(wa, wb) — team A's series win prob from state (wa, wb).
    pa_memo: Dict[Tuple[int, int], float] = {}
    def p_a(wa: int, wb: int) -> float:
        if wa == 3: return 1.0
        if wb == 3: return 0.0
        if (wa, wb) in pa_memo:
            return pa_memo[(wa, wb)]
        gi = wa + wb
        p = ps[gi] if gi < len(ps) else ps[-1]
        r = p * p_a(wa + 1, wb) + (1 - p) * p_a(wa, wb + 1)
        pa_memo[(wa, wb)] = r
        return r

    # HC(wa, wb, live_ask) — total hedge cost from state (wa, wb) until series
    # resolution, given that THIS step's hedge transaction pays `live_ask`
    # cents per share. Future re-hedges (HC_win, HC_lose recurse) use the
    # model-implied ask at their own state, so the recursion is truly self-
    # financing: under fair pricing assumptions the total cost equals
    # 100 - loaded(P_A * 100), i.e. zero edge.
    hc_memo: Dict[Tuple[int, int, float], float] = {}
    def hc(wa: int, wb: int, live_ask: float) -> float:
        if wa == 3 or wb == 3:
            return 0.0
        key = (wa, wb, live_ask)
        if key in hc_memo:
            return hc_memo[key]
        gi = wa + wb
        # Next state's model-implied ask (for the team-B side, since hedge
        # is shares of opp winning the next map). Caps at ps[-1] beyond G5.
        next_gi = gi + 1
        if next_gi < len(ps):
            future_ask = (1.0 - ps[next_gi]) * 100.0
        else:
            future_ask = (1.0 - ps[-1]) * 100.0
        V_win = p_a(wa + 1, wb) * 100.0
        V_lose = p_a(wa, wb + 1) * 100.0
        H = (V_win - V_lose) / 100.0
        HC_win = hc(wa + 1, wb, future_ask)
        HC_lose = hc(wa, wb + 1, future_ask)
        # Cash reserve must cover whichever branch demands more: future
        # hedge funding after a win (HC_win) vs after a loss (HC_lose less
        # the H*100 we collect when B wins this map).
        C = max(HC_win, HC_lose - H * 100.0)
        cost = H * live_ask + C
        hc_memo[key] = cost
        return cost

    synthetic_cost = hc(wins_a, wins_b, float(next_game_b_ask_cents))

    # Shares + cash split for compatibility with downstream callers that read
    # those fields (build_profile API parity). Edge calc only needs
    # synthetic_cost.
    V_win = p_a(wins_a + 1, wins_b) * 100.0
    V_lose = p_a(wins_a, wins_b + 1) * 100.0
    shares_b = (V_win - V_lose) / 100.0
    cash_reserve = synthetic_cost - shares_b * float(next_game_b_ask_cents)

    return {
        "shares_b": shares_b,
        "cash_reserve": cash_reserve,
        "synthetic_cost": synthetic_cost,
    }


# ── Shared Utilities ──

def parse_series_ticker(ticker: str) -> Optional[Tuple[str, str, str]]:
    """Parse a series ticker into (series_base, team_suffix, map_base).
    Returns None if ticker is not a valid GAME/MATCH ticker."""
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    series_base = parts[0] + "-" + parts[1]
    # 2026-07-28: test/rewrite the PREFIX only — a TEAM NAME can contain the
    # token ("2GAME Esports"), which made series_base.replace() emit ...SR2MAP
    # (404). See memory/project_map_base_game_substring_collision.md
    _mpfx = parts[0]
    if "GAME" not in _mpfx and "MATCH" not in _mpfx:
        return None
    team_suffix = parts[2]
    if "MATCH" in _mpfx:
        map_base = _mpfx.replace("MATCH", "SETWINNER") + "-" + parts[1]
    else:
        map_base = _mpfx.replace("GAME", "MAP") + "-" + parts[1]
    return series_base, team_suffix, map_base


BO5_DISABLED_FLAG = "disable_bo5.flag"


def bo5_disabled() -> bool:
    """Runtime kill-switch for BO5 trading. Returns True when disable_bo5.flag
    exists in the working directory. Mirrors the populate-time gate at
    populate_configs.py:1531 so existing BO5 rows in template_quoter_config.csv
    cannot trade either. Delete the flag to re-enable BO5."""
    return os.path.exists(BO5_DISABLED_FLAG)


def load_probabilities(prob_file: str = "esports_probabilities.csv") -> Dict[str, Dict[str, Any]]:
    """Load pre-game baseline probabilities from CSV.

    CSV column order:
      0: ticker
      1: p1
      2: p2
      3: series
      4: g2_momentum (optional)
      5: g3_momentum (optional)
      6: p3 (optional — present indicates BO5)
      7: spread_penalty (optional, ignored here)
      8: "VERIFIED" flag (optional)
      9: p4 (optional — BO5, present when Poly exposes Game 4 Winner)
     10: p5 (optional — BO5, present when Poly exposes Game 5 Winner)

    p4 / p5 fall back to (p1+p2+p3)/3 when absent (matching the legacy
    one-scalar BO5 approximation). When present they come from Polymarket
    Game 4/5 Winner markets via populate_configs.best_prob_for_leg.

    Hallucination rejection: entries where p1, p2, and series are all ≈0.5
    are rejected unless explicitly flagged VERIFIED.

    Returns {ticker: {p1, p2, series, g2_momentum, g3_momentum, p3, p4, p5, bo5, verified}}."""
    cache = {}
    if not os.path.exists(prob_file):
        return cache
    try:
        with open(prob_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    ticker = row[0].strip()
                    p1 = float(row[1])
                    p2 = float(row[2])
                    series = float(row[3])
                    g2_momentum = float(row[4]) if len(row) > 4 and row[4].strip() else 0.0
                    g3_momentum = float(row[5]) if len(row) > 5 and row[5].strip() else 0.0

                    # p3 (col 6): explicit value indicates BO5; otherwise default to avg(p1,p2)
                    bo5 = len(row) > 6 and row[6].strip() != ""
                    p3 = float(row[6]) if bo5 else (p1 + p2) / 2.0

                    verified = len(row) >= 9 and row[8].strip().upper() == "VERIFIED"

                    # p4 (col 9) and p5 (col 10): explicit values from Poly when present;
                    # non-numeric (e.g., '-' sentinel meaning "captured but Poly didn't
                    # expose G4/G5") or missing → fall back to (p1+p2+p3)/3 → p4 chain.
                    p4_avg = (p1 + p2 + p3) / 3.0
                    def _parse_optional_p(idx, fallback):
                        if len(row) > idx and row[idx].strip():
                            try:
                                return float(row[idx])
                            except ValueError:
                                return fallback
                        return fallback
                    p4 = _parse_optional_p(9, p4_avg)
                    p5 = _parse_optional_p(10, p4)  # fall back to p4 (g5=g4 carry-forward)

                    # Reject hallucinated 50/50 entries unless VERIFIED
                    if (not verified
                        and abs(p1 - 0.5) < 0.01
                        and abs(p2 - 0.5) < 0.01
                        and abs(series - 0.5) < 0.01):
                        log.warning(f"[PROBS] Rejecting hallucinated 50/50 entry: {ticker} "
                                    f"(p1={p1:.3f} p2={p2:.3f} series={series:.3f})")
                        continue

                    cache[ticker] = {
                        "p1": p1,
                        "p2": p2,
                        "series": series,
                        "g2_momentum": g2_momentum,
                        "g3_momentum": g3_momentum,
                        "p3": p3,
                        "p4": p4,
                        "p5": p5,
                        "bo5": bo5,
                        "verified": verified,
                    }
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        log.error(f"Failed loading probabilities: {e}")
    return cache


# ── Forfeit Detection ──

def evaluate_forfeit(
    series_base: str,
    map1_ticker: str,
    map2_ticker: str,
    full_top_bids: Dict[str, float],
    full_top_offers: Dict[str, float],
    series_ask: float = 50.0,
    series_no_ask: float = 50.0,
    data_source: str = "kalshi",
    disable_forfeit_check: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Centralized forfeit detection.

    Single source of truth — both arber_bot and detect_bo3_state consult this
    so we never have one component refusing to trade while another doesn't.

    Two forfeit signals:
      1. General: maps 1 and 2 both have real books pinned ~50/50 while series
         is extreme (>=95 on either side). Cross-source sanity check.
      2. CS2 Poly: a single map pinned 49-51 — Poly voids forfeited maps to 50/50.
         Only meaningful when prices are sourced from Poly.

    `data_source` controls when the CS2 check runs:
      - "poly":              run always (Tier 1 CS, post-data_source-refactor)
      - "kalshi_na_poly_eu": run only when not is_na_hours (current esports default)
      - "kalshi":            never run (prices are pure Kalshi; 50/50 ≠ forfeit)

    Default series_ask/series_no_ask = 50 → general check won't fire unless
    caller provides actual series prices (allows callers without series price
    context, like detect_bo3_state, to still get CS2 protection).

    Returns (forfeited: bool, reason: str | None).
    """
    if disable_forfeit_check:
        return False, None

    # Signal 1: General forfeit — maps both pinned ~50/50, series extreme
    m1_bid = full_top_bids.get(map1_ticker, 0)
    m1_ask = full_top_offers.get(map1_ticker, 100)
    m1_no_ask = 100.0 - m1_bid
    m2_bid = full_top_bids.get(map2_ticker, 0)
    m2_ask = full_top_offers.get(map2_ticker, 100)
    m2_no_ask = 100.0 - m2_bid

    m1_has_book = m1_bid > 0 and m1_no_ask < 100
    m2_has_book = m2_bid > 0 and m2_no_ask < 100
    if m1_has_book and m2_has_book:
        m1_mid = (m1_bid + (100.0 - m1_no_ask)) / 2.0
        m2_mid = (m2_bid + (100.0 - m2_no_ask)) / 2.0
        maps_near_50 = abs(m1_mid - 50) < 1 and abs(m2_mid - 50) < 1
        series_extreme = series_ask >= 95 or series_no_ask >= 95
        if maps_near_50 and series_extreme:
            reason = (f"general: m1_mid={m1_mid:.0f} m2_mid={m2_mid:.0f} "
                      f"series_ask={series_ask:.0f} series_no_ask={series_no_ask:.0f}")
            log.warning(f"[FORFEIT ABORT] {series_base}: maps real books at ~50/50 "
                        f"but series extreme — match likely forfeited.")
            return True, reason

    # Signal 2: CS2 Poly forfeit — single map pinned 49-51 (Poly forfeit voids to 50)
    if "KXCS2" in series_base:
        from esports_config import is_na_hours
        if data_source == "poly":
            is_poly_sourced = True
        elif data_source == "kalshi_na_poly_eu":
            is_poly_sourced = not is_na_hours(series_base)
        else:  # "kalshi" or unknown
            is_poly_sourced = False
        if is_poly_sourced:
            for map_t in [map1_ticker, map2_ticker]:
                m_bid = full_top_bids.get(map_t, 0)
                m_ask = full_top_offers.get(map_t, 100)
                if 49 <= m_bid <= 51 and 49 <= m_ask <= 51:
                    reason = f"cs2_poly: {map_t} bid={m_bid}c ask={m_ask}c"
                    log.warning(f"[POLY FORFEIT] {map_t} bid={m_bid}c ask={m_ask}c — "
                                f"both near 50/50, map likely forfeited on Poly.")
                    return True, reason

    return False, None


# ── 1. State Detection ──

def detect_bo3_state(
    map_base: str,
    team_suffix: str,
    full_market_state: Optional[Dict[str, Any]],
    full_top_bids: Dict[str, float],
    full_top_offers: Dict[str, float],
    series_base: Optional[str] = None,
    data_source: str = "kalshi",
    disable_forfeit_check: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Detect BO3 game state from map prices and settlement data.

    Returns None if series is decided / forfeited / ambiguous.

    Otherwise returns a dict:
      {
        "state": int,                    # 0=g1 active, 1=won g1 (g2 active),
                                         # 2=lost g1 (g2 active), 3=1-1 (game 3)
        "active_map": str | None,        # active map ticker (None for state 3)
        "active_map_opp": str | None,    # opponent's active map ticker (None for state 3)
        "m1_won": bool, "m1_lost": bool,
        "m2_won": bool, "m2_lost": bool,
      }
    """
    # Find opponent suffix
    opp_suffix = None
    for k in (full_top_bids or {}):
        if k.startswith(f"{map_base}-1-") and not k.endswith(f"-{team_suffix}"):
            opp_suffix = k.split("-")[-1]
            break
    # Also check full_market_state keys
    if not opp_suffix and full_market_state:
        for k in full_market_state:
            if k.startswith(f"{map_base}-1-") and not k.endswith(f"-{team_suffix}"):
                opp_suffix = k.split("-")[-1]
                break
    if not opp_suffix:
        return None

    map1_ticker = f"{map_base}-1-{team_suffix}"
    map1_opp = f"{map_base}-1-{opp_suffix}"
    map2_ticker = f"{map_base}-2-{team_suffix}"
    map2_opp = f"{map_base}-2-{opp_suffix}"

    # Forfeit check via centralized oracle. Without series prices the general
    # check defaults to non-firing; only the CS2 Poly check runs here.
    # data_source controls when CS2 forfeit fires (poly → 24/7, kalshi_na_poly_eu → EU only,
    # kalshi → never).
    if series_base:
        forfeited, _ = evaluate_forfeit(
            series_base, map1_ticker, map2_ticker,
            full_top_bids, full_top_offers,
            data_source=data_source,
            disable_forfeit_check=disable_forfeit_check,
        )
        if forfeited:
            return None

    # ── Map 1 resolution ──
    m1_bid = full_top_bids.get(map1_ticker, 0.0)
    m1_ask = full_top_offers.get(map1_ticker, 100.0)

    m1_won = (m1_bid >= 99 and m1_ask == 100)
    m1_lost = (m1_ask <= 1 and m1_bid == 0)

    if full_market_state and map1_ticker in full_market_state:
        m1_data = full_market_state[map1_ticker]
        m1_status = m1_data.get("status", "")
        if m1_status in ["determined", "settled"]:
            m1_result = m1_data.get("result", "")
            if m1_result == "yes":
                m1_won = True
            elif m1_result == "no":
                m1_lost = True
        elif m1_status in ["closed", "finalized"]:
            last_price = float(m1_data.get("last_price_dollars", "0") or "0") * 100.0
            if last_price >= 85:
                m1_won = True
            elif last_price > 0 and last_price <= 15:
                m1_lost = True
            else:
                # Cross-validate against opponent
                opp_data = (full_market_state or {}).get(map1_opp, {})
                opp_last = float(opp_data.get("last_price_dollars", "0") or "0") * 100.0
                if opp_last >= 85:
                    m1_lost = True
                elif opp_last > 0 and opp_last <= 15:
                    m1_won = True

        if m1_status in ["closed", "finalized"] and not m1_won and not m1_lost:
            return None  # Ambiguous

    # ── Map 2 resolution (moved up: M2's settlement state is needed to
    # disambiguate the M1-stuck dead-zone case, where M1's book stays at a
    # mid-range price like 59/0 forever even though the actual game has
    # ended and M2 has come and gone). ──
    m2_bid = full_top_bids.get(map2_ticker, 0.0)
    m2_ask = full_top_offers.get(map2_ticker, 100.0)

    m2_won = (m2_bid >= 99 and m2_ask == 100)
    m2_lost = (m2_ask <= 1 and m2_bid == 0)

    # Empty orderbook cross-validation
    if not m2_won and not m2_lost and m2_bid == 0 and m2_ask == 100:
        opp_m2_bid = full_top_bids.get(map2_opp, 0.0)
        opp_m2_ask = full_top_offers.get(map2_opp, 100.0)
        if opp_m2_bid >= 99 and opp_m2_ask == 100:
            m2_lost = True
        elif opp_m2_ask <= 1:
            m2_won = True

    if full_market_state and map2_ticker in full_market_state:
        m2_data = full_market_state[map2_ticker]
        m2_status = m2_data.get("status", "")
        if m2_status in ["determined", "settled"]:
            m2_result = m2_data.get("result", "")
            if m2_result == "yes":
                m2_won = True
            elif m2_result == "no":
                m2_lost = True
        elif m2_status in ["closed", "finalized"]:
            last_price = float(m2_data.get("last_price_dollars", "0") or "0") * 100.0
            if last_price >= 85:
                m2_won = True
            elif last_price > 0 and last_price <= 15:
                m2_lost = True
            else:
                opp_data = (full_market_state or {}).get(map2_opp, {})
                opp_last = float(opp_data.get("last_price_dollars", "0") or "0") * 100.0
                if opp_last >= 85:
                    m2_lost = True
                elif opp_last > 0 and opp_last <= 15:
                    m2_won = True

        if m2_status in ["closed", "finalized"] and not m2_won and not m2_lost:
            return None

    if not m1_won and not m1_lost:
        # M1 is not clearly decided by strict thresholds or Kalshi status.
        # Three sub-cases, in order:
        #
        # (a) M2 has strictly settled. The series has clearly moved past M2,
        #     so the actual state is either 1-1 (game 3 active) or 2-0
        #     (series decided). In both cases we MUST stop publishing theos
        #     anchored to the dead M1 book. Return state 3 to halt
        #     downstream consumers (arber/quoter both skip state 3). The
        #     2-0 case is also covered by the series-decided guard upstream.
        # (b) LOOSE 5c "essentially-decided" path: if either M1 side's bid
        #     sits in [0, 5), the market believes that side has lost. Empty
        #     book (bid == 0) is treated the same as a stuck 1c bid — both
        #     mean nobody is willing to buy that side. The pre-game case
        #     (BOTH sides empty/loose) is caught and halted explicitly.
        # (c) Otherwise M1 truly looks active: return state 0.
        if m2_won or m2_lost:
            return {
                "state": 3,
                "active_map": None,
                "active_map_opp": None,
                "m1_won": False, "m1_lost": False,
                "m2_won": m2_won, "m2_lost": m2_lost,
            }
        m1_opp_bid = full_top_bids.get(map1_opp, 0.0)
        m2_bid_opp = full_top_bids.get(map2_opp, 0.0)
        m1_us_loose  = (0 <= m1_bid < 5)
        m1_opp_loose = (0 <= m1_opp_bid < 5)
        m2_us_loose  = (0 <= m2_bid < 5)
        m2_opp_loose = (0 <= m2_bid_opp < 5)
        # M1 BOTH sides loose means the M1 book is informationally void
        # (pre-game with no orders, or post-settlement with bids pulled on
        # both teams). Halt rather than infer state from a dead book.
        if m1_us_loose and m1_opp_loose:
            return None
        m1_loose = m1_us_loose or m1_opp_loose
        m2_loose = m2_us_loose or m2_opp_loose
        if m1_loose and m2_loose:
            return None  # both maps in endgame fog — refuse to trade
        return {
            "state": 0,
            "active_map": map1_ticker,
            "active_map_opp": map1_opp,
            "m1_won": False, "m1_lost": False,
            "m2_won": False, "m2_lost": False,
            "use_live_m2_state0": m1_loose,
        }

    # M1 known: state 1, 2, or (with M2 also known) state 3 / 2-0.
    if m2_won or m2_lost:
        if (m1_won and m2_lost) or (m1_lost and m2_won):
            # 1-1 → game 3. No separate map 3 market in BO3 — series IS game 3.
            return {
                "state": 3,
                "active_map": None,
                "active_map_opp": None,
                "m1_won": m1_won, "m1_lost": m1_lost,
                "m2_won": m2_won, "m2_lost": m2_lost,
            }
        # 2-0 → series decided
        return None

    state = 1 if m1_won else 2
    # State 1/2 halt: M2 is the active hedge map. If M2 has a bid in [0, 5)
    # on either side, the market believes that side has lost but Kalshi
    # hasn't settled M2 — live M2 prices unreliable, refuse to trade.
    # Empty book (bid == 0) counts the same as a stuck 1c bid.
    # EXCEPTION: if the drop below 5 happened in the last 60s (real teamfight
    # movement), allow trading. is_fresh on an empty book will return False,
    # so the empty-book case naturally halts.
    m2_opp_bid = full_top_bids.get(map2_opp, 0.0)
    if (0 <= m2_bid < 5) or (0 <= m2_opp_bid < 5):
        try:
            import staleness_tracker as _st
            us_loose_fresh  = (0 <= m2_bid < 5) and _st.is_fresh(map2_ticker)
            opp_loose_fresh = (0 <= m2_opp_bid < 5) and _st.is_fresh(map2_opp)
            if not (us_loose_fresh or opp_loose_fresh):
                return None
        except Exception:
            return None
    return {
        "state": state,
        "active_map": map2_ticker,
        "active_map_opp": map2_opp,
        "m1_won": m1_won, "m1_lost": m1_lost,
        "m2_won": False, "m2_lost": False,
    }


def detect_bo5_state(
    map_base: str,
    team_suffix: str,
    full_market_state: Optional[Dict[str, Any]],
    full_top_bids: Dict[str, float],
    full_top_offers: Dict[str, float],
    series_base: Optional[str] = None,
    data_source: str = "kalshi",
    disable_forfeit_check: bool = False,
) -> Optional[Dict[str, Any]]:
    """Detect BO5 series state from map prices and settlement data.

    Mirrors the logic that previously lived inline in arber_bot.py:491-641.
    Both arber and live_series_model now call this so the two bots can never
    diverge on BO5 state detection.

    Returns None when:
      - forfeit detected (M1/M2 forfeit oracle)
      - series decided (wins_a >= 3 or wins_b >= 3)
      - G5 reached (wins_a + wins_b >= 4) — G5 trading is intentionally disabled
        because the live G5 cross-book has no anchor beyond itself
      - M1 or M2 has status closed/finalized with ambiguous last_price

    Otherwise returns:
      {
        "state": tuple (wins_a, wins_b),
        "wins_a": int, "wins_b": int,
        "active_map": str (map ticker that's currently in play),
        "active_map_opp": str | None (opponent's same-map ticker for cross-book),
        "m1_won": bool, "m1_lost": bool,
        "m2_won": bool, "m2_lost": bool,
        "m3_won": bool, "m3_lost": bool,
        "m4_won": bool, "m4_lost": bool,
      }
    """
    # Find opponent suffix
    opp_suffix = None
    for k in (full_top_bids or {}):
        if k.startswith(f"{map_base}-1-") and not k.endswith(f"-{team_suffix}"):
            opp_suffix = k.split("-")[-1]
            break
    if not opp_suffix and full_market_state:
        for k in full_market_state:
            if k.startswith(f"{map_base}-1-") and not k.endswith(f"-{team_suffix}"):
                opp_suffix = k.split("-")[-1]
                break
    if not opp_suffix:
        return None

    map1_ticker = f"{map_base}-1-{team_suffix}"
    map1_opp    = f"{map_base}-1-{opp_suffix}"
    map2_ticker = f"{map_base}-2-{team_suffix}"
    map2_opp    = f"{map_base}-2-{opp_suffix}"

    # Forfeit check (mirrors detect_bo3_state)
    if series_base:
        forfeited, _ = evaluate_forfeit(
            series_base, map1_ticker, map2_ticker,
            full_top_bids, full_top_offers,
            data_source=data_source,
            disable_forfeit_check=disable_forfeit_check,
        )
        if forfeited:
            return None

    # ── M1 detection (inlined to match arber_bot's prior behavior exactly) ──
    m1_bid = full_top_bids.get(map1_ticker, 0.0)
    m1_ask = full_top_offers.get(map1_ticker, 100.0)
    m1_won  = (m1_bid >= 99 and m1_ask == 100)
    m1_lost = (m1_ask <= 1 and m1_bid == 0)
    m1_auth = False  # True only when settlement status (not price) resolved it

    if full_market_state and map1_ticker in full_market_state:
        m1_data = full_market_state[map1_ticker]
        m1_status = m1_data.get("status", "")
        if m1_status in ["determined", "settled"]:
            r = m1_data.get("result", "")
            if r == "yes": m1_won = True; m1_auth = True
            elif r == "no": m1_lost = True; m1_auth = True
        elif m1_status in ["closed", "finalized"]:
            lp = float(m1_data.get("last_price_dollars", "0") or "0") * 100.0
            if lp >= 85: m1_won = True; m1_auth = True
            elif lp > 0 and lp <= 15: m1_lost = True; m1_auth = True
            else:
                # Cross-validate via opponent's same-map
                opp_data = (full_market_state or {}).get(map1_opp, {})
                opp_last = float(opp_data.get("last_price_dollars", "0") or "0") * 100.0
                if opp_last >= 85: m1_lost = True; m1_auth = True
                elif opp_last > 0 and opp_last <= 15: m1_won = True; m1_auth = True
        if m1_status in ["closed", "finalized"] and not m1_won and not m1_lost:
            return None  # Ambiguous M1 — refuse to detect state

    # Persist / recover M1 result across reactivations (settled maps drop out
    # of full_market_state → empty book → false (0,0) without this).
    m1_won, m1_lost, m1_unstable = _recover_map_wl(
        map1_ticker, m1_won, m1_lost, authoritative=m1_auth,
        live_bid=m1_bid, live_ask=m1_ask)
    if m1_unstable:
        return None  # cached M1 contradicted by live book — halt pending confirm

    # M1 not resolved → G1 active, state (0, 0)
    if not m1_won and not m1_lost:
        return {
            "state": (0, 0), "wins_a": 0, "wins_b": 0,
            "active_map": map1_ticker, "active_map_opp": map1_opp,
            "m1_won": False, "m1_lost": False,
            "m2_won": False, "m2_lost": False,
            "m3_won": False, "m3_lost": False,
            "m4_won": False, "m4_lost": False,
        }

    # ── M2 detection ──
    m2_bid = full_top_bids.get(map2_ticker, 0.0)
    m2_ask = full_top_offers.get(map2_ticker, 100.0)
    m2_won  = (m2_bid >= 99 and m2_ask == 100)
    m2_lost = (m2_ask <= 1 and m2_bid == 0)
    m2_auth = False  # True only when settlement status (not price) resolved it

    # Empty-orderbook cross-validation against M2 opp (arber's existing M2 dead-book check)
    if not m2_won and not m2_lost and m2_bid == 0 and m2_ask == 100:
        opp_m2_bid = full_top_bids.get(map2_opp, 0.0)
        opp_m2_ask = full_top_offers.get(map2_opp, 100.0)
        if opp_m2_bid >= 99 and opp_m2_ask == 100:
            m2_lost = True
        elif opp_m2_ask <= 1:
            m2_won = True

    if full_market_state and map2_ticker in full_market_state:
        m2_data = full_market_state[map2_ticker]
        m2_status = m2_data.get("status", "")
        if m2_status in ["determined", "settled"]:
            r = m2_data.get("result", "")
            if r == "yes": m2_won = True; m2_auth = True
            elif r == "no": m2_lost = True; m2_auth = True
        elif m2_status in ["closed", "finalized"]:
            lp = float(m2_data.get("last_price_dollars", "0") or "0") * 100.0
            if lp >= 85: m2_won = True; m2_auth = True
            elif lp > 0 and lp <= 15: m2_lost = True; m2_auth = True
            else:
                opp_data = (full_market_state or {}).get(map2_opp, {})
                opp_last = float(opp_data.get("last_price_dollars", "0") or "0") * 100.0
                if opp_last >= 85: m2_lost = True; m2_auth = True
                elif opp_last > 0 and opp_last <= 15: m2_won = True; m2_auth = True
        if m2_status in ["closed", "finalized"] and not m2_won and not m2_lost:
            return None  # Ambiguous M2

    # Persist / recover M2 result across reactivations (see M1 note).
    m2_won, m2_lost, m2_unstable = _recover_map_wl(
        map2_ticker, m2_won, m2_lost, authoritative=m2_auth,
        live_bid=m2_bid, live_ask=m2_ask)
    if m2_unstable:
        return None  # cached M2 contradicted by live book — halt pending confirm

    # M2 not resolved → G2 active
    if not m2_won and not m2_lost:
        wa = 1 if m1_won else 0
        wb = 1 if m1_lost else 0
        return {
            "state": (wa, wb), "wins_a": wa, "wins_b": wb,
            "active_map": map2_ticker, "active_map_opp": map2_opp,
            "m1_won": m1_won, "m1_lost": m1_lost,
            "m2_won": False, "m2_lost": False,
            "m3_won": False, "m3_lost": False,
            "m4_won": False, "m4_lost": False,
        }

    # ── M1 and M2 both resolved → walk M3/M4/M5 ──
    wins_a = (1 if m1_won else 0) + (1 if m2_won else 0)
    wins_b = (1 if m1_lost else 0) + (1 if m2_lost else 0)

    if wins_a >= 3 or wins_b >= 3:
        return None  # Series decided

    def _check_map_result(map_num: int):
        mt = f"{map_base}-{map_num}-{team_suffix}"
        mt_opp = f"{map_base}-{map_num}-{opp_suffix}"
        won = False
        lost = False
        auth = False  # True only when settlement status (not price) resolved it
        bid = full_top_bids.get(mt, 0.0)
        ask = full_top_offers.get(mt, 100.0)
        if bid >= 99 and ask == 100: won = True
        if ask <= 1 and bid == 0: lost = True
        if full_market_state and mt in full_market_state:
            mdata = full_market_state[mt]
            mstatus = mdata.get("status", "")
            if mstatus in ["determined", "settled"]:
                r = mdata.get("result", "")
                if r == "yes": won = True; auth = True
                elif r == "no": lost = True; auth = True
            elif mstatus in ["closed", "finalized"]:
                lp = float(mdata.get("last_price_dollars", "0") or "0") * 100.0
                if lp >= 85: won = True; auth = True
                elif lp > 0 and lp <= 15: lost = True; auth = True
        # Persist / recover this map's result across reactivations (see M1 note).
        won, lost, unstable = _recover_map_wl(
            mt, won, lost, authoritative=auth, live_bid=bid, live_ask=ask)
        return won, lost, unstable

    m3_won = m3_lost = False
    m4_won = m4_lost = False
    active_map = None
    active_map_opp = None

    for extra_map in (3, 4, 5):
        if wins_a >= 3 or wins_b >= 3:
            break
        mw, ml, m_unstable = _check_map_result(extra_map)
        if m_unstable:
            return None  # cached map result contradicted — halt pending confirm
        if extra_map == 3:
            m3_won, m3_lost = mw, ml
        elif extra_map == 4:
            m4_won, m4_lost = mw, ml
        if mw:
            wins_a += 1
        elif ml:
            wins_b += 1
        else:
            active_map = f"{map_base}-{extra_map}-{team_suffix}"
            active_map_opp = f"{map_base}-{extra_map}-{opp_suffix}"
            break

    if wins_a >= 3 or wins_b >= 3:
        return None  # Series decided after walk
    if wins_a + wins_b >= 4:
        return None  # G5 reached — intentionally blocked

    return {
        "state": (wins_a, wins_b), "wins_a": wins_a, "wins_b": wins_b,
        "active_map": active_map, "active_map_opp": active_map_opp,
        "m1_won": m1_won, "m1_lost": m1_lost,
        "m2_won": m2_won, "m2_lost": m2_lost,
        "m3_won": m3_won, "m3_lost": m3_lost,
        "m4_won": m4_won, "m4_lost": m4_lost,
    }


# ── 2. Cross-Book Map Prices ──

def get_cross_book_map_prices(
    active_map: str,
    active_map_opp: str,
    full_top_bids: Dict[str, float],
    full_top_offers: Dict[str, float],
) -> Tuple[float, float]:
    """
    Compute conservative cross-book map prices.

    Returns (best_map_no_ask, best_map_yes_ask):
      best_map_no_ask = cheapest way to buy "our team loses this map"
                      = min(our_map_NO_ask, opp_map_YES_ask)
      best_map_yes_ask = cheapest way to buy "our team wins this map"
                       = min(our_map_YES_ask, opp_map_NO_ask)
    """
    our_bid = full_top_bids.get(active_map, 0.0)
    our_ask = full_top_offers.get(active_map, 100.0)
    opp_bid = full_top_bids.get(active_map_opp, 0.0)
    opp_ask = full_top_offers.get(active_map_opp, 100.0)

    our_no_ask = 100.0 - our_bid       # cost to buy our NO
    opp_no_ask = 100.0 - opp_bid       # cost to buy opp NO

    # Direction 1 hedge (long our team → buy our map NO):
    #   our NO ask vs opponent YES ask (both = "opponent wins map")
    best_map_no_ask = min(our_no_ask, opp_ask) if opp_ask < 100 else our_no_ask

    # Direction 2 hedge (short our team → buy our map YES):
    #   our YES ask vs opponent NO ask (both = "our team wins map")
    best_map_yes_ask = min(our_ask, opp_no_ask) if opp_bid > 0 else our_ask

    return best_map_no_ask, best_map_yes_ask


# ── 3. Hedge Profile Dispatch ──

def compute_hedge_profiles(
    state: int,
    p1: float, p2: float, series: float,
    g2: float, g3: float,
    map_no_ask: float,
    map_yes_ask: float,
    m2_no_ask: float = None,
    m2_yes_ask: float = None,
    use_live_m2_state0: bool = False,
    is_bo5: bool = False,
    p3: float = None,
    p4: float = None,
    p5: float = None,
    wins_a: int = 0,
    wins_b: int = 0,
) -> Tuple[dict, dict]:
    """
    Compute hedge profiles for both directions given the current state.

    Returns (hedge_profile_1, hedge_profile_2):
      hedge_profile_1 = hedge for going LONG our team (buy map NO)
      hedge_profile_2 = hedge for going SHORT our team (buy map YES)

    Each profile has keys: shares_b, cash_reserve, synthetic_cost

    `m2_no_ask`/`m2_yes_ask` and `use_live_m2_state0`: when set in state 0,
    the M2 portion of the hedge math swaps the baseline-p2 assumption for the
    live M2 ask. This handles "M1 essentially decided but Kalshi hasn't
    settled it" scenarios where baseline diverges from real M2 prices.

    Bo5: when `is_bo5=True`, dispatches to `calculate_bo5_hedge` with state
    `(wins_a, wins_b)`. Caller is responsible for deriving wins from active
    map outcomes; Bo3-style `state` is ignored on the Bo5 path. `p3` is the
    map-3 baseline probability (defaults to (p1+p2)/2 if not provided).
    """
    map_no_int = int(round(map_no_ask))
    map_yes_int = int(round(map_yes_ask))

    if is_bo5:
        p3_eff = p3 if p3 is not None else (p1 + p2) / 2.0
        # Anchored per-game probs only. Active map's live signal enters via
        # map_no_int / map_yes_int (the real hedge trade prices), not as a
        # per-game probability override.
        p4_b = (1.0 - p4) if p4 is not None else None
        p5_b = (1.0 - p5) if p5 is not None else None
        hp1 = calculate_bo5_hedge(wins_a, wins_b, p1, p2, p3_eff, g2, map_no_int,
                                  p4=p4, p5=p5)
        hp2 = calculate_bo5_hedge(wins_b, wins_a, 1-p1, 1-p2, 1-p3_eff, g2, map_yes_int,
                                  p4=p4_b, p5=p5_b)
        return hp1, hp2

    if state == 0:
        m2_b_for_hp1 = m2_no_ask if (use_live_m2_state0 and m2_no_ask is not None) else None
        m2_b_for_hp2 = m2_yes_ask if (use_live_m2_state0 and m2_yes_ask is not None) else None
        hp1 = calculate_state_0_hedge(p1, p2, series, g2, g3, map_no_int, m2_b_ask_cents=m2_b_for_hp1)
        hp2 = calculate_state_0_hedge(1-p1, 1-p2, 1-series, g2, g3, map_yes_int, m2_b_ask_cents=m2_b_for_hp2)
    elif state == 1:
        hp1 = calculate_state_1_hedge(p1, p2, series, g2, g3, map_no_int)
        hp2 = calculate_state_2_hedge(1-p1, 1-p2, 1-series, g2, g3, map_yes_int)
    elif state == 2:
        hp1 = calculate_state_2_hedge(p1, p2, series, g2, g3, map_no_int)
        hp2 = calculate_state_1_hedge(1-p1, 1-p2, 1-series, g2, g3, map_yes_int)
    elif state == 3:
        hp1 = calculate_state_3_hedge(map_no_int)
        hp2 = calculate_state_3_hedge(map_yes_int)
    else:
        hp1 = {"shares_b": 1.0, "cash_reserve": 0.0, "synthetic_cost": float(map_no_int)}
        hp2 = {"shares_b": 1.0, "cash_reserve": 0.0, "synthetic_cost": float(map_yes_int)}

    return hp1, hp2


# ── 4. Synthetic Cost Assembly ──

def compute_synthetic_costs(
    series_yes_ask: float,
    series_no_ask: float,
    hedge_profile_1: dict,
    hedge_profile_2: dict,
) -> Tuple[float, float, float, float]:
    """
    Assemble total synthetic costs with loaded fees on the series leg.

    Returns (synthetic_1, synthetic_2, edge_1, edge_2):
      synthetic_1 = loaded(series_YES_ask) + hedge_1_cost  (cost to go LONG)
      synthetic_2 = loaded(series_NO_ask)  + hedge_2_cost  (cost to go SHORT)
      edge_X = 100 - synthetic_X
    """
    series_loaded = get_loaded_cost_cents(series_yes_ask)
    series_no_loaded = get_loaded_cost_cents(series_no_ask)

    synthetic_1 = series_loaded + hedge_profile_1["synthetic_cost"]
    synthetic_2 = series_no_loaded + hedge_profile_2["synthetic_cost"]

    edge_1 = 100.0 - synthetic_1
    edge_2 = 100.0 - synthetic_2

    return synthetic_1, synthetic_2, edge_1, edge_2


# ── 5. Series Theo Derivation ──

def compute_series_theos(
    hedge_profile_1: dict,
    hedge_profile_2: dict,
) -> Tuple[float, float]:
    """
    Derive quoter bid/offer theos from hedge profiles — no fee loading.

    bid_theo  = highest YES price where going LONG is break-even (P + h1 = 100)
    offer_theo = lowest  YES price where going SHORT is break-even (P = h2)

    The Kalshi fee on the trade leg is intentionally NOT loaded into the theo:
    it would widen the bid/offer theo gap on thin hedge books by a fee-shaped
    amount that the quoter then re-pays via min_edge on top, producing
    artificially wide quotes. Edge / fee coverage is handled by min_edge.
    """
    h1_cost = hedge_profile_1["synthetic_cost"]
    h2_cost = hedge_profile_2["synthetic_cost"]

    bid_theo = 100.0 - h1_cost
    offer_theo = h2_cost

    bid_theo = max(1.0, min(99.0, bid_theo))
    offer_theo = max(1.0, min(99.0, offer_theo))

    return bid_theo, offer_theo


def _inverse_loaded(target: float) -> float:
    """
    Solve for x in: x + 7*(x/100)*(1-x/100) = target

    This is a quadratic: x + 7x(100-x)/10000 = target
    → x + 0.07x - 0.0007x² = target
    → -0.0007x² + 1.07x - target = 0
    → 0.0007x² - 1.07x + target = 0

    Using quadratic formula: x = (1.07 - sqrt(1.07² - 4*0.0007*target)) / (2*0.0007)
    """
    if target <= 0:
        return 0.0
    if target >= 100:
        return 100.0

    a = 0.0007
    b = -1.07
    c = target
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return target  # fallback to raw
    # We want the smaller root (the one in [0, 100])
    x = (-b - math.sqrt(discriminant)) / (2 * a)
    return max(0.0, min(100.0, x))


# ── Map-arber additions (ported from map-taker branch) ───────────────────
# These functions support EsportsMapArberBot, the inverse of EsportsArberBot.
# Map_arber trades the active map leg in response to series-side price
# discovery. The existing series-side functions above are unchanged.

def get_cross_book_series_prices(
    series_ticker: str,
    opp_series_ticker: str,
    full_top_bids: Dict[str, float],
    full_top_offers: Dict[str, float],
) -> Tuple[float, float]:
    """
    Conservative cross-book series prices (mirror of get_cross_book_map_prices).

    Returns (best_series_no_ask, best_series_yes_ask):
      best_series_no_ask  = cheapest way to buy "our team loses series"
                          = min(our_series_NO_ask, opp_series_YES_ask)
      best_series_yes_ask = cheapest way to buy "our team wins series"
                          = min(our_series_YES_ask, opp_series_NO_ask)
    """
    our_bid = full_top_bids.get(series_ticker, 0.0)
    our_ask = full_top_offers.get(series_ticker, 100.0)
    opp_bid = full_top_bids.get(opp_series_ticker, 0.0)
    opp_ask = full_top_offers.get(opp_series_ticker, 100.0)

    our_no_ask = 100.0 - our_bid
    opp_no_ask = 100.0 - opp_bid

    best_series_no_ask = min(our_no_ask, opp_ask) if opp_ask < 100 else our_no_ask
    best_series_yes_ask = min(our_ask, opp_no_ask) if opp_bid > 0 else our_ask

    return best_series_no_ask, best_series_yes_ask

def calculate_state_0_map_hedge(p1_A: float, p2_base: float, series_A: float,
                                g2_momentum: float, g3_momentum: float,
                                series_no_ask_cents: float) -> dict:
    """
    State 0 (G1 active). Trade leg: m1_YES. Hedge leg: series_NO.

    Conditional series probabilities by G1 outcome (with momentum cascade):
      A_eff = P(A wins series | A won G1)
            = p2_A1 + (1 - p2_A1) * (p3_base - g3_momentum)
            where p2_A1 = p2_base + g2_momentum
      B_eff = P(A wins series | A lost G1)
            = p2_B1 * (p3_base + g3_momentum)
            where p2_B1 = p2_base - g2_momentum

    Hedge math (1 m1_YES + W series_NO + Z cash, payout = $1 either branch):
      Win G1:  1 + W*(1 - A_eff) + Z = 1
      Lose G1: 0 + W*(1 - B_eff) + Z = 1
      → W = 1 / (A_eff - B_eff)
      → Z = -(1 - A_eff) / (A_eff - B_eff)        (negative; algebraic, not borrowed)
    """
    p3_base = solve_game_3_probability(p1_A, p2_base, series_A, g2_momentum, g3_momentum)
    p2_A1 = max(0.01, min(0.99, p2_base + g2_momentum))
    p2_B1 = max(0.01, min(0.99, p2_base - g2_momentum))
    # Asymmetric per-side g3 caps (see _g3_eff_up). In state 0 both branches
    # are contingent: p3_lose_g2 means B has G2 momentum (cap B side),
    # p3_win_g2 means A has G2 momentum (cap A side).
    g3_A_up = _g3_eff_up(p3_base, g3_momentum)
    g3_B_up = _g3_eff_up(1.0 - p3_base, g3_momentum)
    p3_lose_g2 = max(0.01, min(0.99, p3_base - g3_B_up))
    p3_win_g2 = max(0.01, min(0.99, p3_base + g3_A_up))
    A_eff = p2_A1 + (1.0 - p2_A1) * p3_lose_g2
    B_eff = p2_B1 * p3_win_g2
    spread = A_eff - B_eff
    if spread <= 0.01:
        # Degenerate (would only occur with extreme priors). Fall back to a
        # shares_b that won't blow up; map_arber consumer should treat this
        # as "no edge available" via the resulting synthetic_cost.
        return build_profile(1.0, 0.0, get_raw_cost_cents(series_no_ask_cents))
    shares_b = 1.0 / spread
    cash_reserve = -100.0 * (1.0 - A_eff) / spread
    return build_profile(shares_b, cash_reserve, get_raw_cost_cents(series_no_ask_cents))

def calculate_state_1_map_hedge(p1_A: float, p2_base: float, series_A: float,
                                g2_momentum: float, g3_momentum: float,
                                series_no_ask_cents: float) -> dict:
    """
    State 1 (won G1, G2 active). Trade leg: m2_YES. Hedge leg: series_NO.

    Series mark by G2 outcome:
      Win G2: series resolves YES → series_NO = 0
      Lose G2 (state 3, A on loss-momentum): series_NO marks at
        (1 - p3_A_if_A_lost_G2) = 1 - (p3_base - g3_momentum)

    Portfolio (1 m2_YES + W series_NO):
      Win G2:  1 + 0 = 1
      Lose G2: 0 + W * (1 - p3_A_if_A_lost_G2) = 1
      → W = 1 / (1 - p3_A_if_A_lost_G2)
    Cash reserve = 0 — synthetic is $1 deterministic without normalization.
    """
    p3_base = solve_game_3_probability(p1_A, p2_base, series_A, g2_momentum, g3_momentum)
    # State 1 (A won G1): A's loss-G2 branch is reachable; B has momentum, so
    # cap on B's side.
    g3_B_up = _g3_eff_up(1.0 - p3_base, g3_momentum)
    p3_lose_g2 = max(0.01, min(0.99, p3_base - g3_B_up))
    denom = 1.0 - p3_lose_g2
    if denom <= 0.01:
        return build_profile(100.0, 0.0, get_raw_cost_cents(series_no_ask_cents))
    shares_b = 1.0 / denom
    return build_profile(shares_b, 0.0, get_raw_cost_cents(series_no_ask_cents))

def calculate_state_2_map_hedge(p1_A: float, p2_base: float, series_A: float,
                                g2_momentum: float, g3_momentum: float,
                                series_no_ask_cents: float) -> dict:
    """
    State 2 (lost G1, G2 active). Trade leg: m2_YES. Hedge leg: series_NO.

    Series mark by G2 outcome:
      Win G2 (state 3, A on win-momentum): series_NO marks at
        (1 - p3_A_if_A_won_G2) = 1 - (p3_base + g3_momentum)
      Lose G2: series resolves NO (B wins 2-0) → series_NO = 1

    Portfolio (1 m2_YES + W series_NO + Z cash):
      Win G2:  1 + W*(1 - p3_A_if_A_won_G2) + Z = 1
      Lose G2: 0 + W*1 + Z = 1
      → W = 1 / p3_A_if_A_won_G2
      → Z = 1 - W = (p3_A_if_A_won_G2 - 1) / p3_A_if_A_won_G2  (negative)
    """
    p3_base = solve_game_3_probability(p1_A, p2_base, series_A, g2_momentum, g3_momentum)
    # State 2 (A lost G1): A's win-G2 branch is reachable; A has momentum, so
    # cap on A's side.
    g3_A_up = _g3_eff_up(p3_base, g3_momentum)
    p3_win_g2 = max(0.01, min(0.99, p3_base + g3_A_up))
    if p3_win_g2 <= 0.01:
        return build_profile(100.0, 0.0, get_raw_cost_cents(series_no_ask_cents))
    shares_b = 1.0 / p3_win_g2
    cash_reserve = 100.0 * (p3_win_g2 - 1.0) / p3_win_g2
    return build_profile(shares_b, cash_reserve, get_raw_cost_cents(series_no_ask_cents))

def compute_map_hedge_profiles(
    state: int,
    p1: float, p2: float, series: float,
    g2: float, g3: float,
    series_no_ask: float,
    series_yes_ask: float,
) -> Tuple[dict, dict]:
    """
    Map-side hedge profile dispatcher. Mirror of compute_hedge_profiles.

    Returns (hedge_profile_long, hedge_profile_short):
      hedge_profile_long  = long active-map YES, hedge with series_NO
      hedge_profile_short = long active-map NO,  hedge with series_YES
                          = direction 1 from opponent's frame (inverted args)

    State 3 (1-1, no live map) returns degenerate profiles — caller should
    skip when state == 3. Returned shape matches series-side profiles so
    downstream consumers can treat them uniformly.
    """
    # Map-side hedge keeps full float precision (no int rounding) — the trade
    # leg gets rounded at fire time, but the theoretical math should be exact
    # to avoid spurious 0.5–1c errors that compound across shares_b.
    series_no_f = float(series_no_ask)
    series_yes_f = float(series_yes_ask)

    if state == 0:
        hp1 = calculate_state_0_map_hedge(p1, p2, series, g2, g3, series_no_f)
        hp2 = calculate_state_0_map_hedge(1 - p1, 1 - p2, 1 - series, g2, g3, series_yes_f)
    elif state == 1:
        hp1 = calculate_state_1_map_hedge(p1, p2, series, g2, g3, series_no_f)
        # state 1 from us = state 2 from opp (opp lost G1)
        hp2 = calculate_state_2_map_hedge(1 - p1, 1 - p2, 1 - series, g2, g3, series_yes_f)
    elif state == 2:
        hp1 = calculate_state_2_map_hedge(p1, p2, series, g2, g3, series_no_f)
        # state 2 from us = state 1 from opp (opp won G1)
        hp2 = calculate_state_1_map_hedge(1 - p1, 1 - p2, 1 - series, g2, g3, series_yes_f)
    else:
        # state 3: no separate live-map market in BO3. Return inert profiles.
        hp1 = {"shares_b": 0.0, "cash_reserve": 100.0, "synthetic_cost": 100.0}
        hp2 = {"shares_b": 0.0, "cash_reserve": 100.0, "synthetic_cost": 100.0}

    return hp1, hp2

def compute_map_synthetic_costs(
    map_yes_ask: float,
    map_no_ask: float,
    hedge_profile_long: dict,
    hedge_profile_short: dict,
) -> Tuple[float, float, float, float]:
    """
    Mirror of compute_synthetic_costs — assemble total synthetic on the map side.

    Returns (synthetic_long, synthetic_short, edge_long, edge_short):
      synthetic_long  = loaded(map_YES_ask) + hedge_long_cost
      synthetic_short = loaded(map_NO_ask)  + hedge_short_cost
      edge_X          = 100 - synthetic_X

    map leg incurs Kalshi fees (loaded). Hedge leg uses raw cents — same
    convention as the series-side: hedge is a theoretical reference, not a
    fee-bearing trade in the math (real fees on hedge fills are slippage).
    """
    map_yes_loaded = get_loaded_cost_cents(map_yes_ask)
    map_no_loaded = get_loaded_cost_cents(map_no_ask)

    synthetic_long = map_yes_loaded + hedge_profile_long["synthetic_cost"]
    synthetic_short = map_no_loaded + hedge_profile_short["synthetic_cost"]

    edge_long = 100.0 - synthetic_long
    edge_short = 100.0 - synthetic_short

    return synthetic_long, synthetic_short, edge_long, edge_short

def compute_map_theos(
    hedge_profile_long: dict,
    hedge_profile_short: dict,
) -> Tuple[float, float]:
    """
    Mirror of compute_series_theos — derive map bid/offer theo from hedge
    profiles. No fee loading on the map leg (see compute_series_theos docstring).

    bid_theo  = highest YES price where going LONG map is break-even (P + h_long = 100)
    offer_theo = lowest  YES price where going SHORT map is break-even (P = h_short)

    Both clamped to [1, 99].
    """
    h1_cost = hedge_profile_long["synthetic_cost"]
    h2_cost = hedge_profile_short["synthetic_cost"]

    bid_theo = 100.0 - h1_cost
    offer_theo = h2_cost

    bid_theo = max(1.0, min(99.0, bid_theo))
    offer_theo = max(1.0, min(99.0, offer_theo))

    return bid_theo, offer_theo

