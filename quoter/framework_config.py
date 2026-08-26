import csv
import os
from dataclasses import dataclass, field


# Toggle for the V2 amend-orders dispatch path. When True, manager.run_tick
# uses `_diff_quotes_with_amend` and routes same-level price/size deltas
# through `kalshi_client.amend_order` instead of cancel+post pairs. Preserves
# queue position on amended quotes.
#
# Default flipped 2026-06-21 from OFF → ON. The original flag-file gate
# (`enable_amend.flag`) was fragile: a single `rm` / backup operation or
# `git clean` silently wiped it and reverted production to cancel+post
# without anyone noticing. Discovered when 4 days of "amend is on" turned
# out to be 4 days of cancel+post on every reprice, costing queue position
# and adding ~100ms market-absence per move.
#
# Production default: ON. Disable only by explicit opt-out:
#   - touch disable_amend.flag       (no restart needed; checked per tick)
#   - export DISABLE_AMEND=1          (set before launch)
#
# Legacy `enable_amend.flag` is no longer required — its presence is
# harmless (and historically meaningful for audit) but ignored for
# decision-making. Remove anytime.
_DISABLE_AMEND_FLAG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "disable_amend.flag"
)


def amend_enabled() -> bool:
    """True unless explicitly disabled via flag-file or env var.

    Re-checked per tick so opt-out toggles take effect without restart."""
    if os.environ.get("DISABLE_AMEND") == "1":
        return False
    try:
        if os.path.exists(_DISABLE_AMEND_FLAG_PATH):
            return False
    except Exception:
        pass
    return True


def favourite_edge_mult(our_side_p: float, k: float) -> float:
    """Adverse-selection premium for BUYING THE FAVOURITE (2026-08-09).

    `our_side_p` is the probability-price of the side we would go LONG, in
    [0,1] — NOT the ticker's YES price. Callers that quote/sell YES must pass
    (100 - yes_price)/100, because selling YES is buying NO.

    Rationale: flow prefers backing favourites, so the edge the book offers at
    high prices is systematically worse than Bernoulli variance alone implies.
    The variance term in logit_scaled_edge() *shrinks* the requirement toward
    both extremes, which means we historically demanded the LEAST edge exactly
    in the 70-90c band where the maker bled hardest (measured 2026-08-09:
    -0.63c/contract at 70-80c, -0.78c at 80-90c, versus +4.63c at 40-50c;
    FIFO round-trip attribution, see _pnl_by_entry_price.py).

    SHAPE — variance-of-the-variance. Below 50c nothing changes. On [0.5, 1.0]
    rescale the half-range onto its own Bernoulli variable:

        n = 2*(1 - p)        # 1.0 at 50c  ->  0.0 at 100c
        mult = 1 + k * 4n(1-n)

    4n(1-n) is the same normalized variance term used for the price itself, so
    the bump is 0 at BOTH ends of the half-range and peaks at n=0.5, i.e. 75c.
    That makes it exactly 1.0 at 50c — the join to the untouched lower half is
    seamless, no discontinuity, no piecewise seam to tune.

    `k` is simply the peak excess: k=0.5 -> 1.50x at 75c. k=0.0 disables it
    (returns 1.0) — the legacy behaviour.

      50c 1.00 | 60c 1.32 | 65c 1.42 | 70c 1.48 | 75c 1.50 (peak)
      80c 1.48 | 85c 1.42 | 90c 1.32 | 95c 1.18 | 100c 1.00

    NOTE the multiplier peaks at 75c but the resulting edge REQUIREMENT peaks
    near 65c, because the base 4p(1-p) term is already falling by then. See the
    composite table on logit_scaled_edge().
    """
    if not k or our_side_p < 0.5:
        return 1.0
    q = min(1.0, our_side_p)
    n = 2.0 * (1.0 - q)
    return 1.0 + k * (4.0 * n * (1.0 - n))


def logit_scaled_edge(
    theo_cents: float,
    min_edge: float,
    min_absolute_edge: float,
    fav_edge_k: float = 0.0,
    our_side_cents: float | None = None,
) -> float:
    """Returns the effective edge requirement at a given theo price level.

    Edge required scales with Bernoulli variance: 4*p*(1-p) where p = theo/100.
    Normalized so the scale factor = 1 at p=0.5 (i.e. behaves like flat min_edge
    in the middle of the book) and SHRINKS toward the extremes. The intuition:
    at extreme prices the per-contract variance is small, so a smaller cent-edge
    suffices to compensate. min_absolute_edge floors the requirement so we never
    quote inside an unconditional safety bound.

    Example with min_edge=3.0, min_absolute_edge=1.5:
      p=0.50: scale=1.00 → max(3.00, 1.5) = 3.0c required
      p=0.20: scale=0.64 → max(1.92, 1.5) = 1.92c required
      p=0.10: scale=0.36 → max(1.08, 1.5) = 1.5c required  ← floor binds
      p=0.05: scale=0.19 → max(0.57, 1.5) = 1.5c required  ← floor binds

    fav_edge_k (2026-08-09, default 0.0 = EXACT legacy behaviour) layers the
    favourite-side adverse-selection premium on top of the variance term:

        scale = 4*p*(1-p) * favourite_edge_mult(our_side_p, fav_edge_k)

    The variance term is symmetric, so it never cared which side we were long.
    The premium is ASYMMETRIC and does. `our_side_cents` is the price of the
    side we would go long; it defaults to theo_cents, which is correct wherever
    theo_cents is already the price we pay (every arber call site — they pass
    series_ask_p). The quoter's OFFER side must pass 100 - offer_theo.

    With min_edge=3.0 / min_absolute_edge=1.5 / fav_edge_k=0.5, edge required:
      <=50c unchanged | 55c 2.97→3.50 | 60c 2.88→3.80 | 65c 2.73→3.88 (peak)
      70c 2.52→3.73 | 75c 2.25→3.38 | 80c 1.92→2.84 | 85c 1.53→2.17
      90c+ unchanged — the min_absolute_edge floor already binds up there.
    """
    p = max(0.01, min(0.99, theo_cents / 100.0))
    scale = 4.0 * p * (1.0 - p)
    if fav_edge_k:
        side_c = theo_cents if our_side_cents is None else our_side_cents
        scale *= favourite_edge_mult(max(0.01, min(0.99, side_c / 100.0)), fav_edge_k)
    return max(min_edge * scale, min_absolute_edge)


@dataclass
class QuoterConfig:
    """
    Configuration for a single level-based quoting strategy instance.
    """
    market_id: str  # Unique identifier for this quoting bot strategy
    ticker: str
    min_distance_from_top_level: float
    min_edge: float
    min_absolute_edge: float
    volumes: list[int] = field(default_factory=lambda: [1])
    tick_step: int = 1
    reprice_buffer: float = 0.5
    stop_quoting: bool = False

    # Conditional cluster-cooldown re-fire knobs (arber taker path,
    # 2026-08-01). Within maker_refill_cooldown_sec of a fire, allow up to
    # refire_max_per_window re-fires when the prior fire verifiably filled
    # (net position moved >= refire_min_fill_lots in the fire direction) and
    # the book refilled (>= refire_min_depth_lots displayed at the fire
    # level) with price/hedge not worse. See EsportsArberBot._material_refire.
    refire_max_per_window: int = 2
    refire_min_fill_lots: int = 10
    refire_min_depth_lots: int = 100
    # Decay veto (2026-08-02, BLGJDG): within the cooldown window, suppress any
    # fire at a no-better ask whose hedge is > this many TENTHS of a cent worse
    # than the window's best fire. See EsportsArberBot._cluster_gate.
    cluster_hedge_fade_tenths: int = 5

    # Maker refill cooldown: when > 0, after a maker fill on (ticker, side)
    # the QuoterBot cancels resting quotes in that exposure direction and
    # blocks refills for this many seconds. Cross-ticker via a shared
    # direction-keyed store: a YES fill on team_X cools both BID YES on
    # team_X's ticker AND BID NO on team_Y's ticker (both = "long_X").
    # Default 0 = disabled, preserves legacy behavior.
    maker_refill_cooldown_sec: float = 0.0

    # Trade-retreat theo shift (TRADE_RETREAT_SCOPE.md, armed 26AUG18).
    # Recency-flow term applied in PositionAdjuster.adjust_theos — MAKERS
    # ONLY, additive with (not replacing) the static position skew.
    # retreat_cap_cents is the per-row arm switch: 0 = disarmed (shadow
    # readout on LEAD-LAG lines only), 10 = the FINAL LOCKED cap. Defaults
    # below mirror retreat_shadow module constants so absent columns behave
    # exactly like the pre-column shadow build.
    retreat_cap_cents: float = 0.0
    retreat_f_cap: float = 0.18
    retreat_f0: float = 0.022
    retreat_half_life_sec: float = 300.0
    retreat_include_takers: bool = False

    # Position Adjuster Skew Parameters
    #   skew = 0 when |pos| ≤ skew_start_fraction × max_position
    #   skew ramps linearly to skew_max_shift_cents at |pos| = skew_full_fraction × max_position
    #   skew stays capped at skew_max_shift_cents for |pos| above that point
    # Default skew_full_fraction=1.0 = legacy (cap reached at max_position).
    max_position: int = 100
    skew_start_fraction: float = 0.5
    skew_full_fraction: float = 1.0
    skew_max_shift_cents: float = 2.0
    
    # Uni-directional quoting configuration mapping ("both", "bids", "offers")
    quote_side: str = "both"
    
    # Mathematical Override Router
    model_name: str = "vwap"
    # Execution type classification
    execution_type: str = "quoter"

    # Arbitrage Specialized Expansion Limits
    arb_scale_step_cents: float = 1.0
    max_fire_size: int = 500

    # Series-trigger edge multiplier. When the arber's d_trigger == "series",
    # effective min_edge = min_edge * series_edge_mult. Bumping this above 1.0
    # tightens series-triggered fires (e.g. CS T1 = 2.0) without affecting
    # map-triggered fires. Per-row config 2026-06-03 (was a module-level constant).
    series_edge_mult: float = 1.0

    # G3 buffer multiplier. Both arber and quoter widen min_edge by
    # `g3_edge_mult × P(G3) × min_edge` during BO3 state 1/2 (G2 active) to
    # protect against systematic G3 mispricing. Per-row knob 2026-06-23
    # (was a module-level helper in arber_bot + a hardcoded constant in bot.py).
    # Convention: 0.5 for non-CS2-T1 events, 0.25 for CS2 tier-1 (deeper edge
    # surplus, can afford smaller buffer). Setting to 0.0 disables the buffer.
    g3_edge_mult: float = 0.5

    # Asymmetric YES-side widening (QuoterBot only). When > 0, posts YES bids
    # this many cents lower than the symmetric calculation (NO bids unaffected).
    # Theory: YES-side flow is more adversely-selected (sophisticated takers
    # buying NO_team_B), so widening just the YES bid reduces exposure to it
    # without hurting NO-side fills that mostly come from retail YES_team_A buyers.
    # Default 0.0 = legacy symmetric behavior. See _yes_extra_edge_test.py.
    yes_extra_edge_cents: float = 0.0

    # Favourite-side adverse-selection premium (2026-08-09). Multiplies the
    # variance-scaled edge requirement by favourite_edge_mult(our_side_p, k)
    # when we would be LONG the high-priced side. k is the PEAK EXCESS at 75c:
    # 0.5 = 1.50x at 75c, tapering to 1.0x at both 50c and 100c. Applies to
    # BOTH maker (quoter.py)
    # and taker (arber sweep) paths. Default 0.0 = disabled / exact legacy.
    # Fitted to the post-08-June regime only — pre-08-June LoL maker was
    # positive in all 10 price buckets, so this is a per-row knob, not a
    # constant, and should be backed out if the counterparty leaves.
    # See _fav_edge_test.py.
    fav_edge_k: float = 0.0

    # Pregame-taker gate (arber TAKER only). When True, the arber suppresses ALL
    # taker fires (map/series/init) while the event is "pregame" — i.e. before the
    # series mid OR the map-implied hedge has moved >= pregame_block_cents from the
    # event's first-seen init (latched: once it moves, in-game for the rest of the
    # event). Markout study (2wk+4wk to 2026-07-14): pregame takes net -0.4c/contract
    # across every trigger; in-game +2.5c. Blocking pregame takes removes toxic fills
    # AND frees position capacity for in-game edge. Maker/quoter is untouched (separate
    # class). Momentum is a separate bot and NOT gated here. Default OFF.
    pregame_taker_block: bool = False
    pregame_block_cents: float = 5.0

    # Data source routing for esports map prices and forfeit detection.
    # Allowed values:
    #   "kalshi"             — prices always from Kalshi WS; no Poly forfeit check
    #                          (default 2026-05-29 — Poly book freezes burned us;
    #                           flip back to "kalshi_na_poly_eu" once Poly trusted again)
    #   "kalshi_na_poly_eu"  — Poly during EU hours, Kalshi WS during NA hours
    #   "poly"               — prices always from Poly; forfeit check runs 24/7
    data_source: str = "kalshi"

    # Phase 2 feed-integrity: per-row source for Kalshi orderbook reads.
    # Allowed values:
    #   "ws"    — reads from WebSocket (default; current production behavior)
    #   "rest"  — reads from REST-polled OrderbookRegistry (orderbook_poller).
    #             Up to ~1s stale but immune to WS delta-application corruption
    #             that caused the VTCGL incident 2026-06-09 21:52 UTC.
    # Per-row default to "ws" preserves existing behavior unless explicitly opted in.
    book_source: str = "ws"

    # Temporary kill switch for forfeit detection. When True, evaluate_forfeit
    # returns (False, None) immediately for this config. Used for tier 1 CS
    # where matches effectively never get forfeited and the 50/50 pin check
    # was producing false positives.
    disable_forfeit_check: bool = False

    # Per-event capture buffer (minutes before scheduled start at which
    # populate_configs.py captures probabilities). Sourced from parsed_markets.csv
    # via league keyword matching in refresh_tomorrow_markets.py. Default 120
    # = legacy behavior. Tight values (e.g. 20 for LCK) reflect fixed-schedule
    # leagues; wide values (e.g. 120 for LCS / VCT Americas) reflect rolling.
    capture_buffer_min: float = 120.0

    # ── Momentum strategy parameters (execution_type="momentum") ──
    # Standalone bot that fires top-of-book IOCs in the band BELOW the regular
    # arber's threshold, triggered by sudden hedge-cost moves (map jumps).
    # All defaults are "off" — bot returns no orders unless explicitly
    # configured per row in market_parameters.csv.
    momentum_lookback_sec: float = 5.0          # baseline cycle must be ≥ this many seconds old
    momentum_delta_hedge_min: float = 0.0       # |Δhedge| floor in cents; 0 = bot disabled (no triggers)
    momentum_edge_floor: float = -1.0           # min edge to fire (cents); below this = skip
    momentum_require_hedge_driven: bool = True  # require |Δhedge| > |Δbest|
    momentum_require_hedge_confirms: bool = True  # opening fires: require Δhedge<0 (hedge move must CONFIRM direction, not oppose it)
    momentum_event_cap_per_dir: int = 5         # max fires per direction per series (BIGEWI defense)
    momentum_throttle_sec: float = 30.0         # cooldown between fires on same direction
    momentum_volumes: int = 0                   # standard fire cap (lots); 0 = disabled
    momentum_reducing_cap: int = 0              # fire cap when reducing position; 0 = use momentum_volumes

def load_quoter_configs_from_csv(filepath: str) -> list[QuoterConfig]:
    """
    Parses a CSV into a list of QuoterConfig objects.
    Expected CSV Columns:
    market_id, ticker, min_distance_from_top_level, min_edge, min_absolute_edge, volumes, tick_step
    (volumes should be comma separated like "5,10,15")
    """
    configs = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
                
            # Parse volumes string (e.g. "10,20,30" -> [10, 20, 30])
            vol_str = row.get("volumes", "1")
            volumes = [int(v.strip()) for v in vol_str.split(",") if v.strip()]
            
            config = QuoterConfig(
                market_id=row.get("market_id", row.get("config_id", "")).strip(),
                ticker=row.get("ticker", row.get("market_prefix", "")).strip(),
                min_distance_from_top_level=float(row.get("min_distance_from_top_level", 0.0)),
                min_edge=float(row.get("min_edge", 0.0)),
                min_absolute_edge=float(row.get("min_absolute_edge", 0.0)),
                volumes=volumes,
                tick_step=int(row.get("tick_step", 1)),
                reprice_buffer=float(row.get("reprice_buffer", 0.5)),
                stop_quoting=str(row.get("stop_quoting", "false")).lower() in ["true", "1", "yes", "y"],
                maker_refill_cooldown_sec=float(row.get("maker_refill_cooldown_sec", 0.0)),
                retreat_cap_cents=float(row.get("retreat_cap_cents", 0) or 0),
                retreat_f_cap=float(row.get("retreat_f_cap", 0.18) or 0.18),
                retreat_f0=float(row.get("retreat_f0", 0.022) or 0.022),
                retreat_half_life_sec=float(row.get("retreat_half_life_sec", 300) or 300),
                retreat_include_takers=str(row.get("retreat_include_takers", "0") or "0").strip().lower() in ["true", "1", "yes", "y"],
                refire_max_per_window=int(float(row.get("refire_max_per_window", 2) or 2)),
                refire_min_fill_lots=int(float(row.get("refire_min_fill_lots", 10) or 10)),
                refire_min_depth_lots=int(float(row.get("refire_min_depth_lots", 100) or 100)),
                cluster_hedge_fade_tenths=int(float(row.get("cluster_hedge_fade_tenths", 5) or 5)),
                max_position=int(row.get("max_position", 100)),
                skew_start_fraction=float(row.get("skew_start_fraction", 0.5)),
                skew_full_fraction=float(row.get("skew_full_fraction", 1.0) or 1.0),
                skew_max_shift_cents=float(row.get("skew_max_shift_cents", 2.0)),
                quote_side=str(row.get("quote_side", "both")).lower().strip(),
                model_name=str(row.get("model_name", "vwap")).lower().strip(),
                execution_type=str(row.get("execution_type", "quoter")).lower().strip(),
                arb_scale_step_cents=float(row.get("arb_scale_step_cents", 1.0)),
                max_fire_size=int(row.get("max_fire_size", 500)),
                series_edge_mult=float(row.get("series_edge_mult", 1.0) or 1.0),
                pregame_taker_block=str(row.get("pregame_taker_block", "false")).strip().lower() in ["true", "1", "yes", "y"],
                pregame_block_cents=float(row.get("pregame_block_cents", 5.0) or 5.0),
                g3_edge_mult=float(row.get("g3_edge_mult", 0.5) or 0.5),
                yes_extra_edge_cents=float(row.get("yes_extra_edge_cents", 0.0) or 0.0),
                fav_edge_k=float(row.get("fav_edge_k", 0.0) or 0.0),
                data_source=str(row.get("data_source", "kalshi")).strip().lower() or "kalshi",
                book_source=str(row.get("book_source", "ws")).strip().lower() or "ws",
                disable_forfeit_check=str(row.get("disable_forfeit_check", "false")).lower() in ["true", "1", "yes", "y"],
                capture_buffer_min=float(row.get("capture_buffer_min", 120.0) or 120.0),
                # Momentum strategy fields — all default to disabled if absent.
                momentum_lookback_sec=float(row.get("momentum_lookback_sec", 5.0) or 5.0),
                momentum_delta_hedge_min=float(row.get("momentum_delta_hedge_min", 0.0) or 0.0),
                momentum_edge_floor=float(row.get("momentum_edge_floor", -1.0) or -1.0),
                momentum_require_hedge_driven=str(row.get("momentum_require_hedge_driven", "true")).lower() in ["true", "1", "yes", "y"],
                momentum_require_hedge_confirms=str(row.get("momentum_require_hedge_confirms", "true")).lower() in ["true", "1", "yes", "y"],
                momentum_event_cap_per_dir=int(row.get("momentum_event_cap_per_dir", 5) or 5),
                momentum_throttle_sec=float(row.get("momentum_throttle_sec", 30.0) or 30.0),
                momentum_volumes=int(row.get("momentum_volumes", 0) or 0),
                momentum_reducing_cap=int(row.get("momentum_reducing_cap", 0) or 0),
            )
            configs.append(config)
            
    return configs
