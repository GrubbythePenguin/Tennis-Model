"""
Polar Bear Trading Strategy

Detects anomalously large resting orders ("polar bears") on Kalshi and trades against them
when Polymarket confirms a sustained price dislocation, indicating the polar bear is on the
wrong side of a real price move.

Fire conditions (ALL must be true):
1. Polar bear detected: anomalously large order near top of book (within 5c)
2. Polymarket dislocation: Poly BID > polar bear offer price + threshold, sustained 10+ seconds
3. Chip confirmation: other market participants have consumed 25%+ of the polar bear's order

Uses Poly BID (not mid) to avoid false triggers from artificial spread widening.
"""

import logging
import time
import requests
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from framework_config import QuoterConfig
from models import QuoteSide
import position_store

# Import Polymarket helpers from populate_configs
from populate_configs import (
    KALSHI_TO_POLY_SPORT, TEAM_ALIASES, POLY_SPORT_TAGS,
    _match_poly_event, _get_poly_winner_markets,
    _align_poly_to_kalshi,
)

log = logging.getLogger(__name__)

# ── Tunable constants ──
NEAR_TOP_CENTS = 5               # Only actively track bears within 5c of best price
DISLOCATION_SUSTAIN_SECS = 10.0  # Poly dislocation must persist this long
CHIP_THRESHOLD_PCT = 0.25        # 25% of original order consumed before firing
FIRE_COOLDOWN_SECS = 30.0        # Per-ticker cooldown after firing
BEAR_STALE_SECS = 10.0           # Remove bears not seen in orderbook for this long
BEAR_DROP_THRESHOLD = 500        # Remove bear if size drops below this (arber can handle it)
POLY_DISCOVERY_SECS = 60.0       # How often to re-discover Polymarket events via gamma API
ANOMALY_RATIO = 5.0              # Order must be 5x the median to qualify


@dataclass
class TrackedBear:
    """A tracked large resting order that may be a polar bear."""
    ticker: str
    book_side: str          # "yes_dollars" or "no_dollars"
    price_dollars: float    # The orderbook price level (in dollars)
    original_size: int      # Size when first detected
    current_size: int       # Current remaining size
    first_seen: float
    last_seen: float
    dislocation_since: float | None = None

    @property
    def buy_side(self) -> str:
        """The Kalshi side we'd buy to trade against this bear.

        yes_dollars = NO bids (= YES asks). Bear is selling YES → we BUY YES.
        no_dollars = YES bids (= NO asks). Bear is buying YES → we BUY NO.
        """
        return "yes" if self.book_side == "yes_dollars" else "no"

    @property
    def offer_price_cents(self) -> int:
        """Price at which we can buy (cents)."""
        return int(round(100 - self.price_dollars * 100))

    @property
    def chip_pct(self) -> float:
        """Fraction of original order consumed by others."""
        if self.original_size <= 0:
            return 0.0
        return 1.0 - (self.current_size / self.original_size)

    @property
    def bear_id(self) -> str:
        return f"{self.ticker}:{self.book_side}:{self.price_dollars:.4f}"


class PolarBearBot:
    """
    Detects anomalously large resting orders and trades against them when
    Polymarket confirms a sustained price dislocation.
    """

    def __init__(self, config: QuoterConfig):
        self.config = config
        self.active = True

        # Config-driven thresholds
        # min_edge → minimum Poly dislocation in cents
        self.min_dislocation_cents = max(config.min_edge, config.min_absolute_edge, 5.0)
        # volumes[0] → minimum absolute size to consider as a polar bear
        self.min_bear_size = config.volumes[0] if config.volumes else 2000

        # Tracked bears: bear_id -> TrackedBear
        self.tracked_bears: dict[str, TrackedBear] = {}
        # Per-ticker fire cooldowns
        self.fire_cooldowns: dict[str, float] = {}

        # Independent position tracking for polar bear strategy only.
        # The shared position_store includes arber/quoter fills, which would
        # incorrectly block polar bear trades. This tracks only PB-fired contracts.
        self._pb_position: dict[str, int] = {}  # event_base -> net contracts

        # Periodic summary logging
        self._last_summary_ts: float = 0.0

        # Polymarket caching
        self._poly_sport: str | None = None
        for prefix, sport in KALSHI_TO_POLY_SPORT.items():
            if prefix in config.ticker:
                self._poly_sport = sport
                break

        self._poly_event_cache: dict[str, dict] = {}   # event_base -> {winners dict with token IDs}
        self._poly_events_ts: float = 0.0              # last gamma API fetch (for event discovery only)

    # ────────────────────────────────────────────────────────────────
    # Orderbook scanning
    # ────────────────────────────────────────────────────────────────

    def _scan_orderbook(self, ticker: str, raw_ob: dict, market_ref_cents: int = 0) -> list[TrackedBear]:
        """Find anomalously large orders near the market price.

        market_ref_cents: implied YES market price from opponent's book (0 if unknown).

        Key insight: polar bears are large orders near the MARKET price.
        - On yes_dollars (YES bids): polar bears are at HIGH prices (near market).
          Sorted descending, they're near levels[0]. Anchor = market YES price.
        - On no_dollars (NO bids): polar bears are at LOW prices (near 100-market).
          Sorted descending, they're near levels[-1]. Anchor = market NO price.
        """
        now = time.time()
        found = []

        for book_side in ["yes_dollars", "no_dollars"]:
            levels = raw_ob.get(book_side, [])
            if not levels:
                continue

            raw_top = int(round(float(levels[0][0]) * 100))

            if book_side == "yes_dollars":
                # yes_dollars = YES bids. Polar bears here are large YES buyers.
                # Anchor to market YES price so we scan levels near market.
                if market_ref_cents > 0:
                    anchor = max(raw_top, market_ref_cents)
                else:
                    anchor = raw_top
            else:
                # no_dollars = NO bids (= YES asks at 100-P).
                # Polar bears here are large NO buyers (selling YES).
                # Anchor to market NO price = 100 - market_ref.
                if market_ref_cents > 0:
                    anchor = max(raw_top, 100 - market_ref_cents)
                else:
                    anchor = raw_top

            # Collect sizes ONLY for levels near the anchor
            near_top = []
            for pt in levels:
                if len(pt) < 2:
                    continue
                pc = int(round(float(pt[0]) * 100))
                sz = float(pt[1])
                if abs(anchor - pc) <= NEAR_TOP_CENTS and sz > 0:
                    near_top.append((float(pt[0]), int(sz), pc))

            if not near_top:
                continue

            # Debug: show all near-anchor levels for diagnosis
            if any(s >= self.min_bear_size for _, s, _ in near_top):
                level_str = " | ".join(f"{s}@{pc}c" for _, s, pc in near_top)
                log.debug(f"[PB SCAN] {ticker} {book_side} anchor={anchor}c ref={market_ref_cents}c "
                          f"near_market: {level_str}")

            for price_d, size, price_c in near_top:
                # Must exceed minimum absolute size
                if size < self.min_bear_size:
                    continue

                # Compute median of OTHER near-top levels (excluding this candidate).
                # A polar bear is anomalous relative to its neighbors, not itself.
                other_sizes = sorted([s for pd, s, pc in near_top if abs(pd - price_d) >= 0.0001])

                if other_sizes:
                    median_other = other_sizes[len(other_sizes) // 2]
                    if median_other > 0 and size < ANOMALY_RATIO * median_other:
                        log.debug(f"[PB SCAN] REJECTED {ticker} {book_side} {size}@{price_c}c: "
                                  f"median_other={median_other} threshold={ANOMALY_RATIO*median_other:.0f} "
                                  f"others={other_sizes}")
                        continue
                # else: order is alone near top of book — a lone massive order
                # with nothing around it qualifies if it meets min_size (already checked)

                found.append(TrackedBear(
                    ticker=ticker,
                    book_side=book_side,
                    price_dollars=price_d,
                    original_size=size,
                    current_size=size,
                    first_seen=now,
                    last_seen=now,
                ))

        return found

    def _update_bears(self, ticker: str, raw_ob: dict, market_ref_cents: int = 0):
        """Reconcile tracked bears against current orderbook state."""
        now = time.time()
        current_bears = self._scan_orderbook(ticker, raw_ob, market_ref_cents)
        current_map = {b.bear_id: b for b in current_bears}

        # Update existing bears for this ticker
        expired = []
        for bear_id, bear in self.tracked_bears.items():
            if bear.ticker != ticker:
                continue

            # Read raw level size regardless of whether it still qualifies as anomalous
            raw_level_size = 0
            levels = raw_ob.get(bear.book_side, [])
            for pt in levels:
                if abs(float(pt[0]) - bear.price_dollars) < 0.0001:
                    raw_level_size = int(float(pt[1]))
                    break

            if raw_level_size <= 0:
                # Price level gone entirely — bear pulled or fully consumed
                expired.append(bear_id)
                continue

            if raw_level_size < BEAR_DROP_THRESHOLD:
                # Size dropped below threshold — no longer a "polar bear",
                # the arber can handle normal-sized orders on its own
                expired.append(bear_id)
                continue

            # Re-validate: if this bear no longer qualifies as anomalous
            # in the current scan, expire it (e.g. larger orders appeared behind it)
            if bear_id not in current_map:
                expired.append(bear_id)
                log.info(f"[POLAR BEAR] No longer anomalous: {bear_id} "
                         f"({raw_level_size} lots, book changed)")
                continue

            bear.last_seen = now
            bear.current_size = raw_level_size

            # If level size grew, other orders were added (not our bear growing).
            # Ratchet original_size up so chip% stays conservative.
            if raw_level_size > bear.original_size:
                bear.original_size = raw_level_size
                bear.dislocation_since = None

        for bear_id in expired:
            removed = self.tracked_bears.pop(bear_id, None)
            if removed:
                log.info(f"[POLAR BEAR] Expired: {bear_id} "
                         f"({'pulled' if removed.chip_pct < 0.5 else 'consumed'})")

        # Register new bears
        for bear in current_bears:
            if bear.bear_id not in self.tracked_bears:
                self.tracked_bears[bear.bear_id] = bear
                log.info(f"[POLAR BEAR] Tracking: {bear.ticker} {bear.book_side} "
                         f"{bear.current_size} lots @ {bear.offer_price_cents}c "
                         f"(buy {bear.buy_side})")

    # ────────────────────────────────────────────────────────────────
    # Polymarket price feed
    # ────────────────────────────────────────────────────────────────

    def _discover_poly_event(self, event_base: str):
        """Discover and cache Polymarket event/token structure via gamma API.

        Only used for initial event matching and token ID discovery.
        Actual PRICES come from CLOB endpoints (real-time), NOT from
        gamma outcomePrices (which are delayed 30-60+ seconds).
        """
        now = time.time()
        if event_base in self._poly_event_cache and now - self._poly_events_ts < POLY_DISCOVERY_SECS:
            return
        self._poly_events_ts = now

        if not self._poly_sport:
            return

        tag_id = POLY_SPORT_TAGS.get(self._poly_sport)
        if not tag_id:
            return

        try:
            r = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"tag_id": tag_id, "active": "true", "closed": "false", "limit": 500},
                timeout=10,
            )
            if r.status_code == 200:
                fresh_events = r.json()
                poly_event = _match_poly_event(event_base, fresh_events)
                if poly_event:
                    winners = _get_poly_winner_markets(poly_event)
                    self._poly_event_cache[event_base] = winners
        except Exception as e:
            log.debug(f"[POLAR BEAR] Poly discovery error: {e}")

    def _find_aligned_token(self, team_suffix: str, poly_winner: dict,
                            kalshi_full_name: str = "") -> tuple[int, str | None]:
        """Find the Polymarket token index and ID aligned to a Kalshi team.
        Uses kalshi_full_name (from yes_sub_title) for reliable matching.
        Falls back to alias matching only if full name is unavailable.
        Returns (-1, None) if no match — caller must skip, never guess.
        """
        outcomes = poly_winner.get("outcomes", [])
        tokens = poly_winner.get("tokens", [])
        if not outcomes or not tokens:
            return -1, None

        # Primary: use full team name from Kalshi yes_sub_title
        aligned_idx = -1
        if kalshi_full_name:
            from populate_configs import _align_team_to_poly_outcome
            aligned_idx = _align_team_to_poly_outcome(kalshi_full_name, outcomes)

        # Fallback: alias-based matching (only if yes_sub_title unavailable)
        if aligned_idx == -1 and not kalshi_full_name:
            import re as _re
            def _alias_matches(alias, outcome_lower):
                return bool(_re.search(r'\b' + _re.escape(alias) + r'\b', outcome_lower))
            for alias_key, aliases in TEAM_ALIASES.items():
                if alias_key.upper() == team_suffix or team_suffix.lower() in [a.lower() for a in aliases]:
                    for oi, outcome in enumerate(outcomes):
                        if any(_alias_matches(a, outcome.lower()) for a in aliases):
                            aligned_idx = oi
                            break
                    break
            if aligned_idx == -1:
                pattern = r'\b' + _re.escape(team_suffix.lower()) + r'\b'
                for oi, outcome in enumerate(outcomes):
                    if _re.search(pattern, outcome.lower()):
                        aligned_idx = oi
                        break

        if aligned_idx < 0 or aligned_idx >= len(tokens):
            return -1, None
        return aligned_idx, tokens[aligned_idx]

    def _fetch_clob_live(self, token_id: str) -> tuple[float | None, float | None]:
        """Fetch LIVE mid and spread from Polymarket CLOB. Returns (mid_cents, spread_cents).

        These endpoints return real-time data, unlike gamma API outcomePrices
        which can be delayed 30-60+ seconds.
        """
        mid_cents = None
        spread_cents = None
        try:
            r = requests.get(f"https://clob.polymarket.com/midpoint?token_id={token_id}", timeout=3)
            if r.status_code == 200:
                v = r.json().get("mid")
                if v:
                    mid_cents = float(v) * 100
        except Exception:
            pass
        try:
            r = requests.get(f"https://clob.polymarket.com/spread?token_id={token_id}", timeout=3)
            if r.status_code == 200:
                v = r.json().get("spread")
                if v:
                    spread_cents = float(v) * 100
        except Exception:
            pass
        return mid_cents, spread_cents

    def _get_poly_bid(self, event_base: str, team_suffix: str, buy_side: str) -> float | None:
        """
        Get LIVE Polymarket BID price for the asset we'd buy. Returns cents or None.

        Uses CLOB midpoint and spread endpoints (real-time) instead of gamma API
        outcomePrices (which are delayed 30-60+ seconds and caused false triggers).
        bid = CLOB_midpoint - CLOB_spread / 2
        """
        # For MAP tickers, derive GAME event base and poly leg
        poly_event_base = event_base
        poly_leg = "series"
        if "MAP" in event_base or "SETWINNER" in event_base:
            map_parts = event_base.split("-")
            map_num = map_parts[-1] if map_parts[-1] in ("1", "2", "3") else None
            if map_num:
                game_prefix = map_parts[0].replace("MAP", "GAME").replace("SETWINNER", "MATCH")
                poly_event_base = game_prefix + "-" + map_parts[1]
                poly_leg = f"game{map_num}"

        # Discover event structure (token IDs) via gamma API — infrequent
        self._discover_poly_event(poly_event_base)

        winners = self._poly_event_cache.get(poly_event_base)
        if not winners:
            return None

        pw = winners.get(poly_leg)
        if not pw:
            return None

        # Find the token aligned to our team
        aligned_idx, token_id = self._find_aligned_token(team_suffix, pw)
        if token_id is None:
            return None

        # Fetch LIVE price from CLOB (real-time, not gamma cache)
        mid_cents, spread_cents = self._fetch_clob_live(token_id)
        if mid_cents is None:
            return None
        if spread_cents is None:
            spread_cents = 0.0

        # Diagnostic: log the alignment so we can verify correct team
        outcomes = pw.get("outcomes", [])
        aligned_name = outcomes[aligned_idx] if aligned_idx < len(outcomes) else "?"
        log.debug(f"[PB POLY] team={team_suffix} aligned_to='{aligned_name}' (idx={aligned_idx}) "
                  f"mid={mid_cents:.1f}c spread={spread_cents:.1f}c leg={poly_leg} buy_side={buy_side}")

        if buy_side == "yes":
            # CLOB mid is for this token (aligned team's YES)
            return mid_cents - spread_cents / 2.0
        else:
            # We want NO price: bid for NO = (100 - YES mid) - spread/2
            return (100.0 - mid_cents) - spread_cents / 2.0

    # ────────────────────────────────────────────────────────────────
    # Main evaluation
    # ────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        market_state: Any,
        top_level_bid: float,
        top_level_offer: float,
        bid_theo: float,
        offer_theo: float,
        full_market_state: Dict[str, Any] = None,
        full_top_bids: Dict[str, float] = None,
        full_top_offers: Dict[str, float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not self.active or not full_market_state:
            return []

        ticker = self.config.ticker
        if not any(k in ticker for k in ["GAME", "MATCH", "MAP", "SETWINNER"]):
            return []

        now = time.time()
        parts = ticker.split("-")
        if len(parts) < 3:
            return []

        # MAP tickers: KXLOLMAP-26APR190400NSBRO-1-NS (4 parts)
        # GAME tickers: KXLOLGAME-26APR190400NSBRO-NS (3 parts)
        team_suffix = parts[-1]
        event_base = "-".join(parts[:-1])  # everything except team suffix

        # Alphabetical dedup — only one bot per event runs
        opp_ticker = None
        for k in (full_top_bids or {}):
            if k.startswith(event_base + "-") and k != ticker:
                opp_ticker = k
                break
        opp_suffix = opp_ticker.split("-")[-1] if opp_ticker else ""
        if opp_suffix and team_suffix > opp_suffix:
            return []

        # Both tickers must be active
        our_status = full_market_state.get(ticker, {}).get("status", "")
        if our_status != "active":
            return []

        # ── Compute implied market price from opponent's book ──
        # Used to anchor "near top" and prevent phantom detections on depleted books.
        # If our ticker is SEN and opponent is 100T with a 25c YES bid,
        # then SEN implied market = 100 - 25 = 75c.
        opp_bid = (full_top_bids or {}).get(opp_ticker, 0) if opp_ticker else 0
        our_bid = (full_top_bids or {}).get(ticker, 0)
        # Use opponent-implied price as our market reference
        our_market_ref = (100 - opp_bid) if opp_bid > 0 else 0
        opp_market_ref = (100 - our_bid) if our_bid > 0 else 0

        # ── Scan orderbooks for both teams ──
        raw_ob = full_market_state.get(ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})
        if raw_ob:
            self._update_bears(ticker, raw_ob, our_market_ref)

        if opp_ticker:
            opp_raw_ob = full_market_state.get(opp_ticker, {}).get("raw_ob", {}).get("orderbook_fp", {})
            if opp_raw_ob:
                self._update_bears(opp_ticker, opp_raw_ob, opp_market_ref)

        # ── Periodic summary: all tracked bears across all events, every 2 min ──
        if now - self._last_summary_ts >= 120.0:
            self._last_summary_ts = now
            all_bears = list(self.tracked_bears.values())

            # Show near-top-of-book stats for this ticker so we can see why bears do/don't qualify
            diag_lines = []
            for side in ["yes_dollars", "no_dollars"]:
                levels = raw_ob.get(side, [])
                if not levels:
                    continue
                top_c = int(round(float(levels[0][0]) * 100))
                near = [(int(round(float(p[0])*100)), int(float(p[1])))
                        for p in levels if abs(int(round(float(p[0])*100)) - top_c) <= NEAR_TOP_CENTS and float(p[1]) > 0]
                if near:
                    sizes = sorted([s for _, s in near])
                    largest = max(sizes)
                    others = sorted([s for s in sizes if s != largest])
                    med_others = others[len(others)//2] if others else 0
                    threshold = int(ANOMALY_RATIO * med_others) if med_others > 0 else self.min_bear_size
                    top5 = " ".join(f"{s}@{p}c" for p, s in near[:6])
                    diag_lines.append(f"    {ticker} {side}: largest={largest} median_others={med_others} threshold={threshold} | {top5}")

            print(f"\n{'='*90}")
            print(f"  [POLAR BEAR TRACKER] {len(all_bears)} bears across all events")
            print(f"  Config: min_size={self.min_bear_size} | min_disloc={self.min_dislocation_cents}c | anomaly={ANOMALY_RATIO}x")
            if diag_lines:
                print(f"  Current book diagnostics:")
                for dl in diag_lines:
                    print(dl)
            print(f"{'='*90}")
            if all_bears:
                for b in sorted(all_bears, key=lambda x: x.ticker):
                    age = now - b.first_seen
                    disloc_str = f"DISLOCATED {now - b.dislocation_since:.0f}s" if b.dislocation_since else "no disloc"
                    print(f"  {b.ticker:<50} {b.book_side:<12} "
                          f"{b.current_size:>6}/{b.original_size:<6} @ {b.offer_price_cents:>3}c "
                          f"chip={b.chip_pct*100:4.0f}% | age={age:5.0f}s | {disloc_str}")
                pb_pos_str = ", ".join(f"{k}:{v:+d}" for k, v in self._pb_position.items() if v != 0)
                if pb_pos_str:
                    print(f"  PB Positions: {pb_pos_str}")
            else:
                print(f"  No bears qualify.")
            print(f"{'='*90}\n")

        # ── Evaluate tracked bears for fire conditions ──
        desired_quotes = []

        for bear_id, bear in list(self.tracked_bears.items()):
            if not bear.ticker.startswith(event_base):
                continue

            # Check per-ticker cooldown
            if bear.ticker in self.fire_cooldowns:
                if now - self.fire_cooldowns[bear.ticker] < FIRE_COOLDOWN_SECS:
                    continue

            # Determine team for Poly alignment
            bear_parts = bear.ticker.split("-")
            bear_team = bear_parts[-1] if len(bear_parts) >= 3 else team_suffix

            # ── Condition 1: Polymarket dislocation ──
            poly_bid = self._get_poly_bid(event_base, bear_team, bear.buy_side)
            if poly_bid is None:
                continue

            dislocation = poly_bid - bear.offer_price_cents

            if dislocation < self.min_dislocation_cents:
                # No dislocation — reset timer
                if bear.dislocation_since is not None:
                    bear.dislocation_since = None
                continue

            # ── Condition 2: Sustained dislocation ──
            if bear.dislocation_since is None:
                bear.dislocation_since = now
                # Show team, side, and action so alignment can be verified
                action = f"BUY {bear.buy_side.upper()} {bear_team} @ {bear.offer_price_cents}c"
                log.info(
                    f"[POLAR BEAR] Dislocation START: {bear.bear_id} | "
                    f"{action} | poly_{bear.buy_side}_bid={poly_bid:.0f}c "
                    f"disloc={dislocation:.0f}c | Waiting {DISLOCATION_SUSTAIN_SECS}s..."
                )
                continue

            sustained = now - bear.dislocation_since
            if sustained < DISLOCATION_SUSTAIN_SECS:
                continue

            # ── Condition 3: Chip confirmation ──
            if bear.chip_pct < CHIP_THRESHOLD_PCT:
                log.debug(
                    f"[POLAR BEAR] {bear.bear_id} | Sustained {sustained:.0f}s but "
                    f"only {bear.chip_pct*100:.0f}% chipped (need {CHIP_THRESHOLD_PCT*100:.0f}%)"
                )
                continue

            # ═══════════════════════════════════════════
            #  ALL CONDITIONS MET — FIRE
            # ═══════════════════════════════════════════
            # Use polar-bear-only position tracking (independent of arber/quoter)
            pb_pos = self._pb_position.get(event_base, 0)
            pos_room = max(0, self.config.max_position - abs(pb_pos))

            fire_size = min(
                bear.current_size,
                self.config.max_fire_size,
                pos_room,
            )

            if fire_size <= 0:
                continue

            arb_theo = poly_bid  # Our "theo" is the Polymarket bid

            log.warning(
                f"[POLAR BEAR FIRE] {bear.ticker} | "
                f"Buy {bear.buy_side.upper()} {fire_size}x @ {bear.offer_price_cents}c | "
                f"Poly bid: {poly_bid:.0f}c | Disloc: {dislocation:.0f}c sustained {sustained:.0f}s | "
                f"Chipped: {bear.chip_pct*100:.0f}% "
                f"({bear.original_size} -> {bear.current_size})"
            )

            desired_quotes.append({
                "side": QuoteSide.BID,
                "ticker": bear.ticker,
                "kalshi_side": bear.buy_side,
                "size": fire_size,
                "limit_cents": bear.offer_price_cents,
                "time_in_force": "immediate_or_cancel",
                "flag": "POLAR_BEAR",
                "_raw_theo": arb_theo,
                "_adj_theo": arb_theo,
                "_arb_context": (
                    f"[POLAR BEAR] {bear.bear_id} | "
                    f"{fire_size}x @ {bear.offer_price_cents}c vs Poly {poly_bid:.0f}c | "
                    f"Chipped {bear.chip_pct*100:.0f}%"
                ),
            })

            # Update PB-specific position tracking
            direction = 1 if bear.buy_side == "yes" else -1
            # If bear is on opponent ticker, direction flips relative to our event position
            if bear.ticker != ticker:
                direction = -direction
            self._pb_position[event_base] = pb_pos + (direction * fire_size)

            # Cooldown and cleanup
            self.fire_cooldowns[bear.ticker] = now
            self.tracked_bears.pop(bear_id, None)

        return desired_quotes
