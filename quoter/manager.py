import asyncio
import logging
import os
from typing import Callable, Dict, Any, List

from framework_config import load_quoter_configs_from_csv, amend_enabled

# Touch this file (anchored to this script's directory, so cwd-independent)
# to pause the periodic orphan-sweep — lets you place manual orders without
# them being cancelled. Remove the file to resume.
MANUAL_TRADE_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".manual_trade")
from bot import QuoterBot
from arber_bot import EsportsArberBot
from momentum_bot import MomentumBot
from map_arber_bot import EsportsMapArberBot
from taker_bot import SeriesTakerBot
from mma_arber_bot import MMAArberBot
from polar_bear_bot import PolarBearBot
from hedge_bot import HedgeBot
from execution import QuoterExecutionEngine
from models import ActiveQuote

# Ticker alias layer — synthetic events route via alias on the wire. The
# `/portfolio/orders` shutdown sweep needs to know BOTH the synthetic name
# (what bot.config.ticker holds) AND the real name (what Kalshi tags the
# order with) to correctly identify our resting orders.
import ticker_aliases
import write_gate

log = logging.getLogger(__name__)

# ── TENNIS / SHARD 3 ──────────────────────────────────────────────────────────
# Order READS are shard-scoped and DEFAULT TO SHARD 0. Verified 26AUG26:
#   GET /portfolio/orders (no index)      -> 6 orders, all exchange_index=0
#   GET /portfolio/orders exchange_index=0 -> the same 6
#   GET /portfolio/orders exchange_index=3 -> 0
# So a tennis quoter that omits the parameter sees NONE of its own resting orders,
# concludes it has nothing working, and reposts every cycle - an unbounded duplicate
# loop that the position caps cannot catch, because the manager's view of its own
# exposure comes from this very read. Every order read below passes it explicitly.
TENNIS_SHARD = 3



def _quote_signature(side, kalshi_side, ticker, limit_cents, size):
    """Stable comparable key for resting-vs-desired quote diff.

    Two quotes with the same signature represent the same intent and don't
    need to be cancelled+re-posted. Mixing QuoteSide enum and string forms
    is normalized via `.value`.
    """
    side_str = side.value if hasattr(side, "value") else str(side)
    return (str(ticker), str(kalshi_side).lower(), side_str,
            int(limit_cents), int(size))


def _diff_quotes(existing, desired):
    """Compute the minimum cancel/post diff between resting quotes and desired.

    Args:
      existing: List[ActiveQuote] currently resting for this bot.
      desired:  List[dict] of desired quote payloads from bot.evaluate().

    Returns: (kept_aqs, cancel_aqs, post_qs)
      kept_aqs:   ActiveQuotes whose signature matches a desired quote (no-op).
      cancel_aqs: ActiveQuotes with no matching desired (must be cancelled).
      post_qs:    desired quotes with no matching resting twin (must be POSTed).

    Handles duplicates correctly via multi-set counting: if 2 resting orders
    and 3 desired share one signature, 2 are kept and 1 is posted.
    """
    existing_by_sig: Dict[tuple, List] = {}
    for aq in existing:
        key = _quote_signature(aq.side, aq.kalshi_side, aq.ticker,
                               aq.limit_cents, aq.size)
        existing_by_sig.setdefault(key, []).append(aq)

    desired_with_sig: List[tuple] = []
    desired_count: Dict[tuple, int] = {}
    for q in desired:
        key = _quote_signature(q.get("side"), q.get("kalshi_side", "yes"),
                               q.get("ticker", ""), q.get("limit_cents", 0),
                               q.get("size", 0))
        desired_with_sig.append((key, q))
        desired_count[key] = desired_count.get(key, 0) + 1

    kept = []
    cancel = []
    for key, aq_list in existing_by_sig.items():
        keep_n = min(desired_count.get(key, 0), len(aq_list))
        kept.extend(aq_list[:keep_n])
        cancel.extend(aq_list[keep_n:])

    post = []
    remaining = {k: len(v) for k, v in existing_by_sig.items()}
    for key, q in desired_with_sig:
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
        else:
            post.append(q)

    return kept, cancel, post


def _diff_quotes_with_amend(existing, desired):
    """Compute kept/amend/cancel/post diff with level-based pairing.

    Same intent as `_diff_quotes` but recognizes that a price OR size change
    at the same (ticker, side, kalshi_side, level) slot can be expressed as
    a single amend instead of a cancel+post pair. Amend preserves order_id
    and queue position; cancel+post does not.

    Args:
      existing: List[ActiveQuote] currently resting for this bot.
      desired:  List[dict] of desired quote payloads from bot.evaluate().

    Returns: (kept_aqs, amend_pairs, cancel_aqs, post_qs)
      kept_aqs:    ActiveQuotes whose full (price+size) signature matches a
                   desired quote — no wire action.
      amend_pairs: List of (ActiveQuote, desired_dict) tuples. Same
                   (ticker, side, kalshi_side) group, paired by level
                   (position after price-sorting). Differ in price and/or
                   size; sent as `amend_order` calls by the execution layer.
      cancel_aqs:  ActiveQuotes at levels with no desired counterpart.
      post_qs:     desired quotes at levels with no existing counterpart.

    Algorithm — two phases:
      Phase A: multiset-match by full signature (same as _diff_quotes' kept
               logic). Identical quotes → kept; the rest carry over.
      Phase B: group remainder by (ticker, side, kalshi_side). Within each
               group, sort by price — BID descending, OFFER ascending — so
               level 0 = closest to top of book on both sides. Zip-pair by
               position. Pairs → amend_pairs. Excess existing → cancel.
               Excess desired → post.

    Sort direction matters: pairing best-priced existing with best-priced
    desired produces the intuitive reprice behavior. A move from [55, 52]
    BIDs to [56, 53] BIDs yields 2 amends (level-by-level), not 4 wires.
    """
    from collections import defaultdict

    # ─ Phase A: exact-signature multiset → kept ─
    existing_by_sig: Dict[tuple, List] = defaultdict(list)
    for aq in existing:
        sig = _quote_signature(aq.side, aq.kalshi_side, aq.ticker,
                               aq.limit_cents, aq.size)
        existing_by_sig[sig].append(aq)

    desired_by_sig: Dict[tuple, List] = defaultdict(list)
    for q in desired:
        sig = _quote_signature(q.get("side"), q.get("kalshi_side", "yes"),
                               q.get("ticker", ""), q.get("limit_cents", 0),
                               q.get("size", 0))
        desired_by_sig[sig].append(q)

    kept = []
    existing_remaining = []
    desired_remaining = []

    for sig in set(existing_by_sig.keys()) | set(desired_by_sig.keys()):
        e_list = existing_by_sig.get(sig, [])
        d_list = desired_by_sig.get(sig, [])
        n_match = min(len(e_list), len(d_list))
        kept.extend(e_list[:n_match])
        existing_remaining.extend(e_list[n_match:])
        desired_remaining.extend(d_list[n_match:])

    # ─ Phase B: level-pair remainder within (ticker, side, kalshi_side) ─
    existing_grouped: Dict[tuple, List] = defaultdict(list)
    for aq in existing_remaining:
        side_str = aq.side.value if hasattr(aq.side, "value") else str(aq.side)
        key = (str(aq.ticker), str(aq.kalshi_side).lower(), side_str.lower())
        existing_grouped[key].append(aq)

    desired_grouped: Dict[tuple, List] = defaultdict(list)
    for q in desired_remaining:
        side = q.get("side")
        side_str = side.value if hasattr(side, "value") else str(side)
        key = (str(q.get("ticker", "")),
               str(q.get("kalshi_side", "yes")).lower(),
               side_str.lower())
        desired_grouped[key].append(q)

    amend_pairs = []
    cancel = []
    post = []

    for key in set(existing_grouped.keys()) | set(desired_grouped.keys()):
        _ticker, _kside, side_str = key
        is_bid = (side_str == "bid")
        e_list = sorted(existing_grouped.get(key, []),
                        key=lambda aq: aq.limit_cents,
                        reverse=is_bid)
        d_list = sorted(desired_grouped.get(key, []),
                        key=lambda q: q.get("limit_cents", 0),
                        reverse=is_bid)

        n_pairs = min(len(e_list), len(d_list))
        for i in range(n_pairs):
            amend_pairs.append((e_list[i], d_list[i]))
        cancel.extend(e_list[n_pairs:])
        post.extend(d_list[n_pairs:])

    return kept, amend_pairs, cancel, post


def _partition_ioc(desired):
    """Split desired quote payloads into (ioc, resting).

    IOC (immediate-or-cancel / taker) orders NEVER rest on the book — they
    fill-or-cancel on arrival. There is therefore no live order to amend or to
    diff against: amending a dead IOC 404s and drops the caller into a
    cancel+post churn that executes NOTHING (the 2026-07-24 amend-IOC incident
    that silently killed taker fills for ~a month). IOC quotes must be
    fresh-POSTed every tick and kept entirely out of the amend/diff path.
    """
    ioc, resting = [], []
    for q in desired:
        if q.get("time_in_force") == "immediate_or_cancel":
            ioc.append(q)
        else:
            resting.append(q)
    return ioc, resting


def _plan_bot_orders(existing, desired, amend_on):
    """Pure per-bot tick plan. RESTING quotes go through the normal
    amend-or-cancel+post diff; IOC quotes bypass it entirely and are always
    appended to `post` (fresh-post-only: never amended, never kept, never
    diff-cancelled).

    Returns (kept_aqs, amend_pairs, cancel_aqs, post_qs). IOC quotes appear ONLY
    in post_qs. Callers MUST also keep IOC posts out of active_quotes_by_bot
    (see the registration guard in run_tick) so they never become `existing`
    and get amended on a later tick.
    """
    ioc, resting = _partition_ioc(desired)
    if amend_on:
        kept, amend_pairs, cancel, post = _diff_quotes_with_amend(existing, resting)
    else:
        kept, cancel, post = _diff_quotes(existing, resting)
        amend_pairs = []
    return kept, amend_pairs, cancel, list(post) + ioc


def _find_self_match_targets(arber_ioc_payloads, active_quotes_by_bot):
    """Identify resting maker quotes that would self-match against pending IOCs.

    Kalshi semantics (every order is BUY): a resting BUY on kalshi_side X
    crosses an incoming IOC BUY on kalshi_side Y when the two are on opposite
    sides AND their prices sum to >= 100 cents. Same-side resting+IOC are
    both buys on the same book and can never match each other.

    Math: BUY YES @ Py + BUY NO @ Pn costs $1 in total when Py + Pn = 100
    (guaranteed wash). When Py + Pn > 100, the two sides cross — Kalshi
    would match them and waste an IOC slot on our own resting maker.

    Replaces the broken inline guard in run_tick (2026-06-19 fix). The
    previous code required `aq.kalshi_side == ioc_side` in the outer
    condition AND `aq.kalshi_side != ioc_side` in the inner — logically
    impossible, so `self_match_targets` was always empty. Result: the
    pre-emptive cancel never fired; we relied entirely on Kalshi's
    server-side STP (`self_trade_prevention_type=taker_at_cross`) to
    reject crossing IOCs. The STP rejection silently dropped IOCs that
    we should have been clearing space for.

    Args:
      arber_ioc_payloads: list of pending IOC payload dicts (each with
        keys: ticker, kalshi_side, limit_cents).
      active_quotes_by_bot: dict[bot_id, list[ActiveQuote]] — current
        resting maker quotes across all bots.

    Returns:
      list[ActiveQuote] — every resting quote that crosses at least one
      pending IOC. Caller must cancel these before submitting the IOCs.
      Order is deterministic by (ioc index, bot iteration); duplicates
      possible if a single resting crosses multiple IOCs — caller should
      dedup by order_id.
    """
    self_match_targets = []
    for ioc_q in arber_ioc_payloads:
        ioc_ticker = ioc_q["ticker"]
        ioc_side = ioc_q.get("kalshi_side", "yes")
        ioc_price = ioc_q.get("limit_cents", 0)
        for bot_quotes in active_quotes_by_bot.values():
            for aq in bot_quotes:
                if aq.ticker != ioc_ticker:
                    continue
                if aq.kalshi_side == ioc_side:
                    # Same-side BUY + BUY can never cross (both in same book).
                    continue
                if aq.limit_cents + ioc_price >= 100:
                    self_match_targets.append(aq)
    return self_match_targets


def _failed_amend_should_repost(kind: str) -> bool:
    """Decide whether a failed-amend fallback should repost immediately.

    (2026-06-28 P0-2) Only a hard "reject" — the amend definitively did NOT
    apply (order already gone, or rejected before processing) — is safe to
    repost now: the order's resting state is known, so cancel+post carries the
    same risk profile as ordinary repricing.

    Anything else defers (returns False): an "ambiguous" failure (timeout / 5xx
    / network, where the amend MIGHT have applied and the order may now be live
    at the NEW price) — and, defensively, any unrecognized kind — must NOT
    blind-repost, or we double the resting size against a possibly-live order.
    Deferred pairs are still cancelled; the bot reposts on a later tick once
    _verify_pending_cancels confirms the cancel and evicts the AQ.
    """
    return kind == "reject"


class MarketManager:
    """
    Centralized coordinator that initializes QuoterBots, feeds them market state,
    aggregates their quote requests, and dispatches them efficiently through 
    the Global Execution Engine and isolated Theo Generators.
    """
    def __init__(self, kalshi_client: Any, theo_generator: Any, trading_enabled: bool = True):
        self.exec_engine = QuoterExecutionEngine(kalshi_client, trading_enabled)
        self.theo_generator = theo_generator
        self.bots: List[QuoterBot] = []
        self.client_ref = kalshi_client
        self.trading_enabled = trading_enabled
        self.active_quotes_by_bot: Dict[str, List[ActiveQuote]] = {}
        # Pending-cancel verification: order_id -> {"attempts": int, "first_attempt_ts": float, "source_aq": ActiveQuote}
        # Each tick, we re-check Kalshi to confirm the order really did die. Up to MAX_CANCEL_ATTEMPTS retries.
        # Manual orders never enter this dict, so they are never touched.
        self._pending_cancels: Dict[str, Dict[str, Any]] = {}
        self.MAX_CANCEL_ATTEMPTS = 5

        # Periodic reconciliation: every RECONCILIATION_INTERVAL_SEC, fetch ALL our resting
        # orders from Kalshi and cancel any not in our tracking. Catches orphans from any
        # source (race conditions, restarts with stale state, etc.).
        # Defaults to OFF when trading is disabled (dry-run): otherwise the dry-run process
        # treats the production process's resting orders as orphans and cancels them, since
        # both share the same Kalshi account. Override after construction if needed.
        self.reconciliation_enabled = trading_enabled
        self.RECONCILIATION_INTERVAL_SEC = 60
        # Don't cancel orphans younger than this — buffer for our own placement→tracking lag
        self.RECONCILIATION_MIN_AGE_SEC = 30
        self._last_reconciliation_ts = 0.0

        # Per-ticker routing registries. Initialized empty here so external
        # consumers (e.g. RouterBookView constructed in run.py BEFORE
        # load_bots_from_csv runs) can hold a stable reference. Populated
        # in load_bots_from_csv / reload_bots_from_csv via clear()+update
        # so the reference stays valid across hot reloads.
        self.data_source_by_ticker: Dict[str, str] = {}
        self.book_source_by_ticker: Dict[str, str] = {}
        self.disable_forfeit_by_ticker: Dict[str, bool] = {}

    def shutdown_sync(self) -> None:
        """
        Gracefully executes synchronously to bypass event loop cancellation errors.
        """
        if not self.trading_enabled:
            return

        # Best-effort: any pending cancels left mid-retry are also wiped via the
        # full /portfolio/orders sweep below, but we drop the tracking dict so
        # the next run starts clean.
        if self._pending_cancels:
            log.warning(f"MANAGER SHUTDOWN | {len(self._pending_cancels)} pending-cancel verifications dropped (full sweep follows).")
            self._pending_cancels.clear()

        log.warning("MANAGER SHUTDOWN | Initiating Hard Order Wipe...")
        # Build the comparison set with BOTH synthetic and real ticker names.
        # bot.config.ticker is the synthetic (internal) name; orders Kalshi
        # reports via /portfolio/orders are tagged with the real wire ticker.
        # Without the real-name expansion the shutdown skips aliased orders
        # (FALFURIA/LEVPR/etc.) — they survive process exit instead of being
        # cancelled, leaving live exposure with no bot watching.
        active_tickers = set()
        for bot in self.bots:
            synth = bot.config.ticker
            active_tickers.add(synth)
            real = ticker_aliases.to_kalshi(synth)
            if real:
                active_tickers.add(real)

        if not active_tickers:
            return

        try:
            path = "/trade-api/v2/portfolio/orders"
            resp = self.client_ref._get(path, params={"status": "resting", "limit": 1000,
                                                      "exchange_index": TENNIS_SHARD})

            resting_orders = resp.get("orders", [])
            orphans_to_delete = []

            # Since we switched mapping to Bot ID inherently, we don't have bot states exposed here cleanly.
            # But the orchestrator natively understands active_tickers dynamically!
            # For teardowns, we simply grab ALL open IDs mapping to valid tickers.
            for o in resting_orders:
                if o.get("ticker") in active_tickers:
                    orphans_to_delete.append(o.get("order_id"))
                    
            if orphans_to_delete:
                log.warning(f"MANAGER SHUTDOWN | Tracing {len(orphans_to_delete)} physical resting orphans. Dispatching sync obliteration batch...")
                self._execute_sync_cancel(orphans_to_delete)
                log.info("MANAGER SHUTDOWN | Wipe successful.")
            else:
                log.info("MANAGER SHUTDOWN | Clean teardown natively. No orphans found.")
        except Exception as e:
            log.error(f"MANAGER SHUTDOWN | CRITICAL ERROR during teardown sequence: {e}")

    def _execute_sync_cancel(self, order_ids: List[str]) -> None:
        """Batch-cancel via the canonical V2 endpoint through kalshi_client.

        Bug fixed 2026-06-27: the prior implementation hit
        `/trade-api/v2/portfolio/orders/batched` with body `{"ids": [...]}`,
        which is the deprecated legacy shape. Kalshi returns 410
        `deprecated_v1_order_endpoint`, so neither shutdown sweeps nor
        reconciliation orphan sweeps were actually cancelling anything.
        Symptom: 16+ identical orphan IDs persisting across reconciliation
        cycles after run.py restarts. Routing through `cancel_order_batch`
        uses the v2 path `/trade-api/v2/portfolio/events/orders/batched`
        with body `{"orders": [{"order_id": ...}, ...]}` (auto-chunked to
        20-per-batch internally) and respects the write-token limiter.
        """
        if not order_ids:
            return
        try:
            resp = self.client_ref.cancel_order_batch(order_ids)
        except Exception as e:
            log.error("Sync cancel batch failed (network/exception): %s", e)
            return
        # Per-order error reporting — V2 returns {orders:[{order_id,error}]}.
        results = (resp or {}).get("orders", []) if isinstance(resp, dict) else []
        failed = [r for r in results if r.get("error")]
        if failed:
            log.error("Sync cancel: %d of %d order(s) failed; first error: %s",
                      len(failed), len(order_ids), failed[0].get("error"))

    def _expand_wildcard_configs(self, configs: List[Any]) -> List[Any]:
        import copy
        expanded = []
        for conf in configs:
            # Detect wildcards natively (e.g. KXMLS, KXNHL, KXLOLGAME) avoiding hyphens!
            if conf.ticker.startswith("KX") and "-" not in conf.ticker:
                log.info(f"MANAGER FAT PIPE | Detonating Wildcard Anchor for Series: {conf.ticker}...")
                try:
                    resp = self.client_ref._get(f"/trade-api/v2/markets?series_ticker={conf.ticker}&status=active&limit=500")
                    active_markets = resp.get("markets", [])
                    log.info(f"WILDCARD BURST | Found {len(active_markets)} active markets across {conf.ticker}!")
                    
                    for mk in active_markets:
                        cloned_conf = copy.deepcopy(conf)
                        cloned_conf.ticker = mk["ticker"]
                        cloned_conf.market_id = f"{conf.market_id}_{mk['ticker']}"
                        expanded.append(cloned_conf)
                except Exception as e:
                    log.error(f"Failed to expand native wildcard cluster {conf.ticker}: {e}")
            else:
                expanded.append(conf)
        return expanded

    def load_bots_from_csv(self, filepath: str) -> None:
        """
        Loads strategy configs from a CSV and instantiates QuoterBots.
        """
        raw_configs = load_quoter_configs_from_csv(filepath)
        configs = self._expand_wildcard_configs(raw_configs)
        
        for conf in configs:
            if conf.stop_quoting:
                continue
                
            if conf.execution_type == "arber":
                bot = EsportsArberBot(config=conf)
            elif conf.execution_type == "map_arber":
                bot = EsportsMapArberBot(config=conf)
            elif conf.execution_type == "taker":
                bot = SeriesTakerBot(config=conf)
            elif conf.execution_type == "mma_arber":
                bot = MMAArberBot(config=conf)
            elif conf.execution_type == "polar_bear":
                bot = PolarBearBot(config=conf)
            elif conf.execution_type == "momentum":
                bot = MomentumBot(config=conf)
            elif conf.execution_type == "hedger":
                bot = HedgeBot(config=conf)
            else:
                bot = QuoterBot(config=conf)
            self.bots.append(bot)

        # Map configurations downward into the Theo Generation Pipeline.
        # Only quoter configs — arbers/hedgers bypass theos entirely and must
        # never influence the theo model routing.
        quoter_configs = [c for c in configs if c.execution_type not in ("arber", "map_arber", "mma_arber", "polar_bear", "hedger", "taker", "momentum")]
        if hasattr(self.theo_generator, 'configs'):
            self.theo_generator.configs = {c.ticker: c for c in quoter_configs}
        if hasattr(self.theo_generator, 'position_adjuster'):
            self.theo_generator.position_adjuster.configs = {c.ticker: c for c in quoter_configs}

        # Per-ticker data_source registry (Tier 1 CS work). Multiple configs may share a ticker
        # (e.g. ARB + QUOTE rows) — we expect them to agree on data_source per the populator.
        # Take the first non-default we see; warn if subsequent configs disagree.
        # Mutate-in-place rather than reassigning so external references
        # (e.g. RouterBookView holding a pointer to book_source_by_ticker)
        # observe updated routing after this initial load.
        self.data_source_by_ticker.clear()
        self.book_source_by_ticker.clear()
        self.disable_forfeit_by_ticker.clear()
        for c in configs:
            ds = (c.data_source or "kalshi").strip().lower()
            existing = self.data_source_by_ticker.get(c.ticker)
            if existing is None:
                self.data_source_by_ticker[c.ticker] = ds
            elif existing != ds:
                log.warning("[DATA_SOURCE] %s has conflicting values: %s vs %s — keeping %s",
                            c.ticker, existing, ds, existing)
            bs = (c.book_source or "ws").strip().lower()
            existing_bs = self.book_source_by_ticker.get(c.ticker)
            if existing_bs is None:
                self.book_source_by_ticker[c.ticker] = bs
            elif existing_bs != bs:
                log.warning("[BOOK_SOURCE] %s has conflicting values: %s vs %s — keeping %s",
                            c.ticker, existing_bs, bs, existing_bs)
            # If any config for a ticker disables the forfeit check, treat the ticker as disabled.
            if c.disable_forfeit_check:
                self.disable_forfeit_by_ticker[c.ticker] = True

        # Override: map_arber trades the Kalshi map leg directly, so the parent
        # series's child map tickers must NOT be substituted from Poly (substituted
        # Poly prices would diverge from where we'd execute). Force kalshi.
        for c in configs:
            if c.execution_type == "map_arber":
                if self.data_source_by_ticker.get(c.ticker) != "kalshi":
                    log.info("[DATA_SOURCE] %s: forcing kalshi for map_arber execution leg",
                             c.ticker)
                self.data_source_by_ticker[c.ticker] = "kalshi"

        log.info("Loaded %d QuoterBots natively from %s", len(configs), filepath)

    def reload_bots_from_csv(self, filepath: str) -> None:
        """
        Dynamically cross-checks active bots against the newly populated CSV organically.
        Identifies orphaned/closed markets, executes literal native batch deletions for their limits,
        and instantiates any new QuoterBots without tearing down structurally identically matches.
        """
        raw_new_configs = load_quoter_configs_from_csv(filepath)
        new_configs = self._expand_wildcard_configs(raw_new_configs)
        
        active_map = {f"{b.config.market_id}_{b.config.ticker}": b for b in self.bots}
        new_map = {f"{c.market_id}_{c.ticker}": c for c in new_configs if not c.stop_quoting}
        
        orphans_str = []
        orphaned_ids = []
        
        for u_key, bot in list(active_map.items()):
            if u_key not in new_map:
                bot.active = False
                bot_id = str(id(bot))

                # Snag every physical orphaned order UUID linked securely to this bot
                orphaned_aqs_this_bot = []
                if bot_id in self.active_quotes_by_bot:
                    for aq in self.active_quotes_by_bot[bot_id]:
                        if getattr(aq, "order_id", None):
                            orphaned_ids.append(aq.order_id)
                            orphaned_aqs_this_bot.append(aq)
                    del self.active_quotes_by_bot[bot_id]

                # Track these for verification — they may need retries if cancel races with fills
                self._track_pending_cancels(orphaned_aqs_this_bot)

                self.bots.remove(bot)
                orphans_str.append(u_key)

        if orphaned_ids:
            log.warning(f"MANAGER HOT RELOAD | Tracing {len(orphaned_ids)} resting limits for eradicated markets. Engaged synchronous atomic wipe...")
            self._execute_sync_cancel(orphaned_ids)
            
        new_spins = []
        for u_key, config in new_map.items():
            if u_key not in active_map:
                if config.execution_type == "arber":
                    bot = EsportsArberBot(config=config)
                elif config.execution_type == "map_arber":
                    bot = EsportsMapArberBot(config=config)
                elif config.execution_type == "taker":
                    bot = SeriesTakerBot(config=config)
                elif config.execution_type == "mma_arber":
                    bot = MMAArberBot(config=config)
                elif config.execution_type == "polar_bear":
                    bot = PolarBearBot(config=config)
                elif config.execution_type == "momentum":
                    bot = MomentumBot(config=config)
                elif config.execution_type == "hedger":
                    bot = HedgeBot(config=config)
                else:
                    bot = QuoterBot(config=config)
                self.bots.append(bot)
                new_spins.append(u_key)
            else:
                # HOT RELOAD FIX: Organically swap in the new config payload if the user altered edge constants!
                active_map[u_key].config = config

        # Rebake Theo mappings — quoter configs only.
        _NON_THEO_TYPES = ("arber", "map_arber", "mma_arber", "polar_bear", "hedger", "taker", "momentum")
        quoter_bots = [b for b in self.bots if b.config.execution_type not in _NON_THEO_TYPES]

        if hasattr(self.theo_generator, 'configs'):
            self.theo_generator.configs = {b.config.ticker: b.config for b in quoter_bots}
        if hasattr(self.theo_generator, 'position_adjuster'):
            self.theo_generator.position_adjuster.configs = {b.config.ticker: b.config for b in quoter_bots}

        if orphans_str or new_spins:
            quoter_bots = [b for b in self.bots if b.config.execution_type not in _NON_THEO_TYPES]
            if hasattr(self.theo_generator, 'configs'):
                self.theo_generator.configs = {b.config.ticker: b.config for b in quoter_bots}
            if hasattr(self.theo_generator, 'position_adjuster'):
                self.theo_generator.position_adjuster.configs = {b.config.ticker: b.config for b in quoter_bots}

            log.info(f"--- HOT RELOAD ENGAGED --- | Swept: {len(orphans_str)} markets | Bound: {len(new_spins)} new markets.")

        # Rebuild per-ticker registries from the live config set so new tickers
        # bound by hot-reload inherit their disable_forfeit / data_source flags.
        # Without this, live_series_model reads stale dicts and treats new
        # tier-1 CS maps as forfeited (50/50 trips the Poly check).
        # Mutate-in-place rather than reassigning so external references
        # (e.g. RouterBookView holding a pointer to book_source_by_ticker)
        # observe the updated routing after hot-reload.
        self.data_source_by_ticker.clear()
        self.book_source_by_ticker.clear()
        self.disable_forfeit_by_ticker.clear()
        for c in new_configs:
            ds = (c.data_source or "kalshi").strip().lower()
            existing = self.data_source_by_ticker.get(c.ticker)
            if existing is None:
                self.data_source_by_ticker[c.ticker] = ds
            elif existing != ds:
                log.warning("[DATA_SOURCE] %s has conflicting values: %s vs %s — keeping %s",
                            c.ticker, existing, ds, existing)
            bs = (c.book_source or "ws").strip().lower()
            existing_bs = self.book_source_by_ticker.get(c.ticker)
            if existing_bs is None:
                self.book_source_by_ticker[c.ticker] = bs
            elif existing_bs != bs:
                log.warning("[BOOK_SOURCE] %s has conflicting values: %s vs %s — keeping %s",
                            c.ticker, existing_bs, bs, existing_bs)
            if c.disable_forfeit_check:
                self.disable_forfeit_by_ticker[c.ticker] = True

        # Same map_arber data_source override applied after rebuild (mirror of __init__).
        for c in new_configs:
            if c.execution_type == "map_arber":
                self.data_source_by_ticker[c.ticker] = "kalshi"

    def reload_probabilities(self) -> None:
        """Re-read esports_probabilities.csv and push fresh values into running bots
        and the LiveSeriesTheoGenerator cache. Called on esports_probabilities.csv mtime change."""
        try:
            from hedge_engine import load_probabilities
            new_probs = load_probabilities()
        except Exception as e:
            log.error(f"[PROB RELOAD] failed to read esports_probabilities.csv: {e}")
            return

        if hasattr(self.theo_generator, "models"):
            live_series = self.theo_generator.models.get("live_series")
            if live_series is not None:
                live_series._prob_cache = new_probs

        refreshed = 0
        for bot in self.bots:
            if hasattr(bot, "refresh_probabilities"):
                try:
                    if bot.refresh_probabilities(new_probs):
                        refreshed += 1
                except Exception as e:
                    log.error(f"[PROB RELOAD] {getattr(bot.config,'ticker','?')}: {e}")

        log.info(f"--- PROB RELOAD --- | {len(new_probs)} entries in CSV | refreshed {refreshed} bot(s)")

    # Note: 'top_level_bids' and 'top_level_offers' could be dictionaries keyed by ticker if preferred.
    async def _verify_pending_cancels(self) -> None:
        """Confirm each previously-attempted cancel actually killed the order.

        Polls /portfolio/orders/{oid} for every entry in self._pending_cancels.
        - status canceled/executed/expired → drop from pending (order is dead)
        - status resting (or unknown) → re-fire cancel; bump attempt count
        - attempts >= MAX_CANCEL_ATTEMPTS → log loud orphan error and drop

        Manual orders are never tracked here so they're never touched.
        """
        if not self._pending_cancels:
            return
        import time
        to_drop: List[str] = []
        to_retry: List[Any] = []
        for oid, info in list(self._pending_cancels.items()):
            try:
                d = await asyncio.to_thread(
                    lambda o=oid: self.client_ref._get(f"/trade-api/v2/portfolio/orders/{o}",
                                                       params={"exchange_index": TENNIS_SHARD})
                )
                o = (d or {}).get("order", {}) if d else {}
                status = (o.get("status") or "").lower()
            except Exception as e:
                log.debug(f"PENDING CANCEL | order lookup failed for {oid}: {e}")
                status = ""
            if status in ("canceled", "executed", "expired"):
                # Order is conclusively dead on Kalshi (whether by our cancel,
                # an external fill, or expiry). Remove it from local tracking
                # so the orderbook scrubber stops subtracting its volume from
                # the book — otherwise we'd phantom-suppress real liquidity at
                # that price level for the next many ticks.
                src = info.get("source_aq")
                if src is not None:
                    cid = getattr(src, "client_order_id", None)
                    if cid:
                        self.exec_engine.active_quotes.pop(cid, None)
                    # 2026-06-28 orphan-prevention follow-up: now that failed
                    # amends keep their AQ in active_quotes_by_bot (so cycle
                    # interruption can't orphan it), we MUST also remove the
                    # AQ from active_quotes_by_bot once its cancel verifies.
                    # Without this, the dead AQ keeps showing up in the bot's
                    # `existing` set, gets re-amended next cycle, 404s, and
                    # loops forever.
                    for bot_id, aq_list in self.active_quotes_by_bot.items():
                        if src in aq_list:
                            aq_list.remove(src)
                            break
                to_drop.append(oid)
                continue
            if info["attempts"] >= self.MAX_CANCEL_ATTEMPTS:
                age = time.time() - info["first_attempt_ts"]
                log.error(f"--- ORPHANED ORDER --- {oid} survived {self.MAX_CANCEL_ATTEMPTS} cancel attempts. "
                          f"Status={status!r}. {age:.1f}s old. Handing back to the orphan sweep.")
                # 2026-06-28 P0-1: when we give up cancelling a still-resting
                # order, we MUST also evict its AQ from active_quotes_by_bot
                # (and exec_engine.active_quotes). Otherwise its order_id stays
                # in `known_ids` (built from active_quotes_by_bot ∪
                # _pending_cancels) and _reconcile_orders' ORPHAN SWEEP skips
                # it FOREVER — it rests live at a stale price, invisible to
                # both safety nets, until it fills (the 3d048426 pickoff
                # class). Dropping it here hands it back to the sweep, which
                # uses the working v2 batched cancel as the true last resort.
                # Mirrors the confirmed-dead removal above (642-657).
                src = info.get("source_aq")
                if src is not None:
                    cid = getattr(src, "client_order_id", None)
                    removed_from_b = False
                    if cid:
                        removed_from_b = self.exec_engine.active_quotes.pop(cid, None) is not None
                    removed_from_a = False
                    for bot_id, aq_list in self.active_quotes_by_bot.items():
                        if src in aq_list:
                            aq_list.remove(src)
                            removed_from_a = True
                            break
                    log.warning(
                        f"[P0-1 EVICT] {getattr(src, 'ticker', '?')} order={oid} "
                        f"side={getattr(src, 'kalshi_side', '?')} "
                        f"price={getattr(src, 'limit_cents', '?')}c "
                        f"cid={cid} — cancel exhausted, evicted from tracking "
                        f"(active_quotes_by_bot={removed_from_a}, "
                        f"exec.active_quotes={removed_from_b}); now visible to ORPHAN SWEEP"
                    )
                else:
                    log.warning(
                        f"[P0-1 EVICT] order={oid} — cancel exhausted but no source_aq "
                        f"on the pending-cancel record; nothing to evict from tracking"
                    )
                to_drop.append(oid)
                continue
            info["attempts"] += 1
            to_retry.append(info["source_aq"])
        for oid in to_drop:
            self._pending_cancels.pop(oid, None)
        if to_retry:
            log.warning(f"--- CANCEL RETRY --- Re-issuing cancel on {len(to_retry)} unconfirmed order(s)")
            try:
                await self.exec_engine.cancel_quotes_batch(to_retry)
            except Exception as e:
                log.error(f"PENDING CANCEL | retry batch failed: {e}")

    async def _reconcile_orders(self) -> None:
        """Periodic safety net: GET all our resting orders from Kalshi and cancel any
        not in active_quotes_by_bot or _pending_cancels. Detects orphans from any
        source (silent cancel failures, race conditions, restart with stale state).

        Throttled by RECONCILIATION_INTERVAL_SEC and orphans must be older than
        RECONCILIATION_MIN_AGE_SEC to avoid racing our own freshly-placed orders.

        Set self.reconciliation_enabled = False to skip — useful when you're
        placing manual orders that the bot would otherwise sweep.
        """
        # Reconciliation trace — separate from KALSHI_API_TRACE so enabling the
        # CSV trace doesn't auto-enable run.log spam. Opt in deliberately:
        #   export KALSHI_RECON_TRACE=1   OR   touch kalshi_recon_trace.flag
        import os as _os
        _RECON_TRACE = (
            _os.environ.get("KALSHI_RECON_TRACE", "0") == "1"
            or _os.path.exists("kalshi_recon_trace.flag")
        )
        if not self.reconciliation_enabled:
            if _RECON_TRACE:
                log.warning("RECON-TRACE | early-exit: reconciliation_enabled=False")
            return
        import time
        now = time.time()
        elapsed = now - self._last_reconciliation_ts
        if elapsed < self.RECONCILIATION_INTERVAL_SEC:
            if _RECON_TRACE:
                # Sample 1 in 100 to avoid flooding — but still visible
                if int(now) % 30 == 0:
                    log.info(f"RECON-TRACE | throttled: {elapsed:.1f}s since last (need ≥{self.RECONCILIATION_INTERVAL_SEC}s)")
            return
        if _RECON_TRACE:
            log.warning(f"RECON-TRACE | passed throttle (elapsed={elapsed:.1f}s); proceeding")
        self._last_reconciliation_ts = now

        if os.path.exists(MANUAL_TRADE_FLAG):
            log.warning(f"RECONCILIATION | paused — manual-trade flag present ({MANUAL_TRADE_FLAG}). "
                        f"`rm` the file to resume orphan sweeps.")
            return

        if _RECON_TRACE:
            log.warning("RECON-TRACE | about to GET resting orders from Kalshi")
        try:
            d = await asyncio.to_thread(
                lambda: self.client_ref._get("/trade-api/v2/portfolio/orders",
                                              params={"status": "resting", "limit": 1000,
                                                      "exchange_index": TENNIS_SHARD})
            )
        except Exception as e:
            log.error(f"RECONCILIATION | failed to fetch resting orders: {e}")
            return
        if _RECON_TRACE:
            sample = (str(d)[:160] + "...") if d else repr(d)
            log.warning(f"RECON-TRACE | GET returned: type={type(d).__name__} sample={sample}")
        # kalshi_client._get returns {} on 429/timeout/5xx (any non-200). Without
        # this check we silently treat the fetch failure as "no resting orders"
        # and bail; the throttle ts is already bumped to `now` so the next 60s
        # of attempts also early-bail. The whole reconciliation chain goes dark.
        # Distinguish "legitimate empty response" (has 'orders' key, value is
        # empty list) from "request failed" (empty {} from _get's error path).
        if not isinstance(d, dict) or "orders" not in d:
            log.warning("RECONCILIATION | resting-orders fetch returned no 'orders' key — "
                        "Kalshi API call likely 429/timeout. Skipping cycle; will retry next tick.")
            return

        kalshi_orders = d.get("orders", [])
        if _RECON_TRACE:
            log.warning(f"RECON-TRACE | proceeding with {len(kalshi_orders)} resting orders from Kalshi")

        # Build set of known order_ids from both tracking dicts
        known_ids = set()
        for aq_list in self.active_quotes_by_bot.values():
            for aq in aq_list:
                oid = getattr(aq, "order_id", None)
                if oid:
                    known_ids.add(oid)
        known_ids.update(self._pending_cancels.keys())

        from datetime import datetime
        orphan_ids: List[str] = []
        for o in kalshi_orders:
            oid = o.get("order_id")
            if not oid or oid in known_ids:
                continue
            created_str = o.get("created_time", "") or ""
            try:
                created_ts = datetime.fromisoformat(created_str.replace("Z", "+00:00")).timestamp()
            except Exception:
                created_ts = 0.0
            age = now - created_ts
            if age < self.RECONCILIATION_MIN_AGE_SEC:
                # Too fresh — could be racing our own placement→tracking lag. Skip this round.
                continue
            orphan_ids.append(oid)

        if orphan_ids:
            preview = orphan_ids[:10]
            log.error(f"--- RECONCILIATION ORPHAN SWEEP --- {len(orphan_ids)} orphan(s) "
                      f"on Kalshi but not in our tracking. Cancelling. ids[:10]={preview}"
                      f"{' ...' if len(orphan_ids) > 10 else ''}")
            try:
                # _execute_sync_cancel takes a list of order IDs (strings) and uses requests.delete
                self._execute_sync_cancel(orphan_ids)
            except Exception as e:
                log.error(f"RECONCILIATION | sync cancel batch failed: {e}")
        else:
            log.debug(f"RECONCILIATION | clean — {len(kalshi_orders)} resting orders all tracked")

        # ── Inverse direction: WE think we have an order resting, but Kalshi
        # doesn't list it. Means it was filled (most common), expired, or
        # cancelled externally. Leaving the stale ActiveQuote in place causes
        # the orderbook scrubber (run.py:341-352) to keep subtracting its
        # volume from the snapshot, distorting top-of-book by a level for
        # every subsequent tick. Remove from exec_engine.active_quotes once
        # the entry is old enough to be conclusive (placement_ts older than
        # RECONCILIATION_MIN_AGE_SEC). active_quotes_by_bot can keep stale
        # references; they're only used for cancel-on-reprice, which is a
        # no-op against an order Kalshi already doesn't have.
        kalshi_resting_ids = {o.get("order_id") for o in kalshi_orders if o.get("order_id")}
        cleaned: List[str] = []
        for cid, aq in list(self.exec_engine.active_quotes.items()):
            oid = getattr(aq, "order_id", None)
            placement_ts = getattr(aq, "placement_ts", 0.0) or 0.0
            if not oid:
                continue
            if oid in self._pending_cancels:
                # Already being verified via the cancel retry loop; leave it
                # to that path so we don't double-pop and lose the retry.
                continue
            if oid in kalshi_resting_ids:
                continue
            age = now - placement_ts
            if age < self.RECONCILIATION_MIN_AGE_SEC:
                continue
            self.exec_engine.active_quotes.pop(cid, None)
            cleaned.append(oid)
        if cleaned:
            preview = cleaned[:10]
            log.warning(f"--- RECONCILIATION FILL CLEANUP --- {len(cleaned)} tracked order(s) "
                        f"no longer resting on Kalshi (filled/expired). Removed from local tracking. "
                        f"ids[:10]={preview}{' ...' if len(cleaned) > 10 else ''}")

        # Phase 3: force-refire maker bots whose actual resting count on
        # Kalshi is below their last cached intent. See _refire_stale_makers
        # docstring for rationale.
        refired = self._refire_stale_makers(now, kalshi_resting_ids)
        if refired:
            preview = [f"{t}({a}/{e})" for t, e, a in refired[:10]]
            log.warning(f"--- RECONCILIATION REFIRE --- {len(refired)} maker bot(s) had "
                        f"cached intent but fewer resting orders on Kalshi than expected. "
                        f"Cleared cache → next cycle will refire. tickers[:10]={preview}"
                        f"{' ...' if len(refired) > 10 else ''}")

    def _refire_stale_makers(self, now: float, kalshi_resting_ids: set) -> List[tuple]:
        """Clear `last_desired_quotes` on maker bots whose actual resting count
        on Kalshi is below their cached intent. Forces the next `evaluate()` to
        recompute desired quotes from scratch, so the manager's diff path can
        re-post the missing orders.

        Per-bot, not per-ticker — sibling bots on the same ticker (e.g. L1/L2
        quoter layers) are independent. Incident 2026-06-21 on T1ADKC: L1 bot's
        offer disappeared while L2 and the L1 bid stayed; the per-ticker
        variant skipped REFIRE because the ticker still had resting orders.
        L1's `last_desired_quotes` cache stayed stale → evaluate() kept
        returning None for ~3+ minutes, no re-post.

        Comparison: count bot's aqs that are confirmed-resting OR too fresh
        (placement_ts within RECONCILIATION_MIN_AGE_SEC — may still be
        propagating). If that count < len(last_desired_quotes), at least one
        expected quote is definitively missing → clear cache.

        ONLY applies to `execution_type="quoter"` (QuoterBot + MapQuoter);
        arbers/takers post ephemeral makers tied to edge conditions and
        shouldn't be force-refired.

        Returns the list of (ticker, expected, confirmed_or_fresh) tuples for
        the refired bots. Side-effect: mutates `bot.last_desired_quotes = None`
        on each refired bot.
        """
        MAKER_TYPES = ("quoter",)
        refired: List[tuple] = []
        for bot in self.bots:
            if not getattr(bot, "active", True):
                continue
            if bot.config.execution_type not in MAKER_TYPES:
                continue
            if bot.config.stop_quoting:
                continue
            last_desired = getattr(bot, "last_desired_quotes", None)
            if not last_desired:
                # None (never quoted) or empty list (intentionally flat) —
                # nothing to refire.
                continue
            expected = len(last_desired)
            bot_id = str(id(bot))
            aq_list = self.active_quotes_by_bot.get(bot_id, [])
            confirmed_or_fresh = 0
            for aq in aq_list:
                oid = getattr(aq, "order_id", None)
                if oid and oid in kalshi_resting_ids:
                    confirmed_or_fresh += 1
                    continue
                placement_ts = getattr(aq, "placement_ts", 0.0) or 0.0
                if (now - placement_ts) < self.RECONCILIATION_MIN_AGE_SEC:
                    # Too fresh — may not have propagated to /portfolio/orders
                    # yet. Don't count its absence against us this round.
                    confirmed_or_fresh += 1
            if confirmed_or_fresh < expected:
                # Purge phantom AQs from active_quotes_by_bot so the next-tick
                # diff sees the missing-quote gap and produces POST actions.
                # Without this, evaluate() recomputes desired but the diff
                # reads stale phantom refs as "existing" → kept-match → no
                # POST → 1-2 min silence until something else clears the list.
                # Keep AQs that are still resting on Kalshi OR too fresh to
                # judge (placement within RECONCILIATION_MIN_AGE_SEC).
                pruned = [
                    aq for aq in aq_list
                    if (
                        (getattr(aq, "order_id", None) and aq.order_id in kalshi_resting_ids)
                        or (now - (getattr(aq, "placement_ts", 0.0) or 0.0))
                            < self.RECONCILIATION_MIN_AGE_SEC
                    )
                ]
                self.active_quotes_by_bot[bot_id] = pruned
                bot.last_desired_quotes = None
                refired.append((bot.config.ticker, expected, confirmed_or_fresh))
        return refired

    def _track_pending_cancels(self, attempted: List[Any]) -> None:
        """Record cancel attempts so the next tick can verify they succeeded."""
        if not attempted:
            return
        import time
        now = time.time()
        for aq in attempted:
            oid = getattr(aq, "order_id", None)
            if not oid:
                continue
            if oid in self._pending_cancels:
                # Already tracked (mid-retry). Don't reset attempts.
                continue
            self._pending_cancels[oid] = {
                "attempts": 1,
                "first_attempt_ts": now,
                "source_aq": aq,
            }

    # ── TRACK D1: continuous two-sided WS-vs-REST book audit (2026-08-09) ────
    _BOOKDIFF_FLAG = "book_audit.flag"
    _BOOKDIFF_MIN_QTY = 1.0        # MUST match the production dust filter
    _BOOKDIFF_MAX_REST_AGE_S = 2.0
    _BOOKDIFF_SUMMARY_EVERY_S = 60.0

    @staticmethod
    def _bookdiff_best(levels, min_qty):
        """Top-of-book from raw REST levels.

        TWO TRAPS, both of which have already cost us incidents:
          1. Kalshi REST returns levels UNSORTED — `levels[0]` grabs the 1c tail,
             not the top. Always max(). (That parser bug caused the 07:20
             fake-blip storm where REST 'looked' (1,99) on every ticker.)
          2. A >=1.0 lot dust filter is applied on the production path
             (run.py:1785) — a persistent 0.35-lot bid at a better price made
             REST and cleaned-WS disagree every tick forever (100TRRQ 26JUL02).
        `orderbook_poller` stores its own best_* WITHOUT the dust filter, so we
        recompute from orderbook_fp rather than reading RestBook.best_yes_ask.
        """
        best = 0
        for lv in (levels or []):
            if not lv or len(lv) < 2:
                continue
            try:
                p, q = int(round(float(lv[0]) * 100)), float(lv[1])
            except (TypeError, ValueError):
                continue
            if q >= min_qty and p > best:
                best = p
        return best

    def _book_audit_tick(self, tickers, top_level_bids, top_level_offers) -> None:
        """Compare the WS top-of-book we are about to trade on against the REST
        book the poller already holds. Zero new API calls.

        WHY THIS EXISTS. Every prior instrument was conditioned on us FIRING,
        so it could only ever observe the half of the error where our book was
        too optimistic (-> zero-fill). The other half — our book pessimistic,
        so we never fire at all — generates no event and has never been
        measured. Firing SELECTS the optimistic tail of a two-sided error;
        that selection effect is why the zero-fill investigation kept coming
        back inconclusive.

            gap = ws - rest
              ask side, gap > 0 : we see a WORSE ask than exists -> FALSE NEGATIVE
              ask side, gap < 0 : we see an ask that isn't there -> fire into air

        Instantaneous disagreement is mostly timing (REST is up to
        _BOOKDIFF_MAX_REST_AGE_S old). What indicts the book is disagreement
        that PERSISTS, so consecutive-tick run length is tracked per ticker and
        only runs are escalated to WARNING.
        """
        if not os.path.exists(self._BOOKDIFF_FLAG):
            return
        reg = getattr(self, "orderbook_registry", None)
        if reg is None:
            return
        import time   # module has no top-level `time` — local, as at 681/793/1002
        st = getattr(self, "_bookdiff_state", None)
        if st is None:
            # `last` seeds to NOW, not 0.0 — otherwise the first sample is
            # instantly >= the summary interval, so every run emits a junk
            # 1-sample summary and zeroes the counters before anything is read.
            st = self._bookdiff_state = {"runs": {}, "n": 0, "fn": 0, "fp": 0,
                                         "last": time.time(), "worst": 0}
        mq = self._BOOKDIFF_MIN_QTY
        for t in tickers:
            rb = reg.get(t)
            if rb is None or rb.age_sec > self._BOOKDIFF_MAX_REST_AGE_S:
                continue
            ob = rb.orderbook_fp or {}
            r_yes_bid = self._bookdiff_best(ob.get("yes_dollars"), mq)
            r_no_bid = self._bookdiff_best(ob.get("no_dollars"), mq)
            if r_yes_bid == 0 and r_no_bid == 0:
                continue
            r_ask = (100 - r_no_bid) if r_no_bid > 0 else 100
            w_bid = top_level_bids.get(t)
            w_ask = top_level_offers.get(t)
            if w_bid is None or w_ask is None:
                continue
            g_bid, g_ask = int(w_bid) - r_yes_bid, int(w_ask) - r_ask
            st["n"] += 1
            if g_ask > 0:
                st["fn"] += 1          # ask looks worse than it is -> we don't fire
            elif g_ask < 0:
                st["fp"] += 1          # ask looks better than it is -> fire into air
            st["worst"] = max(st["worst"], abs(g_ask), abs(g_bid))
            run = st["runs"].get(t, 0) + 1 if (g_bid or g_ask) else 0
            st["runs"][t] = run
            if run == 3 or (run and run % 10 == 0):
                log.warning(
                    f"[BOOKDIFF PERSIST] {t} ws={w_bid:.0f}/{w_ask:.0f} "
                    f"rest={r_yes_bid}/{r_ask} gap_bid={g_bid:+d} gap_ask={g_ask:+d} "
                    f"— {run} consecutive ticks (rest_age={rb.age_sec*1000:.0f}ms). "
                    f"{'FALSE NEGATIVE: real ask is better than we think' if g_ask > 0 else ''}"
                    f"{'FIRE-INTO-AIR: ask we see is not there' if g_ask < 0 else ''}")
        now = time.time()
        if st["n"] and now - st["last"] >= self._BOOKDIFF_SUMMARY_EVERY_S:
            st["last"] = now
            ex = st["n"] - st["fn"] - st["fp"]
            log.warning(
                f"[BOOKDIFF SUMMARY] {st['n']} ticker-ticks | exact={ex} "
                f"({100.0*ex/st['n']:.1f}%) | FALSE-NEG(ws ask>rest)={st['fn']} | "
                f"fire-into-air(ws ask<rest)={st['fp']} | worst_gap={st['worst']}c")
            st.update(n=0, fn=0, fp=0, worst=0)

    async def run_tick(self, dt_market_state: Dict[str, Any], top_level_bids: Dict[str, float], top_level_offers: Dict[str, float]) -> None:
        """
        Execute a single loop logic path for all bots.
        """
        # One-time fast-cancel context registration (FAST_CANCEL_SCOPE.md
        # Change B): hands trade_logger's fills thread the running loop + the
        # shared engine so maker fills dispatch P0 direction cancels without
        # waiting for this tick loop. Kill switch: fast_cancel_off.flag.
        import execution as _exec_mod
        if not _exec_mod.fast_cancel_ready():
            try:
                _exec_mod.set_fast_cancel_context(asyncio.get_running_loop(), self.exec_engine)
                log.info("[FAST-CANCEL] context registered (main loop + shared engine)")
            except Exception:
                pass
        # Verify any cancels that were attempted in prior ticks actually went through.
        # This protects against the race where cancel_quotes_batch returns success
        # but the order is still resting on Kalshi (e.g. cancel/fill collision).
        await self._verify_pending_cancels()

        # Periodic safety sweep: catch any orphans the per-cancel verification missed
        # (e.g. orders we never tracked due to a startup glitch, manual orders we no
        # longer want, etc.). Throttled internally; disable via reconciliation_enabled
        # when intentionally placing manual orders.
        await self._reconcile_orders()

        all_desired_quotes = []

        # Batch process theoretical routing
        tickers = list({bot.config.ticker for bot in self.bots if bot.active})
        log.debug(f"--- MANAGER TICK START --- | Evaluating Market Bots: {tickers}")
        # Inject top-of-book into dt_market_state so theo models (e.g. live_series)
        # can access map prices even when they come from Poly CLOB (no raw_ob)
        dt_market_state["__top_bids__"] = top_level_bids
        dt_market_state["__top_offers__"] = top_level_offers
        # Inject the per-ticker data_source registry so live_series_model can pass
        # the right value to evaluate_forfeit (Tier 1 CS uses Poly 24/7).
        dt_market_state["__data_source__"] = self.data_source_by_ticker
        dt_market_state["__disable_forfeit__"] = self.disable_forfeit_by_ticker
        # ── TRACK D (2026-08-09) ─────────────────────────────────────────────
        # D2: hand the exec engine the exact top-of-book this tick will decide
        # on, so the IOC phantom probe can tell STALE-BOOK from LATE.
        # D1: continuous two-sided WS-vs-REST audit. Both are logging-only and
        # D1 is flag-gated; neither can touch order flow.
        try:
            self.exec_engine.set_ws_top(top_level_bids, top_level_offers)
        except Exception:
            pass
        try:
            self._book_audit_tick(tickers, top_level_bids, top_level_offers)
        except Exception as _bae:
            log.debug(f"[BOOKDIFF] audit error (ignored): {_bae}")

        theos_map = self.theo_generator.get_theos(tickers, dt_market_state)
        log.debug(f"--- MANAGER MAPPED THEOS --- | Result: {theos_map}")

        # --- DEBUG MODE: Console theos summary to visually track live limits ---
        DEBUG_THEO_CONSOLE = False
        if DEBUG_THEO_CONSOLE and theos_map:
            print("\n" + "─" * 65)
            for tkr, theo_dict in theos_map.items():
                b_theo = theo_dict.get('bid_theo', 0.0)
                o_theo = theo_dict.get('offer_theo', 100.0)
                # Formatting ticker nicely to extract suffix
                suffix = tkr.split("-")[-1] if "-" in tkr else tkr
                print(f" [DEBUG THEO] {suffix:<8} | BID THEO: {b_theo:>5.1f}¢  |  OFFER THEO: {o_theo:>5.1f}¢")
            print("─" * 65)

        # 1. Ask every bot what they want to quote based on their unique setup
        stale_quotes_to_cancel = []
        ordered_payloads = []
        # Phase 3: collector for amend-eligible pairs. Same-level price/size
        # deltas land here when amend_enabled() is True; otherwise stays empty
        # and behavior is byte-identical to the cancel+post hot path.
        amend_pairs_to_send: List[tuple] = []
        payload_to_bot_map = [] # Track strictly identically ordered indexes!
        
        for bot in self.bots:
            ticker = bot.config.ticker
            
            # CRITICAL FIX: Securely sandbox each active bot tracker to its direct Python memory reference!
            # If multiple CSV rows share the identical `market_id` arbitrarily inputted by the user,
            # they must not iteratively cross-pollinate and forcefully tear down each other's resting positions!!
            bot_id = str(id(bot))
            market_tag = bot.config.market_id
            
            bot_theos = theos_map.get(ticker, {})
            bid_theo = bot_theos.get("bid_theo")
            offer_theo = bot_theos.get("offer_theo")

            # ── SAFETY INVARIANT (2026-07-18): no theo → cancel all + fire nothing ──
            # When the theo pipeline AFFIRMATIVELY signals it has no valid fair value
            # for this series ticker (forfeit, series decided, undetectable state —
            # see live_series_model's `halt` marker), cancel EVERY resting order for
            # this bot and skip evaluate — for ALL bot types, arbers included.
            # Previously arbers were exempt from any no-theo cancel and re-derived
            # state internally; that second, independent forfeit check disagreed with
            # the pipeline and kept the arber quoting straight through a forfeit
            # freeze (ATRNTR run-over, 2026-07-18). The pipeline is now the single
            # source of truth for "do we have a theo." Only esports-series tickers the
            # generator actually examined ever carry a halt marker, so hedger /
            # mma_arber / polar_bear / non-esports bots are untouched.
            # ── MAKER-ONLY TOXICITY SUPPRESSION (2026-07-23) ──
            # A softer signal than `halt`: the bo3 in-game-state gate flags the
            # pickoff regime (close/late map, informed flow). Analysis of 7d of
            # fills showed our resting QUOTES get run over there (VAL maker
            # ultra-toxic tail settled ~-32c/lot), but our map-triggered TAKERS
            # are on the informed side of that same move and PROFIT (+~$3-5k,
            # settle +35c/lot). So `maker_suppress` cancels + skips ONLY the
            # dedicated quoter; arbers / momentum / hedgers ignore it and keep
            # firing. Sport-agnostic (VAL + CS2-T2). See bo3_score_feed /
            # live_series_model. (`halt`, below, still stops EVERY bot — reserved
            # for no-fair-value cases: forfeit, series decided, implausible score.)
            if (isinstance(bot_theos, dict) and bot_theos.get("maker_suppress")
                    and bot.config.execution_type == "quoter"):
                if self.active_quotes_by_bot.get(bot_id):
                    stale_quotes_to_cancel.extend(self.active_quotes_by_bot[bot_id])
                    self.active_quotes_by_bot[bot_id] = []
                continue

            if isinstance(bot_theos, dict) and bot_theos.get("halt"):
                if self.active_quotes_by_bot.get(bot_id):
                    stale_quotes_to_cancel.extend(self.active_quotes_by_bot[bot_id])
                    self.active_quotes_by_bot[bot_id] = []
                continue

            # Quoters permanently rely on mathematical pipelines. Arbers structurally bypass them entirely!
            is_arber = bot.config.execution_type in ["arber", "mma_arber", "polar_bear", "hedger"]
            if not is_arber and (bid_theo is None or offer_theo is None):
                # Cancel any resting orders for this bot — theos disappeared (game over, etc.)
                if bot_id in self.active_quotes_by_bot and self.active_quotes_by_bot[bot_id]:
                    stale_quotes_to_cancel.extend(self.active_quotes_by_bot[bot_id])
                    self.active_quotes_by_bot[bot_id] = []
                continue

            t_state = dt_market_state.get(ticker, {})
            t_bid = top_level_bids.get(ticker, 0.0)
            t_offer = top_level_offers.get(ticker, 100.0)

            log.debug(f"BOT EVALUATION | Calling evaluate() on Bot for {market_tag} [{bot_id}] ({ticker}) | top_bid={t_bid}, top_offer={t_offer}, bid_theo={bid_theo}, offer_theo={offer_theo}")
            
            # Pass full system states to the bot since Arbers must cross-reference maps against series.
            # series_eval is the rich SeriesEvaluation dict from LiveSeriesTheoGenerator (Phase 5):
            # contains state, active_map, hedge profiles, synthetic costs, edges, etc. for esports series.
            # Bots that don't consume it (mma_arber, soccer, mlb_total) ignore the kwarg.
            eval_kwargs = {
                "market_state": t_state,
                "top_level_bid": t_bid,
                "top_level_offer": t_offer,
                "bid_theo": bid_theo,
                "offer_theo": offer_theo,
                "full_market_state": dt_market_state,
                "full_top_bids": top_level_bids,
                "full_top_offers": top_level_offers,
                "series_eval": bot_theos if isinstance(bot_theos, dict) and "state" in bot_theos else None,
            }
            desired_quotes = bot.evaluate(**eval_kwargs)
            
            # None = Optimization block triggered (reprice buffer says no change).
            if desired_quotes is None:
                continue
            
            log.debug(f"BOT EVALUATION RESULT | {bot_id} ({ticker}) Output quotes: {desired_quotes}")

            # Stamp ticker so the diff signature uses the right one.
            for q in (desired_quotes or []):
                if "ticker" not in q:
                    q["ticker"] = ticker

            # DIFF-BASED RECONCILIATION (2026-06-07): only cancel resting orders
            # whose (ticker, side, kalshi_side, price, size) signature is NOT in
            # the desired set, and only POST desired quotes that don't already
            # have an identical resting twin. Replaces the prior cancel-all-then-
            # post-all pattern that burned ~96 tokens/cycle on arber re-evals
            # even when the actual quotes hadn't changed. Cuts the steady-state
            # token cost to ~0 for stable books, eliminating the Cloudflare
            # edge-429 storm.
            existing = self.active_quotes_by_bot.get(bot_id, [])
            # Phase 3: gated on flag. When ON, same-level price/size deltas on
            # RESTING quotes produce amend_pairs (preserving queue position);
            # when OFF, the resting path is byte-identical to legacy cancel+post.
            # IOC (taker) quotes are ALWAYS fresh-posted and NEVER amended —
            # amending a never-resting IOC 404s and churns cancel+post, executing
            # nothing (2026-07-24 amend-IOC bug). _plan_bot_orders splits them off;
            # the registration guard (~1490) keeps IOC posts out of
            # active_quotes_by_bot so they never resurface as `existing`.
            kept_aqs, amend_pairs_this_bot, cancel_aqs, post_qs = _plan_bot_orders(
                existing, desired_quotes or [], amend_enabled()
            )

            # Orphan-prevention (2026-06-28): keep amend candidates in the
            # bot's active list from the moment the amend is requested. The
            # previous behavior was `= kept_aqs`, which stripped amend pairs
            # immediately and only re-added them after a successful amend.
            # If the amend coroutine never reached the success-or-failure
            # branch (async cancel, exception swallowed, cycle interrupted,
            # whatever), the order would silently be dropped from local
            # tracking while remaining alive on Kalshi → orphan → eventual
            # pickoff. ECHSHK-ECH 2026-06-28 15:27 incident: 3d048426 (46c
            # bid) was lost this way, then ate 746 of 750 lots at a stale
            # price.
            #
            # New invariant: an ActiveQuote stays tracked from POST through
            # confirmed CANCEL. AMEND no longer moves it through a tracking
            # gap. Failed amends still get routed to cancel+post below; the AQ
            # is removed from active_quotes_by_bot only once the cancel is
            # CONFIRMED dead in _verify_pending_cancels (654-657) — NOT in
            # _track_pending_cancels, which merely registers the pending cancel.
            self.active_quotes_by_bot[bot_id] = (
                list(kept_aqs) + [aq for aq, _ in amend_pairs_this_bot]
            )
            stale_quotes_to_cancel.extend(cancel_aqs)
            if amend_pairs_this_bot:
                # Stamp bot id onto each pair's desired dict so the failed-amend
                # cancel+post fallback can bind the reposted order back to the
                # right bot (see post-dispatch binding ~1356).
                for aq, q in amend_pairs_this_bot:
                    q["_req_bot"] = bot_id
                amend_pairs_to_send.extend(amend_pairs_this_bot)

            # Per-tick diff visibility. Logs EVERY evaluate() that produced a
            # diff action (or, if neither produced one, logs as a no-op when
            # there were resting quotes to compare against). Lets you grep the
            # log and prove whether the diff is or isn't catching duplicates:
            #   grep "\[DIFF\]" run.log | awk ...
            # Format: [DIFF] {bot_id} {ticker} type={execution_type}
            #         existing={N} kept={N} cancel={N} post={N}
            #         cancels=[(price,size,side,kside), ...]
            #         posts=[(price,size,side,kside), ...]
            if cancel_aqs or post_qs or amend_pairs_this_bot or existing:
                def _fmt_aq(a):
                    sv = a.side.value if hasattr(a.side, "value") else str(a.side)
                    return f"({a.limit_cents},{a.size},{sv},{a.kalshi_side})"
                def _fmt_q(q):
                    s = q.get("side")
                    sv = s.value if hasattr(s, "value") else str(s)
                    return (f"({q.get('limit_cents')},{q.get('size')},"
                            f"{sv},{q.get('kalshi_side','yes')})")
                def _fmt_amend(pair):
                    a, q = pair
                    sv = a.side.value if hasattr(a.side, "value") else str(a.side)
                    new_p = q.get("limit_cents", a.limit_cents)
                    new_s = q.get("size", a.size)
                    return f"({a.limit_cents}->{new_p},{a.size}->{new_s},{sv},{a.kalshi_side})"
                cancels_str = ("[" + ",".join(_fmt_aq(a) for a in cancel_aqs) + "]"
                               if cancel_aqs else "[]")
                posts_str = ("[" + ",".join(_fmt_q(q) for q in post_qs) + "]"
                             if post_qs else "[]")
                amends_str = ("[" + ",".join(_fmt_amend(p) for p in amend_pairs_this_bot) + "]"
                              if amend_pairs_this_bot else "[]")
                # Amend field only emitted when the flag is on, so existing
                # log-parsers stay compatible with the legacy format.
                amend_field = (f" amend={len(amend_pairs_this_bot)} amends={amends_str}"
                               if amend_enabled() else "")
                log.info(
                    f"[DIFF] {bot_id} {ticker} type={bot.config.execution_type} "
                    f"existing={len(existing)} kept={len(kept_aqs)} "
                    f"cancel={len(cancel_aqs)} post={len(post_qs)}{amend_field} "
                    f"cancels={cancels_str} posts={posts_str}"
                )

                # Phase 1 feed-integrity shadow log. Phase 0 proved that 100%
                # of cancel-no-post cycles happen while REST shows a healthy
                # book. Print that mismatch inline next to the [DIFF] so we
                # can see it live and confirm in-process registry agrees with
                # the standalone CSV. Decision logic unchanged.
                if (
                    len(post_qs) == 0
                    and len(cancel_aqs) > 0
                    and len(existing) > 0
                    and getattr(self, "orderbook_registry", None) is not None
                ):
                    rest = self.orderbook_registry.get(ticker)
                    if rest is not None and not rest.is_empty and rest.age_sec < 3.0:
                        log.info(
                            f"[FEED CHECK] {ticker} bot=cancel_all_no_post "
                            f"rest_yes_bid={rest.best_yes_bid} rest_yes_ask={rest.best_yes_ask} "
                            f"rest_depth_yes={rest.depth_yes_count:.0f} "
                            f"rest_depth_no={rest.depth_no_count:.0f} "
                            f"rest_age_ms={rest.age_sec*1000:.0f}"
                        )

            # Theo-annotation helper. Stamps _adj_theo + _raw_theo onto a
            # desired-quote dict so downstream telemetry / trades.csv carry
            # the bot's view at quote-emit time. Skips if already set (arb
            # orders pre-stamp their own context).
            def _annotate_theo(q):
                if "_adj_theo" in q:
                    return
                if bot.config.execution_type == "hedger":
                    # Hedger orders are risk-reduction trades on MAP tickers, not
                    # edge-capture trades. They don't carry meaningful "edge"
                    # against the bot's SERIES theo (which is what bid_theo/
                    # offer_theo would be here). Stamp theo = limit price so
                    # downstream edge calc resolves to ~0 in trades.csv and
                    # watch_fills.py instead of inheriting a misleading series
                    # theo and producing fake edges like +18c on a hedge buy.
                    px = q.get("limit_cents", 0)
                    ks = q.get("kalshi_side", "yes").lower()
                    yes_equiv = px if ks == "yes" else (100 - px)
                    q["_adj_theo"] = yes_equiv
                    q["_raw_theo"] = yes_equiv
                elif q.get("kalshi_side", "yes").lower() == "yes":
                    if q.get("side") and q["side"].name == "BID":
                        q["_adj_theo"] = bid_theo
                        q["_raw_theo"] = bot_theos.get("raw_bid_theo", bid_theo)
                    else:
                        q["_adj_theo"] = offer_theo
                        q["_raw_theo"] = bot_theos.get("raw_offer_theo", offer_theo)
                else:
                    # All NO-side orders (BUY NO or OFFER NO) result in SHORT YES.
                    # The relevant theo is offer_theo (our sell-side fair value).
                    q["_adj_theo"] = offer_theo
                    q["_raw_theo"] = bot_theos.get("raw_offer_theo", offer_theo)

            # Annotate + enqueue ONLY the quotes that actually need to be POSTed.
            for q in post_qs:
                q["_req_bot"] = bot_id
                _annotate_theo(q)
                ordered_payloads.append(q)

            # CRITICAL: amend pair q dicts ALSO get annotated. If an amend
            # fails and falls back to cancel+post, the fallback POST goes
            # through the same telemetry-stamping path — without this, the
            # new order's telemetry stores theo=0.0 and all subsequent fills
            # show edge=garbage in trades.csv. Discovered 2026-06-17 on the
            # REFGAM live test: order 4c49ddf6 was a fallback-POST after an
            # amend FAIL, accumulated fills with raw_theo=0/adj_theo=0.
            for aq, q in amend_pairs_this_bot:
                _annotate_theo(q)

        # 2. Synchronize Physical Bound Execution securely against the Kalshi Book

        # Phase 3 dispatch: amend → cancel → post. Amends fire first so their
        # HTTP roundtrip overlaps with cancel processing on Kalshi's side.
        # Failed amends are routed back through cancel + post — the bot never
        # freezes on a stuck amend.
        if amend_pairs_to_send:
            log.debug(f"MANAGER FAT PIPE | Amending {len(amend_pairs_to_send)} resting orders in-place...")
            # 2026-06-28: diagnostic instrumentation. Catch BaseException so
            # CancelledError doesn't slip through and abandon amend pairs in
            # a half-tracked state. If we catch one, log loudly and force
            # everything into failed (cancel+post fallback). We re-raise so
            # the outer task cancellation still propagates correctly.
            try:
                succeeded_aqs, failed_amend_pairs = await self.exec_engine.amend_quotes_batch(amend_pairs_to_send)
            except asyncio.CancelledError as ce:
                log.error(
                    f"[AMEND BATCH CANCELLED] n_pairs={len(amend_pairs_to_send)} — "
                    f"outer task cancelled mid-batch. Routing ALL pairs to failed "
                    f"(cancel+post fallback) before re-raising."
                )
                # Important: the cancel re-raise means stale_quotes_to_cancel
                # / post_quotes_batch won't execute this tick — those pairs
                # will be reprocessed via the fix #1 path (they remain in
                # active_quotes_by_bot) on the next non-cancelled cycle.
                succeeded_aqs = []
                failed_amend_pairs = [(aq, q, "ambiguous") for aq, q in amend_pairs_to_send]
                raise
            except Exception as e:
                log.error(
                    f"[AMEND BATCH EXCEPTION] n_pairs={len(amend_pairs_to_send)} "
                    f"err_type={type(e).__name__} err={e!r} — "
                    f"falling back to cancel+post for all pairs"
                )
                # Unknown whether any amend landed → ambiguous for all pairs.
                succeeded_aqs = []
                failed_amend_pairs = [(aq, q, "ambiguous") for aq, q in amend_pairs_to_send]

            # Refresh telemetry for each succeeded amend. The order_id stayed
            # the same but the theos (_raw_theo / _adj_theo on the new q dict)
            # reflect the bot's view at amend time, not the original POST.
            # Without this refresh, watch_fills + trades.csv attribute every
            # amend-then-fill to the STALE post-time theo, displaying good
            # trades as negative-edge "bad" fills. Bug surfaced 2026-06-21
            # right after amend was re-enabled — went unnoticed in the original
            # 2026-06-17 amend ship because cancel+post was still the default.
            #
            # NOTE (2026-06-28): there is NO bind-back step anymore. The AQ is
            # already in active_quotes_by_bot from the sticky-keep above
            # (`list(kept_aqs) + amend AQs`), set before dispatch and never
            # removed before this point, so a re-append would be a guaranteed
            # no-op. The old `[AMEND REBIND MISS]` warning was therefore
            # unreachable/misleading and has been removed. This block now does
            # telemetry only.
            import telemetry_store
            succeeded_ids = {id(aq) for aq in succeeded_aqs}
            for pair_aq, pair_q in amend_pairs_to_send:
                if id(pair_aq) not in succeeded_ids:
                    continue
                # Refresh BOTH keys: execution.py pre-POST stashes telemetry
                # under client_order_id (to bridge the WS-fill-arrives-before-
                # POST-response race), then manager.py post-success stashes
                # again under order_id. trade_logger.get_telemetry looks up
                # client_order_id FIRST, so if we only refresh order_id, the
                # CID key holds the stale post-time theo forever and edge
                # display reads it. Discovered 2026-06-22 via PLUCAM-PL
                # trade where adj_theo logged as 55c (post-time, stale)
                # vs the real 34c at fill time, producing fake -15c edge.
                def _refresh(key):
                    if not key:
                        return
                    telemetry_store.put_telemetry(
                        key,
                        pair_q.get("_raw_theo", 0.0),
                        pair_q.get("_adj_theo", 0.0),
                        pair_q.get("_arb_context", ""),
                        trigger_type=pair_q.get("_trigger_type", ""),
                        quoted_size=int(pair_aq.size or 0),
                        was_size_capped=bool(pair_q.get("_was_size_capped", False)),
                        sweep_id=pair_q.get("_sweep_id", ""),
                        sweep_total_vol=int(pair_q.get("_sweep_total_vol", 0) or 0),
                    )
                _refresh(pair_aq.order_id)
                _refresh(pair_aq.client_order_id)

            # Route failed amends to cancel+post, branching on the failure kind
            # (2026-06-28 P0-2):
            #   reject    — the amend definitively did NOT apply (order already
            #               gone, or rejected before processing). The order's
            #               resting state is KNOWN, so cancel the old oid AND
            #               repost now — same risk profile as ordinary
            #               cancel+post repricing.
            #   ambiguous — timeout/5xx/network: the amend MIGHT have applied,
            #               so the order may now be live at the NEW price. Do
            #               NOT blind-repost (that would double the resting
            #               size). Cancel the old oid and DEFER the repost: once
            #               _verify_pending_cancels confirms the cancel and
            #               evicts the AQ, the bot re-derives + reposts the
            #               desired quote on a later tick with no double-exposure
            #               window.
            if failed_amend_pairs:
                n_reject = sum(1 for _, _, k in failed_amend_pairs if k == "reject")
                n_defer = len(failed_amend_pairs) - n_reject
                log.warning(
                    f"[P0-2 AMEND FAILED] {len(failed_amend_pairs)}/{len(amend_pairs_to_send)} "
                    f"amend(s) failed → fallback: {n_reject} reject (repost now), "
                    f"{n_defer} ambiguous/other (cancel + defer repost). "
                    f"oids=[{','.join(str(aq.order_id) for aq, _, _ in failed_amend_pairs[:8])}]"
                )
            for aq, q, kind in failed_amend_pairs:
                stale_quotes_to_cancel.append(aq)
                if _failed_amend_should_repost(kind):
                    ordered_payloads.append(q)
                    log.info(
                        f"[AMEND FALLBACK] {aq.ticker} order={aq.order_id} "
                        f"side={aq.kalshi_side} kind={kind} routed to cancel+post"
                    )
                else:
                    # ambiguous (or any non-reject kind): the amend MIGHT have
                    # applied, so the order may be live at the NEW price. Cancel
                    # the old oid but DEFER the repost — the bot re-derives and
                    # reposts next tick once _verify_pending_cancels confirms the
                    # cancel and evicts the AQ. No double-exposure window.
                    log.warning(
                        f"[AMEND FALLBACK DEFER] {aq.ticker} order={aq.order_id} "
                        f"side={aq.kalshi_side} kind={kind} — cancelling old oid; "
                        f"repost deferred until cancel confirms (no double-exposure)"
                    )

        if stale_quotes_to_cancel:
            log.debug(f"MANAGER FAT PIPE | Tearing down {len(stale_quotes_to_cancel)} stale limits exclusively tracked to active reprice boundaries...")
            await self.exec_engine.cancel_quotes_batch(stale_quotes_to_cancel)
            self._track_pending_cancels(stale_quotes_to_cancel)

        if ordered_payloads:
            # ---> SELF MATCH GUARD FOR ARBERS <---
            # 2026-06-19: replaced broken inline guard with _find_self_match_targets
            # helper. Old guard required `aq.kalshi_side == ioc_side` AND
            # `aq.kalshi_side != ioc_side` simultaneously — impossible — so
            # `self_match_targets` was always empty. See helper docstring for
            # full Kalshi cross-math.
            arber_ioc_payloads = [q for q in ordered_payloads if q.get("time_in_force") == "immediate_or_cancel"]
            if arber_ioc_payloads:
                self_match_targets = _find_self_match_targets(arber_ioc_payloads, self.active_quotes_by_bot)
                if self_match_targets:
                    log.warning(f"--- SELF MATCH PREVENTED --- Executing pre-emptive synchronized cancel on {len(self_match_targets)} crossing limits!")
                    # De-duplicate
                    dedup_targets = list({aq.order_id: aq for aq in self_match_targets}.values())
                    # P0: this cancel protects an imminent fire — it rides the
                    # fire lane so it can't queue behind routine cancels.
                    await self.exec_engine.cancel_quotes_batch(
                        dedup_targets, priority=write_gate.P0_FIRE)
                    self._track_pending_cancels(dedup_targets)
                    # Erase from state mapping
                    for aq in dedup_targets:
                        for bq_list in self.active_quotes_by_bot.values():
                            if aq in bq_list:
                                bq_list.remove(aq)

            log.debug(f"MANAGER FAT PIPE | Pushing {len(ordered_payloads)} freshly mapped limits down the wire...")
            new_active_quotes = await self.exec_engine.post_quotes_batch(ordered_payloads)
            
            # Strictly bind the resultant Active Quotes directly back to their parent Bot UUID natively!
            for aq in new_active_quotes:
                if hasattr(aq, "_source_q") and "_req_bot" in aq._source_q:
                    req_bot = aq._source_q["_req_bot"]
                    
                    # Trap fatal market closure failures instantly and completely break the execution loop for this ticker
                    if aq.status.name == "FAILED":
                        bot = next((b for b in self.bots if str(id(b)) == req_bot), None)
                        if bot and bot.active:
                            log.warning(f"--- MARKET CLOSED / NOT FOUND --- | Structurally Deactivating QuoterBot Engine for: {bot.config.ticker}")
                            bot.active = False
                        continue
                        
                    if req_bot not in self.active_quotes_by_bot:
                        self.active_quotes_by_bot[req_bot] = []
                    # IOC orders never rest — do NOT track them as active/resting,
                    # or next tick's diff will try to amend a dead IOC (404 churn,
                    # the 2026-07-24 bug). Fresh-post-only: leave out of registry.
                    # (Telemetry below still runs so the fill is attributed.)
                    if aq._source_q.get("time_in_force") != "immediate_or_cancel":
                        self.active_quotes_by_bot[req_bot].append(aq)

                    # Telemetry Memory Bridge: Map Order UUID to Mathematical Models securely!
                    if aq.order_id:
                        import telemetry_store
                        telemetry_store.put_telemetry(
                            aq.order_id,
                            aq._source_q.get("_raw_theo", 0.0),
                            aq._source_q.get("_adj_theo", 0.0),
                            aq._source_q.get("_arb_context", ""),
                            quoted_size=int(aq.size or 0),
                        )
        else:
            if not stale_quotes_to_cancel:
                log.debug("MANAGER FAT PIPE | No structural changes or quotes generated across any active bots.")
