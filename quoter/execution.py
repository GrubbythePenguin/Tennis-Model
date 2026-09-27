import asyncio
import threading
import uuid
import logging
import os
import time
from typing import Any, List, Dict, Optional
from models import ActiveQuote, QuoteStatus, QuoteSide

# Ticker alias layer — translate the bot's synthetic ticker → real Kalshi
# ticker at the order create/amend request boundary. Internal state
# (ActiveQuote.ticker, client_order_id maps, telemetry) stays keyed on
# synthetic. Pass-through for any non-aliased ticker.
import ticker_aliases
import write_gate

log = logging.getLogger(__name__)

# ── 429 TRACKER (2026-09-01) ──────────────────────────────────────────────────
# Every write-side 429 is a probable collision with the esports system on the
# shared account bucket (the two processes cannot serialize in-flight writes
# against each other). Logged loudly with running totals so a slate's collision
# rate is one grep:  grep -c '429 TRACKER' run.log
_429_counts = {"post": 0, "amend": 0, "cancel": 0}

def _note_429(kind: str, detail: str = "") -> None:
    _429_counts[kind] = _429_counts.get(kind, 0) + 1
    log.warning("[429 TRACKER] write 429 on %s%s — session totals post=%d amend=%d "
                "cancel=%d (shared-bucket collision with esports likely)",
                kind, f" ({detail})" if detail else "",
                _429_counts.get("post", 0), _429_counts.get("amend", 0),
                _429_counts.get("cancel", 0))

# ── TENNIS / SHARD 3 ──────────────────────────────────────────────────────────
# This is the tennis fork of the esports quoter. Tennis markets moved to exchange
# shard 3 on 26AUG24. Two things follow and both are load-bearing:
#   * every ORDER WRITE carries exchange_index explicitly. An auto-routed write bills
#     multiple rate buckets, and would draw on whichever shard the ticker implies.
#   * every ORDER READ must pass exchange_index too. Verified 26AUG26: GET
#     /portfolio/orders with no index returned exactly the shard-0 orders (6 of 6) and
#     zero shard-3 orders. A reconciliation pass that omits it sees NO resting tennis
#     orders and reposts them - an unbounded duplicate loop.
# Collateral is per-shard and per-subaccount: the funded tennis balance lived on
# shard 3 / sub 0 until 2026-09-09; both sports now trade sub 1 (see below).
#
# ── SPORT MODE (2026-09-09) ──────────────────────────────────────────────────
# This process runs ONE shard, chosen by QUOTER_SPORT. Table tennis
# (KXTTELITEMATCH) lives on exchange_index 0 — verified 2026-09-09 with authed
# per-ticker GETs (tennis_populate_configs.verify_shard) on 6 live tickers, all
# shard 0 — so it CANNOT share a process with tennis: every order write stamps
# one exchange_index and every order read passes one, and splitting them
# per-ticker would fork the reconciliation sweep too. A whole-process mode
# keeps the single-shard invariant this file was built on.
#
#   QUOTER_SPORT unset / "tennis"  -> shard 3, tennis series only (unchanged)
#   QUOTER_SPORT=tt                -> shard 0, KXTTELITEMATCH only
#
# BOTH sports run on SUBACCOUNT 1 (see TENNIS_SUBACCOUNT below): shard 0 sub 0
# is the esports system's collateral pool, and sub 1 isolates this book from it
# (verified: sub 1 exists, $5,000 funded, no resting orders — only terminal
# Valorant test orders from 26AUG13). Rate buckets are still per-shard, so TT
# and esports SHARE order-rate limits on shard 0 regardless of subaccount.
#
# READS ARE SUBACCOUNT-SCOPED. Verified 2026-09-09 against /portfolio/orders
# on shard 0: subaccount=1 -> 3 orders, subaccount=0 -> 100, param omitted ->
# the same 100 (defaults to sub 0). So every order read must pass subaccount
# explicitly, or a sub-1 quoter sees NONE of its own resting orders and
# reposts forever — the same unbounded duplicate loop as the shard parameter
# (26AUG26), one level down.
#
# The series tuple is the safety property, same as ever: an order for a ticker
# outside it is refused at construction, so a wrong config row cannot route
# flow onto this process's shard no matter how it got there (26AUG27: 36
# leftover esports rows in template_quoter_config.csv; nothing posted only
# because the process was down). In TT mode that guard also refuses every
# TENNIS ticker, which is exactly right — they live on shard 3.
V2_BATCHED_PATH = "/trade-api/v2/portfolio/events/orders/batched"

_SPORT = os.environ.get("QUOTER_SPORT", "tennis").strip().lower()

# BOTH sports trade on subaccount 1 (operator decision 2026-09-09, "ease of
# access"): sub 0 stays the esports pool, sub 1 is the racquet-sports book —
# one place to fund, one place to read fills. Collateral is per-shard AND
# per-subaccount, so tennis orders now need funds on SHARD 3 / SUB 1 (the old
# tennis balance sits on shard 3 / sub 0 — transfer before the next tennis
# session, or every order rejects on collateral). Legacy sub-0 tennis
# positions are likewise invisible to position_store after this change.
# QUOTER_SUBACCOUNT (2026-09-10, set3 dog maker): env override kept for future
# isolation, but the set-3 dog maker runs on SUB 1 / SHARD 3 — the existing
# tennis book — by operator decision 26SEP10 ("use subaccount 1 shard 3"), i.e.
# the default below, no env var needed. Every order write stamps this value and
# every order/fill/position read passes it (manager, position_store,
# trade_logger all import TENNIS_SUBACCOUNT from here).
#
# ONE FRAMEWORK INSTANCE PER (SHARD, SUBACCOUNT) — this is load-bearing:
# _reconcile_orders sweeps EVERY resting order on its shard+sub that is not in
# its own tracking, so a second instance on the same pair has its quotes
# cancelled as orphans within a reconciliation cycle. Sharing sub 1 therefore
# makes the set3 instance and the recenter tennis instance MUTUALLY EXCLUSIVE
# while running; positions are also pooled per ticker, so a recenter bot that
# quotes a ticker carrying set3 inventory (>1.5x its max_position) goes
# reduce-only and would unwind fills that are meant to ride to settlement.
TENNIS_SUBACCOUNT = int(os.environ.get("QUOTER_SUBACCOUNT", "1"))

if _SPORT == "tt":
    TENNIS_SHARD = 0                       # verified 2026-09-09, see above
    # TT series come from tt_quotable_series.json — written only for series
    # whose shard was VERIFIED 0 by an authed per-ticker GET (initial entries
    # by hand 2026-09-09, new ones by tt_populate_configs' auto-verify). The
    # file is mtime-cache-reloaded inside _is_tennis_ticker so a newly
    # verified series becomes orderable without a run.py restart, and the
    # guard's trust model is unchanged: file membership == shard-verified.
    _TT_SERIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "tt_quotable_series.json")
    _tt_series_cache = {"mtime": -1.0, "series": ("KXTTELITEMATCH",)}

    def _tt_quotable_series() -> tuple:
        import json as _json
        try:
            mt = os.path.getmtime(_TT_SERIES_FILE)
        except OSError:
            return _tt_series_cache["series"]
        if mt != _tt_series_cache["mtime"]:
            try:
                with open(_TT_SERIES_FILE) as f:
                    data = _json.load(f)
                good = tuple(sorted(s for s, v in data.items()
                                    if isinstance(v, dict) and v.get("shard") == 0))
                if good:
                    _tt_series_cache["series"] = good
                _tt_series_cache["mtime"] = mt
            except Exception:
                pass                       # keep last-good on a torn/bad file
        return _tt_series_cache["series"]

    TENNIS_SERIES = _tt_quotable_series()
else:
    TENNIS_SHARD = 3
    TENNIS_SERIES = ("KXATPMATCH", "KXWTAMATCH",
                     "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH",
                     "KXITFMATCH", "KXITFWMATCH",      # ITF added 26AUG28, verified shard 3
                     # Per-set winner series (26AUG31). An order for one of these can only
                     # arise from a config row, and tennis_populate_configs --sets verifies
                     # every SETWINNER ticker on-shard with a per-ticker GET before writing
                     # the row - so listing them here extends the guard, not the trust.
                     "KXATPSETWINNER", "KXWTASETWINNER",
                     "KXATPCHALLENGERSETWINNER", "KXWTACHALLENGERSETWINNER",
                     "KXITFSETWINNER", "KXITFWSETWINNER")


def _is_tennis_ticker(ticker: str) -> bool:
    """True only for a market in a series known to be on THIS process's shard."""
    series = _tt_quotable_series() if _SPORT == "tt" else TENNIS_SERIES
    return str(ticker or "").split("-", 1)[0] in series



def _is_not_found(err) -> bool:
    """True iff a Kalshi per-order cancel error means the order does not exist.

    Kalshi returns {'code': 'not_found', 'details': '', 'message': 'not found'}.
    Deliberately STRICT — anything we cannot positively identify as not_found
    must fall through to the retry path, because confirming a still-resting
    order pops it from active_quotes and orphans it (the 26JUN27 bug).
    Matches only the structured `code`, or a bare string form, and never a
    substring of some other code.
    """
    if isinstance(err, dict):
        code = err.get("code")
        return isinstance(code, str) and code.strip().lower() == "not_found"
    if isinstance(err, str):
        return err.strip().lower() in ("not_found", "not found")
    return False


# Module-level order_id → ActiveQuote registry. Consumed by trade_logger
# on maker fills to bump aq.filled_count so the amend math
# (new_count = filled_count + desired.size) is correct on partial-filled
# orders. Without this, filled_count stays 0 and new_count equals the
# existing total — the amend wire is a no-op and the residual sits.
_orders_by_oid: Dict[str, ActiveQuote] = {}
_orders_by_oid_lock = threading.Lock()


def register_order(aq: ActiveQuote) -> None:
    if not aq.order_id:
        return
    with _orders_by_oid_lock:
        _orders_by_oid[aq.order_id] = aq


def unregister_order(order_id: str) -> None:
    if not order_id:
        return
    with _orders_by_oid_lock:
        _orders_by_oid.pop(order_id, None)


def bump_filled_count(order_id: str, qty: int) -> bool:
    """Increment aq.filled_count by qty for the AQ matching order_id.
    Returns True iff a matching ActiveQuote was found and updated."""
    if not order_id or qty <= 0:
        return False
    with _orders_by_oid_lock:
        aq = _orders_by_oid.get(order_id)
        if aq is None:
            return False
        aq.filled_count = (aq.filled_count or 0) + qty
        return True


def get_registered_order(order_id: str) -> Optional[ActiveQuote]:
    """Local parent-order lookup for trade_logger's maker-fill side
    resolution (FAST_CANCEL_SCOPE.md Change A). The order is ours, so its
    true side lives here — no REST round-trip. None on a miss (orders
    placed before this process started; caller falls back to REST)."""
    if not order_id:
        return None
    with _orders_by_oid_lock:
        return _orders_by_oid.get(order_id)


# ── Fast cancel (FAST_CANCEL_SCOPE.md Change B) ─────────────────────────────
# Fire the existing batched-cancel machinery from the FILL EVENT instead of
# waiting for the quote cycle to notice the maker-fill watermark (0.5-1.5s).
# Semantics are identical to the cooldown filter path — any maker fill cancels
# the direction's quotes on both tickers — only the timing changes. Context
# (main asyncio loop + the manager's shared engine) is registered lazily by
# MarketManager.run_tick; until then every call is a silent no-op, so this is
# inert on old processes and self-arms after the next restart.
# Kill switch: create fast_cancel_off.flag in the repo root (checked per
# call — no restart needed to disable).
_fc_loop = None
_fc_engine = None
_fc_last_fire: Dict[tuple, float] = {}   # direction key -> monotonic ts
_fc_lock = threading.Lock()
_FC_DEBOUNCE_S = 0.05      # N fills on one direction within 50ms → one batch
_FC_COOLDOWN_S = 15.0      # conservative stamp; each bot's evaluate re-stamps
                           # with its configured duration and the LATER
                           # until_ts wins (see trigger_direction_cooldown)
_FC_OFF_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fast_cancel_off.flag")


def set_fast_cancel_context(loop, engine) -> None:
    global _fc_loop, _fc_engine
    _fc_loop = loop
    _fc_engine = engine


def fast_cancel_ready() -> bool:
    return _fc_loop is not None and _fc_engine is not None


def fast_cancel_from_fill(fill_ticker: str, true_side: str) -> int:
    """Called from trade_logger's fills-WS thread on every MAKER fill.

    fill_ticker must be in the bot's SYNTHETIC ticker space (caller passes it
    through ticker_aliases.reverse) so prefixes match ActiveQuote.ticker.

    Stamps the shared direction cooldown immediately and dispatches a P0
    batched cancel of every live same-exposure quote (both tickers) onto the
    main loop. Returns the number of quotes dispatched. Never raises."""
    try:
        if _fc_loop is None or _fc_engine is None:
            return 0
        if os.path.exists(_FC_OFF_FLAG):
            return 0
        side = (true_side or "").lower()
        if side not in ("yes", "no") or "-" not in fill_ticker:
            return 0
        event_base, our_team = fill_ticker.rsplit("-", 1)
        prefix = event_base + "-"
        with _orders_by_oid_lock:
            aqs = [a for a in _orders_by_oid.values() if a.ticker.startswith(prefix)]
        # Opponent suffix from the registry itself (both tickers' orders share
        # the event_base) — closes the "opponent unknown" gap the quote-cycle
        # filter has via full_market_state discovery.
        opp_team = None
        for a in aqs:
            sfx = a.ticker.rsplit("-", 1)[-1]
            if sfx != our_team:
                opp_team = sfx
                break
        long_team = our_team if side == "yes" else opp_team
        dir_key = (event_base, long_team if long_team is not None
                   else f"SAMETICKER:{fill_ticker}:{side}")
        now = time.monotonic()
        with _fc_lock:
            if now - _fc_last_fire.get(dir_key, 0.0) < _FC_DEBOUNCE_S:
                return 0
            _fc_last_fire[dir_key] = now
        if long_team is not None:
            try:
                import position_store
                position_store.trigger_direction_cooldown(event_base, long_team,
                                                          _FC_COOLDOWN_S)
            except Exception:
                pass
        targets = []
        for a in aqs:
            st = getattr(a.status, "name", str(a.status))
            if st not in ("LIVE", "PARTIALLY_FILLED"):
                continue
            ks = (a.kalshi_side or "").lower()
            sfx = a.ticker.rsplit("-", 1)[-1]
            if long_team is not None:
                if ks == "yes":
                    aq_long = sfx
                elif opp_team is not None:
                    aq_long = opp_team if sfx == our_team else our_team
                else:
                    aq_long = None
                if aq_long != long_team:
                    continue
            else:
                # opponent undiscoverable: same-ticker, same-book fallback —
                # mirrors the quote-cycle filter's documented fallback.
                if a.ticker != fill_ticker or ks != side:
                    continue
            targets.append(a)
        if not targets:
            return 0
        fut = asyncio.run_coroutine_threadsafe(
            _fc_engine.cancel_quotes_batch(targets, priority=write_gate.P0_FIRE),
            _fc_loop)

        def _log_exc(f):
            try:
                exc = f.exception()
                if exc:
                    log.error(f"[FAST-CANCEL] batch failed: {exc}")
            except Exception:
                pass
        fut.add_done_callback(_log_exc)
        log.info(f"[FAST-CANCEL] {event_base} dir={dir_key[1]} fill_side={side} "
                 f"→ {len(targets)} quote(s) dispatched P0 (fill on {fill_ticker})")
        return len(targets)
    except Exception as ex:
        log.exception(f"[FAST-CANCEL] error for {fill_ticker}/{true_side}: {ex}")
        return 0


class QuoterExecutionEngine:
    """
    Handles batched order placement and cancellation layer, separate from strategy.
    Designed with native batching to eliminate execution lag during high-velocity updates.
    """
    def __init__(self, client: Any, trading_enabled: bool = True):
        self.client = client
        self.trading_enabled = trading_enabled
        # ── TENNIS RATE BUDGET: HALF of the shared-client defaults (2026-09-01,
        # operator request). The tennis quoter shares one Kalshi account bucket
        # with the esports system in general_level_based_quoting; when both run
        # in tandem, tennis must not starve it. Applied to THIS process's client
        # INSTANCE only (the module is shared on disk — never edit the class
        # defaults, that would halve esports too). Read: _max_rps 75 -> 37.5.
        # Write: rolling-window cap_tokens 600 -> 300 (same 2s window; per-op
        # costs unchanged: POST=10/order, DELETE=2/order). Guarded so a second
        # engine construction on the same client cannot compound the halving.
        if not getattr(client, "_tennis_budget_halved", False):
            try:
                client._max_rps = client._max_rps * 0.5
                client._write_window.cap_tokens = client._write_window.cap_tokens * 0.5
                client._tennis_budget_halved = True
                log.info("TENNIS RATE BUDGET | halved shared-client budgets for this "
                         "process: read=%.1f rps, write=%.0f tokens/%.0fs window",
                         client._max_rps, client._write_window.cap_tokens,
                         client._write_window.window_sec)
            except AttributeError as e:
                # Loud by design — a client refactor must not silently restore
                # full budgets while esports runs in tandem.
                log.error("TENNIS RATE BUDGET | FAILED to halve client budgets: %s — "
                          "running at FULL shared budget", e)
        # Write-priority gate (WRITE_PRIORITY_GATE_PLAN.md): P0 = IOC fires
        # (incl. hedger — IOC by construction), P1 = cancels, posts/amends by
        # admission only. Flag-gated (write_gate.flag); flag absent =
        # passthrough = today's FIFO behavior exactly.
        self.write_gate = write_gate.GATE
        # Unified tracking logic for standard quote operations
        self.active_quotes: dict[str, ActiveQuote] = {}
        # Order IDs for which DELETE has been sent but response not yet
        # received. The orderbook scrubber (run.py) MUST skip these — during
        # the in-flight window Kalshi has likely already removed the order
        # from the book, but active_quotes still contains the entry, so
        # subtracting our size phantom-drops levels and creates 20c top
        # oscillations that the diff faithfully chases. Set ops on a Python
        # set are GIL-atomic so this is safe to mutate from to_thread.
        self._cancel_in_flight: set[str] = set()

        # IOC zero-fill phantom probe throttle. On a zero-fill we pull the REST
        # orderbook (source of truth we trust over WS) to check whether the ask
        # we fired into actually existed. During a stale-book episode the arber
        # re-fires every ~0.5s for many seconds, so throttle the REST probe to
        # at most one per ticker per interval to avoid hammering /markets.
        self._last_phantom_probe_ts: Dict[str, float] = {}
        self._PHANTOM_PROBE_MIN_INTERVAL_SEC = 3.0

    def _amend_order_shard3(self, order_id: str, ticker: str, side: str,
                            new_count: int, new_limit_cents: int,
                            client_order_id: str) -> dict:
        """Amend wrapper that adds 429 accounting. Delegates to the shared
        client's amend_order UNCHANGED on the wire: the amend endpoint routes
        by order_id (verified 26SEP01 morning session — 5/6 amends succeeded
        with no exchange_index in the body; the 404s were post-fill races,
        NOT shard mis-routing). Contrast with the batched DELETE, which DOES
        require exchange_index per entry (the 26SEP01 06:55 wipe bug). This
        wrapper exists only so a 429 on the amend path lands in the
        [429 TRACKER] totals — a probable in-flight collision with esports."""
        # exchange_index + subaccount added 2026-09-09: the amend lookup
        # defaults to shard 0 / sub 0 like every portfolio read, so the old
        # "routes by order_id" claim in the docstring above held only while
        # this fork traded the default sub. On sub 1 every amend 404'd
        # (9/9 on the first TT session) and fell back to cancel+post.
        res = self.client.amend_order(
            order_id=order_id,
            ticker=ticker,
            side=side,
            new_count=new_count,
            new_limit_cents=new_limit_cents,
            client_order_id=client_order_id,
            exchange_index=TENNIS_SHARD,
            subaccount=TENNIS_SUBACCOUNT,
        )
        if isinstance(res, dict) and res.get("status") == 429:
            _note_429("amend", ticker)
        return res

        # ── TRACK D2 (2026-08-09): what did OUR book say at probe time? ──────
        # The probe reads REST only AFTER the miss, so "REST ask is worse than
        # we fired" is EQUALLY consistent with two different failures:
        #   (a) our book was wrong        -> latency cannot fix it
        #   (b) the level was real and got taken before we arrived -> latency CAN
        # Every PHANTOM verdict logged to date conflates the two, which is why
        # the zero-fill work has stalled. The manager pushes its per-tick
        # top-of-book here (the exact dicts the fire decision was made on) so
        # the probe can compare all three views and split the verdict.
        self._ws_top_bids: Dict[str, float] = {}
        self._ws_top_offers: Dict[str, float] = {}
        self._ws_top_ts: float = 0.0

    def set_ws_top(self, top_bids: dict, top_offers: dict) -> None:
        """Manager hands over the tick's materialised top-of-book (D2).

        Stores references, not copies — the manager rebuilds these dicts every
        tick rather than mutating them in place, and copying every tick for a
        diagnostic would be real per-tick cost for no benefit.
        """
        self._ws_top_bids = top_bids or {}
        self._ws_top_offers = top_offers or {}
        self._ws_top_ts = time.time()

    async def _rest_phantom_probe(self, q: Dict[str, Any], requested: int) -> None:
        """On an IOC zero-fill, pull the REST orderbook and log whether the ask
        we fired into actually existed on Kalshi.

        We trust REST over WS (repeated live incidents where a stale/phantom WS
        book showed a takeable ask that wasn't really there — e.g. NIP/LNG
        2026-07-26, where every big arber sweep was accepted by Kalshi and
        immediately canceled with zero fill because the 32c/60c ask only
        existed in our WS view). This probe pulls /markets/{ticker}/orderbook,
        computes the real ask on the side we tried to BUY, and classifies the
        zero-fill as PHANTOM (real ask worse than our fire price → stale book)
        or REAL (book had takeable size → zero-fill is NOT explained by the
        book, investigate execution).

        Throttled per ticker; best-effort. NEVER raises into the order flow —
        an exception here must not affect trading. Killable via
        IOC_PHANTOM_REST_PROBE_DISABLE.
        """
        if os.environ.get("IOC_PHANTOM_REST_PROBE_DISABLE", "").strip():
            return
        synth = q["ticker"]
        now = time.time()
        if now - self._last_phantom_probe_ts.get(synth, 0.0) < self._PHANTOM_PROBE_MIN_INTERVAL_SEC:
            return
        self._last_phantom_probe_ts[synth] = now
        ks = (q.get("kalshi_side", "yes") or "yes").lower()
        limit = q["limit_cents"]
        try:
            real = ticker_aliases.resolve(synth)
            resp = await asyncio.to_thread(self.client.get_orderbook, real)
            book = (resp or {}).get("orderbook_fp") or {}
            # yes_dollars = YES bids; no_dollars = NO bids (= YES asks at 100-price).
            yes_levels = book.get("yes_dollars") or []
            no_levels = book.get("no_dollars") or []
            def _cents(p): return int(round(float(p) * 100))
            def _qty(x): return float(x)
            best_yes_bid = max((_cents(l[0]) for l in yes_levels), default=0)
            best_no_bid = max((_cents(l[0]) for l in no_levels), default=0)
            # Ask on the side we BOUGHT, and takeable depth at/inside our limit.
            # To BUY YES @ L we lift NO bids priced >= (100 - L); vice-versa.
            if ks == "yes":
                rest_ask = (100 - best_no_bid) if best_no_bid > 0 else 0
                depth = sum(_qty(l[1]) for l in no_levels if _cents(l[0]) >= (100 - limit))
            else:
                rest_ask = (100 - best_yes_bid) if best_yes_bid > 0 else 0
                depth = sum(_qty(l[1]) for l in yes_levels if _cents(l[0]) >= (100 - limit))
        except Exception as e:
            log.warning("IOC PHANTOM PROBE FAILED | ticker=%s side=%s err=%s", synth, ks, str(e)[:150])
            return

        if rest_ask == 0:
            verdict = "PHANTOM: REST book has no ask on this side (empty) — WS showed one that isn't there"
        elif rest_ask > limit:
            verdict = f"PHANTOM: real REST ask {rest_ask}c > fired {limit}c — fired through a stale WS book"
        else:
            verdict = (f"REAL: REST ask {rest_ask}c <= fired {limit}c with depth={depth:.0f} — "
                       f"zero-fill NOT explained by the book; investigate execution")

        # ── D2 discriminator: STALE-BOOK vs LATE ─────────────────────────────
        # Read our own top-of-book NOW and ask whether it still disagrees with
        # REST. `ws_ask` is derived exactly as the fire path derives it
        # (arber_bot.py:1253/1265): YES ask straight from top_offers; NO ask as
        # 100 - top_bid. Best-effort — never let this break the probe.
        cause = ""
        try:
            if rest_ask > limit and self._ws_top_ts:
                if ks == "yes":
                    ws_ask = self._ws_top_offers.get(synth)
                else:
                    _tb = self._ws_top_bids.get(synth)
                    ws_ask = (100.0 - _tb) if _tb is not None else None
                ws_age_ms = (now - self._ws_top_ts) * 1000.0
                if ws_ask is None:
                    cause = " | D2=UNKNOWN (no WS top for this ticker)"
                elif ws_ask <= limit:
                    # Our book STILL shows the price we fired at, after the miss.
                    # Nothing was taken from under us — the book is simply wrong.
                    cause = (f" | D2=STALE-BOOK ws_ask={ws_ask:.0f}c still<=fired "
                             f"{limit}c after the miss (ws_age={ws_age_ms:.0f}ms) "
                             f"— LATENCY WILL NOT FIX THIS")
                else:
                    # Our book has already caught up to REST. The level was real
                    # when we decided and was gone when we arrived.
                    cause = (f" | D2=LATE ws_ask={ws_ask:.0f}c now agrees with REST "
                             f"{rest_ask}c (ws_age={ws_age_ms:.0f}ms) — level was "
                             f"REAL and taken before we arrived; latency helps")
        except Exception:
            cause = " | D2=ERR"

        log.warning("IOC PHANTOM PROBE | ticker=%s side=%s fired=%dc requested=%d filled=0 | "
                    "REST ask=%sc depth@limit=%.0f | %s%s",
                    synth, ks, limit, requested, rest_ask, depth, verdict, cause)

    async def post_quotes_batch(self, quotes_to_post: List[Dict[str, Any]]) -> List[ActiveQuote]:
        """
        Takes a list of dictionaries outlining quotes to post with structure:
        {
            "side": QuoteSide.BID,
            "ticker": "...",
            "kalshi_side": "yes",  # typically always 'yes' when trading kalshi directly
            "size": 10,
            "limit_cents": 50
        }
        Places orders asynchronously through kalshi client.
        """
        if not self.trading_enabled:
            # ANSI colors — IOC fires highlighted bright magenta (most attention),
            # REST quotes yellow, BID green / OFFER red, edge green/red by sign.
            BOLD = "\033[1m"; RESET = "\033[0m"; DIM = "\033[2m"
            MAGENTA = "\033[95m"; YELLOW = "\033[93m"; CYAN = "\033[96m"
            GREEN = "\033[92m"; RED = "\033[91m"
            for q in quotes_to_post:
                tif = q.get("time_in_force", "GTC")
                is_ioc = tif == "immediate_or_cancel"
                tif_tag = f"{BOLD}{MAGENTA}IOC{RESET}" if is_ioc else f"{YELLOW}REST{RESET}"

                side_val = q["side"].value if hasattr(q.get("side"), "value") else q.get("side", "?")
                side_upper = str(side_val).upper()
                side_color = GREEN if side_upper == "BID" else (RED if side_upper == "OFFER" else "")
                side_str = f"{side_color}{side_val}{RESET}" if side_color else str(side_val)

                edge_part = ""
                adj = q.get("_adj_theo")
                if adj is not None:
                    # All orders are Kalshi buys; kalshi_side picks the book.
                    # Edge = theo on that side − our buy price.
                    kalshi_side = (q.get("kalshi_side", "yes") or "yes").lower()
                    eff_theo = adj if kalshi_side == "yes" else (100.0 - adj)
                    edge_c = eff_theo - q["limit_cents"]
                    edge_color = GREEN if edge_c >= 0 else RED
                    edge_part = f" edge={edge_color}{BOLD}{edge_c:+.1f}c{RESET}"

                trig = q.get("_trigger_type", "")
                trig_color = CYAN if trig == "series_move" else (YELLOW if trig == "map_move" else "")
                trig_part = f" trig={trig_color}{trig}{RESET}" if trig else ""

                tag_color = BOLD + MAGENTA if is_ioc else YELLOW
                log.info(f"{tag_color}[DRY]{RESET} %s %s %s %s @ {BOLD}%dc{RESET} x %d%s%s",
                         tif_tag, side_str,
                         q.get("kalshi_side", "?").upper(),
                         q["ticker"], q["limit_cents"], q["size"],
                         edge_part, trig_part)
            return []

        orders = []
        calc_map = {}
        
        for q in quotes_to_post:
            cid = str(uuid.uuid4())
            calc_map[cid] = q
            
            import telemetry_store
            telemetry_store.put_telemetry(
                cid,
                q.get("_raw_theo", 0.0),
                q.get("_adj_theo", 0.0),
                trigger_type=q.get("_trigger_type", ""),
                quoted_size=int(q.get("size", 0) or 0),
                was_size_capped=bool(q.get("_was_size_capped", False)),
                sweep_id=q.get("_sweep_id", ""),
                sweep_total_vol=int(q.get("_sweep_total_vol", 0) or 0),
                skew_shift_cents=q.get("_skew_shift_cents"),
            )
            
            # ── V2 EVENT-ORDER SCHEMA (tennis / shard 3) ──────────────────────
            # This fork posts to /portfolio/events/orders/batched, NOT the legacy
            # /portfolio/orders/batched the esports book uses. Kalshi lists the legacy
            # path for deprecation, and only the V2 path accepts exchange_index, which
            # tennis requires: tennis moved to shard 3 on 26AUG24, and a write without
            # an explicit index auto-routes and bills multiple rate buckets.
            #
            # SIDE MAPPING, derived from real orders rather than inferred:
            #     action=buy  + outcome_side=yes  ->  book_side=bid,  price = yes_price
            #     action=sell + outcome_side=no   ->  book_side=ask,  price = 1 - no_price
            # A sampled ask order carried yes_px=0.0900 / no_px=0.9100, confirming that
            # buying NO at X is the same order as selling YES at (100 - X). Getting this
            # backwards would invert every quote, so it is asserted below.
            #
            # count and price are FIXED-POINT STRINGS in V2, not integer cents.
            # self_trade_prevention_type is REQUIRED - omitting it returns
            # 400 missing_parameters (verified 26AUG26 with a live 1-lot test order).
            # SHARD GUARD. Refuse anything that is not a shard-3 tennis market before a
            # payload exists. Skips the single quote rather than raising, so one bad
            # config row cannot take down the whole batch.
            _wire = ticker_aliases.resolve(q["ticker"])
            if not _is_tennis_ticker(_wire):
                log.error("SHARD GUARD | refusing %s — not a tennis series, and this "
                          "fork stamps exchange_index=%d on every order. Check "
                          "template_quoter_config.csv.", _wire, TENNIS_SHARD)
                continue

            ks = q["kalshi_side"].lower()
            if ks not in ("yes", "no"):
                raise ValueError(f"unexpected kalshi_side {ks!r}")
            limit_c = int(q["limit_cents"])
            if ks == "yes":
                v2_side, v2_price_c = "bid", limit_c
            else:
                v2_side, v2_price_c = "ask", 100 - limit_c
            if not (0 < v2_price_c < 100):
                raise ValueError(f"V2 price {v2_price_c}c out of range for {ks} @ {limit_c}c")

            entry = {
                # Wire ticker: synthetic → real (pass-through for non-aliased).
                # Internal state (q["ticker"], ActiveQuote.ticker, telemetry
                # by client_order_id) stays on the synthetic name.
                "ticker": _wire,
                "client_order_id": cid,
                "side": v2_side,
                "count": f"{int(q['size']):.2f}",
                "price": f"{v2_price_c/100:.4f}",
                "exchange_index": TENNIS_SHARD,
                "subaccount": TENNIS_SUBACCOUNT,
            }

            if "time_in_force" in q:
                entry["time_in_force"] = q["time_in_force"]
                if q["time_in_force"] != "immediate_or_cancel":
                    # Maker quotes must never cross. Kalshi cancels (rather than
                    # crosses) the order if it would take liquidity. Prevents
                    # taker fills when our local view of top-of-book is stale.
                    entry["post_only"] = True
            else:
                entry["time_in_force"] = "good_till_canceled"
                # Default path is also a maker quote (no IOC requested).
                entry["post_only"] = True
            # taker_at_cross cancels an incoming taker rather than our resting order,
            # which is what a quoter wants; the docs pair `maker` with IOC.
            entry["self_trade_prevention_type"] = (
                "maker" if entry.get("time_in_force") == "immediate_or_cancel"
                else "taker_at_cross")

            # NOTE: sell_position_cap and cancel_order_on_pause are legacy-only fields
            # and are NOT sent on V2 - the verified-working test payload omitted them.
            # cancel_order_on_pause was a real safety property (orders auto-cancel when
            # a market pauses); without it, a paused tennis market leaves quotes resting.
            # Position caps are enforced locally by the manager instead.

            orders.append(entry)

        # Splice intersecting levels securely into bounded batch arrays.
        # Bumped 5 → 10 on 2026-06-06: Kalshi's per-batch hard cap is 20, our
        # writes were getting 47.8% 429-rate at small batch counts because each
        # batched POST is a separate API call against a tight per-endpoint
        # write limit. Doubling batch size halves the number of POSTs for the
        # same N quotes, materially cutting 429 frequency.
        BATCH_SIZE = 10

        # ── Write-priority partition (2026-08-02, WRITE_PRIORITY_GATE_PLAN) ──
        # IOC entries (arber/momentum fires + hedger, IOC by construction, or
        # an explicit q["_priority"]=="P0" tag) ride the P0 lane: they take
        # the next write slot ahead of any queued housekeeping. Resting posts
        # go by ADMISSION: granted only when the gate is idle; declined
        # entries are DISCARDED here and re-derived by the caller's next tick
        # — a declined post is never stored, so no stale price can reach the
        # wire later (operator invariant). Batches never mix classes.
        def _is_p0(entry):
            if entry.get("time_in_force") == "immediate_or_cancel":
                return True
            q = calc_map.get(entry.get("client_order_id"))
            return bool(q and q.get("_priority") == "P0")

        def _splice(entry_list):
            spliced = []
            seen_keys = []
            for entry in entry_list:
                key = f"{entry['ticker']}_{entry['side']}"
                placed = False
                # Map structural uniqueness by sequentially checking parallel
                # batch layers organically
                for b_idx in range(len(spliced)):
                    if key not in seen_keys[b_idx] and len(spliced[b_idx]) < BATCH_SIZE:
                        spliced[b_idx].append(entry)
                        seen_keys[b_idx].add(key)
                        placed = True
                        break
                if not placed:
                    spliced.append([entry])
                    seen_keys.append({key})
            return spliced

        p0_batches = _splice([e for e in orders if _is_p0(e)])
        admit_batches = _splice([e for e in orders if not _is_p0(e)])
        batches = [("P0", b) for b in p0_batches] + [("ADMIT", b) for b in admit_batches]

        batch_orders = []

        # 429 retry with exponential backoff. kalshi_client._post returns {} on
        # any error (429, timeout, 5xx) so we treat empty response as "retry".
        # Backoff sequence chosen so the cumulative delay (0.5 + 1.5 + 3.0 ≈ 5s)
        # is less than the quoter's typical reprice interval — if all 3 attempts
        # fail, the next reprice cycle will try again rather than dropping the
        # order silently. Was: silent swallow of 429 left bot in stale-cache
        # state where reconciliation had to clean up (2026-06-06 incident).
        _POST_RETRY_DELAYS_SEC = ()

        for i, (gate_class, batch) in enumerate(batches):
            # 2026-06-08: removed 0.35s inter-batch sleep. The kalshi_client
            # _write_inflight_lock already serializes batched POSTs at HTTP-
            # RTT speed (~30ms each), and the rolling-window limiter handles
            # rate gating. With V2 endpoints in place, 350ms padding is pure
            # latency drag (was needed to dodge legacy concurrent-bucket caps,
            # which V2 doesn't have).

            # ── Gate: admission check for maker posts ──
            gate_tok = None
            if gate_class == "ADMIT":
                gate_tok = self.write_gate.try_slot("post")
                if gate_tok is None:
                    # Declined: discard the whole batch. The quoter's next
                    # tick re-derives these quotes from live state — nothing
                    # is stored. Track per-ticker darkness (WARN 5s/ALERT 15s).
                    for entry in batch:
                        q = calc_map.get(entry.get("client_order_id")) or {}
                        self.write_gate.note_post_declined(q.get("ticker", entry.get("ticker", "?")))
                    log.info(f"[POST DECLINED] gate contended — dropped batch of "
                             f"{len(batch)} maker post(s); next tick re-derives")
                    continue
            else:
                gate_tok = self.write_gate.slot(write_gate.P0_FIRE, "fire")

            batch_body = {"orders": batch}
            log.debug(f"KALSHI API TRACE | Firing dynamically spliced physical batch {i+1}/{len(batches)} size: {len(batch)} Payload: {batch_body}")

            resp = None
            async with gate_tok as slt:
                for attempt, delay in enumerate([0.0] + list(_POST_RETRY_DELAYS_SEC)):
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # client_order_ids deliberately preserved across retries.
                    # Tested 2026-06-06: regenerating UUIDs on retry DOUBLED the
                    # 429 rate (34.6% → 78.9%). Hypothesis: Kalshi uses
                    # client_order_id for idempotency — same CID on retry is
                    # treated as duplicate and skipped, while fresh CID is treated
                    # as a new order against some creation-rate limit.
                    # 2026-09-01: was `self.client._post(V2_BATCHED_PATH, ...)` — that
                    # passthrough runs _post_internal(order_count=0, batched=False):
                    # NO write tokens charged, NO inflight lock, status invisible.
                    # Call _post_internal directly: tokens metered (10/order),
                    # batched inflight lock held, and a 429 is now countable.
                    def _post_batched_tracked(bb=batch_body, n=len(batch)):
                        r, status = self.client._post_internal(
                            V2_BATCHED_PATH, bb, order_count=n, batched=True,
                            return_status=True)
                        if status == 429:
                            _note_429("post", f"batch of {n}")
                        return r
                    resp = await asyncio.to_thread(_post_batched_tracked)
                    log.debug(f"KALSHI RESPONSE NATIVE (attempt {attempt+1}): {resp}")
                    # A successful response is a dict with 'orders' key (even if some
                    # entries inside have per-order errors). An empty {} indicates the
                    # POST itself failed (429/timeout/5xx) — retry.
                    if isinstance(resp, dict) and resp:
                        break
                    if attempt < len(_POST_RETRY_DELAYS_SEC):
                        log.warning(f"BATCHED POST | attempt {attempt+1} returned empty (likely 429/timeout). "
                                    f"Retrying batch {i+1}/{len(batches)} after {_POST_RETRY_DELAYS_SEC[attempt]}s "
                                    f"(size={len(batch)} orders)")
                    else:
                        log.error(f"BATCHED POST EXHAUSTED | {len(_POST_RETRY_DELAYS_SEC)+1} attempts failed for "
                                  f"batch {i+1}/{len(batches)} (size={len(batch)} orders). Orders NOT placed; "
                                  f"reconciliation will refire on next cycle.")

            # Post-batch gate accounting: fires get their latency split logged
            # (queue_wait = time behind other writes — THE number the gate
            # exists to crush); admitted maker posts clear maker-dark state.
            if gate_class == "P0":
                log.info(f"[FIRE LATENCY] n={len(batch)} "
                         f"queue_wait={slt.queue_wait_s * 1000:.0f}ms "
                         f"wire={slt.wire_s * 1000:.0f}ms "
                         f"tickers={','.join(sorted({e['ticker'].split('-')[-1] for e in batch}))}")
            else:
                for entry in batch:
                    q = calc_map.get(entry.get("client_order_id")) or {}
                    self.write_gate.note_post_admitted(q.get("ticker", entry.get("ticker", "?")))

            # Aggregate cleanly safely
            if isinstance(resp, dict):
                batch_orders.extend(resp.get("orders", []))

        new_active = []
        for entry in batch_orders:
            if not isinstance(entry, dict): 
                continue
            
            # Kalshi V2 Batched Response format sometimes omits the nested "order" dict
            order_data = entry.get("order", entry)
            err = entry.get("error", order_data.get("error"))
            
            cid = order_data.get("client_order_id", "")
            q = calc_map.get(cid)
            
            if not q: 
                log.error("CRITICAL FATAL: Failed to resolve Client ID map back to math array!!")
                continue
            
            if err:
                # A rejected IOC (taker) is another "edge fired, captured nothing"
                # path — most often self-trade prevention cancelling at the cross.
                # Tag the source so arber fire rejections are greppable, not buried
                # among routine maker reprice failures.
                if q.get("time_in_force") == "immediate_or_cancel":
                    log.warning("IOC FIRE REJECTED | ticker=%s side=%s kalshi_side=%s price=%dc requested=%d error=%s"
                                " — edge fired, captured NOTHING.",
                                q["ticker"], q["side"].value, q["kalshi_side"], q["limit_cents"], q["size"], err)
                else:
                    log.warning("QUOTE FAIL | ticker=%s side=%s error=%s", q["ticker"], q["side"].value, err)
                # Catch closed/erased markets instantly to flag upstream for strict algorithmic teardown
                if isinstance(err, dict) and err.get("code") in ["market_not_found", "market_closed", "market_settled"]:
                    quote = ActiveQuote(
                        order_id="",
                        client_order_id=cid,
                        side=q["side"],
                        ticker=q["ticker"],
                        kalshi_side=q["kalshi_side"],
                        limit_cents=q["limit_cents"],
                        size=q["size"],
                        filled_count=0,
                        status=QuoteStatus.FAILED
                    )
                    quote._source_q = q
                    new_active.append(quote)
                continue
                
            oid = order_data.get("order_id", "")
            fc = int(float(order_data.get("fill_count_fp", 0))) if isinstance(order_data, dict) else 0
            status = QuoteStatus.LIVE if fc == 0 else QuoteStatus.PARTIALLY_FILLED

            import time as _time
            quote = ActiveQuote(
                order_id=oid,
                client_order_id=cid,
                side=q["side"],
                ticker=q["ticker"],
                kalshi_side=q["kalshi_side"],
                limit_cents=q["limit_cents"],
                size=q["size"],
                filled_count=fc,
                status=status,
                placement_ts=_time.time(),
            )
            # Track quote locally
            self.active_quotes[cid] = quote
            register_order(quote)

            # Map original payload logic securely for orchestration tracking
            quote._source_q = q
            
            new_active.append(quote)

            # log.info("QUOTE POST | side=%s ticker=%s price=%dc size=%d order=%s",
            #          q["side"].value, q["ticker"], q["limit_cents"], q["size"], oid)

            is_taker = q.get("time_in_force") == "immediate_or_cancel"
            source = "TAKER/ARBER" if is_taker else "MAKER/QUOTER"
            arb_ctx = q.get("_arb_context", "")

            if fc > 0:
                log.info("IMMEDIATE FILL | side=%s filled=%d/%d on post [%s]", q["side"].value, fc, q["size"], source)
                ctx_line = f" Context: {arb_ctx}\n" if arb_ctx else ""
                log.info(f"\n======== PHYSICAL TRADE EXECUTED [{source}] ========\n"
                         f" Action : BUY {q['kalshi_side'].upper()} {fc}x/{q['size']}x\n"
                         f" Ticker : {q['ticker']}\n"
                         f" Price  : {q['limit_cents']}c\n"
                         f" Source : {source}\n"
                         f"{ctx_line}"
                         f"{'=' * 48}\n")

            # Surplus over-fire that actually filled = depth WS under-reported.
            # The whole surplus order is "extra" (it's fired on top of the visible
            # sweep), so fc IS the extra lots captured. This is the payoff metric
            # for the over-fire change — grep `ARBER SURPLUS CAPTURED`.
            if q.get("_is_surplus") and fc > 0:
                log.info("ARBER SURPLUS CAPTURED | ticker=%s side=%s price=%dc EXTRA=%d lots "
                         "(WS-visible on this leg was %s) — over-fire paid off.",
                         q["ticker"], q["kalshi_side"], q["limit_cents"], fc, q.get("_visible_vol", "?"))

            # 2026-07-26: An IOC (taker/arber) order that returns fc==0 was
            # ACCEPTED by Kalshi but matched NOTHING — the taker slot is dead on
            # arrival (no crossing liquidity by the time it landed, or self-trade
            # prevention cancelled it at the cross). Previously this logged
            # nothing (the only fire log was gated on fc>0), so the arber's
            # LEAD-LAG line would claim a live edge and "fire" while the fill
            # silently returned zero — activity in the log, nothing on the book.
            # An unfilled fire on a live edge is a first-class incident. Surface
            # every IOC outcome loudly and greppably (tags: IOC ZERO-FILL /
            # IOC PARTIAL FILL / IOC FULL FILL; rejections: IOC FIRE REJECTED).
            if is_taker:
                ctx_line = f" ctx={arb_ctx}" if arb_ctx else ""
                # ── SURPLUS TAG (2026-08-09) ─────────────────────────────────
                # Surplus over-fire (arber_bot.py, enable_arber_surplus.flag, on
                # since 26JUL) deliberately sends ~2x extra IOC size at a price
                # whose visible liquidity the PRIMARY order just consumed. It is
                # SUPPOSED to mostly zero-fill — that is the cost of the option
                # on the book being deeper than WS showed.
                #
                # Untagged, those zero-fills were indistinguishable from real
                # ones and dominated every census: measured 26AUG09 over ~3.5h,
                # 27 of 31 zero-fills were surplus, so the headline rate read 38%
                # while the TRUE rate on visible fires was 4/43 = 9%.
                is_surplus = bool(q.get("_is_surplus"))
                sfx = f" surplus={'T' if is_surplus else 'F'}"
                if fc == 0:
                    log.warning(
                        "IOC ZERO-FILL | ticker=%s side=%s kalshi_side=%s price=%dc requested=%d filled=0 order=%s%s"
                        " — edge fired, captured NOTHING.%s",
                        q["ticker"], q["side"].value, q["kalshi_side"], q["limit_cents"], q["size"], oid or "-",
                        sfx, ctx_line,
                    )
                    # Probe ONLY visible fires. On a surplus zero-fill the probe
                    # reads REST *after our own primary order took the level*, so
                    # it sees a worse ask and returns PHANTOM every time — we were
                    # measuring our own market impact and calling it a stale book.
                    # That artifact is what produced the 97:1 PHANTOM:REAL split
                    # against a book the Track D audit shows is ~99% accurate.
                    if not is_surplus:
                        await self._rest_phantom_probe(q, q["size"])
                elif fc < q["size"]:
                    log.warning(
                        "IOC PARTIAL FILL | ticker=%s side=%s kalshi_side=%s price=%dc filled=%d/%d order=%s%s"
                        " — %d lots of edge left unfilled.%s",
                        q["ticker"], q["side"].value, q["kalshi_side"], q["limit_cents"], fc, q["size"],
                        oid or "-", sfx, q["size"] - fc, ctx_line,
                    )
                else:
                    log.info(
                        "IOC FULL FILL | ticker=%s side=%s kalshi_side=%s price=%dc filled=%d/%d order=%s%s%s",
                        q["ticker"], q["side"].value, q["kalshi_side"], q["limit_cents"], fc, q["size"], oid or "-",
                        sfx, ctx_line,
                    )

        return new_active


    async def amend_quotes_batch(
        self,
        amend_pairs: List[tuple],
    ) -> tuple:
        """Loop-amend (aq, desired_q) pairs one at a time via V2 amend endpoint.

        V2 has no batched-amend endpoint, so we sequence single calls. Each
        amend pays 10 write tokens via the same `_post_internal` plumbing
        as POST. Caller's rolling-window budget governs throughput.

        For each pair:
          - new_count = aq.filled_count + desired.size  (TOTAL, not delta)
          - amend_order returns a discriminable dict: {"ok": True, ...} on
            success, else {"ok": False, "kind": "reject"|"ambiguous", ...}
          - On success: mutate aq.limit_cents and aq.size in-place (preserves
            order_id + client_order_id; aq is already in the manager's
            active_quotes_by_bot via kept binding)
          - On failure: pair routes through cancel+post fallback by caller,
            carrying its `kind` so the caller can repost immediately (reject)
            or defer the repost until the cancel confirms (ambiguous)

        Returns:
          (succeeded_aqs, failed_pairs)
            succeeded_aqs: List[ActiveQuote] — mutated in-place, same ref
            failed_pairs:  List[Tuple[ActiveQuote, Dict, str]] — (aq, q, kind)
                           where kind is "reject" or "ambiguous"; re-route via
                           cancel + post in the caller

        Dry-run: short-circuits with all-succeeded, mutating local state only.
        """
        if not amend_pairs:
            return [], []

        succeeded: List[ActiveQuote] = []
        failed: List[tuple] = []
        # Pairs declined by the write gate (admission contended). Deliberately
        # in NEITHER succeeded NOR failed: the aq stays live at its old price
        # and next tick's diff re-derives. Tracked separately so the
        # conservation check below can tell a benign decline from a genuinely
        # orphaned pair.
        gate_declined: List[ActiveQuote] = []

        if not self.trading_enabled:
            DIM = "\033[2m"; RESET = "\033[0m"
            for aq, q in amend_pairs:
                old_price = aq.limit_cents
                old_size = aq.size
                aq.limit_cents = q.get("limit_cents", aq.limit_cents)
                aq.size = q.get("size", aq.size)
                log.info(
                    f"{DIM}[DRY] AMEND %s %s %s @ %dc x %d -> %dc x %d (order=%s){RESET}",
                    aq.side.value if hasattr(aq.side, "value") else aq.side,
                    aq.kalshi_side.upper() if isinstance(aq.kalshi_side, str) else "?",
                    aq.ticker,
                    old_price,
                    old_size,
                    aq.limit_cents,
                    aq.size,
                    aq.order_id,
                )
                succeeded.append(aq)
            return succeeded, failed

        # 2026-06-28 LOUD DIAGNOSTIC LOGGING — hunting the silent amend
        # dropout that caused the 3d048426 orphan at 15:27 ECHSHK. Every
        # path in this function must emit a log line. After every pair is
        # processed we check sum(succeeded) + sum(failed) == len(input);
        # if not, we know the loop bailed early without telling us.
        n_input = len(amend_pairs)
        log.info(
            f"[AMEND BATCH START] n={n_input} oids=[{','.join(str(aq.order_id) or '?' for aq, _ in amend_pairs[:8])}]"
        )
        try:
            for idx, (aq, q) in enumerate(amend_pairs):
                if not aq.order_id:
                    log.warning(
                        f"[AMEND SKIP NO_OID] {aq.ticker} side={aq.kalshi_side} "
                        f"price={aq.limit_cents} idx={idx}/{n_input} — "
                        f"client_order_id={aq.client_order_id} — routing to cancel+post"
                    )
                    # No order_id means we never reached the wire — nothing is
                    # live at a new price, and there's no oid to cancel against,
                    # so reposting is the only recourse. Treat as a hard reject.
                    failed.append((aq, q, "reject"))
                    continue

                new_count = (aq.filled_count or 0) + int(q.get("size", 0))
                new_limit_cents = int(q.get("limit_cents", aq.limit_cents))

                def _do_amend(order_id=aq.order_id,
                              # Wire ticker: synthetic → real (pass-through for
                              # non-aliased). aq.ticker stays synthetic for the
                              # bot's internal state.
                              ticker=ticker_aliases.resolve(aq.ticker),
                              side=aq.kalshi_side, count=new_count,
                              price=new_limit_cents, coid=aq.client_order_id):
                    return self._amend_order_shard3(
                        order_id=order_id,
                        ticker=ticker,
                        side=side,
                        new_count=count,
                        new_limit_cents=price,
                        client_order_id=coid,
                    )

                # BaseException catches asyncio.CancelledError too (Python 3.8+
                # made CancelledError inherit from BaseException, NOT Exception).
                # If the outer task is cancelled while await is pending, this
                # is the path; without explicit handling the cancellation
                # would propagate up without ever logging the in-flight aq.
                # ── Gate: amends are admission-class (never queued). Declined
                # → this pair goes to NEITHER succeeded NOR failed: the aq
                # stays live at its old price and the next tick's diff
                # re-derives the amend from fresh state. Routing declines to
                # `failed` would trigger cancel+post fallback — wrong: the
                # quote isn't broken, the write slot is just busy.
                _gate_tok = self.write_gate.try_slot("amend")
                if _gate_tok is None:
                    self.write_gate.stats["amend_declined"] += 1
                    gate_declined.append(aq)
                    continue
                try:
                    async with _gate_tok:
                        resp = await asyncio.to_thread(_do_amend)
                except asyncio.CancelledError as ce:
                    log.error(
                        f"[AMEND CANCELLED] {aq.ticker} order={aq.order_id} "
                        f"side={aq.kalshi_side} price={aq.limit_cents}->{new_limit_cents} "
                        f"idx={idx}/{n_input} — outer task cancelled mid-await. "
                        f"Routing remaining {n_input - idx - 1} unprocessed pair(s) to failed. "
                        f"Re-raising so the caller's await also cancels cleanly."
                    )
                    # Sweep all remaining (current + rest) into failed BEFORE
                    # re-raising so the caller can route them to cancel+post.
                    # NOTE: caller currently doesn't capture this — it has
                    # try/except Exception which won't catch CancelledError —
                    # but the partial-failed-list is at least visible in logs
                    # post-incident. The in-flight pair is ambiguous (request
                    # may have landed); mark the rest ambiguous too for safety.
                    failed.append((aq, q, "ambiguous"))
                    for rest_aq, rest_q in amend_pairs[idx + 1:]:
                        failed.append((rest_aq, rest_q, "ambiguous"))
                        log.error(
                            f"[AMEND CANCELLED-SWEEP] {rest_aq.ticker} order={rest_aq.order_id} "
                            f"side={rest_aq.kalshi_side} — added to failed list (will not retry mid-cancel)"
                        )
                    raise
                except BaseException as e:
                    log.warning(
                        f"[AMEND EXCEPTION] {aq.ticker} order={aq.order_id} "
                        f"side={aq.kalshi_side} price={aq.limit_cents}->{new_limit_cents} "
                        f"idx={idx}/{n_input} err_type={type(e).__name__} err={e!r} — "
                        f"routing to cancel+post"
                    )
                    # Unknown whether the amend reached the wire → ambiguous.
                    failed.append((aq, q, "ambiguous"))
                    continue

                if resp.get("ok"):
                    old_price = aq.limit_cents
                    old_size = aq.size
                    aq.limit_cents = new_limit_cents
                    aq.size = int(q.get("size", aq.size))
                    log.info(
                        f"[AMEND OK] {aq.ticker} order={aq.order_id} "
                        f"side={aq.kalshi_side} {old_price}c x {old_size} -> "
                        f"{aq.limit_cents}c x {aq.size}"
                    )
                    succeeded.append(aq)
                else:
                    # Discriminable failure: "reject" (order in a known state,
                    # safe to cancel+post now) vs "ambiguous" (amend may have
                    # applied — caller cancels and DEFERS the repost). Default
                    # to ambiguous if a caller/mocked client omits the kind.
                    kind = resp.get("kind", "ambiguous") if isinstance(resp, dict) else "ambiguous"
                    log.info(
                        f"[AMEND FAIL] {aq.ticker} order={aq.order_id} "
                        f"side={aq.kalshi_side} price={aq.limit_cents}->{new_limit_cents} "
                        f"size={aq.size}->{q.get('size')} idx={idx}/{n_input} "
                        f"kind={kind} status={resp.get('status') if isinstance(resp, dict) else '?'} — "
                        f"routing to cancel+post"
                    )
                    failed.append((aq, q, kind))
        finally:
            # Conservation check: every input pair must end up in either
            # succeeded or failed. If we see a mismatch, the bug we're hunting
            # is alive and we can pinpoint the unaccounted pairs.
            n_out = len(succeeded) + len(failed) + len(gate_declined)
            if n_out != n_input:
                missing_idxs = [
                    idx for idx, (aq, _) in enumerate(amend_pairs)
                    if aq not in succeeded
                    and not any(aq is fa[0] for fa in failed)
                    and aq not in gate_declined
                ]
                missing_oids = [amend_pairs[i][0].order_id for i in missing_idxs]
                log.error(
                    f"[AMEND BATCH CONSERVATION VIOLATION] in={n_input} "
                    f"out={n_out} (succeeded={len(succeeded)}, failed={len(failed)}, "
                    f"gate_declined={len(gate_declined)}) "
                    f"missing_idxs={missing_idxs} missing_oids={missing_oids} — "
                    f"these orders are now ORPHANED on Kalshi and untracked locally."
                )
            else:
                log.info(
                    f"[AMEND BATCH END] in={n_input} "
                    f"succeeded={len(succeeded)} failed={len(failed)} "
                    f"gate_declined={len(gate_declined)} — conservation OK"
                )

        return succeeded, failed

    async def cancel_quotes_batch(self, quotes_to_cancel: List[ActiveQuote],
                                  priority: Optional[int] = None) -> None:
        """
        Batch cancels existing quotes using one network request.

        `priority`: write-gate class, default P1 (routine cancels). The
        manager's pre-fire self-match guard passes P0 so the protective
        cancel that clears our own crossing quotes rides the fire lane and
        can't queue behind unrelated housekeeping cancels in a storm.
        """
        if not quotes_to_cancel:
            return

        order_ids = [q.order_id for q in quotes_to_cancel if q.order_id]
        if not order_ids:
            return

        if not self.trading_enabled:
            DIM = "\033[2m"; RESET = "\033[0m"
            for q in quotes_to_cancel:
                log.info(f"{DIM}[DRY] CANCEL %s %s %s @ %dc x %d (order=%s){RESET}",
                         q.side.value if hasattr(q.side, "value") else q.side,
                         q.kalshi_side.upper() if isinstance(q.kalshi_side, str) else "?",
                         q.ticker,
                         q.limit_cents,
                         q.size,
                         q.order_id)
                q.status = QuoteStatus.CANCELLED
                self.active_quotes.pop(q.client_order_id, None)
                unregister_order(q.order_id)
            return

        # V2 batched DELETE — body uses `orders` (array of {order_id} objects)
        # not `ids` (array of strings). Endpoint is /portfolio/events/orders/batched.
        # Migrated 2026-06-08 after probe established V2 enforces documented
        # Advanced tier vs legacy's ~60-token-per-2s cap.
        path = "/trade-api/v2/portfolio/events/orders/batched"

        def send_cancel():
            """Returns the set of order_ids whose DELETE chunk got 200/204.

            Order_ids in failed chunks are intentionally NOT included so the
            caller leaves them in active_quotes. The orderbook scrubber will
            keep subtracting their volume (since they're still resting on
            Kalshi), and manager._verify_pending_cancels will retry until
            confirmed dead. This is the fix for the silent-cancel-failure
            bug: previously the bot pop'd locally regardless of HTTP outcome,
            so a timed-out DELETE left orphan resting orders on Kalshi while
            the local view forgot they existed → scrubber misalignment →
            stale top → taker fills.
            """
            import requests
            import time
            confirmed: set[str] = set()
            # Attempt to safely resolve the user's KALSHI_API_BASE dynamically
            api_base = "https://external-api.kalshi.com"
            import sys
            if "config" in sys.modules and hasattr(sys.modules["config"], "KALSHI_API_BASE"):
                api_base = sys.modules["config"].KALSHI_API_BASE

            url = f"{api_base}{path}"
            headers = self.client.auth.get_headers("DELETE", path)
            headers["Content-Type"] = "application/json"

            # Chunk mathematically to safely respect Kalshi's strict 20 items per Array maximum limits
            chunk_size = 20
            chunks = [order_ids[i:i + chunk_size] for i in range(0, len(order_ids), chunk_size)]

            for index, chunk in enumerate(chunks):
                if index > 0:
                    time.sleep(0.35) # Stagger structurally to dodge micro-burst execution locks

                # Charge the kalshi_client write-token bucket BEFORE firing the DELETE.
                # 2 tokens per cancel; uses cancel reservation (high priority — will
                # always go through, posts back off to leave headroom for these).
                # CRITICAL: this DELETE path bypasses kalshi_client._post, so we
                # explicitly route through _consume_write_tokens to stay accounted.
                throttle_ms = 0.0
                try:
                    throttle_ms += self.client._wait_for_token()
                    throttle_ms += self.client._consume_write_tokens(len(chunk) * 2, is_cancel=True)
                except Exception as _e:
                    log.warning(f"cancel batch: token-bucket gating failed ({_e}); proceeding")

                # V2 batched DELETE body: {"orders": [{"order_id": "..."}, ...]}
                # exchange_index is REQUIRED per cancel entry, not just per order.
                # Cancel entries are shard-scoped exactly like reads: a DELETE that
                # omits it auto-routes, and on a shard-3 order that risks a silent
                # no-op leaving the quote resting. Verified 26AUG26 - a 1-lot shard-3
                # order cancelled cleanly with the index present (reduced_by 1.00).
                v2_body = {"orders": [{"order_id": oid,
                                       "exchange_index": TENNIS_SHARD,
                                       "subaccount": TENNIS_SUBACCOUNT}
                                      for oid in chunk]}
                _t0 = time.time()
                try:
                    resp = requests.delete(url, json=v2_body, headers=headers, timeout=5.0)
                    # Route through kalshi_client._api_trace so this DELETE appears
                    # in the trace CSV alongside everything else (was a blind spot —
                    # cancels via this path were invisible in the trace until now).
                    try:
                        from kalshi_client import _api_trace as _tr
                        _tr("DELETE", path, resp.status_code, throttle_ms,
                            (time.time() - _t0) * 1000.0)
                    except Exception:
                        pass
                    if resp.status_code == 429:
                        _note_429("cancel", f"chunk of {len(chunk)}")
                    if resp.status_code in (200, 204):
                        # Kalshi's batched DELETE returns 200 with per-order
                        # errors embedded in the response body:
                        #   {"orders":[{"order_id":"...","reduced_by":"1"},
                        #              {"error":{"code":"not_found",...},
                        #               "order_id":"..."}]}
                        # Confirming an order whose per-order entry actually
                        # carried an error pops it from active_quotes locally
                        # even though Kalshi may NOT have cancelled it — the
                        # bot then forgets about a still-resting order →
                        # orphan layering on the next quote cycle.
                        # Bug verified 2026-06-27 via mixed-real+fake DELETE
                        # probe (T7b in the endpoint audit). Fix: trust the
                        # per-order result, not the HTTP status, when the body
                        # is parseable. Fall back to optimistic-confirm-all
                        # only on body parse failure (preserves prior behavior
                        # for any caller that doesn't return JSON).
                        try:
                            body = resp.json() if resp.content else {}
                        except Exception as _je:
                            log.warning("Batch cancel: 200 with unparseable body (%s); "
                                        "optimistically confirming chunk", _je)
                            confirmed.update(chunk)
                        else:
                            entries = body.get("orders", []) if isinstance(body, dict) else []
                            if not entries:
                                # 200 with no per-order array — treat as
                                # blanket success to preserve prior behavior.
                                confirmed.update(chunk)
                            else:
                                for entry in entries:
                                    oid = entry.get("order_id")
                                    err = entry.get("error")
                                    if oid and not err:
                                        confirmed.add(oid)
                                    elif err:
                                        # `not_found` is TERMINAL, not ambiguous.
                                        # The orphan-layering fix above (26JUN27)
                                        # rightly refuses to confirm errored orders
                                        # — but its premise is "Kalshi may NOT have
                                        # cancelled it, so it may still be resting".
                                        # That premise fails for not_found, where
                                        # Kalshi states the order does not exist
                                        # (filled / already cancelled / expired).
                                        # Leaving it unconfirmed keeps it in
                                        # active_quotes, `_track_pending_cancels`
                                        # re-enqueues it, and we cancel it forever:
                                        # measured 26AUG19, 3,426 errors from just
                                        # 152 oids (96% duplicates, worst oid
                                        # retried 28x), ~18k/day burning
                                        # P1_CANCEL write-gate slots AHEAD of
                                        # posts/amends. Evict it; every OTHER error
                                        # code stays unconfirmed and retries.
                                        if _is_not_found(err):
                                            if oid:
                                                confirmed.add(oid)
                                            log.info("Batch cancel oid=%s already gone "
                                                     "(not_found) — treating as cancelled",
                                                     oid or "?")
                                        else:
                                            log.warning("Batch cancel per-order error oid=%s: %s",
                                                        oid or "?", err)
                                # Any chunk order_id NOT present in the response
                                # array is also untrusted — leave unconfirmed.
                    else:
                        log.error("Native batch cancel error chunk %d: %s - %s", index, resp.status_code, resp.text)
                except Exception as e:
                    try:
                        from kalshi_client import _api_trace as _tr
                        _tr("DELETE", path, -1, throttle_ms, (time.time() - _t0) * 1000.0)
                    except Exception:
                        pass
                    log.error("Native batch cancel network exception chunk %d: %s", index, e)
            return confirmed

        # Mark all order_ids as cancel-in-flight BEFORE the DELETE goes out.
        # Scrubber will skip these so a Kalshi-already-processed-cancel
        # doesn't get phantom-subtracted by a still-present active_quotes
        # entry. Cleared below regardless of DELETE outcome (the scrub-skip
        # only needs to hold during the actual in-flight window).
        for oid in order_ids:
            self._cancel_in_flight.add(oid)
        try:
            # ── Gate: cancels are P1 — delay-safe (a late cancel is merely
            # conservative), queued behind fires only, ahead of everything else.
            # Pre-fire self-match cancels arrive with priority=P0 (see docstring).
            _prio = write_gate.P1_CANCEL if priority is None else priority
            async with self.write_gate.slot(_prio, "cancel_batch"):
                confirmed_ids = await asyncio.to_thread(send_cancel)
        finally:
            for oid in order_ids:
                self._cancel_in_flight.discard(oid)

        for q in quotes_to_cancel:
            if q.order_id in confirmed_ids:
                q.status = QuoteStatus.CANCELLED
                self.active_quotes.pop(q.client_order_id, None)
                unregister_order(q.order_id)
            # else: leave in active_quotes. Scrubber needs the entry because
            # the order is (most likely) still resting on Kalshi. Manager's
            # _track_pending_cancels has already enqueued the retry path.
