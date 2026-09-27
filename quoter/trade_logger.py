import csv
import logging
import os
import threading
import time
from datetime import datetime
from typing import List, Dict, Any

# Ticker alias layer — translate real Kalshi tickers (on the fill stream)
# back to the bot's synthetic names so internal consumers (position_store,
# hedge buffer, trades.csv, is_series gate) all stay keyed on synthetic.
# Pass-through for any non-aliased ticker (the common case).
import ticker_aliases

# A session must survive this long before its reconnect backoff is forgiven.
# Without it, resetting the delay on every connect turns a connect->error->
# reconnect cycle into a 1s hot loop (see WSSubscriptionDead below).
WS_STABLE_SESSION_S = 60.0


class WSSubscriptionDead(Exception):
    """Kalshi sent an `error` frame — the subscription is gone, the socket is not.

    26AUG27 (esports stack): `{'code': 25, 'msg': 'Subscription buffer overflow'}`
    arrived at 12:01:52Z and was merely logged, so no exception reached the
    reconnect handler and `_ws_ready` stayed True. The listener sat connected and
    deaf for ~3h while 193 fills (~24,000 lots across 12 tickers) bypassed
    position_store, the hedge buffer and trades.csv — max_position and the
    position adjuster traded blind the whole time. An error frame is a
    DISCONNECT: raise it so the existing handler resets _ws_ready and reconnects
    with backoff. Ported to the tennis stack the same day.
    """

log = logging.getLogger(__name__)

class TradeLogger:
    """
    Independent autonomous watcher. It polls the server natively bypassing all internal Quoter 
    state logic to maintain pristine tracking of actual physical fills confirmed by Kalshi!
    """
    def __init__(self, kalshi_client):
        self.client = kalshi_client
        self.recorded_trades: List[Dict[str, Any]] = []
        self.seen_fills = set()
        self.csv_path = "trades.csv"
        # PTA-only: cumulative-fill tracking per order_id for taker full-fill
        # tagging. Multi-fill IOC orders (one POST → N WS events) need to be
        # aggregated to detect "we got ≥80% of what we asked for" across the
        # whole order, not per-event. Map: order_id → {"cum": int, "qsz": int,
        # "tagged": bool}. Never read by execution; logging-only.
        self._taker_cumulative: Dict[str, Dict[str, Any]] = {}
        
        self._shutdown = threading.Event()
        self._ws_ready = False  # True only after WS subscription confirmed — prevents replay fills from triggering hedger
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TradeLogger")
        
    def start(self):
        """Ignite the autonomous physical detection background thread natively so it never blocks the quoter!"""
        self._thread.start()
        log.info("TRADE LOGGER | Autonomous background physical capture daemon engaged seamlessly.")

    def _run_loop(self) -> None:
        """Seamlessly shift to a decoupled asynchronous WebSocket sequence internally!"""
        import asyncio
        asyncio.run(self._ws_loop())

    async def _ws_loop(self) -> None:
        import websockets
        import json
        import asyncio
        import time
        
        # 2026-06-08: migrated to Kalshi's dedicated external WS host
        # (per Kalshi Discord). Old: wss://api.elections.kalshi.com — the
        # CloudFront-fronted legacy that may stop working at some point.
        WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
        WS_PATH = "/trade-api/ws/v2"
        
        reconnect_delay = 1.0
        log.info("TRADE LOGGER | Background WebSocket thread successfully ignited...")
        # Pre-bound: the reconnect handler reads it, and the try block can raise
        # before the per-session stamp (e.g. auth header signing failing).
        session_start = time.monotonic()
        
        # ── One-time REST pull for recent historical trades! ──
        try:
            log.info("TRADE LOGGER | Syphoning recent native fills from Kalshi REST gateway...")
            # Scope the preload to our subaccount: fills reads default to sub 0
            # (same convention as /portfolio/orders, verified 2026-09-09), and
            # the preload only exists to dedupe OUR fills against the WS stream.
            from execution import TENNIS_SUBACCOUNT as _our_sub
            resp = self.client._get("/trade-api/v2/portfolio/fills",
                                    {"limit": 100, "subaccount": _our_sub})
            if resp and "error" not in resp:
                fills = resp.get("fills", [])
                fills.sort(key=lambda x: x.get("created_time", ""), reverse=False)
                for f in fills:
                    trade_id = f.get("trade_id")
                    if trade_id:
                        self.seen_fills.add(trade_id)
                        action_str = str(f.get("action", "")).upper()
                        contract_side = str(f.get("side", "")).upper()
                        side = f"{action_str} {contract_side}".strip()
                        yes_side = str(f.get("side", "")).lower() == "yes"
                        qty = int(float(f.get("count_fp", "0")))
                        prc_raw = f.get("yes_price_dollars" if yes_side else "no_price_dollars", "0")
                        price = int(float(prc_raw) * 100) if prc_raw else 0
                        # Reverse-alias real Kalshi ticker → synthetic for the
                        # bot's view. Mutate f too so downstream code paths
                        # that read from the dict (CSV writer, etc.) see the
                        # synthetic name. Pass-through for non-aliased.
                        ticker = ticker_aliases.reverse(f.get("ticker", f.get("market_ticker", "UNKNOWN")))
                        f["ticker"] = ticker
                        f["market_ticker"] = ticker
                        ts = f.get("created_time", "UNKNOWN")
                        
                        # TODO: Remove this NHL console-silencing once the NHL algorithm gets wired in
                        if "NHL" not in ticker.upper():
                            log.info(f"\n======== RECENT PHYSICAL TRADE (REST) ========\n"
                                     f" Action : {side} {qty}x\n"
                                     f" Ticker : {ticker}\n"
                                     f" Price  : {price}c\n"
                                     f" Time   : {ts}\n"
                                     f"==============================================\n")
        except Exception as e:
            log.warning(f"TRADE LOGGER | Initial REST syphon bypass failed: {e}")

        # ── Live WebSocket Streaming ──
        while not self._shutdown.is_set():
            try:
                headers = self.client.auth.get_headers("GET", WS_PATH)
                session_start = time.monotonic()
                self._ws_ready = False  # Reset on each reconnect — fills before subscription confirm are replays
                log.info(f"TRADE LOGGER | ws_ready=False (connecting), seen_fills={len(self.seen_fills)} in dedup set")
                async with websockets.connect(WS_URL, additional_headers=headers) as ws:
                    sub_msg = {
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {"channels": ["fill"]}
                    }
                    await ws.send(json.dumps(sub_msg))
                    log.info("TRADE LOGGER | Authenticated Live WebSocket Protocol Engaged. Listening for fills seamlessly...")
                    # NB: the backoff is NOT reset here. Connecting proves
                    # nothing — the 26AUG27 failure mode is a socket that opens
                    # fine and then immediately loses its subscription. The
                    # delay is forgiven in the reconnect handler below, and only
                    # for a session that actually stayed up WS_STABLE_SESSION_S.

                    async for raw in ws:
                        if self._shutdown.is_set():
                            break

                        msg = json.loads(raw)
                        msg_type = msg.get("type")

                        if msg_type == "subscribed":
                            self._ws_ready = True
                            log.info(f"TRADE LOGGER | ws_ready=True — hedge buffer now ARMED. {msg}")
                        elif msg_type == "fill":
                            f = msg.get("msg", {})
                            
                            trade_id = f.get("trade_id", f.get("fill_id"))
                            if not trade_id or trade_id in self.seen_fills:
                                continue
                                
                            self.seen_fills.add(trade_id)
                            log.debug(f"TRADE LOGGER | Raw fill msg received: {f}")

                            # Subaccount gate: the fill WS delivers ALL subaccounts
                            # on one connection (see watch_fills_sub1.py). Only
                            # THIS process's subaccount (execution.TENNIS_SUBACCOUNT:
                            # tennis -> 0, QUOTER_SPORT=tt -> 1) may reach
                            # position_store / hedge accounting / trades.csv —
                            # fills on any other sub (manual, esports) must stay
                            # invisible to max_position and the position adjuster.
                            # Field is absent on primary fills, so missing → 0.
                            try:
                                _sub = int(f.get("subaccount") or f.get("subaccount_number") or 0)
                            except (TypeError, ValueError):
                                _sub = 0
                            from execution import TENNIS_SUBACCOUNT as _our_sub
                            if _sub != _our_sub:
                                log.info(f"TRADE LOGGER | SUBACCOUNT-{_sub} fill IGNORED (not ours, sub={_our_sub}): "
                                         f"{f.get('ticker', f.get('market_ticker', '?'))} "
                                         f"{f.get('action', '')} {f.get('side', '')} "
                                         f"x{f.get('count_fp', f.get('size', '?'))}")
                                continue

                            action_str = str(f.get("action", "")).upper()
                            contract_side = str(f.get("side", "")).upper()
                            # Accumulation-frame side (DIRECTION FIX 26AUG15);
                            # refined below when the parent order is resolved.
                            # Safe default = raw side (matches legacy watermark
                            # behavior for fills with no order_id).
                            accum_side = contract_side

                            # Kalshi's fill API can flip `side` for maker fills (shows the
                            # book side, not the order's true side). For all maker fills we
                            # look up the parent order to get the true side.
                            is_taker = bool(f.get("is_taker", True))
                            if not is_taker:
                                oid = f.get("order_id", "")
                                if oid:
                                    import execution
                                    # Registry-first true-side resolution
                                    # (FAST_CANCEL_SCOPE.md Change A): the
                                    # parent order is OURS, so its side lives
                                    # in execution's order registry — no REST
                                    # round-trip (~100-300ms saved on the
                                    # fill→cancel hot path). REST only on a
                                    # registry miss (orders placed before this
                                    # process started).
                                    _aq = execution.get_registered_order(oid)
                                    _reg_side = str(getattr(_aq, "kalshi_side", "") or "").upper() if _aq is not None else ""
                                    # DIRECTION FIX (26AUG15): the ACCUMULATION
                                    # side (internal frame — 'YES' = this fill
                                    # made us longer the ticker's team) drives
                                    # fast-cancel and the cooldown watermark.
                                    # It is DISTINCT from contract_side (order
                                    # frame, used for accounting/CSV/hedge):
                                    # an eaten offer is order-frame sell-YES
                                    # but accumulation-frame NO.
                                    accum_side = contract_side
                                    if _reg_side in ("YES", "NO") and action_str in ("BUY", "SELL"):
                                        # FRAME FIX (26AUG15, see FAST_CANCEL_SCOPE.md
                                        # DEFECT): the registry side is the INTERNAL
                                        # frame (offer-as-NO-bid); the fill's action is
                                        # ORDER-frame. Order-frame side follows the
                                        # identity sell-Y ≡ buy-opposite(Y): on a SELL
                                        # fill the order side is the OPPOSITE of the
                                        # internal side. Reproduces the REST parent-
                                        # order lookup byte-for-byte (verified live:
                                        # internal NO-bids fill as action=sell, order
                                        # side=yes). NEVER mix frames member-wise.
                                        if action_str == "SELL":
                                            contract_side = "NO" if _reg_side == "YES" else "YES"
                                        else:
                                            contract_side = _reg_side
                                        # registry side IS the internal frame
                                        accum_side = _reg_side
                                    else:
                                        log.info(f"TRADE LOGGER | [SIDE-LOOKUP MISS] {oid} not in registry — REST fallback")
                                        try:
                                            ord_resp = self.client._get(f"/trade-api/v2/portfolio/orders/{oid}")
                                            true_side = str(ord_resp.get("order", {}).get("side", "")).upper()
                                            if true_side in ("YES", "NO"):
                                                contract_side = true_side
                                                # derive internal frame from the
                                                # order frame via the same
                                                # identity, inverted
                                                if action_str == "SELL":
                                                    accum_side = "NO" if true_side == "YES" else "YES"
                                                else:
                                                    accum_side = true_side
                                        except Exception as ex:
                                            log.debug(f"TRADE LOGGER | Order lookup failed for {oid}: {ex}")
                                    # Fast cancel (FAST_CANCEL_SCOPE.md Change
                                    # B): dispatch the direction's P0 cancel
                                    # from the fill event — the quote-cycle
                                    # cooldown path stays as the idempotent
                                    # safety net. Silent no-op until the
                                    # manager registers context (post-restart)
                                    # or if fast_cancel_off.flag exists.
                                    try:
                                        _fc_tkr = ticker_aliases.reverse(
                                            f.get("ticker", f.get("market_ticker", "")))
                                        # ACCUMULATION side, not order side —
                                        # cancels the eaten ladder + siblings,
                                        # not the opposite book (26AUG15
                                        # direction fix, MIBRKRU evidence).
                                        execution.fast_cancel_from_fill(_fc_tkr, accum_side)
                                    except Exception as ex:
                                        log.debug(f"TRADE LOGGER | fast-cancel dispatch failed: {ex}")
                                    # Bump filled_count on the parent ActiveQuote so the next
                                    # amend computes new_count = filled + desired correctly.
                                    # Without this the residual after a partial fill sits at
                                    # the post-fill size until the residual itself fills.
                                    try:
                                        import execution
                                        fill_qty = int(float(f.get("count_fp", f.get("size", 0)) or 0))
                                        execution.bump_filled_count(oid, fill_qty)
                                    except Exception as ex:
                                        log.debug(f"TRADE LOGGER | filled_count bump failed for {oid}: {ex}")

                            side = f"{action_str} {contract_side}".strip() if action_str else contract_side

                            qty_raw = f.get("count_fp", f.get("size", "0"))
                            qty = int(float(qty_raw)) if qty_raw else 0

                            yes_side = (contract_side == "YES")

                            # Reverse-alias real Kalshi ticker → synthetic
                            # BEFORE any downstream consumer touches it. Critical:
                            # the is_series gate below (line ~190) checks for
                            # "GAME" in ticker — the real KXCS2-IEMCOL26-FAL
                            # would FAIL that check and silently skip hedge
                            # accounting. Mutate f too so the trades.csv writer
                            # (reads from f) stays consistent. Pass-through for
                            # any non-aliased ticker.
                            ticker = ticker_aliases.reverse(f.get("ticker", f.get("market_ticker", "UNKNOWN")))
                            f["ticker"] = ticker
                            f["market_ticker"] = ticker

                            # ── TENNIS-ONLY FILTER (2026-09-01) ──────────────
                            # The fills WS is ACCOUNT-wide: the esports system's
                            # fills arrive here too (observed 21:56Z — a
                            # KXLOLGAME maker fill got position-tracked,
                            # hedge-pushed, and written into tennis trades.csv).
                            # This process only trades shard-3 tennis series;
                            # everything else is another system's fill — log it
                            # at debug and skip ALL downstream accounting.
                            from execution import _is_tennis_ticker
                            if not _is_tennis_ticker(ticker):
                                log.debug(f"TRADE LOGGER | non-tennis fill ignored: {ticker}")
                                continue

                            # Immediately push execution vectors mathematically into the local position container securely tracking the fills!
                            import position_store
                            position_store.apply_fill(ticker, action_str, contract_side, qty)
                            
                            # Safely extract from both Kalshi V1/Legacy cent representations and modern V2 decimal string variables mapping identically into memory!
                            raw_yes = f.get("yes_price_dollars", f.get("yes_price"))
                            raw_no = f.get("no_price_dollars", f.get("no_price"))
                            pure_price = f.get("price_dollars", f.get("price"))
                            
                            price = 0
                            if raw_yes is not None:
                                p = float(raw_yes)
                                p = int(round(p * 100)) if p <= 1.0 else int(p)
                                price = p if yes_side else (100 - p)
                            elif raw_no is not None:
                                p = float(raw_no)
                                p = int(round(p * 100)) if p <= 1.0 else int(p)
                                price = (100 - p) if yes_side else p
                            elif pure_price is not None:
                                p = float(pure_price)
                                price = int(round(p * 100)) if p <= 1.0 else int(p)
                            else:
                                price = 0
                            
                            # Push maker fills to buffer for hedge bot.
                            # Push the YES-equivalent cost and the true side.
                            # YES cost = what it costs to be LONG this ticker's team.
                            #   If YES fill: yes_cost = yes_price
                            #   If NO fill: yes_cost = 100 - no_price (= yes_price)
                            # This way the hedger always knows the cost to go long.
                            is_taker = f.get("is_taker", True)
                            is_series = ("GAME" in ticker or "MATCH" in ticker)

                            # Log EVERY fill's hedge-eligibility decision
                            if is_series:
                                log.info(f"TRADE LOGGER HEDGE CHECK | {ticker} {contract_side} {qty}x @ {price}c | "
                                         f"is_taker={is_taker} ws_ready={self._ws_ready} trade_id={trade_id}")

                            if not is_taker and price > 0 and is_series and self._ws_ready:
                                if raw_yes is not None:
                                    p_yes = float(raw_yes)
                                    yes_cost = int(round(p_yes * 100)) if p_yes <= 1.0 else int(p_yes)
                                elif raw_no is not None:
                                    p_no = float(raw_no)
                                    no_cents = int(round(p_no * 100)) if p_no <= 1.0 else int(p_no)
                                    yes_cost = 100 - no_cents
                                else:
                                    yes_cost = price
                                position_store.push_maker_fill(ticker, qty, yes_cost, contract_side.lower(),
                                                               watermark_side=(accum_side.lower() if not is_taker else None))
                                log.info(f"TRADE LOGGER HEDGE PUSH | {ticker} → pushed {qty}x yes_cost={yes_cost}c side={contract_side.lower()} to hedge buffer")
                            elif is_series and not is_taker and not self._ws_ready:
                                log.warning(f"TRADE LOGGER HEDGE BLOCKED | {ticker} {qty}x — ws_ready=False (replay fill, not pushing to hedge buffer)")
                            elif is_series and is_taker:
                                log.debug(f"TRADE LOGGER HEDGE SKIP | {ticker} {qty}x — taker fill, no hedge needed")

                            # Shadow recency-flow tracker (TRADE_RETREAT_SCOPE.md) —
                            # instrumentation only. Records BOTH maker and taker live
                            # series fills (separate accumulators; the taker lever
                            # decides at query time). ws_ready gate keeps replay
                            # fills out of the recency window, same as hedge push.
                            if is_series and self._ws_ready and price > 0:
                                try:
                                    import retreat_shadow
                                    retreat_shadow.record_fill(ticker, action_str, contract_side, qty,
                                                               is_taker=bool(is_taker))
                                except Exception:
                                    log.exception("RETREAT SHADOW | record_fill failed")

                            ts_raw = f.get("created_time", f.get("ts", time.time()))
                            try:
                                ts_obj = datetime.fromtimestamp(int(ts_raw))
                                ts_str = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
                            except Exception:
                                ts_str = str(ts_raw)

                            raw_t = 0.0
                            adj_t = 0.0
                            m_edge = 0.0
                            
                            order_id = f.get("order_id")
                            client_order_id = f.get("client_order_id")
                            
                            t_metrics = None
                            import telemetry_store
                            if client_order_id:
                                t_metrics = telemetry_store.get_telemetry(client_order_id)
                                
                            # If client_order_id is missing/fails, retry fetching natively by order_id 
                            # to bypass the WebSocket intercepting Kalshi's fill millseconds before the POST payload completes!
                            retry_count = 0
                            while not t_metrics and retry_count < 4 and order_id:
                                t_metrics = telemetry_store.get_telemetry(order_id)
                                if t_metrics:
                                    break
                                await asyncio.sleep(0.35)
                                retry_count += 1
                                
                            if t_metrics:
                                raw_t = round(float(t_metrics.get("raw_theo", 0.0)), 2)
                                adj_t = round(float(t_metrics.get("adjusted_theo", 0.0)), 2)
                                
                                # Transform pure YES theos securely into NO theos structurally!
                                if not yes_side:
                                    raw_t = round(100.0 - raw_t, 2)
                                    adj_t = round(100.0 - adj_t, 2)
                                
                                # After structural transformation, mathematical edge flawlessly universalizes!
                                is_buy = ("BUY" in side.upper())
                                m_edge = (adj_t - price) if is_buy else (price - adj_t)
                                m_edge = round(m_edge, 2)
                            
                            # TODO: Remove this NHL console-silencing once the NHL algorithm gets wired in
                            if "NHL" not in ticker.upper():
                                arb_ctx = t_metrics.get("arb_context", "") if t_metrics else ""
                                ctx_line = f" Context: {arb_ctx}\n" if arb_ctx else ""
                                is_taker = f.get("is_taker", None)
                                fee_cost = f.get("fee_cost", "?")
                                source = "TAKER" if is_taker else "MAKER" if is_taker is False else "?"
                                log.info(f"\n======== PHYSICAL TRADE EXECUTED [{source}] ========\n"
                                         f" Action : {side} {qty}x\n"
                                         f" Ticker : {ticker}\n"
                                         f" Price  : {price}c\n"
                                         f" Edge   : {m_edge:>+5.1f}c  | (Raw: {raw_t}c -> Adj: {adj_t}c)\n"
                                         f" Source : {source} | Fee: ${fee_cost}\n"
                                         f" Time   : {ts_str}\n"
                                         f"{ctx_line}"
                                         f"{'=' * 48}\n")
                                         
                            if price == 0:
                                try:
                                    import json
                                    with open("ws_fill_dump.json", "a") as df:
                                        df.write(json.dumps(msg) + "\n")
                                except: pass
                                
                            f["raw_theo"] = raw_t
                            f["adjusted_theo"] = adj_t
                            f["monetary_edge"] = m_edge
                            f["cash_edge"] = round(m_edge * qty / 100.0, 4)
                            f["trigger_type"] = t_metrics.get("trigger_type", "") if t_metrics else ""
                            # PTA-only flag — never read by execution. Semantics:
                            # "raising our sizing config would have given us more fills here."
                            # TAKER: True on the FIRST fill where cumulative qty across all
                            #   WS events on this order_id crosses 0.8 * qsz, AND the
                            #   sweep was sizing-bound (`was_size_capped`). Subsequent
                            #   fills on the same order tag False (already crossed). A
                            #   1-lot fire that fills 1 lot is NOT full — sweep wasn't
                            #   sizing-bound (edge/book set the size, not our config).
                            # MAKER: True iff the entire quoted order filled in a single
                            #   Kalshi event (qty == qsz). Means we got everything we
                            #   offered at this level — raising L1/L2 vol might fit more.
                            # Multi-fill maker orders show False on partial rows; aggregate
                            # by order_id in PTA to recover the cumulative case (no
                            # cross-fill state for makers because they share order_id).
                            try:
                                qsz = int(t_metrics.get("quoted_size", 0)) if t_metrics else 0
                            except (TypeError, ValueError):
                                qsz = 0
                            if qsz > 0:
                                if is_taker:
                                    was_capped = bool(t_metrics.get("was_size_capped", False)) if t_metrics else False
                                    # Cross-book sweeps generate one order per
                                    # leg but the cap binds on the combined
                                    # sweep total. Aggregate fills by sweep_id
                                    # (not per-order) and compare to the full
                                    # sweep_total_vol so a 393-lot leg of a
                                    # 1626-lot sweep can't get tagged "full"
                                    # off its own 80% threshold when the
                                    # sweep itself wasn't sized to the cap.
                                    sweep_id = t_metrics.get("sweep_id", "") if t_metrics else ""
                                    sweep_total = int(t_metrics.get("sweep_total_vol", 0) or 0) if t_metrics else 0
                                    oid_key = sweep_id or order_id or trade_id
                                    threshold_size = sweep_total if sweep_total > 0 else qsz
                                    state = self._taker_cumulative.get(oid_key)
                                    if state is None:
                                        state = {"cum": 0, "qsz": threshold_size, "tagged": False}
                                        self._taker_cumulative[oid_key] = state
                                    state["cum"] += qty
                                    crosses_now = (not state["tagged"]) and state["cum"] >= 0.8 * state["qsz"]
                                    if was_capped and crosses_now:
                                        f["is_full_fill"] = True
                                        state["tagged"] = True
                                    else:
                                        f["is_full_fill"] = False
                                    # Keep dict bounded; reap stale orders periodically.
                                    if len(self._taker_cumulative) > 5000:
                                        old_keys = list(self._taker_cumulative.keys())[:1000]
                                        for k in old_keys:
                                            del self._taker_cumulative[k]
                                else:
                                    f["is_full_fill"] = bool(qty == qsz)
                            else:
                                f["is_full_fill"] = ""
                            # Overwrite `side` with the resolved true side so CSV is correct
                            # even for maker-sell fills (Kalshi's raw API returns the flipped side).
                            f["side"] = contract_side.lower()

                            f["computed_price_dollars"] = price / 100.0
                            self.recorded_trades.append(f)
                            
                        elif msg_type == "error":
                            # NOT recoverable in place: Kalshi has torn down the
                            # subscription but leaves the socket open, so the
                            # `async for` would block forever on a stream that
                            # will never deliver another fill. Raise into the
                            # reconnect handler (which resets _ws_ready first).
                            raise WSSubscriptionDead(str(msg))
                        else:
                            # Log heartbeat or unknown just so we see the stream alive!
                            log.debug(f"TRADE LOGGER | Unhandled WS Type {msg_type}: {msg}")
                            
            except WSSubscriptionDead as e:
                log.error(f"TRADE LOGGER | SUBSCRIPTION DEAD — Kalshi error frame, "
                          f"ws_ready set to False, reconnecting. Frame: {e}")
            except Exception as e:
                log.error(f"TRADE LOGGER | WS DISCONNECTED — ws_ready set to False, reconnecting. Error: {e}")

            # Forgive the backoff only for a session that actually held up. A
            # connect that dies immediately must keep escalating, or a repeating
            # error frame becomes a 1s reconnect hot loop against Kalshi.
            if time.monotonic() - session_start >= WS_STABLE_SESSION_S:
                reconnect_delay = 1.0
            if self._shutdown.is_set():
                break
            log.info(f"TRADE LOGGER | Reconnecting in {reconnect_delay:.1f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30.0)
            
    def shutdown_sync(self) -> None:
        """Write all recorded physical trades to CSV synchronously bypassing async death states."""
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

        if not self.recorded_trades:
            log.info("TRADE LOGGER | No trades recorded this active session limit. Exiting cleanly.")
            return
            
        file_exists = os.path.isfile(self.csv_path)
        
        try:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                # Write header if entirely fresh file
                if not file_exists:
                    writer.writerow(["trade_id", "created_ts", "ticker", "action", "count", "price", "order_id", "raw_theo", "adjusted_theo", "edge_cents", "cash_edge", "side", "is_taker", "fee_cost", "trigger_type", "is_full_fill", "settlement_pnl"])

                for trade in self.recorded_trades:
                    writer.writerow([
                        trade.get("trade_id", trade.get("fill_id", "")),
                        trade.get("created_time", trade.get("ts", "")),
                        trade.get("ticker", trade.get("market_ticker", "")),
                        trade.get("action", ""),
                        trade.get("count_fp", trade.get("size", "")),
                        trade.get("computed_price_dollars", trade.get("yes_price_dollars", trade.get("price", ""))),
                        trade.get("order_id", ""),
                        trade.get("raw_theo", ""),
                        trade.get("adjusted_theo", ""),
                        trade.get("monetary_edge", ""),
                        trade.get("cash_edge", ""),
                        trade.get("side", ""),
                        trade.get("is_taker", ""),
                        trade.get("fee_cost", ""),
                        trade.get("trigger_type", ""),
                        trade.get("is_full_fill", ""),
                        # settlement_pnl: empty at fill time (outcome not yet known).
                        # Backfilled later by `_backfill_settlement_pnl.py` once
                        # `_settlement_cache.json` covers this ticker.
                        "",
                    ])
                    
            log.info(f"TRADE LOGGER | Flush Success! Pushed {len(self.recorded_trades)} physical executions into '{self.csv_path}'.")
        except Exception as e:
            log.error(f"TRADE LOGGER | CRITICAL CSV FLUSH ERROR: {e}")
