import logging
import threading
from typing import Dict, Any, List

# Ticker alias layer — reverse-translate real Kalshi tickers on inbound
# REST boot-load so positions are keyed on the bot's synthetic name.
# Pass-through for any non-aliased ticker.
import ticker_aliases

log = logging.getLogger(__name__)

# Global thread-safe map: ticker -> int (net physical YES positional limit)
_POSITIONS: Dict[str, int] = {}
_lock = threading.Lock()
_initialized = False

def init_positions(client: Any, active_tickers: List[str]) -> None:
    """
    Safely boot-strap the memory container globally by invoking the Kalshi REST API exactly once.
    This safely prevents the algorithm from flying completely blind if it restarted midway into a game!
    """
    global _POSITIONS, _initialized
    with _lock:
        if _initialized:
            return
            
        try:
            path = "/trade-api/v2/portfolio/positions"
            log.info("POSITION STORE | Initiating critical boot-load physical REST synchronization globally...")
            # Natively traverse user REST gateway directly!
            # subaccount=0 is the documented API default, but pin it explicitly:
            # manual positions on subaccount 1+ must never enter this store
            # (max_position + position adjuster read from here).
            resp_data = client._get(path, params={"limit": 1000, "subaccount": 0})
            
            if resp_data and "market_positions" in resp_data:
                # Pick up any new alias entries from disk (alias module is
                # eager-loaded at import; this catches mid-session CSV edits).
                ticker_aliases.reload_if_changed()
                aliased_count = 0
                for p in resp_data["market_positions"]:
                    t_real = p.get("ticker", "")
                    # Reverse-alias real Kalshi ticker → bot's synthetic name
                    # so position_store reads/writes by the same key the bot
                    # uses internally. Pass-through for non-aliased.
                    t = ticker_aliases.reverse(t_real)
                    if t != t_real:
                        aliased_count += 1
                    # Initialize all natively found positions universally
                    pos = int(float(p.get("position_fp", 0)))
                    _POSITIONS[t] = pos
                _initialized = True
                log.info(f"POSITION STORE | Synchronous boot REST API sync completed "
                         f"successfully ({len(resp_data['market_positions'])} positions, "
                         f"{aliased_count} reverse-aliased). Decoupling sequence initialized!")
            else:
                log.warning("POSITION STORE | REST boot payload natively returned empty! Starting with blank global exposure array...")
        except Exception as e:
            log.error(f"POSITION STORE | FATAL boot REST payload execution failure: {e}")

def get_position(ticker: str) -> int:
    """Safely ping RAM to evaluate exposure cleanly without executing network traffic!"""
    with _lock:
        pos_primary = _POSITIONS.get(ticker, 0)
        
        # Aggressively net mathematically opposing sides (Team A vs Team B) universally
        if "LOL" in ticker or "CS2" in ticker or "VALORANT" in ticker or "DOTA2" in ticker or "COD" in ticker or "ATP" in ticker or "WTA" in ticker:
            base_group = "-".join(ticker.split("-")[:-1])
            for t, p in _POSITIONS.items():
                if t != ticker and t.startswith(base_group + "-"):
                    # We identically found the opponent. 
                    # Formula: Net(Team1) = Pos(Team1) - Pos(Team2)
                    return pos_primary - p
                    
        return pos_primary

def get_all_positions() -> Dict[str, int]:
    """Retrieve an instant copy of physical bounds structurally preventing mid-tick manipulation."""
    with _lock:
        return dict(_POSITIONS)

def apply_fill(ticker: str, action: str, side: str, qty: int) -> None:
    """
    Instantly adjust RAM memory the EXACT millisecond the physical WebSocket detects a fill payload!
    Mathematical alignment:
        BUY YES  -> +qty (Increases limit)
        SELL YES -> -qty (Decreases limit)
        BUY NO   -> -qty (Natively behaves identical to SELL YES)
        SELL NO  -> +qty (Natively behaves identical to BUY YES)
    """
    global _POSITIONS
    if not qty:
        return
        
    with _lock:
        current = _POSITIONS.get(ticker, 0)
        
        is_buy = ("BUY" in action.upper())
        is_yes = ("YES" in side.upper())
        
        delta = 0
        if is_buy and is_yes:
            delta = qty
        elif not is_buy and is_yes:
            delta = -qty
        elif is_buy and not is_yes:
            delta = -qty
        elif not is_buy and not is_yes:
            delta = qty
            
        _POSITIONS[ticker] = current + delta
        log.debug(f"POSITION STORE | Zero-latency algorithmic state natively updated! {ticker} shifted {delta:>+d} ➔ {_POSITIONS[ticker]} net aggregate lots.")


# ── Maker fill buffer for hedge bot ──
_MAKER_FILLS: Dict[str, list] = {}  # ticker -> [(qty, price_cents, side, timestamp)]
_maker_lock = threading.Lock()

# Non-destructive parallel to _MAKER_FILLS: tracks only the latest maker-fill
# timestamp per (ticker, side). Used by QuoterBot's refill-cooldown so it can
# detect "did a maker fill just happen" without consuming the queue that
# hedge_bot needs via pop_maker_fills.
_LAST_MAKER_FILL_TS: Dict[tuple, float] = {}  # (ticker, side_lower) -> ts

def push_maker_fill(ticker: str, qty: int, price_cents: int, side: str,
                    watermark_side: str = None) -> None:
    """Called by trade logger when a maker fill arrives.

    `side` ('yes'/'no') feeds the HEDGE queue and keeps the legacy ORDER-frame
    convention (hedge_bot netting semantics — unchanged 26AUG15).
    `watermark_side` (DIRECTION FIX 26AUG15): the ACCUMULATION-frame side for
    the refill-cooldown watermark — internal AQ frame, where 'yes' always
    means "this fill made us longer the ticker's team". The quoter's trigger
    maps watermark 'yes'→cool long-own-team, 'no'→cool long-opponent, so an
    eaten OFFER (order-frame sell-yes, accumulation-frame 'no') now cools the
    re-offering direction instead of the opposite side. Defaults to `side`
    for backward compat with any caller that predates the fix."""
    import time
    now = time.time()
    side_lower = side.lower()
    wm = (watermark_side or side).lower()
    with _maker_lock:
        if ticker not in _MAKER_FILLS:
            _MAKER_FILLS[ticker] = []
        _MAKER_FILLS[ticker].append((qty, price_cents, side_lower, now))
        _LAST_MAKER_FILL_TS[(ticker, wm)] = now

def pop_maker_fills(ticker: str) -> list:
    """Returns and clears all buffered maker fills for a ticker."""
    with _maker_lock:
        fills = _MAKER_FILLS.pop(ticker, [])
        return fills

def get_last_maker_fill_ts(ticker: str, side: str) -> float:
    """Latest maker-fill timestamp on (ticker, side), or 0.0 if none recorded.
    Non-destructive — does not affect pop_maker_fills."""
    with _maker_lock:
        return _LAST_MAKER_FILL_TS.get((ticker, side.lower()), 0.0)


# ── Direction-keyed refill cooldowns for QuoterBot ──
# Key is (event_base, long_team_suffix) — the team we'd grow long in if a
# fill in this direction occurred. Lets two QuoterBots on opposite tickers
# share one cooldown clock: BID YES on -FKS and BID NO on -RBN both target
# "long FKS" and share key (event_base, "FKS").
_DIRECTION_COOLDOWNS: Dict[tuple, float] = {}  # (event_base, long_team) -> until_ts
_cooldown_lock = threading.Lock()

def trigger_direction_cooldown(event_base: str, long_team: str, duration_sec: float) -> None:
    """Set a cooldown until now+duration_sec on (event_base, long_team).
    If a later cooldown is already in place, leaves it alone."""
    import time
    until_ts = time.time() + duration_sec
    with _cooldown_lock:
        existing = _DIRECTION_COOLDOWNS.get((event_base, long_team), 0.0)
        if until_ts > existing:
            _DIRECTION_COOLDOWNS[(event_base, long_team)] = until_ts

def is_direction_in_cooldown(event_base: str, long_team: str) -> bool:
    import time
    with _cooldown_lock:
        return _DIRECTION_COOLDOWNS.get((event_base, long_team), 0.0) > time.time()
