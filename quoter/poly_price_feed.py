"""
Polymarket CLOB Price Feed for Esports Map Prices

Replaces Kalshi orderbook polling for MAP/SET tickers with Polymarket CLOB
midpoint fetches. The arber only trades series (GAME) tickers — map prices
are purely inputs to the theo model. Poly has 1-2c spreads vs Kalshi's 10-15c.

Usage:
    feed = PolyMapPriceFeed()
    # In tick loop, after fetching Kalshi series orderbooks:
    feed.inject_all(active_events, top_level_bids, top_level_offers, dt_market_state)

The arber reads map prices from full_top_bids/full_top_offers dicts.
This module writes Poly-derived prices into those dicts so the arber
sees tighter, more accurate map prices without any code changes.
"""

import logging
import os
import time
import requests
import json
from typing import Optional

from populate_configs import (
    KALSHI_TO_POLY_SPORT, TEAM_ALIASES,
    _get_poly_winner_markets, _split_kalshi_teams,
    _load_poly_mappings, _fetch_poly_event_by_slug,
)

log = logging.getLogger(__name__)

# ---- WS book source (flag-gated, 26AUG04) --------------------------------
# When ws_feed.flag exists, best bid/ask come from the Poly CLOB WebSocket
# listener's snapshot file instead of REST /book. The listener
# (_poly_ws_listener.py) maintains delta-correct books and rewrites the
# snapshot atomically ~4x/s. Freshness is enforced per token AND for the
# whole file; any miss falls through to the unchanged REST path, and a dead
# snapshot while the flag is on warns loudly (silent WS->REST demotion is
# exactly the failure mode that hides feed bugs).
_WS_DIR = os.path.dirname(os.path.abspath(__file__))
WS_FEED_FLAG = os.path.join(_WS_DIR, "ws_feed.flag")
WS_SNAPSHOT = os.path.join(_WS_DIR, "_poly_ws_snapshot.json")
# Per-token bound. MUST exceed the listener's worst-case refresh interval,
# which is HEARTBEAT_S (5s) + the recv timeout granularity (3s) = 8.0s, because
# the heartbeat sweep only runs on message arrival or on that timeout. At the
# old value of 8.0 the margin was ZERO by construction: idle books grazed the
# threshold constantly, briefly dropping to REST and (once the STALE beacon
# landed 26AUG04) logging a false alarm roughly once a minute — which would
# have masked the real frozen-book case it exists to catch. 12s leaves 4s of
# slack and costs nothing: a genuinely frozen book ages to MINUTES, not 12s.
WS_FRESH_S = 12.0
WS_FILE_FRESH_S = 10.0    # whole-file bound (listener dead => all REST)
_ws_snap = {"mtime": 0.0, "data": {}, "flag_ts": 0.0, "flag": False,
            "dead_log_ts": 0.0}

# ---- token publication (26AUG04) ----------------------------------------
# The WS listener used to learn its token set from
# _venue_lead_capture_state.json — a RESEARCH script's state file. That put
# an instrumentation tool in the production critical path: if it died, the
# listener kept serving yesterday's tokens, tomorrow's games silently never
# entered the WS universe, and production dropped to REST for them with no
# warning (the snapshot stays "fresh" for the tokens it does have, so
# [WS-BOOK DEAD] never fires — a silent demotion).
#
# Inverted here: production, which has already resolved every token it
# trades, publishes them and the listener consumes. The listener therefore
# subscribes to exactly the set we price off — no more, no less.
WS_TOKENS_FILE = os.path.join(_WS_DIR, "_poly_ws_tokens.json")
_ws_tok_pub = {"last": 0.0, "sig": None}


def _publish_ws_tokens(label_map):
    """Atomically publish {token: leg_label} for the WS listener."""
    sig = hash(frozenset(label_map))
    now = time.time()
    if sig == _ws_tok_pub["sig"] and now - _ws_tok_pub["last"] < 60:
        return
    _ws_tok_pub["sig"] = sig
    _ws_tok_pub["last"] = now
    tmp = WS_TOKENS_FILE + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(label_map, fh)
        os.replace(tmp, WS_TOKENS_FILE)
    except OSError as e:
        log.debug(f"[WS TOKENS] publish failed (non-fatal): {e}")

def _note_ws_gap(c, now, kind, token_id):
    """Loudly surface PER-TOKEN WS gaps (throttled 60s).

    [WS-BOOK DEAD] is a WHOLE-FILE check and cannot catch this: the
    listener's snap_flush rewrites the snapshot every ~1s even on a
    silent connection (deliberate — file mtime is a process-liveness
    signal, added after the 26AUG04 04:39 silent death). So a listener
    that is alive and connected but whose BOOKS are frozen — exactly the
    26AUG03 23:18-00:27 wire-format change, where price_change moved to
    per-market `price_changes` batches and every book froze for 69
    minutes — keeps the file fresh while individual tokens go stale.
    Production then silently falls back to REST per token.

    Silent degradation is the failure mode we cannot afford overnight.
    """
    c[kind] = c.get(kind, 0) + 1
    c.setdefault("gap_tok", str(token_id)[:16])
    if now - c.get("gap_log_ts", 0) < 60:
        return
    miss, stale = c.get("gap_miss", 0), c.get("gap_stale", 0)
    if not (miss or stale):
        return
    c["gap_log_ts"] = now
    log.warning(
        f"[WS-BOOK STALE] per-token WS gaps in the last 60s: "
        f"{stale} stale (>{WS_FRESH_S}s old), {miss} missing from snapshot "
        f"— those tokens fell back to REST while the snapshot FILE looked "
        f"healthy (e.g. token {c.get('gap_tok')}…). Persistent = listener "
        f"alive but books frozen, or subscribed to a stale token set.")
    c["gap_miss"] = c["gap_stale"] = 0
    c.pop("gap_tok", None)


def _ws_book_lookup(token_id):
    """(bid_cents, ask_cents) from the WS snapshot, or None to use REST."""
    c = _ws_snap
    now = time.time()
    if now - c["flag_ts"] > 1.0:
        c["flag"] = os.path.exists(WS_FEED_FLAG)
        c["flag_ts"] = now
    if not c["flag"]:
        return None
    try:
        mt = os.path.getmtime(WS_SNAPSHOT)
    except OSError:
        mt = 0.0
    if now - mt > WS_FILE_FRESH_S:
        if now - c["dead_log_ts"] > 60:
            c["dead_log_ts"] = now
            log.warning(
                f"[WS-BOOK DEAD] ws_feed.flag is ON but the WS snapshot is "
                f"{now - mt:.0f}s old — ALL tokens falling back to REST "
                f"(is _poly_ws_listener.py running?)")
        return None
    if mt != c["mtime"]:
        try:
            with open(WS_SNAPSHOT) as fh:
                c["data"] = json.load(fh)
            c["mtime"] = mt
        except (OSError, ValueError):
            return None
    e = c["data"].get(str(token_id))
    if not e:
        _note_ws_gap(c, now, "gap_miss", token_id)
        return None
    bid_c, ask_c, ts_ms = e
    if now - ts_ms / 1000.0 > WS_FRESH_S:
        _note_ws_gap(c, now, "gap_stale", token_id)
        return None
    if bid_c is None or ask_c is None:
        return None
    # CROSSED-BOOK GUARD (26AUG04). A real CLOB cannot have bid > ask, but a
    # streamed book can: Poly sends a burst of price_change messages and we
    # briefly hold an intermediate state where the new ask has landed and the
    # stale bid levels have not yet been removed. Measured 26AUG04: 298 of
    # 430,209 states (0.069%), median cross 2c, max 22c, lifetime ~3ms,
    # concentrated on the busiest book (274/298 on HLET1).
    #
    # Such a tick can only hurt. Its mid sits OUTSIDE the true spread, so it
    # either dampens a genuine signal (07:38:54: crossed mid 73.0 vs true
    # 69.5 — would have made us less aggressive into a real HLE move) or
    # manufactures a fake one. It can never be what correctly triggers a
    # trade, because the true book is present milliseconds either side of it.
    #
    # Strict '>' only: a LOCKED book (bid == ask) has a well-defined mid and
    # occurred inside genuine moves today (70/70 at 07:38:54, 80/80 at
    # 07:35:17), so it is still served. Returning None falls through to the
    # unchanged REST path, which returns a valid book.
    if float(bid_c) > float(ask_c):
        if now - c.get("cross_log_ts", 0) > 30:
            c["cross_log_ts"] = now
            c["cross_n"] = c.get("cross_n", 0) + 1
            log.warning(
                f"[WS-BOOK CROSSED] token {str(token_id)[:16]}… bid={bid_c} "
                f"ask={ask_c} (cross {float(bid_c) - float(ask_c):.0f}c) — "
                f"invalid book, falling back to REST for this fetch "
                f"(episodes logged: {c['cross_n']})")
        return None
    if not c.get("served_logged"):
        c["served_logged"] = True
        log.info("[WS-BOOK LIVE] serving Poly books from the WebSocket "
                 "snapshot (ws_feed.flag on, snapshot fresh)")
    return (float(bid_c), float(ask_c))

DISCOVERY_INTERVAL_SECS = 120.0   # Re-discover Poly events every 2 min

# ─── Settled-series retirement (2026-07-27) ────────────────────────────────
# A decided series keeps its map legs in _event_cache forever, and Poly 404s
# /book for every resolved map — so we re-polled dead games on EVERY tick, for
# hours. Measured 2026-07-26: KXDOTA2GAME-26JUL261400STXJEN, a 14:00 game, was
# still being fetched at 23:02 at ~110 retries/min ("NO book for any map
# ['1','2']"), each retry a blocking REST call on the trading loop.
#
# Retirement signal is the KALSHI SERIES book (already in top_level_bids/offers,
# WS-fed and accurate — costs no extra fetch) rather than a Poly 404, because a
# 404 cannot distinguish "map resolved" from "map not created yet": game 3 of a
# BO3 404s exactly like a finished game 1, and retiring on that would make us
# LATE TO THE DECIDER. A series pinned at an extreme for a sustained window is
# a positive observation of resolution, not an inference from missing data.
#
# The 15-minute dwell is what makes this safe: a BO5 sitting 2-0 with map 3 live
# wobbles as map 3 plays, so it never accumulates 15 continuous minutes bonded.
SERIES_BOND_SECS = 900.0    # 15 min continuously bonded → series is over
SERIES_BOND_BID = 99        # bid >= 99  → this side has won
SERIES_BOND_ASK = 1         # ask <= 1   → this side has lost
# LOOSENED 2026-07-27 (per operator). The bid>=99 / ask<=1 test was too tight:
# real finished markets do not park at exactly 99/0. AVIVDS-VDS sat at bid=0 /
# ask=4 — the winning side at 96, not 99 — 5.5h after its game ended, and sailed
# straight through the bonded test. The robust signal is not a price level but
# the ABSENCE of any yes bid: nobody will buy that side at any price.
#   bid == 0            → no yes values at all
#   AND offer < 100     → but a real ask exists, so this is a live book with a
#                         dead side, NOT an empty/pregame/unquoted book (0/100),
#                         which must never be retired on this basis.
SERIES_NO_YES_BID = 0
SERIES_NO_YES_MAX_ASK = 100

# ─── Frozen-book detection (2026-07-27) ───────────────────────────────────
# The 2s cache TTL bounds CACHE age, not DATA age. A Poly market that is halted
# or dead on Polymarket's side keeps returning HTTP 200 with an unchanging book,
# so every fetch caches a *fresh timestamp* over a *stale price*. The TTL is
# satisfied, the leg injects normally, and the map theo freezes while the real
# Kalshi book walks away — phantom edge, and we trade into it. This is the
# opposite of a fetch failure, which returns None and safely suppresses.
#
# Precedent: the 2026-05-29 data_source=kalshi flip was forced by exactly this
# ("Poly freezes burned trades"), and the staleness guard it called for was
# never built. Same shape as 100TG2 (theo pinned at 46.4 for 4 min while the map
# collapsed 68→75→90; a taker fired 1,000 lots into a 9.4c phantom edge).
#
# DETECT-ONLY on purpose. An auto-halt would be a new, untested trading
# behaviour, and an illiquid academy-league book can legitimately sit unchanged
# for a while — a false halt costs real trading on a thin slate. So this logs
# loudly and changes nothing. Grep: "POLY FROZEN".
POLY_FREEZE_WARN_SECS = 90.0     # unchanged this long → warn
POLY_FREEZE_RELOG_SECS = 60.0    # then re-warn at most this often
# STALE_PRICE_SECS (was 10.0) removed 2026-07-16: the stale-cache fallback let a
# CLOB timeout re-inject a stale (often pregame) mid as a fresh mid±1c book →
# pregame-priced phantom edges → falling-knife blowups (>$100k cumulative). On a
# fetch timeout we now serve NOTHING beyond the 2s fresh cache, so the map ticker
# is left absent and the live-series active-map guard suppresses trading.

_POLY_PARSED_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "poly_parsed_markets.csv"
)


class PolyMapPriceFeed:
    """Fetches map prices from Polymarket CLOB and injects them into the
    arber's pricing dicts, replacing Kalshi map orderbook polling.

    Source of truth for Kalshi→Poly event mapping is poly_parsed_markets.csv
    (written by the populator with date-pinned slugs and verified team
    alignments). If the CSV does not pin a mapping for an event, Poly is
    disabled for it and the runner falls back to Kalshi map books — no
    fuzzy gamma-API guessing in the live path. Every missing/failed mapping
    emits a WARNING so silent demotion to Kalshi is visible in logs.
    """

    def __init__(self, kalshi_fetcher=None):
        # event_base -> {sport, poly_event_title, game1: {token_A, token_B, team_A, team_B}, game2: {...}}
        self._event_cache: dict[str, dict] = {}
        self._discovery_ts: dict[str, float] = {}  # event_base -> last discovery time

        # token_id -> (mid_cents, timestamp)
        self._price_cache: dict[str, tuple[float, float]] = {}

        # Pre-verified Kalshi→Poly mappings (source of truth).
        # event_base -> {poly_slug, poly_url, poly_title}
        self._csv_mappings: dict[str, dict] = {}
        self._csv_mtime: float = 0.0
        self._reload_csv_if_changed()

        # Optional Kalshi fetcher: callable taking event_base, returning a list
        # of market dicts (each with at least "ticker" and "yes_sub_title").
        # When set, _build_entry uses it as the authoritative source of Kalshi
        # team suffixes — bypassing the antiquated _split_kalshi_teams string
        # heuristic that fails on team-suffix concatenations missing from
        # TEAM_ALIASES (e.g., WOLFPX = WOL + FPX, neither aliased).
        self._kalshi_fetcher = kalshi_fetcher

        # Dedupe alignment-failure warnings — one log per (event_base, leg)
        # per process lifetime instead of one per discovery tick.
        self._warned_alignment: set[tuple[str, str]] = set()

        # Throttle the "no book for map(s)" logs. A resolved map keeps 404ing
        # /book for the rest of the series, so the unthrottled line fired every
        # inject cycle — several times a second per event, and immediately on
        # restart for any event past map 1. Keyed by event_base; re-logs when
        # the stale-leg SET changes or the interval elapses, and reports how
        # many repeats it swallowed so the condition is never silently hidden.
        # event_base -> (stale_legs, last_logged_ts, suppressed_count)
        self._stale_leg_log: dict[str, tuple[frozenset, float, int]] = {}

        # ─── Settled-series retirement (see SERIES_BOND_SECS) ───
        # event_base -> ts when the series book was FIRST seen bonded (reset the
        # moment it un-bonds, so only a continuous window counts).
        self._series_bonded_since: dict[str, float] = {}
        # Retired events. Also gates _discover_event, otherwise the next tick
        # would re-discover straight from the CSV and undo the eviction.
        self._settled_events: set[str] = set()

        # Frozen-book detection (see POLY_FREEZE_WARN_SECS).
        # token_id -> (book, ts_first_seen_at_this_value, ts_last_warned)
        self._book_seen: dict[str, tuple[tuple, float, float]] = {}
        # token_id -> event/leg label, for readable freeze warnings
        self._token_label: dict[str, str] = {}

        # Tokens batch-fetched this tick (Phase 3 of inject_all). Lets the
        # cache_only path tell "the batch tried and failed" (expected, quiet)
        # from "this token was never batched" (a bug — see _fetch_book).
        self._batched_this_tick: set[str] = set()

        # Pooled session: the old code called module-level requests.get, paying
        # a fresh TLS handshake per book fetch. Measured 2026-07-27 against
        # clob.polymarket.com: 130ms/call cold vs 105ms/call keep-alive.
        # pool_maxsize covers the _batch_fetch_books thread pool.
        self._session = requests.Session()
        self._session.mount(
            "https://",
            requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20),
        )

    # Re-log intervals for stale map legs. The all-legs-missing case is a real
    # outage (theos suppressed for the whole event) so it repeats often; the
    # partial case is almost always a resolved map and repeats rarely.
    _STALE_LOG_INTERVAL_ERROR = 60.0
    _STALE_LOG_INTERVAL_WARN = 600.0

    def _log_stale_legs(self, event_base: str, stale_legs: list,
                        injected_any: bool) -> None:
        """Emit the stale-map-leg log, throttled per event.

        Always logs when the set of stale legs CHANGES, or when severity
        changes (a partial outage escalating to "no book for ANY map" is news
        and must never sit behind the throttle), otherwise at most once per
        interval, carrying the count of suppressed repeats so a persistent
        condition stays visible.
        """
        # Severity is part of the key: WARN->ERROR on the same leg set is an
        # escalation, not a repeat.
        legs = (frozenset(stale_legs), bool(injected_any))
        now = time.time()
        interval = (self._STALE_LOG_INTERVAL_WARN if injected_any
                    else self._STALE_LOG_INTERVAL_ERROR)

        repeat = ""
        prev = self._stale_leg_log.get(event_base)
        if prev is not None:
            prev_legs, last_ts, suppressed = prev
            if prev_legs == legs:
                if (now - last_ts) < interval:
                    self._stale_leg_log[event_base] = (legs, last_ts, suppressed + 1)
                    return
                if suppressed:
                    repeat = (f" [{suppressed} identical repeats suppressed in "
                              f"the last {now - last_ts:.0f}s]")

        self._stale_leg_log[event_base] = (legs, now, 0)
        if not injected_any:
            log.error(f"[POLY CLOB] {event_base}: NO book for any map "
                      f"{stale_legs} — no prices injected, theos will be "
                      f"suppressed (no stale-cache fallback){repeat}")
        else:
            log.warning(f"[POLY CLOB] {event_base}: no book for map(s) "
                        f"{stale_legs} (resolved leg or fetch failure — "
                        f"indistinguishable here); other legs injected OK. "
                        f"Theos suppressed only if this is the active map."
                        f"{repeat}")

    def _note_books(self, results: dict) -> None:
        """Flag Poly books that have not moved (see POLY_FREEZE_WARN_SECS).

        DETECT ONLY — never suppresses. Called single-threaded after the batch
        pool joins, so no locking is needed despite the fetches being parallel.
        A None result is a fetch FAILURE, which is the safe path (absent leg →
        suppressed) and is deliberately not treated as a freeze: it clears the
        tracker so a recovered book starts a fresh window.
        """
        now = time.time()
        for token_id, book in results.items():
            if book is None:
                self._book_seen.pop(token_id, None)
                continue
            prev = self._book_seen.get(token_id)
            if prev is None or prev[0] != book:
                if prev is not None and (now - prev[1]) >= POLY_FREEZE_WARN_SECS:
                    log.warning(
                        f"[POLY FROZEN] RECOVERED {self._token_label.get(token_id, token_id[:12])}: "
                        f"book moved {prev[0]} → {book} after "
                        f"{now - prev[1]:.0f}s unchanged"
                    )
                self._book_seen[token_id] = (book, now, 0.0)
                continue

            book_val, first_seen, last_warn = prev
            held = now - first_seen
            if held < POLY_FREEZE_WARN_SECS:
                continue
            if last_warn and (now - last_warn) < POLY_FREEZE_RELOG_SECS:
                continue
            self._book_seen[token_id] = (book_val, first_seen, now)
            log.warning(
                f"[POLY FROZEN] {self._token_label.get(token_id, token_id[:12])}: "
                f"book {book_val} UNCHANGED for {held:.0f}s while still returning "
                f"HTTP 200 — the 2s cache TTL cannot catch this (fresh timestamp, "
                f"stale data). If the Kalshi book is moving, the map theo is pinned "
                f"and any edge against it is phantom. Detection only; nothing "
                f"suppressed."
            )

    def _is_series_settled(self, event_base: str, top_level_bids: dict,
                           top_level_offers: dict) -> bool:
        """True if this event should be retired from Poly polling.

        Retires an event once its Kalshi SERIES book has sat bonded at an
        extreme (bid >= 99, i.e. this side won, or ask <= 1, i.e. it lost) for
        SERIES_BOND_SECS continuously. See the SERIES_BOND_SECS comment for why
        the signal is the Kalshi series book and not a Poly 404.

        Deliberately conservative — an event is retired only on a POSITIVE
        observation of a bonded book:

          * A series ticker absent from the dicts does NOT count. Callers
            default missing tickers to 0/100, and treating "no data" as
            "decided" would retire live events during a book outage. Absence
            resets the dwell timer, exactly like an un-bonded book.
          * The dwell must be CONTINUOUS. Any tick where the book is not bonded
            clears the timer and the 15 minutes restart.
        """
        if event_base in self._settled_events:
            return True

        # Series tickers are "{event_base}-{TEAM}". Map tickers cannot collide:
        # their prefix has GAME→MAP (or MATCH→SETWINNER) substituted, so they
        # do not start with event_base.
        prefix = event_base + "-"
        bonded = False
        seen_any = False
        for tkr, bid in top_level_bids.items():
            if not tkr.startswith(prefix) or tkr not in top_level_offers:
                continue
            seen_any = True
            ask = top_level_offers[tkr]
            # Either the classic bonded extreme, OR (loosened 2026-07-27) a side
            # with no yes bids at all while still quoting a real ask.
            if (bid >= SERIES_BOND_BID or ask <= SERIES_BOND_ASK
                    or (bid == SERIES_NO_YES_BID and ask < SERIES_NO_YES_MAX_ASK)):
                bonded = True
                break

        if not (seen_any and bonded):
            self._series_bonded_since.pop(event_base, None)
            return False

        now = time.time()
        first = self._series_bonded_since.setdefault(event_base, now)
        held = now - first
        if held < SERIES_BOND_SECS:
            return False

        self._settled_events.add(event_base)
        self._series_bonded_since.pop(event_base, None)

        # Drop cached books so a retired event stops consuming memory, and so a
        # later re-discovery (should one ever happen) cannot read a stale price.
        entry = self._event_cache.pop(event_base, {}) or {}
        for leg in ["game1", "game2", "game3", "game4", "game5"]:
            leg_data = entry.get(leg)
            if leg_data and leg_data.get("token_a"):
                self._price_cache.pop(leg_data["token_a"], None)
        self._discovery_ts.pop(event_base, None)
        self._stale_leg_log.pop(event_base, None)

        log.info(
            f"[POLY FEED] {event_base}: series book bonded for {held / 60:.0f} min "
            f"— series decided, retiring event from Poly polling (was re-fetching "
            f"its resolved map legs every tick)."
        )
        return True

    def _reload_csv_if_changed(self) -> None:
        """Reload poly_parsed_markets.csv when its mtime changes. Lets the
        populator's writes take effect in a live-running process without
        restarts. On reload, also clears the event-discovery cache so stale
        slugs from the previous CSV don't linger."""
        try:
            mtime = os.path.getmtime(_POLY_PARSED_CSV)
        except OSError:
            mtime = 0.0
        if mtime == self._csv_mtime:
            return
        self._csv_mappings = _load_poly_mappings(_POLY_PARSED_CSV)
        self._csv_mtime = mtime
        if self._event_cache:
            log.info(f"[POLY FEED] poly_parsed_markets.csv changed (mtime={mtime}); "
                     f"clearing discovery cache ({len(self._event_cache)} entries)")
            self._event_cache.clear()
            self._discovery_ts.clear()

    # ──────────────────────────────────────────────
    # Event discovery (CSV-driven, deterministic)
    # ──────────────────────────────────────────────

    def _discover_event(self, event_base: str, team_suffix: str, kalshi_markets: list = None) -> bool:
        """Resolve a Kalshi event to its Polymarket counterpart via the
        pre-verified poly_parsed_markets.csv mapping.

        There is NO fuzzy gamma-API fallback. If the CSV does not pin a
        slug for this event, Poly is disabled for it and the caller falls
        back to Kalshi map books (existing behavior in run.py). This makes
        the mapping deterministic and surfaces missing populator runs as
        loud WARNINGs instead of silently mispricing.

        event_base: e.g., "KXDOTA2GAME-26APR200300GLTS"
        team_suffix: kept for signature compatibility (no longer used here).
        kalshi_markets: market dicts with yes_sub_title for suffix→full-name
            alignment. The runner passes these via full_market_state.
        Returns True if discovery succeeded.
        """
        # Retired by _is_series_settled. Must be checked here too: eviction only
        # clears _event_cache, and discovery re-populates it straight from the
        # CSV, so without this gate the next tick would silently un-retire the
        # event and resume polling its dead legs.
        if event_base in self._settled_events:
            return False

        now = time.time()
        if event_base in self._event_cache:
            if now - self._discovery_ts.get(event_base, 0) < DISCOVERY_INTERVAL_SECS:
                return True
        self._discovery_ts[event_base] = now

        sport = None
        for prefix, s in KALSHI_TO_POLY_SPORT.items():
            if prefix in event_base:
                sport = s
                break
        if not sport:
            return False

        csv_entry = self._csv_mappings.get(event_base)
        if csv_entry is None:
            log.warning(f"[POLY FEED] {event_base}: no entry in poly_parsed_markets.csv "
                        f"— Poly disabled, caller will use Kalshi map books. "
                        f"Run the populator to pin this mapping.")
            return False

        slug = csv_entry.get("poly_slug", "")
        if not slug:
            # CSV explicitly records "no Poly listing" (empty poly_url) —
            # Kalshi-only event by design. Quiet skip.
            log.debug(f"[POLY FEED] {event_base} flagged Kalshi-only in CSV — skipping Poly")
            return False

        poly_event = _fetch_poly_event_by_slug(slug)
        if not poly_event:
            if event_base not in self._warned_alignment:
                self._warned_alignment.add(event_base)
                log.warning(f"[POLY FEED] {event_base}: CSV slug {slug!r} did not resolve "
                            f"to a Poly event — Poly disabled, caller falls back to Kalshi "
                            f"(further failures for this event silenced)")
            return False

        if not self._build_entry(event_base, sport, poly_event, kalshi_markets, source="csv"):
            if event_base not in self._warned_alignment:
                self._warned_alignment.add(event_base)
                log.warning(f"[POLY FEED] {event_base}: CSV slug {slug!r} resolved but "
                            f"team alignment failed — Poly disabled, caller falls back to Kalshi "
                            f"(further failures for this event silenced)")
            return False
        return True

    def _build_entry(self, event_base: str, sport: str, poly_event: dict,
                     kalshi_markets: list, source: str) -> bool:
        """Build and cache an event entry from a resolved Polymarket event.

        Kalshi team suffixes are derived from `kalshi_markets` ticker names
        (deterministic) — NOT from _split_kalshi_teams, which silently drops
        unknown abbreviations like 'EXH'. _split_kalshi_teams is only used as
        a last-ditch fallback when kalshi_markets is unavailable.
        """
        winners = _get_poly_winner_markets(poly_event)
        if not winners:
            log.warning(f"[POLY FEED] {event_base}: no winner markets in Poly event "
                        f"(source={source})")
            return False

        # suffix → full team name from Kalshi yes_sub_title (preferred).
        team_full_names: dict[str, str] = {}
        suffixes: list[str] = []
        if kalshi_markets:
            for m in kalshi_markets:
                t = m.get("ticker", "")
                if not t:
                    continue
                suffix = t.split("-")[-1]
                sub = (m.get("yes_sub_title") or "").strip()
                if suffix and sub and suffix not in team_full_names:
                    team_full_names[suffix] = sub
                    suffixes.append(suffix)

        # If we couldn't derive suffixes from kalshi_markets, try the on-demand
        # Kalshi fetcher (deterministic; replaces _split_kalshi_teams heuristic).
        # The fetcher returns the same shape as kalshi_markets and we re-run the
        # same extraction loop above. Only when the fetcher is unavailable OR
        # fails do we fall back to the string-splitting heuristic.
        if len(suffixes) < 2 and self._kalshi_fetcher is not None:
            try:
                fetched = self._kalshi_fetcher(event_base) or []
            except Exception as e:
                log.debug(f"[POLY FEED] {event_base}: kalshi_fetcher raised: {e}")
                fetched = []
            for m in fetched:
                t = m.get("ticker", "")
                if not t: continue
                suffix = t.split("-")[-1]
                sub = (m.get("yes_sub_title") or "").strip()
                if suffix and sub and suffix not in team_full_names:
                    team_full_names[suffix] = sub
                    suffixes.append(suffix)
            if len(suffixes) >= 2:
                log.debug(f"[POLY FEED] {event_base}: derived suffixes via kalshi_fetcher → {suffixes[:2]}")

        if len(suffixes) < 2:
            tail = event_base.split("-", 1)[1] if "-" in event_base else ""
            teams_str = tail[11:] if len(tail) > 11 else ""
            split = _split_kalshi_teams(teams_str)
            if len(split) >= 2:
                suffixes = split[:2]
                log.debug(f"[POLY FEED] {event_base}: kalshi_markets lacked "
                          f"yes_sub_title; falling back to _split_kalshi_teams → {suffixes}")
            else:
                log.warning(f"[POLY FEED] {event_base}: cannot derive Kalshi team "
                            f"suffixes (kalshi_markets={'provided' if kalshi_markets else 'missing'}, "
                            f"_split_kalshi_teams={split}) — Poly disabled")
                return False

        team_a, team_b = suffixes[0], suffixes[1]
        full_a = team_full_names.get(team_a, "")

        entry = {
            "sport": sport,
            "title": poly_event.get("title", "")[:60],
            "teams": [team_a, team_b],
            "source": source,
        }

        # game1, game2: always for BO3; game3, game4, game5: only if Poly
        # exposes them (BO5 events). BO3 events naturally skip g3-g5 because
        # winners.get("game3") returns None.
        for leg in ["game1", "game2", "game3", "game4", "game5"]:
            pw = winners.get(leg)
            if not pw:
                continue
            outcomes = pw.get("outcomes", [])
            tokens = pw.get("tokens", [])
            if not outcomes or not tokens or len(outcomes) < 2 or len(tokens) < 2:
                continue
            idx_a = self._find_aligned_idx(team_a, outcomes, full_a, event_base=event_base)
            if idx_a < 0:
                key = (event_base, leg)
                if key not in self._warned_alignment:
                    self._warned_alignment.add(key)
                    log.warning(f"[POLY FEED] {event_base} {leg}: alignment failed for "
                                f"{team_a} ('{full_a}') vs {outcomes} — skipping leg "
                                f"(further failures for this leg silenced)")
                continue
            idx_b = 1 - idx_a
            entry[leg] = {
                "token_a": tokens[idx_a], "token_b": tokens[idx_b],
                "team_a": team_a, "team_b": team_b,
                "outcome_a": outcomes[idx_a], "outcome_b": outcomes[idx_b],
            }

        pw_series = winners.get("series")
        if pw_series:
            outcomes = pw_series.get("outcomes", [])
            tokens = pw_series.get("tokens", [])
            if outcomes and tokens and len(outcomes) >= 2:
                idx_a = self._find_aligned_idx(team_a, outcomes, full_a, event_base=event_base)
                if idx_a >= 0:
                    entry["series"] = {
                        "token_a": tokens[idx_a], "token_b": tokens[1 - idx_a],
                        "team_a": team_a, "team_b": team_b,
                        "outcome_a": outcomes[idx_a], "outcome_b": outcomes[1 - idx_a],
                    }

        # Require at least one map leg to consider discovery successful.
        if not any(k in entry for k in ("game1", "game2", "game3", "game4", "game5")):
            log.warning(f"[POLY FEED] {event_base}: no aligned map legs (source={source})")
            return False

        self._event_cache[event_base] = entry
        legs = [l for l in ["series", "game1", "game2"] if l in entry]
        log.info(f"[POLY FEED] Discovered ({source}): {event_base} → "
                 f"{entry['title']} | teams={[team_a, team_b]} legs={legs}")
        return True

    def _find_aligned_idx(self, team_abbr: str, outcomes: list[str],
                          kalshi_full_name: str = "", event_base: str = "") -> int:
        """Find the Poly outcome index matching a Kalshi team.

        Deterministic-only path via poly_parsed_markets.csv. First tries the
        Kalshi yes_sub_title (full name); if that's missing or doesn't resolve,
        falls back to the ticker suffix (team_abbr) — both go through the same
        unique-substring _align_team_to_poly_outcome, so an ambiguous
        abbreviation still returns -1 rather than guessing.

        Rationale: yes_sub_title can be unavailable (kalshi_markets not yet
        loaded, or empty in the API response). In that case a literal ticker
        suffix like "LGD" should still align to a Poly outcome like
        "LGD Gaming" — it's the same name written shorter. Refusing to align
        when only the suffix is known produced silent "kalshi-only" fallbacks
        that broke Poly mid-pricing for LGDTY 2026-06-04.
        """
        if not event_base:
            return -1
        from populate_configs import _align_team_to_poly_outcome
        if kalshi_full_name:
            idx = _align_team_to_poly_outcome(kalshi_full_name, outcomes,
                                              event_base=event_base)
            if idx >= 0:
                return idx
        if team_abbr:
            idx = _align_team_to_poly_outcome(team_abbr, outcomes,
                                              event_base=event_base)
            if idx >= 0:
                log.warning(f"[POLY FEED] {event_base}: aligned {team_abbr!r} "
                            f"via ticker suffix (full_name={kalshi_full_name!r}) "
                            f"— if this is wrong, pin in kalshi_poly_team_map.csv")
                return idx
        return -1

    # ──────────────────────────────────────────────
    # CLOB price fetching (every tick)
    # ──────────────────────────────────────────────

    def _fetch_book(self, token_id: str,
                    cache_only: bool = False) -> Optional[tuple[float, float]]:
        """Fetch best bid/ask from CLOB book for a token. Returns (bid_cents, ask_cents) or None.
        Uses fresh cache if available (< 2s old, e.g. from batch fetch).
        Otherwise fetches from CLOB. On timeout/failure returns None (NO stale
        fallback) so the caller leaves the map ticker absent and trading halts.

        cache_only=True: serve from cache or return None, never hit the network.
        Used by inject_all's Phase 4, where Phase 3 has ALREADY batch-fetched
        every token this tick. Before this flag, a token the batch failed on was
        re-fetched here SEQUENTIALLY and BLOCKING, ~130ms each, on the trading
        loop — and since Poly 404s /book forever once a map resolves, dead legs
        were retried every tick indefinitely. Measured 2026-07-27: the tick
        period was linear in active Poly events (1→257ms, 2→800ms, 3→1324ms),
        ~530ms per event, essentially all of it these serial retries.

        The retry it removes was near-worthless anyway: it fires milliseconds
        after the batch failed on the same token under identical network
        conditions. Note this does NOT weaken the no-stale-price invariant — it
        returns None *sooner*, never a stale value.
        """
        ws = _ws_book_lookup(token_id)
        if ws is not None:
            self._price_cache[token_id] = (ws, time.time())
            return ws

        cached = self._price_cache.get(token_id)
        if cached and time.time() - cached[1] < 2.0:
            return cached[0]

        if cache_only:
            # A miss for a token Phase 3 batched = that fetch failed. Expected
            # (resolved leg 404s, transient timeout) — the caller leaves the map
            # absent and the active-map guard suppresses. Quiet.
            #
            # A miss for a token Phase 3 NEVER batched is a BUG: cache_only is
            # only sound while Phase 2's token set covers everything Phase 4
            # reads. If someone adds a fetch to inject_map_prices (a token_b, a
            # 'series' leg, a 6th map) without adding it to Phase 2, this would
            # silently return None forever → map absent → that event quietly
            # stops trading. Silent demotion is exactly the failure mode that
            # hid the Poly map-pricing bug for a month, so make it loud.
            if token_id not in self._batched_this_tick:
                log.warning(
                    f"[POLY FEED] cache_only miss for token {str(token_id)[:16]}… "
                    f"that was NOT batch-fetched this tick — inject_map_prices is "
                    f"reading a token inject_all's Phase 2 does not collect. This "
                    f"leg will be permanently absent (silent no-trade) until the "
                    f"two token sets are reconciled."
                )
            return None

        try:
            r = self._session.get(
                f"https://clob.polymarket.com/book?token_id={token_id}",
                timeout=2,
            )
            if r.status_code == 200:
                book = r.json()
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                best_bid = max((float(b["price"]) for b in bids), default=0) * 100
                best_ask = min((float(a["price"]) for a in asks), default=1.0) * 100
                result = (best_bid, best_ask)
                self._price_cache[token_id] = (result, time.time())
                return result
        except Exception:
            pass

        # Poly CLOB fetch failed / timed out. Do NOT serve a stale cached price:
        # return None so the caller leaves this map ticker ABSENT and the
        # live-series active-map guard suppresses theos for the cycle (we don't
        # trade). Removed the prior <10s stale-cache fallback 2026-07-16 —
        # re-injecting a stale (often pregame) mid as a fresh mid±1c book
        # manufactured phantom edges and repeated falling-knife blowups
        # (GMPCFIC 2026-07-16; >$100k cumulative).
        return None

    def _check_poly_depth(self, token_id: str, mid_cents: float,
                          min_lots: float = 50.0, band_cents: float = 5.0) -> bool:
        """Check if a Polymarket token has real liquidity near the midpoint.
        Returns True if there are at least min_lots within band_cents of mid.
        Prevents trusting phantom prices from illiquid books."""
        try:
            r = self._session.get(
                f"https://clob.polymarket.com/book?token_id={token_id}",
                timeout=2,
            )
            if r.status_code != 200:
                return False
            book = r.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            mid = mid_cents / 100.0
            depth = 0.0
            for level in bids + asks:
                price = float(level.get("price", 0))
                if abs(price - mid) <= band_cents / 100.0:
                    depth += float(level.get("size", 0))
            if depth < min_lots:
                log.warning(f"[POLY DEPTH] Illiquid: {depth:.0f} lots near mid={mid_cents:.0f}c "
                            f"(need {min_lots:.0f}) — rejecting price")
                return False
            return True
        except Exception:
            return False

    def _batch_fetch_books(self, token_ids: list[str]) -> dict[str, Optional[tuple[float, float]]]:
        """Fetch best bid/ask from CLOB book for multiple tokens in parallel.
        Returns {token_id: (bid_cents, ask_cents)}."""
        import concurrent.futures

        results = {}
        if not token_ids:
            return results

        def _fetch_one(token_id):
            ws = _ws_book_lookup(token_id)
            if ws is not None:
                self._price_cache[token_id] = (ws, time.time())
                return token_id, ws
            try:
                r = self._session.get(
                    f"https://clob.polymarket.com/book?token_id={token_id}",
                    timeout=2,
                )
                if r.status_code == 200:
                    book = r.json()
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    best_bid = max((float(b["price"]) for b in bids), default=0) * 100
                    best_ask = min((float(a["price"]) for a in asks), default=1.0) * 100
                    result = (best_bid, best_ask)
                    self._price_cache[token_id] = (result, time.time())
                    return token_id, result
            except Exception:
                pass
            # Poly CLOB fetch failed / timed out — return None (NO stale-cache
            # fallback; removed 2026-07-16). Caller leaves the map ticker absent
            # → live-series active-map guard suppresses trading for the cycle.
            return token_id, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_fetch_one, tid) for tid in token_ids]
            for f in concurrent.futures.as_completed(futures):
                tid, book_data = f.result()
                results[tid] = book_data

        # Frozen-book detection. Additive and exception-safe: it must never be
        # able to break a fetch path that is working.
        try:
            self._note_books(results)
        except Exception as _e:
            log.debug(f"[POLY FROZEN] detector error (ignored): {_e}")

        return results

    def fetch_map_prices(self, event_base: str) -> dict[str, Optional[tuple[float, float]]]:
        """Fetch CLOB best bid/ask for game1..game5 of an event.

        Returns {gameN_book: (bid, ask)|None} in cents, from team A's perspective
        (the alphabetically-first team). game3-5 are populated for BO5 events
        that have Poly winner markets exposed.
        """
        entry = self._event_cache.get(event_base)
        if not entry:
            return {f"{leg}_book": None for leg in ["game1", "game2", "game3", "game4", "game5"]}

        result = {}
        for leg in ["game1", "game2", "game3", "game4", "game5"]:
            leg_data = entry.get(leg)
            if not leg_data:
                result[f"{leg}_book"] = None
                continue

            book = self._fetch_book(leg_data["token_a"])
            result[f"{leg}_book"] = book

        return result

    # ──────────────────────────────────────────────
    # Injection into arber pricing dicts
    # ──────────────────────────────────────────────

    def inject_map_prices(
        self,
        event_base: str,
        team_suffix: str,
        top_level_bids: dict,
        top_level_offers: dict,
        dt_market_state: dict,
        injected_out: set = None,
        cache_only: bool = False,
    ) -> bool:
        """Inject Poly-derived map prices into the arber's pricing dicts.

        cache_only is forwarded to _fetch_book. inject_all passes True because
        its Phase 3 has already batch-fetched every token this method reads;
        it defaults to False so direct callers keep the fetching behaviour.

        Writes the REAL Poly CLOB top-of-book (best_bid/best_ask, fetched
        <2s fresh — see _fetch_book) for both teams' map tickers. This is NOT
        a synthetic "mid ± 1c" fabrication: bid_a/offer_a come straight from
        the live book (these Poly map books just happen to be ~1c wide). There
        is deliberately NO fallback that manufactures a book from a bare mid or
        a pregame probability — on a CLOB timeout the leg is left ABSENT and
        trading halts (removed the stale-cache fallback 2026-07-16, GMPCFIC).
        Do NOT reintroduce any mid±1 synthesis here.
        Returns True if at least one map price was injected.

        The arber reads:
          m1_ask = full_top_offers[map1_ticker]      # YES ask for our team
          m1_bid = full_top_bids[map1_ticker]         # YES bid for our team (used for NO ask calc)
          m1_no_ask = 100 - full_top_bids[map1_ticker]  # NO ask for our team
        """
        entry = self._event_cache.get(event_base)
        if not entry:
            return False

        teams = entry.get("teams", [])
        if len(teams) < 2:
            return False

        # Determine map ticker prefix
        # KXLOLGAME-26APR... → KXLOLMAP-26APR...
        parts = event_base.split("-")
        if len(parts) < 2:
            return False
        series_prefix = parts[0]
        date_teams = parts[1]

        if "MATCH" in series_prefix:
            map_prefix = series_prefix.replace("MATCH", "SETWINNER")
        else:
            map_prefix = series_prefix.replace("GAME", "MAP")

        injected_any = False
        stale_legs = []

        for leg, map_num in [("game1", "1"), ("game2", "2"),
                             ("game3", "3"), ("game4", "4"), ("game5", "5")]:
            leg_data = entry.get(leg)
            if not leg_data:
                continue

            book_a = self._fetch_book(leg_data["token_a"], cache_only=cache_only)
            if book_a is None:
                # No book for this leg. _fetch_book returns None on ANY non-200
                # or exception, so this covers two very different cases and
                # cannot tell them apart here:
                #   (a) the leg RESOLVED — Polymarket 404s /book once a market
                #       closes. Expected and harmless for an already-decided
                #       map; verified 2026-07-19 on PARIBB (closed=True legs
                #       404, open legs 200).
                #   (b) a genuine timeout / network failure — unknown price.
                # Either way we leave the map ticker ABSENT (no stale-cache
                # fallback; removed 2026-07-16). That only suppresses theos if
                # this leg is the ACTIVE map — the caller's active-map guard
                # decides that, not this function.
                stale_legs.append(map_num)
                continue

            bid_a_raw, ask_a_raw = book_a

            # Sanity: clamp to 1-99
            if bid_a_raw >= 99:
                bid_a, offer_a = 99, 100
            elif ask_a_raw <= 1:
                bid_a, offer_a = 0, 1
            else:
                bid_a = max(0, round(bid_a_raw))
                offer_a = min(100, round(ask_a_raw))

            bid_b = max(0, 100 - offer_a)
            offer_b = min(100, 100 - bid_a)

            team_a = teams[0]
            team_b = teams[1]
            ticker_a = f"{map_prefix}-{date_teams}-{map_num}-{team_a}"
            ticker_b = f"{map_prefix}-{date_teams}-{map_num}-{team_b}"

            top_level_bids[ticker_a] = bid_a
            top_level_offers[ticker_a] = offer_a
            top_level_bids[ticker_b] = bid_b
            top_level_offers[ticker_b] = offer_b

            # Record the map tickers we FRESHLY injected this cycle so the caller
            # can force-absent any poly-source map ticker that did NOT get a fresh
            # inject (i.e. still carries the pregame Kalshi default seeded upstream).
            if injected_out is not None:
                injected_out.add(ticker_a)
                injected_out.add(ticker_b)

            # Ensure dt_market_state has entries so the arber doesn't skip
            for t in [ticker_a, ticker_b]:
                if t not in dt_market_state:
                    dt_market_state[t] = {}
                # Don't overwrite status/result if already set from Kalshi metadata
                if "status" not in dt_market_state[t]:
                    dt_market_state[t]["status"] = "active"

            injected_any = True

        if stale_legs:
            # Severity reflects blast radius, not the count of missing legs:
            #   nothing injected at all -> the event has no map prices, theos
            #     will be suppressed. Genuinely serious.
            #   some legs injected      -> almost always already-resolved maps
            #     (Poly 404s /book after a market closes). Theos are suppressed
            #     ONLY if the missing leg is the active map, which the caller's
            #     active-map guard determines — so do not assert it here.
            # Reworded 2026-07-19: the old line claimed "timed out" (never
            # checked) and "suppressing theos" (unconditional text), which read
            # as an outage while the event was quoting normally.
            self._log_stale_legs(event_base, stale_legs, injected_any)

        if injected_any:
            log.debug(f"[POLY FEED] Injected map prices for {event_base}: "
                      f"teams={teams}")

        return injected_any

    def inject_all(
        self,
        active_events: list[tuple[str, str]],
        top_level_bids: dict,
        top_level_offers: dict,
        dt_market_state: dict,
        full_market_state: dict = None,
        injected_out: set = None,
    ) -> list[str]:
        """Discover and inject Poly map prices for all active events.

        active_events: [(event_base, team_suffix), ...]
        full_market_state: optional dict with raw market data including yes_sub_title
        Returns list of event_bases that failed (need Kalshi fallback).
        """
        failed = []

        # Pick up populator writes without requiring a restart.
        self._reload_csv_if_changed()

        # Phase 0: Retire events whose series has been decided for 15 min.
        # Retired events are dropped silently — NOT added to `failed`, because
        # `failed` means "Poly unavailable, suppress this cycle" and the caller
        # logs a warning for each one. A settled game is not an outage, and
        # warning about it every tick would just replace one log storm with
        # another. Its map tickers still get force-absented downstream by the
        # PREGAME MAP SCRUB (they receive no fresh inject), so a retired event
        # cannot trade on a stale book.
        # Kill switch, matching disable_pregame_map_scrub.flag. Retirement is
        # permanent for the process lifetime, so if it ever fires wrongly there
        # is no in-band recovery — `touch disable_series_retirement.flag` stops
        # further retirements without a restart. Already-retired events stay
        # retired; restart to clear those.
        if os.path.exists("disable_series_retirement.flag"):
            live_events = list(active_events)
        else:
            live_events = [
                (eb, ts) for eb, ts in active_events
                if not self._is_series_settled(eb, top_level_bids, top_level_offers)
            ]

        # Phase 1: Discover all events (gamma API — cached, infrequent)
        discovered_events = []
        for event_base, team_suffix in live_events:
            kalshi_mkts = None
            if full_market_state:
                kalshi_mkts = [
                    m_data.get("market", m_data) for t, m_data in full_market_state.items()
                    if t.startswith(event_base + "-") and isinstance(m_data, dict)
                ]
            if self._discover_event(event_base, team_suffix, kalshi_mkts):
                discovered_events.append((event_base, team_suffix))
            else:
                failed.append(event_base)

        # Phase 2: Collect all token IDs that need price fetches
        all_tokens = set()
        ws_pub = {}          # {token: leg_label} published to the WS listener
        for event_base, _ in discovered_events:
            entry = self._event_cache.get(event_base, {})
            for leg in ["game1", "game2", "game3", "game4", "game5"]:
                leg_data = entry.get(leg)
                if leg_data and leg_data.get("token_a"):
                    all_tokens.add(leg_data["token_a"])
                    # Readable label for [POLY FROZEN] warnings
                    self._token_label[leg_data["token_a"]] = f"{event_base} {leg}"
                    # Also publish token_b: the listener maintains books for
                    # both sides so its logs are complete, and subscribing to
                    # a token we never read costs nothing.
                    ws_pub[leg_data["token_a"]] = (
                        f"{event_base}|M{leg[-1]}" if leg.startswith("game")
                        else f"{event_base}|{leg}")
                    if leg_data.get("token_b"):
                        ws_pub[leg_data["token_b"]] = (
                            f"{event_base}|M{leg[-1]}B" if leg.startswith("game")
                            else f"{event_base}|{leg}B")

        # Publish the token set for the WS listener (throttled, atomic,
        # exception-safe — must never be able to break the trading loop).
        try:
            if ws_pub:
                _publish_ws_tokens(ws_pub)
        except Exception as _e:
            log.debug(f"[WS TOKENS] publish error (ignored): {_e}")

        # Phase 3: Batch fetch all books in parallel (~200ms instead of N*200ms)
        # Record what we attempted so Phase 4's cache_only path can distinguish
        # "the batch tried and failed" from "never batched" (see _fetch_book).
        self._batched_this_tick = set(all_tokens)
        if all_tokens:
            self._batch_fetch_books(list(all_tokens))

        # Phase 4: Inject prices using cached results.
        # cache_only=True: Phase 3 just fetched every one of these tokens, so a
        # cache miss here means that fetch failed — re-fetching would block the
        # trading loop ~130ms per failed leg, serially, for a retry issued
        # milliseconds later under identical conditions.
        for event_base, team_suffix in discovered_events:
            injected = self.inject_map_prices(
                event_base, team_suffix,
                top_level_bids, top_level_offers, dt_market_state,
                injected_out=injected_out,
                cache_only=True,
            )
            if not injected:
                failed.append(event_base)

        return failed

    def get_diagnostics(self) -> str:
        """Return a summary of cached events and recent prices for logging."""
        lines = []
        for event_base, entry in sorted(self._event_cache.items()):
            teams = entry.get("teams", [])
            for leg in ["game1", "game2", "game3", "game4", "game5"]:
                leg_data = entry.get(leg)
                if not leg_data:
                    continue
                cached_a = self._price_cache.get(leg_data["token_a"])
                if cached_a and isinstance(cached_a[0], tuple):
                    bid, ask = cached_a[0]
                    mid_a = f"{bid:.0f}/{ask:.0f}c"
                elif cached_a:
                    mid_a = f"{cached_a[0]}c"
                else:
                    mid_a = "?"
                age = f"{time.time() - cached_a[1]:.0f}s" if cached_a else "?"
                lines.append(f"  {event_base} {leg}: {teams[0]}={mid_a} (age={age})")
        return "\n".join(lines) if lines else "  No cached prices"
