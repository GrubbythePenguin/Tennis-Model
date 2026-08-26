import sys
import os
import csv
import time
import json
import unicodedata
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# WS market-status probe — replaces /markets?event_ticker= REST polling for
# the gate-data needs in `_refresh_g2_probabilities`. One batched WS subscribe
# classifies every map ticker (active vs finalized) AND returns top-of-book
# in the same response. The dedicated ~3-5 token/sec /markets? bucket was
# the source of the ~30 × 429 retries per populate cycle (2026-06-24).
from ws_market_status import probe_markets as _ws_probe_markets


# ─── Kalshi market metadata cache ─────────────────────────────────────────
# Caches the fields the WS orderbook payload does NOT carry: `yes_sub_title`
# (team name, needed for Poly alignment) and `rules_primary` (used by the
# VCT detector). Together with WS-probed status + book, this lets the
# per-event series fetch (line ~2069) skip its REST call for any event we
# already learned the metadata for — the last per-event /markets? hit on
# the populate hot path.
#
# Cache structure (JSON on disk):
#   {
#     "KXVALORANTGAME-26JUN241500LEVKR": {
#       "tickers": [
#         {"ticker": "KX...-LEV", "yes_sub_title": "Leviatan", "rules_primary": "..."},
#         {"ticker": "KX...-KR",  "yes_sub_title": "KRU Esports", "rules_primary": "..."}
#       ]
#     }, ...
#   }
#
# yes_sub_title and rules_primary are static for an event's lifetime, so
# the cache is write-once-per-event. Entries for settled events linger
# harmlessly; cleanup is deferred to a future refactor.
_KALSHI_METADATA_CACHE_FILE = "_kalshi_market_metadata_cache.json"


def _load_kalshi_metadata_cache(path: str = _KALSHI_METADATA_CACHE_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.loads(f.read())
    except Exception as e:
        print(f"  [METADATA-CACHE] load failed: {type(e).__name__}: {e}")
        return {}


def _save_kalshi_metadata_cache(cache: dict, path: str = _KALSHI_METADATA_CACHE_FILE) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(cache, indent=2, sort_keys=True))
        os.replace(tmp, path)
    except Exception as e:
        print(f"  [METADATA-CACHE] save failed: {type(e).__name__}: {e}")


# 429-retry wrapper added 2026-06-10. Drop-in replacement for requests.get
# that sleeps 5s on every 429 and retries (up to max_retries=20 → ~100s
# max wait). populate_configs is not time-sensitive — better to wait out
# the rate limit than silently drop a live ticker from template_quoter_config.
# Internal call uses requests.request("GET", ...) so a global
# `_get_with_retry(` → `_get_with_retry(` rewrite doesn't cause recursion.
def _get_with_retry(url, **kwargs):
    max_retries = kwargs.pop("max_retries", 20)
    last_resp = None
    for attempt in range(max_retries):
        try:
            r = requests.request("GET", url, **kwargs)
            last_resp = r
            if r.status_code != 429:
                return r
            print(f"  [429-RETRY] {url[:100]} attempt {attempt + 1}/{max_retries} — sleeping 5s")
            time.sleep(5.0)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [REQ-ERR] {url[:80]} attempt {attempt + 1}: {e} — sleeping 1s")
            time.sleep(1.0)
    return last_resp

# Phase 3 active-ticker scanner (2026-06-10): subscribes to Kalshi WS trade
# feed and decides whether each candidate ticker should be in the live config
# based on recent trade activity. Markets with no trades 10+ min after their
# scheduled start get dropped from template_quoter_config.csv on the next
# daemon cycle, which the bot picks up via hot-reload. Has a built-in health
# fail-open so a broken WS feed cannot cause every ticker to be marked inactive.
sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
from dotenv import load_dotenv
load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
from kalshi_auth import KalshiAuth
from trade_activity_tracker import TradeActivityTracker, parse_scheduled_start

# ── Kalshi exchange-status write blackout (26AUG13, operator rule) ──────────
# "Do not write ANYTHING when Kalshi markets are down." During the 07-09Z
# maintenance 26AUG13, frozen-but-readable Kalshi books fed the intermission
# solver stale snapshots -> confident wrong writes -> ~$50k
# (FOXDRX/OSGDRX, see memory). The exchange status endpoint is the
# authoritative down signal. FAIL-CLOSED FOR WRITERS: only a confirmed
# exchange_active AND trading_active returns True — errors, timeouts and
# unknown states all read as "down", because a MISSED write costs ~nothing
# and a poisoned write costs five figures. Cached 120s (one status call per
# ~cycle). Consumers: the anchor write path and the G2-refresh write path.
_KALSHI_STATUS = {"ts": 0.0, "active": False, "auth": None}


def _kalshi_auth():
    if _KALSHI_STATUS["auth"] is None:
        _key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
        _pk = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not _pk.startswith("/"):
            _pk = ("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/"
                   f"esports_arbitrage/{_pk}")
        _KALSHI_STATUS["auth"] = KalshiAuth(_key_id, _pk)
    return _KALSHI_STATUS["auth"]


def kalshi_trading_active(max_age_s: float = 120.0) -> bool:
    now = time.time()
    if now - _KALSHI_STATUS["ts"] < max_age_s:
        return _KALSHI_STATUS["active"]
    active = False
    try:
        path = "/trade-api/v2/exchange/status"
        r = requests.get(
            "https://external-api.kalshi.com" + path,
            headers=_kalshi_auth().get_headers("GET", path),
            timeout=6)
        if r.status_code == 200:
            d = r.json()
            # STRICT booleans only: a string "false" is truthy in Python, and
            # this guard must never let a type quirk open the write path.
            active = (d.get("exchange_active") is True
                      and d.get("trading_active") is True)
        else:
            print(f"  [KALSHI-STATUS] HTTP {r.status_code} — treating as DOWN "
                  f"(writes suppressed)")
    except Exception as e:
        print(f"  [KALSHI-STATUS] unreachable ({type(e).__name__}) — treating "
              f"as DOWN (writes suppressed)")
        active = False
    # Record the INACTIVE -> ACTIVE edge so writers can hold off while the
    # books reprice (see kalshi_writes_safe).
    if active and _KALSHI_STATUS.get("active") is False:
        _KALSHI_STATUS["resumed_ts"] = now
        print(f"  [KALSHI-STATUS] trading RESUMED — probability writes held "
              f"for {POST_HALT_COOLDOWN_S:.0f}s while books reprice")
    _KALSHI_STATUS["ts"] = now
    _KALSHI_STATUS["active"] = active
    return active


# ── Post-halt stabilisation window (26AUG20, HLEDK) ────────────────────────
# Kalshi reporting trading_active again does NOT mean its books are repriced.
# HLEDK 26AUG20: the 07:00-09:00Z halt meant "0 pregame snapshot(s) in the
# 15min window", then game 1 finished right as trading resumed and ALL SIX
# intermission snapshots landed in the first 5 minutes of a book still
# catching up from its frozen pre-game level. The leader back-out
#     p3 = (series - m2) / (1 - m2)
# ran on a series price ~13c stale; with m2~0.50 the denominator is 0.50, so
# the error DOUBLED to ~27 points (true p3 ~0.52, computed 0.250). Only the
# 0.15 delta cap kept it from landing in full — it still wrote 0.374.
#
# INTERMISSION_MAX_AMPLIFICATION cannot catch this: it bounds how much a SMALL
# error grows (a conditioning test on the denominator) and says nothing about
# whether the input is fresh. 1c at 2x is fine; 13c at 2x is a disaster, and
# both look identical to a denominator check.
#
# So: gate probability writes on trading being active AND having been active
# long enough for the book to reprice. The existing rule checks the WRITE
# moment; this checks the PROVENANCE of the data being written.
POST_HALT_COOLDOWN_S = 300.0


# ── WEEKLY MAINTENANCE BLACKOUT (26AUG20, operator) ────────────────────────
# HARD blackout: Thursdays 03:00-05:10 America/New_York. NO probability writes
# and NO snapshot collection, regardless of what the exchange-status endpoint
# claims. Operator: refreshing across this window has cost ~$50k in two weeks.
#
# Why hard-coded and not just status-driven: the status endpoint is a REPORT.
# On 26AUG20 it flipped trading_active=True at 09:00Z while books were still
# 20c wide, and the HLEDK intermission back-out took all six snapshots inside
# that window (see the width guard in pregame_anchor). A wall-clock blackout
# does not depend on the venue telling us the truth about itself.
#
# 05:10 not 05:00: the extra 10 minutes is the reprice tail. Books reopen wide
# and need time to tighten before any mid is meaningful.
#
# America/New_York (NOT a fixed UTC offset) so the window tracks EST/EDT.
WEEKLY_BLACKOUT_DOW = 3            # Monday=0 ... Thursday=3
WEEKLY_BLACKOUT_START = (3, 0)     # 03:00 ET
WEEKLY_BLACKOUT_END = (5, 10)      # 05:10 ET
WEEKLY_BLACKOUT_TZ = "America/New_York"
WEEKLY_BLACKOUT_OFF_FLAG = "disable_weekly_blackout.flag"


def in_weekly_blackout(now=None) -> tuple:
    """(blocked, reason) for the hard Thursday 03:00-05:10 ET blackout.

    Fails CLOSED on a timezone error: if we cannot establish the local time we
    cannot prove we are outside the window, and the whole point is to stop
    writing blind.
    """
    if os.path.exists(WEEKLY_BLACKOUT_OFF_FLAG):
        return False, "blackout disabled by flag"
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime as _dt
        et = (now or _dt.now(ZoneInfo(WEEKLY_BLACKOUT_TZ)))
        if et.tzinfo is None:
            et = et.replace(tzinfo=ZoneInfo(WEEKLY_BLACKOUT_TZ))
        else:
            et = et.astimezone(ZoneInfo(WEEKLY_BLACKOUT_TZ))
    except Exception as e:
        return True, f"blackout: timezone unavailable ({type(e).__name__}) — failing closed"
    if et.weekday() != WEEKLY_BLACKOUT_DOW:
        return False, "ok"
    mins = et.hour * 60 + et.minute
    start = WEEKLY_BLACKOUT_START[0] * 60 + WEEKLY_BLACKOUT_START[1]
    end = WEEKLY_BLACKOUT_END[0] * 60 + WEEKLY_BLACKOUT_END[1]
    if start <= mins < end:
        return True, (f"WEEKLY BLACKOUT Thu {WEEKLY_BLACKOUT_START[0]:02d}:"
                      f"{WEEKLY_BLACKOUT_START[1]:02d}-{WEEKLY_BLACKOUT_END[0]:02d}:"
                      f"{WEEKLY_BLACKOUT_END[1]:02d} ET (now {et:%a %H:%M} ET)")
    return False, "ok"


def kalshi_writes_safe() -> tuple:
    """(safe, reason) — hard blackout, trading active, AND past post-halt cooldown.

    Use for ANY probability write OR snapshot collection. `kalshi_trading_active()`
    alone is not enough: it returns True the instant Kalshi reopens, while the
    books are still repriced from a frozen state.
    """
    _bl, _blwhy = in_weekly_blackout()
    if _bl:
        return False, _blwhy
    if not kalshi_trading_active():
        return False, "kalshi inactive/unconfirmed"
    resumed = _KALSHI_STATUS.get("resumed_ts") or 0.0
    if resumed:
        age = time.time() - resumed
        if age < POST_HALT_COOLDOWN_S:
            return False, (f"post-halt cooldown {age:.0f}s/"
                           f"{POST_HALT_COOLDOWN_S:.0f}s — books still repricing")
    return True, "ok"


# ── Order-probe: PROOF of open trading, not a report of it (26AUG13) ────────
# The exchange-status endpoint is a REPORT and can disagree with reality —
# during the 26AUG13 maintenance the books were frozen-but-readable and the
# FOXDRX intermission write landed anyway. The one statement Kalshi cannot
# make wrongly is ACCEPTING AN ORDER. So before any anchor write lands, place
# a 1-lot YES bid at 1c on one of the slate's own live books and cancel it:
#   accepted                        -> trading is really open, write proceeds
#   rejected / error / no candidate -> treat as PAUSED, suppress the write
# Safety of the probe itself: candidates require yes_ask >= 5c so a 1c bid
# can NEVER cross (unfillable by construction, not just unlikely), and
# place_order stamps expiration_ts=now+60s on GTC, so even a failed cancel
# self-destructs in 60s. Orders MUST go through KalshiClient — raw posts to
# the order paths 410 (reference_kalshi_order_endpoints_v2_410).
# Fail-closed like the status gate: a missed write costs ~nothing, a poisoned
# write costs five figures. Cached 120s either way; it is only invoked when a
# write is imminent, so real traffic is 1 order + 1 cancel a few times a day.
_ORDER_PROBE = {"ts": 0.0, "ok": False, "client": None}


def kalshi_order_probe_ok(candidates, book_fn, max_age_s: float = 120.0) -> bool:
    now = time.time()
    if now - _ORDER_PROBE["ts"] < max_age_s:
        return _ORDER_PROBE["ok"]
    ok = False
    try:
        import uuid
        from kalshi_client import KalshiClient   # esports_arbitrage sys.path
        if _ORDER_PROBE["client"] is None:
            _ORDER_PROBE["client"] = KalshiClient(_kalshi_auth())
        cl = _ORDER_PROBE["client"]
        ticker, seen = None, set()
        for t in candidates:
            if not t or t in seen:
                continue
            seen.add(t)
            b = book_fn(t)
            if (b and b[0] is not None and b[1] is not None
                    and b[0] < b[1] <= 99 and b[1] >= 5):
                ticker = t
                break
        if ticker is None:
            print("  [KALSHI-PROBE] no live two-sided book to probe on — "
                  "treating trading as PAUSED (writes suppressed)")
        else:
            resp = cl.place_order(str(uuid.uuid4()), ticker, "yes", 1, 1,
                                  time_in_force="gtc") or {}
            oid = (resp.get("order_id")
                   or (resp.get("order") or {}).get("order_id"))
            if oid:
                ok = True
                canceled = cl.cancel_order(oid)
                print(f"  [KALSHI-PROBE] trading OPEN — 1-lot @1c accepted on "
                      f"{ticker} (order {oid}, canceled={canceled}"
                      + ("" if canceled else ", auto-expires in 60s") + ")")
            else:
                print(f"  [KALSHI-PROBE] 1-lot @1c REJECTED on {ticker} — "
                      f"trading PAUSED/unconfirmed (writes suppressed)")
    except Exception as e:
        print(f"  [KALSHI-PROBE] probe failed ({type(e).__name__}: {e}) — "
              f"treating trading as PAUSED (writes suppressed)")
        ok = False
    _ORDER_PROBE["ts"] = now
    _ORDER_PROBE["ok"] = ok
    return ok

# Ticker alias layer — synthetic events (e.g. tournament finals listed on
# Kalshi as KXCS2-IEMCOL26-{FAL,FURIA} but traded internally as
# KXCS2GAME-26JUN211100FALFUR-{FAL,FUR}) bypass Kalshi's event-discovery
# API. We synthesize active_markets from ticker_aliases.csv directly.
import ticker_aliases


def _teams_for_synthetic_event_base(event_base: str) -> list[str]:
    """Return sorted team suffixes registered under this synthetic event_base.

    Derived from ticker_aliases.csv. For 'KXCS2GAME-26JUN211100FALFURIA'
    with entries '-FAL' and '-FURIA', returns ['FAL', 'FURIA']. Empty if no
    matching aliases (caller skips emission).

    Always calls `reload_if_changed()` first so a long-running populate
    daemon picks up mid-session CSV edits without a restart. This was the
    2026-06-21 FALFURIA wipe — populate ran with stale FALFUR aliases
    after the rename, missed the synth bypass, and emitted an empty
    template_quoter_config row set for the event."""
    ticker_aliases.reload_if_changed()
    teams = set()
    for synth, _ in ticker_aliases.all_pairs():
        if synth.startswith(event_base + "-"):
            teams.add(synth.rsplit("-", 1)[1])
    return sorted(teams)


def _real_series_event_base(event_base: str) -> str:
    """For a synthetic event_base, return the REAL series event_base by
    resolving any registered synthetic market ticker through the alias map and
    stripping the team suffix. Non-synthetic events return unchanged.

    e.g. synthetic 'KXLOLGAME-26JUL0219007DVTC' whose alias maps
    '...-26JUL02...-7D' -> '...-26JUL14...-7D' yields the real series event
    'KXLOLGAME-26JUL1419007DVTC'. Used so the probability-capture series fetch
    can read the REAL market price for an aliased event (the synthetic event
    ticker doesn't exist on Kalshi)."""
    if not ticker_aliases.event_base_is_synthetic(event_base):
        return event_base
    for synth, real in ticker_aliases.all_pairs():
        if synth.startswith(event_base + "-") and real:
            return real.rsplit("-", 1)[0]
    return event_base

# Import MLB Tracker natively to test pure LIVE constraints!
if "/Users/bradleyguan/Documents/Coding/kalshi_mlb_tracker" not in sys.path:
    sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_mlb_tracker")
from trading.mlb_game_state import get_instance, KALSHI_TO_MLB

# ── Polymarket fallback for esports probability capture ──
# Uses Polymarket prices when Kalshi spreads are too wide (>2c).
# Polymarket has dramatically tighter spreads on esports, especially Asian leagues.

POLY_SPORT_TAGS = {"dota2": 102366, "lol": 65, "cs2": 100780, "val": 101672, "cod": 100230}
KALSHI_TO_POLY_SPORT = {"KXDOTA2": "dota2", "KXLOL": "lol", "KXCS2": "cs2", "KXVALORANT": "val", "KXCOD": "cod"}

# Per-game momentum bumps written into the probs CSV (g2_momentum, g3_momentum columns).
# Keys are matched against `event_base` via substring containment.
MOMENTUM_BY_GAME = {
    # (g2_momentum, g3_momentum) — bonus to map-N prob given who won map-N-1.
    # g3 raised 0.04 → 0.10 for Valorant on 2026-06-28 per fresh G2→G3 audit
    # (n=36 G3 deciders, actual G2-winner-wins-G3 = 61.1% vs market-implied 49.2%
    # → ~12pp lift ≈ 0.11 symmetric g3_momentum). Cap may be needed at extremes.
    "KXVALORANT": (0.06, 0.10),   # tier-2 VAL g2 0.03→0.06 (2026-07-05, map-2 market pricing)
    "KXCS2":      (0.04, 0.02),   # tier-2 CS g2 0.04; g3 0.04→0.02 (2026-07-18, FUTGM experiment)
    "KXDOTA2":    (0.03, 0.01),
}
# Per-tier override applied AFTER the base lookup (see capture loop). VCT
# (Valorant tier-1) lowered to CS2 values 2026-06-29; the VAL g3=0.10 audit
# was VCL-dominated and tier-1 books behave differently.
# TEMPORARY (2026-07-04): bumped tier-1 g3_momentum. VCT normal value is
# (0.02, 0.06); ran at g3=0.10 for a stretch, then tuned back to 0.07 on
# 2026-07-17 (per user — a step down toward the 0.06 baseline, not a full revert).
# 2026-07-05: CS2 g3 unified to 0.08 across both tiers (T1 0.10→0.08, T2 0.06→0.08).
# 2026-07-18 EXPERIMENT: VCT + both CS2 tiers g3 → 0.02 after the FUTGM post-mortem.
# In state 1/2 the bump enters the hedge as [P(win G3) + g3_momentum] * map-2 ask, so
# the phantom edge is g3_momentum * map-2 ask — largest exactly when the trailing team
# is winning map 2 (i.e. mid-comeback). On FUTGM (26JUL18, VCT) g3=0.06 with a 74c
# map-2 ask manufactured +4.46c of the +5.22c apparent edge; at g3=0 the edge fell to
# +0.76c, under min_edge. Also note solve_game_3_probability documents that real G3
# prices sit -0.3c from avg(p1,p2), i.e. the base needs no bump at all.
# VCL deliberately UNCHANGED at 0.10 (MOMENTUM_BY_GAME["KXVALORANT"]) as the control.
MOMENTUM_VAL_VCT = (0.04, 0.02)   # tier-1 VAL g2 0.04; g3 0.06→0.02 (2026-07-18, FUTGM experiment)
MOMENTUM_CS_T1   = (0.04, 0.02)   # tier-1 CS2 g2 0.04; g3 0.04→0.02 (2026-07-18, FUTGM experiment)
DEFAULT_MOMENTUM = (0.0, 0.0)

TEAM_ALIASES = {
    # Dota 2
    "NAVI": ["natus vincere", "navi"], "XTREME": ["xtreme gaming", "xtreme"],
    "MOUZ": ["mouz"], "TS": ["team spirit", "spirit", "ts"], "VG": ["vici gaming"],
    "GL": ["gamerlegion"], "LIQUID": ["team liquid", "liquid"],
    "SAR": ["south america rejects", "sar"], "AUR": ["aurora"],
    "PARI": ["parivision", "pari"], "VP": ["virtus.pro", "virtus"],
    "BB": ["betboom", "bet boom"], "FLC": ["team falcons", "falcons"],
    "FAL": ["team falcons", "falcons"],
    "TY": ["team yandex", "yandex"], "HEROIC": ["heroic"], "TUNDRA": ["tundra"],
    # LoL
    "GEN": ["gen.g", "geng", "gen g"], "GENG": ["gen.g", "geng", "gen.g esports"],
    "HLE": ["hanwha life", "hle"], "BLG": ["bilibili gaming", "blg"],
    "IG": ["invictus gaming"], "DCG": ["deep cross gaming", "dcg"],
    "CFO": ["ctbc flying oyster", "cfo"], "KT": ["kt rolster", "kt"],
    "DNF": ["dnf", "dplus"], "JDG": ["jd gaming", "jdg"],
    "WB": ["weibo gaming", "weibo"], "MVK": ["mvk"],
    "DFM": ["detonation", "dfm"], "NAVI_LOL": ["navi"],
    "GX": ["giantx", "giant"], "FUR": ["furia", "fur"], "FURIA": ["furia"],
    "FLU": ["fluxo", "flu"], "SK": ["sk gaming", "sk"],
    "G2": ["g2 esports", "g2"], "PNG": ["png", "pain gaming"],
    "LOS": ["los", "loud"], "C9": ["cloud9", "c9"],
    "DSG": ["disguised", "dsg"], "FLY": ["flyquest", "fly"],
    "FQ":  ["flyquest", "fly"],
    "SEN": ["sentinels", "sen"], "T1": ["t1"], "DK": ["dplus kia", "dk"],
    "FOX": ["foxit", "fox"], "DRX": ["drx", "kiwoom drx"],
    "BRO": ["brion", "bro", "hanjin brion"],
    "NIP": ["ninjas in pyjamas", "nip"], "TES": ["top esports", "tes"],
    "OMG": ["oh my god", "omg"], "TT": ["thundertalk", "tt"],
    "WE": ["team we"], "AL": ["anyone's legend", "anyone"],
    "NS": ["nongshim", "nongshim redforce", "nongshim red force"],
    "SHG": ["softbank hawks", "fukuoka softbank hawks"],
    "TSW": ["team secret whales", "secret whales"],
    "GAM": ["gam esports", "gam"], "GZ": ["ground zero", "ground zero gaming"],
    "VIT": ["vitality", "team vitality"], "LY": ["lyon"],
    "SR": ["shopify rebellion", "shopify", "srb"], "DIG": ["dignitas", "dig"],
    "MKOI": ["movistar koi", "movistar"], "TH": ["team heretics", "heretics"],
    "VKS": ["vivo keyd", "keyd stars"], "RED": ["red canids", "red academy"],
    "LLL": ["lll", "loud leviatan"],
    # Valorant
    "GE": ["gentle mates", "global esports", "ge"],
    "EG": ["evil geniuses", "eg"], "FNC": ["fnatic", "fnc"],
    "PCFIC": ["pacific", "pcific", "pcific esports"], "TL": ["team liquid", "liquid"],
    "FUT": ["fut esports", "fut academy", "fut"],
    "FUTA": ["fut academy", "futa"], "NV": ["natus vincere", "navi"],
    "NVU": ["nvu"], "LEV": ["leviatan", "leviatán"],
    "LOUD": ["loud"], "MIBR": ["mibr"], "ECO": ["ecoround", "eco"],
    "AH": ["apeks heroes", "apeks"], "WIP": ["wip esports", "wip"],
    "F9": ["f9 eicar", "f9"], "AGOS": ["ghosts", "gos"], "GOS": ["ghosts", "gos"],
    "FKS": ["fks", "fokus"], "RZN": ["rzn", "rizon"],
    "CAL": ["caldya", "cal", "cla"], "GAL": ["galions", "gal", "gl"],
    "DRG": ["dragon ranger", "drg"], "TYLOO": ["tyloo"],
    "EDG": ["edward gaming", "edg"], "ZETA": ["zeta division", "zeta"],
    "VARS": ["varrel", "var"], "RRQ": ["rex regum qeon", "rrq"],
    "TE": ["trace esports", "trace"], "TEC": ["tec esports"],
    "XLG": ["xlg gaming", "xlg"], "AG": ["all gamers", "alliance guardians", "axg"],
    "ES": ["esprit", "esprit shōnen", "eintracht spandau", "spandau"],
    "GS": ["galatasaray", "galatasaray esports"],
    "RBN": ["reborn"], "CGN": ["cgn esports", "cgn"],
    "SGE": ["eintracht frankfurt", "eintracht"],
    "ATTAX": ["alternate attax", "attax", "atn"],
    "DOR": ["dortmund", "dortmund esports", "dxg"],
    "MDR": ["mandatory", "mdr"], "JL": ["joblife", "jl"],
    "NUX": ["nuxeria", "nux", "nxr"], "100T": ["100 thieves", "100t"],
    "BJK": ["beşiktaş", "besiktas", "beşiktaş esports"],
    "PCFX": ["pcific", "pcific esports"], "EF": ["eternal fire"],
    "BBL": ["bbl esports", "bbl"], "KC": ["karmine corp", "karmine"],
    # LoL Academy / Challengers
    "GENGA": ["gen.g global academy", "gen.g academy", "geng academy"],
    "NSEA": ["nongshim esports academy", "nongshim academy"],
    "SHFT": ["shifters", "shft"],
    "KTC": ["kt challengers", "kt rolster challengers"],
    "FOXY": ["bnk fearx youth", "fearx youth", "foxy"],
    "DRXC": ["drx challengers", "kiwoom drx challengers"],
    "CDKC": ["dplus kia challengers", "dk challengers", "dkc"],
    "DNSC": ["dn sports challengers", "dnsc"],
    "CHLE": ["hanwha life challengers", "hle challengers"],
    # LoL Rift/Road of Legends
    "SLY": ["solary", "sly"],
    "DV1": ["division 1", "dv1"], "LDS": ["lds", "lidakost"],
    "BAN": ["banan", "ban"], "SEC": ["sector one", "sec"],
    "BOM": ["bombat", "bombat gaming"], "DOC": ["docler", "doc"],
    "ZNT": ["zenith", "znt"], "MYTH": ["myth esports", "myth"],
    "NES": ["eintracht spandau", "spandau"],
    "G2N": ["g2 nord", "g2nord"],
    "UB": ["ub alma mater", "ub alma"], "FLK": ["falke", "falke esports"],
    "USE": ["unicorns of love sexy", "unicorns sexy edition"],
    "TOG": ["team orange gaming", "orange gaming"],
    "OA": ["orbit anonymo", "orbit"], "BCE": ["barcząca", "barczaca esports"],
    "UCAM": ["ucam esports", "ucam"],
    # Valorant VCL
    "DP": ["dark passage"], "ETE": ["eternal fire passion", "efp", "ete"],
    "FIRE": ["fire flux", "fire flux esports", "ff"],
    "D1": ["division one"], "NBG": ["nightblood", "nightblood gaming"],
    "SAD": ["sad esports", "sad"], "NRGA": ["nrg academy"],
    "PGS": ["pigeons", "pgs", "coo"],
    "OTA": ["otakar", "otakar esports", "otk"],
    "EVI": ["team evictix", "evictix", "evx"],
    "TLANA": ["team liquid academy", "tl academy"],
    "VIJ": ["navi juniors", "natus vincere juniors", "navi junior"],
    "ADG": ["adventus gaming", "adg"], "QOR": ["qor gaming", "qor"],
    "BAR": ["barça", "barca esports", "barça esports", "barca", "bar"],
    "HRTS": ["heretics", "heretics esports", "hrts"],
    # IEM Cologne 2026 (added 2026-06-03 for splitter robustness; subtitle
    # matching at Stage 2 already handles these events, but Stage 1 is faster).
    "BET": ["betboom", "betboom team", "bb"],
    "M80": ["m80"],
    "B8E": ["b8", "b8 esports"],
    "EXG": ["ex-gaimin gladiators", "gaimin gladiators", "gg"],
    "HERO": ["heroic"],
    "DSY": ["dnsty", "dynasty"],
    "SRE": ["shopify rebellion", "srb"], "GA": ["gaming academy"],
    "EGA": ["ega", "eagles"],
    # CS2
    # 2026-07-26: The Mongolz vs Wildcard (KXCS2GAME-...MGLZWC, BLAST Bounty)
    # went unmapped and untraded — "WC" had no alias, so it fell back to the bare
    # token "wc" and \bwc\b never matches "wildcard" (title) or "wc1" (slug). With
    # only 1/2 teams matching, _match_poly_event returned no event → poly_url empty
    # → CS2-T3 gate dropped the whole game. Alias the WC code to the real name.
    "WC": ["wildcard gaming", "wildcard"],
    "SASHI": ["sashi esport", "sashi"], "SIN": ["sinners"],
    "ALL": ["alliance"], "KOL": ["kolesie"],
    "QUA": ["qual4", "qualifier"],
    # CS2 ESL Challenger / lower tier
    "BIGA": ["big academy"], "JUL": ["julie", "julie&cie"],
    "BMB": ["basement boys", "basement"], "ATR": ["atreides"],
    "DRIP": ["dripmen"], "CYB": ["cybershoke", "cybershoke prospects"],
    "EJ": ["endless journey"], "EXZ": ["ex-zero tenacity", "zero tenacity"],
    "MEG": ["megoshort"], "AURYB": ["aurora young blood"],
    "PRO": ["project 91"], "ECB": ["ec banga"],
    "RUST": ["rustec"], "HAS": ["hashiras"],
    "UNITY": ["unity esports"],
    "ALKAA": ["alka"], "9Z": ["9z"],
    "FC": ["fisher college", "fisher"], "MGC": ["magic"],
    "MGLZ": ["themongolz", "the mongolz", "mongolz"],
    "PRV": ["parivision"],
    "CSDIILIT": ["csdiilit"], "PLA": ["playersclub"],
    "DON": ["donstu", "donstu esports"], "G2A": ["g2 ares"],
    "JUS": ["just_players", "just players"], "BL": ["brazylijski"],
    "MIS": ["misa esports", "misa"], "CO": ["clair obscur"],
    "MSC": ["masonic"], "OLD": ["oldboys"],
    "YNG": ["yngods"], "AIM": ["aimclub"],
    "YN": ["yn", "young ninjas"], "VSC": ["vasco", "vasco esports"],
    "MW": ["metanoia wolves", "metanoia"], "ALZ": ["alzon"],
    "LGC": ["legacy"], "CRA": ["crashers"], "KEYD": ["keyd"],
}

_poly_event_cache = {}
_poly_slug_cache = {}

# Tier 1 CS = top-tier tournaments with deep Poly liquidity. We treat these as
# a different sport: bigger size + Poly is source of truth. Detection is purely
# from poly_parsed_markets.csv `poly_title`. Extend this list as new tier-1
# tournaments appear.
# 2026-07-19: "BLAST" RESTORED to tier-1 ahead of BLAST Open Porto. It had been
# removed 2026-07-09 because the BLAST Open *Qualifiers* then listed are tier-3
# and were matching this substring, taking full size + Poly source-of-truth.
# Verified safe at restore time: no BLAST event was listed in
# poly_parsed_markets.csv / parsed_markets.csv, so there is nothing for the
# substring to over-match today. The collision returns the moment qualifiers for
# a future BLAST are listed alongside the main event — if that happens, exclude
# them via blacklist.txt (one-off) rather than pulling the keyword again.
TIER1_CS_KEYWORDS = ["BLAST", "IEM", "DREAMHACK", "PGL", "ASIA CHAMPIONSHIP",
                     # TEMPORARY 2026-06-28: Super DraculaN finals (Acend vs
                     # Inner Circle, KXCS2GAME-26JUN281300ACEICE). BO5 finals
                     # behave more like tier-1 than tier-2 — bigger size, Poly
                     # source-of-truth. REMOVE this entry once the finals settle
                     # (other DraculaN brackets stay tier-2 via TIER2_CS_TOURNAMENTS).
                     "DRACULAN",
                     # TEMPORARY 2026-07-14: Stake Ranked Episode 3 Playoffs —
                     # main-event games trade tier-1 (Poly source-of-truth, full
                     # size), same as Episode 2. REMOVE after the event settles
                     # (Closed Qualifier brackets stay non-tier-1 / skipped in
                     # refresh_tomorrow_markets).
                     "STAKE RANKED",
                     "XSE PRO LEAGUE",
                     # 2026-07-21: StarLadder StarSeries → tier-1. Both brand
                     # strings listed because events surface under either name
                     # ("StarLadder StarSeries Budapest", "StarSeries i-League").
                     # Same qualifier-collision caveat as BLAST above: StarLadder
                     # also runs lower-division / qualifier brackets — if one is
                     # ever listed alongside a main event, exclude it via
                     # blacklist.txt rather than pulling the keyword.
                     "STARLADDER", "STARSERIES",
                     # 2026-08-11 (operator): EWC CS2 MAIN event → tier-1, now
                     # that play-ins are done. Qualifier brackets stay demoted
                     # via TIER1_CS_EXCLUSIONS below and re-admitted tier-2 by
                     # refresh's TIER2_CS_TOURNAMENTS entry — exactly the
                     # StarSeries pattern. Full phrase, never bare "EWC" (hides
                     # inside "Newcastle").
                     "ESPORTS WORLD CUP"]

# Substrings that FORCE tier-2 even when a TIER1_CS_KEYWORDS entry also matches.
# Checked after the tier-1 match, so it always wins.
#
# This is the mechanism the BLAST/StarLadder comments above kept deferring to
# blacklist.txt for. Blacklisting is the wrong tool when the event is still
# worth trading: it DROPS the event (and per the blacklist memory, destroys its
# poly_parsed row). A qualifier we want at 1/5 size needs a demotion, not a drop.
#
# 2026-07-28 (operator): StarLadder StarSeries → the EUROPEAN qualifiers were
# genuinely tier-1, but the NORTH and SOUTH AMERICAN qualifier brackets are
# tier-2 fields. The brand keyword alone can't tell them apart — both brand
# spellings listed because events surface under either.
TIER1_CS_EXCLUSIONS = [
    "STARSERIES NORTH AMERICAN QUALIFIER",
    "STARSERIES SOUTH AMERICAN QUALIFIER",
    "STARLADDER NORTH AMERICAN QUALIFIER",
    "STARLADDER SOUTH AMERICAN QUALIFIER",
    # EWC qualifier/play-in brackets stay tier-2 while the MAIN event (added
    # to TIER1_CS_KEYWORDS 2026-08-11) trades tier-1. Narrow phrases per the
    # guidance above; the "LCQ Qualifiers" futures event has no " vs " and is
    # already excluded by the matchup gate regardless.
    "ESPORTS WORLD CUP OPEN QUALIFIER",
    "ESPORTS WORLD CUP PLAY-IN",
    "ESPORTS WORLD CUP LCQ",
]

_cs_tier_cache = {}  # event_base -> "tier1" | "tier2"

# NOTE (2026-07-05): The VCL_EUROPE_KEYWORDS "skip" hard-block was removed here.
# EU VCL (esp. VCL EMEA) was originally blocked as map-led/bleeding, but after a
# month of iteration it is comfortably profitable, so EU VCL now classifies as
# normal tier-2 ("vcl") like every other VCL region and trades via the _VCL
# templates. Tier-2 VAL as a whole is still gated by enable_val_tier2.flag.


def is_bo2_event(s_markets) -> bool:
    """True if the event has a Tie sub-market or 3+ team markets.

    BO2 esports events (LoL HLL, LoL Prime League, etc.) list three markets:
    team A YES, team B YES, and Tie YES. Series-state math and hedge engine
    only support binary outcomes — a BO2 event with a Tie market mis-prices
    the tie tail and misclassifies state. Detection is structural (market
    count + Tie yes_sub_title), so it catches any BO2 tournament without
    needing per-league keyword maintenance.
    """
    if not s_markets or len(s_markets) < 3:
        return False
    for m in s_markets:
        sub = (m.get("yes_sub_title", "") or "").strip().lower()
        ticker = (m.get("ticker", "") or "")
        if sub == "tie" or ticker.endswith("-TIE"):
            return True
    return False


def is_vct_event(s_markets) -> bool:
    """True if any series market's rules_primary tags this as VCT (Tier 1).
    Excludes Challengers and Game Changers (Tier 2 / dev league)."""
    if not s_markets:
        return False
    for m in s_markets:
        rules = m.get("rules_primary", "") or ""
        if "VCT" in rules and "Challengers" not in rules and "Game Changers" not in rules:
            return True
    return False


def classify_cs_tier(event_base: str, poly_mappings: dict) -> str:
    """Return "tier1" or "tier2" for a CS2 event_base; "tier2" for non-CS2.

    Tier 1 iff there is a Poly mapping AND the cached poly_title contains a
    TIER1_CS_KEYWORDS entry (case-insensitive). No Poly link → guaranteed tier 2.
    Accepts both KXCS2GAME-... and KXCS2MAP-...-N tickers; normalizes to the
    KXCS2GAME-... key used in poly_parsed_markets.csv.
    """
    if not event_base or "KXCS2" not in event_base:
        return "tier2"
    key = event_base.replace("KXCS2MAP", "KXCS2GAME")
    parts = key.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        key = parts[0]
    if key in _cs_tier_cache:
        return _cs_tier_cache[key]
    m = poly_mappings.get(key)
    tier = "tier2"
    has_poly = bool(m and m.get("poly_url"))
    if has_poly:
        title_upper = (m.get("poly_title") or "").upper()
        if (any(k in title_upper for k in TIER1_CS_KEYWORDS)
                and not any(x in title_upper for x in TIER1_CS_EXCLUSIONS)):
            tier = "tier1"
    # Only cache once we have a Poly mapping. Tier2-because-no-poly-yet must
    # remain re-checkable so a later poly_parsed_markets fill-in promotes the
    # event to tier1 on the next cycle.
    if has_poly:
        _cs_tier_cache[key] = tier
    return tier


# LoL is tier-1 by DEFAULT; only these regional leagues route to tier-2 (1/5
# size via _T2 template rows). Opposite default from CS2 (which defaults tier2).
# NOTE (2026-07-04): keyword strings are PROVISIONAL — no LRS/LRN/LJL event was
# in the data to confirm the exact Poly title. Verify against a real title
# before relying on tier-2 routing (a wrong keyword silently keeps it tier-1).
# LoL is DEFAULT tier-1 (full size); only these league keywords (matched in Kalshi
# rules_primary, or Poly title) route to tier-2 / _T2 rows (~1/5–1/10 size).
# Keywords are the exact league string in rules_primary ("...wins the <LEAGUE> <YEAR>:...").
# TIER-1 (default, deliberately NOT here): LCK/LPL/LEC/LCS/LCP/CBLOL/VCS/TCL/MSI/Worlds,
#   EMEA Masters, Asia Masters, Esports World Cup(+quals), LCK Challengers League,
#   LES, Prime League, LFL.
#   (LFL was briefly moved to tier-2 on 26AUG19 and REVERTED the same day — operator
#   misread the league; it stays TIER-1 alongside the other big ERLs. Never shipped:
#   populate_configs was not restarted in between.)
#   STALE ENTRY: " TCL " above IS in TIER2_LOL_KEYWORDS as of 26AUG11, so the TCL
#   in this tier-1 line is wrong — left in place only to flag the discrepancy.
# Care: "NORTH AMERICAN CHALLENGERS LEAGUE" is spelled in full — bare "NACL" matches
#   "pinnacle" and "CHALLENGERS LEAGUE" would wrongly catch tier-1 LCK Challengers League.
#   " LIT " is space-padded — bare "LIT" matches "elite"/"facility"/"split".
# ── LEAGUE BLACKLIST (operator 26AUG20) ──────────────────────────────────────
# Leagues we refuse to trade at ANY size — integrity risk, not a sizing question.
# Distinct from TIER2_LOL_KEYWORDS (which only routes to smaller _T2 rows): a
# league here emits NO template rows at all, so no quoter/arber/momentum bot is
# ever constructed for the event.
#
# Matched exactly like the tier keywords — against Kalshi rules_primary first
# (authoritative, present even with no Poly link) then the cached Poly title —
# so it works for the minor leagues that frequently have no Poly mapping.
#
# LPLOL = Liga Portuguesa. Operator 26AUG20: "worst league and has rigged games".
# Was already tier-2 (added 26AUG05); this promotes it to a hard exclusion.
# CAUTION: "LPLOL" must never be shortened to "LPL" — tier-1 Chinese LPL and
# CBLOL both contain "LPL" and would be silently killed. Keep the full code.
BLACKLIST_LOL_LEAGUES = (
    "LPLOL",
)


def lol_league_blacklisted(event_base: str, poly_mappings: dict, s_markets=None) -> str | None:
    """Return the matched blacklist keyword, or None. Same lookup order as
    classify_lol_tier: rules_primary (authoritative) then Poly title."""
    if not event_base or "KXLOL" not in event_base or not BLACKLIST_LOL_LEAGUES:
        return None
    if s_markets:
        for m in s_markets:
            rules = (m.get("rules_primary", "") or "").upper()
            for k in BLACKLIST_LOL_LEAGUES:
                if k in rules:
                    return k
    key = event_base.replace("KXLOLMAP", "KXLOLGAME")
    parts = key.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        key = parts[0]
    m = poly_mappings.get(key)
    if m and m.get("poly_url"):
        title = (m.get("poly_title") or "").upper()
        for k in BLACKLIST_LOL_LEAGUES:
            if k in title:
                return k
    return None


TIER2_LOL_KEYWORDS = (
    "LJL", "LRN", "LRS",                    # JP + LATAM regional leagues
    "NLC",                                  # Northern EU ERL (per user 2026-07-29).
                                            # rules_primary reads "wins the NLC 2026: ...",
                                            # Poly titles "... - NLC Regular Season".
    "LPLOL",                                # Liga Portuguesa (per user 2026-08-05).
                                            # rules_primary "wins the LPLOL 2026: ...",
                                            # Poly titles "... - LPLOL Group Stage".
                                            # Safe next to tier-1 "LPL"/"CBLOL": neither
                                            # contains "LPLOL". Order matters the OTHER
                                            # way — see _lol_t2_series_stats.py, where a
                                            # bare " LPL" tier-1 stem DID swallow LPLOL.
    "ROAD OF LEGENDS", "RIFT LEGENDS",      # minor ERLs
    "HITPOINT MASTERS",                     # Czech/Slovak ERL (per user 26AUG13).
                                            # Poly titles "... - Hitpoint Masters
                                            # Regular Season". Full string — safe
                                            # next to tier-1 "EMEA Masters"/"Asia
                                            # Masters" (neither contains HITPOINT).
    "ARABIAN LEAGUE",                       # MENA
    " HLL ",                                # Hellenic Legends League, Greek ERL
                                            # (per user 26AUG25). rules_primary
                                            # "wins the HLL 2026: ...", Poly
                                            # titles "... - HLL Playoffs" — both
                                            # carry surrounding spaces. Space-
                                            # padded like " LIT "/" TCL " so a
                                            # 3-letter code can never fire inside
                                            # a team name.
    " EBL ",                                # Esports Balkan League (per user
                                            # 26AUG25). rules_primary "wins the
                                            # EBL 2026: ...", Poly titles
                                            # "... - EBL Playoffs". Space-padded
                                            # like " HLL "/" LIT "/" TCL ": a
                                            # bare "EBL" is a live substring of
                                            # tier-1 LPL ticker bodies (WEBLG =
                                            # Weibo vs Bilibili, TEBLG = TE vs
                                            # Bilibili), so keep the padding.
    "CIRCUITO DESAFIANTE",                  # Brazil tier-2 (per user 26AUG17,
                                            # NERDPNGA). Was tier-1 ("high
                                            # liquidity") until 26AUG17. Poly
                                            # titles "... - Circuito Desafiante
                                            # Playoffs". Full string — safe next
                                            # to tier-1 "CBLOL" (no overlap).
    "EQUAL ESPORTS CUP",                    # academy-team cup (per user 26AUG16,
                                            # RBG2H: G2 HEL vs Vitality Rising
                                            # Bees). Poly title "... - Equal
                                            # eSports Cup Playoffs".
    "NORTH AMERICAN CHALLENGERS LEAGUE",    # NACL
    " LIT ",                                # LIT ERL (space-padded standalone code)
    " TCL ",                                # Turkish league (per user 2026-08-11,
                                            # SUSHA). Poly titles "... - TCL
                                            # Play-Ins", rules "wins the TCL 2026:"
                                            # — space-padded like LIT so the code
                                            # never fires inside a team name.
)
import prob_reconcile  # noqa: E402  (continuation backsolve)


# ── G2/G3/G4 prob-refresh scope exclusion (2026-08-06, operator) ──
# LoL and Dota are excluded from _refresh_g2_probabilities — BOTH tiers (tier is
# not in the ticker, so the prefix covers T1 and T2 alike).
#
# Why: the refresh overwrites p2 (and p3/p4 on BO5) with the RAW live G(N) mid.
# That is exactly the per-leg pricing that produced the pregame lean — each leg
# taken from its own book with nothing forcing the set to agree. Once
# `prob_reconcile.solve_continuation` backsolves a reconciled continuation
# probability at populate time, a later refresh would silently overwrite it and
# reintroduce the lean mid-series. See PREGAME_LEAN_SCOPE.md.
#
# CS2 / VAL / COD keep refreshing — they were never in the reconcile scope.
# Re-enable a sport here only once the refresh itself is lean-aware (i.e. it
# re-solves rather than writing a raw mid).
G2_REFRESH_EXCLUDED_PREFIXES = ("KXLOLGAME", "KXDOTA2GAME")

# Scope of the pregame/intermission anchor re-solve. LoL ONLY, deliberately:
# the start-time source is Riot's lolesports feed, which carries no Dota2 at all
# (verified 26AUG11 — zero Dota leagues in getLeagues). Including Dota here just
# means every Dota event fails its join forever and retries every 15 minutes.
# Dota needs a different start-time source before it can be added.
PREGAME_ANCHOR_PREFIXES = ("KXLOLGAME",)
# Intermission-ONLY sports (26AUG12, operator — after HLEBRO's 0.594 write
# landed within 0.5c of the market's decider): no per-game feed exists, so no
# kickoff phase and no pregame sizing signal — just the book-pinned map-1
# detector plus TIMEOUT adoption, with per-sport (timeout_min, adopt_min)
# windows.
#
# VAL timing, MEASURED 26AUG11 (probe, VCT China, 3 transitions): map-end ->
# next-map skeleton (agent select, bo3 publishes round-1 BUY_TIME + map_name)
# at +9m07s/+9m08s/+9m50s; pistol live at +11m00s/+11m01s/+11m49s.
#
# HARDCODED +5min (operator, 26AUG12): adopt 5 minutes after we first SEE the
# map-1 pin (decided_at, 0..1 populate cycle after the true end), window = the
# same 5 minutes. Executes at +5..+10 real time — pre-pistol — because a
# skeleton-triggered exit can't be caught reliably at populate's ~3.3min
# sampling (skeleton leads the pistol by only 2-3.5min). Same numbers for CS2
# until its breaks are measured.
# Dota added 26AUG12 (operator): THE INTERNATIONAL ONLY (poly_title gate in
# the assembly loop below), tier-1 only, bo3-backed g1-end detection via
# dota_game_state layered over the same book-pin/settlement/timeout path.
# Measured Dota inter-game breaks (14 transitions, tier-a, 26AUG06-08 capture):
# g1-end -> g2 first live data 21.5-29.5 min, median ~24.7. Dota runs (8,8)
# not the VAL/CS2 (5,5) (operator, 26AUG13): the book-pin "decided" fires
# EARLY in Dota — 99/0 books while the stomp is still being played out — so
# decided_at precedes the true throne fall and the earliest post-pin snapshots
# can still be mid-game. Adopt at +8 with the median over the full 8 min so
# the window reaches into genuine intermission; still pre-draft (draft starts
# ~true-end +6..13 min, and true end >= pin).
INTERMISSION_ONLY_PREFIXES = ("KXVALORANTGAME", "KXCS2GAME", "KXDOTA2GAME")
INTERMISSION_WINDOWS = {"KXVALORANTGAME": (5.0, 5.0), "KXCS2GAME": (5.0, 5.0),
                        "KXDOTA2GAME": (8.0, 8.0)}
# Per-sport arming flags — Dota deliberately NOT under the VAL/CS2 flag so
# either can be killed alone.
INTERMISSION_ONLY_FLAGS = {"KXDOTA2GAME": "enable_dota_intermission.flag"}
_INTERMISSION_DEFAULT_FLAG = "enable_valcs_intermission.flag"
# Poly-title markers that classify a VAL event tier-2 for intermission scoping
# (thin books — same principle as the LoL tier-2 exclusion).
_VAL_T2_TITLE_MARKERS = ("GAME CHANGERS", "CHALLENGERS", "VCL")


_league_blacklist_logged: set = set()
_lol_tier_cache: dict[str, str] = {}  # event_base -> "tier1" | "tier2"


def classify_lol_tier(event_base: str, poly_mappings: dict, s_markets=None) -> str:
    """Return "tier1" (default) or "tier2" for a LoL event_base.

    Tier-2 iff a TIER2_LOL_KEYWORDS regional-league code (LJL/LRN/LRS) appears in
    EITHER Kalshi's rules_primary (authoritative — present even with NO Poly link)
    OR the cached Poly title. No signal → "tier1" (per user: assume tier-1 unless a
    known tier-2 league). Accepts KXLOLGAME-... and KXLOLMAP-...-N; normalizes to
    the KXLOLGAME key.

    rules_primary is checked FIRST because minor regional leagues frequently have
    no Poly mapping (e.g. KXLOLGAME-26JUL121500FUESDM = "LRN 2026: Fuego vs. SDM
    Tigres", which had an empty poly_parsed_markets row). The old Poly-title-only
    path silently left those at tier-1 / full size. Mirrors classify_val_tier.
    """
    if not event_base or "KXLOL" not in event_base:
        return "tier1"
    key = event_base.replace("KXLOLMAP", "KXLOLGAME")
    parts = key.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        key = parts[0]
    if key in _lol_tier_cache:
        return _lol_tier_cache[key]
    # 1) Authoritative: Kalshi rules_primary (static per event, no Poly link
    #    needed). Match case-insensitively — short league codes (LRS/LRN/LJL)
    #    appear uppercase, but multi-word league names (e.g. "Road Of Legends")
    #    are title-case in the rules text, so an uppercase-only match would miss
    #    them. Keywords are specific enough (3-letter codes or full league names)
    #    that false substring hits inside team names are not a real risk.
    if s_markets:
        for m in s_markets:
            rules = (m.get("rules_primary", "") or "").upper()
            if any(k in rules for k in TIER2_LOL_KEYWORDS):
                _lol_tier_cache[key] = "tier2"
                return "tier2"
    # 2) Fallback: cached Poly title.
    m = poly_mappings.get(key)
    has_poly = bool(m and m.get("poly_url"))
    if has_poly:
        title_upper = (m.get("poly_title") or "").upper()
        tier = "tier2" if any(k in title_upper for k in TIER2_LOL_KEYWORDS) else "tier1"
        # Only cache once a Poly mapping exists, so a not-yet-mapped event
        # stays re-checkable and can flip to tier2 on a later cycle.
        _lol_tier_cache[key] = tier
        return tier
    # No rules_primary keyword and no Poly link → assume tier-1, but leave
    # uncached so a later cycle with rules_primary/Poly can still flip it.
    return "tier1"


# Dota is tier-2 by DEFAULT (smaller-prize-pool tournaments where the top-10
# teams aren't competing → 1/4 size via base template rows). Only premier-circuit
# events whose Poly title matches TIER1_DOTA_KEYWORDS route to full-size `_T1`
# rows. Same default direction as CS2 (default tier2, whitelist → tier1); opposite
# of LoL. No Poly link → tier-2 (conservative 1/4 until confirmed premier).
TIER1_DOTA_KEYWORDS = ["THE INTERNATIONAL", "BLAST", "ESPORTS WORLD CUP",
                       "ESL ONE", "DREAMLEAGUE", "PGL", "RIYADH MASTERS",
                       "FISSURE",
                       # Games of the Future — ~$1M prize pool, operator call
                       # 2026-07-31. Premier-scale field (Xtreme Gaming, LGD,
                       # Vici, Zero Tenacity). Poly titles read "... - Games of
                       # the Future Group A/B/C/D", so the phrase alone covers
                       # the group stage and any later playoff bracket.
                       "GAMES OF THE FUTURE",
                       # 1win Essence — operator call 2026-08-03. Premier field
                       # (LGD, OG, Falcons, BetBoom, Liquid, Vici). "1WIN
                       # ESSENCE" phrase, NOT bare "1WIN": 1win is also a team
                       # name (e.g. CS2 "1WIN vs ex-RUBY") and appears in other
                       # tournaments' matchup titles.
                       "1WIN ESSENCE"]
_dota_tier_cache: dict[str, str] = {}  # event_base -> "tier1" | "tier2"


def classify_dota_tier(event_base: str, poly_mappings: dict) -> str:
    """Return "tier1" or "tier2" for a Dota2 event_base; "tier2" for non-Dota.

    Tier 1 iff there is a Poly mapping AND the cached poly_title contains a
    TIER1_DOTA_KEYWORDS entry (case-insensitive). No Poly link → guaranteed
    tier 2. Accepts KXDOTA2GAME-... and KXDOTA2MAP-...-N; normalizes to the
    KXDOTA2GAME-... key used in poly_parsed_markets.csv. Mirrors
    classify_cs_tier's cache-once-mapped semantics.
    """
    if not event_base or "KXDOTA2" not in event_base:
        return "tier2"
    key = event_base.replace("KXDOTA2MAP", "KXDOTA2GAME")
    parts = key.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        key = parts[0]
    if key in _dota_tier_cache:
        return _dota_tier_cache[key]
    m = poly_mappings.get(key)
    tier = "tier2"
    has_poly = bool(m and m.get("poly_url"))
    if has_poly:
        title_upper = (m.get("poly_title") or "").upper()
        if any(k in title_upper for k in TIER1_DOTA_KEYWORDS):
            tier = "tier1"
        # Only cache once a Poly mapping exists so a not-yet-mapped event stays
        # re-checkable and can promote to tier1 on a later cycle.
        _dota_tier_cache[key] = tier
    return tier


_val_tier_cache: dict[str, str] = {}  # event_base -> "vct" | "vcl"

# p2/p3/p4 refresh baseline cache — anchors the FIRST p_a observed for each
# (event_base, leg_num) since process start. Refresh rejects any new_p_a that
# deviates from this baseline by > P_BASELINE_MAX_DEV (10pp). Added 2026-05-29
# after SHKALL p2 jumped 0.495 → 0.735 in one cycle and triggered a 1500-lot
# arber+momentum cascade off a likely-bad Poly G2 mid.
_p_baseline_cache: dict[tuple[str, int], float] = {}
P_BASELINE_MAX_DEV = 0.15

# 2026-07-20 — velocity guard, and why the displacement guard alone was wrong.
#
# SHKALL was a VELOCITY failure: p2 jumped 0.495 → 0.735 in ONE cycle. The guard
# built to catch it was a DISPLACEMENT limit against an anchor fixed at process
# start. Those measure different things, and using one for the other freezes the
# value permanently: as a game legitimately travels away from its pregame prior,
# every correct update looks like an outlier and is rejected forever.
#
# Measured on PCKCPLYNX 26JUL20, p_a observed every ~3min:
#   0.350 0.360 0.350 0.350 0.365 0.400 [28min blocked] 0.500 0.515 0.565
#   0.560 0.565 0.560 0.585 0.590 0.580
# Pure drift — worst true step ~3.5pp/cycle; the lone +10pp step spans ~9 missed
# cycles (~1.1pp/cycle). Total displacement reached 24pp, so the 10pp guard
# rejected every correct update for 90 minutes while p2 sat frozen at 0.400.
# That frozen p2 understated the hedge ~10c and drove a 7,500-lot accumulation.
#
# So: bound the STEP, ratchet the anchor. A 24pp one-cycle jump is still
# rejected; a 3pp-per-cycle drift accumulating past 15pp is allowed through.
# The step allowance scales with elapsed time so a spread-blocked gap doesn't
# re-freeze the leg the moment the book tightens again.
_p_last_obs_cache: dict[tuple[str, int], tuple[float, float]] = {}  # key -> (p, ts)
P_STEP_MAX_PER_CYCLE = 0.08   # max |Δp| per nominal refresh cycle
P_STEP_NOMINAL_CYCLE_SEC = 190.0
P_STEP_MAX_ABS = 0.30         # ceiling on the gap-scaled allowance


def check_p_sanity(key, new_p: float, cur_p: float, now: float):
    """Velocity + ratcheting-displacement sanity check for a refreshed prob.

    Returns (ok, skip_reason). Mutates the two module caches on the accept
    path. Module-level (not nested in _refresh_g2_probabilities) so tests
    exercise the real code rather than a reimplementation of it.

      key    : (event_base, leg_num)
      new_p  : candidate probability for team A
      cur_p  : current CSV value, used to seed the anchor on first sight
      now    : epoch seconds (injected so tests can control gaps)
    """
    baseline = _p_baseline_cache.setdefault(key, cur_p)

    # Velocity (primary): bound the per-cycle STEP, scaled by elapsed time so a
    # spread-blocked gap doesn't re-freeze the leg when the book tightens again.
    prev = _p_last_obs_cache.get(key)
    if prev is not None:
        prev_p, prev_ts = prev
        cycles = max(1.0, (now - prev_ts) / P_STEP_NOMINAL_CYCLE_SEC)
        allowed = min(P_STEP_MAX_ABS, P_STEP_MAX_PER_CYCLE * cycles)
        if abs(new_p - prev_p) > allowed:
            return False, (f"step_too_large(new={new_p:.3f}, prev={prev_p:.3f}, "
                           f"step={abs(new_p-prev_p):.3f}, "
                           f"allowed={allowed:.3f}, gap={now-prev_ts:.0f}s)")
    # Record every value that CLEARS velocity, accepted downstream or not —
    # otherwise a run of displacement-rejected reads leaves prev stale and the
    # next legitimate step measures against an old value and looks enormous.
    _p_last_obs_cache[key] = (new_p, now)

    # Displacement (backstop): anchor RATCHETS on accept, so sustained drift
    # tracks instead of exhausting a fixed budget set at process start.
    if abs(new_p - baseline) > P_BASELINE_MAX_DEV:
        return False, (f"baseline_dev(new={new_p:.3f}, base={baseline:.3f}, "
                       f"dev={abs(new_p-baseline):.3f})")
    _p_baseline_cache[key] = new_p
    return True, None

def classify_val_tier(event_base: str, s_markets) -> str:
    """Return "vct" or "vcl" for a KXVALORANT event; "vcl" for non-Valorant
    (callers gate on KXVALORANT prefix before applying the filter, so the
    non-Valorant return is unused).

    Keys off Kalshi's rules_primary. Sizing tier:
      - "vct"  if is_vct_event() (real VCT, excluding Challengers / Game Changers)
      - "vct"  if rules_primary mentions "Esports World Cup" (EWC qualifiers —
               empirically perform well, comparable to VCT)
      - "vct"  if rules_primary mentions "Evolution" (Chinese VCT Evolution
               series — added 2026-05-25 after observing tier-1-like behavior:
               high volume and maps leading the series)
      - "vcl"  otherwise (Challengers, Game Changers, or explicit non-VCT)

    Caches results once we have a definitive answer from rules_primary so a
    transient series API failure (s_markets=None) doesn't downgrade a known
    VCT/EWC event to VCL sizing. Cache mirrors classify_cs_tier's pattern.
    Cause-of-bug 2026-05-21: SENEG (EWC Americas Qualifier) flipped 12500→4000
    when the populator hit a 5s API timeout, triggering hot-reload.
    """
    # Fast path: return cached classification if we've seen this event
    # definitively before. Survives transient API failures.
    if event_base in _val_tier_cache:
        return _val_tier_cache[event_base]

    if is_vct_event(s_markets):
        _val_tier_cache[event_base] = "vct"
        return "vct"
    if s_markets:
        for m in s_markets:
            rules = m.get("rules_primary", "") or ""
            # EWC / Evolution ⇒ tier-1. (EWC EMEA Qualifier is six-figure-volume
            # tier-1, so this must be checked before any Europe/EMEA heuristic —
            # the old VCL-Europe blacklist, removed 2026-07-05, used to mis-skip
            # these until it was reordered on 2026-05-30.)
            if any(kw in rules for kw in ("Esports World Cup", "Evolution")):
                _val_tier_cache[event_base] = "vct"
                return "vct"
        # Reached here = we have a real rules_primary block but no VCT/EWC/Evolution
        # signal → definitive VCL (Challengers, Game Changers, or other).
        _val_tier_cache[event_base] = "vcl"
        return "vcl"
    # s_markets is None or empty: API failure on this cycle. Don't cache —
    # let next cycle re-derive. Default to "vcl" (current behavior) but the
    # cache check above already handles the common case where we previously
    # determined VCT/EWC.
    return "vcl"


# Kalshi-sub-title → Polymarket-outcome team name map. Loaded once,
# mtime-reloaded by _maybe_reload_team_map below. Primary lookup in
# _align_via_poly_parsed_markets — overrides substring matching so the same
# Kalshi short form (e.g. "DRX") doesn't fall through to ambiguous fuzzy
# matching when a multi-letter alignment exists in the CSV.
#
# SPORT-SCOPED PINS (2026-08-02, DRXVAR incident): keys are
# (series_scope, sub_title) where series_scope is the Kalshi series prefix
# ("KXLOLGAME") or "" for a global pin. The global pin `DRX → Kiwoom DRX`
# (correct for the LCK org on Poly) poisoned every VALORANT DRX event: the
# alignment translated VAL "DRX" to "Kiwoom DRX", found no such Poly outcome,
# and correctly refused to guess — so the leg silently never traded. A pin
# that is only true for one sport MUST carry its series_scope. Lookups try
# the event's own scope first, then global — via _team_map_lookup, never
# _team_map.get directly.
_team_map: dict[tuple, str] = {}
_team_map_mtime: float = 0.0

def _maybe_reload_team_map(path: str = "kalshi_poly_team_map.csv") -> None:
    """Reload kalshi_poly_team_map.csv on mtime change. CSV columns:
    `kalshi_sub_title,polymarket_outcome[,series_scope]` — series_scope is
    optional ("" / absent = global pin; else a Kalshi series prefix like
    KXLOLGAME). Manual edits take effect on the next discovery cycle — no
    process restart needed."""
    global _team_map, _team_map_mtime
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return  # file missing — leave map empty, callers fall through
    if mt == _team_map_mtime and _team_map:
        return
    new_map: dict[tuple, str] = {}
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (row.get("kalshi_sub_title") or "").strip()
                p = (row.get("polymarket_outcome") or "").strip()
                scope = (row.get("series_scope") or "").strip().upper()
                if k and p:
                    new_map[(scope, k)] = p
        _team_map = new_map
        _team_map_mtime = mt
        print(f"[TEAM MAP] Loaded {len(new_map)} kalshi→poly pairs from {path}")
    except Exception as e:
        print(f"[TEAM MAP] Failed to load {path}: {e}")


def _team_map_lookup(sub_title: str, kalshi_event: str = ""):
    """Scoped-first pin lookup: (event's series prefix, sub) then ("", sub).
    Returns the pinned Poly outcome string or None."""
    sub = sub_title.strip()
    prefix = kalshi_event.split("-", 1)[0].strip().upper() if kalshi_event else ""
    if prefix:
        hit = _team_map.get((prefix, sub))
        if hit:
            return hit
    return _team_map.get(("", sub))


def _load_poly_mappings(path: str = "poly_parsed_markets.csv") -> dict:
    """Load pre-verified Kalshi→Polymarket mappings.
    Returns dict: kalshi_event_base -> {"poly_slug": str, "poly_url": str, "poly_title": str}
    Empty poly_url means confirmed no Poly match — skip Poly entirely.
    """
    mappings = {}
    if not os.path.exists(path):
        return mappings
    try:
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if not row:
                    continue
                kalshi_event = row[0].strip()
                poly_url = row[2].strip() if len(row) > 2 else ""
                poly_title = row[3].strip() if len(row) > 3 else ""
                # team_alignments column: "Outcome A Name; Outcome B Name", in
                # the same order as the Polymarket outcomes for the event. Used
                # as the deterministic source of truth for Kalshi team → Poly
                # outcome alignment, eliminating the fuzzy-matching guesswork
                # that's bitten us multiple times (FALTS leak 2026-05-17, FS/PR
                # alignment fail 2026-05-17). Falls back to the fuzzy matcher
                # only when this column is empty.
                team_alignments_raw = row[4].strip() if len(row) > 4 else ""
                team_alignments = [a.strip() for a in team_alignments_raw.split(";") if a.strip()]
                # Format column (7th): "BO3" | "BO5" | "" — written by
                # refresh_tomorrow_markets at discovery time. Source of truth
                # for is_bo5 routing across populate_configs; eliminates the
                # per-cycle m3-existence REST check that was unreliable
                # mid-series (Kalshi creates m3 lazily, and the REST
                # event_ticker filter hides settled m3 markets, both of which
                # mis-labeled BO5 as BO3 at different lifecycle stages).
                fmt = row[6].strip().upper() if len(row) > 6 else ""
                # `verified` column (col 7): operator-set "VERIFIED" tag means
                # the row is hand-curated and the Kalshi event is allowed
                # through discovery even with no per-map tickers. Used by
                # refresh_tomorrow_markets to bypass the BO1 reject path
                # when Kalshi neglects to list individual map markets.
                verified = (row[7].strip().upper() == "VERIFIED") if len(row) > 7 else False
                # Extract slug from URL (last path segment)
                # Handles both https://polymarket.com/event/{slug}
                # and https://polymarket.com/esports/.../slug
                poly_slug = ""
                if poly_url:
                    poly_slug = poly_url.rstrip("/").split("/")[-1]
                mappings[kalshi_event] = {
                    "poly_slug": poly_slug,
                    "poly_url": poly_url,
                    "poly_title": poly_title,
                    "team_alignments": team_alignments,
                    "format": fmt,
                    "verified": verified,
                }
        print(f"[POLY MAP] Loaded {len(mappings)} pre-verified mappings "
              f"({sum(1 for v in mappings.values() if v['poly_slug'])} with Poly, "
              f"{sum(1 for v in mappings.values() if not v['poly_slug'])} Kalshi-only)")
    except Exception as e:
        print(f"[POLY MAP] Failed to load {path}: {e}")
    return mappings


def _max_map_n(event_base: str, poly_mappings: dict) -> int:
    """Return the highest valid map index for this event.

      BO5 confirmed → 5
      BO3 confirmed → 2  (NOT 3 — Kalshi creates m3 lazily ONLY when the
                          match goes to a decider; for 2-0 results, the
                          m3 ticker is NEVER created. Querying for it
                          returns 0 markets and burns /markets? tokens
                          on every cycle for the rest of the event's life.)
      Truly unknown → 2  (default to BO3 — the common case; downstream
                          BO5-detection fallback in the per-prefix loop
                          will re-probe m3 once if other signals suggest BO5)

    Confirmation order:
      1. `format` column in poly_parsed_markets.csv  ("BO3"|"BO5")
      2. `poly_title`     substring scan              ("(BO3)" / "(BO5)")
         — refresh_tomorrow_markets sometimes leaves
         format empty for newly-discovered events, but
         the poly_title scrape carries the marker.
      3. Default → BO3 (max_n=2). Avoids 429 storms on
         non-existent m3 tickers for the common case.

    Verified against memory note `reference_bo3_bo5_detection.md`:
    "Kalshi m4 ticker presence is NOT reliable" — same lazy creation
    applies to m3 in BO3 events."""
    m = (poly_mappings or {}).get(event_base) or {}
    fmt = (m.get("format") or "").upper()
    if fmt == "BO5":
        return 5
    if fmt == "BO3":
        return 2
    title = (m.get("poly_title") or "").upper()
    if "BO5" in title and "BO3" not in title:
        return 5
    if "BO3" in title and "BO5" not in title:
        return 2
    return 2


def _fetch_poly_event_by_slug(slug: str) -> dict | None:
    """Fetch a specific Polymarket event by slug. Cached."""
    if slug in _poly_slug_cache:
        return _poly_slug_cache[slug]
    try:
        r = _get_with_retry("https://gamma-api.polymarket.com/events",
                         params={"slug": slug, "limit": 1}, timeout=10)
        if r.status_code == 200:
            events = r.json()
            if events:
                _poly_slug_cache[slug] = events[0]
                return events[0]
    except Exception:
        pass
    _poly_slug_cache[slug] = None
    return None


def _fetch_poly_events(sport: str) -> list:
    """Fetch active Polymarket events for a sport, cached per session.

    Paginates via offset: the gamma API caps each response at 100 regardless of
    the requested limit, so a single call silently truncates large tags (e.g.
    CS2 has 400+ events, with today's games past offset 300).
    """
    if sport in _poly_event_cache:
        return _poly_event_cache[sport]
    tag_id = POLY_SPORT_TAGS.get(sport)
    if not tag_id:
        _poly_event_cache[sport] = []
        return []
    PAGE = 100
    MAX_PAGES = 50
    all_events = []
    try:
        for page in range(MAX_PAGES):
            r = _get_with_retry("https://gamma-api.polymarket.com/events",
                             params={"tag_id": tag_id, "active": "true", "closed": "false",
                                     "limit": PAGE, "offset": page * PAGE}, timeout=10)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            all_events.extend(batch)
            if len(batch) < PAGE:
                break
    except Exception:
        pass
    _poly_event_cache[sport] = all_events
    return all_events


def _split_kalshi_teams(teams_str: str) -> list[str]:
    """Split concatenated Kalshi team string into individual teams.

    Pass 1: both halves are known TEAM_ALIASES (deterministic).
    Pass 2 (added 2026-06-02): known alias on one side + plausibly-shaped
      unknown abbrev (2-6 alphanumerics) on the other side. Lets the matcher's
      direct-slug check (which doesn't require aliases) catch HRTS-class
      teams where Kalshi+Poly share a short code (e.g., `lol-hrts-bar-...`)
      but the team isn't yet in TEAM_ALIASES. Sorted DESC by alias length so
      longest known prefix wins, avoiding mis-splits.
    """
    import re as _re
    teams_str_upper = teams_str.upper()
    sorted_aliases = sorted(TEAM_ALIASES.keys(), key=len, reverse=True)
    # Pass 1: both halves known
    for a1 in sorted_aliases:
        if teams_str_upper.startswith(a1):
            remainder = teams_str_upper[len(a1):]
            if remainder in TEAM_ALIASES:
                return [a1, remainder]
            for a2 in sorted_aliases:
                if remainder == a2:
                    return [a1, a2]
        if teams_str_upper.endswith(a1):
            prefix = teams_str_upper[:-len(a1)]
            if prefix in TEAM_ALIASES:
                return [prefix, a1]
    # Pass 2: known alias + plausibly-shaped unknown abbrev (2-6 alphanumerics)
    _plausible = _re.compile(r'^[A-Z0-9]{2,6}$')
    for a1 in sorted_aliases:
        if teams_str_upper.startswith(a1):
            remainder = teams_str_upper[len(a1):]
            if _plausible.match(remainder):
                return [a1, remainder]
        if teams_str_upper.endswith(a1):
            prefix = teams_str_upper[:-len(a1)]
            if _plausible.match(prefix):
                return [prefix, a1]
    return [teams_str_upper]


def _is_settled_event(event: dict) -> bool:
    """Check if a Polymarket event's main market has settled (prices at 0 or 1)."""
    event_slug = event.get("slug", "")
    for m in event.get("markets", []):
        if m.get("slug", "") == event_slug:
            prices_raw = m.get("outcomePrices", "")
            try:
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            except (json.JSONDecodeError, TypeError):
                prices = []
            if prices and len(prices) >= 2:
                try:
                    p0, p1 = float(prices[0]), float(prices[1])
                    if p0 <= 0.01 or p0 >= 0.99 or p1 <= 0.01 or p1 >= 0.99:
                        return True
                except (ValueError, TypeError):
                    pass
            break
    return False


def _extract_kalshi_date(kalshi_event: str) -> str | None:
    """Extract date as YYYY-MM-DD from Kalshi ticker like KXLOLGAME-26APR221700LYDSG."""
    from datetime import datetime
    tail = kalshi_event.split("-", 1)[1] if "-" in kalshi_event else ""
    if len(tail) < 7:
        return None
    date_str = tail[:7]  # e.g. "26APR22"
    try:
        dt = datetime.strptime(date_str, "%y%b%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


_SUBTITLE_STOPWORDS = {
    "esports", "esport", "gaming", "team", "club", "academy", "challengers",
    "the", "official", "vs",
    # 2026-07-27: league/roster suffixes carried by EVERY team in a league, so
    # they identify the league, never the team. "gc" (VCT Game Changers) let
    # Karmine Corp GC vs Barca eSports GC match HIMMERS vs ROSO GC. The
    # distinct-token rule in _match_poly_event Stage 2 is the general guard;
    # these are belt-and-braces so such tokens never score at all.
    "gc", "corp", "gc.",
}


def _fetch_kalshi_subtitles(event_base: str) -> list[str]:
    """Fetch yes_sub_title for each market in a Kalshi event via public REST.
    Used as a fallback signal when alias-based matching fails."""
    try:
        r = _get_with_retry(
            "https://external-api.kalshi.com/trade-api/v2/markets",
            params={"event_ticker": event_base, "limit": 10},
            headers={"Accept": "application/json"},
            timeout=4,
        )
        if r.status_code != 200:
            return []
        out = []
        for m in r.json().get("markets", []):
            sub = (m.get("yes_sub_title") or "").strip()
            if sub and sub not in out:
                out.append(sub)
        return out
    except Exception:
        return []


def _deaccent(s: str) -> str:
    """Strip diacritics so Poly<->Kalshi team matching survives accents:
    'KRÜ Esports' == 'KRU', 'Leviatán' == 'Leviatan', 'Movistar KOI Fénix'.
    Without this the [^a-z0-9] tokenizer/word-boundary regexes treat the accented
    char as a delimiter and truncate the token ('KRÜ' -> 'kr'), so a Kalshi
    sub-title never matches Poly's de-accented slug ('kru1') and the event goes
    unmapped (no hedge). Non-str/errors -> passthrough."""
    if not isinstance(s, str):
        return s or ""
    try:
        return "".join(c for c in unicodedata.normalize("NFKD", s)
                       if not unicodedata.combining(c))
    except (TypeError, ValueError):
        return s


def _tokenize_team_name(name: str) -> list[str]:
    """Lowercase alnum tokens of length >= 2, minus generic stop-words.
    Length-2 is required to catch short team codes like G2, T1, B8, C9, EG."""
    import re as _re
    name = _re.sub(r'[^a-z0-9 ]', ' ', _deaccent(name).lower())
    return [t for t in name.split() if len(t) >= 2 and t not in _SUBTITLE_STOPWORDS]


def _match_poly_event(kalshi_event: str, poly_events: list,
                      subtitles: list[str] | None = None) -> dict | None:
    """Find matching Polymarket event for a Kalshi ticker, skipping settled events.
    Verifies date match to avoid cross-day team collisions.

    Three-stage match:
      0. Deterministic via kalshi_poly_team_map.csv (NEW 2026-05-27): if BOTH
         Kalshi yes_sub_titles are pinned in the team map, find a Poly event
         whose Game 1 Winner / series winner outcomes contain EXACTLY both
         pinned poly_outcome names. Skips all fuzzy logic when the map covers
         both teams. Prevents the DRXFOX-class bug where main-team Kalshi
         events were silently mapped to academy Poly events because both
         shared a substring.
      1. Alias-based: tail-suffix → TEAM_ALIASES → poly title/slug.
         Fast, no API call. Defeats by missing aliases for new teams.
      2. Subtitle-token fallback: tokenize each Kalshi yes_sub_title (full team
         name) and require >=2 distinct subs to land a token in poly title/slug.
         Robust to new teams without alias updates.

    Subtitles can be passed in by callers that already have them; otherwise we
    fetch via public Kalshi REST (one extra request per unmatched event).
    """
    import re as _re
    tail = kalshi_event.split("-", 1)[1] if "-" in kalshi_event else ""
    teams_str = tail[11:] if len(tail) > 11 else ""
    if not teams_str:
        return None

    kalshi_date = _extract_kalshi_date(kalshi_event)  # e.g. "2026-04-22"

    # Academy/sub-team tier filter: second-tier rosters (LoL academies, VCT
    # Challengers, etc.) reliably have "youth"/"academy"/"junior"/"challenger(s)"
    # in their official names. If the Kalshi yes_sub_titles include any of
    # these keywords, only consider Poly events whose title also includes one
    # (and vice versa). Eliminates DRXFOX/T1KT-class main→academy mismatches.
    # CS2 is the only sport with rare exceptions like MOUZ NXT / Young Ninjas
    # where this filter could mis-reject, but user noted those don't surface
    # as confusion cases in practice — handle via team-map overrides if so.
    # Added 2026-05-27.
    _ACADEMY_KEYWORDS = ("youth", "academy", "junior", "challenger")
    # League-name whitelist: official tier-2 leagues whose NAME contains an
    # academy keyword but whose teams are MAIN rosters, not academy squads.
    # Strip these league names from the text BEFORE checking academy keywords
    # so the league name doesn't trigger a false "academy" flag.
    # Add to this list as new legitimate Challenger-named leagues appear.
    _LEAGUE_NAME_WHITELIST = (
        "lck challengers",
        "north american challengers league",
        "nacl",
    )
    def _is_academy(text):
        t = (text or "").lower()
        for league in _LEAGUE_NAME_WHITELIST:
            t = t.replace(league, "")
        return any(kw in t for kw in _ACADEMY_KEYWORDS)

    # Subtitles may be passed in; defer fetch until we need them (and only
    # once). When unavailable, we can't apply the tier filter — fall back
    # to existing behavior.
    _subtitles_cache = [subtitles] if subtitles is not None else [None]
    def _get_subs():
        if _subtitles_cache[0] is None:
            _subtitles_cache[0] = _fetch_kalshi_subtitles(kalshi_event)
        return _subtitles_cache[0] or []

    def _tier_ok(event):
        subs = _get_subs()
        if not subs:
            return True  # no info, don't filter
        # Kalshi's own name does not always carry the academy marker that Poly
        # carries: 'Los Heretics' -> 'Team Heretics Academy', 'NSEA' ->
        # 'Nongshim Esports Academy', 'T1A' -> 'T1 Academy'. Judged on the
        # Kalshi text alone those read as MAIN rosters, so this filter rejected
        # the one correct Poly event (which does say "Academy") and the event
        # went unmapped — KXLOLGAME-26AUG061100UBHRTS, 2026-08-06. Stage 0
        # normally rescues these, but only when BOTH sub_titles are pinned, so
        # one unpinned opponent was enough to lose the match.
        #
        # Resolve each sub_title through the team map first and judge tier on
        # the pinned Poly name, which is the side that carries the keyword.
        # Unpinned subs fall back to the Kalshi text (prior behaviour).
        _maybe_reload_team_map()
        resolved = [(_team_map_lookup(s, kalshi_event) or s) for s in subs]
        kalshi_acad = _is_academy(" ".join(resolved))
        poly_acad = _is_academy(event.get("title", "") + " " + event.get("slug", ""))
        return kalshi_acad == poly_acad

    def _alias_matches_title(alias, text):
        return bool(_re.search(r'\b' + _re.escape(alias) + r'\b', text))

    def _date_ok(event):
        if not kalshi_date:
            return True
        poly_slug = event.get("slug", "")
        slug_date_match = _re.search(r'(\d{4}-\d{2}-\d{2})$', poly_slug)
        if not slug_date_match:
            return True
        # ±1 day tolerance: Kalshi tickers are ET-dated, Poly slugs sometimes
        # use UTC or the league's own time zone, so late-evening ET games can
        # legitimately appear as next-day on Poly (e.g., G2 vs KC silently
        # dropped on 2026-05-23 because poly_slug said 2026-05-24).
        from datetime import datetime as _dt
        try:
            k = _dt.strptime(kalshi_date, "%Y-%m-%d")
            p = _dt.strptime(slug_date_match.group(1), "%Y-%m-%d")
        except ValueError:
            return False
        return abs((p - k).days) <= 1

    def _is_matchup(event):
        """Reject non-head-to-head Poly markets (placement / outright / futures).

        A real game/series event reads 'X vs Y' in its title
        ("LoL: Anyone's Legend vs Bilibili Gaming (BO3) - ..."). Season
        placement / outright markets ("Will BLG Place Higher than AL in LPL
        Split 3 2026") contain BOTH team names — so the fuzzy team-token stages
        happily match them — but never ' vs ', and their slugs carry no trailing
        YYYY-MM-DD so `_date_ok` fails open and can't reject them either. This is
        exactly how KXLOLGAME-26JUL260700BLGAL was mis-mapped to the BLG-vs-AL
        placement market (2026-07-26). Require the head-to-head structure the
        rest of the code already assumes (parse_poly_team_names keys on ' vs ').
        """
        title = (event.get("title") or "").lower()
        return " vs " in title or " vs. " in title

    # Stage 0: deterministic via kalshi_poly_team_map.csv. When both Kalshi
    # yes_sub_titles are pinned in the team map, only consider Poly events
    # whose Game 1 / series winner outcomes contain EXACTLY the two pinned
    # poly_outcome names. Returns the date-closest match. No fuzzy.
    if subtitles is None:
        subtitles = _fetch_kalshi_subtitles(kalshi_event)
    if subtitles and len(subtitles) >= 2:
        _maybe_reload_team_map()
        pinned_outs = []
        for sub in subtitles[:2]:
            p = _team_map_lookup(sub, kalshi_event)
            if p:
                pinned_outs.append(_deaccent(p.strip().lower()))
        if len(pinned_outs) == 2 and pinned_outs[0] != pinned_outs[1]:
            best = None  # (date_diff, event)
            for event in poly_events:
                if _is_settled_event(event) or not _date_ok(event):
                    continue
                event_slug = event.get("slug", "")
                # Look at Game 1 Winner OR series winner (slug == event_slug)
                for m in event.get("markets", []):
                    q = m.get("question", "")
                    m_slug = m.get("slug", "")
                    if not (m_slug == event_slug or "Map 1 Winner" in q or "Game 1 Winner" in q):
                        continue
                    outcomes_raw = m.get("outcomes", "")
                    try:
                        outs = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                    except (json.JSONDecodeError, TypeError):
                        outs = []
                    if not outs or len(outs) < 2:
                        continue
                    outs_lower = [_deaccent(o.strip().lower()) for o in outs[:2]]
                    if pinned_outs[0] in outs_lower and pinned_outs[1] in outs_lower:
                        # Compute date_diff for tie-break (closer = better)
                        date_diff = 0
                        slug_match = _re.search(r'(\d{4}-\d{2}-\d{2})$', event_slug)
                        if slug_match and kalshi_date:
                            from datetime import datetime as _dt
                            try:
                                pd = _dt.strptime(slug_match.group(1), "%Y-%m-%d")
                                kd = _dt.strptime(kalshi_date, "%Y-%m-%d")
                                date_diff = abs((pd - kd).days)
                            except ValueError:
                                pass
                        cand = (date_diff, event)
                        if best is None or cand[0] < best[0]:
                            best = cand
                        break  # don't double-count this event from a second market
            if best:
                return best[1]

    # Stage 1: alias-based matching
    teams = _split_kalshi_teams(teams_str)
    if len(teams) >= 2:
        for event in poly_events:
            if _is_settled_event(event) or not _date_ok(event) or not _tier_ok(event) or not _is_matchup(event):
                continue
            title = _deaccent(event.get("title", "").lower())
            slug = _deaccent(event.get("slug", "").lower())
            matched = 0
            for team_abbr in teams:
                aliases = TEAM_ALIASES.get(team_abbr, [team_abbr.lower()])
                if any(_alias_matches_title(alias, title) for alias in aliases):
                    matched += 1
            if matched >= 2:
                return event
            slug_matched = 0
            for team_abbr in teams:
                if _alias_matches_title(team_abbr.lower(), slug):
                    slug_matched += 1
                else:
                    aliases = TEAM_ALIASES.get(team_abbr, [])
                    if any(_alias_matches_title(a, slug) for a in aliases):
                        slug_matched += 1
            if slug_matched >= 2:
                return event

    # Stage 2: subtitle-token fallback
    if subtitles is None:
        subtitles = _fetch_kalshi_subtitles(kalshi_event)
    if not subtitles or len(subtitles) < 2:
        return None

    sub_tokens = [_tokenize_team_name(s) for s in subtitles]
    sub_tokens = [toks for toks in sub_tokens if toks]
    if len(sub_tokens) < 2:
        return None

    # Tiebreaker: (subs_matched, exact_slug_hits, total_hits).
    # exact_slug_hits counts how many Kalshi team suffixes appear as
    # word-bounded tokens in the Poly slug (e.g., `\bdrx\b` matches
    # `lol-fox1-drx-...` but NOT `lol-drxc-foxy-...` because `c` is a
    # word char so no boundary forms after `drx`). This is what stops
    # main-team Kalshi events (DRX/FOX, KT/T1) from being misrouted to
    # academy Poly events (DRXC/FOXY, KTC/T1A) when Stage 2's subtitle
    # tokens score the same (the DRXFOX bug, 2026-05-27). Added as a
    # band-aid until the kalshi_poly_team_map.csv covers every team.
    best = None  # (subs_matched, exact_slug_hits, total_hits, event)
    for event in poly_events:
        if _is_settled_event(event) or not _date_ok(event) or not _tier_ok(event) or not _is_matchup(event):
            continue
        title_text = _deaccent(event.get("title", "").lower())
        slug_text = _deaccent(event.get("slug", "").lower())
        text = title_text + " " + slug_text
        subs_matched = 0
        total_hits = 0
        matched_tokens = []          # the actual tokens each sub landed
        for toks in sub_tokens:
            hits = [t for t in toks
                    if _re.search(r'\b' + _re.escape(t) + r'\b', text)]
            if hits:
                subs_matched += 1
                total_hits += len(hits)
                matched_tokens.append(set(hits))

        # ── DISTINCT-TOKEN REQUIREMENT (2026-07-27) ───────────────────────
        # `subs_matched >= 2` counted SUBS that landed a token, not distinct
        # TEAMS identified. When both subtitles land the SAME generic token,
        # that is one shared league suffix — not two team identifications.
        #
        # KXVALORANTGAME-26JUL281100KCBAR (Karmine Corp GC vs Barca eSports GC)
        # was matched to `val-him-roso-2026-07-27` (HIMMERS vs ROSO GC, a
        # different match on a different day) purely on the token `gc`:
        #     'Karmine Corp GC'  tokens=[karmine, corp, gc] -> hits ['gc']
        #     'Barca eSports GC' tokens=[barca, gc]         -> hits ['gc']
        # Both subs "matched", so the event was written to
        # poly_parsed_markets.csv pointing at another event's Poly slug — the
        # only duplicate slug in the file. `gc` is the VCT Game Changers league
        # suffix carried by EVERY team in the league, so any two GC teams
        # matched any GC Poly event.
        #
        # Requiring the two subs to land DIFFERENT tokens generalises past `gc`
        # to any future league suffix (…GC, …Academy, …Youth) without needing to
        # enumerate them as stop-words. A genuine match identifies two teams and
        # therefore lands two different tokens.
        if subs_matched >= 2 and len(matched_tokens) >= 2:
            if not (matched_tokens[0] - matched_tokens[1]) or \
               not (matched_tokens[1] - matched_tokens[0]):
                # Both subs landed the same token set — one shared league
                # suffix, no distinguishing team evidence. Not a match.
                continue
        if subs_matched >= 2:
            exact_slug_hits = sum(
                1 for team in teams
                if _re.search(r'\b' + _re.escape(team.lower()) + r'\b', slug_text)
            )
            cand = (subs_matched, exact_slug_hits, total_hits, event)
            if best is None or cand[:3] > best[:3]:
                best = cand
    return best[3] if best else None


_poly_alignment_cache: dict | None = None  # lazy-loaded poly_mappings
_poly_alignment_mtime: float = 0.0


def _maybe_reload_poly_alignment_cache(path: str = "poly_parsed_markets.csv") -> None:
    """Reload poly_parsed_markets.csv on mtime change. Without this, events
    added or edited AFTER process startup (e.g. refresh_tomorrow_markets
    daemon's 30-min rewrite, or manual user edits) silently miss the
    alignment cache, causing _align_via_poly_parsed_markets to return -1 →
    poly_price_feed disables Poly and falls back to Kalshi for that event.
    Symptom: silent ~12-sec Kalshi-only window after a fresh row appears
    (LOSLLL incident 2026-06-11 19:20:56 → 19:21:08).

    Mirrors _maybe_reload_team_map: mtime-watched, reload on change, no
    process restart required.
    """
    global _poly_alignment_cache, _poly_alignment_mtime
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return  # file missing — leave cache as-is, callers handle the miss
    if mt == _poly_alignment_mtime and _poly_alignment_cache is not None:
        return
    try:
        _poly_alignment_cache = _load_poly_mappings(path)
        _poly_alignment_mtime = mt
    except Exception as e:
        print(f"[POLY ALIGN CACHE] Failed to reload {path}: {e}")


def _align_via_poly_parsed_markets(kalshi_full_name: str, event_base: str,
                                   poly_outcomes: list[str]) -> int:
    """Deterministic alignment using two source-of-truth CSVs.

    1) `kalshi_poly_team_map.csv` — global Kalshi-sub → Poly-outcome map.
       If `kalshi_full_name` is in the map, the mapped Poly-outcome string is
       used directly (must appear exactly in `poly_outcomes`).
    2) `poly_parsed_markets.csv` `team_alignments` column — per-event list of
       Poly outcomes in some order. Used when the global map doesn't have an
       entry yet (self-bootstrapping fallback).

    For (2), each of the two lookups (kalshi sub-title → alignment slot, and
    alignment string → poly outcome) accepts:
      (a) exact / spaceless-exact match, OR
      (b) unique-containment — `needle` is substring of EXACTLY ONE candidate.
    Multi-match or no-match returns -1. Never a fuzzy ranking guess.

    Returns -1 on any miss — caller must skip the leg, never guess.
    """
    global _poly_alignment_cache
    _maybe_reload_team_map()
    _maybe_reload_poly_alignment_cache()

    def _unique_match(needle: str, candidates: list[str]) -> int:
        n = _deaccent(needle.strip().lower())
        n_ns = n.replace(" ", "").replace("_", "")
        if not n:
            return -1
        for i, c in enumerate(candidates):
            cl = _deaccent(c.strip().lower())
            cl_ns = cl.replace(" ", "").replace("_", "")
            if cl == n or cl_ns == n_ns:
                return i
        matches = []
        for i, c in enumerate(candidates):
            cl = _deaccent(c.strip().lower())
            cl_ns = cl.replace(" ", "").replace("_", "")
            if n in cl or n_ns in cl_ns:
                matches.append(i)
        return matches[0] if len(matches) == 1 else -1

    # Path 1: explicit team map (kalshi_poly_team_map.csv), event-scope first.
    pinned = _team_map_lookup(kalshi_full_name, event_base)
    if pinned:
        # The map says Kalshi "X" pairs with Poly "Y" — find Y exactly in poly_outcomes.
        idx = _unique_match(pinned, poly_outcomes)
        if idx >= 0:
            return idx
        # The pin says Y but Y isn't in this event's outcomes — Poly drift.
        # Don't fall through (would be silent guessing); return -1.
        return -1

    # Path 2: per-event team_alignments column (fallback when team map lacks the entry).
    # _maybe_reload_poly_alignment_cache (called above) keeps this fresh on mtime change.
    if _poly_alignment_cache is None:
        return -1
    mapping = _poly_alignment_cache.get(event_base)
    if not mapping:
        return -1
    alignments = mapping.get("team_alignments") or []
    if len(alignments) < 2 or len(poly_outcomes) < 2:
        return -1
    aligned_slot = _unique_match(kalshi_full_name, alignments)
    if aligned_slot == -1:
        return -1
    return _unique_match(alignments[aligned_slot], poly_outcomes)


def _align_team_to_poly_outcome(kalshi_full_name: str, poly_outcomes: list[str],
                                event_base: str = "") -> int:
    """Align a Kalshi team to a Polymarket outcome.

    Source of truth: poly_parsed_markets.csv `team_alignments` column. The
    deterministic lookup via _align_via_poly_parsed_markets is the ONLY path.
    Returns -1 on any miss — caller must skip the leg (never guess).

    Fuzzy fallbacks (containment / acronym / word-overlap) were removed
    2026-05-27 after the DRXFOX inversion: a stale poly_parsed_markets entry
    pointed today's main-team Kalshi event at tomorrow's academy Poly event;
    the fuzzy fallback silently aligned around the bad mapping and produced
    inverted hedge prices for ~hours. Per the silent-fallbacks memory note,
    a bad CSV must fail loud, not get rescued by heuristics.
    """
    if not event_base:
        return -1
    return _align_via_poly_parsed_markets(kalshi_full_name, event_base, poly_outcomes)


def _extract_team_names(kalshi_markets: list) -> dict:
    """Extract suffix → full team name from Kalshi market yes_sub_title fields.
    Returns e.g. {"C9": "Cloud9", "TL": "Team Liquid"}
    """
    names = {}
    for m in kalshi_markets:
        ticker = m.get("ticker", "")
        suffix = ticker.split("-")[-1]
        sub_title = m.get("yes_sub_title", "")
        if suffix and sub_title:
            names[suffix] = sub_title
    return names


def _get_poly_winner_markets(event: dict) -> dict:
    """Extract series + game1..game5 winner markets from a Polymarket event.

    BO5 events on Poly expose Game/Map 4 and (less consistently) Game/Map 5
    winner markets in addition to the standard 1-3. We capture all five so
    the BO5 hedge math can use market-priced future-game probabilities
    instead of the (p1+p2+p3)/3 stand-in.
    """
    result = {}
    event_slug = event.get("slug", "")
    for m in event.get("markets", []):
        q = m.get("question", "")
        slug = m.get("slug", "")
        tokens_raw = m.get("clobTokenIds", "")
        try:
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) and tokens_raw else tokens_raw
        except (json.JSONDecodeError, TypeError):
            tokens = []
        prices_raw = m.get("outcomePrices", "")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) and prices_raw else prices_raw
        except (json.JSONDecodeError, TypeError):
            prices = []
        if not tokens or len(tokens) < 2:
            continue
        outcomes_raw = m.get("outcomes", "")
        try:
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) and outcomes_raw else outcomes_raw
        except (json.JSONDecodeError, TypeError):
            outcomes = []
        entry = {"p": prices, "outcomes": outcomes, "tokens": tokens}
        matched = False
        for gi in range(1, 6):
            if f"Game {gi} Winner" in q or f"Map {gi} Winner" in q:
                result[f"game{gi}"] = entry
                matched = True
                break
        if matched:
            continue
        if slug == event_slug and "handicap" not in slug and "total" not in slug:
            result["series"] = entry
    return result


def _get_poly_spread(token_id: str) -> float | None:
    """Get the spread for a Polymarket token in cents."""
    try:
        r = _get_with_retry(f"https://clob.polymarket.com/spread?token_id={token_id}", timeout=5)
        if r.status_code == 200:
            v = r.json().get("spread")
            if v:
                return float(v) * 100
    except Exception:
        pass
    return None


def _align_poly_to_kalshi(k_team: str, poly_market: dict, kalshi_full_name: str = "",
                          event_base: str = "") -> float | None:
    """
    Get Polymarket mid aligned to Kalshi's YES team perspective.
    Uses kalshi_full_name (from yes_sub_title) for reliable matching.
    Returns probability as 0.0-1.0 or None.
    """
    poly_outcomes = poly_market.get("outcomes", [])
    poly_prices = poly_market.get("p", [])
    if not poly_outcomes or len(poly_outcomes) < 2 or not poly_prices or len(poly_prices) < 2:
        return None

    # Use full team name from Kalshi if available
    aligned_idx = -1
    if kalshi_full_name:
        aligned_idx = _align_team_to_poly_outcome(kalshi_full_name, poly_outcomes, event_base=event_base)

    if aligned_idx == -1:
        return None  # Can't align — don't guess

    try:
        return float(poly_prices[aligned_idx])
    except (ValueError, TypeError):
        return None


def _load_force_include_patterns(path: str = "force_include_tickers.txt") -> set:
    """Read substring patterns that bypass refresh_tomorrow's tier-3 drop AND
    force data_source=kalshi on the emitted template rows.

    Format per line:
      <pattern>           # substring matched against event_base
      # comment           # ignored

    Returns a set of patterns. Empty set when the file doesn't exist.
    """
    out: set = set()
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                # Take first whitespace-separated token (allows future extension).
                out.add(ln.split()[0])
    except Exception:
        pass
    return out


def _clock_skew_seconds(timeout: float = 3.0):
    """|local - network| time in seconds via an HTTPS Date header (kalshi.com,
    fallback polymarket). Returns None if unverifiable this cycle.

    Added 26AUG15 after the FNCSK/SHFTG2 miss: the system clock ran ~10h slow,
    so the pregame time buffer classified in-progress LEC games as many hours
    away and never admitted them. The buffer must never trust an unverified
    local clock — see the fail-open at the buffer check."""
    import urllib.request
    import urllib.error
    import email.utils
    import time as _t
    for url in ("https://gamma-api.polymarket.com", "https://kalshi.com"):
        hdr = None
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "clock-guard/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                hdr = r.headers.get("Date")
        except urllib.error.HTTPError as e:
            # 405/429/etc still carry a trustworthy Date header
            hdr = e.headers.get("Date") if e.headers else None
        except Exception:
            continue
        if hdr:
            try:
                return _t.time() - email.utils.parsedate_to_datetime(hdr).timestamp()
            except Exception:
                continue
    return None


def _load_force_start_patterns(path: str = "force_start_tickers.txt") -> list:
    """Read substring patterns that bypass time gates.

    Format per line:
      <pattern>                       # always force-start (legacy)
      <pattern> @ 2026-05-04T03:00    # force-start only on/after this time (ET)

    Only patterns whose activation time has been reached are returned. Lets you
    schedule overrides ahead of time (e.g. before bed) without having probabilities
    captured at stale prices many hours pre-match.
    """
    if not os.path.exists(path):
        return []
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo as _ZI
        et = _ZI("America/New_York")
    except Exception:
        et = None
    now = _dt.now(et) if et else _dt.now()
    out = []
    try:
        with open(path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "@" in ln:
                    pattern, _, ts_str = ln.partition("@")
                    pattern = pattern.strip()
                    ts_str = ts_str.strip()
                    if not pattern:
                        continue
                    try:
                        d = _dt.fromisoformat(ts_str)
                        if d.tzinfo is None and et is not None:
                            d = d.replace(tzinfo=et)
                    except Exception:
                        continue  # bad datetime → fail safe (don't activate)
                    if now >= d:
                        out.append(pattern)
                else:
                    out.append(ln)
    except Exception:
        pass
    return out


def _probe_settled_series_events(rows: list[list[str]],
                                  min_age_minutes: float = 30.0) -> set[str]:
    """Identify event_bases whose series (KX*GAME) markets are all settled.

    Methodology:
      1. From the candidate template rows, extract every distinct series ticker
         (any ticker whose prefix contains "GAME", excluding "*MAP*"). Series
         markets carry the canonical "is this event live?" signal — when both
         team sides return finalized/empty on a WS subscribe, the event is
         resolved and no map row should remain in the template either.
      2. WS-probe those tickers via ws_market_status.probe_markets (one-shot
         orderbook_snapshot subscribe). For each ticker, classify as:
            - SETTLED:    active=False (code:28 "Markets not found"), OR
                          active=True with zero levels on BOTH yes & no books
                          (Kalshi keeps finalized markets listed with empty
                          books for a while after settlement before removing
                          them from the WS channel entirely).
            - LIVE:       active=True with at least one level on either side.
            - UNKNOWN:    active=None (timeout) — KEEP the event (fail-open).
      3. Group by event_base. An event is "settled" iff EVERY series ticker
         for it is SETTLED. Mixed or any-UNKNOWN → keep the event.
      4. Temporal guard: only drop events whose scheduled start (parsed from
         ticker via parse_scheduled_start) was ≥ `min_age_minutes` ago. This
         protects pre-game events (which legitimately have empty books until
         book-keepers arrive) and short-lived intermissions. Override via the
         `min_age_minutes=0` argument or by force-keep file at caller level.

    Returns:
      Set of event_bases (e.g. {"KXVALORANTGAME-26JUN261900PGSD1", ...}) that
      can be safely dropped from template_quoter_config.csv.
    """
    series_tickers: set[str] = set()
    for r in rows:
        if len(r) < 2 or not r[1]:
            continue
        t = r[1]
        u = t.upper()
        if "MAP" in u:
            continue  # MAP-level row — series row will carry the signal
        if "GAME" not in u:
            continue
        series_tickers.add(t)
    if not series_tickers:
        return set()

    try:
        from ws_market_status import probe_markets as _ws_probe
        statuses = _ws_probe(sorted(series_tickers), timeout_s=4.0)
    except Exception as e:
        print(f"[SETTLED-SCANNER] WS probe failed ({e}); fail-open, no events dropped")
        return set()

    # Per-ticker classification
    def _is_settled(st):
        """True = settled, False = live, None = unknown (treat as live)."""
        if st is None:
            return None
        if st.active is False:
            return True
        if st.active is None:
            return None
        # active is True: settled iff empty book on BOTH sides
        no_levels = not (st.yes_levels or st.no_levels)
        no_top = (st.yes_top_bid_c is None and st.no_top_bid_c is None)
        return bool(no_levels and no_top)

    # Group by event_base
    from collections import defaultdict as _dd
    by_event: dict[str, list] = _dd(list)
    for tk in series_tickers:
        parts = tk.split("-")
        if len(parts) < 2:
            continue
        eb = f"{parts[0]}-{parts[1]}"
        by_event[eb].append(_is_settled(statuses.get(tk)))

    # Temporal guard: require event start >= min_age_minutes ago
    import time as _time
    now_ts = _time.time()
    settled: set[str] = set()
    for eb, verdicts in by_event.items():
        if not verdicts:
            continue
        # All sides settled (no UNKNOWN, no LIVE)?
        if not all(v is True for v in verdicts):
            continue
        # Use any series ticker for this event for the start-time lookup —
        # they all share the same scheduled-start prefix.
        eb_ticker = next(t for t in series_tickers
                         if t.startswith(eb + "-"))
        sched = parse_scheduled_start(eb_ticker)
        if sched is not None and (now_ts - sched) < min_age_minutes * 60.0:
            continue  # Too young — pre-game empty book risk
        settled.add(eb)
    return settled


_TICKER_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                  "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _kalshi_body_to_date(body: str) -> str:
    """'26AUG100415DRXDNF' -> '2026-08-10' (or '' if unparseable).

    Only used to narrow the Riot schedule lookup, which accepts neighbouring
    days — the ticker date can diverge from the real one
    ([[project_kalshi_map_date_divergence]]) and `hhmm` is Eastern, so this is a
    filter, never a key.
    """
    try:
        yy, mon, dd = body[:2], body[2:5].upper(), body[5:7]
        return f"20{yy}-{_TICKER_MONTHS[mon]:02d}-{int(dd):02d}"
    except Exception:
        return ""


def _anchor_team_names(pm: dict, event_base: str) -> tuple[list, str]:
    """Team names to feed the Riot join: Poly alignments, else Kalshi sub_titles.

    Returns (names, source) with source in {'poly', 'kalshi-subtitle', ''}.

    The fallback exists because a Poly-matching miss used to CASCADE: an event
    with empty team_alignments was skipped before the Riot lookup ever ran, so
    the anchor stayed frozen on a market that was otherwise fine (ESBOT /
    CKZMEA, 26AUG11). Kalshi's own yes_sub_titles are full team names and the
    Riot matcher normalises them at least as well as Poly outcome names.
    find_match is orientation-agnostic, so sub_title order doesn't matter.
    The caller must log source == 'kalshi-subtitle' loudly — it means the
    Poly mapping is broken upstream and someone should look at THAT too.
    """
    teams = (pm or {}).get("team_alignments") or []
    if len(teams) == 2:
        return teams, "poly"
    subs = _fetch_kalshi_subtitles(event_base)
    if len(subs) >= 2:
        return subs[:2], "kalshi-subtitle"
    return [], ""


def _p3_preserving_p1(p1_a: float, p2_a_old: float, p2_a_new: float) -> float:
    """p1 counter-move that holds p3 = (p1+p2)/2 fixed across a p2 write.

    BO3 only. The decider continuation is DERIVED — solve_game_3_probability
    reads p3 as (p1+p2)/2 — so writing p2 alone silently drags p3 with it
    while p1 stays frozen at its capture value. That was the JLNAVI/EPGM
    defect (26AUG11, −$10k week): the decider phase ran on a p3 still
    carrying pregame p1. Counter-moving p1 makes p2−p1 track the market's
    veto/momentum skew while p3 only changes when something deliberately
    writes it (capture, anchor, a future intermission solve). Holding p3
    also matches the veto structure itself: map-2 news is anti-correlated
    with map-1 (a team stronger on its opponent's pick is weaker on its
    own), which is exactly p2 up / p1 down around a fixed level.

    Returns the new p1_a. Since p3_pre = (p1+p2_old)/2, holding it gives
    p1_new = p1 + p2_old − p2_new, clamped to [0.01, 0.99] — a clamp means
    p3 must shift; the caller logs that loudly.
    """
    return min(0.99, max(0.01, p1_a + p2_a_old - p2_a_new))


def _refresh_g2_probabilities(valid_rows, poly_mappings, metadata_cache=None):
    """Pass 2: refresh future-game probabilities (p2 always; p3/p4 for BO5)
    in esports_probabilities.csv for events already captured, while the
    preceding map hasn't decided yet and Poly's GN mid is tight.

    Activated by marker files in cwd:
      enable_p2_refresh_shadow.flag : audit-log only, no CSV writeback
      enable_p2_refresh_live.flag   : audit-log + modify valid_rows in place
    Both absent → no-op.

    Per-leg gates (ALL required to refresh leg N, where N ∈ {2,3,4}):
      1. m(N-1) not finalized (Game(N-1) still active)
      2. Operative G(N) spread ≤ 6c (Poly or Kalshi, per data_source)
      3. Each team's m(N-1) yes_bid ≥ 5c (Game(N-1) result not yet priced in)

    For BO3 events, only G2 is refreshed (no p3/p4). For BO5 events, all
    three legs are refreshed each cycle, each writing to its own CSV col:
      p2 → col 2, p3 → col 6, p4 → col 9
    """
    shadow_on = os.path.exists("enable_p2_refresh_shadow.flag")
    live_on = os.path.exists("enable_p2_refresh_live.flag")
    # Pregame / intermission anchor re-solve (BO3 LoL + Dota — the sports
    # G2-REFRESH excludes by scope). ON BY DEFAULT: starting populate_configs is
    # all it takes. It lives inside this function only to reuse the per-cycle WS
    # probe cache, and must NOT inherit the p2-refresh flags — turning p2 off
    # would otherwise silently kill the anchor too.
    #
    #   disable_pregame_anchor.flag        kill switch, reverts to pre-26AUG11
    #                                      behavior (frozen populate anchors)
    #   pregame_anchor_shadow.flag         evaluate and log, write nothing
    #
    # Riot being unavailable (the 26AUG05-10 403 window) degrades to exactly the
    # same thing as the kill switch — see riot_game_state.unavailable().
    anchor_off = os.path.exists("disable_pregame_anchor.flag")
    anchor_shadow = os.path.exists("pregame_anchor_shadow.flag")
    anchor_live = not anchor_off and not anchor_shadow
    # 26AUG13 operator rule: never write when Kalshi is down/unconfirmed —
    # detection and snapshots continue (they self-invalidate via the clamp and
    # tomorrow's tick-check), but nothing lands in the CSV this cycle.
    if anchor_live:
        _ok, _why = kalshi_writes_safe()
        if not _ok:
            anchor_live = False
            print(f"  [ANCHOR] WRITES SUPPRESSED this cycle — {_why} "
                  f"(26AUG13 FOXDRX rule + 26AUG20 HLEDK post-halt cooldown)")
    anchor_on = not anchor_off
    # ── Per-phase and per-tier scope (26AUG11, operator) ──────────────────────
    #   disable_anchor_kickoff.flag       kickoff phase off (both tiers)
    #   disable_anchor_intermission.flag  intermission phase off (both tiers)
    #   enable_t2_intermission.flag       opt tier-2 BACK IN to the intermission
    #
    # Default: tier-1 gets both phases; TIER-2 GETS KICKOFF ONLY.
    # The anchor's premise is that the series price is the level authority and
    # any gap is our staleness. Tier-2 books run 5c+ wide, where "the market
    # moved" is often one participant repricing, so that premise is weakest
    # exactly where the intermission back-out is most fragile — it DIVIDES by a
    # book price. Both bad writes on 26AUG11 (OTNBS 13.3x, 3BLPIV 3.8x) were
    # thin tier-2 books; 3BLPIV landed 22 points off the live back-out.
    # The kickoff phase does not divide, so it stays on for both tiers.
    kickoff_off = os.path.exists("disable_anchor_kickoff.flag")
    interm_off = os.path.exists("disable_anchor_intermission.flag")
    t2_interm_on = os.path.exists("enable_t2_intermission.flag")
    if not (shadow_on or live_on or anchor_on):
        return

    audit_path = "esports_probabilities_refresh_log.csv"
    from datetime import datetime as _dt
    ts_str = _dt.now().isoformat(timespec="seconds")
    mode = "live" if live_on else "shadow"

    by_event: dict = {}
    for row in valid_rows:
        ticker = row[0].strip() if row else ""
        if not ticker:
            continue
        parts = ticker.rsplit("-", 1)
        if len(parts) != 2:
            continue
        by_event.setdefault(parts[0], []).append(row)

    # New audit format includes `leg` column (G2/G3/G4) — one row per leg per cycle.
    audit_header = ["ts", "event_base", "ticker_a", "ticker_b", "leg",
                    "cur_p_a", "cur_p_b", "new_p_a", "new_p_b",
                    "m_target_bid", "m_target_ask", "g_spread_c",
                    "m_prev_bid_a", "m_prev_bid_b",
                    "m_prev_finalized", "gate_passed", "skip_reason", "mode"]
    write_header = not os.path.exists(audit_path)
    audit_f = open(audit_path, "a", newline="")
    audit_w = csv.writer(audit_f)
    if write_header:
        audit_w.writerow(audit_header)

    def _fetch_markets(event_ticker):
        for attempt in range(2):
            try:
                r = _get_with_retry("https://external-api.kalshi.com/trade-api/v2/markets",
                                 params={"event_ticker": event_ticker, "limit": 5},
                                 timeout=4)
                if r.status_code == 200:
                    return r.json().get("markets", []), ""
                if r.status_code == 429 and attempt == 0:
                    time.sleep(1.0)
                    continue
                return [], "rate_limited" if r.status_code == 429 else f"http_{r.status_code}"
            except Exception as e:
                return [], f"exc_{type(e).__name__}"
        return [], "rate_limited"

    # ─── WS map-status probe (one batched subscribe for all per-event maps) ───
    # Collects every (m1/m2/m3-side) ticker the loop below will need, probes
    # them in one WS connection, and exposes a fetch helper that synthesizes
    # the same (markets_list, err_str) shape `_fetch_markets` returns. Maps
    # are the bulk of the call volume — 4 calls/event × ~30 events drained
    # the /markets? bucket. The series fetch (s_markets, used only for
    # yes_sub_title → sub_a/sub_b alignment) stays on REST for now;
    # yes_sub_title isn't in the WS orderbook payload.
    _map_tickers_to_probe: list[str] = []
    for _eb, _rows in by_event.items():
        if len(_rows) != 2:
            continue
        _ta = (_rows[0][0] or "").strip()
        _tb = (_rows[1][0] or "").strip()
        if not _ta or not _tb or "-" not in _ta or "-" not in _tb:
            continue
        _sa = _ta.rsplit("-", 1)[-1]
        _sb = _tb.rsplit("-", 1)[-1]
        # BO5 flag (same logic as the main loop — row[6] non-empty/non-"-")
        _is_bo5 = False
        if len(_rows[0]) > 6:
            _p3s = (_rows[0][6] or "").strip()
            _is_bo5 = _p3s not in ("", "-")
        if "MATCH" in _eb:
            _map_base = _eb.replace("MATCH", "SETWINNER")
        else:
            _map_base = _eb.replace("GAME", "MAP")
        # Probe m1, m2; for BO5 also m3. Leg-4 refresh is disabled (see
        # `legs` construction below), so m4 tickers don't need probing.
        _map_idxs = [1, 2] + ([3] if _is_bo5 else [])
        for _n in _map_idxs:
            _map_tickers_to_probe.append(f"{_map_base}-{_n}-{_sa}")
            _map_tickers_to_probe.append(f"{_map_base}-{_n}-{_sb}")
        # Series legs too — the pregame anchor re-solve needs the series book
        # alongside map 1, and adding them here keeps it at zero extra REST.
        if anchor_on:
            _map_tickers_to_probe.append(_ta)
            _map_tickers_to_probe.append(_tb)
    _map_tickers_to_probe = sorted(set(_map_tickers_to_probe))

    _ws_status_cache: dict = {}
    if _map_tickers_to_probe:
        _t_probe = time.time()
        try:
            _ws_status_cache = _ws_probe_markets(_map_tickers_to_probe, timeout_s=4.0)
            _classified = sum(1 for s in _ws_status_cache.values() if s.active is not None)
            print(f"  [G2-REFRESH] WS-probed {len(_map_tickers_to_probe)} map tickers "
                  f"in {time.time() - _t_probe:.2f}s "
                  f"({_classified}/{len(_map_tickers_to_probe)} classified)")
        except Exception as _e:
            print(f"  [G2-REFRESH] WS probe failed: {type(_e).__name__}: {_e}")
            _ws_status_cache = {}

    def _fetch_map_markets_via_ws(*target_tickers):
        """Look up specific map tickers in the cycle's WS probe cache.

        Returns the same (markets, err) shape `_fetch_markets` returns so the
        downstream `_evaluate_leg` consumer needs zero changes. Synthesizes
        each market dict with the fields `_evaluate_leg` actually reads:
        `ticker`, `status`, `yes_bid_dollars`, `yes_ask_dollars`. Finalized
        tickers get `status="finalized"` and no bid/ask — the bid≥5c gate
        already handles missing data gracefully via `m_prev_bid_by_suffix`.

        Any unclassified ticker (probe timeout) → "ws_unknown" err. The
        caller's `any(fetch_errs)` check then skips refresh for this event
        on this cycle, which is the same conservative behavior as a REST
        429/timeout."""
        markets = []
        for t in target_tickers:
            ms = _ws_status_cache.get(t)
            if ms is None or ms.active is None:
                return [], "ws_unknown"
            m = {"ticker": t}
            if ms.active is False:
                m["status"] = "finalized"
            else:
                m["status"] = "active"
                if ms.yes_top_bid_c is not None:
                    m["yes_bid_dollars"] = f"{ms.yes_top_bid_c / 100:.4f}"
                if ms.no_top_bid_c is not None:
                    m["yes_ask_dollars"] = f"{(100 - ms.no_top_bid_c) / 100:.4f}"
            markets.append(m)
        return markets, ""

    def _to_cents(s):
        try:
            return round(float(s) * 100)
        except (TypeError, ValueError):
            return None

    def _suffix(t):
        return t.rsplit("-", 1)[-1]

    def _evaluate_leg(*, leg_num, event_base, ticker_a, ticker_b, sub_a, sub_b,
                      suffix_a, suffix_b, m_prev_markets, m_target_markets,
                      poly_leg_key, cur_p_a, cur_p_b, use_poly_source,
                      m_prev_prev_markets=None):
        """Evaluate one leg's gates and compute new prob. Returns audit dict.

        m_target_markets is the Kalshi map-N book (used ONLY when source=kalshi
        — Poly source doesn't need it). Important for G4: Kalshi m4 ticker
        often doesn't exist until G3 starts, but Poly G4 winner markets can
        still be tight and refreshable in state 0/1. Don't early-exit on
        empty Kalshi m_target if we're in Poly source mode.
        """
        result = {
            "leg": f"G{leg_num}",
            "cur_p_a": cur_p_a, "cur_p_b": cur_p_b,
            "new_p_a": "", "new_p_b": "",
            "m_target_bid": "", "m_target_ask": "", "g_spread": "",
            "m_prev_bid_a": "", "m_prev_bid_b": "",
            "m_prev_finalized": "",
            "gate_passed": False, "skip_reason": "",
        }

        m_prev_finalized = any(m.get("status") in ("finalized", "settled")
                               for m in m_prev_markets) if m_prev_markets else False
        # Poly-source fallback: Kalshi may never spawn prev-leg sub-ticker for
        # a BO5 (e.g., KCFNC had no m3 ticker), so an empty m_prev_markets list
        # silently disables Gate 3. For Poly source we treat Polymarket as the
        # source of truth — if Poly's game{N-1} outcomePrice on either side is
        # > 0.95, that map has already resolved and we must freeze p{N} rather
        # than overwriting it with Poly's live in-progress mid for game{N}.
        m_prev_via_poly = False
        # leg_num >= 2 (was >= 3, 2026-07-15): G2 refresh must judge whether
        # game 1 is decided via POLY when data_source=poly, NOT the Kalshi map
        # book. For date-divergent games the Kalshi game-1 ticker is a phantom
        # under the wrong date (empty), which falsely tripped m1_bid_dead and
        # froze p2 all game (EFTL: TL G2 stuck at 43% vs real 33%).
        if not m_prev_finalized and use_poly_source and leg_num >= 2:
            try:
                _mp = poly_mappings.get(event_base) or {}
                _slug = _mp.get("poly_slug") or ""
                if _slug:
                    _pe = _fetch_poly_event_by_slug(_slug)
                    if _pe:
                        _gp = _get_poly_winner_markets(_pe).get(f"game{leg_num - 1}")
                        if _gp:
                            try:
                                _nums = [float(x) for x in (_gp.get("p") or [])]
                                if _nums and (max(_nums) > 0.95 or min(_nums) < 0.05):
                                    m_prev_finalized = True
                                    m_prev_via_poly = True
                            except (TypeError, ValueError):
                                pass
            except Exception:
                pass
        result["m_prev_finalized"] = "poly_resolved" if m_prev_via_poly else m_prev_finalized

        # Populate Kalshi m_target columns if available (for audit transparency
        # and for Kalshi-source fallback). Empty m_target is fine for poly source.
        # CROSS-BOOK spread: when both sides' books are available, compute the
        # tradable cross-book spread rather than one side's direct yes_bid/ask
        # gap. The previous single-side read [0] reported phantom 20-30c spreads
        # for thin-LVG/tight-TYLOO scenarios where the actual tradable Kalshi
        # spread was ~5c — silently blocking p2 refresh on LVGTYLOO 2026-06-05.
        kalshi_target_spread = None
        if m_target_markets:
            a_bid = _to_cents(m_target_markets[0].get("yes_bid_dollars")) or 0
            a_ask = _to_cents(m_target_markets[0].get("yes_ask_dollars")) or 100
            if len(m_target_markets) >= 2:
                b_bid = _to_cents(m_target_markets[1].get("yes_bid_dollars")) or 0
                b_ask = _to_cents(m_target_markets[1].get("yes_ask_dollars")) or 100
                # Cross-book A YES bid/ask: best of (direct, synthetic via B).
                xb_bid = max(a_bid, 100 - b_ask)
                xb_ask = min(a_ask, 100 - b_bid)
                kalshi_target_spread = max(0, xb_ask - xb_bid)
                result["m_target_bid"] = xb_bid
                result["m_target_ask"] = xb_ask
            else:
                # Single-side fallback (legacy) — used when only one M_N market
                # has spawned. Worse than cross-book but better than nothing.
                kalshi_target_spread = a_ask - a_bid
                result["m_target_bid"] = a_bid
                result["m_target_ask"] = a_ask

        # Gate 2: operative spread (Poly when poly source available, else Kalshi)
        g_spread = None
        g_source = None
        if use_poly_source:
            g_spread = 999
            g_source = "poly_unreachable"
            mapping = poly_mappings.get(event_base) or {}
            poly_slug = mapping.get("poly_slug") or ""
            if poly_slug:
                try:
                    pe = _fetch_poly_event_by_slug(poly_slug)
                    if pe:
                        g_w = _get_poly_winner_markets(pe).get(poly_leg_key)
                        if g_w and g_w.get("tokens"):
                            ps = _get_poly_spread(g_w["tokens"][0])
                            if ps is not None:
                                g_spread = round(ps)
                                g_source = "poly"
                except Exception:
                    pass
        else:
            # Kalshi-source: require Kalshi m_target spread
            if kalshi_target_spread is None:
                result["skip_reason"] = f"no_kalshi_m{leg_num}_book"
                return result
            g_spread = kalshi_target_spread
            g_source = "kalshi"
        result["g_spread"] = g_spread

        # Gate 3: m(N-1) bids — read on m_prev_markets.
        # If m_prev_markets is empty (Kalshi hasn't seeded map(N-1) yet —
        # common for G4 in state 1 where m3 ticker doesn't exist), don't
        # apply Gate 3. m_prev hasn't started, so we can't infer "previous
        # game decided" from missing bids; default to "continue" rather than
        # falsely tripping with bid_dead(0,0).
        m_prev_bid_by_suffix = {}
        if m_prev_markets:
            m_prev_bid_by_suffix = {
                _suffix(m.get("ticker", "")): (_to_cents(m.get("yes_bid_dollars")) or 0)
                for m in m_prev_markets
            }
        bid_a = m_prev_bid_by_suffix.get(suffix_a)
        bid_b = m_prev_bid_by_suffix.get(suffix_b)
        result["m_prev_bid_a"] = bid_a if bid_a is not None else ""
        result["m_prev_bid_b"] = bid_b if bid_b is not None else ""

        # 2026-07-20: 6c → 8c, matching the pregame probability-capture
        # tolerance. Gate 2 bounds the spread of the book we READ the new
        # probability from (Poly for data_source=poly, else Kalshi). It does
        # NOT bound the spread of the Kalshi G_N book we trade — see
        # GATE2_TRADED_BOOK_MAX below.
        GATE2_THRESHOLD = 8
        if leg_num == 4:
            # G4 policy 2026-05-25: refresh ONLY while G3 is ongoing (state 2).
            # Pre-G3 (state 0/1): Poly G4 books are dominated by thin/manipulated
            # quotes (saw 5 lots 1c wide at 46.5% when real prob was 36%).
            # Post-G3 (state ≥3): freeze whatever real-G3-ongoing value we got;
            # G4 imminent/playing means quoter must rely on the captured value.
            # Three conditions for refresh:
            #   1. m2 finalized (G2 done — state ≥ 2)
            #   2. m3 not finalized (G3 still playing — state < 3)
            #   3. m4 not finalized (G4 not yet over)
            m2_finalized = bool(m_prev_prev_markets) and any(
                m.get("status") in ("finalized", "settled") for m in m_prev_prev_markets
            )
            # Poly fallback for events whose Kalshi m2 ticker is missing.
            if not m2_finalized and use_poly_source:
                try:
                    _mpp = poly_mappings.get(event_base) or {}
                    _slugpp = _mpp.get("poly_slug") or ""
                    if _slugpp:
                        _pepp = _fetch_poly_event_by_slug(_slugpp)
                        if _pepp:
                            _g2pp = _get_poly_winner_markets(_pepp).get("game2")
                            if _g2pp:
                                try:
                                    _npp = [float(x) for x in (_g2pp.get("p") or [])]
                                    if _npp and (max(_npp) > 0.95 or min(_npp) < 0.05):
                                        m2_finalized = True
                                except (TypeError, ValueError):
                                    pass
                except Exception:
                    pass
            if not m2_finalized:
                result["skip_reason"] = "g2_unresolved_g4_baseline"
                return result
            if m_prev_finalized:
                # m3 done → G3 ended → freeze G4 at last captured value
                result["skip_reason"] = (f"poly_g3_resolved_g4_frozen"
                                         if m_prev_via_poly
                                         else "m3_finalized_g4_frozen")
                return result
            m_target_finalized = bool(m_target_markets) and any(
                m.get("status") in ("finalized", "settled") for m in m_target_markets
            )
            if m_target_finalized:
                result["skip_reason"] = "m4_finalized"
                return result
        else:
            if m_prev_finalized:
                result["skip_reason"] = (f"poly_g{leg_num-1}_resolved"
                                         if m_prev_via_poly
                                         else f"m{leg_num-1}_finalized")
                return result
        if g_spread > GATE2_THRESHOLD:
            result["skip_reason"] = f"m{leg_num}_spread={g_spread}c>{GATE2_THRESHOLD}({g_source})"
            return result
        # m_prev bid_dead = early-warning that prev game is ending.
        # For G4: m3 bids are actively traded during G3; check applies normally
        # and acts as the freeze trigger before m3_finalized status flips.
        #
        # 2026-07-20: RE-ENABLED for data_source=poly. The 07-15 change skipped
        # this entirely for poly-source events, on the reasoning that prev-game
        # state must come from Poly and that a date-diverged Kalshi map ticker
        # reads as a phantom (a=0,b=0) that would falsely freeze p2. But the
        # m_prev_via_poly path never actually detected game end, so poly-source
        # events had NO working stop condition: gate 1 (m_prev_finalized) waits
        # on Kalshi settlement, which lags hours. PCKCPLYNX 26JUL20 refreshed
        # p2 all through game 2 off the LIVE in-progress G2 price (mid 0.92-0.955
        # at 13:21-13:36, series state S2, m1 bids 0/0) — only the baseline_dev
        # sanity gate stopped those writes, which is not what that gate is for.
        #
        # The phantom concern is already handled upstream: when the map(N-1)
        # ticker is absent, m_prev_markets is empty, so bid_a/bid_b are None and
        # the is-not-None guard below skips the check. `not use_poly_source` was
        # therefore redundant protection. Replay over 26JUL20: 81 of 511
        # gate-passing refreshes would now be blocked, all of them post-game-start.
        if bid_a is not None and bid_b is not None:
            if bid_a < 5 or bid_b < 5:
                result["skip_reason"] = f"m{leg_num-1}_bid_dead(a={bid_a},b={bid_b})"
                return result

        # Gates passed — compute new prob from the source we trade off.
        mid_a = None
        if use_poly_source:
            mapping = poly_mappings.get(event_base) or {}
            poly_slug = mapping.get("poly_slug") or ""
            if not poly_slug:
                result["skip_reason"] = "no_poly_mapping"
                return result
            try:
                pe = _fetch_poly_event_by_slug(poly_slug)
            except Exception:
                pe = None
            if not pe:
                result["skip_reason"] = "poly_fetch_failed"
                return result
            g_w = _get_poly_winner_markets(pe).get(poly_leg_key)
            if not g_w or len(g_w.get("tokens", [])) != 2 or len(g_w.get("outcomes", [])) != 2:
                result["skip_reason"] = f"no_poly_{poly_leg_key}"
                return result
            poly_outcomes = g_w.get("outcomes", [])
            poly_tokens = g_w.get("tokens", [])
            idx_a = _align_via_poly_parsed_markets(sub_a, event_base, poly_outcomes)
            idx_b = _align_via_poly_parsed_markets(sub_b, event_base, poly_outcomes)
            if idx_a < 0 or idx_b < 0 or idx_a == idx_b:
                result["skip_reason"] = f"alignment_failed(a={idx_a},b={idx_b})"
                return result
            try:
                mr = _get_with_retry(
                    f"https://clob.polymarket.com/midpoint?token_id={poly_tokens[idx_a]}",
                    timeout=5)
                if mr.status_code == 200:
                    v = mr.json().get("mid")
                    if v:
                        mid_a = float(v)
            except Exception:
                pass
            if mid_a is None:
                result["skip_reason"] = "poly_midpoint_failed"
                return result
        else:
            # Kalshi mid: find team-A's market in m_target_markets and use
            # (yes_bid + yes_ask) / 2 as the fair mid. m_target_markets is
            # already loaded by the caller for the gate-2 spread check.
            team_a_market = None
            for _m in (m_target_markets or []):
                _t = _m.get("ticker", "") or ""
                if _t.endswith(f"-{suffix_a}"):
                    team_a_market = _m
                    break
            if not team_a_market:
                result["skip_reason"] = f"no_kalshi_m{leg_num}_team_a"
                return result
            _bid = _to_cents(team_a_market.get("yes_bid_dollars"))
            _ask = _to_cents(team_a_market.get("yes_ask_dollars"))
            if _bid is None or _ask is None or _bid <= 0 or _ask >= 100:
                result["skip_reason"] = f"kalshi_m{leg_num}_empty_book(bid={_bid},ask={_ask})"
                return result
            mid_a = (_bid + _ask) / 200.0  # avg in cents / 100 → fraction
        if mid_a >= 0.97 or mid_a <= 0.03:
            result["skip_reason"] = f"extreme_p{leg_num}(a={mid_a:.3f})"
            return result

        # Sum-to-1: round team_a, derive team_b = 1 - team_a
        result["new_p_a"] = round(mid_a, 3)
        result["new_p_b"] = round(1.0 - result["new_p_a"], 3)
        # Baseline-deviation sanity gate (anchor = first cur_p_a seen this process).
        # Rejects single-cycle Poly G_N mid spikes that pass the spread gate but
        # are still anomalous vs the populate-time baseline.
        ok, skip = check_p_sanity((event_base, leg_num), result["new_p_a"],
                                  cur_p_a, time.time())
        if not ok:
            result["skip_reason"] = skip
            result["new_p_a"] = ""
            result["new_p_b"] = ""
            return result
        result["gate_passed"] = True
        return result

    def _write_audit(leg_str, cur_a, cur_b, skip_reason):
        """Write an early-exit audit row (fetch failure / empty markets)."""
        audit_w.writerow([ts_str, event_base, ticker_a, ticker_b, leg_str,
                          cur_a, cur_b, "", "",
                          "", "", "", "", "",
                          "", False, skip_reason, mode])

    # ── Pregame anchor re-solve (LoL/Dota) ───────────────────────────────────
    # The capture loop is gated on `sm_ticker not in existing_probs`, so a row's
    # p1/p2/series are written once and never revisited — median 127 min before
    # the real game-1 start, p90 183. Anything the market reprices in between
    # becomes a standing one-sided basis, because solve_game_3_probability
    # derives G3 as (p1+p2)/2 and discards the series price. DRX/DNS 26AUG10:
    # p3 frozen at 0.612 vs a market back-out of 0.534, -$29,605.
    #
    # This records a reconciled solve every cycle while the event is pregame and
    # rewrites the row once riot_game_state confirms the TRUE game-1 start,
    # using the cached solves from strictly before that instant. Scheduled times
    # are useless here (only 8% land within +/-5 min of the real start).
    if anchor_on:
        try:
            import pregame_anchor
            import riot_game_state as _rgs
        except Exception as _e:
            print(f"[ANCHOR] module import failed ({type(_e).__name__}: {_e}) — "
                  f"probabilities keep their populate-time values")
            _rgs = None
    if anchor_on and _rgs is not None and _rgs.unavailable():
        # Designed degradation, not an error: identical to pre-26AUG11 behavior.
        print(f"[ANCHOR] Riot unavailable ({_rgs.disabled_reason()}) — "
              f"re-anchoring OFF this cycle, frozen populate anchors kept")
        anchor_on = False
    if anchor_on and _rgs is not None:
        try:
            def _anchor_book(t):
                """(yes_bid, yes_ask) in cents; EITHER may be None.

                A one-sided book is not "no data" — on a decided map the losing
                side stops being quoted entirely, and that emptiness is the
                signal pregame_anchor.map1_decided looks for. Collapsing it to
                None (the 26AUG10 bug) made the pinned-book fallback dead code
                in exactly the case it existed for. Consumers that need a real
                two-sided book check for None themselves.
                """
                ms = _ws_status_cache.get(t)
                if ms is None or ms.active is not True:
                    return None
                yb, nb = ms.yes_top_bid_c, ms.no_top_bid_c
                if yb is None and nb is None:
                    return None
                return yb, (None if nb is None else 100 - nb)

            def _anchor_settled(t):
                """True/False/None — has Kalshi settled this market?

                The intermission G3 back-out is only valid once map 1 is
                DECIDED; this is Kalshi's own settlement, never a guess from
                the price.
                """
                ms = _ws_status_cache.get(t)
                if ms is None or ms.active is None:
                    return None
                return ms.active is False

            _anchor_events = []
            _skipped_bo5 = _stale = 0
            # NOTE (26AUG12): the VAL pregame taker gate was measured in shadow
            # over 4 covered VCT games and RETIRED by operator decision. Three
            # games cleared with 9-11min veto->pistol margins, but the fourth
            # (NAVITH) had bo3 publish the veto 2m49s before the pistol after
            # an earlier-reschedule — the release chain cannot beat that, and
            # a live gate would have blocked the pistol take, the single most
            # valuable taker window in VAL. Pregame protection for VAL is the
            # delta-preserving capture reconcile (no standing lean) plus the
            # intermission reprice; there is deliberately NO taker gate.
            for _eb, _rows in by_event.items():
                _nofeed = _eb.startswith(INTERMISSION_ONLY_PREFIXES)
                if _nofeed and not os.path.exists(
                        INTERMISSION_ONLY_FLAGS.get(_eb.split("-", 1)[0],
                                                    _INTERMISSION_DEFAULT_FLAG)):
                    # GATED (operator, 26AUG12): each no_feed sport arms via its
                    # own flag (VAL/CS2 share enable_valcs_intermission.flag,
                    # Dota has enable_dota_intermission.flag) — touch to arm,
                    # rm to kill, no restart.
                    continue
                if (not (_eb.startswith(PREGAME_ANCHOR_PREFIXES) or _nofeed)
                        or len(_rows) != 2):
                    continue
                # esports_probabilities.csv keeps every event it has ever seen
                # (400+ LoL/Dota rows), while poly_parsed_markets.csv holds only
                # the CURRENT slate. An event missing from the latter is settled
                # and long gone — skip it quietly, or the log gains ~400 lines
                # every populate cycle.
                _pm = (poly_mappings or {}).get(_eb)
                if _pm is None:
                    _stale += 1
                    continue
                if _eb.startswith("KXDOTA2GAME") and "THE INTERNATIONAL" not in (
                        (_pm.get("poly_title") or "").upper()):
                    # Operator scope 26AUG12: Dota intermission is THE
                    # INTERNATIONAL only. Everything else keeps frozen anchors.
                    continue
                if _nofeed:
                    # No Riot join for the no_feed sports. VAL/CS2 don't need
                    # team names at all; Dota uses them for the bo3 binding —
                    # but a failed name resolve must NOT kill the event, the
                    # book-pin path works nameless.
                    _teams, _tsrc = ["-", "-"], "none"
                    if _eb.startswith("KXDOTA2GAME"):
                        _dteams, _dsrc = _anchor_team_names(_pm, _eb)
                        if len(_dteams) == 2:
                            _teams, _tsrc = _dteams, _dsrc
                        else:
                            print(f"  [ANCHOR] {_eb}: TI event but no team names "
                                  f"(team_alignments {_pm.get('team_alignments')!r}) "
                                  f"— bo3 g1-end detection unavailable, book-pin "
                                  f"path only")
                else:
                    _teams, _tsrc = _anchor_team_names(_pm, _eb)
                    if len(_teams) != 2:
                        # On the current slate but unusable — that IS worth saying.
                        print(f"  [ANCHOR] {_eb}: on the slate but team_alignments is "
                              f"{_pm.get('team_alignments')!r} and Kalshi sub_titles "
                              f"unavailable — cannot resolve the Riot match, anchor "
                              f"stays frozen")
                        continue
                    if _tsrc == "kalshi-subtitle":
                        # LOUD demotion: the Riot join proceeds, but the Poly
                        # mapping is broken upstream and needs its own look.
                        print(f"  [ANCHOR] {_eb}: team_alignments missing — FALLBACK "
                              f"to Kalshi sub_titles {_teams} for the Riot join "
                              f"(Poly mapping failed upstream, investigate that too)")
                _ta = (_rows[0][0] or "").strip()
                _sa = _ta.rsplit("-", 1)[-1]
                _map_base = _eb.replace("MATCH", "SETWINNER") if "MATCH" in _eb \
                    else _eb.replace("GAME", "MAP")
                try:
                    _cur_p1 = float(_rows[0][1]); _cur_p2 = float(_rows[0][2])
                except (ValueError, IndexError):
                    continue
                # BO5 gets the KICKOFF anchor (solve_continuation handles both
                # formats); pregame_anchor refuses it the intermission on its own.
                _bo5 = len(_rows[0]) > 6 and (_rows[0][6] or "").strip() not in ("", "-")
                if _bo5 and _nofeed:
                    # Intermission-only sport + BO5 = nothing this pass can do
                    # (BO5 intermission does not invert; kickoff needs a feed).
                    _skipped_bo5 += 1
                    continue
                if _bo5:
                    _skipped_bo5 += 1        # counted as "BO5 (kickoff only)"
                _body = _eb.split("-", 1)[1] if "-" in _eb else ""
                _ewc_cs2 = False
                try:
                    if _eb.startswith("KXCS2GAME"):
                        _tier = classify_cs_tier(_eb, poly_mappings)
                        # EWC intermission exception (operator, 26AUG13): EWC
                        # CS2 is tier-2 as a SIZING decision (deliberate — see
                        # the 1.5x bump), but its books are top-event liquid,
                        # nothing like the thin-tier-2 books the intermission
                        # exclusion protects against (26AUG11 OTNBS/3BLPIV).
                        # Admit it to the intermission phase; sizing unchanged.
                        _ewc_cs2 = "ESPORTS WORLD CUP" in (
                            (_pm.get("poly_title") or "")).upper()
                    elif _eb.startswith("KXVALORANTGAME"):
                        _title_up = ((_pm.get("poly_title") or "")).upper()
                        _tier = ("tier2" if any(k in _title_up
                                                for k in _VAL_T2_TITLE_MARKERS)
                                 else "tier1")
                    elif _eb.startswith("KXDOTA2GAME"):
                        # NOT classify_lol_tier: Dota is a default-TIER-2 sport
                        # and the LoL classifier would call any non-LoL-T2 title
                        # tier-1, handing thin Dota books the intermission
                        # divide. (TI, the only admitted event today, is tier-1
                        # by the THE INTERNATIONAL keyword either way.)
                        _tier = classify_dota_tier(_eb, poly_mappings)
                    else:
                        _tier = classify_lol_tier(_eb, poly_mappings)
                except Exception:
                    _tier = "tier1"   # unknown -> treat as tier-1 (both phases)
                _win = INTERMISSION_WINDOWS.get(_eb.split("-", 1)[0])
                _anchor_events.append({
                    "event_base": _eb,
                    "ticker_a": _ta,
                    "map1_ticker_a": f"{_map_base}-1-{_sa}",
                    "map2_ticker_a": f"{_map_base}-2-{_sa}",
                    "team_a": _teams[0], "team_b": _teams[1],
                    "date": _kalshi_body_to_date(_body),
                    "is_bo5": _bo5,
                    # Phase permissions — policy decided HERE, enforced in
                    # pregame_anchor. Tier-2 is intermission-excluded by default.
                    "tier": _tier,
                    "no_feed": _nofeed,
                    "inter_timeout_min": (_win[0] if _win else None),
                    "inter_adopt_min": (_win[1] if _win else None),
                    "allow_kickoff": (not kickoff_off) and not _nofeed,
                    "allow_intermission": (
                        not interm_off and (_tier != "tier2" or t2_interm_on
                                            or _ewc_cs2)),
                    # The continuation we currently use: BO5 stores it
                    # explicitly in col 6, BO3 derives it as (p1+p2)/2.
                    "cur_p3": (float(_rows[0][6]) if _bo5 else (_cur_p1 + _cur_p2) / 2.0),
                })

            # ── bo3-backed g1-end for TI Dota (26AUG12) ──────────────────────
            # One bo3 list request per cycle; dota_game_state observes the
            # game-1-end transition and persists it. Injected as g1_state_ext
            # so map1_decided treats it exactly like Riot's g1 'completed'
            # (priority 1, ahead of the book pin). Any failure here degrades
            # to the book-pin path — never to a dead intermission.
            _dota_cands = [e for e in _anchor_events
                           if e["event_base"].startswith("KXDOTA2GAME")
                           and e.get("team_a") not in (None, "", "-")]
            if _dota_cands:
                try:
                    import dota_game_state as _dgs
                    _dgs.poll([{"event_base": e["event_base"],
                                "team_a": e["team_a"], "team_b": e["team_b"]}
                               for e in _dota_cands])
                    for e in _dota_cands:
                        if _dgs.g1_ended_at(e["event_base"]) is not None:
                            e["g1_state_ext"] = "completed"
                            e["g1_src"] = "bo3:g1-ended"
                except Exception as _e:
                    print(f"  [ANCHOR] dota bo3 state unavailable "
                          f"({type(_e).__name__}: {_e}) — book-pin path only")

            # ── bo3-frames pregame release for LoL (26AUG15) ─────────────────
            # Riot livestats first-frame lags true game start 5-20 min (BLGTT,
            # LNGNIP); bo3.gg live_updates.net_worth flows within minutes of
            # the real start (KTHLE validation). One list request per cycle.
            # GATE-ONLY: pregame_anchor releases the config latch on the ext
            # stamp; kickoff anchor adoption still waits for Riot's start.
            # Failure degrades to riot-only — never to a stuck latch.
            _lol_cands = [e for e in _anchor_events
                          if e["event_base"].startswith("KXLOLGAME")
                          and e.get("team_a") not in (None, "", "-")]
            if _lol_cands:
                try:
                    import lol_game_state as _lgs
                    _lgs.poll([{"event_base": e["event_base"],
                                "team_a": e["team_a"], "team_b": e["team_b"]}
                               for e in _lol_cands])
                    for e in _lol_cands:
                        _ts = _lgs.frames_live_at(e["event_base"])
                        if _ts is not None:
                            e["ext_started_at"] = _ts
                except Exception as _e:
                    print(f"  [ANCHOR] lol bo3 frames unavailable "
                          f"({type(_e).__name__}: {_e}) — riot-only this cycle")

            _anchors = pregame_anchor.run_pass(_anchor_events, _anchor_book,
                                               settled_fn=_anchor_settled)
            # ── PROOF-OF-TRADING gate (26AUG13, FOXDRX follow-up) ────────────
            # The status endpoint above is only a report; this is the foolproof
            # check — Kalshi must ACCEPT a real 1-lot order before any anchor
            # value lands in the CSV. Only fires when there is actually
            # something to write. Probe against the written events' own series
            # books first (a per-market pause counts as paused), then any
            # tracked book. NOTE run_pass one-shots each phase, so a write
            # suppressed here is DROPPED, not retried — same accepted
            # semantics as the status gate: missed write ~free, poisoned
            # write five figures.
            if _anchors and anchor_live:
                _probe_cands = (
                    [e["ticker_a"] for e in _anchor_events
                     if e["event_base"] in _anchors]
                    + [e["ticker_a"] for e in _anchor_events]
                    + [e.get("map2_ticker_a") for e in _anchor_events])
                if not kalshi_order_probe_ok(_probe_cands, _anchor_book):
                    anchor_live = False
                    print("  [ANCHOR] ORDER PROBE FAILED — Kalshi did not "
                          "accept a 1-lot probe order; anchor writes DEMOTED "
                          "TO SHADOW this cycle (only an accepted order "
                          "proves open trading)")
            _applied = 0
            for _eb, _new in _anchors.items():
                _rows = by_event.get(_eb) or []
                if len(_rows) != 2:
                    continue
                # series is left alone on the intermission write: post-G1 the
                # hedge math discards series_A entirely, and rewriting it would
                # change the number the operator reads without changing any
                # pricing.
                _ser = "" if _new.get("series") is None else f"{_new['series']:.3f}"
                if not anchor_live:
                    print(f"  [ANCHOR] {_eb}: SHADOW ({_new.get('phase')}) — would write "
                          f"p1={_new['p1']:.3f} p2={_new['p2']:.3f}"
                          + (f" series={_ser}" if _ser else " series=unchanged"))
                    continue
                if (_eb.startswith("KXDOTA2GAME")
                        and os.path.exists("dota_intermission_shadow.flag")):
                    # TI day-1 tracking-only (operator, 26AUG12): the full
                    # pipeline runs — bo3 g1-end, decided detection, snapshots,
                    # adoption — and the would-be write is logged here, but
                    # nothing is written. `rm dota_intermission_shadow.flag` to
                    # go live (checked per cycle, no restart). NOTE: run_pass
                    # one-shots each intermission, so removing the flag arms the
                    # NEXT series; an already-shadowed one does not re-fire.
                    print(f"  [ANCHOR] {_eb}: DOTA SHADOW ({_new.get('phase')}) — "
                          f"would write p1={_new['p1']:.3f} p2={_new['p2']:.3f}"
                          + (f" series={_ser}" if _ser else " series=unchanged"))
                    continue
                _rows[0][1] = f"{_new['p1']:.3f}"
                _rows[0][2] = f"{_new['p2']:.3f}"
                _rows[1][1] = f"{1 - _new['p1']:.3f}"
                _rows[1][2] = f"{1 - _new['p2']:.3f}"
                if _new.get("series") is not None:
                    _rows[0][3] = f"{_new['series']:.3f}"
                    _rows[1][3] = f"{1 - _new['series']:.3f}"
                # BO5: the continuation also lives in p3/p4/p5 (cols 6/9/10),
                # exactly as RECONCILE writes them at capture time. Leaving them
                # stale would let the hedge keep using the OLD continuation for
                # maps 3-5 while p2 moved, which is worse than not re-anchoring.
                if _new.get("p3") is not None:
                    _c = _new["p3"]
                    for _r, _v in ((_rows[0], _c), (_rows[1], 1 - _c)):
                        while len(_r) < 11:
                            _r.append("")
                        _r[6] = f"{_v:.3f}"
                        _r[9] = f"{_v:.3f}"
                        _r[10] = f"{_v:.3f}"
                _applied += 1
            _n_t2 = sum(1 for e in _anchor_events if e.get("tier") == "tier2")
            _n_nointer = sum(1 for e in _anchor_events if not e.get("allow_intermission"))
            print(f"[ANCHOR] {len(_anchor_events)} BO3 event(s) tracked "
                  f"({_skipped_bo5} BO5 kickoff-only, {_stale} settled/off-slate); "
                  f"{_applied} re-anchored "
                  f"{'LIVE' if anchor_live else '(shadow - no writes)'}"
                  f" | {_n_t2} tier-2, {_n_nointer} intermission-excluded"
                  + ("" if not kickoff_off else " | KICKOFF PHASE OFF")
                  + ("" if not interm_off else " | INTERMISSION PHASE OFF")
                  + ("" if not t2_interm_on else " | tier-2 intermission FORCED ON"))
        except Exception as _e:
            # Never let the anchor pass break populate.
            print(f"[ANCHOR] pass failed: {type(_e).__name__}: {_e}")

    if not (shadow_on or live_on):
        return
    # 26AUG13 operator rule: no probability writes of ANY kind while Kalshi
    # is down/unconfirmed (same blackout as the anchor path).
    _ok, _why = kalshi_writes_safe()
    if not _ok:
        print(f"[G2-REFRESH] WRITES SUPPRESSED this cycle — {_why} "
              f"(26AUG13 rule + 26AUG20 post-halt cooldown)")
        return

    evaluated = 0

    skipped_scope = 0   # LoL/Dota excluded by G2_REFRESH_EXCLUDED_PREFIXES
    refreshed = 0
    print(f"\n[G2-REFRESH] mode={mode}; evaluating {len(by_event)} existing event(s)...")

    # Build BOTH a (prefix, is_bo5) dict (legacy fallback) AND a config_id keyed
    # dict. The prefix dict is broken for CS because T1 (data_source=poly) and
    # T2 (data_source=kalshi) rows share market_prefix=KXCS2GAME, and
    # setdefault locks in the first row. ds_by_cid lets us route per-event by
    # classifying the tier first.
    ds_by_prefix = {}  # (prefix, is_bo5) → data_source — legacy fallback
    ds_by_cid = {}     # config_id → data_source — primary path
    try:
        with open("market_parameters.csv", "r") as _mf:
            _reader = csv.DictReader(_mf)
            for _row in _reader:
                _cid = (_row.get("config_id") or "").strip()
                _pfx = (_row.get("market_prefix") or "").strip()
                _ds = (_row.get("data_source") or "").strip().lower()
                if not _cid or not _pfx or not _ds:
                    continue
                _is_bo5 = _cid.endswith("_BO5")
                ds_by_prefix.setdefault((_pfx, _is_bo5), _ds)
                ds_by_cid[_cid] = _ds
    except FileNotFoundError:
        pass

    def _resolve_data_source(event_base: str, is_bo5: bool) -> str:
        """Per-event data_source lookup, tier-aware for CS2.

        For KXCS2GAME-* events, classify_cs_tier(event_base) → tier1/tier2,
        then look up the matching arber config_id (CS2_ARB_L1[_T1][_BO5]).
        For KXVALORANTGAME-* events, classify_val_tier → vct/vcl and probe the
        matching config_id (VAL_ARB_L1[_VCL][_BO5]) — VCT rows may carry
        kalshi_na_poly_eu (Americas-evening Kalshi) while VCL stays poly, and
        the per-prefix dict can't tell them apart (shared KXVALORANTGAME prefix).
        For other sports, fall back to the prefix-keyed dict.
        """
        _prefix = event_base.split("-", 1)[0]
        if _prefix == "KXCODGAME":
            return "poly"
        if _prefix == "KXCS2GAME":
            tier = classify_cs_tier(event_base, poly_mappings)
            t_suf = "_T1" if tier == "tier1" else ""
            bo_suf = "_BO5" if is_bo5 else ""
            # Probe arber row (any execution_type would do; data_source is
            # consistent across all rows for the same tier+BO mode).
            cid = f"CS2_ARB_L1{t_suf}{bo_suf}"
            ds = ds_by_cid.get(cid, "")
            if ds:
                return ds
        if _prefix == "KXVALORANTGAME":
            # VCT/VCL split (2026-07-18): keep prob-refresh source consistent
            # with trading routing. VCT rows → kalshi_na_poly_eu (window decides),
            # VCL rows → poly. Mirrors the CS2 tier branch above.
            v_suf = "" if classify_val_tier(event_base, s_markets) == "vct" else "_VCL"
            bo_suf = "_BO5" if is_bo5 else ""
            cid = f"VAL_ARB_L1{v_suf}{bo_suf}"
            ds = ds_by_cid.get(cid, "")
            if ds:
                return ds
        # Fallback: legacy per-prefix lookup
        return ds_by_prefix.get((_prefix, is_bo5), "")

    # is_na_hours retained for "kalshi_na_poly_eu" data_source mode (deferred
    # time-of-day decision — not used when data_source is explicitly poly/kalshi).
    try:
        from esports_config import is_na_hours
        _is_na_hours_fn = is_na_hours
        _is_na_hours_ok = True
    except Exception as e:
        _is_na_hours_fn = None
        _is_na_hours_ok = False
        try:
            with open("_g2_refresh_warnings.log", "a") as _lf:
                _lf.write(f"{ts_str} is_na_hours import failed: {e!r}; defaulting use_poly_source=True\n")
        except Exception:
            pass

    for event_base, rows in by_event.items():
        if event_base.startswith(G2_REFRESH_EXCLUDED_PREFIXES):
            skipped_scope += 1
            continue
        if len(rows) != 2:
            continue
        evaluated += 1

        ticker_a, ticker_b = rows[0][0].strip(), rows[1][0].strip()
        try:
            cur_p2_a = float(rows[0][2])
            cur_p2_b = float(rows[1][2])
        except (ValueError, IndexError):
            continue

        # BO5 detection: row[6] (p3 column) non-empty and not "-"
        is_bo5 = False
        if len(rows[0]) > 6:
            p3_str = (rows[0][6] or "").strip()
            is_bo5 = p3_str not in ("", "-")

        cur_p3_a = cur_p3_b = cur_p4_a = cur_p4_b = None
        if is_bo5:
            try:
                cur_p3_a = float(rows[0][6])
                cur_p3_b = float(rows[1][6])
            except (ValueError, IndexError):
                cur_p3_a = cur_p3_b = None
            try:
                cur_p4_a = float(rows[0][9])
                cur_p4_b = float(rows[1][9])
            except (ValueError, IndexError):
                cur_p4_a = cur_p4_b = None

        if "MATCH" in event_base:
            base_repl = lambda n: event_base.replace("MATCH", "SETWINNER") + f"-{n}"
        else:
            base_repl = lambda n: event_base.replace("GAME", "MAP") + f"-{n}"

        # Series fetch: cache-first, REST fallback. yes_sub_title is the only
        # field downstream reads, and it's static per ticker — so a cache hit
        # for BOTH tickers fully replaces the REST call. Previously this hit
        # /markets?event_ticker= every cycle for every event, draining the
        # ~3-5 token/sec bucket and spamming [429-RETRY] for minutes per cycle
        # (fix 2026-06-25; the 2026-06-24 WS migration only covered the map
        # probes, not this series fetch).
        s_markets, s_err = [], ""
        _s_cache = None
        if metadata_cache is not None:
            _cached = (metadata_cache.get(event_base) or {}).get("tickers") or []
            if _cached:
                _sub_by_t = {(e.get("ticker") or ""): (e.get("yes_sub_title") or "").strip()
                             for e in _cached}
                if _sub_by_t.get(ticker_a) and _sub_by_t.get(ticker_b):
                    _s_cache = [
                        {"ticker": e.get("ticker", ""),
                         "yes_sub_title": (e.get("yes_sub_title") or "").strip(),
                         # rules_primary MUST be carried through: _resolve_data_source
                         # (VCT/VCL split, 2026-07-18) calls classify_val_tier on this
                         # s_markets. Omitting it made is_vct_event() see empty rules →
                         # cached "vcl" in _val_tier_cache, poisoning tier/data-source
                         # for the whole cycle (all VCT events emitted _VCL rows).
                         "rules_primary": (e.get("rules_primary") or "").strip()}
                        for e in _cached if e.get("ticker")
                    ]
        if _s_cache is not None:
            s_markets = _s_cache
        else:
            s_markets, s_err = _fetch_markets(event_base)
            time.sleep(0.05)
            # Cache writeback on successful REST: same shape as the discovery-
            # path cache write at ~line 2172 so both sites stay interchangeable.
            if metadata_cache is not None and s_markets and not s_err:
                metadata_cache[event_base] = {
                    "tickers": [
                        {
                            "ticker": sm.get("ticker", ""),
                            "yes_sub_title": (sm.get("yes_sub_title") or "").strip(),
                            "rules_primary": (sm.get("rules_primary") or "").strip(),
                        }
                        for sm in s_markets if sm.get("ticker")
                    ]
                }
                _save_kalshi_metadata_cache(metadata_cache)

        # Map fetches: WS cache (built once at the top of this function).
        # Eliminates ~4 /markets?event_ticker= REST calls per event — the
        # bulk of populate_configs's 429 storm.
        _suffix_a = ticker_a.rsplit("-", 1)[-1] if "-" in ticker_a else ""
        _suffix_b = ticker_b.rsplit("-", 1)[-1] if "-" in ticker_b else ""
        _map_base = base_repl  # base_repl(n) already yields the full map base+leg
        m1_markets, m1_err = _fetch_map_markets_via_ws(
            f"{_map_base(1)}-{_suffix_a}", f"{_map_base(1)}-{_suffix_b}")
        m2_markets, m2_err = _fetch_map_markets_via_ws(
            f"{_map_base(2)}-{_suffix_a}", f"{_map_base(2)}-{_suffix_b}")
        if is_bo5:
            m3_markets, m3_err = _fetch_map_markets_via_ws(
                f"{_map_base(3)}-{_suffix_a}", f"{_map_base(3)}-{_suffix_b}")
        else:
            m3_markets, m3_err = [], ""
        # m4 fetch removed 2026-06-24: leg-4 refresh is permanently disabled
        # (see `legs` construction below), so m4 status/bid is never read.
        m4_markets, m4_err = [], ""

        fetch_errs = (s_err, m1_err, m2_err, m3_err, m4_err)
        if any(fetch_errs):
            err_label = "rate_limited" if "rate_limited" in fetch_errs \
                        else f"fetch_err({','.join(e or '-' for e in fetch_errs)})"
            _write_audit("G2", cur_p2_a, cur_p2_b, err_label)
            if is_bo5:
                if cur_p3_a is not None:
                    _write_audit("G3", cur_p3_a, cur_p3_b, err_label)
                if cur_p4_a is not None:
                    _write_audit("G4", cur_p4_a, cur_p4_b, err_label)
            continue

        if not s_markets:
            _write_audit("G2", cur_p2_a, cur_p2_b, "empty_series_markets")
            continue

        sub_by_ticker = {sm.get("ticker", ""): (sm.get("yes_sub_title", "") or "").strip()
                         for sm in s_markets}
        sub_a = sub_by_ticker.get(ticker_a, "")
        sub_b = sub_by_ticker.get(ticker_b, "")
        suffix_a = _suffix(ticker_a)
        suffix_b = _suffix(ticker_b)

        # Source dispatch: per-event data_source lookup. For CS2 this consults
        # classify_cs_tier so T1 IEM events get data_source=poly (not the T2
        # kalshi value that the per-prefix dict would silently return).
        _prefix = event_base.split("-", 1)[0]
        ds = _resolve_data_source(event_base, is_bo5)
        if ds == "poly":
            use_poly_source = True
        elif ds == "kalshi":
            use_poly_source = False
        elif ds == "kalshi_na_poly_eu" and _is_na_hours_ok:
            try:
                use_poly_source = not _is_na_hours_fn(event_base)
            except Exception:
                use_poly_source = True
        else:
            # Unknown / missing config: skip refresh — we have no opinion on
            # source, so don't write probabilities we can't justify.
            _write_audit("G2", cur_p2_a, cur_p2_b,
                         f"no_data_source(prefix={_prefix},bo5={is_bo5})")
            continue

        # Per-leg evaluation queue.
        # Tuple: (leg_num, poly_leg_key, m_prev_prev_markets, m_prev_markets,
        #         m_target_markets, cur_p_a, cur_p_b, csv_col)
        # m_prev_prev is used only for G4 (needs m2 finalized as the gate
        # for "G3 ongoing"); None for G2/G3.
        #
        # G4 is NEVER refreshed. The initial p4 = avg(p1, p2, p3) captured at
        # populate time is the permanent value. Live-refreshing G4 from Poly
        # mid bit us on 4IKIPRAN (2026-06-23): G3 had already settled on Poly
        # but Kalshi m3 was still "active", so the m_prev_finalized gate didn't
        # fire, and we overwrote p4 (and manual operator overrides) every cycle
        # with the live Poly G4 mid while G4 was actively playing. _evaluate_leg
        # leg=4 code is preserved for reference but never invoked.
        legs = [(2, "game2", None, m1_markets, m2_markets, cur_p2_a, cur_p2_b, 2)]
        if is_bo5:
            if cur_p3_a is not None:
                legs.append((3, "game3", None, m2_markets, m3_markets, cur_p3_a, cur_p3_b, 6))

        for leg_num, poly_key, m_prev_prev, m_prev, m_target, cur_a, cur_b, csv_col in legs:
            result = _evaluate_leg(
                leg_num=leg_num,
                event_base=event_base,
                ticker_a=ticker_a, ticker_b=ticker_b,
                sub_a=sub_a, sub_b=sub_b,
                suffix_a=suffix_a, suffix_b=suffix_b,
                m_prev_prev_markets=m_prev_prev,
                m_prev_markets=m_prev,
                m_target_markets=m_target,
                poly_leg_key=poly_key,
                cur_p_a=cur_a, cur_p_b=cur_b,
                use_poly_source=use_poly_source,
            )

            audit_w.writerow([
                ts_str, event_base, ticker_a, ticker_b, result["leg"],
                result["cur_p_a"], result["cur_p_b"], result["new_p_a"], result["new_p_b"],
                result["m_target_bid"], result["m_target_ask"], result["g_spread"],
                result["m_prev_bid_a"], result["m_prev_bid_b"],
                result["m_prev_finalized"], result["gate_passed"], result["skip_reason"], mode
            ])

            if result["gate_passed"] and live_on and result["new_p_a"] != "" and result["new_p_b"] != "":
                # p3-preservation (26AUG11): BO3 leg-2 writes counter-move p1
                # so the derived decider continuation (p1+p2)/2 holds. MUST
                # run BEFORE the p2 overwrite below — it reads rows[0][2].
                # BO5 carries p3 explicitly in col 6, so no counter-move.
                # Kill switch: disable_p1_preserve.flag (no restart needed).
                if (leg_num == 2 and not is_bo5
                        and not os.path.exists("disable_p1_preserve.flag")):
                    try:
                        _p1a = float(rows[0][1])
                        _p2a_old = float(rows[0][2])
                        _p2a_new = float(result["new_p_a"])
                        _p1a_new = _p3_preserving_p1(_p1a, _p2a_old, _p2a_new)
                        _p3_pre = (_p1a + _p2a_old) / 2.0
                        _p3_post = (_p1a_new + _p2a_new) / 2.0
                        if abs(_p3_post - _p3_pre) > 1e-9:
                            print(f"[G2-REFRESH] {event_base}: p1 counter-move CLAMPED "
                                  f"(p1 {_p1a:.3f}->{_p1a_new:.3f}, p2 {_p2a_old:.3f}->"
                                  f"{_p2a_new:.3f}) — p3 shifts {_p3_pre:.4f}->{_p3_post:.4f}")
                        if abs(_p1a_new - _p1a) >= 0.0005:
                            rows[0][1] = f"{_p1a_new:.4f}"
                            rows[1][1] = f"{1.0 - _p1a_new:.4f}"
                            print(f"[G2-REFRESH] {event_base}: p3-preserve — p2 "
                                  f"{_p2a_old:.3f}->{_p2a_new:.3f}, p1 {_p1a:.3f}->"
                                  f"{_p1a_new:.4f}, p3 held {_p3_pre:.4f}")
                    except (ValueError, IndexError) as _e:
                        print(f"[G2-REFRESH] {event_base}: p3-preserve SKIPPED "
                              f"({_e!r}) — p2 write proceeds, p3 will drift")
                rows[0][csv_col] = str(result["new_p_a"])
                rows[1][csv_col] = str(result["new_p_b"])
                refreshed += 1

    audit_f.close()
    if live_on:
        print(f"[G2-REFRESH] evaluated {evaluated} events; refreshed {refreshed} leg(s) LIVE; "
              f"skipped {skipped_scope} LoL/Dota event(s) by scope; audit → {audit_path}")
    else:
        print(f"[G2-REFRESH] evaluated {evaluated} events; shadow-mode (no writes); "
              f"skipped {skipped_scope} LoL/Dota event(s) by scope; audit → {audit_path}")


def run_populator():
    print("="*65)
    print("  [DAEMON] POPULATE QUOTER CONFIGS")
    print("="*65)

    # Pick up ticker_aliases.csv edits without a daemon restart. The 2026-06-21
    # FALFURIA wipe happened because populate ran with stale FALFUR alias state
    # after the synthetic event was renamed mid-session — the synth bypass
    # didn't recognize the event_base and emitted zero rows.
    if ticker_aliases.reload_if_changed():
        print(f"[DAEMON] ticker_aliases.csv reloaded — "
              f"{len(list(ticker_aliases.all_pairs()))} alias pair(s) in effect")

    # Load pre-verified Polymarket mappings
    poly_mappings = _load_poly_mappings()

    force_patterns = _load_force_start_patterns()
    if force_patterns:
        print(f"[FORCE-START] Active patterns: {force_patterns}")

    # Clock guard (26AUG15 FNCSK postmortem): the pregame time buffer is only
    # trustworthy if the local clock is. Verify against network time each
    # cycle; on skew >180s (or unverifiable), FAIL OPEN — admit every mapped
    # event regardless of scheduled start (PREGAME-CFG caps still apply, so
    # far-future events just sit at min_edge=999). Missing a live game is far
    # worse than carrying extra pregame rows.
    _skew = _clock_skew_seconds()
    clock_ok = _skew is not None and abs(_skew) < 180
    if _skew is None:
        print("[CLOCK-GUARD] WARNING: could not verify system clock against "
              "network time this cycle — pregame time buffer DISABLED (fail-open)")
    elif not clock_ok:
        print(f"[CLOCK-GUARD] *** SYSTEM CLOCK SKEW {_skew:+.0f}s vs network "
              f"time — pregame time buffer DISABLED (fail-open), FIX NTP ***")
    else:
        # Heartbeat: a guard whose healthy state is silence is itself a
        # silent-failure risk — one line per cycle proves it ran.
        print(f"[CLOCK-GUARD] ok, skew {_skew:+.0f}s")
    
    # 1. Load Parameters Blueprint
    param_maps = {}
    csv_header = []
    
    with open("market_parameters.csv", "r") as f:
        reader = csv.reader(f)
        try:
            csv_header = next(reader)
        except StopIteration:
            pass
            
        for row in reader:
            if not row or not row[0]: continue
            prefix = row[1] # market_prefix
            if prefix not in param_maps:
                param_maps[prefix] = []
            param_maps[prefix].append(row)
            
    print(f"Loaded {sum(len(v) for v in param_maps.values())} parameter templates across {len(param_maps)} prefixes.")

    # Load the Kalshi metadata cache once per cycle. Mutations within the
    # cycle write back via _save_kalshi_metadata_cache as new events are
    # discovered (incremental persistence so a daemon kill mid-cycle never
    # loses learned metadata).
    _kalshi_metadata_cache = _load_kalshi_metadata_cache()
    if _kalshi_metadata_cache:
        print(f"[METADATA-CACHE] loaded {len(_kalshi_metadata_cache)} cached event(s)")
    
    # 2. Boot native MLB tracker to verify physical game status
    mlb_tracker = get_instance()
    time.sleep(2.0) # Buffer to allow WebSocket & API alignment
    
    # 3. Read Daily Markets
    final_output_rows = []
    # Spread penalty: persisted per-event in column 7 of esports_probabilities.csv.
    # Loaded from col 7 at startup (~line 1508). Capture loop only fires for NEW
    # events (sm_ticker not in existing_probs), so existing rows never have their
    # penalty recomputed. To override a specific event's penalty, either edit
    # column 7 in esports_probabilities.csv directly OR use arb_overrides.csv
    # (which runs AFTER the penalty pass per the 2026-05-27 order swap).
    spread_penalties = {}  # event_base -> extra edge in cents from wide map spreads
    settled_urls = set()   # URLs of events confirmed settled — removed from parsed_markets at end
    # Track event_bases that hit a per-prefix Kalshi API failure during template
    # emission. Used at write time to preserve their prior template_quoter_config
    # rows (preventing the "bot drops all rows mid-game on transient API blip"
    # failure mode where hot-reload cancels active trades for 1-3 min).
    events_with_api_failure: set[str] = set()

    existing_probs = set()
    valid_rows = []  # Rows that pass validation (rewritten to CSV)
    prob_file = "esports_probabilities.csv"
    if os.path.exists(prob_file):
        with open(prob_file, "r") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
            for row in reader:
                if not row:
                    continue
                ticker = row[0].strip()
                try:
                    p1 = float(row[1])
                    p2 = float(row[2])
                    series = float(row[3])
                except (ValueError, IndexError):
                    continue

                # Reject hallucinated 50/50 entries on load (unless manually verified)
                verified = len(row) >= 9 and row[8].strip().upper() == "VERIFIED"
                if not verified and abs(p1 - 0.5) <= 0.005 and abs(p2 - 0.5) <= 0.005 and abs(series - 0.5) <= 0.005:
                    print(f"  🚨 PURGING hallucinated entry on load: {ticker} p1={p1} p2={p2} series={series}")
                    continue  # Don't add to existing_probs — will be re-captured

                # Backfill gate: if this row is a BO5 entry (col 6 = p3 non-empty)
                # but is missing p4 in col 9, drop it from existing_probs so the
                # next capture cycle re-processes it and writes p4/p5 from Poly
                # G4/G5 winner markets. Rows captured pre-Stage-1 (2026-05-17)
                # only have 8 columns and need this backfill to pick up p4.
                is_bo5_entry = len(row) > 6 and row[6].strip() != ""
                has_p4 = len(row) > 9 and row[9].strip() != ""
                if is_bo5_entry and not has_p4:
                    print(f"  [BO5 BACKFILL] {ticker}: missing p4 in row, dropping to force re-capture with Poly G4")
                    continue

                existing_probs.add(ticker)
                valid_rows.append(row)
                # Load spread penalty from column 8 if present
                if len(row) >= 8 and row[7].strip():
                    try:
                        penalty = float(row[7])
                        if penalty > 0:
                            # Extract event_base from ticker (drop team suffix)
                            parts = ticker.split("-")
                            if len(parts) >= 3:
                                event_base_key = "-".join(parts[:2])
                                spread_penalties[event_base_key] = max(spread_penalties.get(event_base_key, 0), penalty)
                    except ValueError:
                        pass

        # G2 refresh pass — refresh p2 in valid_rows in place (if flagged).
        # Runs BEFORE the rewrite so any p2 updates persist in the same write.
        # No-op unless enable_p2_refresh_shadow.flag or enable_p2_refresh_live.flag exists.
        _refresh_g2_probabilities(valid_rows, poly_mappings, _kalshi_metadata_cache)

        # Rewrite CSV without purged entries
        if len(valid_rows) < len(valid_rows) + 1:  # always rewrite to clean up
            with open(prob_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header_row if header_row else ["ticker", "p1_start", "p2_start", "series_start", "g2_momentum", "g3_momentum"])
                writer.writerows(valid_rows)

    new_probs_rows = []
    
    # Per-event capture-buffer registry keyed by event_base (uppercase, no scheme).
    # Filled while reading parsed_markets.csv; consumed below by the esports
    # capture gate and threaded into template_quoter_config rows at write time.
    _buffer_by_event: dict[str, float] = {}

    with open("parsed_markets.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        parsed_rows = [row for row in reader if row and row[0]]
    # Sort by event start time (embedded in ticker as YYMMMDD HHMM)
    def _sort_key(row):
        url = row[0].strip().strip("/").split("/")[-1].upper()
        parts = url.split("-", 1)
        return parts[1][:11] if len(parts) >= 2 and len(parts[1]) >= 11 else ""
    parsed_rows.sort(key=_sort_key)

    # Populate buffer registry. Default 120 when column missing (back-compat
    # with manually-edited rows from before this column was added).
    for row in parsed_rows:
        url = (row[0] if row else "").strip()
        if not url:
            continue
        eb = url.strip("/").split("/")[-1].upper()
        try:
            buf = float(row[1]) if len(row) > 1 and str(row[1]).strip() else 120.0
        except (ValueError, TypeError):
            buf = 120.0
        _buffer_by_event[eb] = buf

    # Per-event Poly start-time registry. Sourced from poly_parsed_markets.csv
    # `poly_start_time_et` column (written by refresh_tomorrow_markets via
    # Poly's gamma `startTime` field). Consumed by the esports capture gate
    # below; precedence is force-start > Poly time > Kalshi ticker time.
    # Empty when refresh_tomorrow hasn't populated the column or the event
    # has no Poly mapping. Added 2026-06-14 after Kalshi mislisted THLEV
    # start by 3h and the bot missed the entire BO3.
    _poly_start_by_event: dict[str, datetime] = {}
    try:
        with open("poly_parsed_markets.csv", "r") as f:
            for r in csv.DictReader(f):
                eb = (r.get("kalshi_event") or "").strip().upper()
                ts = (r.get("poly_start_time_et") or "").strip()
                if eb and ts:
                    try:
                        _poly_start_by_event[eb] = datetime.fromisoformat(ts)
                    except Exception:
                        pass
    except FileNotFoundError:
        pass

    for row in parsed_rows:
            url = row[0].strip()
            # Extract '26apr091340detmin' from 'kxmlbgame-26apr091340detmin'
            url_parts = url.strip("/").split("/")
            event_base = url_parts[-1].upper()
            
            # Simple assumption: Kalshi IDs have a hyphen before the daily tag
            if "-" not in event_base:
                continue
            tail_tag = event_base.split("-", 1)[1]
            is_bo5 = False  # default; set True below for esports series events when m3 markets exist

            # If this is MLB, enforce Live status natively
            if "MLB" in event_base:
                team_hint = None
                for abbr in KALSHI_TO_MLB.keys():
                    if abbr in tail_tag:
                        team_hint = abbr
                        break
                        
                if team_hint:
                    gs = mlb_tracker.get(team_hint)
                    if not gs or gs.get("status") != "Live":
                        print(f"Skipping {event_base} - Game is NOT currently live! Status: {gs.get('status') if gs else 'Unknown'}")
                        continue
                else:
                    print(f"Skipping {event_base} - Cannot parse MLB team hint.")
                    continue
            elif "MLS" in event_base:
                # MLS functionally omits the timestamp from the ticker (e.g. 26APR12CLBORL lacked time chunk)
                pass
            elif any(p in event_base for p in ["KXLOL", "KXCS2", "KXVALORANT", "KXDOTA2", "KXCOD", "KXATP", "KXWTA"]):
                # Esports capture gate: only capture probabilities once we're
                # within `capture_buffer_min` of scheduled start. Per-event
                # value comes from parsed_markets.csv (set by refresh_tomorrow_markets
                # via league keyword matching). Tight buffer (e.g. 20 for LCK) =
                # capture close to tip → fresh probabilities. Wide buffer
                # (e.g. 120 for LCS / VCT Americas) = catch rolling-schedule
                # slippage where actual start lags scheduled by 30-60 min.
                try:
                    buffer_min = _buffer_by_event.get(event_base, 120.0)
                    eastern = ZoneInfo('America/New_York')
                    forced = any(p in event_base for p in force_patterns)
                    # Source precedence: force-start > poly_parsed_markets
                    # start > Kalshi ticker. Poly preferred because Kalshi
                    # has been wrong on marquee listings (THLEV 2026-06-14:
                    # ticker said 13:00 ET, real start was 10:00 ET; bot
                    # missed the BO3 waiting for the listed time).
                    poly_dt = _poly_start_by_event.get(event_base)
                    if poly_dt:
                        scheduled_dt = poly_dt
                        time_src = "poly"
                    else:
                        # Aliased (synthetic) events carry the MAP anchor date in
                        # their ticker (e.g. 26JUL02), not the real game date, so
                        # the Kalshi-ticker start time would always be in the past
                        # → the event would never wait for its real start. When the
                        # time source is the Kalshi ticker, resolve synthetic →
                        # real so the start time reflects the actual game date.
                        _time_tail = tail_tag
                        if ticker_aliases.event_base_is_synthetic(event_base):
                            _real_eb = _real_series_event_base(event_base)
                            if "-" in _real_eb:
                                _time_tail = _real_eb.split("-", 1)[1]
                        es_time_string = _time_tail[:11]
                        scheduled_dt = datetime.strptime(es_time_string, "%y%b%d%H%M").replace(tzinfo=eastern)
                        time_src = "kalshi"
                    current_dt = datetime.now(eastern)
                    minutes_until = (scheduled_dt - current_dt).total_seconds() / 60.0
                    if minutes_until > buffer_min and not forced and clock_ok:
                        print(f"Skipping {event_base} - Esports event is {minutes_until:.0f} min away (>{buffer_min:.0f} min buffer, src={time_src})")
                        continue
                    if (forced or not clock_ok) and minutes_until > buffer_min:
                        print(f"[FORCE-START] Bypassing {buffer_min:.0f}-min buffer for {event_base} ({minutes_until:.0f} min away, src={time_src}"
                              f"{', clock-guard fail-open' if not clock_ok else ''})")
                except Exception:
                    # If time parsing fails, fall through to liquidity checks below
                    pass
            else:
                # Universal Start Time Pre-Market Enforcement for non-MLB / non-Esports
                time_string = tail_tag[:11]
                try:
                    eastern = ZoneInfo('America/New_York')
                    scheduled_dt = datetime.strptime(time_string, "%y%b%d%H%M").replace(tzinfo=eastern)
                    current_dt = datetime.now(eastern)
                    
                    if current_dt < scheduled_dt:
                        diff_minutes = int((scheduled_dt - current_dt).total_seconds() / 60)
                        print(f"Skipping {event_base} - Event has not started! Timer indicates {diff_minutes} minutes remaining.")
                        continue
                except Exception as e:
                    print(f"Skipping {event_base} - Failed to parse native timestamp from {time_string}. Error: {e}")
                    continue
            
            # Game is live or heavily liquid! Cross-pollinate sub-markets for EVERY matched prefix in our parameters.
            
            # Probabilities Baseline Capture
            # All esports (LoL/Valorant/CS2): capture probabilities immediately
            # Map picks are determined pre-game, so pre-game baselines are sufficient
            cs2_prob_delay_ok = True

            is_series_event = "GAME" in event_base or "MATCH" in event_base
            # Set to True only after is_bo5 is resolved inside the series-API
            # block below. If the series API call (or any step before is_bo5
            # resolution) fails, this stays False and the post-block fallback
            # re-resolves is_bo5 from poly_title + a standalone m3 probe so
            # template selection survives transient API failures (KCG2 incident
            # 2026-05-16).
            is_bo5_resolved = False
            s_markets = None  # initialized so later VCL/VCT classification doesn't NameError if series API fails
            if cs2_prob_delay_ok and is_series_event and any(p in event_base for p in ["KXLOL", "KXCS2", "KXVALORANT", "KXDOTA2", "KXCOD", "KXATP", "KXWTA"]):
                parts = event_base.split("-")
                if len(parts) >= 2:
                    # ─── Metadata cache fast path ─────────────────────────
                    # If we've cached yes_sub_title + rules_primary for this
                    # event's tickers (from a prior cycle's REST call), we
                    # can build s_markets skeleton from the cache and fill
                    # in status/yes_bid/yes_ask from the WS probe below —
                    # no series REST call needed.
                    #
                    # Sentinel: _s_from_cache distinguishes the two paths
                    # so the WS-probe block downstream knows whether to
                    # *fill in* missing status/bid fields (cache path) or
                    # leave s_markets alone (REST path, which already has
                    # them).
                    _cached_meta = (_kalshi_metadata_cache.get(event_base) or {}).get("tickers") or []
                    _s_from_cache = False
                    if _cached_meta:
                        s_markets = [
                            {
                                "ticker": e.get("ticker", ""),
                                "yes_sub_title": e.get("yes_sub_title", "") or "",
                                "rules_primary": e.get("rules_primary", "") or "",
                            }
                            for e in _cached_meta
                            if e.get("ticker")
                        ]
                        _s_from_cache = bool(s_markets)
                        if _s_from_cache:
                            print(f"  [METADATA-CACHE-HIT] {event_base}: {len(s_markets)} tickers from cache, skipping series REST")

                    # ─── Cache miss → series REST (only path that hits /markets?) ──
                    if not _s_from_cache:
                        def _series_rest(evt):
                            _r = _get_with_retry(
                                f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={evt}",
                                headers={"Accept": "application/json"}, timeout=5)
                            time.sleep(0.1)
                            if _r.status_code == 200:
                                return _r.json().get("markets", []), 200
                            return [], _r.status_code
                        try:
                            # Try the (possibly synthetic) event ticker first.
                            s_markets, _sc = _series_rest(event_base)
                            # Aliased events: the synthetic event ticker doesn't exist
                            # on Kalshi, so the fetch above is empty. Fall back to the
                            # REAL series event ticker (via ticker_aliases) and re-key
                            # the returned markets back to synthetic so all downstream
                            # keying (prob rows, team alignment) stays on internal names.
                            if not s_markets and ticker_aliases.event_base_is_synthetic(event_base):
                                _real_eb = _real_series_event_base(event_base)
                                if _real_eb and _real_eb != event_base:
                                    _real_markets, _sc = _series_rest(_real_eb)
                                    for _sm in _real_markets:
                                        _syn_t = ticker_aliases.reverse(_sm.get("ticker", ""))
                                        if _syn_t:
                                            _sm["ticker"] = _syn_t
                                    s_markets = _real_markets
                                    if s_markets:
                                        print(f"  [ALIAS-SERIES] {event_base}: fetched real series {_real_eb}, re-keyed {len(s_markets)} tickers → synthetic")
                            if not s_markets and _sc != 200:
                                print(f"Skipping {event_base} - Series API returned {_sc}")
                            # Persist metadata for next cycle — write-once-per-event
                            # since yes_sub_title and rules_primary are static.
                            if s_markets:
                                _kalshi_metadata_cache[event_base] = {
                                    "tickers": [
                                        {
                                            "ticker": sm.get("ticker", ""),
                                            "yes_sub_title": (sm.get("yes_sub_title") or "").strip(),
                                            "rules_primary": (sm.get("rules_primary") or "").strip(),
                                        }
                                        for sm in s_markets
                                        if sm.get("ticker")
                                    ]
                                }
                                _save_kalshi_metadata_cache(_kalshi_metadata_cache)
                                print(f"  [METADATA-CACHE-MISS] {event_base}: cached {len(s_markets)} tickers for next cycle")
                        except Exception as e:
                            print(f"  [SERIES-REST-ERR] {event_base}: {type(e).__name__}: {e}")

                    # all_settled check moved below the WS-probe block so the
                    # cache-hit path can populate status from WS results first.
                    # The REST-path s_markets already has status — the check
                    # works identically once both paths reach the post-probe
                    # synchronization step.
                    #
                    # The outer try/except mirrors the original structure
                    # (orphan except at the bottom of this block catches any
                    # downstream Polymarket/baseline-capture errors so a
                    # single misbehaving event can't crash the cycle).
                    try:
                        if s_markets:

                            # ─── Per-event WS probe ───────────────────────────
                            # One batched subscribe covers every map sub-ticker
                            # we'd otherwise REST-fetch in `fetch_map_markets`
                            # below (5 calls/event) AND every per-prefix derived
                            # event ticker at line ~2788 (6-10 calls/event).
                            # Eliminates the bulk of the populate cycle's
                            # `/markets?event_ticker=` traffic.
                            #
                            # Team suffixes come from s_markets (just fetched).
                            # We probe all map-N sub-tickers (1-5) plus the
                            # per-prefix derived tickers that the per-prefix
                            # loop below will need.
                            _team_suffixes = sorted({
                                (sm.get("ticker", "") or "").rsplit("-", 1)[-1]
                                for sm in s_markets
                                if "-" in (sm.get("ticker", "") or "")
                            }) or []
                            _candidate_tickers: list[str] = []
                            if _team_suffixes:
                                if "MATCH" in event_base:
                                    _map_base_for_probe = event_base.replace("MATCH", "SETWINNER")
                                else:
                                    _map_base_for_probe = event_base.replace("GAME", "MAP")
                                _probe_max_n = _max_map_n(event_base, poly_mappings)
                                for _n in range(1, _probe_max_n + 1):
                                    for _ts in _team_suffixes:
                                        _candidate_tickers.append(f"{_map_base_for_probe}-{_n}-{_ts}")
                                # Per-prefix loop candidates (matching prefixes by
                                # sport-root rule mirrored from line 2721-2724).
                                import re as _re_probe
                                _event_root_m = _re_probe.match(
                                    r'(KX[A-Z0-9]+?)(?:GAME|MAP|MATCH|SETWINNER|TOTAL|SPREAD|F5)',
                                    event_base.split("-")[0],
                                )
                                _event_root = _event_root_m.group(1) if _event_root_m else None
                                _tail_for_probe = event_base.split("-", 1)[1] if "-" in event_base else ""
                                if _event_root and _tail_for_probe:
                                    for _prefix in param_maps.keys():
                                        _prefix_root_m = _re_probe.match(
                                            r'(KX[A-Z0-9]+?)(?:GAME|MAP|MATCH|SETWINNER|TOTAL|SPREAD|F5)',
                                            _prefix,
                                        )
                                        if not _prefix_root_m or _prefix_root_m.group(1) != _event_root:
                                            continue
                                        _pu_probe = _prefix.upper()
                                        # Three prefix families exist:
                                        #   - team-suffix (GAME, MATCH):
                                        #       children = {prefix}-{tail}-{team}
                                        #   - per-map + team (MAP, SETWINNER):
                                        #       children = {prefix}-{tail}-{N}-{team}
                                        #   - threshold (TOTALMAPS, TOTAL,
                                        #     SPREAD, F5, MOV, MOF):
                                        #       children = {prefix}-{tail}-{N}
                                        #     (N is a number, NOT a team).
                                        # Probing the threshold family with
                                        # team suffixes (bug pre-2026-06-25)
                                        # synthesized bogus tickers like
                                        # KXCS2TOTALMAPS-{tail}-AM that don't
                                        # exist on Kalshi — and ended up in
                                        # template_quoter_config.csv when the
                                        # WS probe response was misclassified.
                                        # Skip threshold prefixes here; the
                                        # outer per-prefix REST loop already
                                        # handles them via /markets?event_ticker=.
                                        if _pu_probe.endswith("MAP") or "SETWINNER" in _pu_probe:
                                            for _n in range(1, _probe_max_n + 1):
                                                for _ts in _team_suffixes:
                                                    _candidate_tickers.append(f"{_prefix}-{_tail_for_probe}-{_n}-{_ts}")
                                        elif _pu_probe.endswith("GAME") or "MATCH" in _pu_probe:
                                            for _ts in _team_suffixes:
                                                _candidate_tickers.append(f"{_prefix}-{_tail_for_probe}-{_ts}")
                                        # else: threshold prefix — let REST handle it
                            _candidate_tickers = sorted(set(_candidate_tickers))
                            _per_event_ws_cache: dict = {}
                            if _candidate_tickers:
                                try:
                                    _t_probe = time.time()
                                    _per_event_ws_cache = _ws_probe_markets(
                                        _candidate_tickers, timeout_s=4.0,
                                    )
                                    _n_active = sum(
                                        1 for s in _per_event_ws_cache.values()
                                        if s.active is True
                                    )
                                    _n_final = sum(
                                        1 for s in _per_event_ws_cache.values()
                                        if s.active is False
                                    )
                                    print(
                                        f"  [WS-PROBE] {event_base}: "
                                        f"{len(_candidate_tickers)} probed → "
                                        f"{_n_active} active, {_n_final} finalized "
                                        f"in {time.time() - _t_probe:.2f}s"
                                    )
                                except Exception as _e:
                                    print(f"  [WS-PROBE] {event_base} failed: "
                                          f"{type(_e).__name__}: {_e}")
                                    _per_event_ws_cache = {}

                            # If we got s_markets from the metadata cache, the
                            # entries have ticker/yes_sub_title/rules_primary
                            # but no status/yes_bid/yes_ask. Fill those in from
                            # the WS probe results so downstream all_settled
                            # check and bid-based gates work identically to the
                            # REST path.
                            if _s_from_cache and _per_event_ws_cache:
                                _unfilled = 0
                                for sm in s_markets:
                                    _t = sm.get("ticker", "")
                                    _ms = _per_event_ws_cache.get(_t)
                                    if _ms is None or _ms.active is None:
                                        _unfilled += 1
                                        continue
                                    if _ms.active is False:
                                        sm["status"] = "finalized"
                                    else:
                                        sm["status"] = "active"
                                        if _ms.yes_top_bid_c is not None:
                                            sm["yes_bid_dollars"] = f"{_ms.yes_top_bid_c / 100:.4f}"
                                        if _ms.no_top_bid_c is not None:
                                            sm["yes_ask_dollars"] = f"{(100 - _ms.no_top_bid_c) / 100:.4f}"
                                if _unfilled:
                                    # WS didn't classify every cached ticker —
                                    # safer to refetch from REST this cycle than
                                    # to proceed with partial data. Falls
                                    # through to the cache-miss branch below.
                                    print(f"  [METADATA-CACHE-PARTIAL] {event_base}: "
                                          f"{_unfilled}/{len(s_markets)} unclassified, falling back to REST")
                                    _fallback_url = f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={event_base}"
                                    try:
                                        _fr = _get_with_retry(_fallback_url, headers={"Accept": "application/json"}, timeout=5)
                                        time.sleep(0.1)
                                        if _fr.status_code == 200:
                                            s_markets = _fr.json().get("markets", [])
                                            # Re-cache (the existing cache may be stale if a ticker was added/dropped)
                                            _kalshi_metadata_cache[event_base] = {
                                                "tickers": [
                                                    {
                                                        "ticker": sm.get("ticker", ""),
                                                        "yes_sub_title": (sm.get("yes_sub_title") or "").strip(),
                                                        "rules_primary": (sm.get("rules_primary") or "").strip(),
                                                    }
                                                    for sm in s_markets
                                                    if sm.get("ticker")
                                                ]
                                            }
                                            _save_kalshi_metadata_cache(_kalshi_metadata_cache)
                                    except Exception as _fe:
                                        print(f"  [SERIES-REST-ERR] {event_base}: "
                                              f"{type(_fe).__name__}: {_fe}")

                            # all_settled check — runs after status has been
                            # set (from REST or WS). Same semantics as before,
                            # just relocated below the probe so both paths
                            # converge here.
                            all_settled = s_markets and all(
                                m.get("status") in ("determined", "settled", "closed", "finalized")
                                for m in s_markets
                            )
                            if all_settled:
                                print(f"  [SETTLED] {event_base} — all markets settled, removing from parsed_markets")
                                settled_urls.add(url)
                                continue

                            def _ws_event_markets(probe_event_ticker):
                                """Construct the same shape Kalshi REST returns
                                (list of market dicts) using the per-event WS
                                cache. Returns (markets_list, "ws_hit") if every
                                expected sub-ticker is classified, else
                                (None, "ws_miss") so the caller falls back to
                                REST.

                                Children are team-suffixed for the GAME/MATCH
                                (series-level) and MAP/SETWINNER (per-map)
                                families. The caller passes probe_event_ticker
                                at the right level for each — e.g.
                                  KXCS2GAME-{tail}      → children KX...-{team}
                                  KXCS2MAP-{tail}-{N}   → children KX...-N-{team}
                                so synthesizing {probe}-{team} works for BOTH.

                                Threshold-style prefixes (TOTALMAPS, TOTAL,
                                SPREAD, F5, MOV, MOF) use numeric suffixes,
                                not team suffixes; synthesizing them this way
                                produces bogus tickers. Return ws_miss for
                                those — the per-prefix REST loop handles them."""
                                if not _team_suffixes or not _per_event_ws_cache:
                                    return None, "ws_miss"
                                _probe_root = probe_event_ticker.split("-", 1)[0].upper()
                                _is_team_suffix_family = (
                                    _probe_root.endswith("GAME")
                                    or _probe_root.endswith("MAP")
                                    or "MATCH" in _probe_root
                                    or "SETWINNER" in _probe_root
                                )
                                # endswith("MAP") would also catch TOTALMAPS,
                                # which is a threshold prefix. Exclude it
                                # explicitly (and any other ...MAPS variants).
                                if _probe_root.endswith("MAPS"):
                                    _is_team_suffix_family = False
                                if not _is_team_suffix_family:
                                    return None, "ws_miss"
                                expected = [f"{probe_event_ticker}-{ts}" for ts in _team_suffixes]
                                out = []
                                for t in expected:
                                    ms = _per_event_ws_cache.get(t)
                                    if ms is None or ms.active is None:
                                        return None, "ws_miss"
                                    if ms.active is False:
                                        # finalized — still include so caller's
                                        # status-based filtering sees it. yes_bid
                                        # omitted (no book) — downstream tolerates.
                                        out.append({"ticker": t, "status": "finalized"})
                                    else:
                                        m = {"ticker": t, "status": "active"}
                                        if ms.yes_top_bid_c is not None:
                                            m["yes_bid_dollars"] = f"{ms.yes_top_bid_c / 100:.4f}"
                                        if ms.no_top_bid_c is not None:
                                            m["yes_ask_dollars"] = f"{(100 - ms.no_top_bid_c) / 100:.4f}"
                                        out.append(m)
                                return out, "ws_hit"

                            def fetch_map_markets(map_event):
                                # Try the cycle's WS cache first; only hit REST
                                # if a candidate ticker wasn't classified.
                                _out, _src = _ws_event_markets(map_event)
                                if _src == "ws_hit":
                                    return _out
                                for attempt in range(2):
                                    try:
                                        r = _get_with_retry(f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={map_event}", headers={"Accept": "application/json"}, timeout=5)
                                        if r.status_code == 200:
                                            return r.json().get("markets", [])
                                    except Exception:
                                        pass
                                    if attempt == 0:
                                        time.sleep(0.5)
                                return []

                            if "MATCH" in event_base:
                                map1_event = event_base.replace("MATCH", "SETWINNER") + "-1"
                                map2_event = event_base.replace("MATCH", "SETWINNER") + "-2"
                            else:
                                map1_event = event_base.replace("GAME", "MAP") + "-1"
                                map2_event = event_base.replace("GAME", "MAP") + "-2"
                            m1_markets = fetch_map_markets(map1_event)
                            m2_markets = fetch_map_markets(map2_event)

                            # Fetch m3, m4, m5 markets — m3 used both as the Bo5 detection
                            # fallback (below) and to capture baseline p3. m4 and m5 are used
                            # for baseline p4 / p5 capture so the BO5 hedge math can plug in
                            # market-priced future-game probabilities (Poly typically exposes
                            # G4 for ~65% of BO5 events and G5 for ~45%). Kalshi sometimes
                            # has these map markets too; best_prob_for_leg routes through
                            # whichever side has tighter spread.
                            if "MATCH" in event_base:
                                map3_event = event_base.replace("MATCH", "SETWINNER") + "-3"
                                map4_event = event_base.replace("MATCH", "SETWINNER") + "-4"
                                map5_event = event_base.replace("MATCH", "SETWINNER") + "-5"
                            else:
                                map3_event = event_base.replace("GAME", "MAP") + "-3"
                                map4_event = event_base.replace("GAME", "MAP") + "-4"
                                map5_event = event_base.replace("GAME", "MAP") + "-5"
                            # m3 fetch gated by poly format:
                            #   BO3 confirmed → SKIP (Kalshi creates m3 only
                            #     when match goes to decider; querying it for
                            #     a 2-0 BO3 returns 0 markets and burns the
                            #     /markets? bucket every cycle for hours).
                            #   BO5 or unknown → fetch (need m3 for either
                            #     baseline p3 or the BO5 detection fallback).
                            # m4/m5 only fetched when poly already confirms BO5.
                            _early_max_n = _max_map_n(event_base, poly_mappings)
                            m3_markets = fetch_map_markets(map3_event) if _early_max_n >= 3 else []
                            m4_markets = fetch_map_markets(map4_event) if _early_max_n >= 4 else []
                            m5_markets = fetch_map_markets(map5_event) if _early_max_n >= 5 else []
                            is_bo5 = False  # resolved below from poly_title with m3 fallback

                            # Fetch Polymarket data: use pre-verified mapping first,
                            # fall back to dynamic matching only if not in mapping.
                            poly_event = None
                            poly_winners = {}
                            poly_src = "none"

                            mapping = poly_mappings.get(event_base)
                            if mapping is not None:
                                if mapping["poly_slug"]:
                                    # Pre-verified Poly match — fetch by exact slug
                                    poly_event = _fetch_poly_event_by_slug(mapping["poly_slug"])
                                    if poly_event:
                                        poly_winners = _get_poly_winner_markets(poly_event)
                                        poly_src = "mapped"
                                        print(f"  [POLY] Using pre-verified mapping: {mapping['poly_title']}")
                                    else:
                                        print(f"  [POLY] ⚠ Pre-verified slug '{mapping['poly_slug']}' not found — using Kalshi only")
                                        poly_src = "mapped-miss"
                                else:
                                    # Explicitly mapped as no Poly match — skip entirely
                                    poly_src = "kalshi-only"
                                    print(f"  [POLY] Mapped as Kalshi-only (no Poly match)")
                            else:
                                # Not in mapping — fall back to dynamic matching
                                poly_sport = None
                                for prefix, sport in KALSHI_TO_POLY_SPORT.items():
                                    if prefix in event_base:
                                        poly_sport = sport
                                        break
                                if poly_sport:
                                    poly_events = _fetch_poly_events(poly_sport)
                                    # Pass subtitles so the matcher can use them
                                    # without an extra Kalshi REST round-trip.
                                    sm_subs = []
                                    for sm in s_markets:
                                        sub = (sm.get("yes_sub_title") or "").strip()
                                        if sub and sub not in sm_subs:
                                            sm_subs.append(sub)
                                    poly_event = _match_poly_event(event_base, poly_events, subtitles=sm_subs)
                                    if poly_event:
                                        poly_winners = _get_poly_winner_markets(poly_event)
                                        poly_src = "dynamic"
                                        print(f"  [POLY] Dynamic match: {poly_event.get('title', '?')}")

                            # Bo5 detection — two independent signals from poly_parsed_markets.csv:
                            #   1. poly_title (e.g. "Valorant: X vs Y (BO3) - Tournament") — the
                            #      operator-verified Polymarket question text. Parsed here.
                            #   2. `format` column — the Kalshi has_m3 reading from
                            #      refresh_tomorrow_markets's verification step. Blank when
                            #      refresh short-circuited via the prior cycle's poly_title.
                            # When BOTH are present and AGREE → fast path, no REST.
                            # When BOTH are present and DISAGREE → real mismatch, log it and
                            # fall through to the OR-of-three dispute resolver.
                            # When only ONE is present → trust it.
                            # When NEITHER → fall through to OR-of-three.
                            poly_mapping = poly_mappings.get(event_base) or {}
                            poly_title_verified = poly_mapping.get("poly_title", "")
                            kalshi_fmt = poly_mapping.get("format", "")  # Kalshi-only signal
                            poly_title_says_bo5 = "BO5" in poly_title_verified.upper()
                            poly_title_says_bo3 = "BO3" in poly_title_verified.upper()
                            title_fmt = ""
                            if poly_title_says_bo5 and not poly_title_says_bo3:
                                title_fmt = "BO5"
                            elif poly_title_says_bo3 and not poly_title_says_bo5:
                                title_fmt = "BO3"

                            if kalshi_fmt and title_fmt:
                                if kalshi_fmt == title_fmt:
                                    # Both agree — fastest path.
                                    is_bo5 = (kalshi_fmt == "BO5")
                                else:
                                    # Genuine Kalshi-vs-Poly disagreement. Trust poly_title:
                                    # the operator-verified Polymarket question is the more
                                    # reliable source. Kalshi has_m3 can mislead during BO3
                                    # deciders (m3 spawns mid-series for 2-1 tie-breakers).
                                    # Log the mismatch so the operator can investigate, but
                                    # use poly_title as the answer.
                                    is_bo5 = (title_fmt == "BO5")
                                    m3_exists = len(m3_markets) > 0
                                    poly_has_g4_g5 = ("game4" in poly_winners) or ("game5" in poly_winners)
                                    print(f"  [BO5-MISMATCH] {event_base}: kalshi_fmt={kalshi_fmt}, "
                                          f"poly_title={title_fmt} ('{poly_title_verified[:50]}'); "
                                          f"m3_markets={len(m3_markets)}, poly_g4/g5={poly_has_g4_g5} "
                                          f"→ trusting poly_title, is_bo5={is_bo5}")
                            elif kalshi_fmt:
                                # Only Kalshi signal (poly_title silent or absent). Trust it.
                                is_bo5 = (kalshi_fmt == "BO5")
                            elif title_fmt:
                                # Only Poly signal (refresh short-circuited Kalshi REST). Trust it.
                                is_bo5 = (title_fmt == "BO5")
                            else:
                                # Neither signal — fall back to OR-of-three.
                                m3_exists = len(m3_markets) > 0
                                poly_has_g4_g5 = ("game4" in poly_winners) or ("game5" in poly_winners)
                                is_bo5 = m3_exists or poly_has_g4_g5
                            # COD: all CDL matches are BO5; force-true so the
                            # bot never falls back to BO3 sizing if a signal misses.
                            if "KXCOD" in event_base:
                                is_bo5 = True
                            is_bo5_resolved = True

                            # Collect all team tickers for this event, then process as pairs
                            # to ensure probabilities always sum to 1.0
                            pending_teams = []
                            for sm in s_markets:
                                sm_ticker = sm.get("ticker", "")
                                if sm_ticker not in existing_probs and sm.get("status") in ["active", "open", "unopened"]:
                                    pending_teams.append(sm)

                            for sm in pending_teams:
                                sm_ticker = sm.get("ticker", "")

                                # Extract pure unbiased symmetrical mid-points for Kalshi legs
                                def compute_symmetric_mid(m_array, tm_suffix):
                                    mid_self = None
                                    mid_opp = None
                                    for m in m_array:
                                        bx = m.get("yes_bid_dollars")
                                        ax = m.get("yes_ask_dollars")
                                        if bx is None or ax is None:
                                            continue
                                        mid = (float(bx) + float(ax)) / 2.0
                                        if m.get("ticker", "").endswith(f"-{tm_suffix}"):
                                            mid_self = mid
                                        else:
                                            mid_opp = mid
                                    if mid_self is not None and mid_opp is not None:
                                        return (mid_self + (1.0 - mid_opp)) / 2.0
                                    return mid_self if mid_self is not None else None

                                def kalshi_spread(m_array):
                                    """Max spread in cents across all markets in the array."""
                                    if not m_array:
                                        return 999.0
                                    ms = 0.0
                                    for mx in m_array:
                                        bx = mx.get("yes_bid_dollars")
                                        ax = mx.get("yes_ask_dollars")
                                        if bx is None or ax is None:
                                            return 999.0
                                        ms = max(ms, (float(ax) - float(bx)) * 100)
                                    return ms

                                tm_suffix = sm_ticker.split("-")[-1]
                                # Get full team name from Kalshi yes_sub_title
                                kalshi_team_names = _extract_team_names(s_markets)
                                kalshi_full_name = kalshi_team_names.get(tm_suffix, "")

                                # Per-leg probability: use best spread across Kalshi + Polymarket
                                # Kalshi ≤2c → Kalshi | Poly tighter & ≤8c → Poly | Kalshi ≤8c → Kalshi | else skip
                                def best_prob_for_leg(k_markets_arr, poly_leg_key):
                                    k_spr = kalshi_spread(k_markets_arr)
                                    k_mid = compute_symmetric_mid(k_markets_arr, tm_suffix)

                                    # Try Polymarket — use full team name from yes_sub_title
                                    p_mid = None
                                    p_spr = 999.0
                                    pw = poly_winners.get(poly_leg_key)
                                    if pw:
                                        if pw.get("tokens") and kalshi_full_name:
                                            outcomes = pw.get("outcomes", [])
                                            aligned_idx = _align_team_to_poly_outcome(kalshi_full_name, outcomes, event_base=event_base)
                                            if aligned_idx == -1:
                                                print(f"[PROBS] ALIGNMENT ABORT: Kalshi '{kalshi_full_name}' ({tm_suffix}) "
                                                      f"does not match any Poly outcome: {outcomes} — using Kalshi only")
                                            elif 0 <= aligned_idx < len(pw["tokens"]):
                                                token_id = pw["tokens"][aligned_idx]
                                                # Fetch CLOB midpoint
                                                try:
                                                    mr = _get_with_retry(f"https://clob.polymarket.com/midpoint?token_id={token_id}", timeout=5)
                                                    if mr.status_code == 200:
                                                        v = mr.json().get("mid")
                                                        if v:
                                                            p_mid = float(v)
                                                except Exception:
                                                    pass
                                                # Fetch CLOB spread
                                                spr = _get_poly_spread(token_id)
                                                if spr is not None:
                                                    p_spr = spr
                                        # Fallback to gamma if CLOB failed (still using full name)
                                        if p_mid is None and kalshi_full_name:
                                            p_mid = _align_poly_to_kalshi(tm_suffix, pw, kalshi_full_name, event_base=event_base)

                                    # Tier 1 CS: Poly is source of truth (deep books, tightest
                                    # quotes around the clock). Return Poly whenever it's
                                    # available, ignoring the Kalshi-spread preference.
                                    if classify_cs_tier(event_base, poly_mappings) == "tier1":
                                        if p_mid is not None and p_spr <= 15.0:
                                            return p_mid, "Poly", p_spr

                                    # VCT Valorant: Poly differentiates Map 1 vs Map 2
                                    # (picker advantage); Kalshi market makers often quote
                                    # both maps symmetrically pre-game. Prefer Poly when
                                    # available so the picker-advantage signal isn't lost.
                                    if "KXVALORANT" in event_base and is_vct_event(s_markets):
                                        if p_mid is not None and p_spr <= 15.0:
                                            return p_mid, "Poly", p_spr

                                    best_spread = min(k_spr, p_spr)
                                    if best_spread > 15.0:
                                        return None, "NONE", 999.0  # Both too wide
                                    if k_spr <= 2.0 and k_mid is not None:
                                        return k_mid, "Kalshi", k_spr
                                    if p_mid is not None and p_spr <= k_spr:
                                        return p_mid, "Poly", p_spr
                                    if k_spr <= 15.0 and k_mid is not None:
                                        return k_mid, "Kalshi", k_spr
                                    if p_mid is not None and p_spr <= 15.0:
                                        return p_mid, "Poly", p_spr
                                    return None, "NONE", 999.0

                                def check_kalshi_depth(k_markets_arr, min_lots=100, band_cents=10):
                                    """Check Kalshi has min_lots on both sides within band_cents of mid.

                                    Aggregates across both team books via Team A YES = Team B NO:
                                    bids on the opposite team's NO are bids for our team's YES, and
                                    bids on the opposite team's YES are offers on our team's YES.
                                    """
                                    if not k_markets_arr:
                                        return False
                                    ref = k_markets_arr[0]
                                    bx = ref.get("yes_bid_dollars")
                                    ax = ref.get("yes_ask_dollars")
                                    if bx is None or ax is None:
                                        return False
                                    mid = (float(bx) + float(ax)) / 2.0
                                    band = band_cents / 100.0

                                    bid_depth = 0.0
                                    ask_depth = 0.0
                                    for i, mx in enumerate(k_markets_arr):
                                        try:
                                            r = _get_with_retry(f"https://external-api.kalshi.com/trade-api/v2/markets/{mx['ticker']}/orderbook",
                                                           headers={"Accept": "application/json"}, timeout=3)
                                            if r.status_code != 200:
                                                return False
                                            ob = r.json().get("orderbook_fp", {})
                                        except Exception:
                                            return False

                                        if i == 0:
                                            yes_side = ob.get("yes_dollars", [])
                                            no_side = ob.get("no_dollars", [])
                                        else:
                                            # Opponent ticker: their NO bids = our YES bids,
                                            # their YES bids = our YES asks (sellers of our team).
                                            yes_side = ob.get("no_dollars", [])
                                            no_side = ob.get("yes_dollars", [])

                                        bid_depth += sum(float(pt[1]) for pt in yes_side
                                                         if abs(float(pt[0]) - mid) <= band)
                                        ask_depth += sum(float(pt[1]) for pt in no_side
                                                         if abs((1.0 - float(pt[0])) - mid) <= band)

                                    return bid_depth >= min_lots and ask_depth >= min_lots

                                def check_poly_depth(token_id, min_lots=100, band_cents=10):
                                    """Check if Poly CLOB has min_lots on BOTH sides within band_cents of mid."""
                                    try:
                                        r = _get_with_retry(f"https://clob.polymarket.com/book?token_id={token_id}", timeout=5)
                                        if r.status_code != 200:
                                            return False
                                        book = r.json()
                                        bids = book.get("bids", [])
                                        asks = book.get("asks", [])
                                        if not bids or not asks:
                                            return False
                                        best_bid = max(float(b["price"]) for b in bids)
                                        best_ask = min(float(a["price"]) for a in asks)
                                        mid = (best_bid + best_ask) / 2.0
                                        bid_depth = sum(float(b["size"]) for b in bids
                                                       if abs(float(b["price"]) - mid) <= band_cents / 100.0)
                                        ask_depth = sum(float(a["size"]) for a in asks
                                                       if abs(float(a["price"]) - mid) <= band_cents / 100.0)
                                        return bid_depth >= min_lots and ask_depth >= min_lots
                                    except Exception:
                                        return False

                                series_prob, s_src, s_used_spr = best_prob_for_leg(s_markets, "series")
                                p1_prob, p1_src, p1_used_spr = best_prob_for_leg(m1_markets, "game1")
                                p2_prob, p2_src, p2_used_spr = best_prob_for_leg(m2_markets, "game2")

                                # Spread penalty based on the spread of whichever source
                                # actually provided the probability (Poly or Kalshi).
                                # best_prob_for_leg guarantees used_spr ≤ 15 on a non-None
                                # return, and the row was already skipped above if any prob
                                # is None — so max_map_spread is bounded to [0, 15] here.
                                max_map_spread = max(p1_used_spr, p2_used_spr)
                                spread_penalty = round(max(0, (max_map_spread - 4.0) / 2.0), 1)
                                if spread_penalty > 0:
                                    print(f"    Spread penalty: max_spread={max_map_spread:.0f}c → +{spread_penalty:.1f}c extra edge")
                                    spread_penalties[event_base] = max(spread_penalties.get(event_base, 0), spread_penalty)

                                # For BO5, also capture p3, p4, p5 — Poly typically exposes
                                # G4 (and sometimes G5) winner markets that the BO5 hedge
                                # math can plug in directly instead of using (p1+p2+p3)/3
                                # as a stand-in for future-game probability.
                                p3_prob = None
                                p3_src = "NONE"
                                p4_prob = None
                                p4_src = "NONE"
                                p5_prob = None
                                p5_src = "NONE"
                                if is_bo5:
                                    p3_prob, p3_src, _ = best_prob_for_leg(m3_markets, "game3")
                                    if "KXCOD" in event_base:
                                        # COD: Map 4 reuses Map 1 mode (Hardpoint), Map 5
                                        # reuses Map 2 mode (S&D). Poly G4 exists but trades
                                        # ~47c wide pre-resolution — not usable as a fresh
                                        # signal. Use the same-mode prior instead of avg.
                                        p4_prob = round(p1_prob, 3) if p1_prob is not None else None
                                        p4_src = "p1_copy_cod" if p4_prob is not None else "NONE"
                                        p5_prob = round(p2_prob, 3) if p2_prob is not None else None
                                        p5_src = "p2_copy_cod" if p5_prob is not None else "NONE"
                                    else:
                                        # G4 policy 2026-05-25: Poly G4 books are often a single
                                        # thin/manipulated quote pre-G3-resolution (saw 5 lots
                                        # 1c wide at 46.5% when real prob was 36%, distorted
                                        # downstream pricing). Don't seed p4 from Poly G4 mid.
                                        # Use avg(p1,p2,p3) as the baseline. The G4 refresh path
                                        # in `_evaluate_leg` only overwrites p4 once G3 resolves,
                                        # at which point Poly G4 is imminent/liquid enough to trust.
                                        if all(v is not None for v in (p1_prob, p2_prob, p3_prob)):
                                            p4_prob = round((p1_prob + p2_prob + p3_prob) / 3.0, 3)
                                            p4_src = "avg(p1,p2,p3)"
                                        else:
                                            p4_prob = None
                                            p4_src = "NONE"
                                        p5_prob, p5_src, _ = best_prob_for_leg(m5_markets, "game5")

                                if series_prob is None or p1_prob is None or p2_prob is None:
                                    skipped = [l for l, v in [("Series", series_prob), ("Map1", p1_prob), ("Map2", p2_prob)] if v is None]
                                    print(f"Skipping {sm_ticker} - No tradeable spread for: {', '.join(skipped)}")
                                    continue

                                # MAP LIQUIDITY CHECK: require 100 lots on both bid and ask
                                # within 10c of midpoint on map markets. Illiquid map books
                                # produce unreliable hedge costs and phantom edge.
                                # Check whichever source provided each map's probability.
                                for leg_label, leg_src, leg_markets, poly_leg_key in [
                                    ("Map1", p1_src, m1_markets, "game1"),
                                    ("Map2", p2_src, m2_markets, "game2"),
                                ]:
                                    leg_depth_ok = False
                                    if leg_src == "Kalshi":
                                        leg_depth_ok = check_kalshi_depth(leg_markets)
                                    elif leg_src == "Poly":
                                        pw_leg = poly_winners.get(poly_leg_key)
                                        if pw_leg and pw_leg.get("tokens"):
                                            outcomes = pw_leg.get("outcomes", [])
                                            leg_idx = _align_team_to_poly_outcome(kalshi_full_name, outcomes, event_base=event_base) if kalshi_full_name else -1
                                            if 0 <= leg_idx < len(pw_leg["tokens"]):
                                                leg_depth_ok = check_poly_depth(pw_leg["tokens"][leg_idx])
                                    if not leg_depth_ok:
                                        print(f"Skipping {sm_ticker} - {leg_label} has insufficient liquidity "
                                              f"(need 100 lots within 10c on both sides, source={leg_src})")
                                        p1_prob = None  # force skip
                                        break

                                if p1_prob is None:
                                    continue

                                # MAP FORFEIT DETECTION: if a map is settled on Kalshi
                                # but prob is near 50%, Poly is returning a voided forfeit price.
                                for leg_label, leg_prob, leg_src, leg_markets in [
                                    ("Map1", p1_prob, p1_src, m1_markets),
                                    ("Map2", p2_prob, p2_src, m2_markets)
                                ]:
                                    map_settled = any(
                                        m.get("status") in ("determined", "settled", "closed", "finalized")
                                        for m in leg_markets
                                    )
                                    if map_settled and abs(leg_prob - 0.5) < 0.05:
                                        print(f"  ⚠ MAP FORFEIT {sm_ticker} — {leg_label} is settled on Kalshi "
                                              f"but prob={leg_prob:.3f} (~50%) from {leg_src}. "
                                              f"Map likely forfeited — purging probs and skipping event.")
                                        p1_prob = None
                                        # Purge any existing probs for this event so live_series won't quote
                                        for ex_ticker in list(existing_probs):
                                            if ex_ticker.startswith(event_base + "-"):
                                                existing_probs.discard(ex_ticker)
                                                valid_rows[:] = [r for r in valid_rows if r[0].strip() != ex_ticker]
                                                print(f"    Purged existing prob: {ex_ticker}")
                                        break

                                    if p1_prob is None:
                                        continue

                                # Safeguard: if series or any map prob is >97% or <3%, the game
                                # is either over, forfeited, or so lopsided there's no opportunity.
                                if series_prob > 0.97 or series_prob < 0.03:
                                    print(f"  ⚠ SKIPPING {sm_ticker} — extreme series prob ({series_prob:.3f}). Match likely over or forfeited.")
                                    continue
                                if p1_prob > 0.97 or p1_prob < 0.03 or p2_prob > 0.97 or p2_prob < 0.03:
                                    print(f"  ⚠ SKIPPING {sm_ticker} — extreme map prob (p1={p1_prob:.3f} p2={p2_prob:.3f}). Game likely over or no opportunity.")
                                    continue

                                # Safeguard: if p1 and p2 differ by too much, probabilities were likely
                                # captured mid-game (e.g. game started early). Skip to avoid bad baselines.
                                # CS2 gets a looser 11% gate — legitimate 9–10% spreads are common.
                                delta_limit = 0.11 if "KXCS2" in event_base else 0.08
                                if abs(p1_prob - p2_prob) > delta_limit:
                                    print(f"  ⚠ SKIPPING {sm_ticker} — p1={p1_prob:.3f} p2={p2_prob:.3f} differ by {abs(p1_prob-p2_prob)*100:.1f}% (>{delta_limit*100:.0f}%). Likely captured mid-game.")
                                    continue

                                # Safeguard: if all legs are ~50%, probabilities are likely
                                # hallucinated (e.g. Poly alignment failure → both teams got same
                                # price → complementarity normalized to 50/50). Skip entirely.
                                if abs(p1_prob - 0.5) < 0.01 and abs(p2_prob - 0.5) < 0.01 and abs(series_prob - 0.5) < 0.01:
                                    if sm_ticker in existing_probs:
                                        # Already in CSV (manually entered) — don't skip
                                        print(f"  ✓ {sm_ticker} — probs ≈50% but already in CSV, keeping")
                                    else:
                                        print(f"  ⚠ SKIPPING {sm_ticker} — all probs ≈50% (p1={p1_prob:.3f} p2={p2_prob:.3f} series={series_prob:.3f}). Likely hallucinated alignment.")
                                        continue

                                if is_bo5 and p3_prob is None:
                                    # For BO5, p3 is required. Fallback to avg(p1, p2).
                                    p3_prob = (p1_prob + p2_prob) / 2.0
                                    p3_src = "avg(p1,p2)"

                                # ── VAL delta-preserving level solve (26AUG11) ──
                                # Operator decision: for VAL the SERIES book is
                                # the level authority and the maps carry only
                                # the veto delta. Keep p1−p2, shift both so the
                                # implied series equals the CAPTURED market
                                # series (0.60/0.58 -> 0.58/0.56 shape). Must
                                # run BEFORE the sanity block below — that
                                # block resolves series-vs-maps divergence the
                                # OPPOSITE way (replaces the series with the
                                # map-implied value), which is exactly the VAL
                                # trust ranking inverted. BO3 only; a failed
                                # solve falls through to today's behavior,
                                # loudly. Kill: disable_val_delta_reconcile.flag
                                if (event_base.startswith("KXVALORANTGAME") and not is_bo5
                                        and not os.path.exists("disable_val_delta_reconcile.flag")):
                                    _dp = prob_reconcile.solve_delta_preserving(
                                        series_prob, p1_prob, p2_prob)
                                    if _dp is None:
                                        print(f"    [RECONCILE-VAL] {event_base}: no delta-preserving "
                                              f"solution (series={series_prob:.3f} p1={p1_prob:.3f} "
                                              f"p2={p2_prob:.3f}) — keeping per-leg probs")
                                    else:
                                        _np1, _np2 = _dp
                                        if abs(_np1 - p1_prob) >= 0.0005:
                                            print(f"    [RECONCILE-VAL] {event_base}: delta "
                                                  f"{p1_prob - p2_prob:+.3f} kept, series "
                                                  f"{series_prob:.3f} kept, p1 {p1_prob:.3f}->"
                                                  f"{_np1:.3f}, p2 {p2_prob:.3f}->{_np2:.3f}")
                                        p1_prob, p2_prob = round(_np1, 3), round(_np2, 3)

                                # Sanity check: map-implied series should be close to captured series.
                                # If they diverge by >8c, the series price is likely from a bad source.
                                if is_bo5:
                                    # BO5: use dynamic programming for series implied
                                    p3_val = p3_prob if p3_prob else (p1_prob + p2_prob) / 2.0
                                    p4_val = (p1_prob + p2_prob + p3_val) / 3.0
                                    ps = [p1_prob, p2_prob, p3_val, p4_val, p4_val]
                                    memo = {}
                                    def _win(wa, wb):
                                        if wa == 3: return 1.0
                                        if wb == 3: return 0.0
                                        if (wa,wb) in memo: return memo[(wa,wb)]
                                        gi = wa + wb
                                        p = ps[gi] if gi < len(ps) else ps[-1]
                                        r = p * _win(wa+1, wb) + (1-p) * _win(wa, wb+1)
                                        memo[(wa,wb)] = r
                                        return r
                                    series_implied = _win(0, 0)
                                else:
                                    p3_implied = (p1_prob + p2_prob) / 2.0
                                    series_implied = p1_prob * p2_prob + p1_prob * (1 - p2_prob) * p3_implied + (1 - p1_prob) * p2_prob * p3_implied
                                if abs(series_prob - series_implied) > 0.08:
                                    print(f"  ⚠ SERIES SANITY FAIL {sm_ticker}: captured={series_prob:.3f}({s_src}) implied={series_implied:.3f} diff={abs(series_prob-series_implied)*100:.1f}c → using implied")
                                    series_prob = series_implied

                                # ── Continuation-probability reconciliation ──
                                # (2026-08-06, FLC/LIQUID post-mortem — see
                                # PREGAME_LEAN_SCOPE.md). Each leg above was
                                # priced off its OWN book, so nothing forces the
                                # set to agree: on LIQUIDFLC p1..p5 implied a
                                # series of 79.7 against a market of 76.5, and
                                # that standing +3.2c disagreement never closed
                                # — 57% of map-1 ticks showed edge to buy and 0%
                                # to sell, and the arber ran to its 45,000 cap.
                                #
                                # Backsolve ONE continuation probability so the
                                # two edges are symmetric at the opening
                                # snapshot. p1 is PINNED (it is the real map-1
                                # mid — the best information we have) and
                                # series_prob is left alone; only p2 (and p3/p4/
                                # p5 on BO5) move. LoL + Dota only, both tiers.
                                #
                                # Runs AFTER the series-sanity block above, so
                                # it reconciles against whichever series value
                                # actually gets written. The two do opposite
                                # things — sanity replaces the MARKET series
                                # with the model's at >8c divergence, this
                                # replaces the MODEL's continuation to match the
                                # market — so order matters.
                                if prob_reconcile.applies_to(event_base):
                                    _q = prob_reconcile.solve_continuation_from_probs(
                                        series_prob, max(s_used_spr, 0.5),
                                        p1_prob, max(p1_used_spr, 0.5), is_bo5)
                                    if _q is None:
                                        # Guard tripped (wide/crossed book, or the
                                        # root sits outside max_shift). Keep the
                                        # per-leg probabilities and say so — a
                                        # silent fallback here is how the lean got
                                        # shipped in the first place.
                                        print(f"    [RECONCILE] {event_base}: no solution "
                                              f"(series={series_prob:.3f}/{s_used_spr:.0f}c "
                                              f"p1={p1_prob:.3f}/{p1_used_spr:.0f}c) — "
                                              f"keeping per-leg probs")
                                    else:
                                        _old_p2 = p2_prob
                                        p2_prob = round(_q, 3)
                                        if is_bo5:
                                            p3_prob = p4_prob = p5_prob = round(_q, 3)
                                            p3_src = p4_src = p5_src = "reconciled"
                                        print(f"    [RECONCILE] {event_base}: p1={p1_prob:.3f} kept, "
                                              f"series={series_prob:.3f} kept, "
                                              f"p2 {_old_p2:.3f} -> {p2_prob:.3f} "
                                              f"({p2_prob - _old_p2:+.3f})"
                                              + (" (p3/p4/p5 too)" if is_bo5 else ""))

                                g2_momentum, g3_momentum = DEFAULT_MOMENTUM
                                for _game_key, _bumps in MOMENTUM_BY_GAME.items():
                                    if _game_key in event_base:
                                        g2_momentum, g3_momentum = _bumps
                                        break
                                if "KXVALORANT" in event_base:
                                    try:
                                        if classify_val_tier(event_base, s_markets) == "vct":
                                            g2_momentum, g3_momentum = MOMENTUM_VAL_VCT
                                    except Exception:
                                        pass
                                # CS2 tier-1 g3 override (0.08, matches tier-2 base as of 2026-07-05).
                                if "KXCS2" in event_base:
                                    try:
                                        if classify_cs_tier(event_base, poly_mappings) == "tier1":
                                            g2_momentum, g3_momentum = MOMENTUM_CS_T1
                                    except Exception:
                                        pass

                                # Spread penalty already computed above from the source that provided probs
                                if spread_penalty > 0:
                                    print(f"    Spread penalty: max_map_spread={max_map_spread:.0f}c (p1={p1_src} {p1_used_spr:.0f}c, p2={p2_src} {p2_used_spr:.0f}c) → +{spread_penalty:.1f}c extra edge")
                                    spread_penalties[event_base] = max(spread_penalties.get(event_base, 0), spread_penalty)

                                # CSV column layout (0-indexed):
                                #  0  ticker
                                #  1  p1_start
                                #  2  p2_start
                                #  3  series_start
                                #  4  g2_momentum
                                #  5  g3_momentum
                                #  6  p3 (BO5 only, '' for BO3)
                                #  7  spread_penalty
                                #  8  VERIFIED flag (manually edited)
                                #  9  p4 (BO5: numeric when Poly G4 available; '-' = tried but Poly didn't expose; '' = BO3)
                                # 10  p5 (BO5: numeric when Poly G5 available; '-' = tried but Poly didn't expose; '' = BO3)
                                #
                                # The '-' sentinel for BO5 events without Poly G4/G5 is what
                                # prevents the backfill loop from re-firing every cycle when
                                # Poly never exposes those legs (~35% of BO5 events lack G4
                                # ever, ~55% lack G5). Loader treats non-numeric as None and
                                # falls back to (p1+p2+p3)/3 → p4 → p5 chain.
                                row = [sm_ticker, round(p1_prob, 3), round(p2_prob, 3), round(series_prob, 3), g2_momentum, g3_momentum]
                                if is_bo5:
                                    row.append(round(p3_prob, 3))
                                else:
                                    row.append("")  # p3 placeholder
                                row.append(spread_penalty)  # column 7: spread penalty
                                row.append("")  # column 8: VERIFIED flag — left blank by default
                                if is_bo5:
                                    row.append(round(p4_prob, 3) if p4_prob is not None else "-")
                                    row.append(round(p5_prob, 3) if p5_prob is not None else "-")
                                else:
                                    row.append("")  # column 9: p4 placeholder (BO3 doesn't use)
                                    row.append("")  # column 10: p5 placeholder (BO3 doesn't use)
                                new_probs_rows.append(row)
                                existing_probs.add(sm_ticker)
                                bo_label = "BO5" if is_bo5 else "BO3"
                                p3_str = f" p3={p3_prob:.3f}({p3_src})" if is_bo5 else ""
                                p4_str = f" p4={p4_prob:.3f}({p4_src})" if (is_bo5 and p4_prob is not None) else ""
                                p5_str = f" p5={p5_prob:.3f}({p5_src})" if (is_bo5 and p5_prob is not None) else ""
                                print(f"  Probs {sm_ticker} [{bo_label}]: series={series_prob:.3f}({s_src}) "
                                      f"p1={p1_prob:.3f}({p1_src}) p2={p2_prob:.3f}({p2_src}){p3_str}{p4_str}{p5_str}")

                            # Enforce complementarity: both teams must sum to 1.0 for each leg
                            # Find pairs in new_probs_rows that belong to the same event
                            event_rows = [r for r in new_probs_rows if r[0].startswith(event_base + "-")]
                            if len(event_rows) == 1:
                                # Only one team captured — can't trade without both sides
                                orphan = event_rows[0][0]
                                print(f"  ⚠ REMOVING orphan {orphan} — opponent team failed probability capture")
                                new_probs_rows[:] = [r for r in new_probs_rows if r[0] != orphan]
                                existing_probs.discard(orphan)
                            elif len(event_rows) == 2:
                                r_a, r_b = event_rows
                                abort_event = False
                                for col in [1, 2, 3]:  # p1, p2, series
                                    total = r_a[col] + r_b[col]
                                    if total < 0.95 or total > 1.05:
                                        print(f"  🚨 ABORTING {event_base} — col={['','p1','p2','series'][col]}: "
                                              f"{r_a[0].split('-')[-1]}={r_a[col]:.3f} + {r_b[0].split('-')[-1]}={r_b[col]:.3f} = {total:.3f} "
                                              f"(outside [0.95, 1.05] — likely Poly alignment bug)")
                                        abort_event = True
                                        break
                                if abort_event:
                                    # Remove both rows from new_probs_rows
                                    new_probs_rows[:] = [r for r in new_probs_rows if not r[0].startswith(event_base + "-")]
                                    existing_probs -= {r_a[0], r_b[0]}
                                else:
                                    for col in [1, 2, 3]:
                                        total = r_a[col] + r_b[col]
                                        if total <= 0:
                                            continue
                                        if abs(total - 1.0) > 0.01:
                                            print(f"  ⚠ COMPLEMENT FIX col={['','p1','p2','series'][col]}: "
                                                  f"{r_a[0].split('-')[-1]}={r_a[col]:.3f} + {r_b[0].split('-')[-1]}={r_b[col]:.3f} = {total:.3f} → normalizing")
                                        r_a[col] = round(r_a[col] / total, 3)
                                        r_b[col] = round(1.0 - r_a[col], 3)

                                    # ── Tier-1 CS spread-penalty guardrail (added 2026-06-03)
                                    # Per-team Poly /spread queries can return asymmetric values
                                    # for the two tokens of the same market due to API timing
                                    # (consecutive calls catch the orderbook at different
                                    # microseconds). MIN of the two reads is the trustworthy
                                    # value; the higher one is most likely a transient artifact.
                                    # Tier-1 CS books (IEM/BLAST/PGL/DREAMHACK/ASIA CHAMPIONSHIP)
                                    # are consistently tight in steady state, so > 1.5c is almost
                                    # always a capture artifact. Cap at 1.5c on those events.
                                    if "KXCS2" in event_base and classify_cs_tier(event_base, poly_mappings) == "tier1":
                                        try:
                                            p_a = float(r_a[7] or 0)
                                            p_b = float(r_b[7] or 0)
                                            sym_penalty = min(p_a, p_b)
                                            if sym_penalty > 1.5:
                                                print(f"  [TIER1-CS PENALTY CAP] {event_base}: "
                                                      f"raw_min={sym_penalty}c → 1.5c (transient capture)")
                                                sym_penalty = 1.5
                                            if sym_penalty != p_a or sym_penalty != p_b:
                                                print(f"  [TIER1-CS PENALTY SYMM] {event_base}: "
                                                      f"a={p_a}c b={p_b}c → both={sym_penalty}c")
                                            r_a[7] = sym_penalty
                                            r_b[7] = sym_penalty
                                            # Sync in-memory dict used by section 3.5
                                            # (template-level penalty apply this cycle).
                                            if sym_penalty > 0:
                                                spread_penalties[event_base] = sym_penalty
                                            else:
                                                spread_penalties.pop(event_base, None)
                                        except (ValueError, TypeError) as _e:
                                            print(f"  ⚠ tier1-cs penalty sym failed for {event_base}: {_e}")
                    except Exception as e:
                        print(f"Failed to fetch baseline probs for {event_base}: {e}")

            # Bo5 detection fallback: triggered when the series API call (or any
            # step before is_bo5 resolution) failed, leaving is_bo5_resolved False.
            # Without this, template selection at the loop below uses the line-790
            # default is_bo5=False and silently demotes real Bo5 events to Bo3
            # sizing on transient series-API failures (KCG2 incident 2026-05-16).
            # Priority: persisted `format` column > poly_title > standalone m3
            # probe (legacy fallback for rows without the format column).
            if (not is_bo5_resolved
                    and is_series_event
                    and any(p in event_base for p in ["KXLOL", "KXCS2", "KXVALORANT", "KXDOTA2", "KXCOD", "KXATP", "KXWTA"])):
                poly_mapping = poly_mappings.get(event_base) or {}
                poly_title_verified = poly_mapping.get("poly_title", "")
                kalshi_fmt = poly_mapping.get("format", "")
                _title_upper = poly_title_verified.upper()
                _title_bo5 = "BO5" in _title_upper
                _title_bo3 = "BO3" in _title_upper
                # Priority: COD > poly_title (when present and unambiguous) > kalshi_fmt >
                # m3 REST. poly_title wins disagreements with kalshi_fmt (operator-verified
                # signal beats the lagging Kalshi has_m3 reading).
                if "KXCOD" in event_base:
                    is_bo5 = True
                elif _title_bo5 and not _title_bo3:
                    is_bo5 = True
                elif _title_bo3 and not _title_bo5:
                    is_bo5 = False
                elif kalshi_fmt == "BO5":
                    is_bo5 = True
                elif kalshi_fmt == "BO3":
                    is_bo5 = False
                else:
                    if "MATCH" in event_base:
                        _m3_event = event_base.replace("MATCH", "SETWINNER") + "-3"
                    else:
                        _m3_event = event_base.replace("GAME", "MAP") + "-3"
                    for _attempt in range(2):
                        try:
                            _r = _get_with_retry(
                                f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={_m3_event}",
                                headers={"Accept": "application/json"}, timeout=5,
                            )
                            if _r.status_code == 200:
                                is_bo5 = len(_r.json().get("markets", [])) > 0
                                break
                        except Exception:
                            pass
                        if _attempt == 0:
                            time.sleep(0.5)
                print(f"  [BO5-FALLBACK] {event_base}: series-API path didn't resolve is_bo5; "
                      f"poly_title='{poly_title_verified}' → is_bo5={is_bo5}")

            # Hard kill-switch for BO5 trading. When disable_bo5.flag exists in
            # the working directory, BO5 events skip template emission entirely
            # so no rows reach template_quoter_config.csv. BO3 events are
            # untouched. Created 2026-05-16 after the THFUT/FLURED incident;
            # delete the flag to re-enable BO5 once the math is fixed.
            if is_bo5 and os.path.exists("disable_bo5.flag"):
                print(f"  [BO5-DISABLED] {event_base}: disable_bo5.flag present, skipping template emission")
                continue

            for prefix in param_maps.keys():
                # Only try prefixes that share the same sport root as the event
                # e.g. KXLOLGAME event should only try KXLOL* prefixes, not KXATP*
                #
                # Bug fixed 2026-06-25: the keyword set must include MOV|MOF
                # (UFC method-of-victory prefixes) or KXUFCMOV regex-misses,
                # falls through `event_root and prefix_root and ...` as
                # short-circuit False, and gets tried against EVERY event
                # — burning the /markets? bucket on bogus per-event probes
                # like KXUFCMOV-{valorant_tail}. Same applies to any
                # regex-miss on the prefix side: treat as "unknown family,
                # skip" rather than "fall through to try anyway."
                import re
                _KIND_RE = r'(KX[A-Z0-9]+?)(?:GAME|MAP|MATCH|SETWINNER|TOTAL|SPREAD|F5|MOV|MOF)'
                event_root = re.match(_KIND_RE, event_base.split("-")[0])
                prefix_root = re.match(_KIND_RE, prefix)
                if not event_root or not prefix_root:
                    continue
                if event_root.group(1) != prefix_root.group(1):
                    continue

                # MMA Arber: emit one config row per EVENT (not per child market)
                # The bot manages all 7 children internally via dynamic discovery.
                execution_type = param_maps[prefix][0][-1].strip().lower() if param_maps[prefix] else ""
                if execution_type == "mma_arber":
                    derived_event = f"{prefix}-{tail_tag}"
                    api_url = f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={derived_event}"
                    markets = None
                    for attempt in range(2):
                        try:
                            r = _get_with_retry(api_url, headers={"Accept": "application/json"}, timeout=5)
                            if r.status_code == 200:
                                markets = r.json().get("markets", [])
                                break
                        except Exception:
                            pass
                        if attempt == 0:
                            time.sleep(0.5)
                    if markets:
                        active_markets = [m for m in markets if m.get("status") == "active"]
                        if active_markets:
                            for template_row in param_maps[prefix]:
                                new_row = template_row.copy()
                                new_row[1] = derived_event  # Event-level ticker, not child
                                final_output_rows.append(new_row)
                            print(f"[MMA] Bound event {derived_event} with {len(active_markets)} active child markets")
                    elif markets is None:
                        print(f"[MMA] Network failure checking {derived_event} (2 attempts exhausted)")
                    continue

                derived_tickers_to_check = []

                # Kalshi splits Map/Set markets using strict sequence suffixes natively (-1, -2, -3)
                # Per-map prefixes (KXCS2MAP, KXLOLMAP, etc.) require iterating
                # map numbers 1-5. `endswith("MAP")` avoids matching prop
                # prefixes like KXCS2TOTALMAPS (event-level, single child market
                # with the threshold suffix — queried at the event_ticker level
                # below, just like GAME/MATCH prefixes).
                _pu = prefix.upper()
                if _pu.endswith("MAP") or "SETWINNER" in _pu:
                    _outer_max_n = _max_map_n(event_base, poly_mappings)
                    for map_num in range(1, _outer_max_n + 1):
                        derived_tickers_to_check.append(f"{prefix}-{tail_tag}-{map_num}")
                else:
                    derived_tickers_to_check.append(f"{prefix}-{tail_tag}")

                for derived_event_ticker in derived_tickers_to_check:
                    # ── Synthetic event bypass ─────────────────────────────
                    # event_bases registered in ticker_aliases.csv don't exist
                    # on Kalshi's event-discovery API. Synthesize active_markets
                    # from the alias map directly (one ticker per team suffix
                    # registered under this event_base). For map prefixes
                    # (KXCS2MAP / KXVALORANTMAP / etc.) the synthesized map
                    # tickers don't trade on Kalshi — they're bookkeeping-only,
                    # fed by Poly via the BO5 routing in run.py.
                    if ticker_aliases.event_base_is_synthetic(event_base):
                        teams = _teams_for_synthetic_event_base(event_base)
                        if not teams:
                            # Defensive: event_base_is_synthetic implies non-empty.
                            continue
                        active_markets = [f"{derived_event_ticker}-{team}" for team in teams]
                        print(f"  [SYNTH] {derived_event_ticker}: synthesized {len(active_markets)} active marker(s) from alias map")
                    else:
                        # Try the per-event WS cache first (built right after
                        # s_markets was fetched). For derived_event_tickers that
                        # follow the standard {prefix}-{tail}[-{N}]-{team} layout,
                        # _ws_event_markets reconstructs the same shape Kalshi
                        # REST returns. Only hit REST if the cache misses (e.g.
                        # non-team prefixes like KXCS2TOTALMAPS, or events
                        # before team suffixes were learned).
                        markets, _ws_src = _ws_event_markets(derived_event_ticker)
                        if _ws_src != "ws_hit":
                            markets = None
                        api_url = f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={derived_event_ticker}"
                        headers = {"Accept": "application/json"}

                        for attempt in range(2):
                            if markets is not None:
                                break
                            try:
                                r = _get_with_retry(api_url, headers=headers, timeout=5)
                                if r.status_code == 200:
                                    markets = r.json().get("markets", [])
                                    break
                            except Exception:
                                pass
                            if attempt == 0:
                                time.sleep(0.5)
                        time.sleep(0.1)  # Throttle API calls to avoid rate limiting

                        if markets is None:
                            print(f"Network failure mapping {derived_event_ticker} (2 attempts exhausted)")
                            events_with_api_failure.add(event_base)
                            continue
                        if not markets:
                            continue  # Break empty map loops cleanly

                        # Only bind to physical markets that are currently 'active' on the orderbook
                        active_markets = [m.get("ticker") for m in markets if m.get("status") == "active"]
                        if not active_markets:
                            continue

                    cs_tier = classify_cs_tier(event_base, poly_mappings) if "KXCS2" in prefix else None
                    val_tier = classify_val_tier(event_base, s_markets) if "KXVALORANT" in prefix else None
                    lol_tier = classify_lol_tier(event_base, poly_mappings, s_markets) if "KXLOL" in prefix else None
                    # League blacklist: emit nothing at all for this event.
                    _bl = (lol_league_blacklisted(event_base, poly_mappings, s_markets)
                           if "KXLOL" in prefix else None)
                    if _bl:
                        if event_base not in _league_blacklist_logged:
                            _league_blacklist_logged.add(event_base)
                            print(f"[LEAGUE-BLACKLIST] {event_base}: matched '{_bl}' "
                                  f"— emitting NO rows (event will not be traded)")
                        continue
                    dota_tier = classify_dota_tier(event_base, poly_mappings) if "KXDOTA2" in prefix else None
                    emitted_per_market = 0
                    for mkt_ticker in active_markets:
                        # Apply the exact blueprints
                        for template_row in param_maps[prefix]:
                            # Substring match (not endswith) so combined
                            # `_T1_BO5` template names match BOTH filters.
                            if cs_tier is not None:
                                is_t1_row = "_T1" in template_row[0]
                                if cs_tier == "tier1" and not is_t1_row:
                                    continue  # Tier 1 CS event: only _T1 templates
                                if cs_tier == "tier2" and is_t1_row:
                                    continue  # Tier 2 CS event: skip _T1 templates
                            if val_tier is not None:
                                is_vcl_row = "_VCL" in template_row[0]
                                if val_tier == "vct" and is_vcl_row:
                                    continue  # VCT event: skip _VCL templates
                                if val_tier == "vcl" and not is_vcl_row:
                                    continue  # VCL event: only _VCL templates
                            if lol_tier is not None:
                                is_lol_t2_row = "_T2" in template_row[0]
                                if lol_tier == "tier1" and is_lol_t2_row:
                                    continue  # Tier-1 LoL event: skip _T2 templates
                                if lol_tier == "tier2" and not is_lol_t2_row:
                                    continue  # Tier-2 LoL event: only _T2 templates
                            if dota_tier is not None:
                                is_dota_t1_row = "_T1" in template_row[0]
                                if dota_tier == "tier1" and not is_dota_t1_row:
                                    continue  # Tier-1 Dota event: only _T1 templates
                                if dota_tier == "tier2" and is_dota_t1_row:
                                    continue  # Tier-2 Dota event: skip _T1 templates
                            is_bo5_row = "_BO5" in template_row[0]
                            if is_bo5 and not is_bo5_row:
                                continue  # Bo5 event: only _BO5 templates
                            if (not is_bo5) and is_bo5_row:
                                continue  # Bo3 event: skip _BO5 templates
                            new_row = template_row.copy()
                            new_row[1] = mkt_ticker # Overwrite 'KXMLBTOTAL' with pure ticker 'KXMLBTOTAL-XX-XX-5'
                            # Force-include override: events without a Poly mapping
                            # need data_source=kalshi (the row's default may be poly).
                            # Column 19 = data_source per the schema in framework_config.
                            # Applies to force_include_tickers.txt patterns AND to any
                            # aliased (synthetic) event — those are Kalshi-only by
                            # construction (real maps + alias-resolved series, no Poly),
                            # so poly data_source would leave the quoter waiting on a
                            # hedge feed that never arrives.
                            try:
                                _force_pats = _load_force_include_patterns()
                                _force_kalshi = (
                                    (_force_pats and any(p in event_base for p in _force_pats))
                                    or ticker_aliases.event_base_is_synthetic(event_base)
                                )
                                if _force_kalshi and len(new_row) > 19 and new_row[19] != "kalshi":
                                    new_row[19] = "kalshi"
                            except Exception:
                                pass
                            final_output_rows.append(new_row)
                            if mkt_ticker == active_markets[0]:
                                emitted_per_market += 1

                    layer_desc = f"{emitted_per_market} layers" + (f" [{cs_tier}]" if cs_tier else "")
                    print(f"Mapped {len(active_markets)} active markets across {layer_desc} for {derived_event_ticker}")

    # 3.5a. Intermission swap: replace arber rows with intermission quoter rows
    # when map 1 is settled (inter-game break)
    intermission_templates = {}  # prefix -> [template_rows] for esports_intermission
    for prefix, rows in param_maps.items():
        for row in rows:
            if len(row) >= 14 and row[13] == "esports_intermission":
                if prefix not in intermission_templates:
                    intermission_templates[prefix] = []
                intermission_templates[prefix].append(row)

    if intermission_templates:
        # Check each series event for intermission state
        esports_game_prefixes = [p for p in intermission_templates.keys()]
        intermission_events = set()  # event_bases currently in intermission
        intermission_start_times = {}  # persist across daemon runs via file

        # Load intermission state file
        intermission_state_file = "_intermission_state.json"
        try:
            if os.path.exists(intermission_state_file):
                with open(intermission_state_file, "r") as f:
                    intermission_start_times = json.loads(f.read())
        except Exception:
            intermission_start_times = {}

        # Check map 1 status for each active series event
        for row in list(final_output_rows):
            ticker = row[1]
            if not any(gp in ticker for gp in esports_game_prefixes):
                continue
            # Per-map tickers (KXCS2MAP-...) are skipped here; prop tickers
            # (KXCS2TOTALMAPS-...) ARE series-level so they fall through.
            _pp = ticker.split("-", 1)[0]
            if _pp.endswith("MAP") or "SETWINNER" in _pp:
                continue

            parts = ticker.split("-")
            if len(parts) < 3:
                continue
            series_base = parts[0] + "-" + parts[1]
            team_suffix = parts[2]

            # 2026-07-28: rewrite the PREFIX only. series_base.replace() mangles
            # the date+teams half when a TEAM NAME contains the token — "2GAME
            # Esports" turned ...SR2GAME into ...SR2MAP (404).
            # See memory/project_map_base_game_substring_collision.md
            _mpfx = parts[0]
            if "MATCH" in _mpfx:
                map_base = _mpfx.replace("MATCH", "SETWINNER") + "-" + parts[1]
            else:
                map_base = _mpfx.replace("GAME", "MAP") + "-" + parts[1]

            map1_event = f"{map_base}-1"

            # Check map 1 status via Kalshi API
            try:
                api_url = f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={map1_event}"
                r = _get_with_retry(api_url, headers={"Accept": "application/json"}, timeout=5)
                if r.status_code == 200:
                    m1_markets = r.json().get("markets", [])
                    m1_settled = False
                    for m in m1_markets:
                        status = m.get("status", "")
                        if status in ["determined", "settled", "closed", "finalized"]:
                            m1_settled = True
                            break

                    # Check map 2 status — if map 2 is also settled, intermission is OVER
                    map2_event = f"{map_base}-2"
                    r2 = _get_with_retry(f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={map2_event}",
                                     headers={"Accept": "application/json"}, timeout=5)
                    m2_active = False
                    if r2.status_code == 200:
                        m2_markets = r2.json().get("markets", [])
                        for m in m2_markets:
                            if m.get("status") == "active":
                                m2_active = True
                                break
                            if m.get("status") in ["determined", "settled", "closed", "finalized"]:
                                # Map 2 also settled — past intermission
                                m1_settled = False
                                break

                    if m1_settled:
                        # Record when intermission started
                        if series_base not in intermission_start_times:
                            intermission_start_times[series_base] = time.time()
                            print(f"[INTERMISSION] Started for {series_base}")

                        elapsed = time.time() - intermission_start_times[series_base]
                        # 2 min delay before activating, 12 min total window
                        if 120 <= elapsed <= 720:
                            intermission_events.add(series_base)
                            print(f"[INTERMISSION] Active for {series_base} ({elapsed/60:.1f} min elapsed)")
                        elif elapsed > 720:
                            print(f"[INTERMISSION] Expired for {series_base} ({elapsed/60:.1f} min > 12 min)")
                    else:
                        # Not in intermission — clean up
                        intermission_start_times.pop(series_base, None)

                time.sleep(0.1)
            except Exception as e:
                print(f"[INTERMISSION] Error checking {series_base}: {e}")

        # Save intermission state
        try:
            with open(intermission_state_file, "w") as f:
                f.write(json.dumps(intermission_start_times))
        except Exception:
            pass

        # Swap rows: remove arber GAME+MAP rows for intermission events, add intermission quoter rows
        if intermission_events:
            new_rows = []
            for row in final_output_rows:
                ticker = row[1]
                parts = ticker.split("-")
                series_base_check = parts[0] + "-" + parts[1] if len(parts) >= 2 else ""

                # Check if this row belongs to an intermission event
                if series_base_check in intermission_events:
                    # Skip arber rows (both GAME and MAP) for this event
                    if row[14] == "arber" if len(row) > 14 else False:
                        continue
                    # Also skip per-map rows (KXCS2MAP, KXATPSETWINNER);
                    # prop event-level tickers (KXCS2TOTALMAPS) fall through.
                    _pp = ticker.split("-", 1)[0]
                    if _pp.endswith("MAP") or "SETWINNER" in _pp:
                        continue
                else:
                    pass  # Keep non-intermission rows

                new_rows.append(row)

            # Add intermission quoter rows for each intermission event
            for series_base in intermission_events:
                prefix = series_base.split("-")[0]  # e.g., KXLOLGAME
                templates = intermission_templates.get(prefix, [])
                cs_tier = classify_cs_tier(series_base, poly_mappings) if "KXCS2" in prefix else None
                dota_tier = classify_dota_tier(series_base, poly_mappings) if "KXDOTA2" in prefix else None

                # Find all active series tickers for this event
                api_url = f"https://external-api.kalshi.com/trade-api/v2/markets?event_ticker={series_base}"
                try:
                    r = _get_with_retry(api_url, headers={"Accept": "application/json"}, timeout=5)
                    if r.status_code == 200:
                        for m in r.json().get("markets", []):
                            if m.get("status") == "active":
                                mkt_ticker = m["ticker"]
                                for tmpl in templates:
                                    if cs_tier is not None:
                                        is_t1_row = "_T1" in tmpl[0]
                                        if cs_tier == "tier1" and not is_t1_row:
                                            continue
                                        if cs_tier == "tier2" and is_t1_row:
                                            continue
                                    if dota_tier is not None:
                                        is_t1_row = "_T1" in tmpl[0]
                                        if dota_tier == "tier1" and not is_t1_row:
                                            continue
                                        if dota_tier == "tier2" and is_t1_row:
                                            continue
                                    new_row = tmpl.copy()
                                    new_row[1] = mkt_ticker
                                    new_rows.append(new_row)
                                print(f"[INTERMISSION] Added quoter for {mkt_ticker}")
                except Exception as e:
                    print(f"[INTERMISSION] Error adding quoter rows: {e}")

            final_output_rows = new_rows
            print(f"[INTERMISSION] Swapped {len(intermission_events)} events to intermission mode")

    # 3.5. Apply spread-based edge penalties to min_edge (column 3).
    # Runs FIRST so per-ticker overrides in arb_overrides.csv (section 3.6)
    # can stomp the penalty when the operator wants a hard-pinned min_edge.
    # Order swapped 2026-05-27 — previously overrides ran first and the
    # penalty was added on top, so any manual min_edge override was silently
    # inflated by the live spread penalty on every populate cycle.
    if spread_penalties:
        applied_sp = 0
        for row in final_output_rows:
            ticker = row[1]
            for event_base, penalty in spread_penalties.items():
                if event_base in ticker:
                    base_edge = float(row[3])
                    new_edge = round(base_edge + penalty, 1)
                    row[3] = str(new_edge)
                    applied_sp += 1
                    break
        if applied_sp:
            print(f"[SPREAD PENALTY] Applied spread-based min_edge increase to {applied_sp} rows")

    # 3.6. Apply per-ticker overrides from arb_overrides.csv (LAST — wins over
    # spread penalty so manual pins stick).
    override_file = "arb_overrides.csv"
    if os.path.exists(override_file):
        # Column index map (0-indexed from market_parameters.csv header)
        # config_id=0, market_prefix=1, min_distance=2, min_edge=3, min_absolute_edge=4,
        # volumes=5, tick_step=6, reprice_buffer=7, stop_quoting=8, max_position=9,
        # skew_start=10, skew_max=11, quote_side=12, model_name=13, execution_type=14,
        # arb_scale_step=15, max_fire_size=16, series_edge_mult=29, fav_edge_k=37
        override_col_map = {"min_edge": 3, "min_absolute_edge": 4, "volumes": 5, "tick_step": 6, "stop_quoting": 8, "max_position": 9, "max_fire_size": 16, "series_edge_mult": 29, "fav_edge_k": 37}
        overrides = []
        with open(override_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pattern = row.get("ticker_contains", "").strip()
                if not pattern or pattern.startswith("#"):
                    continue
                overrides.append(row)

        if overrides:
            applied = 0
            for row in final_output_rows:
                ticker = row[1]
                row_config_id = row[0] if len(row) > 0 else ""
                row_exec_type = row[14] if len(row) > 14 else ""
                for ov in overrides:
                    pattern = ov["ticker_contains"].strip()
                    ov_exec_type = ov.get("execution_type", "").strip()
                    ov_config_id = ov.get("config_id_contains", "").strip()
                    if pattern in ticker:
                        # If override specifies execution_type, only apply to matching rows
                        if ov_exec_type and ov_exec_type != row_exec_type:
                            continue
                        # If override specifies config_id_contains, only apply to matching config_ids
                        if ov_config_id and ov_config_id not in row_config_id:
                            continue
                        for col_name, col_idx in override_col_map.items():
                            # `or ""` handles missing/None columns (newer keys
                            # absent from older arb_overrides.csv rows).
                            val = (ov.get(col_name) or "").strip()
                            if val:
                                row[col_idx] = val
                        applied += 1
                        break
            if applied:
                print(f"[OVERRIDES] Applied {applied} parameter overrides from {override_file}")

    # 3.7. Pregame favorite tightening: if an event's pregame series_start
    # implies a >=90% favorite on EITHER side, DOUBLE min_absolute_edge on
    # that event's ARBER rows (execution_type=="arber"). Quoter / momentum /
    # map_arber rows untouched. Rationale: external pregame flow consistently
    # buys the underdog YES, oversupplying favorite YES at small discounts —
    # doubling min_absolute_edge effectively only constrains the favorite-buy
    # direction since logit-scaled min_edge dominates at extreme underdog
    # prices. Set threshold to 1.01 to disable. Added 2026-06-03.
    PREGAME_FAVORITE_DOUBLE_THRESHOLD = 0.90
    # Build event_base → max(p, 1-p) over both teams from loaded probabilities.
    event_max_fav = {}
    for vr in valid_rows:
        try:
            tkr = (vr[0] or "").strip()
            p = float(vr[3])
        except (ValueError, IndexError):
            continue
        parts = tkr.rsplit("-", 1)
        if len(parts) != 2:
            continue
        eb = parts[0]
        fav = max(p, 1.0 - p)
        if fav > event_max_fav.get(eb, 0.0):
            event_max_fav[eb] = fav

    # Inline ticker → series-event-base normalizer (handles series + map tickers).
    _MAP2GAME = (("KXCS2MAP", "KXCS2GAME"), ("KXLOLMAP", "KXLOLGAME"),
                 ("KXVALORANTMAP", "KXVALORANTGAME"), ("KXDOTA2MAP", "KXDOTA2GAME"),
                 ("KXCODMAP", "KXCODGAME"))
    def _eb_for_arber_row(ticker: str) -> str:
        t = (ticker or "").upper()
        parts = t.rsplit("-", 1)
        if len(parts) != 2:
            return ""
        rest, _team = parts
        rest_parts = rest.rsplit("-", 1)
        if len(rest_parts) == 2 and rest_parts[1].isdigit():
            rest = rest_parts[0]  # strip map number suffix (KXLOLMAP-...-N → ...)
        for src, dst in _MAP2GAME:
            rest = rest.replace(src, dst)
        return rest

    doubled = 0
    for row in final_output_rows:
        if len(row) <= 14 or row[14].strip().lower() != "arber":
            continue
        eb = _eb_for_arber_row(row[1] if len(row) > 1 else "")
        fav = event_max_fav.get(eb)
        if fav is None or fav < PREGAME_FAVORITE_DOUBLE_THRESHOLD:
            continue
        try:
            current_mae = float(row[4])
        except (ValueError, IndexError):
            continue
        new_mae = current_mae * 2
        row[4] = f"{new_mae:g}"
        doubled += 1
    if doubled:
        print(f"[PREGAME-FAV-DOUBLE] Doubled min_absolute_edge on {doubled} arber rows "
              f"(event favorite >= {PREGAME_FAVORITE_DOUBLE_THRESHOLD*100:.0f}%)")

    # 4. Write output to temporary file, then atomic swap
    tmp_path = "_temp_template_quoter_config.csv"
    tgt_path = "template_quoter_config.csv"

    # Helper: recover the GAME event_base from any ticker (series or map) so
    # we can look up per-event capture buffer regardless of which row variant.
    _MAP_TO_GAME = (("KXCS2MAP", "KXCS2GAME"), ("KXLOLMAP", "KXLOLGAME"),
                    ("KXVALORANTMAP", "KXVALORANTGAME"), ("KXDOTA2MAP", "KXDOTA2GAME"),
                    ("KXCODMAP", "KXCODGAME"))
    def _event_base_from_ticker(t: str) -> str:
        u = (t or "").upper()
        for src, dst in _MAP_TO_GAME:
            if src in u:
                u = u.replace(src, dst)
                break
        parts = u.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return u

    # Preserve prior template rows for events that had per-prefix Kalshi API
    # failures during template emission. Without this, a transient network blip
    # silently drops all rows for an event, which on hot-reload cancels active
    # trades for 1-3 min until the next successful populator run. The cure-on-
    # failure is to fall back to whatever was last successfully written.
    if events_with_api_failure and os.path.exists(tgt_path):
        preserved_events = events_with_api_failure - {u.strip("/").split("/")[-1].upper() for u in settled_urls}
        if preserved_events:
            existing_by_event: dict[str, list[list[str]]] = {}
            try:
                with open(tgt_path, "r") as f:
                    rdr = csv.reader(f)
                    _ = next(rdr, None)  # header
                    for row in rdr:
                        if len(row) < 2: continue
                        eb = _event_base_from_ticker(row[1])
                        if eb in preserved_events:
                            existing_by_event.setdefault(eb, []).append(row[:-1] if len(row) >= 20 else row)
            except Exception as e:
                print(f"[PRESERVE] could not read existing template: {e}")
                existing_by_event = {}

            # Drop any partial-new rows for preserved events, then re-add the
            # full set of prior rows for those events.
            new_rows_for_failed = sum(1 for r in final_output_rows
                                      if _event_base_from_ticker(r[1] if len(r) > 1 else "") in preserved_events)
            final_output_rows = [r for r in final_output_rows
                                 if _event_base_from_ticker(r[1] if len(r) > 1 else "") not in preserved_events]
            restored = 0
            for eb, rows in existing_by_event.items():
                for r in rows:
                    final_output_rows.append(r)
                    restored += 1
            print(f"[PRESERVE] {len(preserved_events)} events hit API failures during template emission. "
                  f"Dropped {new_rows_for_failed} partial new rows; restored {restored} rows from prior template. "
                  f"events={sorted(preserved_events)}")

    # ── Settled-series scanner (WS-based, re-enabled 2026-06-27) ──
    # Drop template rows for events whose series (KX*GAME) markets are all
    # finalized. Replaces the disabled trade-activity ACTIVE_SCANNER. WS-based,
    # stateless, fail-open. Series settlement is a strict signal: once both
    # team sides return "Markets not found" on a WS subscribe, the event is
    # fully resolved on Kalshi and no quoter / arber / hedger row should remain
    # in the template. Drops prevent run.py's REST fallback from polling
    # finalized tickers forever (PGSD1 was being polled every ~600ms 4h after
    # settlement, contributing to 429 pressure).
    #
    # Kill switch: touch `disable_settled_scanner.flag` to skip this pass.
    # Force-keep: any event matching `force_start_tickers.txt` is NEVER dropped,
    # matching the override surface that existed for the prior scanner (DINOSGW
    # incident on 2026-06-10 forced disable; preserved here as a guard).
    if not os.path.exists("disable_settled_scanner.flag"):
        try:
            _settled_events = _probe_settled_series_events(final_output_rows)
        except Exception as _e:
            print(f"[SETTLED-SCANNER] unexpected error ({_e}); fail-open, no events dropped")
            _settled_events = set()

        if _settled_events:
            _force_keep = set(_load_force_start_patterns())
            _kept_back = {
                eb for eb in _settled_events
                if any(p.upper() in eb.upper() for p in _force_keep)
            }
            if _kept_back:
                print(f"[SETTLED-SCANNER] force-keep override (force_start_tickers.txt): "
                      f"{sorted(_kept_back)} — keeping despite settled series")
                _settled_events -= _kept_back

        if _settled_events:
            _orig_n = len(final_output_rows)
            final_output_rows = [
                r for r in final_output_rows
                if _event_base_from_ticker(r[1] if len(r) > 1 else "") not in _settled_events
            ]
            _n_dropped = _orig_n - len(final_output_rows)
            print(f"[SETTLED-SCANNER] dropped {_n_dropped} row(s) from "
                  f"{len(_settled_events)} settled event(s) (all series markets "
                  f"return active=False via WS): {sorted(_settled_events)}")
    else:
        print("[SETTLED-SCANNER] disabled via disable_settled_scanner.flag — no events dropped")

    # ══ PREGAME CONFIG GATE (26AUG11) ═══════════════════════════════════════
    # Decided here, in populate, NOT in the trading process. run.py reads plain
    # config values and knows nothing about pregame — it just sees a row that
    # says stop_quoting=True or a smaller max_position, and hot-reloads it.
    #
    # Why it exists: DNFHLE 26AUG11. KeSPA fielded academy rosters, the series
    # went 80/20 -> 20/80, and we were carrying ~24k lot-equivalents built
    # entirely pregame. The arber's taker gate existed but keyed off a 5c
    # price-move latch that false-fired 9.5 min BEFORE a real start on BROAP;
    # the maker had no pregame concept at all.
    #
    #   TAKERS  (arber/mma_arber/momentum/map_taker) -> stop_quoting = True
    #   MAKERS  (quoter)  -> max_position *= pregame_position_frac (0.1)
    #   HEDGERS           -> UNTOUCHED. Blocking a hedge pregame strands
    #                        exposure instead of protecting it.
    #
    # The maker cap is a FRACTION of each row's own in-game cap, not a fixed
    # number: a hardcoded ceiling silently fails to bind on small rows (an 8000
    # cap does nothing to a tier-2 row already authored at 7000) and has to be
    # re-tuned by hand every time sizing moves. 0.1 scales with whatever the row
    # is authored at, so tier-2 gets 700 and tier-1 BO5 gets 3999 for free.
    #
    # Source of truth is riot's TRUE game-1 first frame via pregame_state, which
    # populate wrote earlier this cycle from the anchor pass. Not-pregame (or no
    # opinion, e.g. every non-LoL sport) leaves rows exactly as authored, so the
    # cap reverts to the real one the moment the game starts.
    _pg_takers = _pg_capped = _pg_sized = 0
    try:
        import pregame_state
        _TAKER_EXEC = {"arber", "mma_arber", "momentum", "map_taker"}
        _i_stop = csv_header.index("stop_quoting")
        _i_maxpos = csv_header.index("max_position")
        _i_exec = csv_header.index("execution_type")
        _i_pfrac = (csv_header.index("pregame_position_frac")
                    if "pregame_position_frac" in csv_header else None)
        _i_cfgid = csv_header.index("config_id")
        _i_minedge = csv_header.index("min_edge")
        _i_vols = csv_header.index("volumes") if "volumes" in csv_header else None
        # Sentinel, not a tuned threshold: no cross-venue edge is ever ~1000c, so
        # every taker fire is suppressed while the bot stays alive and visible.
        # Chosen to be unmistakable in the [LEAD-LAG] line rather than plausible.
        _PG_BLOCK_EDGE = 999.0
        # config_id -> AUTHORED max_position, straight from the blueprint. This is
        # what makes the cap idempotent no matter how often, or from where, a row
        # passes through this block.
        _pg_authored_maxpos = {}
        _pg_authored_vols = {}
        for _tmpl_rows in param_maps.values():
            for _tr in _tmpl_rows:
                if len(_tr) > max(_i_cfgid, _i_maxpos):
                    try:
                        _pg_authored_maxpos[_tr[_i_cfgid]] = int(float(_tr[_i_maxpos] or 0))
                    except ValueError:
                        pass
                if _i_vols is not None and len(_tr) > max(_i_cfgid, _i_vols):
                    _pg_authored_vols[_tr[_i_cfgid]] = _tr[_i_vols]
        _pg_events = set()
        _pg_frac_seen = None
        for r in final_output_rows:
            _eb = _event_base_from_ticker(r[1] if len(r) > 1 else "")
            if not _eb or pregame_state.is_pregame(_eb) is not True:
                continue
            _pg_events.add(_eb)
            _exec = str(r[_i_exec] or "").strip().lower()
            if _exec in _TAKER_EXEC:
                # NOT stop_quoting. `stop_quoting=True` makes the manager skip
                # bot construction entirely (manager.py:457) and drop the row
                # from the hot-reload map (manager.py:537), so the event
                # DISAPPEARS from run.log — there is no object left to emit
                # [LEAD-LAG] and no way to tell "correctly gated" from "fell out
                # of the system". An unreachable edge threshold suppresses every
                # fire while keeping the bot alive, evaluating and logging, with
                # Thresh:999.0 on the [LEAD-LAG] line making the gate visible.
                if str(r[_i_stop]).strip().lower() in ("true", "1", "yes", "y"):
                    continue        # authored-off rows stay off; leave them alone
                try:
                    _prev = float(str(r[_i_minedge]).strip() or 0)
                except ValueError:
                    _prev = 0.0
                if _prev < _PG_BLOCK_EDGE:
                    r[_i_minedge] = str(_PG_BLOCK_EDGE)
                    _pg_takers += 1
            elif _exec == "quoter" and _i_pfrac is not None:
                try:
                    _frac = float(str(r[_i_pfrac]).strip() or 0)
                    _cur = int(float(str(r[_i_maxpos]).strip() or 0))
                except ValueError:
                    continue
                if not (0 < _frac < 1) or _cur <= 0:
                    continue
                # IDEMPOTENCE: scale the AUTHORED cap from market_parameters, not
                # whatever this row currently holds. The [PRESERVE] path above
                # re-injects rows read back out of the PREVIOUS
                # template_quoter_config.csv — already gated — and it runs before
                # this block. Multiplying the row's own value would decay an
                # API-failing event 3499 -> 349 -> 34 -> 3, once per cycle,
                # silently, looking exactly like the gate working correctly.
                _authored = _pg_authored_maxpos.get(r[_i_cfgid])
                _base = _authored if _authored else _cur
                # Floor at 1 so a small row can never be capped to zero, which
                # would read as "no limit" downstream rather than "tiny limit".
                _pg_frac_seen = _frac
                _cap = max(1, int(_base * _frac))
                if _cap < _cur:
                    r[_i_maxpos] = str(_cap)
                    _pg_capped += 1
                # CLIP SIZE too, on the same fraction. The cap bounds TOTAL
                # exposure; it does nothing about how big a single bite is. A
                # 750-lot clip against a 3499 cap is still a full-size fill into
                # a market that may be repricing on news we have not seen — the
                # DNFHLE burst was 95 fills in one minute, and per-fill size is
                # what determines how much of that we eat before the cap binds.
                # Scaled from the AUTHORED volumes for the same idempotence
                # reason as the cap (the [PRESERVE] path re-feeds gated rows).
                if _i_vols is not None:
                    _auth_v = _pg_authored_vols.get(r[_i_cfgid]) or str(r[_i_vols])
                    _parts = [p.strip() for p in str(_auth_v).split(",") if p.strip()]
                    try:
                        _scaled = [max(1, int(int(float(p)) * _frac)) for p in _parts]
                    except ValueError:
                        _scaled = None
                    if _scaled:
                        _new_v = ",".join(str(x) for x in _scaled)
                        if _new_v != str(r[_i_vols]):
                            r[_i_vols] = _new_v
                            _pg_sized += 1
        if _pg_events:
            print(f"[PREGAME-CFG] {len(_pg_events)} pregame event(s): "
                  f"{_pg_takers} taker row(s) -> min_edge={_PG_BLOCK_EDGE:g} "
                  f"(bot stays ALIVE and logs [LEAD-LAG]; stop_quoting would "
                  f"delete it), {_pg_capped} quoter cap(s) + {_pg_sized} clip "
                  f"size(s) x{_pg_frac_seen or 0.1:g} ({sorted(_pg_events)})")
    except Exception as _e:
        # Never block the config write. Failing here means rows go out exactly
        # as authored — i.e. pre-26AUG11 behavior — which is loud, not silent.
        print(f"[PREGAME-CFG] gate FAILED ({type(_e).__name__}: {_e}) — "
              f"configs written UNGATED at full size")

    # ══ TOXIC-STATE MAKER GATE (26AUG20) ════════════════════════════════════
    # Decided HERE, in populate, from Kalshi's own live game state. run.py reads
    # a plain `stop_quoting` value and knows nothing about round scores. This
    # replaces the run.py -> bo3_score_feed -> live_series_model(maker_suppress)
    # -> manager runtime path.
    #
    # Why the source changed: 26AUG20 G2 vs M80 (VCT Americas). bo3 listed the
    # fixture `current` while publishing `live_updates: None`, so the gate
    # failed OPEN and the quoter ran through an 11-11 map that ended 14-12 in
    # OT. Kalshi had the round score throughout, 2-12s fresh, and milestones are
    # keyed BY KALSHI EVENT TICKER — which deletes bo3's whole name-binding
    # layer, the documented "recurring failure mode ... fails SILENTLY".
    #
    # SCOPE: quoter rows only, CS2 + VAL only. Takers/hedgers/momentum are
    # untouched — the 7d fill analysis behind this gate found our resting QUOTES
    # get run over in the toxic regime while our map-triggered TAKERS are the
    # informed side of the same move and PROFIT. Dota/LoL were never gated
    # (bo3_score_feed.event_sport returns None for both).
    #
    # OBSERVABILITY: `stop_quoting=True` makes the manager skip bot construction
    # (manager.py:457) and drop the row from the hot-reload map (manager.py:537)
    # — which is exactly what we want, because dropping the row is what fires
    # `_execute_sync_cancel` on the maker's resting orders. The pregame gate
    # above avoids stop_quoting for TAKERS precisely because it deletes the row,
    # but a maker MUST have its orders cancelled, so here the deletion is the
    # point.
    #
    # The event does NOT disappear from run.log: only the quoter rows are gated,
    # and every CS2/VAL event also carries arber (series + map), momentum and
    # hedger rows which stay alive and keep emitting [LEAD-LAG] (verified
    # 26AUG20 on G2M80: 6 rows, 2 quoter gated, 4 untouched). Still log every
    # gated event LOUDLY below so populate's own log shows WHY the quoter went
    # quiet — that is the part run.log cannot tell you.
    _tox_gated = 0
    _tox_events = {}
    try:
        import kalshi_game_state as _kgs
        from bo3_score_feed import thresholds_for as _thr

        _i_stop = csv_header.index("stop_quoting")
        _i_exec = csv_header.index("execution_type")
        _i_cfgid = csv_header.index("config_id")

        # AUTHORED stop_quoting per config_id, straight from the blueprint.
        # IDEMPOTENCE: the [PRESERVE] path re-injects rows out of the PREVIOUS
        # template_quoter_config.csv — already gated — and runs before this
        # block. Restoring from the row's own value would latch a gated row
        # True forever once a map went toxic. Same trap the pregame cap hit.
        _tox_authored_stop = {}
        for _tmpl_rows in param_maps.values():
            for _tr in _tmpl_rows:
                if len(_tr) > max(_i_cfgid, _i_stop):
                    _tox_authored_stop[_tr[_i_cfgid]] = _tr[_i_stop]

        _tox_ebs = set()
        for r in final_output_rows:
            if str(r[_i_exec] or "").strip().lower() != "quoter":
                continue
            _eb = _event_base_from_ticker(r[1] if len(r) > 1 else "")
            if _eb and ("KXCS2" in _eb or "KXVALORANT" in _eb):
                _tox_ebs.add(_eb)

        _verdicts = _kgs.evaluate(sorted(_tox_ebs), _thr) if _tox_ebs else {}

        for r in final_output_rows:
            if str(r[_i_exec] or "").strip().lower() != "quoter":
                continue
            _eb = _event_base_from_ticker(r[1] if len(r) > 1 else "")
            _v = _verdicts.get(_eb)
            if not _v:
                continue
            _authored = _tox_authored_stop.get(r[_i_cfgid])
            if str(_authored).strip().lower() in ("true", "1", "yes", "y"):
                continue          # authored-off rows stay off; not ours to touch
            if _v["toxic"]:
                r[_i_stop] = "True"
                _tox_gated += 1
                _tox_events[_eb] = _v
            elif _authored is not None:
                r[_i_stop] = _authored    # restore blueprint (undo a prior gate)

        # Per-event LIVE SCORE line, printed every cycle for every CS2/VAL event
        # we evaluated — toxic or not. This is the only place the round score is
        # visible anywhere in the stack (run.log sees a plain stop_quoting value
        # and knows nothing about game state), so print it unconditionally
        # rather than only on suppression.
        for _eb, _v in sorted(_verdicts.items()):
            _sc = _v.get("score")
            _pk = _v.get("period")
            if _sc and _pk:
                try:
                    _gn = str(_pk).rsplit("_", 1)[-1]
                except Exception:
                    _gn = "?"
                _where = f"game {_gn} {_sc[0]}-{_sc[1]}"
            else:
                _where = _v.get("reason", "no live map")
            _age = _v.get("age_s")
            _agestr = "age=?" if _age is None else f"age={round(_age)}s"
            _act = "SUPPRESS quoter" if _v["toxic"] else "quote ok"
            print(f"[GAME-STATE] {_eb}: {_where}  {_agestr}  -> {_act}"
                  f"{'  [' + _v['reason'] + ']' if _v['toxic'] else ''}"
                  f"{'  STICKY' if _v.get('sticky') else ''}")

        if _tox_events:
            print(f"[TOXIC-CFG] {len(_tox_events)} event(s), {_tox_gated} quoter "
                  f"row(s) gated; {len(_verdicts)} evaluated")
        elif _verdicts:
            print(f"[TOXIC-CFG] {len(_verdicts)} CS2/VAL event(s) evaluated, "
                  f"none toxic")
    except Exception as _e:
        # Never block the config write. Failing here writes rows as authored —
        # i.e. UNGATED, the pre-26AUG20 behavior. Loud, not silent.
        print(f"[TOXIC-CFG] gate FAILED ({type(_e).__name__}: {_e}) — "
              f"quoter rows written UNGATED")

    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Override the header so `market_prefix` is replaced with `ticker` matching framework_config!
        # Append `capture_buffer_min` (per-event, sourced from parsed_markets.csv via _buffer_by_event)
        # so QuoterConfig has the value at runtime without re-reading parsed_markets.
        fixed_header = csv_header.copy()
        fixed_header[1] = "ticker"
        fixed_header.append("capture_buffer_min")
        writer.writerow(fixed_header)

        for r in final_output_rows:
            eb = _event_base_from_ticker(r[1] if len(r) > 1 else "")
            buf = _buffer_by_event.get(eb, 120.0)
            writer.writerow(list(r) + [buf])

    os.replace(tmp_path, tgt_path)
    print(f"\n[SUCCESS] Wrote {len(final_output_rows)} total routing layers atomically to {tgt_path}.")
    
    if new_probs_rows:
        file_exists = os.path.exists(prob_file)
        with open(prob_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ticker", "p1_start", "p2_start", "series_start", "g2_momentum", "g3_momentum"])
            for row in new_probs_rows:
                writer.writerow(row)
        print(f"[SUCCESS] Appended {len(new_probs_rows)} new baseline probability lock-ins to {prob_file}")

    # 5. Clean settled events from parsed_markets.csv
    if settled_urls:
        with open("parsed_markets.csv", "r") as f:
            pm_lines = f.readlines()
        with open("parsed_markets.csv", "w") as f:
            f.write(pm_lines[0])  # header
            kept = 0
            for line in pm_lines[1:]:
                if line.strip() not in settled_urls:
                    f.write(line)
                    kept += 1
        print(f"[CLEANUP] Removed {len(settled_urls)} settled events from parsed_markets.csv ({kept} remaining)")

if __name__ == '__main__':
    # ── Self-contained file logging (added 2026-07-15) ──
    # populate uses print() throughout, which previously went ONLY to the
    # launching terminal — no log survived for post-mortems (e.g. the EFTL
    # G2-refresh freeze). Tee stdout+stderr to populate_configs.log so output
    # is captured regardless of how the daemon is launched. Startup rotation
    # keeps the file bounded. Effective on next populate restart.
    import sys as _sys
    _LOG_PATH = "populate_configs.log"
    try:
        # Rotate if the existing log is large (>25MB) so we don't grow forever.
        if os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > 25 * 1024 * 1024:
            try:
                os.replace(_LOG_PATH, _LOG_PATH + ".1")
            except Exception:
                pass

        class _Tee:
            def __init__(self, *streams):
                self._streams = [s for s in streams if s is not None]
            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data); s.flush()
                    except Exception:
                        pass
            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass
            def isatty(self):
                return False

        _logf = open(_LOG_PATH, "a", buffering=1)  # line-buffered
        _stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')
        _logf.write(f"\n===== populate_configs daemon start {_stamp} (pid {os.getpid()}) =====\n")
        _sys.stdout = _Tee(_sys.__stdout__, _logf)
        _sys.stderr = _Tee(_sys.__stderr__, _logf)
        print(f"[LOG] stdout/stderr teed to {_LOG_PATH}")
    except Exception as _e:
        print(f"[LOG] file logging setup FAILED (continuing to terminal only): {_e}")

    print("Initializing Background Auto-Populator Daemon...")

    # Settled-series scanner (re-enabled 2026-06-27, replaces the disabled
    # TradeActivityTracker-based ACTIVE_SCANNER). Stateless: each populate
    # cycle WS-probes the series tickers via ws_market_status.probe_markets,
    # and drops template rows for any event whose series markets are all
    # finalized. See `_probe_settled_series_events` for details. No daemon
    # init needed — the filter runs inline inside run_populator.
    print("[DAEMON] Settled-series scanner ENABLED (WS-based).")

    while True:
        try:
            run_populator()
        except Exception as e:
            print(f"Daemon encountered a fatal pipeline error: {e}")

        now_str = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')
        from datetime import timedelta
        next_str = (datetime.now(ZoneInfo('America/New_York')) + timedelta(seconds=180)).strftime('%H:%M:%S %Z')
        print(f"[DAEMON] Last run completed: {now_str} | Next run: {next_str} (3 min)")
        time.sleep(180)
