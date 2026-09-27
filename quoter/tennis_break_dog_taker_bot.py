"""BREAK-DOG TAKER BOT (ITF only): when a break lands and the player who broke is
still cheap, take them if our model says they are underpriced by >= min_edge net of
fees. Runs THROUGH run.py like the set-1 taker (execution_type = "break_dog_taker",
see set3_market_parameters.csv BREAKDOG_*).

WHY. Measured over 5,774 ITF breaks in 1,065 matches (26AUG28-26SEP24 tapes): buying
the breaker when the fee-loaded ask clears our model fair by >= 6c and the ask is
under 50c returned +13.7c/contract held to settlement (95% CI +10.7..+16.8, clustered
by match, n=930 across 477 matches). Set 1 +14.7c, set 2 +10.3c, set 3 +18.6c. The
edge is NOT a standing discount on ITF dogs: the same dogs bought at a game open with
no break in the prior 120s return -0.03c (n=5,925). Model edge also sorts realised
P&L *within* every price band, so this is a signal and not a blind dog purchase; the
regression is realised ~= 1.33c + 0.599 * model_edge_c. Challenger is excluded because
our model agrees with that market there (mean model edge ~0) and the gate almost never
fires - the signal enforces the tier restriction on its own.

WHAT IT DOES. Tails tapes/<event>.jsonl for the break (a game won by the RETURNER),
takes the breaker-oriented model fair from the freshest tape row, and checks the LIVE
manager book. If fair - loaded_cost >= min_edge on either route to long-breaker
exposure (breaker YES asks, or opponent NO at 100 - opp bid) AND the ask is under
DOG_MAX_C, it fires ONE cross-book IOC sweep via calculate_cross_book_sweep -
depth-aware, fee-loaded, position-netted, capped by max_fire_size (200 for the pilot).

WHY THE ASK AND NOT THE MID. min_edge here means exactly what it means in
tennis_set1_taker_bot: fair_c - get_loaded_cost_cents(ask). The backtest above is
quoted on the same basis (paying the ask, fee loaded), NOT on the mid - median ITF
spread at these moments is 1.0c, so the two are close, but the gate is the strict one.

CAPS, and why they are shaped this way. Two independent latches, because an attempt
that fills NOTHING is not a trade and must not consume the opportunity:

  per BREAK   `<ev>|s<n>|<games>`  set on every attempt, fill or not. This is what
              stops the same break being re-attempted on every tick inside its 30s
              window (one break = one attempt, ever).
  per SET+SIDE `<ev>|s<n>|<side>`  counts FILLS ONLY (total > 0), PER BREAKER. A
              fire_empty -- gate cleared, claim taken, REST book turned out phantom --
              leaves this untouched, so a later break can still trade. In a
              phantom-heavy book a naive latch burns its one attempt on ghosts and
              never reaches a fillable level.
              Default 3 per side per set, NOT 1: fills in a thin ITF book are
              routinely partial, and a cap of 1 would strand a 20-lot fill with no way
              to top up on the next break. max_position does the real bounding.
              KEYED BY SIDE (operator direction 26SEP27): if A breaks and we buy A,
              then B breaks back with edge on B, that is a DIFFERENT trade and we want
              it. Buying both sides of one event is not additive risk -- they are
              sibling markets, one settles 100 and the other 0, so a combined cost
              under 100c is a locked win. The old `<ev>|s<n>` key blocked it.
  per MATCH+SIDE BREAK_DOG_MAX_PER_MATCH (default 9) fills per breaker across sets.
  per TICKER  max_position (200) inside calculate_cross_book_sweep -- this is the real
              money bound, and it is what stops repeated fills on ONE player from
              stacking: the sweep is position-netted, so once 200 lots are held the
              next fire on that ticker sizes to zero on its own.

There is deliberately NO business cap on fires per day (operator direction 26SEP27).
BREAK_DOG_MAX_FIRES survives only as a runaway backstop at an absurd default; the real
bound is per-match fills x the size of the board.
Latch persists to tapes/_break_dog_state.json so a quoter restart cannot double-fire.

SHADOW BY DEFAULT. Live only while ROOT/break_dog_live.flag exists; every candidate,
fire and skip goes to tapes/break_dog_events.jsonl either way. Fills ride to
settlement (exiting into the post-set price measured better risk-adjusted, +16.0c at
half the standard error, but needs an exit path this bot does not have - see NEXT).

Flags: break_dog_live.flag (arm), disable_break_dog.flag (kill).
Env: BREAK_DOG_WINDOW_S (fire window after the break, default 30, clamp 5-90),
     BREAK_DOG_MAX_FIRES (runaway backstop only, default 10000),
     BREAK_DOG_MAX_PER_MATCH (fills per breaker per match, default 3),
     BREAK_DOG_DOG_MAX_C (ask must be under this, default 50),
     BREAK_DOG_MAX_PER_SET (default 1),
     BREAK_DOG_MAX_TAPE_AGE_S (tape staleness bound, default 90),
     BREAK_DOG_THIN_TOP (admit 2nd book level under this top size, default 1000).

NEXT (not built): sell into the post-set price instead of holding to settlement.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import position_store
from arber_bot import calculate_cross_book_sweep
from hedge_engine import get_loaded_cost_cents
from tennis_set1_tau_model import in_maintenance_window

log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAPES = os.path.join(ROOT, "tapes")
STATE_PATH = os.path.join(TAPES, "_break_dog_state.json")
EVENTS_PATH = os.path.join(TAPES, "break_dog_events.jsonl")
LIVE_FLAG = os.path.join(ROOT, "break_dog_live.flag")
KILL_FLAG = os.path.join(ROOT, "disable_break_dog.flag")

SERIES = ("KXITFMATCH", "KXITFWMATCH")   # the tier the edge was measured in
TAIL_BYTES = 400_000                     # a 30s-old break is always inside this
MAX_SPREAD_D = 0.25                      # same real-book bound the study used

_REST_CLIENT = None


def _envf(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(os.environ.get(name, str(default)))))
    except ValueError:
        return default


def _rest_client():
    """Lazy authed KalshiClient for fire-time orderbook truth (reads only).

    Built exactly like run.py's client, mirroring tennis_set1_taker_bot.
    """
    global _REST_CLIENT
    if _REST_CLIENT is None:
        from dotenv import load_dotenv
        load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
        from kalshi_auth import KalshiAuth
        from kalshi_client import KalshiClient
        k = os.getenv("KALSHI_API_KEY_ID", "").strip()
        p = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not p.startswith("/"):
            p = f"/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage/{p}"
        _REST_CLIENT = KalshiClient(KalshiAuth(k, p))
    return _REST_CLIENT


# ---- module-level shared state (one latch and one cache across both legs) ----
_latch: Dict[str, dict] = {}
_latch_loaded = False
_tape_cache: Dict[str, tuple] = {}       # event -> (mtime, detection dict or None)
_meta_cache: Dict[str, tuple] = {}


def _load_latch():
    global _latch, _latch_loaded
    if _latch_loaded:
        return
    try:
        with open(STATE_PATH) as f:
            d = json.load(f)
        _latch = d if isinstance(d, dict) else {}
    except Exception:
        _latch = {}
    _latch_loaded = True


def _save_latch():
    tmp = STATE_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(_latch, f, indent=1)
        os.replace(tmp, STATE_PATH)
    except Exception:
        log.exception("[BREAK-DOG] state persist failed — one-shot latch is "
                      "memory-only until this succeeds")


def _emit(kind: str, event: str, **kw):
    rec = {"ts": time.time(), "type": kind, "event": event}
    rec.update(kw)
    try:
        with open(EVENTS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        log.exception("[BREAK-DOG] events jsonl write failed")


def _fires_today() -> int:
    """Fills (not attempts) recorded today. Runaway backstop only."""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return sum(int(v.get("fills", 0) or 0) for v in _latch.values()
               if v.get("fired_day") == day)


def _fills_this_match(event: str, side: str) -> int:
    """Fills for ONE breaker across every set of one match."""
    return sum(int(v.get("fills", 0) or 0) for k, v in _latch.items()
               if k.startswith(f"{event}|s") and k.endswith(f"|{side}")
               and k.count("|") == 2)


def _meta(event: str) -> Tuple[Optional[str], Optional[str]]:
    """(me_ticker, opp_ticker) from the poller's log json — the 'me'/'opp' mapping."""
    if event in _meta_cache:
        return _meta_cache[event]
    got = (None, None)
    try:
        d = json.load(open(os.path.join(TAPES, event + ".log.json")))
        m = d.get("meta") or {}
        got = (m.get("me_ticker"), m.get("opp_ticker"))
    except Exception:
        pass
    _meta_cache[event] = got
    return got


def _real_book(b: dict) -> bool:
    if not b:
        return False
    if any(b.get(k) is None for k in ("bid_me", "ask_me", "bid_opp", "ask_opp")):
        return False
    return ((b["ask_me"] - b["bid_me"]) <= MAX_SPREAD_D
            and (b["ask_opp"] - b["bid_opp"]) <= MAX_SPREAD_D)


def detect_break(rows: List[dict]) -> Optional[dict]:
    """Most recent break in the tail, or None.

    A break is a game won by the RETURNER. The server is taken as the MODAL
    `server` value across the game's rows, because the raw field flickers without
    the game score changing (measured 10 of 41 rows in one match) — keying on the
    instantaneous value closes games early and records false breaks.

    Also reports the break-advantage transition (breaks made minus conceded within
    the set, breaker-oriented) as TELEMETRY ONLY. It is approximate: the tail may
    not reach the start of the set. The fire gate never reads it.
    """
    last = None
    cur_set = None
    curg = None
    srvcnt: Dict[str, int] = {}
    bmade = bopp = 0                      # within cur_set, 'me' perspective
    for r in rows:
        s = r.get("state") or {}
        if s.get("sets_me") is None or not s.get("_points_known"):
            continue
        sm, so = s["sets_me"], s["sets_opp"]
        si = sm + so + 1
        if si != cur_set:
            cur_set, curg, srvcnt, bmade, bopp = si, None, {}, 0, 0
        gm, go = s["games_me"], s["games_opp"]
        g = (gm, go)
        if curg is None:
            curg = g
        if g != curg:
            won_me = g[0] > curg[0]
            modal = max(srvcnt, key=srvcnt.get) if srvcnt else None
            if modal in ("me", "opp") and (won_me != (modal == "me")) \
                    and curg != (6, 6):              # tiebreak "breaks" excluded
                before = (bmade - bopp) if won_me else (bopp - bmade)
                if won_me:
                    bmade += 1
                else:
                    bopp += 1
                b = r.get("book") or {}
                last = {"ts": r.get("ts"), "breaker_me": won_me, "set_no": si,
                        "games": list(curg), "before": before, "after": before + 1,
                        "model": r.get("model"), "book_ok": _real_book(b)}
            curg, srvcnt = g, {}
        srv = s.get("server")
        if srv in ("me", "opp"):
            srvcnt[srv] = srvcnt.get(srv, 0) + 1
    return last


def _detection(event: str) -> Tuple[Optional[dict], Optional[dict], float]:
    """(break dict, freshest tape row, tape age). mtime-cached per event."""
    p = os.path.join(TAPES, event + ".jsonl")
    try:
        st = os.stat(p)
    except OSError:
        return None, None, 1e9
    age = time.time() - st.st_mtime
    hit = _tape_cache.get(event)
    if hit and hit[0] == st.st_mtime:
        return hit[1], hit[2], age
    rows = []
    try:
        with open(p, "rb") as f:
            f.seek(max(0, st.st_size - TAIL_BYTES))
            for line in f.read().decode("utf-8", "replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        log.warning("[BREAK-DOG] %s tape unreadable: %s", event, str(e)[:120])
        return None, None, age
    brk = detect_break(rows)
    latest = None
    for r in reversed(rows):
        if (r.get("state") or {}).get("sets_me") is not None:
            latest = r
            break
    _tape_cache[event] = (st.st_mtime, brk, latest)
    return brk, latest, age


class BreakDogTakerBot:
    """One-shot cross-book IOC sweeper for the cheap side of a fresh ITF break."""

    def __init__(self, config: Any):
        self.config = config
        self.active = True
        self.baseline_loaded = True
        _load_latch()

    def evaluate(self, market_state: Any = None, top_level_bid: float = 0.0,
                 top_level_offer: float = 100.0, bid_theo: float = None,
                 offer_theo: float = None, full_market_state: Dict[str, Any] = None,
                 full_top_bids: Dict[str, float] = None,
                 full_top_offers: Dict[str, float] = None,
                 **kwargs) -> List[Dict[str, Any]]:
        if not self.active or os.path.exists(KILL_FLAG):
            return []
        ticker = self.config.ticker
        if not ticker.startswith(SERIES):
            return []                      # tier restriction, belt to the config's braces
        event = ticker.rsplit("-", 1)[0]
        now = time.time()

        brk, latest, age = _detection(event)
        if not brk or latest is None:
            return []
        if age > _envf("BREAK_DOG_MAX_TAPE_AGE_S", 90, 10, 600):
            return []                      # poller stalled; the book view is not ours to trust
        if now - (brk.get("ts") or 0) > _envf("BREAK_DOG_WINDOW_S", 30, 5, 90):
            return []                      # break is stale; edge decays ~0.03c/s

        me_tk, opp_tk = _meta(event)
        if not me_tk or not opp_tk:
            return []
        breaker = me_tk if brk["breaker_me"] else opp_tk
        other = opp_tk if brk["breaker_me"] else me_tk
        if ticker != breaker:
            return []                      # one actor per event: the breaker's leg

        # TWO latches (see module docstring): one per SET counting FILLS, one per
        # BREAK counting ATTEMPTS. The per-break key is what keeps a single break from
        # re-attempting every tick inside its 30s window.
        side = breaker.rsplit("-", 1)[-1]          # e.g. "KUB"
        key = f"{event}|s{brk['set_no']}|{side}"   # fills, PER BREAKER
        bkey = f"{event}|s{brk['set_no']}|g{brk['games'][0]}-{brk['games'][1]}"
        slot = _latch.get(key) or {}
        if _latch.get(bkey, {}).get("tried"):
            return []                      # this exact break already attempted
        # LEGACY STATE. Two older shapes exist and both are side-BLIND, so they are
        # consulted only as an upper bound on THIS side -- deliberately conservative
        # across a restart, at the cost of possibly skipping one fill on the other side
        # of a set that already traded before the restart.
        filled = int(slot.get("fills", 0) or 0)
        if not filled:
            legacy = _latch.get(f"{event}|s{brk['set_no']}") or {}
            if legacy.get("fired"):
                filled = int(legacy.get("fires", 0) or 0)
            else:
                filled = int(legacy.get("fills", 0) or 0)
        if filled >= int(_envf("BREAK_DOG_MAX_PER_SET", 3, 1, 12)):
            return []                      # this SIDE already filled in this set
        if _fills_this_match(event, side) >= int(_envf("BREAK_DOG_MAX_PER_MATCH", 9, 1, 40)):
            _emit("skip", event, reason="match_fill_cap", set_no=brk["set_no"])
            return []

        mdl = brk.get("model")
        if mdl is None:
            mdl = latest.get("model")
        if mdl is None:
            return []                      # no fit behind the price -> never price it
        fair_c = round(100.0 * (mdl if brk["breaker_me"] else 1.0 - mdl), 2)

        # Kalshi weekly maintenance paints phantom edge on frozen-but-readable
        # books — same guard the set-1 taker carries.
        if in_maintenance_window(brk.get("ts")) or in_maintenance_window(now):
            _emit("skip", event, reason="maintenance_window", set_no=brk["set_no"])
            return []

        # ---- live top-of-book gate (depth handled by the sweep) ----
        dog_max = _envf("BREAK_DOG_DOG_MAX_C", 50, 5, 100)
        min_edge = max(self.config.min_edge, self.config.min_absolute_edge, 0.0)
        b_ask = top_level_offer if 0 < top_level_offer < 100 else None
        o_bid = (full_top_bids or {}).get(other)
        o_bid = o_bid if (o_bid is not None and 0 < o_bid < 100) else None
        legs = []
        for route, px in (("breaker_yes", b_ask),
                          ("other_no", None if o_bid is None else 100.0 - o_bid)):
            if px is None or px >= dog_max:
                continue                   # the dog gate is on the price we PAY
            edge = fair_c - get_loaded_cost_cents(float(px))
            if edge >= min_edge:
                legs.append({"route": route, "px_c": round(px, 2),
                             "edge_c": round(edge, 2)})
        if not legs:
            return []                      # book can still improve inside the window

        # FIT PROVENANCE. implied_model.check_split warns that a p<=q fit is physically
        # impossible and "prices built off such a fit should not be traded". Measured
        # 26SEP27 over 930 historical gate-clearing breaks: 20.8% were priced off an
        # already-inverted fit, and their realised edge (+12.87c) is indistinguishable
        # from the physical ones (+13.91c). So this is LOGGED, NOT GATED -- a hard p>q
        # bar would cut a fifth of opportunities for no measured benefit. Revisit on
        # live data rather than on the backtest.
        _f = ((latest.get("fit") or {}).get("rolling") or {})
        detail_fit = {"p": _f.get("p"), "q": _f.get("q"), "n": _f.get("n"),
                      "inverted": (None if _f.get("p") is None or _f.get("q") is None
                                   else bool(_f["p"] <= _f["q"]))}
        detail = dict(breaker=ticker, other=other, fair_c=fair_c, fit=detail_fit,
                      set_no=brk["set_no"], games=brk["games"],
                      transition=f"{brk['before']:+d}->{brk['after']:+d}",
                      break_ts=brk["ts"], tape_age_s=round(age, 1),
                      dog_max_c=dog_max, min_edge_c=min_edge,
                      top_book={"breaker_ask": b_ask, "other_bid": o_bid}, legs=legs)

        if not os.path.exists(LIVE_FLAG):
            # shadow never fills, so it must not consume the set's fill slot --
            # only mark this break as seen so it is not re-logged every tick.
            _latch[bkey] = {"tried": True, "reason": "shadow_would_fire", "ts": now}
            _save_latch()
            _emit("would_fire", event, live=False, **detail)
            log.warning("[BREAK-DOG] %s set%d SHADOW would fire (fair %.1fc): %s",
                        event, brk["set_no"], fair_c, legs)
            return []
        if _fires_today() >= int(_envf("BREAK_DOG_MAX_FIRES", 10000, 1, 100000)):
            _emit("skip", event, reason="runaway_backstop", set_no=brk["set_no"])
            return []

        # ---- depth-aware cross-book sweep, REST truth not the feed view ----
        # The feed can show a sibling market's bid mirrored as an ask on this
        # book; Kalshi does not match across sibling books, so an IOC into it
        # cancels 0-filled. Same lesson as set-1 (26SEP18 YANOZX). 2 GETs/fire.
        raw_ob, opp_ob, rest_src = {}, {}, "rest"
        try:
            c = _rest_client()
            raw_ob = (c.get_orderbook(ticker) or {}).get("orderbook_fp") or {}
            opp_ob = (c.get_orderbook(other) or {}).get("orderbook_fp") or {}
        except Exception as e:
            log.warning("[BREAK-DOG] %s REST orderbook fetch failed (%s) — "
                        "falling back to manager view", event, str(e)[:120])
            rest_src = "manager"
            if full_market_state and ticker in full_market_state:
                raw_ob = (full_market_state[ticker].get("raw_ob") or {}).get("orderbook_fp", {}) or {}
            if full_market_state and other in full_market_state:
                opp_ob = (full_market_state[other].get("raw_ob") or {}).get("orderbook_fp", {}) or {}
        books = []
        if raw_ob.get("no_dollars"):
            books.append((raw_ob["no_dollars"], ticker, "yes"))   # NO bids = breaker YES asks
        if opp_ob.get("yes_dollars"):
            books.append((opp_ob["yes_dollars"], other, "no"))    # opp YES bids = opp NO asks
        detail["book_src"] = rest_src
        # Top tick only, with the set-1 thin-top extension: admit the 2nd level
        # when the top holds fewer than THIN_TOP lots. The sweep's per-level edge
        # gate still decides, so a thin top never buys a bad second level.
        thin = _envf("BREAK_DOG_THIN_TOP", 1000, 1, 100000)
        _books = []
        for lvls, tk_, side in books:
            if not lvls:
                continue
            lv = sorted(lvls, key=lambda l: float(l[0]), reverse=True)
            _books.append((lv[:2] if float(lv[0][1]) < thin else lv[:1], tk_, side))
        # WHAT REST ACTUALLY RETURNED. Without this, fire_empty cannot be told apart:
        # (a) REST had no resting levels at all -> the displayed top was phantom, or
        # (b) REST had levels but every one failed the sweep's per-level edge gate.
        # Those mean opposite things and the first live fire_empty (26SEP27 WANKUB) could
        # only be diagnosed by hand-reading the tape. Top 3 levels per side, cheap.
        # ALL FOUR SIDES. 26SEP27 HATGOL: the WS book read HAT 17/18 for six consecutive
        # samples while REST's executable ask was 27c -- a 9.5c error, and the WS pair was
        # a perfect sibling mirror (18 = 100-82, 83 = 100-17). That same book feeds the
        # model's `vig_free` observations, so the fit is being trained on prices ~9c from
        # executable, which is what inverts p/q. To measure the error in the MID (what the
        # model needs, not just the ask) both sides of BOTH books must be logged.
        def _lv(ob, k):
            return [[str(a), str(b)] for a, b in (ob.get(k) or [])[-3:]]
        detail["rest_top"] = {
            "breaker_yes_bids": _lv(raw_ob, "yes_dollars"),   # breaker BID side
            "breaker_no_bids": _lv(raw_ob, "no_dollars"),     # breaker ASK side (100-p)
            "other_yes_bids": _lv(opp_ob, "yes_dollars"),
            "other_no_bids": _lv(opp_ob, "no_dollars"),
            "n_books": len(books),
        }
        # the displayed book the gate actually used, so the pair is self-contained
        detail["shown_book"] = {"breaker_bid": top_level_bid, "breaker_ask": b_ask,
                                "other_bid": o_bid}
        books = _books
        base_vol = self.config.volumes[0] if self.config.volumes else 200
        orders = calculate_cross_book_sweep(
            books=books,
            hedge_cost=100.0 - float(fair_c),
            min_edge=min_edge,
            min_absolute_edge=float(getattr(self.config, "min_absolute_edge", 0) or 0),
            scale_step_c=self.config.arb_scale_step_cents,
            base_vol=base_vol,
            max_fire_size=self.config.max_fire_size,
            max_position=self.config.max_position,
            current_pos=position_store.get_position(ticker),
            direction_sign=1,
        ) if books else []
        total = sum(o["size"] for o in orders)
        # The attempt is always recorded against the BREAK. The SET's fill counter
        # moves only on a real fill, so a phantom book leaves the set open to retry
        # on its next break.
        _latch[bkey] = {"tried": True, "lots": total, "ts": now,
                        "reason": "fired" if total else "fired_empty_book"}
        if total:
            _latch[key] = {"fills": filled + 1,
                           "fired_day": time.strftime("%Y-%m-%d", time.gmtime()),
                           "lots": slot.get("lots", 0) + total, "ts": now}
        _save_latch()
        _emit("fire" if total else "fire_empty", event, live=True, lots=total,
              orders=[{k: o[k] for k in ("ticker", "kalshi_side", "size", "limit_cents")}
                      for o in orders], **detail)
        if total:
            log.warning("[BREAK-DOG] %s set%d FIRED %d lots (fair %.1fc, "
                        "min_edge %.1fc): %s", event, brk["set_no"], total, fair_c,
                        min_edge,
                        [(o["ticker"].rsplit("-", 1)[-1], o["kalshi_side"],
                          o["size"], o["limit_cents"]) for o in orders])
        else:
            log.error("[BREAK-DOG] %s set%d cleared the gate but the sweep found "
                      "NO takeable depth (books=%d) — investigate",
                      event, brk["set_no"], len(books))
        return orders
