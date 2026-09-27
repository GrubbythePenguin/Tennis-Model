"""SET-1 TAU TAKER BOT: the taker half of the tau strategy, run THROUGH run.py the
same way the esports taker is (execution_type = "set1_taker" config rows, one per
ITF ticker — see set3_market_parameters.csv SET1TAKER_*).

WHAT IT DOES. While quoter/tennis_set1_tau_model.py has an event ARMED (fresh 1-0
anchor, tau fair frozen), the LEADER ticker's bot instance checks the live top of
book: if fair - fee-loaded cost >= min_edge (config, 6c) on either route to long-
leader exposure (leader YES asks, or opponent NO at 100 - opp bid), it fires ONE
cross-book IOC sweep via calculate_cross_book_sweep — depth-aware, fee-loaded,
position-netted, capped by volumes/max_fire_size (2000). One shot per match, ever.

WIRING. Registered in manager.py next to SeriesTakerBot; it is in the is_arber
evaluation set because it must be evaluated even when NO theos exist — the tau
maker in shadow emits nothing, and the taker's live/shadow state is independent.
It reads the maker's state file (tapes/_set1_tau_state.json, mtime-cached) for the
armed window + frozen fair; it never computes its own trigger, so maker and taker
always agree on the anchor and the fair.

MAKER/TAKER EXCLUSION (same protocol as the model): tapes/set1_claims/<event>.claim
via O_CREAT|O_EXCL. Live taker claims immediately before firing; a claim owned by
"maker" is a terminal skip. The maker defers arming TAKER_GRACE_S when the anchor
is taker-eligible and set1_taker_live.flag is up, so this bot wins the race when
its condition holds at the anchor. Shadow (flag absent) logs would_fire and NEVER
claims — a shadow taker must not block a live maker.

Flags: set1_taker_live.flag (arm), disable_set1_taker.flag (kill).
State: tapes/_set1_taker_state.json. Events: tapes/set1_taker_events.jsonl.
Env: SET1_TAKER_WINDOW_S (fire window after trigger, default 30, clamp 10-90),
SET1_TAKER_MAX_FIRES (per UTC day, default 20).
"""
import json
import logging
import os
import time
from typing import Any, Dict, List

import position_store
from arber_bot import calculate_cross_book_sweep
from hedge_engine import get_loaded_cost_cents
from tennis_set1_tau_model import claim, claim_owner, in_maintenance_window, \
    _STATE_PATH as TAU_STATE_PATH

log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAPES = os.path.join(ROOT, "tapes")
STATE_PATH = os.path.join(TAPES, "_set1_taker_state.json")
EVENTS_PATH = os.path.join(TAPES, "set1_taker_events.jsonl")
LIVE_FLAG = os.path.join(ROOT, "set1_taker_live.flag")
KILL_FLAG = os.path.join(ROOT, "disable_set1_taker.flag")

MAX_FIRES_PER_DAY = int(os.environ.get("SET1_TAKER_MAX_FIRES", "50"))

_REST_CLIENT = None


def _rest_client():
    """Lazy authed KalshiClient for fire-time orderbook truth (reads only).

    Built exactly like run.py's client. Separate instance = separate read-token
    bucket, acceptable at 2 GETs per fire with fires capped per day.
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


def _window_s() -> float:
    try:
        return min(90.0, max(10.0, float(os.environ.get("SET1_TAKER_WINDOW_S", "30"))))
    except ValueError:
        return 30.0


# ---- module-level shared state (one latch across both legs' bot instances) ----
_tau_cache = {"mtime": -1.0, "events": {}}
_latch: Dict[str, dict] = {}
_latch_loaded = False


def _tau_events() -> Dict[str, dict]:
    try:
        mt = os.path.getmtime(TAU_STATE_PATH)
    except OSError:
        return {}
    if mt != _tau_cache["mtime"]:
        try:
            with open(TAU_STATE_PATH) as f:
                d = json.load(f)
            _tau_cache["events"] = d if isinstance(d, dict) else {}
            _tau_cache["mtime"] = mt
        except Exception:
            pass                       # keep last-good on a torn read
    return _tau_cache["events"]


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
            json.dump({k: {a: b for a, b in v.items() if not a.startswith("_")}
                       for k, v in _latch.items()}, f, indent=1)
        os.replace(tmp, STATE_PATH)
    except Exception:
        log.exception("[SET1-TAKER] state persist failed — one-shot latch is "
                      "memory-only until this succeeds")


def _emit(kind: str, event: str, **kw):
    rec = {"ts": time.time(), "type": kind, "event": event}
    rec.update(kw)
    try:
        with open(EVENTS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        log.exception("[SET1-TAKER] events jsonl write failed")


def _fires_today() -> int:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return sum(1 for v in _latch.values()
               if v.get("fired") and v.get("fired_day") == day)


class Set1TauTakerBot:
    """One-shot cross-book IOC sweeper for tau-armed 1-0 windows (leader leg only)."""

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
        event = ticker.rsplit("-", 1)[0]
        e = _tau_events().get(event)
        if not e or e.get("state") != "armed":
            return []
        if ticker != e.get("leader"):
            return []                          # one actor per event: the leader leg
        if event in _latch and _latch[event].get("state") == "done":
            return []
        now = time.time()
        if now - (e.get("trigger_ts") or 0) > _window_s():
            return []                          # maker window may run on; ours is over
        fair_c = e.get("fair_c")
        other = e.get("other")
        if fair_c is None or not other:
            return []
        # Kalshi weekly maintenance: frozen-but-readable books paint phantom
        # edge — never fire on an anchor priced inside the blackout, and never
        # fire DURING it either (validated 26SEP17: the only in-window eligible
        # was the only reversal). The maker gate lives in the tau model.
        if in_maintenance_window(e.get("anchor_ts")) or in_maintenance_window(now):
            _latch[event] = {"state": "done", "reason": "maintenance_window",
                             "fired": False, "ts": now}
            _save_latch()
            _emit("skip", event, reason="maintenance_window")
            return []

        # ---- top-of-book edge precheck (cheap; depth handled by the sweep) ----
        l_ask = top_level_offer if 0 < top_level_offer < 100 else None
        o_bid = (full_top_bids or {}).get(other)
        o_bid = o_bid if (o_bid is not None and 0 < o_bid < 100) else None
        min_edge = max(self.config.min_edge, self.config.min_absolute_edge, 0.0)
        legs = []
        if l_ask is not None:
            edge = fair_c - get_loaded_cost_cents(float(l_ask))
            if edge >= min_edge:
                legs.append({"route": "leader_yes", "px_c": l_ask, "edge_c": round(edge, 2)})
        if o_bid is not None:
            no_ask = 100.0 - o_bid
            edge = fair_c - get_loaded_cost_cents(float(no_ask))
            if edge >= min_edge:
                legs.append({"route": "other_no", "px_c": no_ask, "edge_c": round(edge, 2)})
        if not legs:
            return []                          # book can still improve inside the window

        detail = dict(leader=ticker, other=other, fair_c=fair_c,
                      pregame_vf=e.get("pregame_vf"), min_edge_c=min_edge,
                      top_book={"leader_ask": l_ask, "other_bid": o_bid}, legs=legs)

        if not os.path.exists(LIVE_FLAG):
            _latch[event] = {"state": "done", "reason": "shadow_would_fire",
                             "fired": False, "ts": now}
            _save_latch()
            _emit("would_fire", event, live=False, **detail)
            log.warning("[SET1-TAKER] %s SHADOW would fire (fair %.1fc): %s",
                        event, fair_c, legs)
            return []
        if _fires_today() >= MAX_FIRES_PER_DAY:
            _latch[event] = {"state": "done", "reason": "daily_fire_cap",
                             "fired": False, "ts": now}
            _save_latch()
            _emit("skip", event, reason="daily_fire_cap")
            return []
        owner = claim(event, "taker")
        if owner != "taker":
            _latch[event] = {"state": "done", "reason": f"claimed_by_{owner}",
                             "fired": False, "ts": now}
            _save_latch()
            _emit("skip", event, reason=f"claimed_by_{owner}")
            return []

        # ---- depth-aware cross-book sweep (same machinery as the esports taker)
        # REST truth, not WS/feed view. 26SEP18 YANOZX: IOC yes@62 into a feed
        # ask that sat at 62 for 19s after the miss — canceled, 0 filled. The
        # displayed ask was the sibling market's bid mirrored, not resting depth
        # on this book, and Kalshi does not match across sibling books. Same
        # lesson execution._rest_phantom_probe learned on esports (NIP/LNG
        # 2026-07-26). Two GETs per fire, fires are capped per day.
        raw_ob, opp_ob, rest_src = {}, {}, "rest"
        try:
            c = _rest_client()
            raw_ob = (c.get_orderbook(ticker) or {}).get("orderbook_fp") or {}
            opp_ob = (c.get_orderbook(other) or {}).get("orderbook_fp") or {}
        except Exception as e:
            log.warning("[SET1-TAKER] %s REST orderbook fetch failed (%s) — "
                        "falling back to manager view", event, str(e)[:120])
            rest_src = "manager"
            if full_market_state and ticker in full_market_state:
                raw_ob = (full_market_state[ticker].get("raw_ob") or {}).get("orderbook_fp", {}) or {}
            if full_market_state and other in full_market_state:
                opp_ob = (full_market_state[other].get("raw_ob") or {}).get("orderbook_fp", {}) or {}
        books = []
        if raw_ob.get("no_dollars"):
            books.append((raw_ob["no_dollars"], ticker, "yes"))     # NO bids = leader YES asks
        if opp_ob.get("yes_dollars"):
            books.append((opp_ob["yes_dollars"], other, "no"))      # opp YES bids = opp NO asks
        if not books and rest_src == "manager":
            # Last resort, only when REST itself failed AND the manager carries
            # no depth arrays: synthetic single-level books at the observed top,
            # sized at the fire cap. IOC limit still protects the price; a
            # phantom top just cancels unfilled (proven harmless 26SEP18).
            cap = self.config.max_fire_size
            if l_ask is not None:
                books.append(([[(100.0 - l_ask) / 100.0, cap]], ticker, "yes"))
            if o_bid is not None:
                books.append(([[o_bid / 100.0, cap]], other, "no"))
        detail["book_src"] = rest_src
        detail["rest_top"] = {
            "leader_no_bids": (raw_ob.get("no_dollars") or [])[-3:],
            "other_yes_bids": (opp_ob.get("yes_dollars") or [])[-3:],
        }
        # TOP TICK ONLY (26SEP18 operator choice): keep each leg's best price
        # level so the IOC limit sits exactly at the top tick — never walk
        # deeper levels, even ones that would clear the edge bar on their own.
        # THIN-TOP EXTENSION (26SEP21, after 39- and 20-lot fires): when the
        # top level holds fewer than SET1_TAKER_THIN_TOP lots, admit the second
        # level too — the sweep's per-level edge gate still decides whether it
        # is actually taken, so a thin top never buys a bad second level.
        thin = float(os.environ.get("SET1_TAKER_THIN_TOP", "1000"))
        _books = []
        for lvls, tk_, side in books:
            if not lvls:
                continue
            lv = sorted(lvls, key=lambda l: float(l[0]), reverse=True)
            keep = lv[:2] if float(lv[0][1]) < thin else lv[:1]
            _books.append((keep, tk_, side))
        books = _books
        base_vol = self.config.volumes[0] if self.config.volumes else 2000
        orders = calculate_cross_book_sweep(
            books=books,
            hedge_cost=100.0 - float(fair_c),
            min_edge=min_edge,
            # Floor the logit-scaled requirement at min_absolute_edge so the live
            # bar matches the backtested flat rule. 26SEP18 rerun on 431 anchors:
            # fires admitted only by the scaled bar were EV-flat (-0.4c/lot,
            # z=-0.10); all measured edge lives in the flat-6c population.
            min_absolute_edge=float(getattr(self.config, "min_absolute_edge", 0) or 0),
            scale_step_c=self.config.arb_scale_step_cents,
            base_vol=base_vol,
            max_fire_size=self.config.max_fire_size,
            max_position=self.config.max_position,
            current_pos=position_store.get_position(ticker),
            direction_sign=1,
        ) if books else []
        total = sum(o["size"] for o in orders)
        _latch[event] = {"state": "done",
                         "reason": "fired" if total else "fired_empty_book",
                         "fired": bool(total),
                         "fired_day": time.strftime("%Y-%m-%d", time.gmtime()),
                         "lots": total, "ts": now}
        _save_latch()
        _emit("fire" if total else "fire_empty", event, live=True, lots=total,
              orders=[{k: o[k] for k in ("ticker", "kalshi_side", "size", "limit_cents")}
                      for o in orders], **detail)
        if total:
            log.warning("[SET1-TAKER] %s FIRED %d lots (fair %.1fc, min_edge %.1fc): %s",
                        event, total, fair_c, min_edge,
                        [(o["ticker"].rsplit("-", 1)[-1], o["kalshi_side"],
                          o["size"], o["limit_cents"]) for o in orders])
        else:
            log.error("[SET1-TAKER] %s claimed but sweep found NO takeable depth "
                      "(books=%d) — maker is now excluded too; investigate",
                      event, len(books))
        return orders
