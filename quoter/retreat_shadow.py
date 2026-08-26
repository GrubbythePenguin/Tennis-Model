"""Recency-flow retreat tracker (TRADE_RETREAT_SCOPE.md).

PRODUCTION as of 26AUG18: PositionAdjuster.adjust_theos applies the retreat
shift to QUOTER theos for rows whose blueprint sets retreat_cap_cents > 0
(per-row arm switch; 0 = off, shadow readout only). MAKERS ONLY — taker
gates (arber/momentum) are untouched by design (FINAL SCOPE, operator
26AUG15). The long-term static position skew is applied ON TOP as a separate
additive term (static = inventory risk, this term = flow information).

Per-row blueprint columns (market_parameters.csv → template → QuoterConfig):
  retreat_cap_cents      0     max shift; 0 = row disarmed (shadow only)
  retreat_f_cap          0.18  decayed f where shift pins at cap (≈25%/300s)
  retreat_f0             0.022 onset (≈3% of max accumulated in 300s)
  retreat_half_life_sec  300   decay half-life
  retreat_include_takers 0     lever: taker fills count toward accumulation
                               (both channels always recorded; retroactive)

Mechanics:
  Fills are stored raw as (ts, team, signed_qty, is_taker) per event
  (sibling tickers net onto one axis: long A == short B), so the decayed
  sum is computed EXACTLY at query time under the querying row's half-life
  — per-row half-lives stay correct and the taker lever is retroactive
  within the window.
  f(team)  = sum(delta_toward_team * 0.5 ** (age / half_life)) / max_position
  Steady accumulation of X of max over one half-life reads f = 0.721 * X,
  so thresholds live in decayed space: onset "3% in 300s" -> F0 = 0.022;
  cap "25% in 300s" -> F_CAP = 0.18.
  shift(f) = cap * ((f - F0) / (F_CAP - F0))^2, clamped to [0, cap] —
  quadratic ramp (FIFO-calibrated toxicity is convex); shift PINS at the
  cap (no outright quote-pull in the final scope; /CAP tag in the readout).
"""
import threading
import time

# Module defaults == FINAL LOCKED CONFIG (operator, 26AUG15). Per-row
# blueprint values override these at query time; the module constants remain
# the shadow-readout fallback for rows/callers that pass no config.
HALF_LIFE_SEC = 300.0
F0 = 0.022         # decayed-f onset (≈3% of max accumulated in 300s)
F_CAP = 0.18       # decayed-f where shift hits the cap (≈25% of max in 300s)
CAP_CENTS = 10.0
_GC_LOTS = 0.5     # drop an event once every decayed net is below this

# Fills older than this many (largest-seen) half-lives contribute < 0.03%
# and are pruned so the per-event store stays bounded in long slates.
_PRUNE_HALF_LIVES = 12.0

# Global default for the taker lever (per-row retreat_include_takers wins
# when a config is passed). BOTH channels are always recorded, so flipping
# is retroactive on flow already inside the decay window.
INCLUDE_TAKER_FILLS = False

_lock = threading.Lock()
# event_base -> {"fills": [(ts, team, delta, is_taker), ...] (ts ascending),
#                "hl_max": largest half-life this event has been queried with}
_events = {}


def _prune(ev, now):
    horizon = now - _PRUNE_HALF_LIVES * ev["hl_max"]
    fills = ev["fills"]
    i = 0
    n = len(fills)
    while i < n and fills[i][0] < horizon:
        i += 1
    if i:
        del fills[:i]


def record_fill(ticker, action, side, qty, is_taker=False, now=None):
    """Record a fill. Direction mapping matches position_store.apply_fill:
    buy-yes / sell-no -> long this ticker's team; sell-yes / buy-no -> long
    the opponent (recorded as negative toward this ticker's team).
    Maker and taker flow are stored per-fill; whether taker flow counts
    toward f is decided at query time (per-row lever)."""
    parts = ticker.split("-")
    if len(parts) != 3 or not qty:
        return  # series-game tickers only (MAP/TOTALMAPS have extra segments)
    event_base = "-".join(parts[:2])
    team = parts[-1]
    is_buy = "BUY" in action.upper()
    is_yes = "YES" in side.upper()
    delta = float(qty) if (is_buy == is_yes) else -float(qty)
    if now is None:
        now = time.time()
    with _lock:
        ev = _events.get(event_base)
        if ev is None:
            ev = _events[event_base] = {"fills": [], "hl_max": HALF_LIFE_SEC}
        ev["fills"].append((now, team, delta, bool(is_taker)))
        _prune(ev, now)


def record_maker_fill(ticker, action, side, qty, now=None):
    """Back-compat wrapper: maker-fill recording."""
    record_fill(ticker, action, side, qty, is_taker=False, now=now)


def get_f(event_base, team, max_position, now=None,
          half_life=None, include_takers=None):
    """Decayed signed accumulation toward `team` (maker flow, plus taker
    flow iff the lever is on), as a fraction of max_position. Exact under
    the caller's half-life. 0.0 if no surviving flow on the event."""
    if not max_position:
        return 0.0
    if half_life is None or half_life <= 0:
        half_life = HALF_LIFE_SEC
    if include_takers is None:
        include_takers = INCLUDE_TAKER_FILLS
    if now is None:
        now = time.time()
    with _lock:
        ev = _events.get(event_base)
        if ev is None:
            return 0.0
        if half_life > ev["hl_max"]:
            ev["hl_max"] = half_life
        _prune(ev, now)
        if not ev["fills"]:
            del _events[event_base]
            return 0.0
        toward = 0.0
        gross = 0.0
        for ts, t, delta, taker in ev["fills"]:
            decayed = delta * 0.5 ** (max(0.0, now - ts) / half_life)
            gross += abs(decayed)
            if taker and not include_takers:
                continue
            toward += decayed if t == team else -decayed
        if gross < _GC_LOTS:
            del _events[event_base]
            return 0.0
    return toward / float(max_position)


def shift_cents(f, cap_cents=None, f0=None, f_cap=None):
    """Quadratic retreat schedule. `f` is the accumulation fraction on the
    side being shifted (callers pass abs(f) / the positive side); <= f0
    yields 0. Defaults = module (FINAL LOCKED) params."""
    if cap_cents is None:
        cap_cents = CAP_CENTS
    if f0 is None:
        f0 = F0
    if f_cap is None:
        f_cap = F_CAP
    if f <= f0 or f_cap <= f0:
        return 0.0
    x = (f - f0) / (f_cap - f0)
    return min(cap_cents, cap_cents * x * x)


def params_from_conf(conf):
    """(cap, f0, f_cap, half_life, include_takers) for a QuoterConfig,
    falling back to module defaults for missing attributes (older configs)
    and for cap==0 rows the DISPLAY cap stays the module default so the
    shadow readout keeps printing the would-be schedule."""
    cap = float(getattr(conf, "retreat_cap_cents", 0.0) or 0.0)
    f0 = float(getattr(conf, "retreat_f0", F0) or F0)
    f_cap = float(getattr(conf, "retreat_f_cap", F_CAP) or F_CAP)
    hl = float(getattr(conf, "retreat_half_life_sec", HALF_LIFE_SEC)
               or HALF_LIFE_SEC)
    inc_tk = bool(getattr(conf, "retreat_include_takers", INCLUDE_TAKER_FILLS))
    return cap, f0, f_cap, hl, inc_tk


def status_str(event_base, team, max_position, now=None, conf=None):
    """Compact readout for the [LEAD-LAG] line, e.g.
    'Rtr:f(TS)=+12.3%/s2.2c' (+ '/CAP' when the shift is pinned at the cap).
    f>0 means we have been passively accumulating `team`; the shift applies
    against maker adds there. With a config the row's own retreat params
    drive the numbers (disarmed rows display the module-default 10c schedule
    so the shadow tape stays live); prefix flips to 'Rtr+tk:' when the taker
    accumulation lever is on, so the tape self-documents its mode."""
    if conf is not None:
        cap, f0, f_cap, hl, inc_tk = params_from_conf(conf)
        if cap <= 0:
            cap = CAP_CENTS  # display-only: row disarmed, show would-be shift
    else:
        cap, f0, f_cap, hl, inc_tk = (CAP_CENTS, F0, F_CAP, HALF_LIFE_SEC,
                                      INCLUDE_TAKER_FILLS)
    f = get_f(event_base, team, max_position, now=now,
              half_life=hl, include_takers=inc_tk)
    s = shift_cents(f, cap_cents=cap, f0=f0, f_cap=f_cap)
    cap_tag = "/CAP" if f >= f_cap else ""
    tag = "Rtr+tk" if inc_tk else "Rtr"
    return f"{tag}:f({team})={f * 100:+.1f}%/s{s:.1f}c{cap_tag}"
