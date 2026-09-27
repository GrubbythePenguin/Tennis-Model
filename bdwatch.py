#!/usr/bin/env python3
"""Classify break_dog events by their ACTUAL cause, not by event name.

QUIET MODE (default): routine `implausible_fit` blocks are COUNTED, not printed -- on
26SEP27 they were 83% of all events and every one looked identical, which buried the
signal. A rolling count is emitted every QUIET_EVERY of them so the rate stays visible
without a line per block. Set BDWATCH_VERBOSE=1 to print them all.

`fire_empty` conflated three unrelated outcomes and the old watcher labelled all of them
"PHANTOM", which pushed every interpretation toward "the book is fake". Over 26SEP27 the
true split was 7 filled, 1 genuine phantom, 2 no-level-cleared, 1 position-at-cap. Reads
stdin (tail -F) and prints one honest line per event."""
import sys, json

def loaded(c): return c + 7.0 * (c / 100.0) * (1 - c / 100.0)

def classify(r):
    t = r.get("type")
    ev = r.get("event", "?").split("-", 1)[-1][:22]
    fit = r.get("fit") or {}
    fitinfo = ""
    if fit.get("p") is not None:
        fitinfo = f" [p={fit['p']:.3f} n={fit.get('n')}]"
    if t == "fire":
        lots = r.get("lots", 0)
        pos = r.get("position") or {}
        room = f" room={pos.get('room')}" if pos else ""
        return f"FILLED    {ev} {lots} lots @ fair {r.get('fair_c')}c{fitinfo}{room}"
    if t == "skip":
        return f"blocked   {ev} {r.get('reason')}: {r.get('detail','')}{fitinfo}"
    if t == "skip_no_room":
        pos = r.get("position") or {}
        return (f"at cap    {ev} position {pos.get('current')}/{pos.get('max')} "
                f"— correct, not a book problem")
    if t == "fire_no_level":
        return f"no level  {ev} book present but nothing cleared the edge bar{fitinfo}"
    if t == "would_fire":
        return f"SHADOW    {ev} would fire, fair {r.get('fair_c')}c{fitinfo}"
    if t == "fire_empty":
        # legacy name: work out which of the three it really was
        rt = r.get("rest_top") or {}
        nb = rt.get("breaker_no_bids") or []
        if not nb:
            return f"PHANTOM   {ev} no REST book at all — the real phantom case"
        fair = r.get("fair_c") or 0
        best = max((fair - loaded(100 - float(a) * 100), 100 - float(a) * 100, float(b))
                   for a, b in nb)
        if best[0] >= 6:
            return (f"at cap?   {ev} a level DID clear (+{best[0]:.1f}c, {best[2]:.0f} lots) "
                    f"— position cap or sweep, NOT the book")
        return f"no level  {ev} best level only {best[0]:+.1f}c, below the bar"
    return f"{t} {ev}"

import os
VERBOSE = os.environ.get("BDWATCH_VERBOSE") == "1"
QUIET_EVERY = int(os.environ.get("BDWATCH_QUIET_EVERY", "10"))
_blocked = 0
_reasons = {}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    t = r.get("type")
    fit = r.get("fit") or {}
    routine = (t == "skip" and r.get("reason") == "implausible_fit")
    if routine and not VERBOSE:
        global_reason = "inverted" if "inverted" in str(fit.get("reason")) else "p at corner"
        _reasons[global_reason] = _reasons.get(global_reason, 0) + 1
        _blocked += 1
        if _blocked % QUIET_EVERY == 0:
            parts = ", ".join(f"{k} {v}" for k, v in sorted(_reasons.items()))
            print(f"[quiet] {_blocked} fit-blocks so far ({parts}) — "
                  f"latest {r.get('event','?').split('-',1)[-1][:20]} n={fit.get('n')}",
                  flush=True)
        continue
    print(classify(r), flush=True)
