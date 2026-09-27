"""poll_tt.py — table tennis (TT Elite Series) tape: theo vs market. READ-ONLY.

Lean sibling of poll_tennis.py for KXTTELITEMATCH. No quoting, no fitting loop —
it backs out (p, q) ONCE from the first usable vig-free market price via
table_tennis_model.pq_from_match_prob-style solve at the observed state, then
prices every subsequent state at that fixed (p, q) and logs theo next to the
book. The whole point is to see how a static pregame anchor tracks the market.

Reuses kalshi_tennis.Feed: same shared GET bucket, same hard rps cap, same
429 surrender, same disable_tennis_poller.flag kill switch.

TT live_data shape (type=table_tennis_match, provider oddsmatrix). VERIFIED on
an ended match only:
    competitorN_overall_score   SETS won
    period_scores               [{competitor1_score, competitor2_score, number,
                                  type: "set"}, ...] — completed sets at least;
                                whether the ongoing set appears with a running
                                score is UNVERIFIED, which is why every row logs
                                the raw details. scores{"1st Set": {home, away}}
                                is the same data keyed by name.
    winner / is_complete / status ("ended" when done)
No server field — TT serve rotation is reconstructible from parity anyway
(2 serves each, alternating; every point from 10-10), so theo is computed under
BOTH first-server hypotheses and both are logged along with their average.
The two differ only by ~1-2c at game points, ~0c elsewhere.

usage:
    python3 poll_tt.py scan [--limit 40]
    python3 poll_tt.py watch --event KXTTELITEMATCH-... [--interval 4]
            [--markets-every 2] [--serve-share 0.5] [--target 0.62]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import kalshi_tennis as kt
from table_tennis_model import match_win_prob, P0_MEN, Q0_MEN, BEST_OF

# TT gets its OWN kill switch. disable_tennis_poller.flag was present when this
# was written (tennis capture parked), and rebinding here means parking tennis
# does not silently kill TT captures — and vice versa. kt.disabled() reads the
# module global, so this one assignment covers every Feed.get() in this process.
kt.KILL_FLAG = "disable_tt_poller.flag"

SERIES = "KXTTELITEMATCH"
MILESTONE_TYPE = "table_tennis_match"
ORDINAL = {1: "1st Set", 2: "2nd Set", 3: "3rd Set", 4: "4th Set", 5: "5th Set"}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------ state parse

def tt_state(det):
    """(sets1, sets2, pts1, pts2) for competitor1, or None if unparsable.

    Points come from the period_scores entry numbered sets_done+1 (the ongoing
    set). A just-finished set that is already counted in overall_score has
    number == sets_done and is deliberately NOT used — pricing it again would
    double-count the set. Between sets the state is simply (a, b, 0, 0).
    """
    try:
        a = int(det.get("competitor1_overall_score"))
        b = int(det.get("competitor2_overall_score"))
    except (TypeError, ValueError):
        return None
    cur = a + b + 1
    x = y = 0
    for ps in det.get("period_scores") or []:
        if ps.get("number") == cur and ps.get("type") == "set":
            x = int(ps.get("competitor1_score") or 0)
            y = int(ps.get("competitor2_score") or 0)
            break
    else:
        sc = (det.get("scores") or {}).get(ORDINAL.get(cur, ""), {})
        if isinstance(sc, dict) and sc.get("home") is not None:
            x, y = int(sc.get("home") or 0), int(sc.get("away") or 0)
    return a, b, x, y


def theo(p, q, a, b, x, y):
    """(avg, theo_if_c1_served_first_this_game, theo_if_not) for competitor1."""
    t0 = match_win_prob(p, q, a, b, x, y, True)
    t1 = match_win_prob(p, q, a, b, x, y, False)
    return 0.5 * (t0 + t1), t0, t1


def solve_pq(target, a, b, x, y, serve_share=0.5, p0=P0_MEN, q0=Q0_MEN):
    """(p, q) such that competitor1's match prob AT THE GIVEN STATE == target,
    averaging over the first-server hypotheses. Same logit-space edge split as
    table_tennis_model.pq_from_match_prob, generalised to a mid-match anchor."""
    lp = math.log(p0 / (1 - p0))
    lq = math.log(q0 / (1 - q0))
    ws, wr = 2 * serve_share, 2 * (1 - serve_share)

    def pq(k):
        return 1 / (1 + math.exp(-(lp + ws * k))), 1 / (1 + math.exp(-(lq + wr * k)))

    lo, hi = -12.0, 12.0
    for _ in range(80):
        k = 0.5 * (lo + hi)
        p, q = pq(k)
        m = theo(p, q, a, b, x, y)[0]
        if m < target:
            lo = k
        else:
            hi = k
    return pq(0.5 * (lo + hi))


# ----------------------------------------------------------------------- scan

def cmd_scan(a):
    feed = kt.Feed(rps=a.rps)
    body = feed.get("/events", {"series_ticker": SERIES, "status": "open",
                                "limit": 200}) or {}
    evs = (body.get("events") or [])[:a.limit]
    print(f"resolving milestones for {len(evs)} open events...", file=sys.stderr)
    mids, meta = [], {}
    for e in evs:
        body = feed.get("/milestones", {"limit": 200,
                                        "related_event_ticker": e["event_ticker"]})
        ms = [m for m in (body or {}).get("milestones") or []
              if m.get("type") == MILESTONE_TYPE]
        if not ms:
            continue
        mids.append(ms[0]["id"])
        meta[ms[0]["id"]] = e
    det = feed.live_data(mids)
    rows = []
    for mid, e in meta.items():
        d = det.get(mid) or {}
        st = (d.get("status") or "?").lower()
        s = tt_state(d)
        sc = f"{s[0]}-{s[1]} ({s[2]}-{s[3]})" if s else ""
        rows.append((d.get("start_time") or "", st, sc, e["event_ticker"],
                     e.get("title") or ""))
    for start, st, sc, ev, title in sorted(rows):
        live = "  <-- LIVE" if st not in kt._NOT_LIVE else ""
        print(f"{start[:16]:<17} {st:<12} {sc:<14} {ev:<38} {title}{live}")


# ---------------------------------------------------------------------- watch

def cmd_watch(a):
    feed = kt.Feed(rps=a.rps)
    body = feed.get("/milestones", {"limit": 200, "related_event_ticker": a.event})
    ms = [m for m in (body or {}).get("milestones") or []
          if m.get("type") == MILESTONE_TYPE]
    if not ms:
        sys.exit(f"no {MILESTONE_TYPE} milestone for {a.event}")
    mid_id = ms[0]["id"]

    mkts = feed.markets(a.event)
    det0 = feed.live_data([mid_id]).get(mid_id) or {}
    c1 = det0.get("competitor1_id") or ""
    by_uuid = {(m.get("custom_strike") or {}).get("table_tennis_competitor"): m
               for m in mkts}
    if c1 not in by_uuid or len(mkts) != 2:
        sys.exit(f"cannot bind markets to competitors by UUID "
                 f"({len(mkts)} markets, c1={c1[:8]}…) — refusing to guess.")
    t1 = by_uuid[c1]["ticker"]
    t2 = next(m["ticker"] for m in mkts if m["ticker"] != t1)
    name1 = det0.get("competitor1_name") or "competitor1"
    name2 = det0.get("competitor2_name") or "competitor2"
    print(f"watching {a.event}: THEO IS FOR {name1} ({t1}); opp {name2} ({t2})")

    tape_p = os.path.join("tapes", f"{a.event}.jsonl")
    os.makedirs("tapes", exist_ok=True)
    meta = {"type": "meta", "ts": _now(), "event": a.event, "series": SERIES,
            "best_of": BEST_OF, "p0": P0_MEN, "serve_share": a.serve_share,
            "me": name1, "me_ticker": t1, "opp": name2, "opp_ticker": t2,
            "anchor": None}
    with open(tape_p, "a") as f:
        f.write(json.dumps(meta) + "\n")

    p = q = None
    cycles = misses = 0
    while True:
        cycles += 1
        if a.max_cycles and cycles > a.max_cycles:
            print(f"\nreached --max-cycles {a.max_cycles}, stopping."); break
        if kt.disabled():
            print(f"\nkill flag {kt.KILL_FLAG} present — stopping."); break
        try:
            det = feed.live_data([mid_id]).get(mid_id)
        except kt.RateLimited as e:
            print(f"\n*** {e} — stopping to protect the shared GET bucket.",
                  file=sys.stderr)
            break
        if det is None:                       # fetch failure, NOT a finished match
            misses += 1
            back = min(a.interval * (2 ** (misses - 1)), 60.0)
            print(f"\n[{time.strftime('%H:%M:%S')}] live_data empty "
                  f"(miss {misses}) — retry in {back:.0f}s", file=sys.stderr)
            time.sleep(back)
            continue
        misses = 0
        status = (det.get("status") or "").strip().lower()
        live = kt.is_live(det)

        # book: real GET every markets_every live cycles, every 15x pre-match
        every = max(a.markets_every if live else a.markets_every * 15, 1)
        book = {}
        px = None
        wide = None
        if cycles % every == 1 % every:
            mm = {m.get("ticker"): m for m in feed.markets(a.event)}
            b1, a1 = kt.top_of_book(mm.get(t1) or {})
            b2, a2 = kt.top_of_book(mm.get(t2) or {})
            book = {"bid_me": b1, "ask_me": a1, "bid_opp": b2, "ask_opp": a2}
            px = kt.vig_free(kt.mid(mm.get(t1) or {}), kt.mid(mm.get(t2) or {}))
            # a vig-free mid derived from a wide book is not an anchor candidate
            wide = (b1 is None or a1 is None or a1 - b1 > a.max_spread) and \
                   (b2 is None or a2 is None or a2 - b2 > a.max_spread)

        st = tt_state(det)
        # anchor once: first cycle with a state and a price from a sane book
        if p is None and st is not None and \
                (a.target is not None or (px is not None and wide is False)):
            p, q = solve_pq(px, *st, serve_share=a.serve_share) if a.target is None \
                else solve_pq(a.target, *st, serve_share=a.serve_share)
            tgt = px if a.target is None else a.target
            meta["anchor"] = {"ts": _now(), "target": tgt, "state": list(st),
                              "p": p, "q": q}
            with open(tape_p, "a") as f:
                f.write(json.dumps({"type": "anchor", **meta["anchor"]}) + "\n")
            print(f"\nanchored at state {st} target={tgt:.3f}: "
                  f"p={p:.4f} q={q:.4f}")

        row = {"ts": _now(), "status": status, "state": list(st) if st else None,
               "vig_free": px, "book": book, "theo": None, "theo_srv1": None,
               "theo_srv2": None, "details": det}
        if p is not None and st is not None:
            avg, h0, h1 = theo(p, q, *st)
            row.update(theo=round(avg, 4), theo_srv1=round(h0, 4),
                       theo_srv2=round(h1, 4))
        with open(tape_p, "a") as f:
            f.write(json.dumps(row) + "\n")

        sc = f"{st[0]}-{st[1]} ({st[2]}-{st[3]})" if st else "?"
        print(f"\r[{time.strftime('%H:%M:%S')}] {status:<10} {sc:<14} "
              f"mkt={'-' if px is None else format(px, '.3f')} "
              f"theo={'-' if row['theo'] is None else format(row['theo'], '.3f')} "
              f"({feed.n_get}g/{feed.n_429}x) ", end="", flush=True)

        if det.get("winner") or status in ("ended", "closed", "finished"):
            print(f"\nmatch over (status={status}).")
            break
        time.sleep(a.interval if live else a.pregame_interval)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--rps", type=float, default=2.0)
    s.set_defaults(fn=cmd_scan)
    w = sub.add_parser("watch")
    w.add_argument("--event", required=True)
    w.add_argument("--interval", type=float, default=4.0)
    w.add_argument("--pregame-interval", type=float, default=20.0)
    w.add_argument("--markets-every", type=int, default=2)
    w.add_argument("--serve-share", type=float, default=0.5)
    w.add_argument("--target", type=float, default=None,
                   help="anchor to this prob for competitor1 instead of the market")
    w.add_argument("--max-spread", type=float, default=0.15,
                   help="widest top-of-book spread that can seed the anchor")
    w.add_argument("--max-cycles", type=int, default=0)
    w.add_argument("--rps", type=float, default=2.0)
    w.set_defaults(fn=cmd_watch)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
