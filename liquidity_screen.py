"""Is this market deep enough to be worth capturing (let alone trading)?

    python3 liquidity_screen.py --series KXITFMATCH KXITFWMATCH [--max-spread 0.01]
                                [--min-depth 10000] [--band 0.02]

THE TEST (operator spec 26AUG24, for ITF futures):
    * spread <= 1c pre-match, AND
    * >= 10,000 contracts of resting liquidity within 2c of the midpoint,
      ON BOTH SIDES.

WHY BOTH SIDES. One-sided depth is not a tradeable market — you can get in and not
out, and a midpoint computed against a hollow side is not a price anyone would deal
at. The whole model is scored against that midpoint, so a fake mid corrupts the
measurement before any trading question arises.

BOOK MECHANICS (the trap). /markets/{t}/orderbook returns `orderbook_fp`, and BOTH
`yes_dollars` and `no_dollars` are BID ladders — there is no ask ladder. The YES ask
side is the NO bids reflected: a NO bid at 0.80 is an offer to sell YES at 0.20.
Treating no_dollars as asks inverts the book and reports depth that is not there.

    YES-side depth  = sum of YES bid sizes at price >= mid - band
    ASK-side depth  = sum of NO  bid sizes at price >= (1 - mid) - band

Ladders are `depth`-truncated by the API, so a large --depth is requested; a market
that needs more than that to clear the bar is not one where the extra rungs matter.
"""
import argparse
import sys
import time

import kalshi_tennis as kt


def ladder_depth(feed, ticker, mid, band, depth=None):
    """(yes_side_contracts, ask_side_contracts) resting within `band` of mid."""
    # NO depth param. The API caps it and rejects anything over the max with a 400,
    # which this client turns into None — i.e. a silent zero-depth reading on every
    # market. Omitting it returns the full ladder.
    body = feed.get(f"/markets/{ticker}/orderbook", {} if depth is None else {"depth": depth})
    ob = (body or {}).get("orderbook_fp") or (body or {}).get("orderbook") or {}
    yes = ob.get("yes_dollars") or []
    no = ob.get("no_dollars") or []

    def total(levels, floor):
        s = 0
        for lv in levels:
            try:
                # sizes are DECIMAL STRINGS ("13718.80"), not ints — int() raises and
                # would silently drop every level
                px, sz = float(lv[0]), float(lv[1])
            except (TypeError, ValueError, IndexError):
                continue
            if px >= floor:
                s += sz
        return s

    return total(yes, mid - band), total(no, (1.0 - mid) - band)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", required=True)
    ap.add_argument("--max-spread", type=float, default=0.01)
    ap.add_argument("--min-depth", type=int, default=10000)
    ap.add_argument("--band", type=float, default=0.02)
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--pregame-only", action="store_true", default=True)
    a = ap.parse_args()

    feed = kt.Feed(rps=a.rps, verbose=False)
    passed, seen = [], 0
    print(f"screen: spread <= {a.max_spread*100:.0f}c, depth >= {a.min_depth:,} within "
          f"{a.band*100:.0f}c of mid, BOTH sides\n")
    print(f"{'event':34} {'player':20} {'mid':>6} {'sprd':>5} {'bid depth':>10} "
          f"{'ask depth':>10}  verdict")

    for s in a.series:
        body = feed.get("/events", {"series_ticker": s, "status": "open", "limit": 200}) or {}
        for e in body.get("events") or []:
            ev = e.get("event_ticker")
            mkts = feed.markets(ev)
            if len(mkts) != 2:
                continue
            seen += 1
            ok_all, lines = True, []
            for m in mkts:
                t = m.get("ticker")
                b, k = kt.top_of_book(m)
                mid = kt.mid(m)
                if mid is None or b is None or k is None:
                    lines.append((t, m.get("yes_sub_title"), None, None, 0, 0))
                    ok_all = False
                    continue
                spread = k - b
                yd, ad = ladder_depth(feed, t, mid, a.band)
                ok = (spread <= a.max_spread + 1e-9 and yd >= a.min_depth
                      and ad >= a.min_depth)
                ok_all &= ok
                lines.append((t, m.get("yes_sub_title"), mid, spread, yd, ad))
            for t, nm, mid, sp, yd, ad in lines:
                v = "" if mid is None else ("PASS" if (sp <= a.max_spread + 1e-9
                                                       and yd >= a.min_depth
                                                       and ad >= a.min_depth) else "fail")
                print(f"{(t or '')[:34]:34} {str(nm)[:20]:20} "
                      f"{'  -' if mid is None else f'{mid:6.3f}'} "
                      f"{'  -' if sp is None else f'{sp*100:4.0f}c'} "
                      f"{yd:10,.0f} {ad:10,.0f}  {v}")
            if ok_all:
                passed.append(ev)
            time.sleep(0.05)

    print(f"\n{len(passed)} of {seen} events pass on BOTH sides")
    for p in passed:
        print(f"  {p}")
    print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")


if __name__ == "__main__":
    main()
