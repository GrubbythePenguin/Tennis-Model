"""CROSS-BOOK depth: how much can actually be done in a direction, both books summed.

    python3 liquidity_xbook.py --series KXITFMATCH KXITFWMATCH
                               [--max-spread 0.01] [--min-depth 10000] [--band 0.02]

WHY CROSS-BOOK. A Kalshi tennis event is two markets on the SAME mutually exclusive
outcome, so YES on player A and NO on player B are the same economic position. The
liquidity to get long A therefore lives in BOTH books, and measuring one in isolation
understates it — potentially by half.

    long  A  =  A's NO-bid ladder    (lifting NO bids buys YES on A)
             +  B's YES-bid ladder   (lifting YES bids on B buys NO on B == YES on A)

    short A  =  A's YES-bid ladder + B's NO-bid ladder

BOTH `yes_dollars` and `no_dollars` are BID ladders — there is no ask ladder. To buy
YES you lift NO bids: a NO bid at 0.72 is an offer of YES at 0.28. Every price below
is converted into the cost of the A-direction position before the 2c band is applied,
so the band always means "within 2c of the A midpoint", never 2c of some other book's
quote.

PRICE FILTER. A level counts toward `long A` only if acquiring A there costs
<= mid_A + band, and toward `short A` only if selling A there yields >= mid_A - band.

    /markets/{t}/orderbook caps `depth` at 100 and 400s above it; the param is
    omitted. Sizes are DECIMAL STRINGS and must be parsed as floats.
"""
import argparse
import sys
import time

import kalshi_tennis as kt


def _levels(feed, ticker):
    body = feed.get(f"/markets/{ticker}/orderbook", {})
    ob = (body or {}).get("orderbook_fp") or {}
    def parse(k):
        out = []
        for lv in ob.get(k) or []:
            try:
                out.append((float(lv[0]), float(lv[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return out
    return parse("yes_dollars"), parse("no_dollars")


def xbook_depth(feed, tick_a, tick_b, mid_a, band):
    """(long_a, short_a) contracts available within `band` of mid_a, both books."""
    a_yes, a_no = _levels(feed, tick_a)
    b_yes, b_no = _levels(feed, tick_b)
    lo, hi = mid_a - band, mid_a + band

    # long A: acquire A-exposure for <= hi. Both routes cost (1 - level_price), so
    # both filter on level_price >= 1 - hi (== mid_b - band).
    long_a = 0.0
    for n, sz in a_no:                 # lift A's NO bid at n -> buy YES on A at (1-n)
        if (1.0 - n) <= hi:
            long_a += sz
    for y, sz in b_yes:                # lift B's YES bid at y -> buy NO on B at (1-y) == long A
        if (1.0 - y) <= hi:
            long_a += sz

    short_a = 0.0
    for y, sz in a_yes:                # hit A's YES bids: sell A at y
        if y >= lo:
            short_a += sz
    for n, sz in b_no:
        # Lifting B's NO bid at n buys YES on B at cost (1-n), which is economically
        # SELLING A at price n. So the filter is n >= lo — NOT (1-n) >= lo, which
        # would sum the deep out-of-the-money end of the ladder and inflate depth
        # several-fold (37,285 vs a true 7,661 on the first market checked).
        if n >= lo:
            short_a += sz
    return long_a, short_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", required=True)
    ap.add_argument("--max-spread", type=float, default=0.01)
    ap.add_argument("--min-depth", type=float, default=10000)
    ap.add_argument("--band", type=float, default=0.02)
    ap.add_argument("--rps", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    feed = kt.Feed(rps=a.rps, verbose=False)
    rows = []
    for s in a.series:
        body = feed.get("/events", {"series_ticker": s, "status": "open", "limit": 200}) or {}
        for e in body.get("events") or []:
            ev = e.get("event_ticker")
            mk = feed.markets(ev)
            if len(mk) != 2:
                continue
            ta, tb = mk[0].get("ticker"), mk[1].get("ticker")
            ba, ka = kt.top_of_book(mk[0])
            bb, kb = kt.top_of_book(mk[1])
            ma, mb = kt.mid(mk[0]), kt.mid(mk[1])
            if None in (ba, ka, bb, kb, ma, mb):
                continue
            spread = max(ka - ba, kb - bb)
            lo_, sh_ = xbook_depth(feed, ta, tb, ma, a.band)
            rows.append((spread, -min(lo_, sh_), ev, ma, lo_, sh_,
                         mk[0].get("yes_sub_title")))
            time.sleep(0.03)

    rows.sort()
    print(f"CROSS-BOOK depth, band {a.band*100:.0f}c, bar {a.min_depth:,.0f} both directions\n")
    print(f"{'event':32} {'mid':>6} {'sprd':>5} {'long':>11} {'short':>11}  verdict")
    for spread, negd, ev, ma, lo_, sh_, nm in rows[:a.top]:
        ok = spread <= a.max_spread + 1e-9 and lo_ >= a.min_depth and sh_ >= a.min_depth
        print(f"{ev[:32]:32} {ma:6.3f} {spread*100:4.0f}c {lo_:11,.0f} {sh_:11,.0f}  "
              f"{'PASS' if ok else 'fail'}")
    ok_rows = [r for r in rows
               if r[0] <= a.max_spread + 1e-9 and -r[1] >= a.min_depth]
    print(f"\n{len(rows)} events; {sum(1 for r in rows if r[0] <= a.max_spread+1e-9)} "
          f"at <= {a.max_spread*100:.0f}c spread; {len(ok_rows)} PASS both criteria")
    for r in ok_rows:
        print(f"  {r[2]}")
    print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")


if __name__ == "__main__":
    main()
