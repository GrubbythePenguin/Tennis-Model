"""tt_fill_lag.py — per-fill feed-lag post-mortem for the TT quoter.

For every TT fill in quoter/trades.csv, joins the screener row tape
(tapes/tt_screener_rows.jsonl, one row per match per ~12s) to answer:

    * what the FEED said the score was when we traded
    * when OUR SYSTEM first saw the match ended (status flip on the tape)
    * the fill -> end-detection delay, and how many POINTS landed in that span
      (per the feed; the real table is 2-10 points ahead of all of this)
    * settlement markout per contract, from Kalshi's own market `result`

A fill with a short delay and many points in the span is the delayed-feed
pickoff the band is supposed to price; a NEGATIVE delay means we filled after
our own system already knew the match was over — a quoting bug, not a feed
problem.

usage:  .venv/bin/python tt_fill_lag.py [--since-hours 24]
"""
import argparse
import csv
import json
import time
import urllib.request

TRADES = "quoter/trades.csv"
ROWS = "tapes/tt_screener_rows.jsonl"
ENDED = {"ended", "closed", "finished", "cancelled", "canceled"}


def _epoch(ts_str):
    return time.mktime(time.strptime(ts_str, "%Y-%m-%dT%H:%M:%S"))


def _points(score_str):
    """Total points played per the feed's score string '11-6, 12-14, 9-5'."""
    tot = 0
    for part in (score_str or "").split(","):
        part = part.strip()
        if "-" in part:
            x, _, y = part.partition("-")
            try:
                tot += int(x) + int(y)
            except ValueError:
                pass
    return tot


def load_rows():
    """event -> ordered [(epoch, status, points, state, score), ...]"""
    out = {}
    with open(ROWS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ev = r.get("event")
            if not ev:
                continue
            out.setdefault(ev, []).append(
                (_epoch(r["ts"]), (r.get("status") or "").lower(),
                 _points(r.get("score")), r.get("state"), r.get("score")))
    for v in out.values():
        v.sort()
    return out


def results_for(events):
    """event -> {ticker: 'yes'|'no'|''} from Kalshi settled results."""
    out = {}
    for ev in events:
        url = ("https://api.elections.kalshi.com/trade-api/v2/markets"
               f"?event_ticker={ev}")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                mkts = json.load(r).get("markets") or []
            out[ev] = {m["ticker"]: (m.get("result") or "") for m in mkts}
        except Exception:
            out[ev] = {}
        time.sleep(0.4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=float, default=24.0)
    a = ap.parse_args()
    cutoff = time.time() - a.since_hours * 3600

    fills = []
    with open(TRADES) as f:
        for r in csv.DictReader(f):
            if not (r.get("ticker") or "").startswith("KXTT"):
                continue
            ts = float(r["created_ts"])
            if ts < cutoff:
                continue
            fills.append(r)
    if not fills:
        print("no TT fills in window")
        return
    rows = load_rows()
    res = results_for(sorted({f["ticker"].rsplit("-", 1)[0] for f in fills}))

    hdr = (f"{'fill time':<9} {'ticker':<38} {'trade':<20} {'state@fill':<11} "
           f"{'det_delay':>9} {'pts_span':>8} {'settle':>6} {'pnl/ct':>7} {'pnl$':>8}")
    print(hdr)
    print("-" * len(hdr))
    tot_pnl = 0.0
    for f in sorted(fills, key=lambda r: float(r["created_ts"])):
        ts = float(f["created_ts"])
        ev = f["ticker"].rsplit("-", 1)[0]
        evrows = rows.get(ev) or []
        at = None
        for row in evrows:                       # last tape row at/before fill
            if row[0] <= ts:
                at = row
            else:
                break
        det = next((row for row in evrows if row[1] in ENDED), None)
        delay = pts = None
        if det is not None:
            delay = det[0] - ts
            if at is not None:
                pts = det[2] - at[2]
        settle = (res.get(ev) or {}).get(f["ticker"], "")
        px, qty = float(f["price"]), float(f["count"])
        pnl_ct = None
        if settle in ("yes", "no"):
            val = 1.0 if settle == f["side"] else 0.0
            pnl_ct = (val - px) if f["action"] == "buy" else (px - val)
        pnl = None if pnl_ct is None else pnl_ct * qty
        if pnl is not None:
            tot_pnl += pnl
        state = "-".join(map(str, at[3])) if at and at[3] else "?"
        print(f"{time.strftime('%H:%M:%S', time.localtime(ts)):<9} "
              f"{f['ticker']:<38} "
              f"{f['action']} {f['side']} {qty:.0f}@{100*px:.0f}c{'':<3} "
              f"{state:<11} "
              f"{'?' if delay is None else format(delay, '+8.0f')+'s':>9} "
              f"{'?' if pts is None else pts:>8} "
              f"{settle or '?':>6} "
              f"{'?' if pnl_ct is None else format(100*pnl_ct, '+.1f')+'c':>7} "
              f"{'?' if pnl is None else format(pnl, '+8.2f'):>8}")
    print(f"\ntotal settled pnl: {tot_pnl:+.2f} across {len(fills)} fills "
          f"({a.since_hours:.0f}h window)")
    print("det_delay: fill -> first tape row with ended status "
          "(negative = filled AFTER our system knew it was over). "
          "pts_span: feed points between fill and detection.")


if __name__ == "__main__":
    main()
