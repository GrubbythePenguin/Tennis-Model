"""Resolve tradeable tennis markets and PROVE each one is on shard 3.

    python3 tennis_parsed_markets.py                     # today's tennis, to stdout
    python3 tennis_parsed_markets.py --out tennis_parsed_markets.csv
    python3 tennis_parsed_markets.py --event KXATPMATCH-26AUG26SAMHAR

WHY A SHARD CHECK IS THE WHOLE POINT. Kalshi split trading across matching engines on
26AUG24; tennis and baseball moved to shard 3 while everything this account has ever
traded stayed on shard 0. Two consequences that make an unchecked ticker dangerous:

  1. Collateral must be PREALLOCATED per shard. Orders against shard 3 fail until funds
     are moved there, and as of 26AUG26 shard 3 holds $0.00 of the account's $568,708.
  2. A write that explicitly targets a nonzero shard bills only that shard's write
     budget. An auto-routed write bills MULTIPLE buckets - so a quoter that omits
     exchange_index silently spends the esports system's budget too.

So every market this emits carries a verified exchange_index, and anything not on the
expected shard is REFUSED rather than defaulted. Verified 26AUG26: three tennis markets
returned exchange_index=3 and a KXLOLGAME market returned 0.

Read-only. No order endpoints are touched. Nothing here can trade.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt

HERE = os.path.dirname(os.path.abspath(__file__))
TENNIS_SHARD = 3
BASE = "https://api.elections.kalshi.com"


def _auth():
    """Signed session, reusing the esports account's key. Returns None if unavailable."""
    ENVF = "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env"
    try:
        sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
        # python-dotenv lives in the esports venv, not here. Parse the file directly so
        # this project does not depend on another project's interpreter.
        try:
            from dotenv import load_dotenv
            load_dotenv(ENVF)
        except ImportError:
            with open(ENVF) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        from kalshi_auth import KalshiAuth
        pk = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not pk.startswith("/"):
            pk = ("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/"
                  "esports_arbitrage/" + pk)
        return KalshiAuth(os.getenv("KALSHI_API_KEY_ID"), pk)
    except Exception as e:
        print(f"(auth unavailable: {e})", file=sys.stderr)
        return None


def market_shard(auth, ticker):
    """exchange_index for one market, or None.

    The /markets collection IGNORES a `ticker` filter - asking it for a tennis ticker
    returned an unrelated KXMVECROSSCATEGORY market on shard 1, which would have been
    read as 'tennis is on shard 1'. The per-ticker PATH form is the only reliable read.
    """
    import requests
    path = f"/trade-api/v2/markets/{ticker}"
    r = requests.get(BASE + path, headers=auth.get_headers("GET", path), timeout=15)
    if r.status_code != 200:
        return None, r.status_code
    m = (r.json() or {}).get("market") or {}
    return m.get("exchange_index"), 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", nargs="*", help="specific event tickers; default = today's board")
    ap.add_argument("--out", help="write CSV here")
    ap.add_argument("--rps", type=float, default=3.0)
    ap.add_argument("--expect-shard", type=int, default=TENNIS_SHARD)
    a = ap.parse_args()

    auth = _auth()
    if auth is None:
        print("cannot verify shards without credentials - refusing to emit markets")
        return 1
    feed = kt.Feed(rps=a.rps, verbose=False)

    events = a.event or [e["event_ticker"] for e in feed.matches()]
    rows, refused = [], []
    for ev in events:
        mkts = feed.markets(ev)
        if len(mkts) != 2:
            continue
        mil = feed.milestone(ev)
        det = (mil or {}).get("details") or {}
        for m in mkts:
            tk = m.get("ticker")
            if not tk:
                continue
            shard, code = market_shard(auth, tk)
            if shard != a.expect_shard:
                refused.append((tk, shard, code))
                continue
            b, ask = kt.top_of_book(m)
            rows.append(dict(event=ev, ticker=tk, exchange_index=shard,
                             player=m.get("yes_sub_title") or "",
                             bid=b, ask=ask, mid=kt.mid(m),
                             tournament=det.get("tournament_name") or "",
                             best_of=det.get("best_of") or "",
                             start=(mil or {}).get("start_date") or ""))

    print(f"{len(rows)} markets verified on shard {a.expect_shard}; "
          f"{len(refused)} refused")
    for tk, shard, code in refused[:10]:
        print(f"  REFUSED {tk}: exchange_index={shard!r} (HTTP {code})")
    for r in rows[:12]:
        print(f"  {r['ticker']:<48} shard={r['exchange_index']} "
              f"bid={r['bid']} ask={r['ask']}  {r['player'][:22]}")
    if len(rows) > 12:
        print(f"  ... {len(rows)-12} more")
    print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")

    if a.out and rows:
        p = a.out if a.out.startswith("/") else os.path.join(HERE, a.out)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
