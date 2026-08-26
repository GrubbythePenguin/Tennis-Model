"""Place ONE resting contract on a tennis market, verify shard-3 routing, then cancel.

    python3 tennis_test_order.py                    # dry run: show the payload, send nothing
    python3 tennis_test_order.py --send             # actually place, verify, cancel

WHAT THIS PROVES, AND WHY IT IS WORTH A REAL ORDER. Nothing on this account has ever
written to exchange shard 3. Tennis moved there on 26AUG24 and the whole quoting stack
predates it. Six things have to work together before a quoter can run, and none of them
can be checked by reading code:

    1. the signed POST is accepted at the V2 events path
    2. exchange_index=3 is honoured rather than auto-routed
    3. shard-3 collateral is actually drawable (the $10,000 funded 26AUG26)
    4. the order comes back on GET with exchange_index=3
    5. subaccount 0 on shard 3 is the account it lands in
    6. the batched DELETE cancels it - cancel entries are shard-scoped too, so a
       cancel that omits exchange_index could silently no-op and leave it resting

SAFETY. One contract. post_only, so it can never cross and take. The limit is placed
--depth cents BELOW the best bid, so it rests behind the entire queue and will not fill
unless the market collapses through it within the few seconds it is alive. It is
cancelled immediately, and cancellation is verified rather than assumed. Worst case if
it somehow filled is one contract, under a dollar.

PATHS. The V2 endpoints are /portfolio/events/orders{,/batched}. The legacy
/portfolio/orders is deprecated no earlier than 2026-05-06. exchange_index and
subaccount go inside EACH order entry, not on the batch body.
"""
import argparse
import json
import os
import sys
import time
import uuid

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt

BASE = "https://api.elections.kalshi.com"
TENNIS_SHARD = 3
V2_ORDERS = "/trade-api/v2/portfolio/events/orders"
V2_BATCHED = "/trade-api/v2/portfolio/events/orders/batched"


def auth():
    envf = "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env"
    sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
    try:
        from dotenv import load_dotenv
        load_dotenv(envf)
    except ImportError:
        for ln in open(envf):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    from kalshi_auth import KalshiAuth
    pk = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not pk.startswith("/"):
        pk = ("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/"
              "esports_arbitrage/" + pk)
    return KalshiAuth(os.getenv("KALSHI_API_KEY_ID"), pk)


def req(a, method, path, body=None, params=None):
    h = a.get_headers(method, path)
    if body is not None:
        h["Content-Type"] = "application/json"
    r = requests.request(method, BASE + path, headers=h, json=body,
                         params=params, timeout=20)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", help="market to rest on; default = pick a live tennis one")
    ap.add_argument("--depth", type=int, default=15,
                    help="cents BELOW the best bid to rest, so it cannot fill")
    ap.add_argument("--subaccount", type=int, default=0)
    ap.add_argument("--send", action="store_true", help="actually place the order")
    a = ap.parse_args()

    A = auth()

    tk = a.ticker
    if not tk:
        feed = kt.Feed(rps=3.0, verbose=False)
        for e in feed.matches():
            mkts = feed.markets(e["event_ticker"])
            for m in mkts:
                b, ask = kt.top_of_book(m)
                if b and ask and 0.20 < b < 0.80:
                    tk = m.get("ticker")
                    break
            if tk:
                break
    if not tk:
        print("no suitable tennis market found")
        return 1

    c, body = req(A, "GET", f"/trade-api/v2/markets/{tk}")
    m = (body or {}).get("market") or {}
    shard = m.get("exchange_index")
    bid = m.get("yes_bid_dollars")
    ask = m.get("yes_ask_dollars")
    print(f"  market  {tk}")
    print(f"  shard   exchange_index={shard!r}   book {bid}/{ask}   status={m.get('status')!r}")
    if shard != TENNIS_SHARD:
        print(f"  REFUSING: expected shard {TENNIS_SHARD}")
        return 1

    try:
        bid_c = int(round(float(bid) * 100))
    except (TypeError, ValueError):
        print("  no usable bid")
        return 1
    limit_c = max(1, bid_c - a.depth)

    cid = str(uuid.uuid4())
    entry = {
        "ticker": tk,
        "client_order_id": cid,
        "side": "bid",                 # V2 single-book side
        "count": "1.00",
        "price": f"{limit_c/100:.4f}",
        "time_in_force": "good_till_canceled",
        # REQUIRED on V2 - omitting it returns 400 missing_parameters. taker_at_cross
        # cancels the incoming taker rather than our resting order, which is what a
        # quoter wants; the docs pair it with good_till_canceled and reserve `maker`
        # for IOC.
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": True,             # can never cross into a take
        "exchange_index": TENNIS_SHARD,
        "subaccount": a.subaccount,
    }
    print(f"\n  POST {V2_BATCHED}")
    print(f"  {json.dumps({'orders': [entry]}, indent=2)}")
    print(f"\n  resting {a.depth}c under the {bid_c}c bid, at {limit_c}c — 1 contract, "
          f"post_only, cancelled immediately")
    if not a.send:
        print("\n  DRY RUN — nothing sent. Re-run with --send.")
        return 0

    c, resp = req(A, "POST", V2_BATCHED, {"orders": [entry]})
    print(f"\n  HTTP {c}")
    print(f"  {json.dumps(resp, indent=1)[:700]}")
    oid = None
    for o in (resp or {}).get("orders", []) if isinstance(resp, dict) else []:
        oid = o.get("order_id") or (o.get("order") or {}).get("order_id")
        if o.get("error"):
            print(f"  per-order error: {o['error']}")
    if not oid:
        print("\n  no order_id returned — nothing to cancel. Check the response above.")
        return 1

    time.sleep(2)
    c, got = req(A, "GET", "/trade-api/v2/portfolio/orders",
                 params={"status": "resting", "limit": 100, "exchange_index": TENNIS_SHARD})
    mine = [o for o in (got or {}).get("orders", []) if o.get("order_id") == oid]
    print(f"\n  read back on exchange_index={TENNIS_SHARD}: "
          f"{'FOUND' if mine else 'NOT FOUND'}")
    if mine:
        o = mine[0]
        print(f"    order_id={o.get('order_id')}  exchange_index={o.get('exchange_index')!r}  "
              f"ticker={o.get('ticker')}  remaining={o.get('remaining_count_fp')}")

    # cancel entries are shard-scoped too
    c, cx = req(A, "DELETE", V2_BATCHED,
                {"orders": [{"order_id": oid, "exchange_index": TENNIS_SHARD,
                             "subaccount": a.subaccount}]})
    print(f"\n  DELETE HTTP {c}")
    print(f"  {json.dumps(cx, indent=1)[:400]}")

    time.sleep(2)
    c, got = req(A, "GET", "/trade-api/v2/portfolio/orders",
                 params={"status": "resting", "limit": 100, "exchange_index": TENNIS_SHARD})
    still = [o for o in (got or {}).get("orders", []) if o.get("order_id") == oid]
    print(f"\n  after cancel: {'STILL RESTING — INVESTIGATE' if still else 'gone (cancel confirmed)'}")
    return 0 if not still else 1


if __name__ == "__main__":
    sys.exit(main())
