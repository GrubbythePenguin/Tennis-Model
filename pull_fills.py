"""Pull every tennis fill from Kalshi into tapes/kalshi_fills.csv (trades.csv schema).

    .venv/bin/python3 pull_fills.py [--tag 26AUG28] [--out tapes/kalshi_fills.csv]
    python3 fill_markouts.py --since 26AUG28 --fills tapes/kalshi_fills.csv

WHY. quoter/trades.csv is flushed only at shutdown, so mid-session it is hours behind;
the exchange record is the truth. One GET per 1000 fills. Direction from `book_side`
(bid = bought YES, ask = sold YES) - the unambiguous field; `action`/`side` are not.
Fees come from `fee_cost`, so the ATP/WTA maker fee and the Challenger/ITF zero are
both real numbers here.
"""
import argparse, csv, datetime as dt, os, sys
sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_general_quoting")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quoter"))
from dotenv import load_dotenv
load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
from kalshi_auth import KalshiAuth
from kalshi_client import KalshiClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="only tickers containing this, e.g. 26AUG28")
    ap.add_argument("--out", default="tapes/kalshi_fills.csv")
    a = ap.parse_args()
    k = os.getenv("KALSHI_API_KEY_ID", "").strip(); p = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not p.startswith("/"):
        p = f"/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage/{p}"
    c = KalshiClient(KalshiAuth(k, p))
    fills, cursor, n_get = [], None, 0
    while True:
        params = {"limit": 1000, "exchange_index": 3}
        if cursor:
            params["cursor"] = cursor
        r = c._get("/trade-api/v2/portfolio/fills", params=params) or {}
        n_get += 1
        for f in r.get("fills") or []:
            t = f.get("ticker", "")
            if t.startswith(("KXATP", "KXWTA", "KXITF")) and (not a.tag or a.tag in t):
                fills.append(f)
        cursor = r.get("cursor")
        if not cursor:
            break
    with open(a.out, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["trade_id", "created_ts", "ticker", "action", "count", "price", "side", "is_taker", "fee_cost", "edge_cents"])
        for f in fills:
            ts = int(dt.datetime.fromisoformat(f["created_time"].replace("Z", "+00:00")).timestamp())
            fee = float(f.get("fee_cost") or 0)
            w.writerow([f["trade_id"], ts, f["ticker"], "buy" if f["book_side"] == "bid" else "sell",
                        float(f["count_fp"]), float(f["yes_price_dollars"]), "yes", bool(f.get("is_taker")), fee, 0.0])
    print(f"{len(fills)} fills -> {a.out}  ({n_get} GET{'s' if n_get != 1 else ''})")
    # settlements: Kalshi's own result per market, for marking finished matches (1 GET / 1000)
    sett, cursor, n2 = [], None, 0
    while True:
        params = {"limit": 1000, "exchange_index": 3}
        if cursor:
            params["cursor"] = cursor
        r = c._get("/trade-api/v2/portfolio/settlements", params=params) or {}
        n2 += 1
        for x in r.get("settlements") or []:
            if str(x.get("ticker", "")).startswith(("KXATP", "KXWTA", "KXITF")):
                sett.append(x)
        cursor = r.get("cursor")
        if not cursor or n2 > 20:
            break
    sp = os.path.join(os.path.dirname(a.out), "kalshi_settlements.csv")
    with open(sp, "w", newline="") as out:
        w = csv.writer(out); w.writerow(["ticker", "market_result", "settled_time", "revenue", "yes_count", "no_count"])
        for x in sett:
            w.writerow([x.get("ticker"), x.get("market_result"), x.get("settled_time"), x.get("revenue_dollars", x.get("revenue")),
                        x.get("yes_count_fp", x.get("yes_count")), x.get("no_count_fp", x.get("no_count"))])
    print(f"{len(sett)} settlements -> {sp}  ({n2} GET{'s' if n2 != 1 else ''})")


if __name__ == "__main__":
    main()
