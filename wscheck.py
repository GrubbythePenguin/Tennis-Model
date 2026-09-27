#!/usr/bin/env python3
"""Is the ITF displayed book now the LIVE orderbook?

Compares, for each live ITF event, the quoter's displayed top-of-book (the newest line
of tapes/wsbook_<ev>.jsonl, which the recenter model writes from dt_market_state's
__top_bids__/__top_offers__) against a fresh REST orderbook. Before the run.py obs_to_fetch
fix, ITF fell through to the full_markets metadata snapshot and ran ~11.5c from executable.

    python3 wscheck.py [max_events]     # 2 GETs per event
"""
import json, os, sys, time, glob

def newest_wsbook(ev):
    p = f"tapes/wsbook_{ev}.jsonl"
    try:
        sz = os.path.getsize(p)
        with open(p, "rb") as f:
            f.seek(max(0, sz - 4000))
            lines = f.read().decode("utf-8", "replace").strip().splitlines()
        for l in reversed(lines):
            try: return json.loads(l)
            except Exception: continue
    except OSError:
        return None

def rest_top(client, tk):
    ob = (client.get_orderbook(tk) or {}).get("orderbook_fp") or {}
    y = [float(p) * 100 for p, _ in (ob.get("yes_dollars") or [])]
    n = [float(p) * 100 for p, _ in (ob.get("no_dollars") or [])]
    bid = max(y) if y else None
    ask = (100 - max(n)) if n else None
    return bid, ask, len(y), len(n)

def main(limit):
    from dotenv import load_dotenv
    load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
    sys.path.insert(0, "quoter")
    sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_general_quoting")
    sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
    from kalshi_auth import KalshiAuth
    from kalshi_client import KalshiClient
    k = os.getenv("KALSHI_API_KEY_ID", "").strip()
    pk = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not pk.startswith("/"):
        pk = f"/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage/{pk}"
    c = KalshiClient(KalshiAuth(k, pk))

    cfg = "quoter/template_quoter_config.csv"
    evs = []
    for line in open(cfg).read().splitlines()[1:]:
        t = line.split(",")[1] if "," in line else ""
        if t.split("-", 1)[0] in ("KXITFMATCH", "KXITFWMATCH"):
            e = t.rsplit("-", 1)[0]
            if e not in evs: evs.append(e)
    evs = evs[:limit]
    if not evs:
        print("no ITF events configured right now"); return
    print(f"{'event':40} {'age':>5} {'shown bid/ask':>14} {'REST bid/ask':>14} "
          f"{'ask gap':>8} {'lvls':>7}")
    gaps = []
    for ev in evs:
        w = newest_wsbook(ev)
        if not w:
            print(f"{ev:40} {'-':>5}  no wsbook line yet"); continue
        me = w.get("me") or [None, None]
        meta = json.load(open(f"tapes/{ev}.log.json")).get("meta") or {}
        tk = meta.get("me_ticker")
        if not tk:
            print(f"{ev:40} no me_ticker"); continue
        try:
            rb, ra, ny, nn = rest_top(c, tk)
        except Exception as e:
            print(f"{ev:40} REST failed: {str(e)[:40]}"); continue
        age = time.time() - w.get("ts", 0)
        gap = (ra - me[1]) if (ra is not None and me[1] is not None) else None
        if gap is not None: gaps.append(gap)
        # one side can be absent (no YES bids but plenty of NO bids); format each
        # independently or a missing bid takes the whole line down
        rs = f"{rb:.0f}" if rb is not None else "-"
        as_ = f"{ra:.0f}" if ra is not None else "-"
        print(f"{ev:40} {age:4.0f}s {str(me[0])+'/'+str(me[1]):>14} "
              f"{rs+'/'+as_:>14} "
              f"{(f'{gap:+.1f}c' if gap is not None else '-'):>8} {ny}y/{nn}n")
    if gaps:
        import statistics as st
        print(f"\n  ask gap (REST - shown): mean {st.mean(gaps):+.2f}c  "
              f"median {st.median(gaps):+.2f}c  n={len(gaps)}")
        m = abs(st.mean(gaps))
        print(f"  -> {'LIVE BOOK: displayed matches executable' if m <= 1.5 else ('STILL STALE: ' + f'{m:.1f}c off executable')}")
        print(f"  (pre-fix baseline was +11.5c over 294 set-1 taker observations)")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
