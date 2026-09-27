#!/usr/bin/env python3
"""Are the arm loop's discovery mids -- the source of every pregame anchor -- accurate?

seed_<ev>.json carries the vig-free discovery mid written before attach, and
_pregame_vf_me falls back to it whenever no clean all-zeros tape tick exists (which is
always, since the quoter only configures LIVE matches so no pregame wsbook is written).
This compares discovered.json mids against a fresh REST orderbook for NOT-YET-STARTED
ITF matches, i.e. the anchor at the moment it matters. 2 GETs per event."""
import json, os, sys, datetime as dt
sys.path.insert(0, "quoter")
sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_general_quoting")
sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage")
from dotenv import load_dotenv
load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
from kalshi_auth import KalshiAuth
from kalshi_client import KalshiClient

def mid_from_rest(c, tk):
    ob = (c.get_orderbook(tk) or {}).get("orderbook_fp") or {}
    y = [float(p)*100 for p, _ in (ob.get("yes_dollars") or [])]
    n = [float(p)*100 for p, _ in (ob.get("no_dollars") or [])]
    if not y or not n: return None, None, None
    bid = max(y); ask = 100 - max(n)
    return bid, ask, (bid + ask) / 2.0

k = os.getenv("KALSHI_API_KEY_ID","").strip()
pk = os.getenv("KALSHI_PRIVATE_KEY_PATH","").strip()
if not pk.startswith("/"):
    pk = f"/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage/{pk}"
c = KalshiClient(KalshiAuth(k, pk))
d = json.load(open("tapes/discovered.json"))
now = dt.datetime.now(dt.timezone.utc)
cands = []
for ev, v in d.items():
    if ev.split("-",1)[0] not in ("KXITFMATCH","KXITFWMATCH"): continue
    tks = v.get("tickers") or []
    mids = v.get("mids") or {}
    if len(tks) != 2 or len(mids) != 2: continue
    try: h = (dt.datetime.fromisoformat(v["start"].replace("Z","+00:00")) - now).total_seconds()/3600
    except Exception: continue
    if 0.1 <= h <= 10: cands.append((h, ev, sorted(tks), mids))
cands.sort()
cands = cands[:int(sys.argv[1]) if len(sys.argv)>1 else 5]
print(f"{'event':38} {'start':>6} {'disc mid':>9} {'REST mid':>9} {'diff':>7} {'REST bid/ask':>13}")
diffs=[]
for h, ev, tks, mids in cands:
    a = tks[0]
    dm = mids.get(a)
    if dm is None: continue
    # vig-free the discovery pair the same way kalshi_tennis.vig_free does
    s = sum(mids.values())
    dvf = 100.0*dm/s if s > 0 else None
    try: bid, ask, rm = mid_from_rest(c, a)
    except Exception as e:
        print(f"{ev:38} REST failed {str(e)[:30]}"); continue
    if rm is None:
        print(f"{ev:38} {h:+5.1f}h {dvf:8.1f}c   no REST book yet"); continue
    diffs.append(rm - dvf)
    print(f"{ev:38} {h:+5.1f}h {dvf:8.1f}c {rm:8.1f}c {rm-dvf:+6.1f}c {bid:5.0f}/{ask:<6.0f}")
if diffs:
    import statistics as st
    print(f"\n  REST mid minus discovery mid: mean {st.mean(diffs):+.2f}c  "
          f"median {st.median(diffs):+.2f}c  n={len(diffs)}")
    print(f"  -> {'ANCHOR SOURCE GOOD' if abs(st.mean(diffs))<=2.0 else 'anchor source drifting'}")
