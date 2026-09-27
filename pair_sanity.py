"""Pair-price sanity watchdog (read-only; 2026-09-03, operator request).

Invariant: our resting quotes on a match pair must express the SAME price for
the same outcome: a resting BUY on ticker A at p and a resting SELL on ticker B
at q are the same direction, so p should equal 100-q (+/- 1c tick rounding).
A violation means per-leg adjustments (skew/retreat/rounding) diverged — the
"poor adjustments" failure mode. Checks every CYCLE seconds via ONE shard-3
resting-orders GET; logs [PAIR MISMATCH] loudly, one OK summary line per cycle.
"""
import sys, time
sys.path.insert(0, '/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage')
from dotenv import load_dotenv; import os
load_dotenv('/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env')
from kalshi_auth import KalshiAuth
from kalshi_client import KalshiClient

pk = os.getenv('KALSHI_PRIVATE_KEY_PATH', '').strip()
if not pk.startswith('/'):
    pk = f'/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/esports_arbitrage/{pk}'
client = KalshiClient(KalshiAuth(os.getenv('KALSHI_API_KEY_ID', '').strip(), pk))
CYCLE = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
TOL_C = 1.0

while True:
    try:
        o = (client._get('/trade-api/v2/portfolio/orders',
                         params={'status': 'resting', 'limit': 1000,
                                 'exchange_index': 3}) or {}).get('orders') or []
        by_ev = {}
        for x in o:
            t = str(x.get('ticker', ''))
            ev = t.rsplit('-', 1)[0]
            px = float(x.get('yes_price_dollars') or 0) * 100
            by_ev.setdefault(ev, []).append((t.rsplit('-', 1)[1], x.get('action'), px))
        ts = time.strftime('%H:%M:%S', time.gmtime())
        pairs = bad = 0
        for ev, rows in by_ev.items():
            legs = sorted({r[0] for r in rows})
            if len(legs) != 2:
                continue
            a, b = legs
            for act_a, act_b in (('buy', 'sell'), ('sell', 'buy')):
                pa = [px for lg, ac, px in rows if lg == a and ac == act_a]
                pb = [px for lg, ac, px in rows if lg == b and ac == act_b]
                if not pa or not pb:
                    continue
                pairs += 1
                diff = abs(pa[0] - (100 - pb[0]))
                if diff > TOL_C:
                    # DEBOUNCE (added after first live catch, 15:55Z): the pair's
                    # two legs amend as separate wire calls ~0.1-0.3s apart every
                    # reprice, so a snapshot can land between them and see a real
                    # but sub-second asymmetry. Re-check the pair once after 3s;
                    # alert only if the mismatch PERSISTS (= genuine bad adjust).
                    time.sleep(3.0)
                    o2 = (client._get('/trade-api/v2/portfolio/orders',
                                      params={'status': 'resting', 'limit': 1000,
                                              'exchange_index': 3}) or {}).get('orders') or []
                    pa2 = [float(x.get('yes_price_dollars') or 0) * 100 for x in o2
                           if str(x.get('ticker','')) == f"{ev}-{a}" and x.get('action') == act_a]
                    pb2 = [float(x.get('yes_price_dollars') or 0) * 100 for x in o2
                           if str(x.get('ticker','')) == f"{ev}-{b}" and x.get('action') == act_b]
                    if pa2 and pb2 and abs(pa2[0] - (100 - pb2[0])) > TOL_C:
                        bad += 1
                        print(f"{ts}Z [PAIR MISMATCH] {ev} {a}-{act_a}@{pa2[0]:.0f} vs "
                              f"{b}-{act_b}@{pb2[0]:.0f} -> implied {100-pb2[0]:.0f} "
                              f"(diff {abs(pa2[0]-(100-pb2[0])):.1f}c, PERSISTED 3s)", flush=True)
        print(f"{ts}Z pair-sanity: {len(o)} orders, {pairs} direction-pairs, "
              f"{bad} mismatch(es)", flush=True)
    except Exception as e:
        print(f"pair-sanity error: {e}", flush=True)
    time.sleep(CYCLE)
