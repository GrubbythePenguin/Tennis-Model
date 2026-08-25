"""How long after a game ends does the market finish repricing?

    python3 market_lag.py tapes/<event>.jsonl [...]

WHY IT MATTERS. A boundary price is only meaningful if the market has absorbed the
game that just finished. Read it too early and you stamp the PREVIOUS game's price
onto the new score, which shows up as the market apparently moving the wrong way —
a player getting broken and going UP.

The lag is NOT a constant. Measured 26AUG24:
  Poljicak/Schoenhaus (ATP Challenger)  market LED the score feed by ~5s
  Koevermans/Monnet   (WTA qualifying)  market LAGGED it by ~19s
so it must be measured per match, not assumed. Liquidity is the obvious suspect.

METHOD. At each boundary, record the vig-free price at the moment the score changed,
then track it forward until it stops moving. The elapsed time to the final value is
the repricing lag; the size of that move is what an early read would have cost you.
"""
import argparse
import json
import statistics

import kalshi_tennis as kt
from rebuild_log import PLAYING, resolve_tracked


def lags(tape, horizon=60.0, tol=0.002):
    rows = [json.loads(l) for l in open(tape)]
    if not rows:
        return []
    me, opp, _, _ = resolve_tracked(rows)
    play = [r for r in rows
            if (r["details"].get("match_status") in PLAYING
                or (r["details"].get("status") == "live"
                    and r["details"].get("match_status") not in ("match_about_to_start", None)))
            and r.get("vig_free") is not None]

    out, last_key = [], None
    for i, r in enumerate(play):
        st = kt.model_state(r["details"], me, opp)
        if st is None:
            continue
        key = kt.boundary_key(st)
        if last_key is None or key == last_key:
            last_key = key
            continue
        p0, t0 = r["vig_free"], r["ts"]
        settle_px, settle_dt = p0, 0.0
        for r2 in play[i:]:
            if r2["ts"] - t0 > horizon:
                break
            s2 = kt.model_state(r2["details"], me, opp)
            if not s2 or kt.boundary_key(s2) != key:
                break              # next game started; stop
            if abs(r2["vig_free"] - settle_px) > tol:
                settle_px, settle_dt = r2["vig_free"], r2["ts"] - t0
        out.append(dict(key=key, first=p0, settled=settle_px, dt=settle_dt,
                        move=(settle_px - p0) * 100))
        last_key = key
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tapes", nargs="+")
    ap.add_argument("--horizon", type=float, default=60.0)
    a = ap.parse_args()

    for t in a.tapes:
        rs = lags(t, a.horizon)
        print(f"\n{t}   {len(rs)} boundaries")
        if not rs:
            continue
        print(f"  {'games':>8} {'first':>7} {'settled':>8} {'move':>7} {'lag s':>6}")
        for r in rs:
            k = r["key"]
            print(f"  {k[2]}-{k[3]:<6} {r['first']:7.3f} {r['settled']:8.3f} "
                  f"{r['move']:+7.1f}c {r['dt']:6.0f}")
        moved = [r for r in rs if abs(r["move"]) >= 0.5]
        dts = [r["dt"] for r in moved]
        print(f"  boundaries that repriced after capture: {len(moved)}/{len(rs)}")
        if dts:
            print(f"  median lag {statistics.median(dts):.0f}s   "
                  f"max {max(dts):.0f}s   "
                  f"median |cost of reading early| "
                  f"{statistics.median([abs(r['move']) for r in moved]):.1f}c")
        print(f"  RECOMMENDED --settle for this match: "
              f"{0 if not dts else int(min(60, max(dts)))}s")


if __name__ == "__main__":
    main()
