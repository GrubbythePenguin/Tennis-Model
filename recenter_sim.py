"""Recenter-after-every-point: a maker-fill backtest.

    python3 recenter_sim.py [--edge 0] [--halflife 0.5] [--size 10] [--event X]

THE DESIGN BEING TESTED. Take no view on the level. At the end of every point, re-anchor
to the market. Quote the one-point bracket around that anchor:

    bid = anchor + delta_lose      offer = anchor + delta_win

where the deltas come from the model at the CURRENT score. Whichever way the unseen point
went, the true price is inside the quote, so a counterparty who already knows the outcome
gets fair value and cannot pick us off. Revenue is the spread from liquidity-motivated
flow, not from a view.

WHY THE SPLIT IS PINNED. The delta depends on who is serving, so it depends on p-q. But
p-q is not identified by match prices (2.91c per 0.01 on p+q vs 0.22c on p-q), and the
live ewma2 fit let it collapse to 0.039 on 26AUG25CHWPAR - implying a WTA player holds
52.5% while she was 6-for-6 on serve. That degenerate split inflated the bracket to 7.57c.
So the split is pinned at the tour prior via SumTracker and only p+q tracks the market.

FILL MODEL - deliberately conservative. A resting quote fills only when the market trades
THROUGH it (market ask <= our bid, or market bid >= our offer), never merely when it
touches. Real maker fills also happen at equality with queue priority, so this understates
fills. One fill per side per point, so a single quote cannot be filled repeatedly while
the book sits crossed with it across several polls.

WHAT WOULD MAKE THIS WRONG. The tape is a 3s poll of top-of-book, so a fill that happened
and reverted between polls is invisible, and queue position is not modelled at all. Treat
the fill COUNT as a lower bound and the P&L as indicative, not as a backtest you can size
from.
"""
import argparse, json, glob, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sum_tracker import SumTracker

FEE_PER_CONTRACT = 0.007          # Kalshi maker fee, ~0.7c/contract at mid prices


def strip(s):
    return {k: v for k, v in s.items()
            if k not in ("points_me", "points_opp") and not k.startswith("_")}


def clean(s):
    return {k: v for k, v in s.items() if not k.startswith("_")}


def run_match(path, edge=0.0, halflife=0.5, size=10, max_pos=200):
    ev = os.path.basename(path)[:-6]
    lp = path[:-6] + ".log.json"
    try:
        log = json.load(open(lp))
    except Exception:
        return None
    prior = log.get("split_prior", 0.20)
    bo = log.get("best_of", 3)
    trk = SumTracker(best_of=bo, split=prior, halflife=halflife, first_server="me")

    rows = []
    for line in open(path, errors="replace"):
        if '"book"' not in line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "live":
            continue
        s, bk = r.get("state") or {}, r.get("book") or {}
        if bk.get("bid_me") is None or bk.get("ask_me") is None:
            continue
        if s.get("sets_me") is None or s.get("games_me") is None:
            continue
        rows.append(r)
    if len(rows) < 100:
        return None

    gkey = lambda s: (s.get("sets_me"), s.get("sets_opp"), s.get("games_me"), s.get("games_opp"))
    pkey = lambda s: gkey(s) + (s.get("points_me"), s.get("points_opp"))

    lastg = lastp = None
    quote = None                       # (bid, offer, filled_bid, filled_offer)
    pos = 0.0                          # signed contracts of "me"
    cash = 0.0
    fills = []
    brackets = []
    for r in rows:
        s, bk = r["state"], r["book"]
        vf = r.get("vig_free")
        bid_m, ask_m = bk["bid_me"] * 100.0, bk["ask_me"] * 100.0

        if gkey(s) != lastg and lastg is not None and vf:
            trk.observe(vf, weight=1.0, **strip(s))
        lastg = gkey(s)

        # ---- recenter on every POINT change (and on the first pricable row)
        if pkey(s) != lastp and trk.s is not None:
            lastp = pkey(s)
            anchor = (bid_m + ask_m) / 2.0
            try:
                base = clean(s)
                w = dict(base); w["points_me"] = base.get("points_me", 0) + 1
                l = dict(base); l["points_opp"] = base.get("points_opp", 0) + 1
                pw = trk.price(**w)[0] * 100.0
                pl = trk.price(**l)[0] * 100.0
                now = trk.price(**base)[0] * 100.0
            except Exception:
                quote = None; continue
            dlo, dhi = min(pw, pl) - now, max(pw, pl) - now      # deltas, not levels
            brackets.append(dhi - dlo)
            quote = [anchor + dlo - edge, anchor + dhi + edge, False, False]
        elif pkey(s) != lastp:
            lastp = pkey(s)

        if quote is None:
            continue
        b, o, fb, fo = quote
        # ---- conservative maker fills: the market must trade THROUGH the quote
        if not fb and ask_m <= b and pos > -max_pos:
            px = b; pos += size; cash -= size * px; quote[2] = True
            fills.append((r.get("ts"), "BUY", px, ask_m, bid_m, dict(s)))
        if not fo and bid_m >= o and pos < max_pos:
            px = o; pos -= size; cash += size * px; quote[3] = True
            fills.append((r.get("ts"), "SELL", px, ask_m, bid_m, dict(s)))

    fin = rows[-1]["state"]
    won = fin.get("sets_me", 0) > fin.get("sets_opp", 0)
    settle = 100.0 if won else 0.0
    pnl = (cash + pos * settle) / 100.0 - len(fills) * size * FEE_PER_CONTRACT
    nb = sum(1 for f in fills if f[1] == "BUY")
    ns = len(fills) - nb
    return dict(ev=ev, fills=len(fills), buys=nb, sells=ns, pos=pos, pnl=pnl,
                settle=settle, brk=(st.median(brackets) if brackets else float("nan")),
                rows=len(rows), fill_list=fills)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.0)
    ap.add_argument("--halflife", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--event", nargs="*")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    paths = ([f"tapes/{e}.jsonl" for e in a.event] if a.event
             else sorted(glob.glob("tapes/*.jsonl")))
    out = []
    for p in paths:
        try:
            r = run_match(p, edge=a.edge, halflife=a.halflife, size=a.size)
        except Exception as e:
            print(f"  {os.path.basename(p)}: {type(e).__name__} {e}"); continue
        if r: out.append(r)
    if not out:
        print("no usable tapes"); return 1
    print(f"edge={a.edge}c  halflife={a.halflife}  size={a.size}\n")
    print(f"{'match':<34}{'rows':>7}{'med brkt':>10}{'fills':>7}{'buy':>6}{'sell':>6}"
          f"{'end pos':>9}{'P&L':>10}")
    for r in sorted(out, key=lambda x: x["pnl"]):
        print(f"{r['ev'].rsplit('-',1)[-1]:<34}{r['rows']:>7}{r['brk']:>10.2f}"
              f"{r['fills']:>7}{r['buys']:>6}{r['sells']:>6}{r['pos']:>9.0f}{r['pnl']:>+10.2f}")
    tf = sum(r["fills"] for r in out); tp = sum(r["pnl"] for r in out)
    tb = sum(r["buys"] for r in out); ts_ = sum(r["sells"] for r in out)
    print(f"\n{len(out)} matches | {tf} fills ({tb} buy / {ts_} sell) | "
          f"P&L {tp:+.2f} | per fill {tp/tf*100/a.size if tf else 0:+.2f}c")
    print(f"one-way share: {max(tb,ts_)/(tb+ts_)*100 if tf else 0:.0f}%  "
          f"| matches profitable: {sum(1 for r in out if r['pnl']>0)}/{len(out)}")
    if a.detail:
        for r in out:
            for ts, sd, px, ask, bid, s in r["fill_list"][:12]:
                print(f"   {r['ev'].rsplit('-',1)[-1]:<14}{sd:<5}{px:>6.1f}  "
                      f"mkt {bid:.0f}/{ask:.0f}  {s.get('games_me')}-{s.get('games_opp')} "
                      f"{s.get('points_me')}-{s.get('points_opp')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
