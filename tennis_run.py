"""Tennis quoter. DRY RUN BY DEFAULT - computes and logs quotes, places nothing.

    python3 tennis_run.py                          # dry run over the configured matches
    python3 tennis_run.py --once                   # one pass, then exit
    python3 tennis_run.py --churn-report           # what the write budget would cost

THERE IS NO ORDER-PLACEMENT CODE IN THIS FILE. Not disabled by a flag - absent. Adding
it is a deliberate, separate change, because as of 26AUG26 shard 3 holds $0.00 and no
tennis order can succeed anyway. What this does is compute exactly what it WOULD post,
log it, and measure the cost of posting it.

WHERE THE PRICES COME FROM. poll_tennis.py is the market-data and model process: it
polls, fits, and writes a tape carrying raw top of book plus each variant priced one
point ahead on BOTH branches. This process consumes that tape. Two reasons to split
them rather than have the quoter poll for itself:
  - no extra GETs against a bucket shared with the live esports system
  - poll_tennis keeps its property of touching no order endpoint at all

THE QUOTE. Our scoreboard runs a point behind, so at an observed 4-3 40-15 the market
already prices whatever the next point did. Each side is therefore valued at the branch
where THAT side lost the unseen point - buy A priced at 40-30, buy B priced at 5-3 0-0 -
and the quote goes `margin` cents below that. With w=model(win), l=model(lose):

    value(tracked)  = min(w, l)              quote = value - margin
    value(opponent) = 1 - max(w, l)          quote = value - margin

Neither side can be surprised by the point we did not see. The two values do not sum to
1; the gap is the bracket, and it is what the lag costs.

WHY THE CHURN NUMBER MATTERS. A quoter that re-prices on every point cancel/replaces
constantly, and a write that explicitly targets a nonzero shard bills THAT shard's write
budget. Nobody has run a shard-3 quoter on this account, so the write rate is an
unmeasured risk. This reports it before a single order exists.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(HERE, "tapes")
TENNIS_SHARD = 3
KILL = os.path.join(HERE, "disable_tennis_poller.flag")


def load_config(path):
    p = path if path.startswith("/") else os.path.join(HERE, path)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("enabled", "1")).strip() not in ("1", "true", "True"):
                continue
            if int(r["exchange_index"]) != TENNIS_SHARD:
                print(f"  REFUSED {r['ticker']}: exchange_index="
                      f"{r['exchange_index']} not {TENNIS_SHARD}", file=sys.stderr)
                continue
            for k in ("size", "max_open", "max_match", "max_total", "subaccount"):
                r[k] = int(float(r[k]))
            for k in ("margin_c", "min_price", "max_price"):
                r[k] = float(r[k])
            out.append(r)
    return out


def last_tape_row(event):
    """Most recent in-play row for an event, or None."""
    p = os.path.join(TAPES, event + ".jsonl")
    if not os.path.exists(p):
        return None
    last = None
    with open(p, "rb") as f:
        try:
            f.seek(max(0, os.path.getsize(p) - 200_000))
        except Exception:
            pass
        tail = f.read().decode("utf-8", "replace").splitlines()
    for line in reversed(tail):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") == "live" and (r.get("ahead") or {}).get("win"):
            last = r
            break
    return last


def quotes_for(cfg, row):
    """The two quotes this match supports right now, or [] if it is unquotable."""
    a = row.get("ahead") or {}
    bk = row.get("book") or {}
    v = cfg["variant"]
    if v not in (a.get("win") or {}) or v not in (a.get("lose") or {}):
        return []
    w, l = a["win"][v], a["lose"][v]
    st = row.get("state") or {}
    tracked_tick = None
    out = []
    for side, value, bid, ask in (
            ("me", min(w, l), bk.get("bid_me"), bk.get("ask_me")),
            ("opp", 1.0 - max(w, l), bk.get("bid_opp"), bk.get("ask_opp"))):
        if bid is None or ask is None:
            continue
        q = value - cfg["margin_c"] / 100.0
        if not (cfg["min_price"] <= q <= cfg["max_price"]):
            continue
        # never quote through the offer: that is taking, not quoting
        q = min(q, ask - 0.01)
        if q <= 0:
            continue
        out.append(dict(side=side, value=round(value, 4), quote=round(q, 2),
                        bid=bid, ask=ask, size=cfg["size"],
                        bracket_c=round(abs(w - l) * 100, 2),
                        state="%s-%s %s-%s %s-%s" % (
                            st.get("sets_me"), st.get("sets_opp"),
                            st.get("games_me"), st.get("games_opp"),
                            st.get("points_me"), st.get("points_opp"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tennis_quoter_config.csv")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--churn-report", action="store_true",
                    help="summarise how often the quote would change, i.e. the write "
                         "rate a live quoter would draw on shard 3's budget")
    ap.add_argument("--max-cycles", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if not cfg:
        print(f"no enabled shard-{TENNIS_SHARD} rows in {a.config} — "
              f"run tennis_parsed_markets.py then tennis_populate_configs.py")
        return 1
    by_event = defaultdict(list)
    for c in cfg:
        by_event[c["event"]].append(c)

    print("=" * 78)
    print("  TENNIS QUOTER — DRY RUN. No order-placement code exists in this process.")
    print(f"  {len(cfg)} configured sides across {len(by_event)} matches, shard {TENNIS_SHARD}")
    c0 = cfg[0]
    print(f"  size {c0['size']} | margin {c0['margin_c']:.1f}c | subaccount {c0['subaccount']}"
          f" | variant {c0['variant']}")
    print(f"  caps {c0['max_open']}/side {c0['max_match']}/match {c0['max_total']} total")
    print("=" * 78)

    prev = {}                 # (event, side) -> last quote
    changes = defaultdict(int)
    looks = 0
    t0 = time.time()
    try:
        while True:
            if os.path.exists(KILL):
                print("kill flag present — stopping.")
                break
            looks += 1
            live = 0
            for ev, rows_cfg in by_event.items():
                row = last_tape_row(ev)
                if row is None:
                    continue
                live += 1
                qs = quotes_for(rows_cfg[0], row)
                for q in qs:
                    key = (ev, q["side"])
                    if prev.get(key) != q["quote"]:
                        changes[key] += 1
                        if not a.churn_report:
                            print(f"  [{time.strftime('%H:%M:%S')}] {ev[-12:]:<12} "
                                  f"{q['side']:<4} state {q['state']:<14} "
                                  f"value {q['value']:.3f} bracket {q['bracket_c']:>5.2f}c "
                                  f"-> QUOTE {q['quote']:.2f} x{q['size']}  "
                                  f"(book {q['bid']:.2f}/{q['ask']:.2f})")
                        prev[key] = q["quote"]
            if a.once or (a.max_cycles and looks >= a.max_cycles):
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\ninterrupted.")

    el = max(time.time() - t0, 1e-9)
    tot = sum(changes.values())
    print("\n" + "=" * 78)
    print(f"  {looks} passes over {len(by_event)} matches in {el:.0f}s")
    print(f"  quote changes: {tot}  -> {tot/el:.2f} writes/s if each were a "
          f"cancel+replace pair, {2*tot/el:.2f} raw order writes/s")
    if changes:
        top = sorted(changes.items(), key=lambda x: -x[1])[:5]
        print("  busiest sides:")
        for (ev, side), n in top:
            print(f"    {ev[-14:]:<16} {side:<4} {n} changes ({n/el*60:.1f}/min)")
    print("  NOTE: shard 3 held $0.00 as of 26AUG26. No tennis order can fill until")
    print("        collateral is preallocated there.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
