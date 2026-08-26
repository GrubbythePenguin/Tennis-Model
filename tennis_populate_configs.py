"""Per-match quoting config for the tennis 10-lot test. Tennis only, shard 3 only.

    python3 tennis_populate_configs.py --from tennis_parsed_markets.csv
    python3 tennis_populate_configs.py --size 10 --margin 2.0 --out tennis_quoter_config.csv

WHAT A CONFIG ROW IS. One tradeable side of one match, with the parameters the quoter
needs and the limits it must not exceed. Both sides of a match get a row, because the
strategy quotes BOTH: at an observed 4-3 40-15 we are willing to buy A priced at 40-30
and buy B priced at 5-3 0-0, each at the branch where THAT side lost the unseen point.

THE MARGIN IS THE EDGE. The quote goes `margin` cents BELOW the conservative branch
price, so a fill only happens at a price better than our already-pessimistic value.
Measured 26AUG26 across 49 point-level tapes, the 30s markout scales with how far the
market is beyond that conservative price:

    edge 0-1c -> +0.376c    1-2c -> +0.490c    2-4c -> +0.634c    4c+ -> +1.011c

so a wider margin is paid for, at least out to 4c. It also trades less: the 4c+ bucket
fired on 1,808 opportunities against 4,200 in 1-2c. Default 2.0c sits in the middle.

WHAT THE LIMITS ARE FOR. This is a MEASUREMENT run, not a P&L run. At 10 lots the P&L
is noise either way; the deliverable is fill-conditioned markout, i.e. how much of that
+0.6c survives being filled only when someone chooses to hit us. So the caps exist to
bound the cost of being wrong, not to size a position:

    size            contracts per quote                     10
    max_open        contracts open on one side of a match   10
    max_match       contracts across both sides of a match  20
    max_total       contracts across every match at once    60

REFUSES ANYTHING NOT ON SHARD 3. tennis_parsed_markets.py verifies exchange_index per
market; this re-checks rather than trusting the file, because a config that silently
carries a shard-0 ticker would auto-route and bill the esports write budget.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TENNIS_SHARD = 3

FIELDS = ["event", "ticker", "exchange_index", "player", "subaccount",
          "size", "margin_c", "max_open", "max_match", "max_total",
          "min_price", "max_price", "variant", "enabled"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="tennis_parsed_markets.csv")
    ap.add_argument("--out", default="tennis_quoter_config.csv")
    ap.add_argument("--size", type=int, default=10, help="contracts per quote")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="cents BELOW the conservative branch price to quote")
    # Subaccount 0 on shard 3. Subaccounts are PER-SHARD: sub 1 exists only on shard 0,
    # so a sub-1 transfer to shard 3 returns
    # 409 subaccount_1_does_not_exist_on_exchange_shard_3, and creating one there
    # returned 404 user_not_found until the account was provisioned on shard 3 via the
    # UI. That provisioning created sub 0 on shard 3, which is what holds the $10,000.
    ap.add_argument("--subaccount", type=int, default=0)
    ap.add_argument("--max-open", type=int, default=10)
    ap.add_argument("--max-match", type=int, default=20)
    ap.add_argument("--max-total", type=int, default=60)
    ap.add_argument("--min-price", type=float, default=0.10,
                    help="do not quote below this; deep longshots settle 0 and the "
                         "model is least reliable there")
    ap.add_argument("--max-price", type=float, default=0.90)
    ap.add_argument("--variant", default="ewma2",
                    help="which live fit to price from. ewma2 led the markout test; "
                         "static was weakest, consistent with a frozen fit being unable "
                         "to track a re-rating")
    a = ap.parse_args()

    src = a.src if a.src.startswith("/") else os.path.join(HERE, a.src)
    if not os.path.exists(src):
        print(f"no {src} - run tennis_parsed_markets.py --out {a.src} first")
        return 1

    rows, refused = [], 0
    with open(src, newline="") as f:
        for m in csv.DictReader(f):
            try:
                shard = int(m.get("exchange_index"))
            except (TypeError, ValueError):
                shard = None
            if shard != TENNIS_SHARD:
                refused += 1
                continue
            rows.append({
                "event": m["event"], "ticker": m["ticker"], "exchange_index": shard,
                "player": m.get("player", ""), "subaccount": a.subaccount,
                "size": a.size, "margin_c": a.margin,
                "max_open": a.max_open, "max_match": a.max_match,
                "max_total": a.max_total,
                "min_price": a.min_price, "max_price": a.max_price,
                "variant": a.variant, "enabled": 1,
            })

    out = a.out if a.out.startswith("/") else os.path.join(HERE, a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} config rows written to {out}"
          + (f"; {refused} refused for not being on shard {TENNIS_SHARD}" if refused else ""))
    print(f"  size {a.size} | margin {a.margin:.1f}c below the conservative branch price")
    print(f"  caps: {a.max_open}/side  {a.max_match}/match  {a.max_total} total")
    print(f"  subaccount {a.subaccount} | variant {a.variant} | "
          f"price band {a.min_price:.2f}-{a.max_price:.2f}")
    ev = {r["event"] for r in rows}
    print(f"  {len(ev)} matches, both sides each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
