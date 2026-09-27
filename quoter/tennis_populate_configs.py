"""Map tennis market_parameters rows onto concrete shard-3 tickers, one line per player.

    python3 tennis_populate_configs.py                        # today's live/upcoming
    python3 tennis_populate_configs.py --event KXATPMATCH-26AUG26PIRGLI
    python3 tennis_populate_configs.py --within-hours 3 --out template_quoter_config.csv

WHAT THIS REPLACES. The esports populate_configs.py is 5,723 lines because it resolves
Polymarket mappings, team-name aliases, CS2 tiers and per-sport sizing. Tennis needs
none of that: a quoter config line IS the market_parameters row with `market_prefix`
swapped for a concrete `ticker` (plus capture_buffer_min). That is the entire job, so
this is the entire file.

TWO LINES PER MATCH. Each tennis event has exactly two markets, one per player, and
both are quoted - the model prices them as complements off the same one-point bracket
(tracked side gets min/max of the branches, the opponent gets 1 minus each). So a match
produces two config rows and the quoter runs both sides against one fit.

SHARD 3 IS ENFORCED, NOT ASSUMED. Every ticker is verified with a per-ticker GET before
it is written, and anything not on exchange_index 3 is refused. Two reasons this is not
paranoia: the /markets COLLECTION silently ignores a `ticker` filter (asking it for a
tennis ticker returned an unrelated shard-1 market), and an order to a mis-shard ticker
auto-routes, drawing on whichever shard the ticker implies and billing multiple rate
buckets.

REQUIRES poll_tennis.py TO BE RUNNING on every event quoted. The tennis_branch model
reads the tape poll_tennis writes - that is where the fit and the one-point branches
come from. A config row without a live tape produces no theo (by design: the model
treats a tape older than 30s as a hard stop and returns nothing, so the quoter pulls
rather than resting on a price nobody is maintaining).
"""
import argparse
import json
import csv
import datetime as dt
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import kalshi_tennis as kt

TENNIS_SHARD = 3
BASE = "https://api.elections.kalshi.com"


def _auth():
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


def verify_shard(auth, ticker):
    """exchange_index via the per-ticker PATH form (the collection ignores filters)."""
    import requests
    p = f"/trade-api/v2/markets/{ticker}"
    r = requests.get(BASE + p, headers=auth.get_headers("GET", p), timeout=15)
    if r.status_code != 200:
        return None
    return ((r.json() or {}).get("market") or {}).get("exchange_index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="tennis_market_parameters.csv")
    ap.add_argument("--out", default="template_quoter_config.csv")
    ap.add_argument("--event", nargs="*", help="specific events; default = the board")
    ap.add_argument("--within-hours", type=float, default=6.0,
                    help="only events starting within this many hours (or already live)")
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--rps", type=float, default=3.0)
    ap.add_argument("--capture-buffer-min", default="0")
    ap.add_argument("--from-discovery", metavar="PATH",
                    help="build rows from the arm loop's tapes/discovered.json (tickers + "
                         "exchange_index captured by its /events call) - NO GETs at all. "
                         "26AUG28: the per-event milestone/live_data/markets/shard GETs here "
                         "were 3+ calls per event per 5-minute cycle.")
    ap.add_argument("--ab-edges", default=None, metavar="A,B",
                    help="A/B test: assign each event under --ab-prefixes one of these min_edge "
                         "values (cents), balanced per series prefix, persisted in --ab-file so an "
                         "event never changes arm. 26AUG29: ITF at 2c vs 4c.")
    ap.add_argument("--ab-prefixes", default="KXITFMATCH,KXITFWMATCH")
    ap.add_argument("--sets", default=None, metavar="N[,N...]",
                    help="ALSO emit KX*SETWINNER rows for these set numbers (e.g. '1,2'). "
                         "Tickers are derived from the match tickers (KXATPMATCH-EV-ABC -> "
                         "KXATPSETWINNER-EV-1-ABC) and EVERY one is verified on-shard with a "
                         "per-ticker GET before a row is written; a set market that does not "
                         "exist is skipped, and a pair is all-or-nothing. Costs 2 GETs per set "
                         "number per match, so run it when the board is quiet. Params come "
                         "from a market_prefix=KX*SETWINNER row if one exists in --params, "
                         "else the match row with config_id+'_SET'. The tennis_recenter "
                         "model only ever QUOTES the set in progress, and only while "
                         "tapes/setwinner_quote.flag exists - without it these rows log "
                         "theos and books to tapes/setwinner_quotes.jsonl (measure first).")
    ap.add_argument("--ab-file", default=os.path.join(ROOT, "tapes", "ab_edges.json"))
    a = ap.parse_args()
    ab = None
    if a.ab_edges:
        arms = [float(x) for x in a.ab_edges.split(",")]
        try: ab = json.load(open(a.ab_file))
        except Exception: ab = {}
        ab_prefixes = set(a.ab_prefixes.split(","))

    pp = a.params if a.params.startswith("/") else os.path.join(HERE, a.params)
    params = list(csv.DictReader(open(pp)))
    if not params:
        print(f"no rows in {pp}")
        return 1
    # FIRST row per prefix is the primary (quoter) row; any FURTHER rows with the
    # same prefix are EXTRA bot rows (e.g. execution_type=set1_taker) emitted per
    # ticker alongside the primary — the manager supports multiple configs per
    # ticker (ARB+QUOTE pattern). A/B edge overrides apply to the primary only.
    by_prefix, extras_by_prefix = {}, {}
    for r in params:
        pfx = r["market_prefix"]
        if pfx in by_prefix:
            extras_by_prefix.setdefault(pfx, []).append(r)
        else:
            by_prefix[pfx] = r
    hdr = [c if c != "market_prefix" else "ticker" for c in params[0]]
    if "capture_buffer_min" not in hdr:
        hdr.append("capture_buffer_min")

    disc = json.load(open(a.from_discovery)) if a.from_discovery else None
    # --sets always needs auth: SETWINNER tickers are not in discovered.json (the arm
    # loop only sweeps the MATCH series), so each one is shard-verified by GET.
    auth = None if (disc is not None and not a.sets) else _auth()
    feed = kt.Feed(rps=a.rps, verbose=False)
    now = dt.datetime.now(dt.timezone.utc)

    events = a.event or ([] if disc is not None else [e["event_ticker"] for e in feed.matches()])
    rows, refused, skipped = [], [], 0
    set_nos = [int(x) for x in a.sets.split(",")] if a.sets else []

    def emit_set_rows(ev, prow, match_tickers, edge=None):
        """Derive, shard-verify and append SETWINNER rows for one match event."""
        prefix, base = ev.split("-", 1)
        set_prefix = prefix.replace("MATCH", "SETWINNER")
        sprow = by_prefix.get(set_prefix)
        for n in set_nos:
            stks, ok = [], True
            for tk in sorted(match_tickers):
                stk = f"{set_prefix}-{base}-{n}-{tk.rsplit('-', 1)[1]}"
                shard = verify_shard(auth, stk)
                if shard != TENNIS_SHARD:
                    refused.append((stk, shard))     # None = market does not exist
                    ok = False
                stks.append(stk)
            if not ok:
                continue                             # pair is all-or-nothing
            for stk in stks:
                r = {k: v for k, v in (sprow or prow).items() if k != "market_prefix"}
                if sprow is None:
                    r["config_id"] = f"{prow.get('config_id', '')}_SET"
                r["ticker"] = stk
                r["capture_buffer_min"] = a.capture_buffer_min
                if edge is not None:
                    r["min_edge"] = str(edge); r["min_absolute_edge"] = str(edge)
                rows.append(r)
    for ev in events:
        prefix = ev.split("-", 1)[0]
        prow = by_prefix.get(prefix)
        if not prow:
            continue
        if disc is not None:
            d = disc.get(ev)
            if not d or len(d.get("tickers") or []) != 2:
                skipped += 1
                continue
            bad = [(t, i) for t, i in zip(d["tickers"], d["exchange_index"]) if i != TENNIS_SHARD]
            if bad:
                refused.extend(bad)
                continue
            edge = None
            if ab is not None and prefix in ab_prefixes:
                if ev not in ab:
                    # balance within this series: give the new event the arm with fewer members
                    counts = {arm: sum(1 for e2, v in ab.items() if e2.split("-", 1)[0] == prefix and v["edge"] == arm) for arm in arms}
                    arm = min(arms, key=lambda x: (counts[x], arms.index(x)))
                    ab[ev] = {"edge": arm, "assigned": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
                    json.dump(ab, open(a.ab_file, "w"), indent=1)
                edge = ab[ev]["edge"]
            for tk in sorted(d["tickers"]):
                r = {k: v for k, v in prow.items() if k != "market_prefix"}
                r["ticker"] = tk
                r["capture_buffer_min"] = a.capture_buffer_min
                if edge is not None:
                    r["min_edge"] = str(edge); r["min_absolute_edge"] = str(edge)
                rows.append(r)
                for xrow in extras_by_prefix.get(prefix, ()):
                    x = {k: v for k, v in xrow.items() if k != "market_prefix"}
                    x["ticker"] = tk
                    x["capture_buffer_min"] = a.capture_buffer_min
                    rows.append(x)
            if set_nos:
                emit_set_rows(ev, prow, d["tickers"], edge)
            continue
        mil = feed.milestone(ev)
        if not mil:
            continue
        d = feed.live_data([mil.get("id")]).get(mil.get("id")) or {}
        is_live = kt.is_live(d) and (d.get("match_status") or "") not in ("", None, "match_about_to_start")
        if not a.event:
            if a.live_only and not is_live:
                skipped += 1
                continue
            if not is_live:
                try:
                    t = dt.datetime.fromisoformat((mil.get("start_date") or "").replace("Z", "+00:00"))
                    if not (-1.0 <= (t - now).total_seconds() / 3600 <= a.within_hours):
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue
        mkts = feed.markets(ev)
        tickers = sorted(m.get("ticker") for m in mkts if m.get("ticker"))
        if len(tickers) != 2:
            skipped += 1
            continue
        ok = True
        for tk in tickers:
            shard = verify_shard(auth, tk)
            if shard != TENNIS_SHARD:
                refused.append((tk, shard))
                ok = False
        if not ok:
            continue
        # one line per PLAYER — both sides of the match
        for tk in tickers:
            r = {k: v for k, v in prow.items() if k != "market_prefix"}
            r["ticker"] = tk
            r["capture_buffer_min"] = a.capture_buffer_min
            rows.append(r)
            for xrow in extras_by_prefix.get(prefix, ()):
                x = {k: v for k, v in xrow.items() if k != "market_prefix"}
                x["ticker"] = tk
                x["capture_buffer_min"] = a.capture_buffer_min
                rows.append(x)
        if set_nos:
            emit_set_rows(ev, prow, tickers)

    op = a.out if a.out.startswith("/") else os.path.join(HERE, a.out)
    with open(op, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_set = sum(1 for r in rows if "SETWINNER" in r["ticker"])
    ev_n = len({r["ticker"].rsplit("-", 1)[0] for r in rows}) - (n_set // 2)
    print(f"{len(rows)} config lines ({n_set} SETWINNER) across {ev_n} markets -> {op}")
    for tk, shard in refused[:8]:
        print(f"  REFUSED {tk}: exchange_index={shard!r}")
    if skipped:
        print(f"  {skipped} events skipped (not live / outside {a.within_hours}h / unpriced)")
    for r in rows[:8]:
        print(f"  {r['config_id']:<14} {r['ticker']:<44} vol={r['volumes']} "
              f"edge={r['min_edge']} model={r['model_name']}")
    if len(rows) > 8:
        print(f"  ... {len(rows)-8} more")
    print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")
    print("\nNOTE: poll_tennis.py must be running on each of these events — the "
          "tennis_branch\n      model reads its tape for the fit and the one-point branches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
