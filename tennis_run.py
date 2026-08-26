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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt
import poll_tennis as pt
from implied_model import ImpliedModel

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


def live_quote_board(a):
    """Self-contained: poll one event, fit, and show the market beside our quotes.

    Standalone rather than tape-consuming so a single event can be watched without a
    poll_tennis process running. Same quoting rule either way.

    STARTING MID-MATCH the fit has no history, so the first market price observed is
    seeded as one observation - enough for ImpliedModel to fit against its split prior,
    but THIN. p/q only becomes meaningful after a few real game boundaries, and the
    board marks how many it has.
    """
    feed = kt.Feed(rps=a.rps, verbose=False)
    mil = feed.milestone(a.event)
    if not mil:
        print(f"milestone not found for {a.event}")
        return 1
    det_m = mil.get("details") or {}
    mid_id = mil.get("id")

    mkts = feed.markets(a.event)
    mm = {m.get("ticker"): m for m in mkts if m.get("ticker")}
    if len(mm) != 2:
        print(f"{a.event}: {len(mm)} priced markets")
        return 1
    # SHARD GATE. Never quote a market we have not proven is on shard 3.
    auth_ok = True
    try:
        import tennis_parsed_markets as tpm
        auth = tpm._auth()
        for tk in mm:
            shard, code = tpm.market_shard(auth, tk)
            if shard != TENNIS_SHARD:
                print(f"REFUSED {tk}: exchange_index={shard!r} (HTTP {code})")
                return 1
        print(f"  shard check: both markets verified exchange_index={TENNIS_SHARD}")
    except Exception as e:
        print(f"  shard check unavailable ({e}) — refusing to quote")
        return 1

    prices = {t: kt.mid(mm[t]) for t in mm}
    me_tick = a.me_ticker or min(prices, key=lambda t: prices[t])
    opp_tick = next(t for t in mm if t != me_tick)
    d0 = feed.live_data([mid_id]).get(mid_id) or {}
    c1, c2 = d0.get("competitor1_id") or "", d0.get("competitor2_id") or ""
    st0 = kt.model_state(d0, c1, c2) or {}
    # which competitor is "me" — match the tracked ticker suffix against the market
    me_id, opp_id = c1, c2
    if kt.model_state(d0, c1, c2) is None:
        me_id, opp_id = c2, c1

    # best_of from the EXACT-SCORE market, which enumerates possible set scores and is
    # therefore ground truth. resolve_best_of deliberately refuses Kalshi's own
    # best_of=3 on a men's Grand Slam event, because it reports 3 for main-draw
    # best-of-FIVE matches; passing the market-verified value satisfies that guard
    # honestly instead of overriding it by hand.
    verified = None
    try:
        verified = kt.best_of_from_exact_market(feed, a.event)
    except Exception:
        pass
    try:
        best_of = kt.resolve_best_of(det_m, a.best_of, verified)
    except ValueError as e:
        print(f"  best_of unresolved: {e}")
        print("  (exact-score market returned %r)" % verified)
        return 1
    print(f"  best_of={best_of} (exact-score market said {verified!r}, "
          f"kalshi said {det_m.get('best_of')!r})")
    prior = kt.split_prior_for(det_m) if hasattr(kt, "split_prior_for") else (
        0.14 if (det_m.get("gender") or "").lower().startswith("f") else 0.28)
    tb = kt.resolve_final_set_tb(det_m, a.final_set_tb) if hasattr(kt, "resolve_final_set_tb") else 7

    print(f"  {mil.get('title')}  [best of {best_of}, split_prior {prior}, tb {tb}]")
    print(f"  quoting BOTH sides | me={me_tick}  opp={opp_tick}")
    print(f"  size {a.size} | margin {a.margin:.1f}c | DRY RUN — no order code in this process")
    print("=" * 100)

    model = ImpliedModel(best_of=best_of, first_server="me", split_prior=prior,
                         warn_split=False, final_set_tb=tb)
    last_key, n_obs, cycles = None, 0, 0
    try:
        while True:
            cycles += 1
            if a.max_cycles and cycles > a.max_cycles:
                break
            if os.path.exists(KILL):
                print("kill flag present — stopping."); break
            d = feed.live_data([mid_id]).get(mid_id)
            if d is None:
                time.sleep(a.interval); continue
            st = kt.model_state(d, me_id, opp_id)
            if st is None:
                time.sleep(a.interval); continue
            mkts = feed.markets(a.event)
            mm = {m.get("ticker"): m for m in mkts if m.get("ticker")}
            bid_me, ask_me = kt.top_of_book(mm.get(me_tick) or {})
            bid_opp, ask_opp = kt.top_of_book(mm.get(opp_tick) or {})
            px = kt.vig_free(kt.mid(mm.get(me_tick) or {}), kt.mid(mm.get(opp_tick) or {}))

            obs_state = {k: v for k, v in st.items()
                         if k not in ("points_me", "points_opp") and not k.startswith("_")}
            key = kt.boundary_key(st)
            if px is not None and key != last_key:
                model.observe(px, **obs_state)
                n_obs += 1
                last_key = key

            if model.p is None or not st.get("_points_known"):
                time.sleep(a.interval); continue
            pstate = {k: v for k, v in st.items() if not k.startswith("_")}
            ah = pt._point_ahead_block([("m", model)], pstate, st.get("server"))
            if "m" not in ah["win"]:
                time.sleep(a.interval); continue
            w, l = ah["win"]["m"], ah["lose"]["m"]
            val_me, val_opp = min(w, l), 1.0 - max(w, l)
            q_me = round(val_me - a.margin / 100.0, 2)
            q_opp = round(val_opp - a.margin / 100.0, 2)
            lbl = ['0', '15', '30', '40', 'AD']
            pm, po = st['points_me'], st['points_opp']
            intb = st['games_me'] == 6 and st['games_opp'] == 6
            pts = f"{pm}-{po}" if intb else (f"{lbl[pm]}-{lbl[po]}" if pm < 5 and po < 5 else "?")
            print(f"\n[{time.strftime('%H:%M:%S')}] {st['sets_me']}-{st['sets_opp']} sets  "
                  f"{st['games_me']}-{st['games_opp']} games  {pts:>7}  srv={st.get('server','?')}"
                  f"   fit p={model.p:.3f} q={model.q:.3f} on {n_obs} boundar"
                  f"{'y' if n_obs==1 else 'ies'}")
            print(f"     one-point bracket {abs(w-l)*100:5.2f}c   "
                  f"(win {w:.3f} / lose {l:.3f})")
            for tag, tick, val, q, b, ak in (
                    ("ME ", me_tick, val_me, q_me, bid_me, ask_me),
                    ("OPP", opp_tick, val_opp, q_opp, bid_opp, ask_opp)):
                if b is None or ak is None:
                    continue
                inband = a.min_price <= q <= a.max_price
                through = q >= ak
                if through:
                    note = "SKIP (would cross the offer — that is taking)"
                elif not inband:
                    note = f"SKIP (outside {a.min_price:.2f}-{a.max_price:.2f})"
                else:
                    note = f"QUOTE {q:.2f} x{a.size}   edge vs bid {(val-b)*100:+5.2f}c"
                print(f"     {tag} mkt {b:.2f}/{ak:.2f}   worst-branch value {val:.3f}   {note}")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    print(f"\n[{feed.n_get} GETs, {feed.n_429} rate-limited]  DRY RUN — nothing was placed.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", help="run self-contained on ONE event (a run.py-style "
                                    "live quote board) instead of consuming tapes")
    ap.add_argument("--me-ticker", help="force the tracked side; default = underdog")
    ap.add_argument("--best-of", type=int, choices=(3, 5))
    ap.add_argument("--final-set-tb", type=int, choices=(7, 10))
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--min-price", type=float, default=0.10)
    ap.add_argument("--max-price", type=float, default=0.90)
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--config", default="tennis_quoter_config.csv")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--churn-report", action="store_true",
                    help="summarise how often the quote would change, i.e. the write "
                         "rate a live quoter would draw on shard 3's budget")
    ap.add_argument("--max-cycles", type=int, default=0)
    a = ap.parse_args()

    if a.event:
        print("=" * 100)
        print("  TENNIS QUOTER — LIVE DRY RUN. No order-placement code in this process.")
        print("=" * 100)
        return live_quote_board(a)

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
