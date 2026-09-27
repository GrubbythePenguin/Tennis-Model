"""Tennis end-of-day report: P&L per match and the parameters that produced it.

    .venv/bin/python3 pull_fills.py                      # refresh the exchange record (1 GET)
    python3 tennis_eod.py [--tag 26AUG28 26AUG29] [--fills tapes/kalshi_fills.csv] [--out PATH]

Per match: series, tournament, edge asked (A/B arm or series default), lots, fills,
contracts, buy share, max |net| position, realised P&L (FIFO round trips), open position
at settlement (tape final score) or last price, fees, net, P&L per fill / per contract,
5s and one-point markouts. Rollups by edge arm, by series, and in total. The fill record
is Kalshi's (quoter/trades.csv only flushes at shutdown); fees are Kalshi's fee_cost.
"""
import argparse, bisect, collections, csv, glob, json, os, statistics as st, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fill_markouts import point_class


def load_tape(ev):
    rows = []
    try:
        for line in open(f"tapes/{ev}.jsonl", errors="replace"):
            if '"status"' not in line: continue
            try: r = json.loads(line)
            except Exception: continue
            rows.append((r["ts"], r.get("vig_free"), r.get("status"), r.get("state") or {}))
    except FileNotFoundError:
        pass
    return rows


def fifo(fl):
    longs, shorts, realised = [], [], 0.0
    for t, px, q in sorted(fl):
        book, opp = (longs, shorts) if q > 0 else (shorts, longs); q = abs(q)
        while q > 0 and opp:
            opx, oq = opp[0]; m = min(q, oq)
            realised += (opx - px) * m if book is longs else (px - opx) * m
            q -= m; oq -= m
            if oq == 0: opp.pop(0)
            else: opp[0] = (opx, oq)
        if q > 0: book.append((px, q))
    pos = sum(q for _, q in longs) - sum(q for _, q in shorts)
    avg = (sum(p * q for p, q in longs) / sum(q for _, q in longs)) if longs else ((sum(p * q for p, q in shorts) / sum(q for _, q in shorts)) if shorts else 0.0)
    return realised, pos, avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="*", help="event date tags, e.g. 26AUG28 26AUG29 (default: fills in the last 30h)")
    ap.add_argument("--fills", default="tapes/kalshi_fills.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = {}
    for lp in glob.glob("tapes/*.log.json"):
        try: d = json.load(open(lp)); m = d.get("meta") or {}
        except Exception: continue
        if m.get("me_ticker"): meta[m["event"]] = m
    disc = {}
    try: disc = json.load(open("tapes/discovered.json"))
    except Exception: pass
    ab = {}
    try: ab = json.load(open("tapes/ab_edges.json"))
    except Exception: pass
    params = {r["market_prefix"]: r for r in csv.DictReader(open("quoter/tennis_market_parameters.csv"))}
    settle = {}                                   # ticker -> 1.0 / 0.0 from Kalshi settlements
    try:
        for r in csv.DictReader(open("tapes/kalshi_settlements.csv")):
            if r.get("market_result") in ("yes", "no"): settle[r["ticker"]] = 1.0 if r["market_result"] == "yes" else 0.0
    except Exception: pass
    now = time.time()
    fills = collections.defaultdict(list)
    for f in csv.DictReader(open(a.fills)):
        t = f["ticker"]; ev = t.rsplit("-", 1)[0]
        if a.tag and not any(tag in ev for tag in a.tag): continue
        if not a.tag and now - int(f["created_ts"]) > 30 * 3600: continue
        fills[ev].append(f)
    rows = []
    for ev, fl in fills.items():
        m = meta.get(ev)
        if not m: continue
        tape = load_tape(ev); ts = [r[0] for r in tape]
        priced = [r for r in tape if r[1] is not None]
        last_state = tape[-1][3] if tape else {}
        settled = last_state.get("sets_me", 0) == 2 or last_state.get("sets_opp", 0) == 2 or (tape and tape[-1][2] in ("closed", "settled", "ended"))
        if m["me_ticker"] in settle:
            mark = settle[m["me_ticker"]]; settled = True
        elif last_state.get("sets_me", 0) == 2 or last_state.get("sets_opp", 0) == 2:
            mark = 1.0 if last_state["sets_me"] > last_state["sets_opp"] else 0.0
        else:
            mark = priced[-1][1] if priced else None
        me_fl, fees, mo5, mo1 = [], 0.0, [], []
        net = 0; maxnet = 0; buys = 0
        for f in sorted(fl, key=lambda x: int(x["created_ts"])):
            px = float(f["price"]); c = float(f["count"]); is_me = f["ticker"] == m["me_ticker"]; buy = f["action"] == "buy"
            me_px = px if is_me else 1 - px; long_me = (buy and is_me) or ((not buy) and (not is_me))
            me_fl.append((int(f["created_ts"]), me_px, c if long_me else -c)); fees += float(f["fee_cost"] or 0)
            net += c if long_me else -c; maxnet = max(maxnet, abs(net)); buys += long_me
            t = int(f["created_ts"]); i0 = bisect.bisect_right(ts, t) - 1
            if i0 >= 0 and t - ts[i0] <= 30 and priced:
                sg = 1 if long_me else -1
                def at(tt):
                    j = min(len(tape) - 1, max(0, bisect.bisect_right(ts, tt) - 1)); r = tape[j]
                    return r[1] if r[1] is not None else None
                v5 = at(t + 5)
                if v5 is not None: mo5.append(sg * (v5 - me_px) * 100)
                k0 = tuple(tape[i0][3].get(x) for x in ("sets_me", "sets_opp", "games_me", "games_opp", "points_me", "points_opp")); j = i0 + 1
                while j < len(tape) and tuple(tape[j][3].get(x) for x in ("sets_me", "sets_opp", "games_me", "games_opp", "points_me", "points_opp")) == k0: j += 1
                if j < len(tape):
                    v1 = at(tape[j][0] + 5)
                    if v1 is not None: mo1.append(sg * (v1 - me_px) * 100)
        realised, pos, avg = fifo(me_fl)
        openpnl = (mark - avg) * pos if mark is not None else 0.0
        contracts = sum(abs(q) for _, _, q in me_fl)
        prefix = ev.split("-", 1)[0]; prow = params.get(prefix, {})
        edge = ab.get(ev, {}).get("edge", float(prow.get("min_edge", 0) or 0))
        d = disc.get(ev, {}); tourn = m.get("tournament") or d.get("tournament") or d.get("competition") or m.get("tour") or "?"
        rows.append(dict(ev=ev, short=ev.rsplit("-", 1)[-1][7:], series=prefix.replace("MATCH", "").replace("KX", "").replace("CHALLENGER", "CH"), tourn=tourn[:22],
                         edge=edge, lots=prow.get("volumes", "?"), fills=len(fl), contracts=contracts, buy_share=buys / len(fl),
                         maxnet=maxnet, realised=realised, pos=pos, avg=avg, mark=mark, settled=settled, open=openpnl, fees=fees,
                         net=realised + openpnl - fees, mo5=st.mean(mo5) if mo5 else None, mo1=st.mean(mo1) if mo1 else None,
                         final=f"{last_state.get('sets_me')}-{last_state.get('sets_opp')} {last_state.get('games_me')}-{last_state.get('games_opp')}"))
    if not rows:
        print("no matches"); return
    rows.sort(key=lambda r: r["net"])
    out = []
    P = out.append
    P(f"TENNIS EOD — {time.strftime('%Y-%m-%d %H:%M', time.gmtime())}Z — {len(rows)} matches, {sum(r['fills'] for r in rows)} fills, {sum(r['contracts'] for r in rows):.0f} contracts   (fills = Kalshi record; marks = settlement where the match ended, else last price)")
    P("")
    hdr = f"{'match':<8}{'series':<7}{'tournament':<23}{'edge':>5}{'lots':>5}{'fills':>6}{'ctrs':>6}{'buy%':>5}{'max|net|':>9}{'realised':>9}{'open':>8}{'fees':>7}{'NET':>8}{'/fill':>7}{'/ctr':>6}{'mo5s':>6}{'mo1pt':>6}  {'mark':>5} final"
    P(hdr); P("-" * len(hdr))
    for r in rows:
        P(f"{r['short']:<8}{r['series']:<7}{r['tourn']:<23}{r['edge']:>4.0f}c{r['lots']:>5}{r['fills']:>6}{r['contracts']:>6.0f}{r['buy_share']*100:>4.0f}%{r['maxnet']:>9.0f}{r['realised']:>+9.2f}{r['open']:>+8.2f}{r['fees']:>7.2f}{r['net']:>+8.2f}{r['net']/r['fills']*100:>+6.1f}c{r['net']/r['contracts']*100 if r['contracts'] else 0:>+5.1f}c{(r['mo5'] if r['mo5'] is not None else float('nan')):>+6.2f}{(r['mo1'] if r['mo1'] is not None else float('nan')):>+6.2f}  {(r['mark']*100 if r['mark'] is not None else float('nan')):>5.0f}{'*' if not r['settled'] else ' '} {r['final']}")
    P("-" * len(hdr))
    def roll(lbl, R):
        if not R: return
        f_ = sum(r["fills"] for r in R); c_ = sum(r["contracts"] for r in R)
        P(f"{lbl:<44}{'':>5}{f_:>6}{c_:>6.0f}{sum(r['buy_share']*r['fills'] for r in R)/f_*100:>4.0f}%{'':>9}{sum(r['realised'] for r in R):>+9.2f}{sum(r['open'] for r in R):>+8.2f}{sum(r['fees'] for r in R):>7.2f}{sum(r['net'] for r in R):>+8.2f}{sum(r['net'] for r in R)/f_*100:>+6.1f}c{sum(r['net'] for r in R)/c_*100 if c_ else 0:>+5.1f}c{st.mean(r['mo5'] for r in R if r['mo5'] is not None) if any(r['mo5'] is not None for r in R) else float('nan'):>+6.2f}{st.mean(r['mo1'] for r in R if r['mo1'] is not None) if any(r['mo1'] is not None for r in R) else float('nan'):>+6.2f}   ({len(R)} matches)")
    P(""); P("BY EDGE ARM")
    for e in sorted({r["edge"] for r in rows}): roll(f"  edge {e:.0f}c", [r for r in rows if r["edge"] == e])
    P(""); P("BY SERIES")
    for s_ in sorted({r["series"] for r in rows}): roll(f"  {s_}", [r for r in rows if r["series"] == s_])
    P(""); roll("TOTAL", rows)
    P(""); P("* = match not finished: open P&L is at last price, not settlement.  mo5s / mo1pt = mean markout vs vig-free mid at 5s / next point (gross, c/contract).")
    txt = "\n".join(out); print(txt)
    op = a.out or f"tapes/eod_{'_'.join(a.tag) if a.tag else time.strftime('%y%b%d', time.gmtime()).upper()}.txt"
    open(op, "w").write(txt + "\n"); print(f"\n-> {op}")


if __name__ == "__main__":
    main()
