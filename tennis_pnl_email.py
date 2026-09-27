"""Daily set-1 taker P&L report — pregame anchor, size fired, size got, edge, cash edge, result.

Replaces the dog-windows OOS tracker in cron (26SEP24). That tracker measured SHADOW
triggers against a hypothetical maker fill; this reports what the live taker actually
did, from the exchange record rather than quoter/trades.csv (which flushes only at
shutdown, so mid-session it is hours behind — see pull_fills.py).

    .venv/bin/python3 tennis_pnl_email.py [--date 2026-09-23] [--days 1]
                                          [--pull] [--email you@x.com] [--once-per-date]

Sources:
  tapes/set1_taker_events.jsonl   type=fire -> pregame_vf, fair_c, lots requested, orders
  tapes/kalshi_fills.csv          exchange fills (--pull refreshes; ~10 GETs)
  tapes/kalshi_settlements.csv    market_result per ticker

Conventions that matter:
  * A fill is matched to a fire by ticker within [-5s, +120s]. The -5 is REQUIRED:
    pull_fills truncates created_time to whole seconds, so a fill 0.2s after the fire
    reads as -0.8s and a 0-floor window silently reports every fire as a zero-fill.
  * pull_fills normalises every fill to the ticker's YES price. action=buy -> cost is
    `price`; action=sell (we sold YES == bought NO) -> cost is 1-price. Both routes are
    long the set-1 leader, so cost is comparable to fair_c directly.
  * edge = fair_c - entry - fee/lot, fee from the exchange record.
  * cash edge = size got x edge (5000 lots x 8c = $400). Expected, not realised.
"""
import argparse, collections, csv, json, os, sys, time, datetime as dt

UTC = dt.timezone.utc
ROOT = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(ROOT, "tapes")
MARKER_F = os.path.join(TAPES, "_tennis_pnl_email_marker.txt")
WLO, WHI = -5.0, 120.0


def load_fires():
    out = []
    p = os.path.join(TAPES, "set1_taker_events.jsonl")
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") == "fire":
                out.append(r)
    out.sort(key=lambda r: r["ts"])
    return out


def load_fills(path):
    fills = collections.defaultdict(list)
    try:
        for r in csv.DictReader(open(path)):
            fills[r["ticker"]].append(dict(
                ts=int(r["created_ts"]), ticker=r["ticker"], action=r["action"],
                count=float(r["count"]), price=float(r["price"]),
                taker=r["is_taker"] == "True", fee=float(r["fee_cost"]), claim=None))
    except FileNotFoundError:
        pass
    return fills


def load_settlements(path):
    out = {}
    try:
        for r in csv.DictReader(open(path)):
            out[r["ticker"]] = (r.get("market_result") or "").lower()
    except FileNotFoundError:
        pass
    return out


def build(fires, fills, sett):
    """Claim each taker fill for its nearest fire, then summarise per fire."""
    for r in fires:
        for o in (r.get("orders") or []):
            for f in fills.get(o["ticker"], []):
                if not f["taker"]:
                    continue
                d = f["ts"] - r["ts"]
                if not (WLO <= d <= WHI):
                    continue
                if f["claim"] and abs(f["claim"][1]) <= abs(d):
                    continue
                f["claim"] = (r["event"], r["ts"], d)
    rows = []
    for r in fires:
        got = cost = fee = 0.0
        for o in (r.get("orders") or []):
            for f in fills.get(o["ticker"], []):
                if f["claim"] and f["claim"][0] == r["event"] and f["claim"][1] == r["ts"]:
                    c = f["price"] if f["action"] == "buy" else (1.0 - f["price"])
                    got += f["count"]; cost += f["count"] * c; fee += f["fee"]
        entry = 100 * cost / got if got else None
        fee_c = 100 * fee / got if got else 0.0
        edge = (r["fair_c"] - entry - fee_c) if entry is not None else None
        won = None
        if got:
            for o in (r.get("orders") or []):
                mr = sett.get(o["ticker"])
                if mr in ("yes", "no"):
                    w = (mr == "yes") if o["kalshi_side"] == "yes" else (mr == "no")
                    won = w if won is None else (won and w)
        pnl = None if (won is None or not got) else got * ((100.0 if won else 0.0) - entry) / 100.0 - fee
        pre = r.get("pregame_vf")
        rows.append(dict(ts=r["ts"], ev=r["event"], pre=pre, fair=r["fair_c"],
                         fired=float(r.get("lots") or 0), got=got, entry=entry,
                         edge=edge, cash=(got * edge / 100.0) if edge is not None else None,
                         won=won, pnl=pnl,
                         bucket=("-" if pre is None else
                                 ("A1" if pre > 0.5 and (entry or 0) >= 85 else
                                  ("A2" if pre > 0.5 else "B")))))
    return rows


def day_range(anchor, days, include_today):
    """The list of UTC dates to report, OLDEST first.

    Date-driven, not "the last N days that happened to have fires" (26SEP24): the cron runs
    at 05:25Z, so keying off fire-days meant a quiet previous day silently vanished from the
    report and the headline day could be a near-empty partial. Anchoring on YESTERDAY and
    listing explicit dates means a zero-fire day is reported AS zero, which is information.
    """
    a = dt.datetime.strptime(anchor, "%Y-%m-%d").replace(tzinfo=UTC)  # %F is strftime-only, not strptime
    out = [(a - dt.timedelta(days=i)).strftime("%F") for i in range(days - 1, -1, -1)]
    if include_today:
        today = dt.datetime.now(UTC).strftime("%F")
        if today not in out:
            out.append(today)
    return out


def default_anchor():
    """Yesterday UTC -- the most recent COMPLETE trading day."""
    return (dt.datetime.now(UTC) - dt.timedelta(days=1)).strftime("%F")


def report(rows, dates):
    by = collections.defaultdict(list)
    for x in rows:
        by[dt.datetime.fromtimestamp(x["ts"], UTC).strftime("%F")].append(x)
    L = []
    L.append(f"TENNIS SET-1 TAKER P&L — generated {dt.datetime.now(UTC):%d%b%y %H:%M} UTC".upper())
    for day in dates:
        s = sorted(by.get(day, []), key=lambda x: x["ts"])
        L.append("")
        L.append(f"=== {day} — {len(s)} fire(s) ===")
        if not s:
            L.append("       no fires")
            continue
        L.append(f"{'time':6} {'event':30} {'bkt':>3} {'pregame':>8} {'fair':>6} "
                 f"{'fired':>6} {'got':>6} {'fill%':>6} {'entry':>6} {'edge':>7} "
                 f"{'cash edge':>10} {'result':>7} {'pnl $':>10}")
        for x in s:
            fp = (100 * x["got"] / x["fired"]) if x["fired"] else 0.0
            L.append(
                f"{dt.datetime.fromtimestamp(x['ts'], UTC):%H:%M} "
                f"{x['ev']:30} {x['bucket']:>3} "
                f"{('%.3f' % x['pre']) if x['pre'] is not None else '?':>8} "
                f"{x['fair']:6.1f} {x['fired']:6.0f} {x['got']:6.0f} {fp:5.0f}% "
                f"{('%.1f' % x['entry']) if x['entry'] is not None else '-':>6} "
                f"{('%+.2f' % x['edge']) if x['edge'] is not None else '-':>7} "
                f"{(f'${x["cash"]:+,.0f}') if x['cash'] is not None else '-':>10} "
                f"{('WIN' if x['won'] else 'loss') if x['won'] is not None else ('pending' if x['got'] else 'NO FILL'):>7} "
                f"{(f'{x["pnl"]:+,.2f}') if x['pnl'] is not None else '-':>10}")
        fired = sum(x["fired"] for x in s); got = sum(x["got"] for x in s)
        cash = sum(x["cash"] or 0 for x in s)
        res = [x for x in s if x["pnl"] is not None]
        pnl = sum(x["pnl"] for x in res)
        lots = sum(x["got"] for x in res)
        L.append(f"{'':6} {'TOTAL':30} {'':3} {'':8} {'':6} {fired:6.0f} {got:6.0f} "
                 f"{(100*got/fired if fired else 0):5.0f}% {'':6} {'':7} {f'${cash:+,.0f}':>10} "
                 f"{'':7} {f'{pnl:+,.2f}':>10}")
        if res:
            w = sum(1 for x in res if x["won"])
            L.append(f"       settled {len(res)}/{len(s)}  win {w}/{len(res)} ({100*w/len(res):.0f}%)  "
                     f"realised {100*pnl/lots:+.2f}c/lot on {lots:,.0f} lots  "
                     f"vs cash edge {100*sum(x['cash'] for x in res)/lots:+.2f}c/lot")
        nofill = [x for x in s if not x["got"]]
        if nofill:
            L.append(f"       ** {len(nofill)} fire(s) with NO FILL: "
                     + ", ".join(x["ev"] for x in nofill) + " **")
        pend = [x for x in s if x["got"] and x["won"] is None]
        if pend:
            L.append(f"       {len(pend)} unsettled (retried next run): "
                     + ", ".join(x["ev"] for x in pend))
    L.append("")
    L.append("cash edge = size got x edge (5000 lots x 8c = $400): EXPECTED, not realised. "
             "pnl rides the position to settlement and is net of the taker fee. "
             "pregame = the leader's frozen pregame vig-free prob; bkt A1 = favourite "
             "re-pricing lag (entry >=85c), A2 = modest favourite, B = pregame dog (the tau trade).")
    return "\n".join(L)


# ---------------------------------------------------------------- HTML email
# Palette: the dataviz skill's STATUS steps (good/critical) on its light surface.
# Rules honoured: status colour is reserved (never reused as a category, so the
# A1/A2/B bucket stays plain ink), and it never carries meaning alone -- every
# result cell pairs the colour with a glyph AND the word. Text greens use the
# skill's light success-text step #006300, because status-good #0ca30c is 3.27:1
# on #fcfcfb and would fail body-text contrast.
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
HAIRLINE  = "rgba(11,11,11,0.10)"
GOOD_TXT  = "#006300"
GOOD_BG   = "#eaf5ea"
BAD_TXT   = "#d03b3b"
BAD_BG    = "#fbecec"
WARN_TXT  = "#8a6100"
WARN_BG   = "#fdf6e6"
ZEBRA     = "#f6f6f4"
MONO      = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS      = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pill(text, glyph, fg, bg):
    return (f'<span style="padding:2px 7px;border-radius:10px;background:{bg};'
            f'color:{fg};font:600 11px/1.4 {SANS}">{glyph}&nbsp;{text}</span>')


def _res_pill(x):
    if not x["got"]:
        return _pill("NO FILL", "&#9675;", INK_2, "#f0efec")
    if x["won"] is None:
        return _pill("pending", "&#9203;", WARN_TXT, WARN_BG)
    return (_pill("WIN", "&#10003;", GOOD_TXT, GOOD_BG) if x["won"]
            else _pill("LOSS", "&#10007;", BAD_TXT, BAD_BG))


def _num(v, fmt="{:+,.2f}", colour=True):
    if v is None:
        return f'<span style="color:{MUTED}">&ndash;</span>'
    c = INK
    if colour:
        c = GOOD_TXT if v > 0 else (BAD_TXT if v < 0 else INK_2)
    return f'<span style="color:{c}">{fmt.format(v)}</span>'


def _tile(label, value, sub=None, fg=None):
    fg = fg or INK
    sub_html = (f'<div style="font:400 11px/1.4 {SANS};color:{MUTED};margin-top:2px">{sub}</div>'
                if sub else "")
    return (f'<td style="padding:10px 14px;border:1px solid {HAIRLINE};border-radius:6px;'
            f'background:{SURFACE};vertical-align:top">'
            f'<div style="font:600 10px/1.3 {SANS};color:{MUTED};letter-spacing:.06em;'
            f'text-transform:uppercase">{label}</div>'
            f'<div style="font:600 19px/1.25 {MONO};color:{fg};margin-top:3px">{value}</div>'
            f'{sub_html}</td>')


def html_report(rows, dates):
    import collections as _c
    by = _c.defaultdict(list)
    for x in rows:
        by[dt.datetime.fromtimestamp(x["ts"], UTC).strftime("%F")].append(x)
    out = [f'<div style="background:{SURFACE};padding:18px;font:{SANS}">']
    out.append(f'<div style="font:600 17px/1.3 {SANS};color:{INK}">Tennis set-1 taker P&amp;L</div>'
               f'<div style="font:400 12px/1.5 {SANS};color:{MUTED};margin:2px 0 16px">'
               f'generated {dt.datetime.now(UTC):%d %b %Y %H:%M} UTC &middot; exchange fill record'
               f'</div>')
    for day in dates:
        s = sorted(by.get(day, []), key=lambda x: x["ts"])
        if not s:
            out.append(f'<div style="font:600 14px/1.3 {SANS};color:{INK};margin:18px 0 6px">{day}</div>'
                       f'<div style="font:400 12px/1.5 {SANS};color:{MUTED};padding:8px 10px;'
                       f'background:{ZEBRA};border-radius:5px">no fires</div>')
            continue
        fired = sum(x["fired"] for x in s); got = sum(x["got"] for x in s)
        cash = sum(x["cash"] or 0 for x in s)
        res = [x for x in s if x["pnl"] is not None]
        pnl = sum(x["pnl"] for x in res); lots = sum(x["got"] for x in res)
        wins = sum(1 for x in res if x["won"])
        out.append(f'<div style="font:600 14px/1.3 {SANS};color:{INK};margin:18px 0 8px">{day}</div>')
        # KPI row
        out.append('<table role="presentation" cellspacing="6" cellpadding="0" '
                   'style="border-collapse:separate;margin-bottom:12px"><tr>')
        out.append(_tile("fires", f"{len(s)}"))
        out.append(_tile("lots filled", f"{got:,.0f}",
                         sub=f"of {fired:,.0f} fired &middot; {(100*got/fired if fired else 0):.0f}%"))
        out.append(_tile("win rate", (f"{wins}/{len(res)}" if res else "&ndash;"),
                         sub=(f"{100*wins/len(res):.0f}%" if res else None)))
        out.append(_tile("cash edge", f"${cash:+,.0f}", sub="expected at entry"))
        out.append(_tile("realised P&amp;L", f"${pnl:+,.0f}",
                         sub=(f"{100*pnl/lots:+.2f}c/lot" if lots else None),
                         fg=(GOOD_TXT if pnl > 0 else BAD_TXT if pnl < 0 else INK)))
        out.append('</tr></table>')
        # detail table
        cols = [("time","left"),("event","left"),("bkt","left"),("pregame","right"),
                ("fair","right"),("entry","right"),("fired","right"),("got","right"),
                ("edge","right"),("cash edge","right"),("result","left"),("P&amp;L","right")]
        out.append(f'<table role="presentation" cellspacing="0" cellpadding="0" '
                   f'style="border-collapse:collapse;width:100%;border:1px solid {HAIRLINE};'
                   f'border-radius:6px;overflow:hidden">')
        out.append(f'<tr style="background:#f0efec">')
        for c, al in cols:
            out.append(f'<th style="text-align:{al};padding:7px 9px;font:600 10px/1.3 {SANS};'
                       f'color:{INK_2};letter-spacing:.05em;text-transform:uppercase;'
                       f'border-bottom:1px solid {HAIRLINE};white-space:nowrap">{c}</th>')
        out.append('</tr>')
        # SIZE DISCIPLINE: Gmail clips a message body over ~102KB ("View entire
        # message"), and a naive per-cell inline style ran 116KB for two days. font,
        # colour and white-space are set ONCE on the <tr> and inherited; each <td>
        # carries only padding, border and alignment. Keep it that way when editing.
        L = f'padding:6px 9px;border-bottom:1px solid {HAIRLINE}'
        R = L + ';text-align:right'
        for i, x in enumerate(s):
            bg = ZEBRA if i % 2 else SURFACE
            short = x["ev"].split("-", 1)[-1]
            partial = x["got"] and x["fired"] and x["got"] < x["fired"] * 0.99
            out.append(f'<tr style="background:{bg};font:400 12px/1.45 {MONO};'
                       f'color:{INK};white-space:nowrap">')
            out.append(f'<td style="{L}">{dt.datetime.fromtimestamp(x["ts"], UTC):%H:%M}</td>')
            out.append(f'<td style="{L};font-family:{SANS}">{_esc(short)}</td>')
            out.append(f'<td style="{L};color:{INK_2}">{x["bucket"]}</td>')
            out.append(f'<td style="{R};color:{INK_2}">'
                       f'{("%.3f" % x["pre"]) if x["pre"] is not None else "&ndash;"}</td>')
            out.append(f'<td style="{R};color:{INK_2}">{x["fair"]:.1f}</td>')
            out.append(f'<td style="{R}">'
                       f'{("%.1f" % x["entry"]) if x["entry"] is not None else "&ndash;"}</td>')
            out.append(f'<td style="{R};color:{INK_2}">{x["fired"]:,.0f}</td>')
            out.append(f'<td style="{R}'
                       + (f';color:{BAD_TXT};font-weight:600' if partial else '') + f'">{x["got"]:,.0f}</td>')
            out.append(f'<td style="{R}">{_num(x["edge"], "{:+.2f}")}</td>')
            out.append(f'<td style="{R}">'
                       f'{_num(x["cash"], "${:+,.0f}", colour=False) if x["cash"] is not None else "&ndash;"}</td>')
            out.append(f'<td style="{L}">{_res_pill(x)}</td>')
            out.append(f'<td style="{R};font-weight:600">{_num(x["pnl"])}</td>')
            out.append('</tr>')
        # total row
        out.append(f'<tr style="background:#f0efec">')
        tdf = (f'padding:7px 9px;font:600 12px/1.45 {MONO};color:{INK};white-space:nowrap')
        out.append(f'<td style="{tdf}" colspan="6">TOTAL &nbsp;<span style="font-family:{SANS};'
                   f'font-weight:400;color:{INK_2}">{len(s)} fires</span></td>')
        out.append(f'<td style="{tdf};text-align:right">{fired:,.0f}</td>')
        out.append(f'<td style="{tdf};text-align:right">{got:,.0f}</td>')
        out.append(f'<td style="{tdf};text-align:right;color:{MUTED}">&ndash;</td>')
        out.append(f'<td style="{tdf};text-align:right">${cash:+,.0f}</td>')
        out.append(f'<td style="{tdf}"></td>')
        out.append(f'<td style="{tdf};text-align:right">{_num(pnl)}</td>')
        out.append('</tr></table>')
        nofill = [x for x in s if not x["got"]]
        if nofill:
            out.append(f'<div style="margin-top:8px;padding:8px 10px;background:{BAD_BG};'
                       f'border-radius:5px;font:400 12px/1.5 {SANS};color:{BAD_TXT}">'
                       f'&#9888; {len(nofill)} fire(s) with no fill: '
                       + ", ".join(_esc(x["ev"]) for x in nofill) + '</div>')
        pend = [x for x in s if x["got"] and x["won"] is None]
        if pend:
            out.append(f'<div style="margin-top:8px;padding:8px 10px;background:{WARN_BG};'
                       f'border-radius:5px;font:400 12px/1.5 {SANS};color:{WARN_TXT}">'
                       f'&#9203; {len(pend)} unsettled, retried next run: '
                       + ", ".join(_esc(x["ev"]) for x in pend) + '</div>')
    out.append(f'<div style="margin-top:18px;font:400 11px/1.6 {SANS};color:{MUTED};'
               f'border-top:1px solid {HAIRLINE};padding-top:10px">'
               f'<b>cash edge</b> = size got &times; edge (5,000 lots &times; 8c = $400) &mdash; '
               f'expected at entry, not realised. <b>P&amp;L</b> rides to settlement, net of the '
               f'taker fee. <b>pregame</b> is the leader\'s frozen pregame vig-free prob. '
               f'<b>bkt</b>: A1 favourite re-pricing lag (entry &ge;85c), A2 modest favourite, '
               f'B pregame dog (the tau trade).</div></div>')
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None, metavar="YYYY-MM-DD",
                    help="anchor date (UTC). Default: YESTERDAY, the most recent COMPLETE "
                         "day -- the 05:25Z cron must never headline a partial current day.")
    ap.add_argument("--days", type=int, default=1,
                    help="how many dates to show, counting BACK from --day (default 1)")
    ap.add_argument("--include-today", action="store_true",
                    help="also append the partial current UTC day")
    ap.add_argument("--pull", action="store_true", help="refresh fills/settlements first (~10 GETs)")
    ap.add_argument("--fills", default=os.path.join(TAPES, "kalshi_fills.csv"))
    ap.add_argument("--settlements", default=os.path.join(TAPES, "kalshi_settlements.csv"))
    ap.add_argument("--email")
    ap.add_argument("--once-per-date", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)          # cwd-independent: cron may invoke this from anywhere
    if a.pull:
        os.system(f"cd {ROOT} && .venv/bin/python3 pull_fills.py --out {a.fills} >/dev/null 2>&1")
    anchor = a.day or default_anchor()
    dates = day_range(anchor, a.days, a.include_today)
    rows = build(load_fires(), load_fills(a.fills), load_settlements(a.settlements))
    rep = report(rows, dates)
    print(rep)
    if a.email:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if a.once_per_date and os.path.exists(MARKER_F) \
           and open(MARKER_F).read().strip() == today:
            print(f"[email] already sent {today}, skipping"); return
        sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/general_level_based_quoting")
        from _eod_report import send_email
        # Gmail clips a body over ~102KB and hides the rest behind "View entire
        # message" -- which would silently swallow the TOTAL row. Shed days until it
        # fits rather than letting that happen.
        d2, html = list(dates), html_report(rows, dates)
        while len(html) > 95_000 and len(d2) > 1:
            d2 = d2[1:]                     # drop the OLDEST date, keep the anchor day
            html = html_report(rows, d2)
            print(f"[email] body over 95KB, trimmed to {d2} ({len(html):,} bytes)")
        if len(html) > 95_000:
            print(f"[email] WARNING body {len(html):,} bytes with a single day — Gmail may clip it")
        ok, msg = send_email(a.email, f"Tennis set-1 taker P&L — {anchor}", html, rep)
        print(f"[email] {'sent' if ok else 'FAILED'}: {msg}")
        if ok and a.once_per_date:
            open(MARKER_F, "w").write(today)


if __name__ == "__main__":
    main()
