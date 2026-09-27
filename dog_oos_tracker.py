"""Daily out-of-sample tracker for the dog-window strategies (set-1 leader, set-3 dog).

Replays every trigger the live models logged to tapes/set1_dog_events.jsonl and
tapes/set3_dog_events.jsonl against the match result, priced at the model's own
frozen anchor (set1: leader vig-free; set3: dog mid), maker fill, ride to
settlement — the same convention as the 26SEP11 study (TENNIS_SET1_DOG_LEADER.md).
The point: the strategies keep SHADOW-logging triggers while disabled, so this
measures whether the edge is holding out of sample before re-enabling.

    .venv/bin/python3 dog_oos_tracker.py [--email you@x.com] [--once-per-date]
                                         [--api-budget 40] [--size 1000]

Match results, tried in order (cached forever in tapes/_dog_oos_results.json):
  1. cache  2. the event's own tape tail (sets==2, else final mid >0.9/<0.1)
  3. wsbook capture tail (mid >90c/<10c)  4. public /markets/{ticker} result —
  only for triggers older than 6h, throttled 1/s, capped by --api-budget.
Unsettled matches stay pending and are retried on the next run.

Cron (installed 26SEP17): daily 05:25 UTC, after the slate and next to the other
EOD reports; emails via EOD_SMTP_* (loaded by _eod_report's import).
"""
import argparse, json, math, os, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(ROOT, "tapes")
CACHE_F = os.path.join(TAPES, "_dog_oos_results.json")
MARKER_F = os.path.join(TAPES, "_dog_oos_email_marker.txt")
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# in-sample references (studies through 26SEP10; see scope docs + model docstrings)
REF = {"set1": "in-sample ref: +11.5pts at mid, wr 76.1%% vs 64.6c (dog-leader n=134)",
       "set3": "in-sample ref: +16.9pts, wr 59.0%% vs 42.1%% (ITF n=61)"}
OOS_SINCE = "26SEP11"   # both models went live / study data ended 26SEP10

_TAG = re.compile(r"-(26[A-Z]{3}\d{2})")
def tag_of(ev):
    m = _TAG.search(ev)
    return m.group(1) if m else "?"

def tag_key(tag):    # 26SEP16 -> sortable
    months = {m: i for i, m in enumerate(
        ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"], 1)}
    return (int(tag[:2]), months.get(tag[2:5], 0), int(tag[5:7]))

def load_triggers(name):
    trigs = {}
    path = os.path.join(TAPES, f"{name}_dog_events.jsonl")
    if not os.path.exists(path): return trigs, None
    last_ts = None
    for line in open(path):
        try: e = json.loads(line)
        except Exception: continue
        last_ts = e.get("ts", last_ts)
        if e.get("type") == "trigger" and e["event"] not in trigs:
            trigs[e["event"]] = e
    return trigs, last_ts

def tail_rows(path, nbytes):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "replace").splitlines()

def winner_from_tape(ev):
    """True/False = 'me' won, None = tape not decisive."""
    fp = os.path.join(TAPES, ev + ".jsonl")
    if not os.path.exists(fp): return None
    last = None
    for line in reversed(tail_rows(fp, 300_000)):
        try: r = json.loads(line)
        except Exception: continue
        st = r.get("state") or {}
        if st.get("sets_me", 0) == 2: return True
        if st.get("sets_opp", 0) == 2: return False
        if last is None: last = r
    if last is not None:
        m = last.get("mid_me")
        if m is not None and m > 0.9: return True
        if m is not None and m < 0.1: return False
    return None

def winner_from_wsbook(ev):
    fp = os.path.join(TAPES, "wsbook_" + ev + ".jsonl")
    if not os.path.exists(fp): return None
    for line in reversed(tail_rows(fp, 50_000)):
        try: r = json.loads(line)
        except Exception: continue
        me = r.get("me")
        if me and len(me) == 2 and me[0] is not None and me[1] is not None:
            mid = (me[0] + me[1]) / 2
            if mid > 90: return True
            if mid < 10: return False
            return None
    return None

def me_ticker_of(ev):
    try:
        return (json.load(open(os.path.join(TAPES, ev + ".log.json")))
                .get("meta") or {}).get("me_ticker")
    except Exception:
        return None

def api_result(ticker):
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"{KALSHI}/markets/{ticker}", timeout=10) as r:
                res = (json.load(r).get("market") or {}).get("result", "").lower()
                return res if res in ("yes", "no") else None
        except Exception:
            time.sleep(2)
    return None

def resolve_all(trig_sets, api_budget):
    cache = {}
    if os.path.exists(CACHE_F):
        try: cache = json.load(open(CACHE_F))
        except Exception: cache = {}
    api_used = 0; pending = []
    now = time.time()
    for name, trigs in trig_sets.items():
        for ev, e in trigs.items():
            # keyed by TARGET ticker, not event: one event can trigger both
            # strategies with different targets (set1 leader vs set3 dog), and
            # "won" is target-relative
            target = e.get("leader") or e.get("dog")
            if target in cache: continue
            me_tk = me_ticker_of(ev)
            won = None; src = None
            wm = winner_from_tape(ev)
            if wm is not None and me_tk:
                won, src = (wm if target == me_tk else not wm), "tape"
            if won is None:
                wm = winner_from_wsbook(ev)
                if wm is not None and me_tk:
                    won, src = (wm if target == me_tk else not wm), "wsbook"
            if won is None and now - e["ts"] > 6 * 3600 and api_used < api_budget:
                api_used += 1; time.sleep(1.0)
                res = api_result(target)
                if res is not None:
                    won, src = res == "yes", "api"
            if won is None:
                pending.append(ev)
            else:
                cache[target] = {"won": won, "src": src}
    tmp = CACHE_F + ".tmp"
    json.dump(cache, open(tmp, "w")); os.replace(tmp, CACHE_F)
    return cache, pending, api_used

def price_of(name, e):
    return e["theo_c"] / 100.0 if name == "set1" else e["dog_mid_c"] / 100.0

def stats(rows, size):
    n = len(rows)
    if not n: return None
    wr = sum(1 for _, w in rows if w) / n
    imp = sum(p for p, _ in rows) / n
    gross = sum(size * ((1 if w else 0) - p) for p, w in rows)
    var = sum(size ** 2 * p * (1 - p) for p, _ in rows)
    z = gross / math.sqrt(var) if var else 0.0
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    return n, wr, imp, 100 * (wr - imp), gross, z, pval

def fmt(label, st):
    if st is None: return f"  {label:<10} n=0"
    n, wr, imp, edge, gross, z, p = st
    return (f"  {label:<10} n={n:3d}  wr={wr:.3f}  implied={imp:.3f}  "
            f"edge={edge:+6.1f}pts  pnl=${gross:+10,.0f}  z={z:+.2f} p={p:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", help="also email the report (EOD_SMTP_* env)")
    ap.add_argument("--once-per-date", action="store_true",
                    help="skip the email if one already went out today (UTC)")
    ap.add_argument("--api-budget", type=int, default=40)
    ap.add_argument("--size", type=int, default=1000, help="lots per event for $ pnl")
    a = ap.parse_args()

    trig_sets = {}; last_ts = 0
    for name in ("set1", "set3"):
        trig_sets[name], lt = load_triggers(name)
        last_ts = max(last_ts, lt or 0)
    cache, pending, api_used = resolve_all(trig_sets, a.api_budget)

    lines = []
    now = time.time()
    lines.append(f"DOG WINDOWS OUT-OF-SAMPLE TRACKER — "
                 f"{time.strftime('%d%b%y %H:%M', time.gmtime(now)).upper()} UTC")
    feed_age_h = (now - last_ts) / 3600 if last_ts else 1e9
    feed = (f"last model event {feed_age_h:.1f}h ago"
            + ("" if feed_age_h < 24 else "  ** FEED DARK — are the dog_windows "
               "config rows still loaded in a running quoter? shadow logging is "
               "what feeds this tracker **"))
    lines.append(f"feed: {feed}   pending results: {len(pending)}   api lookups this run: {api_used}")
    for name, title in (("set1", "SET-1 LEADER"), ("set3", "SET-3 DOG")):
        lines.append("")
        lines.append(f"{title}   ({REF[name] % ()})")
        rows_by_tag = {}
        oos = []
        for ev, e in trig_sets[name].items():
            c = cache.get(e.get("leader") or e.get("dog"))
            if not c: continue
            pr = price_of(name, e)
            tg = tag_of(ev)
            rows_by_tag.setdefault(tg, []).append((pr, c["won"]))
            if tag_key(tg) >= tag_key(OOS_SINCE): oos.append((pr, c["won"]))
        for tg in sorted(rows_by_tag, key=tag_key):
            lines.append(fmt(tg, stats(rows_by_tag[tg], a.size)))
        lines.append(fmt(f"OOS>={OOS_SINCE}", stats(oos, a.size)))
        st = stats(oos, a.size)
        if st:
            verdict = ("HOLDING (edge positive)" if st[3] > 0 else "NOT HOLDING (edge negative)")
            lines.append(f"  verdict: {verdict} over {st[0]} OOS events")
    lines.append("")
    lines.append(f"assumptions: {a.size} lots/event filled at the frozen anchor "
                 "(set1 leader vig-free / set3 dog mid), maker no-fee, ride to "
                 "settlement — the 26SEP11 study convention. Fills are NOT "
                 "guaranteed at that size; treat $ as edge x volume, not realized pnl.")
    report = "\n".join(lines)
    print(report)

    if a.email:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if a.once_per_date and os.path.exists(MARKER_F) \
           and open(MARKER_F).read().strip() == today:
            print(f"[email] already sent {today}, skipping"); return
        sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/general_level_based_quoting")
        from _eod_report import send_email   # loads .env for EOD_SMTP_*
        html = "<pre style='font: 12px/1.4 monospace'>" + (
            report.replace("&", "&amp;").replace("<", "&lt;")) + "</pre>"
        ok, msg = send_email(a.email, f"Dog windows OOS tracker — {today}", html, report)
        print(f"[email] {'sent' if ok else 'FAILED'}: {msg}")
        if ok and a.once_per_date:
            open(MARKER_F, "w").write(today)

if __name__ == "__main__":
    main()
