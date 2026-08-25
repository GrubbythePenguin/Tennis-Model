"""Build and send the tennis in-play model day report.

    python3 email_report.py --to you@example.com [--dry-run]

Reads the REBUILT boundary logs (tapes/rpt_*.json), so every fix made during the
day — advantage decoding, first-ball anchor, settled boundary prices, the
mid-transition state filter, the 10-point deciding-set tiebreak — is applied.
"""
import argparse
import html as _html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from implied_model import ImpliedModel
from tennis_model import G

try:
    from dotenv import load_dotenv
    load_dotenv("/Users/bradleyguan/Documents/Coding/kalshi_nhl_tracker/.env")
except Exception:
    pass

MATCHES = [
    ("Poljicak vs Schoenhaus", "ATP Challenger Augsburg", "rpt_KXATPCHALLENGERMATCH-26AUG24POLSCH",
     "Schoenhaus won 6-4, 6-3", "not measured (attached at 1-1)"),
    ("Cecchinato vs Broady", "US Open Qualifying", "rpt_KXATPMATCH-26AUG24CECBRO",
     "Cecchinato won 7-6, 6-3", "0.0c over 11 min - OK"),
    ("Koevermans vs Monnet", "US Open Qualifying (WTA)", "rpt_KXWTAMATCH-26AUG24KOEMON",
     "Koevermans won 6-4, 7-6", "6.0c over 10 min - UNUSABLE"),
    ("Guerrieri vs Holmgren", "US Open Qualifying", "rpt_KXATPMATCH-26AUG24GUEHOL",
     "Guerrieri won 6-3, 6-7, 6-2", "~1c (operator-observed)"),
    ("Wendelken vs Travaglia", "US Open Qualifying", "rpt_KXATPMATCH-26AUG24WENTRA",
     "in progress at report time", "insufficient (attached 1 min pre-start)"),
    ("Wong vs Moller", "US Open Qualifying", "rpt_KXATPMATCH-26AUG24WONMOL",
     "in progress at report time", "0.0c over 90 min - OK"),
]


def analyse(path, best_of=3, tb=10):
    d = json.load(open(path))
    obs, prior = d["obs"], d["split_prior"]

    def build(n):
        m = ImpliedModel(best_of=d["best_of"], first_server="me", split_prior=prior,
                         warn_split=False, final_set_tb=tb)
        for o in obs[:n]:
            m.observe(o["price"], **o["state"])
        return m

    static = build(2)
    rows, holds = [], []
    for i, o in enumerate(obs):
        s = o["state"]
        sp, _ = static.price(**s)
        run = build(max(i, 1))
        rp = None if i == 0 else run.price(**s)[0]
        holds.append(G(run.p))
        rows.append(dict(
            state=f"{s['sets_me']}-{s['sets_opp']} {s['games_me']}-{s['games_opp']}",
            srv=s.get("server", "?"), mkt=o["price"], st=sp, rn=rp,
            se=(o["price"] - sp) * 100,
            re=None if rp is None else (o["price"] - rp) * 100,
            p=run.p, q=run.q, hold=G(run.p), oos=i >= 2))
    oos = [r for r in rows if r["oos"] and r["re"] is not None]

    def stat(k):
        e = [r[k] for r in oos]
        n = len(e)
        return (sum(e) / n, (sum(x * x for x in e) / n) ** 0.5, max(abs(x) for x in e)) if n else (0, 0, 0)

    jumps = [abs(b - a) * 100 for a, b in zip(holds, holds[1:])]
    return dict(rows=rows, n_oos=len(oos), static=stat("se"), running=stat("re"),
                static_hold=G(static.p), static_p=static.p, static_q=static.q,
                hold_min=min(holds), hold_max=max(holds),
                hold_out=sum(1 for h in holds if not (0.65 <= h <= 0.95)),
                max_jump=max(jumps) if jumps else 0.0)


def build_report():
    out = []
    for name, tour, f, result, screen in MATCHES:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tapes", f + ".json")
        if not os.path.exists(p):
            continue
        a = analyse(p)
        a.update(name=name, tour=tour, result=result, screen=screen)
        out.append(a)
    return out


def to_text(R):
    L = []
    L.append("TENNIS IN-PLAY MODEL - DAY REPORT, 24 Aug 2026")
    L.append("=" * 78)
    L.append("")
    L.append("First live test of the two-parameter model against Kalshi. Six matches")
    L.append("captured game-by-game; every price below is a vig-free midpoint read at a")
    L.append("game boundary, and both model columns are out-of-sample.")
    L.append("")
    L.append("HEADLINE: STATIC vs ROLLING (rms error, cents, out-of-sample)")
    L.append("-" * 78)
    L.append(f"{'match':26} {'n':>3} {'STATIC':>8} {'ROLLING':>8}  winner")
    for a in R:
        w = "ROLLING" if a["running"][1] < a["static"][1] else "static"
        L.append(f"{a['name']:26} {a['n_oos']:3d} {a['static'][1]:8.2f} {a['running'][1]:8.2f}  {w}")
    sw = sum(1 for a in R if a["running"][1] < a["static"][1])
    L.append("")
    L.append(f"ROLLING wins {sw} of {len(R)}.")
    L.append("")
    L.append("WHY ROLLING IS HARDER TO TRUST IN PRODUCTION")
    L.append("-" * 78)
    L.append("The static fit is two numbers, fixed once. Every later price is a")
    L.append("deterministic function of them, so a bug shows up as a CONSTANT offset and")
    L.append("the parameters can be eyeballed against reality (an ATP hold is ~80%).")
    L.append("")
    L.append("The rolling fit re-solves (p,q) by least squares over ALL prices so far,")
    L.append("after every game. Consequences:")
    L.append("  * every prediction depends on the whole history - one bad price")
    L.append("    contaminates all later predictions, and errors autocorrelate")
    L.append("  * it warm-starts from the previous (p,q), so it can persist in a bad basin")
    L.append("  * match prices pin p+q tightly but barely identify p-q, so the SPLIT")
    L.append("    wanders freely while the fit still looks like it is 'working'")
    L.append("  * nothing raises. A degenerate fit returns a number like any other.")
    L.append("")
    L.append("Measured today - how far the rolling fit actually moved:")
    L.append(f"{'match':26} {'STATIC':>7} | {'ROLLING hold':>13} {'range':>7} {'implausible':>12} {'max jump':>9}")
    for a in R:
        L.append(f"{a['name']:26} {a['static_hold']:6.1%} | "
                 f"{a['hold_min']:5.1%}-{a['hold_max']:5.1%} {(a['hold_max']-a['hold_min'])*100:6.1f}p "
                 f"{a['hold_out']:5d}/{len(a['rows']):<6} {a['max_jump']:8.1f}p")
    L.append("")
    L.append("'implausible' = boundaries where the implied hold left 65-95%, i.e. the fit")
    L.append("was describing a player who does not exist. It happened on 3 of 6 matches,")
    L.append("silently - the p>q guard passed every one, because p stayed above q.")
    L.append("")
    for a in R:
        L.append("")
        L.append("=" * 78)
        L.append(f"{a['name']}  ({a['tour']})")
        L.append(f"  result: {a['result']}")
        L.append(f"  pre-match screen: {a['screen']}")
        L.append(f"  STATIC  p={a['static_p']:.3f} q={a['static_q']:.3f} (hold {a['static_hold']:.1%})"
                 f"   mean {a['static'][0]:+.2f}c  rms {a['static'][1]:.2f}c  worst {a['static'][2]:.2f}c")
        L.append(f"  ROLLING                                        "
                 f"   mean {a['running'][0]:+.2f}c  rms {a['running'][1]:.2f}c  worst {a['running'][2]:.2f}c")
        L.append("")
        L.append(f"  {'state':>10} {'srv':>4} {'market':>7} {'static':>7} {'err':>6} "
                 f"{'rolling':>8} {'err':>6} {'roll p':>7} {'roll q':>7} {'hold':>6}")
        for r in a["rows"]:
            rn = "       -" if r['rn'] is None else f"{r['rn']:8.3f}"
            re_ = "     -" if r['re'] is None else f"{r['re']:+6.1f}"
            L.append(f"  {r['state']:>10} {r['srv']:>4} {r['mkt']:7.3f} {r['st']:7.3f} "
                     f"{r['se']:+6.1f} {rn} {re_} "
                     f"{r['p']:7.3f} {r['q']:7.3f} {r['hold']:5.1%}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    R = build_report()
    text = to_text(R)
    if a.dry_run:
        print(text)
        return
    html = "<pre style='font-family:ui-monospace,Menlo,monospace;font-size:12px'>" + \
           _html.escape(text) + "</pre>"
    sys.path.insert(0, "/Users/bradleyguan/Documents/Coding/general_level_based_quoting")
    from _eod_report import send_email
    ok, msg = send_email(a.to, "Tennis in-play model - day report, 24 Aug 2026", html, text)
    print(("SENT: " if ok else "FAILED: ") + msg)


if __name__ == "__main__":
    main()
