"""Archive a day's captures into a self-describing, reproducible bundle.

    python3 archive_day.py --day 26AUG24 [--gzip]

WHAT IS KEPT AND WHY. The raw tape is the only irreplaceable artefact: a live match
cannot be re-run, and every decoder fix made today (advantage=50, first-ball anchor,
settled boundary prices, the both-sets-ongoing filter, the 10-point deciding-set
tiebreak) was applied RETROACTIVELY to tapes already on disk. Derived files —
boundary logs, rebuilds, comparisons — are regenerable from the tape and are archived
only as a convenience snapshot.

A manifest records, per match: the settlement, the tracked side, boundary count,
static-vs-rolling errors, the pre-match screen, and the code version, so a bundle can
be interpreted months later without this conversation.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt
from implied_model import ImpliedModel
from tennis_model import G
from rebuild_log import boundaries_from_tape

HERE = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(HERE, "tapes")


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def settlement(feed, ev, me_suffix):
    """Official Kalshi result for the tracked side: 1, 0, or None if unsettled."""
    body = feed.get("/markets", {"event_ticker": ev, "limit": 10}) or {}
    for m in body.get("markets") or []:
        if (m.get("ticker") or "").endswith("-" + me_suffix):
            r = (m.get("result") or "").strip().lower()
            return 1 if r == "yes" else (0 if r == "no" else None)
    return None


def summarise(tape):
    log = tape.replace(".jsonl", ".log.json")
    meta = {}
    if os.path.exists(log):
        meta = (json.load(open(log)).get("meta") or {})
    obs, ids = boundaries_from_tape(tape)
    rows = [json.loads(l) for l in open(tape)]
    out = dict(event=meta.get("event"), title=meta.get("title"),
               tournament=meta.get("tournament"), tour=meta.get("tour"),
               me_ticker=meta.get("me_ticker"), polls=len(rows), boundaries=len(obs))
    if len(obs) >= 4:
        d = json.load(open(log)) if os.path.exists(log) else {}
        bo = d.get("best_of", 3)
        pr = d.get("split_prior", 0.20)
        tb = d.get("final_set_tb") or 7

        def build(n):
            m = ImpliedModel(best_of=bo, first_server="me", split_prior=pr,
                             warn_split=False, final_set_tb=tb)
            for o in obs[:n]:
                m.observe(o["price"], **o["state"])
            return m

        st = build(2)
        se, re_, holds = [], [], []
        for i, o in enumerate(obs):
            run = build(max(i, 1))
            holds.append(G(run.p))
            if i < 2:
                continue
            se.append((o["price"] - st.price(**o["state"])[0]) * 100)
            re_.append((o["price"] - run.price(**o["state"])[0]) * 100)
        rms = lambda e: (sum(x * x for x in e) / len(e)) ** 0.5 if e else None
        out.update(static_p=round(st.p, 4), static_q=round(st.q, 4),
                   static_hold=round(G(st.p), 4),
                   static_rms=round(rms(se), 3), rolling_rms=round(rms(re_), 3),
                   n_oos=len(se),
                   rolling_hold_min=round(min(holds), 4),
                   rolling_hold_max=round(max(holds), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--gzip", action="store_true", help="gzip tapes (they compress ~10x)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dest = a.out or os.path.join(HERE, "archive", a.day)
    os.makedirs(dest, exist_ok=True)
    feed = kt.Feed(rps=3.0, verbose=False)

    tapes = sorted(t for t in os.listdir(TAPES)
                   if t.endswith(".jsonl") and a.day in t and "pre_restart" not in t)
    manifest = {"day": a.day, "matches": [], "code": {}}
    try:
        manifest["code"]["git"] = subprocess.run(
            ["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "uncommitted"
    except Exception:
        manifest["code"]["git"] = "unknown"

    print(f"archiving {len(tapes)} tapes -> {dest}")
    for t in tapes:
        src = os.path.join(TAPES, t)
        s = summarise(src)
        ev, me = s.get("event"), (s.get("me_ticker") or "").rsplit("-", 1)[-1]
        s["settlement"] = settlement(feed, ev, me) if ev and me else None
        s["tape_sha256_16"] = sha(src)
        s["tape_bytes"] = os.path.getsize(src)
        if a.gzip:
            with open(src, "rb") as fi, gzip.open(os.path.join(dest, t + ".gz"), "wb") as fo:
                shutil.copyfileobj(fi, fo)
            s["stored"] = t + ".gz"
        else:
            shutil.copy2(src, os.path.join(dest, t))
            s["stored"] = t
        for side in (".log.json",):
            p = src.replace(".jsonl", side)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
        manifest["matches"].append(s)
        print(f"  {t[:46]:46} {s['polls']:5d} polls {s['boundaries']:3d} bnd  "
              f"settle={s['settlement']}  "
              f"static={s.get('static_rms')} rolling={s.get('rolling_rms')}")

    for extra in ("pregame_notes.json",):
        p = os.path.join(TAPES, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dest, extra))
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(os.path.getsize(os.path.join(dest, x)) for x in os.listdir(dest))
    print(f"\nmanifest.json written; bundle {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
