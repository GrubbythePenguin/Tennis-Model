"""Emit one line per completed game from a running poll_tennis capture.

Reads only local files (the poller's own log); makes no API calls. Exits when the
poller process is gone, so the watch ends with the match.
"""
import json
import os
import subprocess
import sys
import time

LOG = sys.argv[1] if len(sys.argv) > 1 else "tapes/KXATPCHALLENGERMATCH-26AUG24POLSCH.log.json"
# Match only THIS event's poller, so a watcher ends with its own match rather than
# hanging on until every capture on the box has finished.
PAT = sys.argv[2] if len(sys.argv) > 2 else "poll_tennis.py watch"
TAG = sys.argv[3] if len(sys.argv) > 3 else "GAME"
LBL = ['0', '15', '30', '40', 'AD']


def poller_alive():
    r = subprocess.run(["pgrep", "-f", PAT], capture_output=True)
    return r.returncode == 0


def fit(obs, best_of, prior):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from implied_model import ImpliedModel
    m = ImpliedModel(best_of=best_of, first_server='me', split_prior=prior, warn_split=False)
    pred = None
    for i, o in enumerate(obs):
        if i == len(obs) - 1 and i >= 1:
            try:
                pred, _ = m.price(**o['state'])
            except Exception:
                pred = None
        m.observe(o['price'], **o['state'])
    return m, pred


# TAPE mode: derive boundaries from the raw tape instead of trusting the poller's
# own log. Needed whenever the live log carries a defect the tape can correct — the
# stale pre-match anchor on Koevermans/Monnet being the case that forced this.
TAPE = LOG.replace('.log.json', '.jsonl') if '--tape' in sys.argv else None

seen = 0
while True:
    try:
        d = json.load(open(LOG))
        obs = d.get('obs') or []
        if TAPE and os.path.exists(TAPE):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rebuild_log import boundaries_from_tape
            rebuilt, _ = boundaries_from_tape(TAPE)
            if rebuilt:
                obs = rebuilt
    except Exception:
        obs = []
    if len(obs) > seen:
        seen = len(obs)
        o = obs[-1]
        s = o['state']
        try:
            m, pred = fit(obs, d['best_of'], d['split_prior'])
            err = "     -" if pred is None else f"{(o['price'] - pred) * 100:+5.1f}c"
            pr = "    -" if pred is None else f"{pred:.3f}"
            flag = "  !p<=q" if m.split_violation else ""
            print(f"{TAG} {seen:2d}  sets {s['sets_me']}-{s['sets_opp']}  "
                  f"games {s['games_me']}-{s['games_opp']}  srv={s.get('server','?'):3s}  "
                  f"mkt {o['price']:.3f}  model {pr}  err {err}  "
                  f"| fit p={m.p:.3f} q={m.q:.3f}{flag}", flush=True)
        except Exception as e:
            print(f"{TAG} {seen}: {s} mkt={o['price']} (fit error {e!r})", flush=True)
    if not poller_alive():
        print(f"{TAG} POLLER EXITED — {seen} boundary observations captured", flush=True)
        break
    time.sleep(15)
