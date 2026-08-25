"""End-to-end proof that a match flowing through the feed actually tests the model.

Drives the REAL poll_tennis.cmd_watch loop against a scripted feed that replays a
best-of-3 match, then checks that:
  1. one observation is logged per completed game, at the price seen the instant the
     score changed (not a mid-game sample);
  2. the fitted (p, q) tracks and stays inside the physical range;
  3. the written log replays through live.py unchanged.

Prices are generated from a known (p, q) so the fit has a truth to recover. Run with
no network access; nothing here touches Kalshi.
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt
import poll_tennis as pt
from market_implied import model_prob

EV = "KXATPCHALLENGERMATCH-26AUG24POLSCH"
C1, C2 = "id-poljicak", "id-schoenhaus"
TRUTH = (0.62, 0.38)          # what the simulated market is "running"
fails = []


def chk(name, got, exp):
    ok = got == exp
    print(f"{'OK ' if ok else 'FAIL'} {name:52s} got={got!r} exp={exp!r}")
    if not ok:
        fails.append(name)


# ---- the scripted match: (sets1, sets2, [set games c1], [set games c2], server) ----
# Poljicak serves game 1; server alternates. Set 1 goes to 6-4, then 2 games of set 2.
def build_script():
    """States at each game boundary of a set that finishes 6-4.

    P = Poljicak wins the game, S = Schoenhaus. 10 games -> 6-4 exactly.
    """
    games = "PSPSPSPSPP"           # 6 P, 4 S -> 6-4
    assert games.count("P") == 6 and games.count("S") == 4
    steps, g1, g2 = [], 0, 0
    for i, w in enumerate(games):
        steps.append((0, 0, g1, g2, C1 if i % 2 == 0 else C2))   # before game i
        g1 += (w == "P")
        g2 += (w == "S")
    steps.append((0, 0, g1, g2, C1 if len(games) % 2 == 0 else C2))
    return steps


SCRIPT = build_script()


def details(s1, s2, gm, go, server, status="live"):
    return {
        "competitor1_id": C1, "competitor2_id": C2, "competitor1_is_home": True,
        "competitor1_overall_score": s1, "competitor2_overall_score": s2,
        "competitor1_round_scores": [{"outcome": "ongoing", "score": gm}],
        "competitor2_round_scores": [{"outcome": "ongoing", "score": go}],
        # POINT score, not games — 0-0 because every scripted state is the instant a
        # game completed. Games live in round_scores above.
        "competitor1_current_round_score": 0, "competitor2_current_round_score": 0,
        "server": server, "advantage": "", "status": status,
        "match_status": "live", "round_winners": [], "winner": "",
    }


class FakeFeed:
    """Same surface as kalshi_tennis.Feed, replaying SCRIPT one step per live_data call."""

    def __init__(self, *a, **k):
        self.i = 0
        self.n_get = 0
        self.n_429 = 0
        self.mid = "fake-milestone"
        self.prices = []
        self.cur = SCRIPT[0]        # _bind() calls markets() before any live_data()

    def get(self, path, params=None):
        """No exact-score market in the fixture, so best_of falls back to --best-of.
        Returning None (not {}) is what the real Feed does on a miss."""
        return None

    def milestone(self, ev):
        return {"id": self.mid, "title": "Poljicak vs Schoenhaus",
                "details": {"first_competitor_id": C1, "second_competitor_id": C2,
                            "tour": "ATP Challenger", "gender": "men",
                            "tournament_name": "ATP Challenger Augsburg",
                            "round": "Round Of 32"}}

    def live_data(self, ids):
        """Serve the next scripted state, and REMEMBER it.

        markets() is called after this within the same cycle, so it must quote the
        state just served — not the next one. Advancing a cursor that both methods
        read independently mis-pairs every price with the following game's state,
        which looks like a model that cannot fit.
        """
        self.n_get += 1
        self.cur = SCRIPT[min(self.i, len(SCRIPT) - 1)]
        self.i += 1
        s1, s2, gm, go, srv = self.cur
        return {self.mid: details(s1, s2, gm, go, srv)}

    def markets(self, ev):
        """Quote both sides around the true model price for the state just served."""
        self.n_get += 1
        s1, s2, gm, go, srv = self.cur
        st = {"sets_me": s1, "sets_opp": s2, "games_me": gm, "games_opp": go,
              "i_serve": srv == C1, "points_me": 0, "points_opp": 0}
        p = model_prob(*TRUTH, st, 3)
        p = round(p * 200) / 200.0                      # half-cent market resolution
        self.prices.append(p)
        return [
            {"ticker": f"{EV}-POL", "yes_sub_title": "Mili Poljicak",
             "yes_bid_dollars": f"{p - 0.005:.4f}", "yes_ask_dollars": f"{p + 0.005:.4f}"},
            {"ticker": f"{EV}-SCH", "yes_sub_title": "Max Schoenhaus",
             "yes_bid_dollars": f"{1 - p - 0.005:.4f}", "yes_ask_dollars": f"{1 - p + 0.005:.4f}"},
        ]


def main():
    tmp = tempfile.mkdtemp()
    pt.TAPES = tmp
    real_feed, real_sleep = kt.Feed, pt.time.sleep
    kt.Feed = FakeFeed
    pt.time.sleep = lambda s: None                       # no real waiting
    try:
        args = types.SimpleNamespace(
            event=EV, me="POL", best_of=3, split_prior=None, rps=99,
            interval=0, idle_interval=0, log_first=True, seed=None, final_set_tb=None, pregame_interval=0,
            max_cycles=len(SCRIPT))
        pt.cmd_watch(args)
    finally:
        kt.Feed, pt.time.sleep = real_feed, real_sleep

    log_p = os.path.join(tmp, f"{EV}.log.json")
    tape_p = os.path.join(tmp, f"{EV}.jsonl")
    log = json.load(open(log_p))
    obs = log["obs"]

    print("\n" + "=" * 70)
    chk("log is best_of 3", log["best_of"], 3)
    chk("men's split prior", log["split_prior"], 0.28)
    chk("one obs per distinct game score", len(obs), len(SCRIPT))
    chk("tape has one line per poll", sum(1 for _ in open(tape_p)), len(SCRIPT))
    chk("every obs carries an explicit server",
        all("server" in o["state"] for o in obs), True)
    chk("no obs carries a point score (not published)",
        all("points_me" not in o["state"] for o in obs), True)

    # the game score must advance by exactly one game per observation
    seq = [(o["state"]["games_me"], o["state"]["games_opp"]) for o in obs]
    steps_ok = all(sum(b) - sum(a) == 1 for a, b in zip(seq, seq[1:]))
    chk("each obs is exactly one game later", steps_ok, True)
    chk("final game score", seq[-1], (6, 4))

    # server must alternate across observations
    srv = [o["state"]["server"] for o in obs]
    chk("server alternates every game", all(a != b for a, b in zip(srv, srv[1:])), True)

    # refit from the written log and check we recovered the truth
    from implied_model import ImpliedModel
    m = ImpliedModel(best_of=log["best_of"], first_server="me",
                     split_prior=log["split_prior"], split_prior_sd=5.0)
    for o in obs:
        m.observe(o["price"], **o["state"])
    print(f"\nrefit from log: p={m.p:.3f} q={m.q:.3f}   (truth p={TRUTH[0]} q={TRUTH[1]})")
    chk("recovers p within 0.03", abs(m.p - TRUTH[0]) < 0.03, True)
    chk("recovers q within 0.03", abs(m.q - TRUTH[1]) < 0.03, True)
    chk("fit stays physical (p > q)", m.split_violation, None)

    # the log must be replayable by live.py
    import shutil
    shutil.copy(log_p, os.path.join(tmp, "live_log.json"))
    for f in ("live.py", "implied_model.py", "market_implied.py", "tennis_model.py"):
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), tmp)
    import subprocess
    r = subprocess.run([sys.executable, "live.py", "report"], cwd=tmp,
                       capture_output=True, text=True, timeout=600)
    chk("live.py report exits 0", r.returncode, 0)
    chk("live.py report printed a fit", "Fit on" in r.stdout, True)
    print("\n--- live.py report (from the poller's own log) ---")
    print("\n".join(r.stdout.splitlines()[:len(obs) + 2]))
    if r.returncode != 0:
        print(r.stderr[-1500:])

    print("\n" + "=" * 70)
    print(f"{len(fails)} FAILURES" if fails else "LIVE PATH VERIFIED END TO END")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
