"""Rebuild a boundary log from a capture tape. The tape is the source of truth.

    python3 rebuild_log.py tapes/<event>.jsonl [--out tapes/<event>.fixed.json]

WHY THIS EXISTS. The live poller writes the boundary log as it goes, so any mistake
baked in at capture time — a stale anchor, a decoder bug, a price read before the
market settled — is stuck there, and re-running the match is impossible. The tape
stores the RAW payload and both midpoints for every poll, so the entire boundary
series can be re-derived afterwards with corrected logic. Restarting a live poller to
fix something has already cost this project one game boundary; rebuilding costs
nothing.

WHAT IT FIXES BY DEFAULT.

  Stale pre-match anchor. The poller logs the state it joins at, which may be a
  warm-up reading taken long before first ball. Observed 26AUG24 on Koevermans vs
  Monnet: anchor captured 17:10 at 0.395, play began 17:48 at 0.465 — the market
  drifted 7c on no tennis, and fitting the stale value drove p to 0.117 (below q,
  tripping the split guard). --anchor-at-first-ball takes the anchor from the first
  poll where the match is actually being played instead.

  Prices captured before the market repriced. The boundary price is read on the first
  poll after the score changes, but the market does not always update that fast, and
  the lag is NOT constant: measured 26AUG24, Poljicak/Schoenhaus's market LED the
  score feed by ~5s while Koevermans/Monnet's LAGGED it by ~19s. Reading early stamps
  the previous game's price onto the new score and makes the market look like it moved
  the wrong way (a player broken and going UP).

  By default the price is now read at the LAST poll before the next point is played.
  Settling on a clock instead does not work: most apparent 50s "lags" were the price
  drifting as points were played in the NEXT game, and taking that price would stamp
  information the model cannot have at this state onto the boundary — lookahead bias,
  which corrupts the test worse than reading early. The first-point cut keeps the
  reading strictly inside the gap between games.
"""
import argparse
import json
import os

import kalshi_tennis as kt

PLAYING = {"1st_set", "2nd_set", "3rd_set", "4th_set", "5th_set"}


def resolve_tracked(rows):
    """(me_id, opp_id) — decode both ways, keep the one reproducing recorded states."""
    det0 = rows[0]["details"]
    c1, c2 = det0.get("competitor1_id"), det0.get("competitor2_id")
    scores = {}
    for cand_me, cand_opp in ((c1, c2), (c2, c1)):
        hits = 0
        for r in rows:
            st = kt.model_state(r["details"], cand_me, cand_opp)
            rec = r.get("state") or {}
            if st and (st["games_me"], st["games_opp"], st["sets_me"], st["sets_opp"]) == \
                      (rec.get("games_me"), rec.get("games_opp"),
                       rec.get("sets_me"), rec.get("sets_opp")):
                hits += 1
        scores[(cand_me, cand_opp)] = hits
    (me, opp), best = max(scores.items(), key=lambda kv: kv[1])
    return me, opp, best, min(scores.values())


def boundaries_from_tape(tape, settle=-1):
    """Boundary observations derived from a tape, anchored at first ball.

    Shared by the CLI and the live watcher so both report identical numbers.
    settle=-1 reads the last price before the next point is played (see main()).
    """
    rows = [json.loads(l) for l in open(tape)]
    if not rows:
        return [], None
    me, opp, _, _ = resolve_tracked(rows)
    play = [r for r in rows
            if (r["details"].get("match_status") in PLAYING
                or (r["details"].get("status") == "live"
                    and r["details"].get("match_status") not in ("match_about_to_start", None)))
            and r.get("vig_free") is not None
            and kt.state_is_consistent(r["details"]) is None]
    obs, last_key = [], None
    for i, r in enumerate(play):
        st = kt.model_state(r["details"], me, opp)
        if st is None:
            continue
        key = kt.boundary_key(st)
        if key == last_key:
            continue
        pick = r
        if settle != 0 and obs:                 # never settle the anchor
            for r2 in play[i:]:
                s2 = kt.model_state(r2["details"], me, opp)
                if not s2 or kt.boundary_key(s2) != key:
                    break
                if s2["points_me"] or s2["points_opp"]:
                    break                       # a point was played — stop, no lookahead
                if settle > 0 and r2["ts"] - r["ts"] > settle:
                    break
                if r2.get("vig_free") is not None:
                    pick = r2
        s = kt.model_state(pick["details"], me, opp)
        state = {k: v for k, v in s.items()
                 if k not in ("points_me", "points_opp") and not k.startswith("_")}
        obs.append({"state": state, "price": round(pick["vig_free"], 4),
                    "label": f"rebuilt t={int(pick['ts'])}"})
        last_key = key
    return obs, (me, opp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tape")
    ap.add_argument("--out")
    ap.add_argument("--anchor-at-first-ball", action="store_true", default=True)
    ap.add_argument("--keep-join-anchor", dest="anchor_at_first_ball",
                    action="store_false", help="keep the poller's original anchor")
    ap.add_argument("--settle", type=float, default=-1,
                    help="-1 (default) = read the last price before the next point is "
                         "played; 0 = read immediately at the score change; N>0 = same "
                         "as -1 but capped at N seconds")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.tape)]
    if not rows:
        raise SystemExit("empty tape")
    src_log = a.tape.replace(".jsonl", ".log.json")
    base = json.load(open(src_log)) if os.path.exists(src_log) else {}

    # Which competitor was tracked as "me"? Comparing set scores fails — they are tied
    # 0-0 for most of a match, so the test silently resolves to competitor1 whoever was
    # actually tracked. Instead, decode every poll BOTH ways and keep the assignment
    # that reproduces the state the capture recorded. That is exact, not a heuristic.
    det0 = rows[0]["details"]
    c1, c2 = det0.get("competitor1_id"), det0.get("competitor2_id")
    scores = {}
    for cand_me, cand_opp in ((c1, c2), (c2, c1)):
        hits = 0
        for r in rows:
            st = kt.model_state(r["details"], cand_me, cand_opp)
            rec = r.get("state") or {}
            if st and (st["games_me"], st["games_opp"], st["sets_me"], st["sets_opp"]) == \
                      (rec.get("games_me"), rec.get("games_opp"),
                       rec.get("sets_me"), rec.get("sets_opp")):
                hits += 1
        scores[(cand_me, cand_opp)] = hits
    (me, opp), best = max(scores.items(), key=lambda kv: kv[1])
    other = min(scores.values())
    if best == other:
        raise SystemExit("cannot tell which competitor was tracked — both decodings "
                         "match the recorded states equally; rerun with an explicit --me")
    print(f"  tracked side resolved: {best}/{len(rows)} polls match "
          f"(other assignment {other})")

    # only polls where the match is genuinely being played
    play = [r for r in rows
            if (r["details"].get("match_status") in PLAYING
                or (r["details"].get("status") == "live"
                    and r["details"].get("match_status") not in ("match_about_to_start", None)))
            and r.get("vig_free") is not None
            and kt.state_is_consistent(r["details"]) is None]
    if not play:
        raise SystemExit("no in-play polls with a price on this tape")

    obs, last_key, dropped = [], None, 0
    for i, r in enumerate(play):
        st = kt.model_state(r["details"], me, opp)
        if st is None:
            continue
        key = kt.boundary_key(st)
        if key == last_key:
            continue
        # settle: prefer a poll >= `settle` seconds after the change, same score
        pick = r
        # The ANCHOR is never settled: there is no preceding game for the market to
        # reprice, so waiting only walks the reading into the middle of game 1 and
        # stamps a mid-game price onto a 0-0 state.
        #
        # SETTLING WINDOW = while the next game is still at 0-0 points. Waiting for
        # the price to "stop moving" on a clock does NOT work: measured 26AUG24, most
        # apparent 50s "lags" were the price drifting as points were played in the
        # NEXT game. Taking that price would stamp information the model cannot have
        # at this state onto the boundary — lookahead, which corrupts the test far
        # worse than reading a few seconds early. Stopping at the first point keeps
        # the reading strictly inside the gap between games.
        if a.settle != 0 and obs:
            for r2 in play[i:]:
                s2 = kt.model_state(r2["details"], me, opp)
                if not s2 or kt.boundary_key(s2) != key:
                    break                                   # next game started
                if s2["points_me"] or s2["points_opp"]:
                    break                                   # a point was played
                if a.settle > 0 and r2["ts"] - r["ts"] > a.settle:
                    break                                   # optional extra cap
                if r2.get("vig_free") is not None:
                    pick = r2
        if pick.get("vig_free") is None:
            dropped += 1
            last_key = key
            continue
        s = kt.model_state(pick["details"], me, opp)
        state = {k: v for k, v in s.items()
                 if k not in ("points_me", "points_opp") and not k.startswith("_")}
        obs.append({"state": state, "price": round(pick["vig_free"], 4),
                    "label": f"rebuilt t={int(pick['ts'])}"})
        last_key = key

    if a.anchor_at_first_ball and obs:
        obs[0]["label"] += " (anchor at first ball)"

    out = dict(base)
    out["obs"] = obs
    out.setdefault("best_of", 3)
    out.setdefault("split_prior", 0.14)
    out.setdefault("split_prior_sd", 0.30)
    out.setdefault("first_server", "me")
    out.setdefault("set_servers", {})

    old = len((base.get("obs") or []))
    print(f"{a.tape}")
    print(f"  tape polls {len(rows)}  in-play with price {len(play)}")
    print(f"  boundaries: {len(obs)} rebuilt  (poller had {old})"
          f"{f'  [{dropped} dropped: no price]' if dropped else ''}")
    if obs:
        f = obs[0]["state"]
        print(f"  anchor: {f['sets_me']}-{f['sets_opp']} g{f['games_me']}-{f['games_opp']} "
              f"srv={f.get('server','?')} @ {obs[0]['price']}")
        l = obs[-1]["state"]
        print(f"  last:   {l['sets_me']}-{l['sets_opp']} g{l['games_me']}-{l['games_opp']} "
              f"@ {obs[-1]['price']}")

    dest = a.out or a.tape.replace(".jsonl", ".rebuilt.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
