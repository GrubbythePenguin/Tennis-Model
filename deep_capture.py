"""Pick ONE live match and capture every point at maximum resolution.

    python3 deep_capture.py --pick            # choose and print, launch nothing
    python3 deep_capture.py --launch --interval 1

WHY ONE MATCH. Spread across a fleet, the GET budget forces ~3-10s polling, which is
coarser than a point (a point cycle is ~25-40s, the ball in play 5-10s). One match at a
time can be polled every second for 2 GET/s total, which is less load than five matches
at 3s and gives resolution a fleet can never reach.

WHAT THE MID HIDES. "The market always moves on a point" is right, but the DERIVED mid
often does not show it: the book is 1c-granular, so a bid moving 0.62 -> 0.63 with the
ask unchanged shifts the book and moves the mid only half a cent - and a 0.63/0.64 book
going to 0.64/0.65 moves the mid a full cent while a one-sided move shows nothing.
Measured on a 4.5s tape, the MEDIAN market move across an observed point change was
0.00c, which reads as "no move" but is really "below the mid's resolution".

So detection here keys on the RAW top of book, both sides, which poll_tennis now
records on every row (bid_me/ask_me/bid_opp/ask_opp). A point that moves either side of
either book is visible even when the vig-free mid is unchanged.

SELECTION. Prefers a match that is live, quoting both sides, has a known point score,
and is EARLY - fewest games played - so the capture covers as much of the match as
possible. Ties break toward the tighter book.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt

HERE = os.path.dirname(os.path.abspath(__file__))


def candidates(feed):
    """Live matches with a quoting book on both sides and a visible point score."""
    out = []
    evs = feed.matches()
    mids = {}
    for e in evs:
        mil = feed.milestone(e["event_ticker"])
        if not mil:
            continue
        mids[mil.get("id")] = (e["event_ticker"], mil)
    det = feed.live_data(list(mids))
    for mid, d in det.items():
        if not kt.is_live(d):
            continue
        if (d.get("match_status") or "") in ("", None, "match_about_to_start"):
            continue
        ev, mil = mids[mid]
        mkts = feed.markets(ev)
        mm = {m.get("ticker"): m for m in mkts if m.get("ticker")}
        if len(mm) != 2:
            continue
        ticks = sorted(mm)
        books = {t: kt.top_of_book(mm[t]) for t in ticks}
        if any(b is None or a is None for b, a in books.values()):
            continue
        prices = {t: kt.mid(mm[t]) for t in ticks}
        if any(v is None for v in prices.values()):
            continue
        under = min(ticks, key=lambda t: prices[t])
        # How far along is it? Fewer games played = more match left to capture.
        # Games MUST come from model_state, which reads round_scores.
        # competitor*_current_round_score is the POINT score in tennis notation
        # ("15", "30", "40"), so summing it gives nonsense like 45 - that bug picked
        # a match 90 "games" in on the first dry run.
        c1 = d.get("competitor1_id") or ""
        c2 = d.get("competitor2_id") or ""
        stt = kt.model_state(d, c1, c2) or {}
        games = int(stt.get("games_me") or 0) + int(stt.get("games_opp") or 0)
        sets = int(stt.get("sets_me") or 0) + int(stt.get("sets_opp") or 0)
        if not stt.get("_points_known"):
            continue          # cannot see points -> useless for a point-level capture
        spread = max(a - b for b, a in books.values())
        out.append(dict(event=ev, me=under.rsplit("-", 1)[-1], under_px=prices[under],
                        played=sets * 10 + games, spread_c=round(spread * 100, 1),
                        title=(mil.get("title") or ""),
                        tour=((mil.get("details") or {}).get("tournament_name") or "?")))
    # earliest in the match first, then tightest book
    out.sort(key=lambda r: (r["played"], r["spread_c"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between polls. 1.0 = 2 GET/s for the single match")
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--max-cycles", type=int, default=15000,
                    help="15000 at 1s covers a >4h match")
    ap.add_argument("--cache", type=int, default=25_000)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--event", help="force a specific event instead of picking")
    ap.add_argument("--me", help="market suffix, required with --event")
    a = ap.parse_args()

    feed = kt.Feed(rps=3.0, verbose=False)
    if a.event:
        pick = dict(event=a.event, me=a.me, title="(forced)", tour="", under_px=0,
                    played=0, spread_c=0)
        cands = [pick]
    else:
        cands = candidates(feed)
        if not cands:
            print("no live match with a two-sided book right now")
            print(f"[{feed.n_get} GETs, {feed.n_429} rate-limited]")
            return 1
        pick = cands[0]

    print(f"{len(cands)} candidate(s); picking the earliest with the tightest book:\n")
    for c in cands[:8]:
        mark = "->" if c is pick else "  "
        print(f"  {mark} {c['event']:<44} me={c['me']:<8} px={c['under_px']:.3f} "
              f"played={c['played']:>3} spread={c['spread_c']}c  {c['tour'][:22]}")
    print(f"\n[{feed.n_get} GETs, {feed.n_429} rate-limited]")
    if not a.launch:
        print("\n--launch to start the capture")
        return 0

    log = os.path.join(HERE, "tapes", f"deep_{pick['event'].rsplit('-',1)[-1].lower()}.log")
    # --rps is a TOP-LEVEL flag on poll_tennis.py, before the subcommand. Passing it
    # after "watch" makes argparse reject it and the capture never starts.
    cmd = [sys.executable, "-u", os.path.join(HERE, "poll_tennis.py"),
           "--rps", str(a.rps), "watch",
           pick["event"], "--me", pick["me"], "--interval", str(a.interval),
           "--pregame-interval", "30", "--max-cycles", str(a.max_cycles)]
    env = dict(os.environ, TENNIS_CACHE=str(a.cache))
    with open(log, "a") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE,
                         start_new_session=True, env=env)
    print(f"\nlaunched {pick['event']} --me {pick['me']} at {a.interval}s -> {log}")
    print(f"  ~{2/a.interval:.1f} GET/s for this one match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
