"""Offline verification of the Kalshi tennis decode path (no API calls).

Simulates live_data payloads in the exact shape Kalshi returned, since no live
match is reachable until the US Open starts.
"""
import sys, json, os
sys.path.insert(0, '/Users/bradleyguan/Documents/Coding/Tennis-Model')
import kalshi_tennis as kt

C1, C2 = "aaa-me", "bbb-opp"
fails = []
def chk(name, got, exp):
    ok = got == exp
    print(f"{'OK ' if ok else 'FAIL'} {name:48s} got={got!r} exp={exp!r}")
    if not ok: fails.append(name)

def det(s1, s2, r1, r2, server=None, status="live", p1=0, p2=0, adv=""):
    """Payload in Kalshi's real shape.

    round_scores  = GAMES per set;  current_round_score = POINT score (0/15/30/40,
    or raw counts in a tiebreak); `advantage` names the competitor holding the ad.
    Confirmed live 26AUG24: at deuce both current_round_score read 40 while
    round_scores was still {ongoing, 0}.
    """
    return {
        "competitor1_id": C1, "competitor2_id": C2, "competitor1_is_home": True,
        "competitor1_overall_score": s1, "competitor2_overall_score": s2,
        "competitor1_round_scores": r1, "competitor2_round_scores": r2,
        "competitor1_current_round_score": p1, "competitor2_current_round_score": p2,
        "server": server or "", "advantage": adv, "status": status,
        "round_winners": [], "winner": "",
    }

W = lambda n: {"outcome": "winner", "score": n}
L = lambda n: {"outcome": "loser", "score": n}
O = lambda n: {"outcome": "ongoing", "score": n}

print("=== 1. mid-set state, set 1, 3-2, me serving ===")
d = det(0, 0, [O(3)], [O(2)], server=C1)
st = kt.model_state(d, C1, C2)
chk("games_me", st["games_me"], 3)
chk("games_opp", st["games_opp"], 2)
chk("sets_me", st["sets_me"], 0)
chk("server", st["server"], "me")

print("\n=== 2. set 2 in progress after losing set 1 3-6 ===")
d = det(0, 1, [L(3), O(2)], [W(6), O(4)], server=C2)
st = kt.model_state(d, C1, C2)
chk("sets_me", st["sets_me"], 0)
chk("sets_opp", st["sets_opp"], 1)
chk("games_me (current set only)", st["games_me"], 2)
chk("games_opp (current set only)", st["games_opp"], 4)
chk("server", st["server"], "opp")

print("\n=== 3. opponent's perspective is the mirror ===")
st_opp = kt.model_state(d, C2, C1)
chk("mirror sets", (st_opp["sets_me"], st_opp["sets_opp"]), (1, 0))
chk("mirror games", (st_opp["games_me"], st_opp["games_opp"]), (4, 2))
chk("mirror server", st_opp["server"], "me")

print("\n=== 4. between sets / no ongoing round ===")
d2 = det(1, 1, [W(6), L(4)], [L(4), W(6)], server=C1)
st2 = kt.model_state(d2, C1, C2)
chk("games reset between sets", (st2["games_me"], st2["games_opp"]), (0, 0))

print("\n=== 5. unknown competitor ids -> None, not a wrong state ===")
chk("bad ids", kt.model_state(d, "zzz", C2), None)

print("\n=== 6. unknown/blank server -> key absent (not a guess) ===")
d3 = det(0, 0, [O(1)], [O(0)], server="")
chk("no server key", "server" in kt.model_state(d3, C1, C2), False)

print("\n=== 7. is_live gating ===")
for s, exp in [("live", True), ("not_started", False), ("closed", False),
               ("ended", False), ("cancelled", False), ("", False)]:
    chk(f"is_live({s!r})", kt.is_live({"status": s}), exp)

print("\n=== 8. vig_free strips the overround ===")
chk("normalise 0.505/0.490", round(kt.vig_free(0.505, 0.490), 6), round(0.505/0.995, 6))
chk("one-sided falls back", kt.vig_free(0.42, None), 0.42)
chk("no quote", kt.vig_free(None, 0.5), None)

print("\n=== 9. empty-book sentinels are not quotes ===")
chk("0.00/1.00 -> None mid", kt.mid({"yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}), None)
chk("real book", kt.mid({"yes_bid_dollars": "0.5000", "yes_ask_dollars": "0.5100"}), 0.505)
chk("yes from no side", kt.mid({"no_bid_dollars": "0.4900", "no_ask_dollars": "0.5000"}), 0.505)

print("\n=== 10. best_of refuses to guess ===")
for name, d_, ov, expect_raise in [
    ("US Open men best_of=3 (no override)", {"tour":"ATP","gender":"men","best_of":"3","tournament_name":"US Open Men Singles","round":"Round Of 128"}, None, True),
    ("US Open men with --best-of 5",        {"tour":"ATP","gender":"men","best_of":"3","tournament_name":"US Open Men Singles"}, 5, False),
    ("missing best_of",                     {"tour":"ATP","gender":"men","tournament_name":"ATP Winston Salem"}, None, True),
    ("ATP 250 best_of=3",                   {"tour":"ATP","gender":"men","best_of":"3","tournament_name":"ATP Winston Salem"}, None, False),
    ("US Open WOMEN best_of=3",             {"tour":"WTA","gender":"women","best_of":"3","tournament_name":"US Open Women Singles"}, None, False),
]:
    try:
        v = kt.resolve_best_of(d_, ov)
        chk(name, ("raised" if expect_raise else v), (("raised" if expect_raise else (5 if ov==5 else 3))))
    except ValueError:
        chk(name, "raised", ("raised" if expect_raise else "no-raise"))

print("\n=== 11. split prior from gender ===")
chk("men", kt.resolve_split_prior({"gender":"men"}), 0.28)
chk("women", kt.resolve_split_prior({"gender":"women"}), 0.14)
chk("unknown -> 0.20 (warned)", kt.resolve_split_prior({}), 0.20)
chk("override", kt.resolve_split_prior({"gender":"men"}, 0.5), 0.5)

print("\n=== 12a. POINT SCORE decoding (current_round_score) ===")
for name, p1, p2, adv, exp in [
    ("0-0",            0,  0,  "",  (0, 0)),
    ("15-0",          15,  0,  "",  (1, 0)),
    ("30-15",         30, 15,  "",  (2, 1)),
    ("40-30",         40, 30,  "",  (3, 2)),
    ("deuce",         40, 40,  "",  (3, 3)),
    ("my advantage",  40, 40,  C1,  (4, 3)),
    ("their adv",     40, 40,  C2,  (3, 4)),
    # 50 IS the advantage code, carried in current_round_score itself (live 26AUG24)
    ("my ad via 50",  50, 40,  "",  (4, 3)),
    ("their ad via 50", 40, 50, "",  (3, 4)),
    ("50 + agreeing advantage field", 50, 40, C1, (4, 3)),
]:
    d = det(0, 0, [O(3)], [O(2)], server=C1, p1=p1, p2=p2, adv=adv)
    pm, po, known = kt.points_from_details(d, C1, C2)
    chk(f"points {name}", (pm, po), exp)
    chk(f"points {name} known", known, True)

print("\n  mirror: the opponent's view is the transpose")
d = det(0, 0, [O(3)], [O(2)], server=C1, p1=40, p2=30)
chk("40-30 from opp side", kt.points_from_details(d, C2, C1)[:2], (2, 3))
d = det(0, 0, [O(3)], [O(2)], server=C1, p1=40, p2=40, adv=C1)
chk("my-adv from opp side", kt.points_from_details(d, C2, C1)[:2], (3, 4))

print("\n  tiebreak: raw counts, no 15/30/40 mapping")
d = det(0, 0, [O(6)], [O(6)], server=C1, p1=5, p2=3)
chk("tiebreak 5-3", kt.points_from_details(d, C1, C2, in_tiebreak=True)[:2], (5, 3))
chk("same raw outside a TB is rejected",
    kt.points_from_details(d, C1, C2, in_tiebreak=False), (0, 0, False))

print("\n  model_state wires points in, and flags a tiebreak automatically")
st = kt.model_state(det(0, 0, [O(3)], [O(2)], server=C1, p1=40, p2=40, adv=C2), C1, C2)
chk("state points at their-ad", (st["points_me"], st["points_opp"]), (3, 4))
chk("state marks points known", st["_points_known"], True)
st = kt.model_state(det(0, 0, [O(6)], [O(6)], server=C1, p1=5, p2=3), C1, C2)
chk("6-6 decodes as tiebreak", (st["points_me"], st["points_opp"]), (5, 3))

print("\n  games come from round_scores, NEVER from current_round_score")
st = kt.model_state(det(0, 0, [O(2)], [O(1)], server=C1, p1=40, p2=15), C1, C2)
chk("games unaffected by points", (st["games_me"], st["games_opp"]), (2, 1))

print("\n=== 12. boundary detection fires once per completed game ===")
seq = [([O(0)],[O(0)],C1), ([O(0)],[O(0)],C1), ([O(1)],[O(0)],C2), ([O(1)],[O(0)],C2),
       ([O(1)],[O(1)],C1), ([O(2)],[O(1)],C2)]
keys, fired = None, 0
for r1, r2, sv in seq:
    st = kt.model_state(det(0,0,r1,r2,sv), C1, C2)
    k = kt.boundary_key(st)
    if k != keys:
        if keys is not None: fired += 1
        keys = k
chk("boundaries fired", fired, 3)

print("\n" + "="*60)
print(f"{len(fails)} FAILURES" if fails else "ALL FEED CHECKS PASSED")
for f in fails: print("  -", f)
