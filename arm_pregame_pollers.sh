#!/bin/bash
# Attach a poller to every tennis match BEFORE it starts, on a rolling basis.
#
#   MAX_POLLERS=12 ./arm_pregame_pollers.sh [lookahead_min] [cycle_sec]
#
# WHY PRE-MATCH ATTACHMENT IS NOT OPTIONAL. See tennis_recenter_model / poll_tennis:
# a poller attached mid-match has no seed price and no boundaries behind it.
#
# GET BUDGET (26AUG28 rewrite). The shared bucket is the scarce resource. Per pass:
#     6 x /events?with_nested_markets   one per series - THE discovery call
#     /milestones                        only for events never seen before (cached
#                                        on disk in tapes/milestones_cache.json)
#     nothing else: no /live_data, no /markets. Tickers, shard, prices (for the
#     tracked-side pick) and tournament all come from the events payload; the start
#     time from the cached milestone. Before this rewrite a pass with ITF on the board
#     cost ~200 GETs and 49 rate-limit hits; now it is 6 + new events.
# The poller launched here does its own one-off milestone+markets bind, then only
# /live_data every --interval. --best-of is passed so it skips the exact-market GET.
#
# Idempotent: skips any event that already has a poller. Honours the kill flag.
# Attaches NOT-YET-STARTED matches only (start > now - 5 min).
cd "$(dirname "$0")" || exit 1
LOOKAHEAD=${1:-30}
CYCLE=${2:-600}
LOG=tapes/arm_pregame.log

while true; do
    if [ -f disable_tennis_poller.flag ]; then
        echo "$(date -u '+%F %T')Z kill flag — not attaching" >> "$LOG"
        sleep "$CYCLE"; continue
    fi
    python3 - "$LOOKAHEAD" >> "$LOG" 2>&1 <<'PY'
import sys, os, json, subprocess, datetime as dt, time
sys.path.insert(0, os.getcwd())
import kalshi_tennis as kt

lookahead = float(sys.argv[1])
# 120 (was 50) 26SEP14: cap 50 hit 1,452 times (9 waiting during the ITF morning
# block) and misses matches whose 35-min attach window passes while capped. Post-GET-audit
# the per-poller REST cost is ~0 (state via live_data_mux, markets via quoter wsbook);
# the real cost is RAM: ~57MB/poller, 120 ~= 6.8GB on a 15GB box (11.7GB avail 26SEP14).
# 1.2s spawn stagger unchanged.
#
# 300 (was 160) 26SEP23, operator call: the GET budget no longer binds, so the cap
# should not either. MEASURE BEFORE TRUSTING THIS NUMBER: 120 live pollers that day
# were 6.42GB RSS (mean 55MB, p90 63MB) with 7.67GB available, so ~250 is the
# PHYSICAL ceiling and 300 is not reachable at TENNIS_CACHE=25000 (300 x 58MB =
# 17.4GB > the 15.5GB box). Past ~250 the failure mode stops being the graceful
# "cap reached — N candidates waiting" log line and becomes the OOM killer taking
# random pollers AND potentially quoter/run.py, which shares this box. The lever
# that actually buys 300 is the per-poller cache, not this number: TENNIS_CACHE is
# hardcoded to 25000 in the Popen env below and dominates RSS (tennis_model says
# 200k ~= 235MB, 25k ~= 56MB); lowering it is behaviour-preserving and only costs
# refit latency, which arrives minutes apart. See [[tennis-model-oom-kills-claude]].
MAX_POLLERS = int(os.environ.get("MAX_POLLERS", "300"))
ATTACH = os.environ.get("ATTACH", "1") == "1"        # ATTACH=0: discovery + candidate list only

# RAM GUARD (26SEP23). MAX_POLLERS is no longer the binding constraint — memory is —
# so the count alone cannot keep this box safe. Measured 26SEP23: 120 pollers at
# TENNIS_CACHE=25000 were 6.42GB RSS (mean 55MB, p90 63MB) with 7.67GB MemAvailable,
# i.e. ~250 was the physical ceiling and a 300 cap would have handed the box to the
# OOM killer instead of the graceful cap line. Two changes together fix that:
#   TENNIS_CACHE 25000 -> 10000   MEASURED 26SEP23 (scratchpad cache_rss.py), steady
#                                 state with all three lru_caches filled to maxsize:
#                                 25k = 12MB over the bare import, 10k = 5MB, 5k = 3MB.
#                                 The four fits (static/rolling/ewma2/ewma4) SHARE these
#                                 module-level caches, so the saving is ~7MB per POLLER,
#                                 not per fit: ~55MB -> ~48MB. Latency cost is ~nil, not
#                                 the ~150ms the module quotes for 200k -> 25k: one
#                                 boundary refit's working set is only ~134 set_from_games
#                                 + ~120 tiebreak entries, so 10k still holds ~70 refits
#                                 and nothing thrashes WITHIN a refit. Do not go below
#                                 ~2000 without re-measuring that working set.
#   RAM_FLOOR_MB                  hard stop on attaching while MemAvailable would drop
#                                 below this. THIS is the real cap now; raise/lower it
#                                 rather than MAX_POLLERS. quoter/run.py shares this box
#                                 and must never be the process the OOM killer picks.
# POLLER_MB is the reserve charged per poller spawned THIS pass, because a fresh poller
# takes seconds to build its cache and MemAvailable has not yet fallen when the next
# spawn decision is made — without it one pass walks straight through the floor.
POLLER_CACHE = os.environ.get("TENNIS_CACHE_POLLER", "10000")
RAM_FLOOR_MB = int(os.environ.get("RAM_FLOOR_MB", "2000"))
POLLER_MB = int(os.environ.get("POLLER_MB", "50"))   # measured ~48MB at TENNIS_CACHE=10000

def mem_available_mb():
    """MemAvailable in MB, or None if unreadable (guard then does not bind)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None
CACHE = "tapes/milestones_cache.json"
DISC = "tapes/discovered.json"
now = dt.datetime.now(dt.timezone.utc); nowts = now.timestamp()

running = set()
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    try:
        if open(f"/proc/{p}/comm").read().strip() != "python3":
            continue
        c = open(f"/proc/{p}/cmdline", "rb").read().decode("utf-8", "replace").replace("\0", " ")
    except Exception:
        continue
    if "poll_tennis.py" in c and " watch " in c:
        running.add(c.split(" watch ", 1)[1].split()[0])

try: cache = json.load(open(CACHE))
except Exception: cache = {}
feed = kt.Feed(rps=3.0, verbose=False)
disc = {}
# 429 RESILIENCE (2026-09-01): discovered.json is rebuilt from scratch each pass,
# and auto_quote_configs DROPS any live event missing from it — so one 429 on a
# series' /events call used to erase that whole series from the template and pull
# every resting quote on its LIVE matches (21:52:48Z pass: 2 429s, board 403->239,
# ITF vanished, template emptied mid-match). Carry the previous pass's entries
# forward for any series whose fetch fails; the config loop still requires a
# running poller with a live tape, so stale carried entries cannot quote anything.
try: _prev_disc = json.load(open(DISC))
except Exception: _prev_disc = {}
n = 0; new_mil = 0
for tour, ser in kt.SERIES.items():
    body = feed.get("/events", {"series_ticker": ser, "status": "open", "limit": 200, "with_nested_markets": "true"})
    if not body or body.get("events") is None:
        carried = {ev: d for ev, d in _prev_disc.items() if ev.startswith(ser + "-")}
        disc.update(carried)
        print(f"{now:%F %T}Z series {ser} /events fetch FAILED — carried {len(carried)} event(s) from previous pass")
        continue
    for e in body.get("events") or []:
        ev = e.get("event_ticker") or ""; mk = e.get("markets") or []
        if len(mk) != 2:
            continue
        def px(m):
            try: return (float(m["yes_bid_dollars"]) + float(m["yes_ask_dollars"])) / 2
            except Exception: return None
        tick = {m["ticker"]: px(m) for m in mk}
        comp = (e.get("product_metadata") or {}).get("competition")
        disc[ev] = {"tickers": list(tick), "mids": tick, "exchange_index": [m.get("exchange_index") for m in mk],
                    "competition": comp, "title": e.get("title"), "tour": tour}
        c = cache.get(ev)
        if c is None or (c.get("start") is None and nowts - c.get("checked", 0) > 1800):
            mil = feed.milestone(ev) or {}; det = mil.get("details") or {}; new_mil += 1
            cache[ev] = {"start": mil.get("start_date"), "checked": nowts, "id": mil.get("id"),
                         "gender": det.get("gender"), "tournament": det.get("tournament_name"), "best_of": det.get("best_of")}
            c = cache[ev]
        disc[ev].update({k: c.get(k) for k in ("start", "gender", "tournament", "best_of")})
json.dump(cache, open(CACHE, "w")); json.dump(disc, open(DISC, "w"), indent=0)

cands = []
NO_ARM = ()   # 26SEP14: ATP/WTA main tour re-armed for the taker-execution capture
# window (was excluded 26SEP01 when pollers cost GETs on the shared bucket; post-GET-audit
# a poller is REST-free — prices via the quoter wsbook, state via live_data_mux). The
# set1 dog-leader cell spans all six series, so capture must too.
for ev, d in disc.items():
    if ev.startswith(NO_ARM):
        continue
    if ev in running or not d.get("start"):
        continue
    try: mins = (dt.datetime.fromisoformat(d["start"].replace("Z", "+00:00")) - now).total_seconds() / 60
    except Exception: continue
    if -5 < mins <= lookahead:
        cands.append((mins, ev, d))
cands.sort()
def _dropped(rest):
    """Name what we are NOT attaching. 26SEP23: the old cap line printed only a COUNT,
    so 10 of the 19 ITF singles matches missed on 26SEP22 appeared nowhere in this log
    and had to be recovered by diffing milestones_cache.json against tapes/*.jsonl."""
    names = [e for _, e, _ in rest]
    shown = ", ".join(names[:40]) + (f", +{len(names)-40} more" if len(names) > 40 else "")
    return f"{len(names)} dropped: {shown}"

for ci, (mins, ev, d) in enumerate(cands):
    if len(running) + n >= MAX_POLLERS:
        print(f"{now:%F %T}Z poller cap {MAX_POLLERS} reached — {_dropped(cands[ci:])}")
        break
    am = mem_available_mb()
    # charge the poller about to be spawned as well as the n already spawned this pass,
    # so we never start one that would itself take the box below the floor.
    if am is not None and am - (n + 1) * POLLER_MB < RAM_FLOOR_MB:
        print(f"{now:%F %T}Z RAM floor {RAM_FLOOR_MB}MB reached — MemAvailable {am}MB "
              f"less {n + 1} pollers @~{POLLER_MB}MB — {_dropped(cands[ci:])}")
        break
    priced = {t: v for t, v in d["mids"].items() if v is not None}
    if len(priced) != 2 or not all(i == 3 for i in d["exchange_index"]):
        continue
    me = min(priced, key=priced.get)
    if not ATTACH:
        print(f"{now:%F %T}Z candidate {ev} (+{mins:.0f}m, {d.get('tournament') or d.get('competition')}) mids={ {k.rsplit('-',1)[-1]: round(v, 3) for k, v in priced.items()} }")
        continue
    bo = d.get("best_of")
    slam_men = (d.get("gender") == "men" and any(s in (d.get("tournament") or d.get("competition") or "") for s in ("US Open", "Wimbledon", "French Open", "Australian Open", "Roland")))
    args = ["--best-of", str(bo)] if bo in ("3", "5", 3, 5) else ([] if slam_men else ["--best-of", "3"])
    # SEED boundary 1 from the discovery mids (26AUG29): the poller no longer GETs
    # /markets, and the quoter's WS book log only starts once the match is live and
    # configured, so without this the fit would have no pre-match price at all.
    opp = next(t for t in priced if t != me)
    vf_pre = priced[me] / (priced[me] + priced[opp])
    seed = f"tapes/seed_{ev}.json"
    json.dump([{"state": {"sets_me": 0, "sets_opp": 0, "games_me": 0, "games_opp": 0},
                "price": round(vf_pre, 4), "label": f"pre-match from discovery {now:%H:%M}Z"}], open(seed, "w"))
    args += ["--seed", seed]
    if ev.startswith("KXITF"):
        # ZERO /markets since 26SEP11 (part 3 of the GET audit): the dog-windows
        # model now writes tapes/wsbook_<ev>.jsonl from the quoter's WebSocket
        # books for every configured match, and the poller's _wsbook fallback
        # reads it — same as ATP/WTA always worked. Every-cycle /markets at ~15
        # live pollers was 429ing on nearly every poll (5s backoffs stalling
        # tapes past the staleness cutoff). Books therefore require the quoter
        # to be RUNNING once a match is live; pregame reference comes from the
        # seed file (discovery mids). If the quoter is down, ITF tapes carry
        # state but no book — dog models fail closed on that, nothing quotes.
        args += ["--markets-every", "0"]
    if "DOUBLES" in ev:
        # 26SEP20: the quoter never subscribes doubles (trade-proofing), so the
        # wsbook path that feeds singles books never exists for them — two days
        # of doubles tapes carried scores but zero prices while the markets
        # traded 1k-25k contracts/match. Doubles pollers must GET their own
        # books: every 6th in-play cycle (~24s); PREGAME is a separate time-based
        # throttle in poll_tennis (--pregame-markets-secs, ~5 min) and no longer
        # scales off this number.
        #
        # 6 (was 3) 26SEP23: at 3 this was the ONLY per-poller REST in the fleet
        # (singles get 0) and it was the measured source of the arm loop's /events
        # 429s. Tennis total is only ~1.5 GET/s, but 27-34 concurrent doubles at one
        # GET/12s is 2.3-2.8 GET/s of it, and 429/pass tracked the doubles slate all
        # week -- 0.02-0.03 before doubles existed, 2.01 on 26SEP22 with 120 doubles
        # tapes, hourly r=+0.79 vs concurrent doubles on 26SEP23. 24s still gives the
        # tau study ~11 samples inside its 25-300s post-set-1-flip anchor window.
        args = [x for i, x in enumerate(args)
                if not (x == "--markets-every" or (i > 0 and args[i-1] == "--markets-every"))]
        args += ["--markets-every", "6"]
    # PREGAME CADENCE IS ASYMMETRIC (26SEP25). Singles do not need pregame samples: the tau
    # model falls back to seed_<ev>.json, whose discovery mid is a REAL price for ITF singles.
    # DOUBLES do -- the quoter never subscribes them, so /markets is their only book source and
    # the seed is the same ~0.50 placeholder the pregame book is, so a missing pregame tick
    # means a LOST tau row, not a degraded one. Measured 26SEP24 at a flat 600s: median pregame
    # ticks 7 -> 2, tapes with no usable pregame book 23% -> 37%, ~14 tau rows/day lost.
    # 120s with a 110s markets throttle gives ~13 real books across a 26-min attach window;
    # at ~30 concurrent doubles pregame pollers that is ~0.25 GET/s, about what markets-every
    # 6 just gave back. --pregame-markets-secs MUST be under the interval or the throttle, not
    # the interval, becomes the binding constraint.
    pre_args = (["--pregame-interval", "120", "--pregame-markets-secs", "110"]
                if "DOUBLES" in ev else ["--pregame-interval", "600"])
    log = open(f"tapes/deep_{ev.rsplit('-',1)[-1].lower()}.log", "a")
    subprocess.Popen([sys.executable, "-u", "poll_tennis.py", "--rps", "3", "watch", ev,
                      "--me", me.rsplit("-", 1)[-1], *args, "--interval", "4",
                      # 600 (was 30) 26SEP23, operator call: "not much use for pregame,
                      # all we care about is live". Pre-match we fetch NOTHING over REST -- state
                      # comes from tapes/_livedata_mux.json and the book from _wsbook, both
                      # local files -- so the only cost of a pregame cycle is a 963-byte tape
                      # row that re-serialises an unchanged `details` blob. This override was
                      # also fighting poll_tennis's own --pregame-interval default of 60.
                      # A match whose milestone start is a placeholder (ATP order-of-play
                      # doubles sat `not_started` for 11.8h on 26SEP23) wrote 4.5k identical
                      # rows at 30s. Do NOT "fix" that by reaping the poller: the attach
                      # filter is `-5 < mins <= lookahead`, so an event whose start time has
                      # passed is never re-attached, and killing it loses the pregame capture
                      # that the fit needs (see the header note on pre-match attachment).
                      #
                      # 600s IS THE SETTLED CADENCE, NOT THE ONLY ONE. poll_tennis ramps:
                      # it samples every --pregame-first-interval (20s) until the first row with
                      # a finite vig_free lands, then drops to this value, with a
                      # --pregame-fast-samples (90 = 30 min, our whole lookahead) budget so a
                      # placeholder-start match cannot spin fast forever. The ramp is NOT
                      # cosmetic: quoter _pregame_vf_me() takes the FIRST clean 0-0 row with a
                      # finite vig_free and otherwise falls back to seed_<ev>.json, and at attach
                      # time the quoter WS book file usually does not exist yet -- so a FLAT 600s
                      # would have handed the tau model the seed instead of a real tick, which
                      # for doubles is ~0.50 placeholder garbage. Worse, the pregame /markets
                      # throttle is counted in CYCLES (markets_every*15 = cycle 45 for doubles):
                      # flat 600s puts that first real book fetch at t=7.5 HOURS, i.e. never
                      # before start. The fast phase pulls it back to t=15 min.
                      *pre_args,
                      "--max-cycles", "20000"], stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd(),
                     start_new_session=True, env=dict(os.environ, TENNIS_CACHE=POLLER_CACHE))
    n += 1
    print(f"{now:%F %T}Z attached {ev} --me {me.rsplit('-',1)[-1]} {' '.join(args)} (+{mins:.0f}m, {d.get('tournament') or d.get('competition')})")
    time.sleep(1.2)   # phase-stagger the pollers' GET slots
print(f"{now:%F %T}Z pass: {feed.n_get} GETs ({new_mil} new milestones, {feed.n_429} rate-limited), {len(disc)} events on board, {len(running)} pollers already, {n} attached")
PY
    [ "${ONESHOT:-0}" = "1" ] && exit 0            # ONESHOT=1: one pass, then exit
    sleep "$CYCLE"
done
