"""live_data MULTIPLEXER — one process fetches every watched match's live_data.

    python3 live_data_mux.py            # or: setsid nohup ... >> tapes/mux.log

WHY (26SEP11 GET audit): 25-50 per-match pollers each GETting /live_data every 4s
put 4-6 requests/s on a bucket that 429s ~9-23% of them; the flat 5s backoffs stack
exactly at set boundaries and stretched tape gaps past the dog models' 20s
staleness cutoff, killing 5 live quoting windows in one morning. The batch endpoint
takes 40 milestone_ids per call, so ONE process can carry the entire board in 1-2
GETs per cycle — ~0.5/s total, immune to the per-request 429 lottery at any
plausible account tier.

WHAT IT DOES
  * Discovers the active set locally, zero GETs: running `poll_tennis.py watch`
    processes (from /proc) -> their events -> milestone ids via
    tapes/milestones_cache.json (the arm loop's cache).
  * Every CYCLE seconds: 1-2 batched /live_data GETs via kalshi_tennis.Feed
    (hard rps cap, flat 429 backoff, kill-flag honored).
  * Writes tapes/_livedata_mux.json ATOMICALLY (tmp+rename):
        {"ts": <epoch>, "data": {<milestone_id>: <details>, ...}}

CONSUMERS: poll_tennis.py reads the snapshot first and only falls back to its own
direct GET when the snapshot is stale/absent/missing its milestone — so pollers
lose no independence: kill this process and the fleet degrades to the old
per-poller GETs within one cycle, nothing breaks.

The kill flag (disable_tennis_poller.flag) pauses fetching, matching the fleet.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalshi_tennis as kt

HERE = os.path.dirname(os.path.abspath(__file__))
TAPES = os.path.join(HERE, "tapes")
CACHE = os.path.join(TAPES, "milestones_cache.json")
OUT = os.path.join(TAPES, "_livedata_mux.json")
CYCLE = float(os.environ.get("MUX_CYCLE_S", "3"))
REDISCOVER_S = 15.0
STATUS_EVERY_S = 300.0


def active_events() -> set:
    evs = set()
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
            evs.add(c.split(" watch ", 1)[1].split()[0])
    return evs


class Cache:
    def __init__(self):
        self.mtime = -1.0
        self.map = {}

    def event_to_mid(self):
        try:
            mt = os.path.getmtime(CACHE)
        except OSError:
            return self.map
        if mt != self.mtime:
            try:
                d = json.load(open(CACHE))
                self.map = {ev: (v or {}).get("id") for ev, v in d.items() if (v or {}).get("id")}
                self.mtime = mt
            except Exception as e:
                print(f"[mux] milestones cache unreadable: {e}", file=sys.stderr)
        return self.map


def main():
    feed = kt.Feed(rps=2.0, verbose=True)
    cache = Cache()
    mids: list = []
    last_disc = 0.0
    last_status = 0.0
    cycles = fetched = 0
    print(f"[mux] up — cycle {CYCLE}s, out {OUT}")
    while True:
        now = time.time()
        if kt.disabled():
            time.sleep(10)
            continue
        if now - last_disc >= REDISCOVER_S or not mids:
            last_disc = now
            e2m = cache.event_to_mid()
            evs = active_events()
            mids = sorted({e2m[e] for e in evs if e in e2m})
            missing = [e for e in evs if e not in e2m]
            if missing and now - last_status >= STATUS_EVERY_S:
                print(f"[mux] {len(missing)} active event(s) not in milestones cache "
                      f"(their pollers fall back to direct GETs): {missing[:4]}")
        if not mids:
            time.sleep(CYCLE)
            continue
        try:
            data = feed.live_data(mids)          # chunks by 40 internally
        except kt.RateLimited as e:
            print(f"[mux] {e} — sleeping 30s", file=sys.stderr)
            time.sleep(30)
            continue
        except Exception as e:
            print(f"[mux] fetch error {e!r}", file=sys.stderr)
            time.sleep(CYCLE)
            continue
        cycles += 1
        fetched += len(data)
        if data:
            tmp = OUT + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump({"ts": time.time(), "data": data}, f)
                os.replace(tmp, OUT)
            except Exception as e:
                print(f"[mux] write failed: {e!r}", file=sys.stderr)
        if time.time() - last_status >= STATUS_EVERY_S:
            last_status = time.time()
            print(f"[mux] {cycles} cycles, {len(mids)} milestones tracked, "
                  f"{feed.n_get} GETs ({feed.n_429} 429s), last batch {len(data)} payloads")
        time.sleep(max(0.0, CYCLE - (time.time() - now)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
