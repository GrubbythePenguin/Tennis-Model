"""In-process shadow taker signals for run.py — VAL cluster + LoL pack-eat.

Called once from run.py startup via `taker_shadow.init()`. If
`taker_shadow.flag` is absent this is a no-op; if present it starts ONE
daemon thread that:

  * discovers today's VAL + LoL events from poly_parsed_markets.csv
    (re-read every 10 min, so refresh_tomorrow's evening update is picked
    up automatically) and resolves winner-map + series condition ids via
    gamma;
  * polls data-api trades per active leg (~6 rps global, 1.2s active /
    20s quiet), classifies the taker per txhash INCLUDING levels eaten
    (distinct maker prices);
  * runs two detectors: VAL (CLIP_PRINT/WALLET/BURST tiers) and LoL
    (PACK_EAT tier — the 26AUG08 signature study rule, shadow-validating
    its +5.7c@30s in-sample number);
  * appends signals to `_taker_shadow_signals.csv`.

SHADOW ONLY — no orders, no interaction with trading state. Every
network call and parse is wrapped; the thread can die without touching
the trading loop (daemon=True), and a crashed thread logs once.
"""
import collections
import csv
import os
import threading
import time

import requests

R = os.path.dirname(os.path.abspath(__file__))
FLAG = os.path.join(R, "taker_shadow.flag")
PARSED = os.path.join(R, "poly_parsed_markets.csv")
OUT = os.path.join(R, "_taker_shadow_signals.csv")

SPORTS = ("KXVALORANTGAME", "KXLOLGAME")
DISCOVER_S = 600.0
ACTIVE_POLL_S = 1.2
QUIET_POLL_S = 20.0
QUIET_AFTER_S = 900.0
PACE_S = 0.15
PAGE_LIMIT = 100        # raw fills per tape poll; a storm writes ~2-3/s,
                        # so 100 ≈ 40s of coverage — the 26AUG09 grit-drop
                        # mechanism. Standalone overrides via --page-limit.


def _log(msg):
    print(f"{time.strftime('%H:%M:%S')} [TAKER-SHADOW] {msg}", flush=True)


class _Shadow(threading.Thread):
    daemon = True
    name = "taker-shadow"

    def __init__(self):
        super().__init__()
        from val_swarm_detector import ValSwarmDetector
        self.det = {
            "KXVALORANTGAME": ValSwarmDetector(
                sports=("KXVALORANTGAME",),
                tiers=("CLIP_PRINT", "WALLET", "BURST", "WALLET_ECHOED",
                       "DUAL_WALLET")),
            "KXLOLGAME": ValSwarmDetector(
                sports=("KXLOLGAME",),
                tiers=("PACK_EAT", "WALLET", "WALLET_ECHOED",
                       "DUAL_WALLET", "ORACLE_TAKE")),
        }
        self.s = requests.Session()
        self._last_req = 0.0
        self.legs = {}                     # leg_key -> condition_id
        self.last_row = collections.defaultdict(float)
        self.next_poll = collections.defaultdict(float)
        new = not os.path.exists(OUT)
        self.outf = open(OUT, "a", newline="")
        self.w = csv.writer(self.outf)
        if new:
            self.w.writerow(["wall_ts", "src", "tier", "map_key", "dir",
                             "px", "size", "detail", "event_ts"])
            self.outf.flush()

    # ------------------------------------------------------------- http
    def _get(self, url, params=None):
        err = None
        for a in range(4):
            wait = self._last_req + PACE_S - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.time()
            try:
                r = self.s.get(url, params=params, timeout=12)
                if r.status_code == 429:
                    err = "429"
                    time.sleep(1.5 * (a + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                err = repr(e)[:80]
                time.sleep(0.8 * (a + 1))
        # loud, throttled: a silent None here cost grit's fires on 26AUG09
        now = time.time()
        if now - getattr(self, "_err_log_ts", 0.0) > 30.0:
            self._err_log_ts = now
            _log(f"_get FAILED ({err}) {url.rsplit('/', 1)[-1]}")
        return None

    # -------------------------------------------------------- discovery
    def _discover(self):
        slugs = {}
        try:
            with open(PARSED) as f:
                for r in csv.DictReader(f):
                    ev = r.get("kalshi_event", "")
                    if not ev.startswith(_SPORTS_OVERRIDE or SPORTS):
                        continue
                    if _EVENT_FILTER and not any(t in ev
                                                 for t in _EVENT_FILTER):
                        continue
                    url = r.get("poly_url", "")
                    if "/event/" in url:
                        slugs[ev] = url.rstrip("/").rsplit("/", 1)[-1]
        except OSError as e:
            _log(f"discover: cannot read parsed csv: {e}")
            return
        for kev, slug in slugs.items():
            if any(k.split("|")[0] == kev for k in self.legs):
                continue
            j = self._get("https://gamma-api.polymarket.com/events",
                          {"slug": slug})
            if not j:
                continue
            added = []
            for m in j[0].get("markets", []):
                q = (m.get("question") or "").lower()
                cid = m.get("conditionId")
                if not cid:
                    continue
                key = None
                for n in (1, 2, 3, 4, 5):
                    if "winner" in q and (f"map {n}" in q or f"game {n}" in q):
                        key = f"{kev}|M{n}"
                        break
                if key is None and "(bo" in q and " vs " in q:
                    key = kev
                if key and key not in self.legs:
                    self.legs[key] = cid
                    added.append(key.split("|")[-1] if "|" in key else "SER")
            if added:
                _log(f"tracking {kev} legs={added}")

    # ------------------------------------------------------------ tape
    @staticmethod
    def _taker_rows(batch):
        g = collections.defaultdict(list)
        for t in batch:
            g[t.get("transactionHash")].append(t)
        out = []
        for tx, fs in g.items():
            tot = sum(float(x["size"]) for x in fs)
            t = min(fs, key=lambda x: abs(float(x["size"]) -
                                          (tot - float(x["size"]))))
            oi = int(t.get("outcomeIndex", 0))
            px = float(t["price"]) * 100.0
            if oi == 1:
                px = 100.0 - px
            d = (1 if t["side"] == "BUY" else -1) * (1 if oi == 0 else -1)
            lv = len({round(float(x["price"]), 3) for x in fs if x is not t})
            out.append({"ts": float(t["timestamp"]), "dir": d, "px": px,
                        "sz": float(t["size"]),
                        "wallet": t.get("proxyWallet", ""), "lv": lv})
        out.sort(key=lambda r: r["ts"])
        return out

    def _emit(self, sig):
        try:
            self.w.writerow([f"{time.time():.3f}", "TAPE", sig.tier,
                             sig.map_key, sig.dir, f"{sig.px:.1f}",
                             f"{sig.size:.0f}", sig.detail, f"{sig.ts:.3f}"])
            self.outf.flush()
            _log(f"SIGNAL {sig!r}")
        except Exception:
            pass

    # ------------------------------------------------------------- run
    def run(self):
        _log(f"armed (shadow, in-process) pid={os.getpid()}")
        last_disc = 0.0
        hb = {"t": time.time(), "polls": 0}
        while True:
            try:
                now = time.time()
                if now - hb["t"] > 60.0:
                    _log(f"heartbeat polls/min={hb['polls']} "
                         f"legs={len(self.legs)}")
                    hb["t"], hb["polls"] = now, 0
                if now - last_disc > DISCOVER_S or not self.legs:
                    last_disc = now
                    self._discover()
                for key, cid in list(self.legs.items()):
                    if time.time() < self.next_poll[key]:
                        continue
                    hb["polls"] += 1
                    det = self.det.get(key.split("-")[0])
                    if det is None:
                        continue
                    batch = self._get(
                        "https://data-api.polymarket.com/trades",
                        {"market": cid, "takerOnly": "false",
                         "limit": PAGE_LIMIT, "offset": 0})
                    if batch:
                        rows = self._taker_rows(batch)
                        if rows:
                            self.last_row[key] = max(self.last_row[key],
                                                     rows[-1]["ts"])
                        for sig in det.on_tape_rows(key, rows):
                            self._emit(sig)
                    quiet = time.time() - self.last_row[key] > QUIET_AFTER_S
                    self.next_poll[key] = time.time() + (
                        QUIET_POLL_S if quiet else ACTIVE_POLL_S)
                time.sleep(0.25)
            except Exception as e:
                _log(f"loop error (continuing): {e!r}")
                time.sleep(5.0)


_SPORTS_OVERRIDE = None
_EVENT_FILTER = ()      # standalone --events: substrings, e.g. ("26AUG09",)

# ---------------------------------------------------------------- executors
# run.py owns the taker bot processes (operator spec 26AUG08: "if I restart
# run.py everything will start", same as arber/quoter via manager). On init:
# kill any prior instance via pidfile (takeover — no duplicates across
# restarts), spawn fresh, then respawn any that die. Live/halt stays governed
# by each row's stop_quoting (RowArm re-reads every 30s) — the supervisor
# only guarantees the processes exist.
_VENV_PY = os.path.join(R, ".venv", "bin", "python3")
# 26AUG10 — ALL TAKER BOTS ARE DRY. `--send` was removed from the three
# tape bots, so arming is no longer a one-line CSV edit; it needs a bounce.
# That is deliberate. Operator directive: "we shouldn't be trading on
# anything yet until we get some latency confirmation."
#
# The tape path is the reason. Measured this box, 522 real LoL signals:
# only 1 cleared the MAX_AGE_S=3.0 freshness gate; median signal age is
# 132s and Kalshi has fully echoed by 60s. Previously these bots ran
# `--send` held off ONLY by stop_quoting, so a single CSV edit — or a
# fail-open in RowArm — put a 132s-stale path live. Double-key it instead.
BOTS = {
    "MAP_TAKER_VAL": [_VENV_PY, os.path.join(R, "val_cluster_exec.py"),
                      "--params-id", "MAP_TAKER_VAL",
                      "--tiers", "CLIP_PRINT,BURST,WALLET_ECHOED,DUAL_WALLET",
                      "--include-series"],
    "MAP_TAKER_LOL": [_VENV_PY, os.path.join(R, "val_cluster_exec.py"),
                      "--params-id", "MAP_TAKER_LOL",
                      "--tiers",
                      "PACK_EAT,WALLET_ECHOED,DUAL_WALLET,ORACLE_TAKE",
                      "--leagues", "*", "--exclude", "",
                      "--include-series"],
    "MAP_TAKER_DOTA": [_VENV_PY, os.path.join(R, "map_taker_live.py"),
                       "--params-id", "MAP_TAKER_DOTA",
                       "--sports", "KXDOTA2GAME", "--leagues", "*",
                       "--exclude", "", "--maps", "1,2"],
    # WIRE path, dry. Sources _ws_signal_shadow.csv (0.51s median detect)
    # rather than the data-api tape, and writes its decisions to
    # _ws_taker_live_orders.csv — a SEPARATE file, so wire and tape
    # markouts stay resolvable. Scoped to the Korean leagues where the
    # oracle edge was actually measured (KeSPA = Tarot's turf, LCK CL =
    # Grit's); non-Korean LoL is the only bucket that ever measured
    # negative (-0.68c, p10 -21.68c) and must not be collected as if it
    # were comparable.
    "MAP_TAKER_LOL_WS": [_VENV_PY, os.path.join(R, "ws_cluster_exec.py"),
                         "--params-id", "MAP_TAKER_LOL_WS",
                         "--tiers", "WS_ECHO,WS_CLIP,WS_BIG",
                         "--leagues", "KESPA,LCK CHALLENGERS",
                         "--exclude", "", "--include-series"],
}


def _pidfile(name):
    return os.path.join(R, f"_{name.lower()}.pid")


def _outfile(name):
    return os.path.join(R, f"_{name.lower()}.out")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def _read_pid(name):
    try:
        return int(open(_pidfile(name)).read().strip())
    except (OSError, ValueError):
        return None


class _Supervisor(threading.Thread):
    daemon = True
    name = "taker-bot-supervisor"

    def __init__(self):
        super().__init__()
        self.procs = {}

    @staticmethod
    def _die_with_parent():
        """PR_SET_PDEATHSIG: the kernel SIGTERMs the bot the instant run.py
        exits — ANY exit, including SIGKILL. No ghost bots, ever. Takers
        post IOC-only (nothing rests), so a hard stop leaves no orders."""
        import ctypes
        import signal as _sig
        try:
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(
                1, _sig.SIGTERM)              # 1 = PR_SET_PDEATHSIG
        except Exception:
            pass

    def _spawn(self, name):
        import subprocess
        old = _read_pid(name)
        if old and _alive(old):
            try:
                os.kill(old, 15)
                _log(f"supervisor: killed prior {name} pid={old}")
            except OSError:
                pass
        try:
            out = open(_outfile(name), "a")
            p = subprocess.Popen(BOTS[name], cwd=R, stdout=out,
                                 stderr=subprocess.STDOUT,
                                 preexec_fn=self._die_with_parent)
            self.procs[name] = p
            with open(_pidfile(name), "w") as f:
                f.write(str(p.pid))
            _log(f"supervisor: spawned {name} pid={p.pid}")
        except Exception as e:
            _log(f"supervisor: spawn {name} FAILED: {e!r}")

    def run(self):
        for name in BOTS:
            self._spawn(name)
        while True:
            try:
                time.sleep(30.0)
                for name, p in list(self.procs.items()):
                    if p.poll() is not None:
                        _log(f"supervisor: {name} exited rc={p.returncode} "
                             f"— respawning")
                        self._spawn(name)
            except Exception as e:
                _log(f"supervisor loop error (continuing): {e!r}")


_started = [False]


def init():
    """Flag-gated, idempotent, never raises. Call from run.py startup."""
    try:
        if _started[0] or not os.path.exists(FLAG):
            return False
        _started[0] = True
        _Shadow().start()
        _Supervisor().start()
        return True
    except Exception as e:
        try:
            _log(f"init failed (shadow disabled): {e!r}")
        except Exception:
            pass
        return False


if __name__ == "__main__":
    # Standalone mode (pre-restart coverage / soak): runs the same thread
    # body in the foreground. Kill this process once run.py restarts with
    # taker_shadow.flag armed, or the legs get double-polled.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sports", default="",
                    help="comma list of ticker prefixes to restrict to")
    ap.add_argument("--events", default="",
                    help="comma list of event-ticker substrings (e.g. a "
                         "date like 26AUG09) to restrict tracking to")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="override PACE_S (global seconds between requests)")
    ap.add_argument("--page-limit", type=int, default=0,
                    help="override PAGE_LIMIT (raw fills per tape poll)")
    a = ap.parse_args()
    if a.sports.strip():
        _SPORTS_OVERRIDE = tuple(
            s.strip() for s in a.sports.split(",") if s.strip())
    if a.events.strip():
        _EVENT_FILTER = tuple(
            s.strip() for s in a.events.split(",") if s.strip())
    if a.pace > 0:
        PACE_S = a.pace
    if a.page_limit > 0:
        PAGE_LIMIT = a.page_limit
    _log(f"standalone, sports={_SPORTS_OVERRIDE} events={_EVENT_FILTER} "
         f"pace={PACE_S} page_limit={PAGE_LIMIT}")
    _Shadow().run()
