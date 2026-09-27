"""tt_screener.py — table tennis paper-trading screener. NO ORDERS, EVER.

One process, one table, refreshed every ~12s: every open TT Elite match with
the model's theoretical value next to the live cross-market book, plus the
edge it would see hitting the bid / lifting the offer. Intended trades are
LOGGED (tapes/tt_intents.jsonl), never sent — there is no execution path in
this file and none imported.

    EVENT        SCORE                 PRE    BID    ASK    THEO   eBID   eASK
    ...MBAJNO    11-6, 12-14, 6-3      0.42*  0.38   0.44   0.415  -3.5   -2.5

Data paths (deliberately split):
    books   quoter/ws_delta_book.WSDeltaManager — ONE WebSocket, delta books
            for every tracked market, dynamic subscribe as matches churn.
            No REST fallback for books: a missing WS book renders '-' rather
            than adding GET load next to the live trading system.
    scores  ONE batched /live_data GET per refresh (all milestones at once)
            plus a /events + /milestones discovery sweep every ~2 min, via
            kalshi_tennis.Feed (shared-bucket rps cap, 429 surrender).
            Kill flag: disable_tt_poller.flag (poll_tt's, not tennis's).

Theo: table_tennis_model with (p, q) anchored per match from the FIRST
cross-market mid seen with a sane spread — pregame when we attach early
(anchor marked plain), mid-match otherwise (marked '*'). Anchors persist to
_tt_screener_anchors.json so a restart keeps the pregame numbers.

usage:
    .venv/bin/python tt_screener.py [--interval 12] [--edge 0.04]
        [--size 10] [--max-spread 0.10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "quoter"))

import kalshi_tennis as kt
import poll_tt                                     # rebinds kt.KILL_FLAG to TT's
from poll_tt import tt_state, solve_pq, theo, SERIES
from table_tennis_model import advance_state

# Series discovery is DYNAMIC (2026-09-09): every SERIES_REFRESH_S the screener
# pulls the Sports series list and tracks every 'Table Tennis'-tagged *MATCH
# series with open events — new leagues Kalshi lists get taped automatically.
# THEO (and therefore quoting) is gated separately on
# quoter/tt_quotable_series.json, which only tt_populate_configs' shard
# verification writes; an unverified new series is tracked and taped but
# never priced. *GAME series (per-game markets) are excluded — different
# resolution unit than the match model.
SERIES_REFRESH_S = 1800.0
QUOTABLE_P = os.path.join(_ROOT, "quoter", "tt_quotable_series.json")
from ws_delta_book import WSDeltaManager, KalshiAuth

ANCHORS_P = os.path.join(_ROOT, "_tt_screener_anchors.json")
INTENTS_P = os.path.join(_ROOT, "tapes", "tt_intents.jsonl")
ROWS_P = os.path.join(_ROOT, "tapes", "tt_screener_rows.jsonl")
# Snapshot handoff to the quoter's tt_band model: the CURRENT state of every
# tracked match, atomically replaced each refresh. The quoter treats this file
# as the score feed and hard-stops on staleness — see quoter/tt_band_model.py.
SNAP_P = os.path.join(_ROOT, "tapes", "_tt_state.json")
DISCOVER_EVERY = 10           # refresh cycles between /events sweeps
ENDED_LINGER_S = 60.0         # keep an ended match in the snapshot briefly so
                              # the final state (winner_ticker, ended_ts) lands
                              # on the row tape before the match is dropped


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def top_from_fp(book):
    """(yes_bid, yes_ask) in dollars from a WSDeltaManager orderbook_fp,
    or (None, None). yes_dollars/no_dollars are BID stacks, high first."""
    if not book:
        return None, None
    ob = book.get("orderbook_fp") or {}
    yes, no = ob.get("yes_dollars") or [], ob.get("no_dollars") or []
    bid = float(yes[0][0]) if yes else None
    ask = round(1.0 - float(no[0][0]), 4) if no else None
    return bid, ask


def cross(me_book, opp_book):
    """Effective (bid, ask) on competitor1 combining both markets: selling c1
    at `bid` can also be done by buying c2 at its ask, and vice versa."""
    b1, a1 = top_from_fp(me_book)
    b2, a2 = top_from_fp(opp_book)
    bids = [x for x in (b1, None if a2 is None else round(1 - a2, 4)) if x is not None]
    asks = [x for x in (a1, None if b2 is None else round(1 - b2, 4)) if x is not None]
    return (max(bids) if bids else None), (min(asks) if asks else None)


class Match:
    def __init__(self, event, mid, t1, t2, name1, name2):
        self.event, self.mid = event, mid
        self.t1, self.t2, self.name1, self.name2 = t1, t2, name1, name2
        self.det = {}
        self.anchor = None            # {target, p, q, state, pregame}
        self.ended_ts = None          # first cycle the feed showed it decided
        self.last_intent = {}         # side -> (state, price) last logged

    def winner_ticker(self):
        """Ticker of the feed-declared winner, or None. Trusts the explicit
        winner fields only — never inferred from the score."""
        w = self.det.get("winner")
        wid = self.det.get("winner_competitor_id")
        if w == "competitor1" or (wid and wid == self.det.get("competitor1_id")):
            return self.t1
        if w == "competitor2" or (wid and wid == self.det.get("competitor2_id")):
            return self.t2
        return None

    def score_str(self):
        parts = []
        for ps in sorted(self.det.get("period_scores") or [],
                         key=lambda r: r.get("number") or 0):
            if ps.get("type") == "set":
                parts.append(f"{ps.get('competitor1_score')}-{ps.get('competitor2_score')}")
        return ", ".join(parts)


class Screener:
    def __init__(self, a):
        self.a = a
        self.feed = kt.Feed(rps=2.0, verbose=False)
        self.matches: dict[str, Match] = {}
        self.mgr = None
        self.anchors = {}
        self.series = [SERIES]                # grows via _discover_series
        self._series_ts = 0.0
        self._quotable_cache = {"mtime": -1.0, "data": {}}
        self._bo5_alerted = set()
        if os.path.exists(ANCHORS_P):
            try:
                self.anchors = json.load(open(ANCHORS_P))
            except Exception:
                self.anchors = {}

    # ---------------------------------------------------------- discovery

    @staticmethod
    def _starts_soon(ev):
        """Window filter from the start time embedded in the ticker
        (KXTTELITEMATCH-26SEP071330XXXYYY, US/Eastern): keep matches from
        70 min before now to 45 min after — the rest are hours away and
        would make the discovery sweep minutes long at the rps cap."""
        import re
        from datetime import datetime
        from zoneinfo import ZoneInfo
        m = re.match(r"^KX[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})", ev)
        if not m:
            return True                                    # unknown shape: keep
        yy, mon, dd, hh, mi = m.groups()
        try:
            start = datetime.strptime(f"{yy}{mon}{dd} {hh}{mi}", "%y%b%d %H%M") \
                .replace(tzinfo=ZoneInfo("America/New_York"))
        except ValueError:
            return True
        dt = (start - datetime.now(ZoneInfo("America/New_York"))).total_seconds()
        return -70 * 60 <= dt <= 45 * 60

    def _quotable(self):
        """quoter/tt_quotable_series.json, mtime-cached. Series -> info."""
        try:
            mt = os.path.getmtime(QUOTABLE_P)
        except OSError:
            return self._quotable_cache["data"]
        if mt != self._quotable_cache["mtime"]:
            try:
                self._quotable_cache["data"] = json.load(open(QUOTABLE_P))
                self._quotable_cache["mtime"] = mt
            except Exception:
                pass
        return self._quotable_cache["data"]

    def _discover_series(self):
        """Blocking: refresh the tracked series list from the tag taxonomy."""
        if time.time() - self._series_ts < SERIES_REFRESH_S:
            return
        body = self.feed.get("/series", {"category": "Sports", "limit": 800})
        if not body:
            return                             # transient — keep current list
        self._series_ts = time.time()
        found = []
        for s in body.get("series") or []:
            tick = s.get("ticker") or ""
            tags = [t.lower() for t in (s.get("tags") or [])]
            if "table tennis" in tags and tick.endswith("MATCH"):
                found.append(tick)
        for t in found:
            if t not in self.series:
                self.series.append(t)
                print(f"[series] now tracking {t}", flush=True)

    def _discover(self):
        """Blocking: refresh the set of open TT events and resolve new ones."""
        self._discover_series()
        fresh = []
        for series in self.series:
            body = self.feed.get("/events", {"series_ticker": series,
                                             "status": "open", "limit": 200}) or {}
            for e in body.get("events") or []:
                ev = e.get("event_ticker")
                if not ev or ev in self.matches or not self._starts_soon(ev):
                    continue
                ms = [m for m in (self.feed.get("/milestones",
                          {"limit": 200, "related_event_ticker": ev}) or {})
                      .get("milestones") or []
                      if str(m.get("type") or "").startswith("table_tennis")]
                if ms:
                    fresh.append((ev, ms[0]["id"]))
        dets = self.feed.live_data([mid for _, mid in fresh]) if fresh else {}
        for ev, mid in fresh:
            det = dets.get(mid) or {}
            c1 = det.get("competitor1_id") or ""
            mkts = self.feed.markets(ev)
            by_uuid = {(m.get("custom_strike") or {}).get("table_tennis_competitor"): m
                       for m in mkts}
            if len(mkts) != 2 or c1 not in by_uuid:
                continue                                  # refuse to guess
            t1 = by_uuid[c1]["ticker"]
            t2 = next(m["ticker"] for m in mkts if m["ticker"] != t1)
            mt = Match(ev, mid, t1, t2,
                       det.get("competitor1_name") or "?",
                       det.get("competitor2_name") or "?")
            mt.det = det
            if ev in self.anchors:                        # restart: keep anchor
                mt.anchor = self.anchors[ev]
            self.matches[ev] = mt

    def _scores(self):
        """Blocking: one batched live_data GET for every tracked milestone."""
        dets = self.feed.live_data([m.mid for m in self.matches.values()])
        for m in self.matches.values():
            if m.mid in dets:
                m.det = dets[m.mid]

    # ------------------------------------------------------------- pricing

    def _maybe_anchor(self, m, legs, st, live):
        """Anchor (p, q) off the CROSS-market bid/ask, re-anchored until live.

        2026-09-09 SKOKKA post-mortem, round two. The cross book is the right
        gate: cross bid/ask are EXECUTABLE prices (sellable at the bid,
        buyable at the ask through the complements), so a tight cross is a
        real market even when both legs are 80c wide — a tight cross from
        wide legs just means both legs carry a serious bid. Symmetric junk
        legs (10/90 vs 10/90) make a wide cross and are refused either way.

        What actually poisoned SKOKKA was FREEZING the first qualifying
        print: anchored 0.53 at 22:05 off a genuine 48/58 cross, then ignored
        the market walking to 0.35/0.42 by first ball. So pregame the anchor
        now RE-ANCHORS on every >=1c move of the cross mid and freezes at the
        first live cycle — the value at first ball is the market's final
        pregame word, not its first sketch.
        """
        if st is None:
            return
        if m.anchor and (live or not m.anchor.get("pregame", True)):
            return                        # frozen: match live, or anchored in-play
        b1, a1, b2, a2 = legs
        bids = [x for x in (b1, None if a2 is None else 1 - a2) if x is not None]
        asks = [x for x in (a1, None if b2 is None else 1 - b2) if x is not None]
        if not bids or not asks:
            return
        bid, ask = max(bids), min(asks)
        cap = self.a.max_spread if not live else self.a.max_spread * 1.5
        if ask - bid > cap:
            return
        mid = (bid + ask) / 2.0
        if not 0.02 < mid < 0.98:
            return
        if m.anchor and abs(mid - m.anchor["target"]) < 0.01:
            return                        # pregame re-anchor only on a real move
        p, q = solve_pq(mid, *st, serve_share=self.a.serve_share)
        m.anchor = {"target": round(mid, 4), "p": p, "q": q, "state": list(st),
                    "pregame": not live, "ts": _now()}
        self.anchors[m.event] = m.anchor
        tmp = ANCHORS_P + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.anchors, f, indent=1)
        os.replace(tmp, ANCHORS_P)

    def _intent(self, m, side, price, size, st, t, edge):
        """Log an intended trade once per (side, state, price) — not per cycle."""
        key = (tuple(st), round(price, 2))
        if m.last_intent.get(side) == key:
            return
        m.last_intent[side] = key
        with open(INTENTS_P, "a") as f:
            f.write(json.dumps({"ts": _now(), "event": m.event, "side": side,
                                "ticker": m.t1, "price": price, "size": size,
                                "state": list(st), "theo": round(t, 4),
                                "edge": round(edge, 4),
                                "anchor": m.anchor["target"],
                                "pregame_anchor": m.anchor["pregame"]}) + "\n")

    # -------------------------------------------------------------- render

    def refresh(self):
        rows = []
        drop = []
        tape = []                     # per-refresh rows -> ROWS_P, the dataset
                                      # that grades theo-vs-market afterwards
        snap = {}                     # per-refresh state -> SNAP_P for the quoter
        for ev, m in sorted(self.matches.items()):
            status = (m.det.get("status") or "?").lower()
            live = kt.is_live(m.det)
            ended = bool(m.det.get("winner")) or status in ("ended", "closed", "finished")
            if ended:
                if m.ended_ts is None:
                    m.ended_ts = time.time()
                if time.time() - m.ended_ts > ENDED_LINGER_S:
                    drop.append(ev)
                    continue
            st = tt_state(m.det)
            # MONOTONIC STATE GUARD (2026-09-10, the AGRFGR -$130 lesson): the
            # feed intermittently RESETS a live match's score to 0-0 (48s
            # observed at 2-2 6-5 in a fifth set). Scores only move forward;
            # a big regression is a glitch, and pricing it as a fresh match
            # put tight fresh-match quotes under a match-point market. Hold
            # the state at None (no theo -> quotes pulled) until the
            # regressed reading persists 3 cycles (a real scorer correction
            # does; a transport glitch doesn't).
            if st is not None and m.det.get("status"):
                hwm = getattr(m, "state_hwm", None)
                prog = (st[0] + st[1], sum(st))
                if hwm and (prog[0] < hwm[0] or
                            (prog[0] == hwm[0] and prog[1] < hwm[1] - 2)):
                    # Thresholds from the 2026-09-10 12h glitch survey (55
                    # windows): provider resets last up to 48s and hit every
                    # live match at once; set-level regressions ran 82-113s
                    # and REVERTED — so 3 cycles was far too trusting. A full
                    # reset (near-0 state against a real hwm) is NEVER
                    # believable while live; a partial regression must stand
                    # ~2 min to be a real scorer correction.
                    full_reset = prog[1] <= 2 and hwm[1] >= 8
                    m.glitch_n = getattr(m, "glitch_n", 0) + 1
                    if full_reset or m.glitch_n <= 10:
                        if m.glitch_n == 1:
                            print(f"\n[glitch] {ev[-14:]}: state {st} regressed "
                                  f"below hwm {hwm} — holding theo", flush=True)
                        st = None
                    else:
                        m.state_hwm = prog      # persisted: accept correction
                        m.glitch_n = 0
                else:
                    m.state_hwm = max(hwm, prog) if hwm else prog
                    m.glitch_n = 0
            series = ev.split("-", 1)[0]
            quotable = series in self._quotable()
            # Bo5 TRIPWIRE: a set score past 3 means this league is NOT best
            # of 5 and the whole model is wrong for it. No theo, loud alert;
            # tt_populate_configs also skips the series while flagged.
            bo5_violation = st is not None and max(st[0], st[1]) > 3
            if bo5_violation and ev not in self._bo5_alerted:
                self._bo5_alerted.add(ev)
                print(f"\n*** BO5 VIOLATION {ev}: set score {st[0]}-{st[1]} — "
                      f"series {series} is not best-of-5; not pricing it. ***",
                      flush=True)
            bk1 = self.mgr.get_orderbook_fp(m.t1)
            bk2 = self.mgr.get_orderbook_fp(m.t2)
            bid, ask = cross(bk1, bk2)
            legs = top_from_fp(bk1) + top_from_fp(bk2)
            if not ended and quotable and not bo5_violation:
                # Points on the board = the match HAS started, whatever the
                # status field says this cycle (it flaps on screener boot and
                # feed hiccups) — without this, the pregame re-anchor path
                # runs against an in-play book and overwrites the frozen
                # pregame anchor with a mid-match price (POSKKL, 2026-09-09
                # 23:00: 0.139 -> 0.060 during a status flap).
                started = st is not None and any(st)
                self._maybe_anchor(m, legs, st, live or started)
            t = eb = ea = qb = qa = None
            if m.anchor and st is not None and quotable and not bo5_violation:
                t = theo(m.anchor["p"], m.anchor["q"], *st)[0]
                # Worst-case quote band for a 2-10 point delayed feed: value the
                # state as if the hidden points ALL went one way. Quoting inside
                # this band is unsafe; at qb/qa even a maximally-stale feed
                # cannot make a fill negative-value at the model's own theo.
                d = self.a.delay_points
                qb = theo(m.anchor["p"], m.anchor["q"],
                          *advance_state(*st, n=d, me_wins=False))[0]
                qa = theo(m.anchor["p"], m.anchor["q"],
                          *advance_state(*st, n=d, me_wins=True))[0]
                if bid is not None:
                    eb = bid - t
                if ask is not None:
                    ea = t - ask
                if not ended and live:
                    if eb is not None and eb >= self.a.edge:
                        self._intent(m, "sell", bid, self.a.size, st, t, eb)
                    if ea is not None and ea >= self.a.edge:
                        self._intent(m, "buy", ask, self.a.size, st, t, ea)
            snap[m.event] = {"ts": time.time(), "status": status, "live": live,
                             "series": series, "quotable": quotable,
                             "bo5_violation": bo5_violation,
                             "ended": ended, "ended_ts": m.ended_ts,
                             "winner_ticker": m.winner_ticker() if ended else None,
                             "state": list(st) if st else None,
                             "score": m.score_str(),
                             "t1": m.t1, "t2": m.t2,
                             "name1": m.name1, "name2": m.name2,
                             "theo": None if t is None else round(t, 4),
                             "qbid": None if qb is None else round(qb, 4),
                             "qask": None if qa is None else round(qa, 4),
                             "anchor": None if not m.anchor else m.anchor["target"],
                             "pregame_anchor": None if not m.anchor else m.anchor["pregame"]}
            tape.append({"ts": _now(), "event": m.event, "status": status,
                         "state": list(st) if st else None,
                         "score": m.score_str(), "bid": bid, "ask": ask,
                         "theo": None if t is None else round(t, 4),
                         "qbid": None if qb is None else round(qb, 4),
                         "qask": None if qa is None else round(qa, 4),
                         "anchor": None if not m.anchor else m.anchor["target"],
                         "pregame_anchor": None if not m.anchor else m.anchor["pregame"]})
            if not live or ended:
                continue              # display live-only; taping/anchoring above
                                      # still covers pregame and just-ended rows
            fm = lambda v, n=3: "  -  " if v is None else f"{v:.{n}f}"
            fc = lambda v: "  -  " if v is None else f"{100 * v:+5.1f}"
            pre = "  -  " if not m.anchor else (
                f"{m.anchor['target']:.3f}" + ("" if m.anchor["pregame"] else "*"))
            rows.append(f"{ev[-14:]:<15} {status[:7]:<8} {m.score_str()[:29]:<30} "
                        f"{pre:<7} {fm(bid)}  {fm(ask)}  {fm(t)}  {fc(eb)}  {fc(ea)}  "
                        f"{fm(qb)}  {fm(qa)}  "
                        f"{m.name1[:16]} v {m.name2[:16]}")
        for ev in drop:
            del self.matches[ev]
        if tape:
            with open(ROWS_P, "a") as f:
                for r in tape:
                    f.write(json.dumps(r) + "\n")
        # atomic replace so the quoter can never read a torn snapshot
        tmp = SNAP_P + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"ts": time.time(), "delay_points": self.a.delay_points,
                       "events": snap}, f)
        os.replace(tmp, SNAP_P)
        hdr = (f"{'EVENT':<15} {'STATUS':<8} {'SCORE':<30} {'PRE':<7} "
               f"{'BID':<6} {'ASK':<6} {'THEO':<6} {'eBID':<6} {'eASK':<6} "
               f"{'qBID':<6} {'qASK':<6} MATCH")
        out = [f"tt_screener  {_now()}  live={len(rows)} tracked={len(self.matches)}  "
               f"(gets={self.feed.n_get}/429s={self.feed.n_429})  "
               f"edge>={self.a.edge * 100:.0f}c -> {INTENTS_P}", hdr] + rows
        if sys.stdout.isatty():
            print("\033[2J\033[H" + "\n".join(out), flush=True)
        else:
            print("\n".join(out) + "\n", flush=True)

    # ---------------------------------------------------------------- main

    async def run(self):
        auth = KalshiAuth(os.getenv("KALSHI_API_KEY_ID"),
                          os.getenv("KALSHI_PRIVATE_KEY_PATH"))
        # stale_book_s: production uses 60s because its callers REST-fallback on
        # None. This screener has NO REST fallback by design, and quiet TT
        # pregame books legitimately go 60s+ without a delta — observed in the
        # first live test: four pregame books flapping to '-' every minute.
        # 900s serves the last known book instead; live books tick constantly
        # so intents never fire off anything actually stale.
        self.mgr = WSDeltaManager(auth, stale_book_s=900.0)
        await self.mgr.start()
        print(f"tt_screener starting: WS up, discovering {SERIES} matches "
              f"(first sweep can take ~20s at the rps cap)...", flush=True)
        cycles = 0
        while True:
            if kt.disabled():
                print(f"kill flag {kt.KILL_FLAG} present — stopping.")
                break
            try:
                # SCORES + REFRESH FIRST, discovery after: a multi-series
                # discovery sweep takes 10-20s at the rps cap, and running it
                # before the refresh let the snapshot age toward tt_band's
                # staleness cutoff — spurious quote pulls every DISCOVER_EVERY
                # cycles. Discovery finding a match one cycle late is free;
                # the feed going stale mid-match is not.
                await asyncio.to_thread(self._scores)
                if self.matches:
                    await self.mgr.set_tickers(
                        {t for m in self.matches.values() for t in (m.t1, m.t2)})
                    self.refresh()
                # rediscover every cycle while empty: a single 429 on the boot
                # sweep must not leave the screener blind for DISCOVER_EVERY cycles
                if cycles % DISCOVER_EVERY == 0 or not self.matches:
                    await asyncio.to_thread(self._discover)
            except kt.RateLimited as e:
                # DO NOT exit on a 429 storm (changed 2026-09-09 for unattended
                # runs): dying here kills the feed for the rest of the night.
                # Cool down long enough for the shared bucket to recover; the
                # quoter pulls all quotes via the staleness hard stop meanwhile,
                # which is exactly the safe state.
                print(f"\n*** {e} — cooling down 600s (quoter pulls on stale "
                      f"feed meanwhile).", file=sys.stderr)
                await asyncio.sleep(600)
                continue
            except RuntimeError as e:            # kill flag raised mid-fetch
                print(f"*** {e} — stopping.", file=sys.stderr)
                break
            except Exception as e:
                # An unexpected error must not kill an unattended overnight
                # run. Log it, skip the cycle, keep the feed alive.
                import traceback
                print(f"\n*** cycle error: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc()
                await asyncio.sleep(self.a.interval)
                continue
            if not self.matches:              # refresh already ran pre-discovery
                self.refresh()                # when matches existed; keep the
                                              # empty-table heartbeat otherwise
            cycles += 1
            await asyncio.sleep(self.a.interval)
        await self.mgr.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=12.0)
    ap.add_argument("--edge", type=float, default=0.04,
                    help="min edge (prob) to log an intended trade")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--max-spread", type=float, default=0.10,
                    help="widest pregame spread that can seed an anchor")
    ap.add_argument("--serve-share", type=float, default=0.5)
    ap.add_argument("--delay-points", type=int, default=5,
                    help="feed delay to defend against: qBID/qASK assume the "
                         "next N points all go one way from the shown score")
    a = ap.parse_args()
    os.makedirs(os.path.join(_ROOT, "tapes"), exist_ok=True)
    try:
        asyncio.run(Screener(a).run())
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
