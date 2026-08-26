"""VAL swarm/GRID-desk signal detector — concrete, three validated tiers.

Target actor: the GRID FPS desk (Tremendous-Woodland) + its 8-wallet
fixed-clip swarm (very likely one operator splitting PnL — same birth
date, duplicated clips, same-second co-fires). Validated 26AUG06-07 on
SGEGX + FUREG + NRGNV complete tapes (scratchpad ws_fingerprint_study.py):

  CLIP_PRINT   a single print of an exact swarm clip size
               {3333, 2008, 1500, 1400, 4000, 4016, 6666} +-1 lot
               -> 98-100% desk/swarm precision (n=122, 2 FPs, both 1500).
  BIG_PRINT    >=1500 lots alone, or >=1000 with a same-direction trade
               event <=2s -> the SGEGX composite (86% informed-cluster).
  BURST        >=3 same-direction taker events >=500 lots within 1s
               -> 100% desk/swarm (n=50). Full cardinality usually needs
               the TAPE (tier 3) because the WS last_trade_price channel
               SUPPRESSES same-price prints (measured: only ~25% of tape
               taker events >=300 produce a WS event on NRGNV).
  WALLET       tape rows from watchlist addresses (data-api only).

FEED REALITY (measured, do not redesign around hopes):
  * WS `last_trade_price` is a price-change notification, not a trade
    feed. It undercounts heavily but has ~150ms latency and its size
    field is the FULL taker size when it does fire (clip-exact matches
    confirmed). Tiers CLIP_PRINT/BIG_PRINT run here.
  * data-api /trades per market is complete and wallet-attributed but
    stamps BLOCK time ~2.3s behind the wire; live polling nets ~2.5-4s
    latency. Tiers BURST/WALLET run here via `on_tape_rows`.
  * The Kalshi echo this fires into builds +1.9c@10s -> +4.3c@30s ->
    +5.3c@60s (81% right-dir, n=21), so even tape-lagged fires capture
    most of the move. There is no earlier tell than the first print.
  * Book-depth deltas at 1s granularity do NOT work as a trigger (MM
    refills cancel the drop within the second); a raw-delta-message
    version is future work.

Pure logic, no I/O, never raises — same contract as map_taker_detector.
"""
from collections import deque

CLIPS = (3333.0, 2008.0, 1500.0, 1400.0, 4000.0, 4016.0, 6666.0)
CLIP_TOL = 1.0
BIG_ALONE = 1500.0      # single WS print that fires unaccompanied
BIG_ECHOED = 1000.0     # WS print that fires with a same-dir event <=2s
ECHO_S = 2.0
BURST_N = 3             # tape: >=N same-dir taker events...
BURST_SZ = 500.0        # ...each >= this many lots...
BURST_S = 1.0           # ...inside this window  (100% precision measured)
# PACK_EAT (26AUG08 LoL signature study): a >=500-lot taker eating >=3 book
# levels with >=2 same-dir multi-level companions within +-2s. Contains the
# Grit/Tarot pair 72% of fires; Kalshi echo +5.7c@30s, 69% right (n=27,
# in-sample thresholds — SHADOW-VALIDATE before arming).
PACK_SZ, PACK_LV = 500.0, 3
PACK_COMP_SZ, PACK_COMP_LV, PACK_COMP_N, PACK_S = 500.0, 2, 2, 2.0
# ORACLE_TAKE (26AUG09 SRSEN 00:02:25 exhibit): in a thin late-game book LOT
# COUNT is noise — the oracle expresses conviction as LEVELS EATEN. Grit swept
# 5 levels with only 287 lots (under EVERY size floor) 1s after Tarot swept 2;
# SEN 72->1, Kalshi series 91->64, and we fired nothing. The signal is a NAMED
# oracle wallet acting as an AGGRESSOR, size-independent. Gate on identity +
# aggression (levels eaten), with a dust floor to skip 1-lot noise.
ORACLE_LV = 2           # levels a named oracle must eat to count as aggressive
ORACLE_DUST = 25.0      # ...above this many lots (skip 1-lot round-trip dust)
COOL_S = 20.0           # one signal per leg per this many seconds
PIN_LO, PIN_HI = 5.0, 95.0
# conviction-class labels: prints from these wallets qualify for the
# WALLET_ECHOED / DUAL_WALLET fire tiers (china_whale deliberately not here)
DESK_LABELS = ("grid_desk_val", "grid_desk_cs", "swarm",
               "lol_grit", "lol_tarot")

# ORACLE_TAKE is sport-scoped: a wallet is only an "oracle" in the domain it
# has edge in. The VAL desk/swarm are NEGATIVE outside VAL (reference_val_
# china_whale_eac2), so a swarm wallet touching a LoL leg is NOT a signal —
# the 26AUG09 SRSEN 00:00:54 swarm-buys-SEN-at-69 wrong-way fire is exactly
# what this scoping suppresses. is_desk (the >=800 echo/dual tiers) stays
# broad; only ORACLE_TAKE (size-independent) demands on-domain identity.
SPORT_ORACLES = {
    "KXVALORANTGAME": ("grid_desk_val", "grid_desk_cs", "swarm"),
    "KXLOLGAME": ("lol_grit", "lol_tarot"),
}

WATCHLIST = {
    "0x86df6ce94c9263d09a6052b43d652adb7b51d902": "grid_desk_val",
    "0x0063b23cdeb43166d6c0246c05baaf9b9bd72dd2": "grid_desk_cs",
    "0x2d395d11014415644fe9a8599fe050e7f3a06053": "swarm",
    "0x3471a897e56a8d3621ca79af87dae4325977f17e": "swarm",
    "0xec981ed70ae69c5cbcac08c1ba063e734f6bafcd": "swarm",
    "0xf3ef6ac0510c9eabc42fbb146b5c3c64edc1ca6b": "swarm",
    "0x82dca3851445e005b8f21a2814db1439189f5700": "swarm",
    "0xb6ce76828a51b382f7e88cbb9baa0f03f9c17a7d": "swarm",
    "0x60ec17443af511a21945f01430e61c803465f7b0": "swarm",
    "0xfdc0bd67fbd71aa8edd00121eb2a7fcbddc34b85": "swarm",
    # LoL oracles (26AUG08 SRSEN 23:02:10 exhibit: Grit ate 2,516 lots
    # across 7 levels buying SR in a fight-second; pack rule missed it
    # because the fight flow was two-sided). Wallet tiers cover them now.
    "0x31864feb9d25dee93728c6225ba891530967e9ca": "lol_grit",
    "0x10ef48324b3ebc06d9137794c70285e8e37a7d7b": "lol_tarot",
    # independent VCT-China whale, NOT the GRID desk (found 26AUG07):
    # born 15Jul, +$196k/24d, +$215k of it in China VAL series+maps,
    # NEGATIVE in Pacific/LoL/handicaps; mixed maker/taker; edge is
    # match-level conviction (holds to settle), instant markout only
    # ~+1.7c -> experimental tier, shadow evidence only.
    "0xeac22222ef8d2be457c00101a1a4e4925ec5dddc": "china_whale",
}


def canon(leg):
    """`...|M2B` is the complement token of `...|M2` (map_taker_detector)."""
    if leg.endswith("B") and "|M" in leg:
        return leg[:-1], True
    return leg, False


class Signal:
    __slots__ = ("ts", "map_key", "tier", "dir", "px", "size", "detail",
                 "src_leg")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        return (f"<Signal {self.tier} {self.map_key} dir={self.dir:+d} "
                f"px={self.px:.1f} sz={self.size:.0f} {self.detail}>")


class _LegState:
    __slots__ = ("ws", "tape", "last_sig", "last_info", "pending")

    def __init__(self):
        self.ws = deque(maxlen=200)      # (ts, dir, px, sz) WS trade events
        self.tape = deque(maxlen=400)    # (ts, dir, px, sz, wallet, lv)
        self.last_sig = 0.0     # fire-class tiers
        self.last_info = 0.0    # WALLET (shadow-only) — separate window so
                                # info signals never cooldown-block a fire
        self.pending = []       # unechoed desk prints >=800 awaiting a
                                # cross-batch echo: [ts, d, px, sz, label]


class ValSwarmDetector:
    """Feed WS trade events and (separately) polled tape taker events.

    on_ws_trade(leg, ts, side, price_c, size)      -> Signal | None
    on_tape_rows(map_key, rows)                    -> [Signal, ...]
        rows: iterable of dicts with ts, dir (+1 buys canonical A),
        px (canonical cents), sz, wallet — TAKER events, caller-built
        (one per txhash, taker = row whose size ~ sum of the others).
    """

    DEFAULT_TIERS = ("CLIP_PRINT", "BIG_PRINT", "WALLET", "BURST")

    def __init__(self, sports=("KXVALORANTGAME",), cool_s=COOL_S, tiers=None):
        self._st = {}
        self._sports = set(sports) if sports else None
        self._cool = cool_s
        self._tiers = set(tiers) if tiers else set(self.DEFAULT_TIERS)
        # on-domain oracle labels for ORACLE_TAKE (all desks if unscoped)
        if self._sports is None:
            self._oracles = set(DESK_LABELS)
        else:
            self._oracles = set()
            for sp in self._sports:
                self._oracles |= set(SPORT_ORACLES.get(sp, ()))

    def _want(self, key):
        if self._sports is not None:
            return key.split("-")[0] in self._sports
        return True

    def state(self, key):
        s = self._st.get(key)
        if s is None:
            s = self._st[key] = _LegState()
        return s

    def _emit(self, s, key, ts, tier, d, px, sz, detail, src_leg=""):
        info = tier == "WALLET"
        last = s.last_info if info else s.last_sig
        if ts - last <= self._cool:
            return None
        if not (PIN_LO <= px <= PIN_HI):
            return None
        if info:
            s.last_info = ts
        else:
            s.last_sig = ts
        return Signal(ts=ts, map_key=key, tier=tier, dir=d, px=px, size=sz,
                      detail=detail, src_leg=src_leg)

    # -------------------------------------------------- WS tiers (fast)
    def on_ws_trade(self, leg, ts, side, price_c, size):
        try:
            key, flip = canon(leg)
            if not self._want(key):
                return None
            t = float(ts)
            px = float(price_c)
            sz = float(size or 0)
            if sz <= 0:
                return None
            d = 1 if str(side).upper() == "BUY" else -1
            if flip:
                px = 100.0 - px
                d = -d
            s = self.state(key)
            s.ws.append((t, d, px, sz))

            if "CLIP_PRINT" in self._tiers and \
                    any(abs(sz - c) <= CLIP_TOL for c in CLIPS):
                sig = self._emit(s, key, t, "CLIP_PRINT", d, px, sz,
                                 f"clip={sz:.0f}", leg)
                if sig:
                    return sig
            if "BIG_PRINT" not in self._tiers:
                return None
            if sz >= BIG_ALONE:
                sig = self._emit(s, key, t, "BIG_PRINT", d, px, sz,
                                 "alone>=1500", leg)
                if sig:
                    return sig
            if sz >= BIG_ECHOED:
                echo = [x for x in s.ws
                        if x is not s.ws[-1] and x[1] == d
                        and 0 <= t - x[0] <= ECHO_S]
                if echo:
                    return self._emit(s, key, t, "BIG_PRINT", d, px, sz,
                                      f"echoed n={len(echo)}", leg)
            return None
        except Exception:
            return None

    # ------------------------------------------- tape tiers (complete)
    def on_tape_rows(self, map_key, rows):
        out = []
        try:
            if not self._want(map_key):
                return out
            s = self.state(map_key)
            seen = {(r[0], r[3], r[4]) for r in s.tape}
            fresh = []
            for r in rows:
                tup = (float(r["ts"]), int(r["dir"]), float(r["px"]),
                       float(r["sz"]), str(r.get("wallet", "")),
                       int(r.get("lv", 0) or 0))
                if (tup[0], tup[3], tup[4]) in seen:
                    continue
                s.tape.append(tup)
                fresh.append(tup)
            for t, d, px, sz, w, lv in fresh:
                label = WATCHLIST.get(w)
                is_desk = label in DESK_LABELS
                # retro echo (26AUG08 23:53 SRSEN miss): an oracle print's
                # echo often lands in the NEXT poll batch. Any fresh >=300
                # same-dir row consumes a pending unechoed desk print.
                s.pending = [p for p in s.pending if t - p[0] <= 2.5]
                if "WALLET_ECHOED" in self._tiers and sz >= 300.0:
                    for p in list(s.pending):
                        if p[1] == d and 0 < t - p[0] <= 2.5:
                            s.pending.remove(p)
                            sig = self._emit(s, map_key, p[0], "WALLET_ECHOED",
                                             p[1], p[2], p[3],
                                             f"{p[4]} retro-echo")
                            if sig:
                                out.append(sig)
                            break
                # CLIP first: a watchlist wallet printing an exact clip is
                # the swarm signature — must not be shadowed by WALLET.
                if "CLIP_PRINT" in self._tiers and \
                        any(abs(sz - c) <= CLIP_TOL for c in CLIPS):
                    sig = self._emit(s, map_key, t, "CLIP_PRINT", d, px, sz,
                                     f"clip={sz:.0f}(tape)")
                    if sig:
                        out.append(sig)
                        continue
                # DUAL_WALLET (26AUG08 SENKRU bonus-round lesson): two
                # desk/swarm prints >=800 same-dir within 15s — tonight's
                # desk echoes at 10-15s tempo (Glist 1,227@67.5 then
                # Woodland 1,246@76 twelve seconds later, px -> 83+).
                if "DUAL_WALLET" in self._tiers and is_desk and sz >= 800.0:
                    prior = [x for x in s.tape
                             if x[:5] != (t, d, px, sz, w) and x[1] == d
                             and x[3] >= 800.0 and 0 <= t - x[0] <= 15.0
                             and WATCHLIST.get(x[4]) in DESK_LABELS]
                    if prior:
                        sig = self._emit(s, map_key, t, "DUAL_WALLET", d,
                                         px, sz,
                                         f"{label} pair={len(prior)}")
                        if sig:
                            out.append(sig)
                            continue
                # WALLET_ECHOED (26AUG08 SENKRU M2 pistol-round lesson):
                # desk/swarm print >=800 WITH >=1 same-dir taker >=300 within
                # +-2.5s = conviction (echo screens out round-trip scalps,
                # which print alone). Woodland 1,246 + Pagan 1,054 @ +2s
                # before a 33->90 grind is the canonical exhibit.
                if "WALLET_ECHOED" in self._tiers and is_desk \
                        and sz >= 800.0:
                    echo = [x for x in s.tape
                            if x[:5] != (t, d, px, sz, w) and x[1] == d
                            and x[3] >= 300.0 and abs(t - x[0]) <= 2.5]
                    if echo:
                        sig = self._emit(s, map_key, t, "WALLET_ECHOED", d,
                                         px, sz,
                                         f"{label} echo={len(echo)}")
                        if sig:
                            out.append(sig)
                            continue
                    else:
                        s.pending.append([t, d, px, sz, label])
                if "WALLET" in self._tiers and label and sz >= BURST_SZ:
                    sig = self._emit(s, map_key, t, "WALLET", d, px, sz,
                                     label)
                    if sig:
                        out.append(sig)
                        continue
                # ORACLE_TAKE: a named oracle sweeping >=ORACLE_LV levels is
                # full conviction regardless of lots — the size-independent
                # catch for thin late-game fights the >=500 floors miss.
                if "ORACLE_TAKE" in self._tiers and label in self._oracles \
                        and lv >= ORACLE_LV and sz >= ORACLE_DUST:
                    sig = self._emit(s, map_key, t, "ORACLE_TAKE", d, px, sz,
                                     f"{label} lv={lv}")
                    if sig:
                        out.append(sig)
                        continue
                if "PACK_EAT" in self._tiers and sz >= PACK_SZ \
                        and lv >= PACK_LV:
                    comp = [x for x in s.tape
                            if x[:5] != (t, d, px, sz, w) and x[1] == d
                            and x[3] >= PACK_COMP_SZ and x[5] >= PACK_COMP_LV
                            and abs(t - x[0]) <= PACK_S]
                    if len(comp) >= PACK_COMP_N:
                        sig = self._emit(
                            s, map_key, t, "PACK_EAT", d, px, sz,
                            f"lv={lv} comp={len(comp)}")
                        if sig:
                            out.append(sig)
                            continue
                if "BURST" in self._tiers and sz >= BURST_SZ:
                    burst = [x for x in s.tape if x[1] == d
                             and x[3] >= BURST_SZ
                             and 0 <= t - x[0] <= BURST_S]
                    if len(burst) >= BURST_N:
                        sig = self._emit(
                            s, map_key, t, "BURST", d, px,
                            sum(x[3] for x in burst),
                            f"n={len(burst)} in {BURST_S:.0f}s")
                        if sig:
                            out.append(sig)
            return out
        except Exception:
            return out
