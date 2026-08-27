"""Tennis theo generator: bid and offer come from the two one-point-ahead branches.

Registered as model_name = "tennis_branch".

WHY BID AND OFFER DIFFER, AND WHY THAT IS THE WHOLE POINT. Our scoreboard runs about a
point behind the participants with the fastest feeds: when we read 4-3 40-15 the point
in progress is often already decided, so the market we quote against is pricing either
40-30 or 5-3 0-0. Comparing one model price to that market compares two different
information sets.

So each side is valued at the branch where THAT side lost the unseen point, and offered
at the branch where it won:

    tracked side   bid_theo = min(w, l)        offer_theo = max(w, l)
    opponent       bid_theo = 1 - max(w, l)    offer_theo = 1 - min(w, l)

where w = model price if the tracked side wins the next point and l = if it loses.
Whichever way the point actually went, the true state is at least as good as the side we
priced against. The bid/offer gap IS the one-point bracket, so the model quotes a spread
exactly as wide as its own uncertainty about the point it cannot see - narrow at 0-30
(a 0.86c bracket at p10), wide at break point (16c+ at p99, median 3.42c).

Measured 26AUG26 over 14,915 in-play observations: 81% of raw model-vs-market
disagreements were SMALLER than the bracket, i.e. fully explained by the unseen point.
Those are exactly the trades this pricing declines to make.

WHERE THE NUMBERS COME FROM. poll_tennis.py is the market-data and model process: it
polls, fits (static / rolling / ewma2 / ewma4), and writes a tape carrying each variant
priced on both branches. This reads the tape's most recent in-play row. Two reasons not
to poll here: no duplicate GETs against a bucket shared with the live esports system,
and poll_tennis keeps its property of touching no order endpoint.

STALENESS IS A HARD STOP. If the tape has not been written within --max-age seconds the
poller is dead, paused, or the match ended, and the last branch prices are worthless.
This returns NOTHING for that ticker rather than a stale theo, so the quoter pulls its
quotes instead of resting on a price nobody is maintaining.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List

from base_model import BaseTheoGenerator

log = logging.getLogger(__name__)

TAPES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tapes")
DEFAULT_VARIANT = "ewma2"
MAX_TAPE_AGE_SEC = 30.0
# Minimum real game boundaries behind the fit before it may price anything.
#
# ImpliedModel fits p and q TO OBSERVED MARKET PRICES at boundaries - it carries no
# independent information about tennis. With 20 boundaries the structure genuinely
# interpolates between them. With one or two it does not: the fit is seeded from the
# current market price and the output is that same price with an unconstrained bracket
# around it.
#
# Measured 26AUG27, pollers attached mid-match ~15 minutes earlier:
#     SACLLA  bracket 24.88c    <- 1-2 boundaries, fit meaningless
#     LIURAD  bracket 14.23c
#     KALCOS  bracket  6.52c    <- more boundaries, plausible
# Those thin-fit markets quoted and traded 30 fills for a realised -$90.92, round-
# tripping flat. Config discipline did not prevent it and cannot: matches are added
# while the quoter runs. The gate belongs here, where the theo is produced.
MIN_BOUNDARIES = 8
MIN_THEO_C = 2.0
MAX_THEO_C = 98.0


class TennisBranchTheoGenerator(BaseTheoGenerator):
    """bid/offer theos in CENTS from the one-point-ahead bracket."""

    def __init__(self, client: Any, configs: List[Any] = None,
                 variant: str = DEFAULT_VARIANT, max_age: float = MAX_TAPE_AGE_SEC,
                 min_boundaries: int = MIN_BOUNDARIES):
        self.variant = variant
        self.max_age = max_age
        self.min_boundaries = min_boundaries
        self._cache: Dict[str, Any] = {}      # event -> (mtime, row, meta)
        super().__init__(client, configs)

    # ---------------------------------------------------------------- tape access
    def _meta(self, event: str):
        """(me_ticker, opp_ticker, n_boundaries) from the log poll_tennis writes."""
        p = os.path.join(TAPES, event + ".log.json")
        try:
            d = json.load(open(p))
            m = d.get("meta") or {}
            return m.get("me_ticker"), m.get("opp_ticker"), len(d.get("obs") or [])
        except Exception:
            return None, None, 0

    def _latest(self, event: str):
        """Most recent in-play tape row carrying both branches, or None if stale/absent."""
        p = os.path.join(TAPES, event + ".jsonl")
        try:
            st = os.stat(p)
        except OSError:
            return None
        age = time.time() - st.st_mtime
        if age > self.max_age:
            log.warning("TENNIS THEO | %s tape is %.0fs stale (>%.0fs) — no theo, quotes "
                        "should be pulled", event, age, self.max_age)
            return None
        cached = self._cache.get(event)
        if cached and cached[0] == st.st_mtime:
            return cached[1]
        row = None
        try:
            with open(p, "rb") as f:
                f.seek(max(0, st.st_size - 300_000))
                tail = f.read().decode("utf-8", "replace").splitlines()
            for line in reversed(tail):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == "live" and (r.get("ahead") or {}).get("win"):
                    row = r
                    break
        except Exception as e:
            log.warning("TENNIS THEO | %s tape unreadable: %s", event, e)
            return None
        self._cache[event] = (st.st_mtime, row)
        return row

    # ---------------------------------------------------------------- generation
    def _batch_generate(self, tickers: List[str],
                        dt_market_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for ticker in tickers:
            conf = self.configs.get(ticker)
            if not conf:
                continue
            event = ticker.rsplit("-", 1)[0]
            row = self._latest(event)
            if row is None:
                continue
            ah = row.get("ahead") or {}
            w = (ah.get("win") or {}).get(self.variant)
            l = (ah.get("lose") or {}).get(self.variant)
            if w is None or l is None:
                continue
            me_tick, _opp_tick, n_obs = self._meta(event)
            if not me_tick:
                log.warning("TENNIS THEO | %s has no me_ticker in its log — skipping", event)
                continue
            # THIN-FIT GATE. Below this the "theo" is just the market price with an
            # unconstrained bracket; quoting it trades on fit noise.
            if n_obs < self.min_boundaries:
                log.info("TENNIS THEO | %s only %d/%d boundaries — no theo yet "
                         "(fit too thin to price)", event, n_obs, self.min_boundaries)
                continue

            lo, hi = (w, l) if w <= l else (l, w)
            if ticker == me_tick:
                bid_p, off_p = lo, hi
            else:
                # opponent's price is the complement, so the bracket flips
                bid_p, off_p = 1.0 - hi, 1.0 - lo

            bid_c, off_c = bid_p * 100.0, off_p * 100.0
            # An inverted or degenerate bracket means the branch maths went wrong;
            # refuse rather than quote a crossed pair.
            if not (off_c >= bid_c):
                log.error("TENNIS THEO | %s inverted bracket bid=%.2f offer=%.2f — skipping",
                          ticker, bid_c, off_c)
                continue
            if bid_c < MIN_THEO_C or off_c > MAX_THEO_C:
                continue          # deep longshot / near-certain: the fit is least reliable here

            out[ticker] = {"bid_theo": round(bid_c, 2), "offer_theo": round(off_c, 2)}
            log.debug("TENNIS THEO | %s bid=%.2f offer=%.2f (bracket %.2fc, variant %s)",
                      ticker, bid_c, off_c, off_c - bid_c, self.variant)
        return out
