"""Track p+q fast, hold p-q still. The variant that stops the model leaning one way.

WHY THIS EXISTS. ImpliedModel fits p and q jointly to boundary prices. Measured across
115 matches / 2,475 boundaries, that is the wrong split of effort, because the two
parameters are not equally visible in a price:

    p+q  (overall strength)   2.91c of match price per 0.01     <- identified
    p-q  (serve edge)         0.22c per 0.01                    <- ~13x weaker

A least-squares fit spends its freedom where the residual is cheap, so p-q wanders (it
reached 0.039 on 26AUG25CHWPAR, implying a WTA player holds 52.5% while she was 6-for-6
on serve) while p+q lags behind a moving market. Lag in p+q is the expensive failure: it
is one-signed, so it does not average out, it accumulates as inventory. On CHWPAR, 73 of
75 fills were the same direction and the position reached 677 contracts one way.

So: pin the split, track the sum, and track it FAST.

    scheme              MAE    bias   maxrun   run>=5      (115 matches, no look-ahead)
    halflife 0.5       2.42   -0.12      4.5      43%
    halflife 2 (old)   2.45   -0.28      6.1      78%
    equal weight       2.82   -0.58      8.5      88%

halflife 0.5 beats halflife 2 on every column - there is no accuracy given up for the
decorrelation. `maxrun` is the longest same-signed error streak in a match, and it is the
metric that matters: a 2.4c error that flips sign each boundary is noise you trade
around, a 2.4c error that holds its sign for six boundaries is a directional bet.

DO NOT ADD TREND EXTRAPOLATION. Lag-1 autocorrelation of the implied-sum increments is
-0.313 - the series MEAN-REVERTS. Holt (a=.8, b=.5) was tested: it scores well on run
length only by being noisy, and pays for it with the second-worst MAE (2.83). Plain
last-value dominates it on both. The sum is not trending; do not model it as if it were.

WHAT p-q SHOULD EVENTUALLY COME FROM. Not prices - they barely constrain it. Holds and
breaks identify it directly and we already observe them (CHWPAR: PAR 6/6, CHW 6/8). Until
that is wired, the tour prior is a better estimate than a fit to prices, because a fit to
prices is fitting noise in this direction.
"""
import os
from implied_model import ImpliedModel, _model

DEFAULT_HALFLIFE = 0.5
S_LO, S_HI = 0.60, 1.35          # p+q bounds; outside this a boundary price is unusable


class SumTracker:
    """p+q on a fast EWMA of per-boundary implied values; p-q pinned.

    Mirrors the ImpliedModel surface used by poll_tennis (`observe`, `price`, `.p`, `.q`,
    `.obs`) so it can sit alongside the other variants without special-casing.
    """

    def __init__(self, best_of=3, split=0.20, halflife=DEFAULT_HALFLIFE, final_set_tb=7,
                 first_server="me"):
        self.best_of, self.final_set_tb = best_of, final_set_tb
        # State normalisation (server -> i_serve, and the who-serves-which-game pattern)
        # is delegated to ImpliedModel rather than reimplemented. Getting the serving
        # pattern subtly different from the rest of the codebase would show up as a small
        # persistent bias that looks exactly like the one we are trying to remove.
        self._norm = ImpliedModel(best_of=best_of, first_server=first_server,
                                  final_set_tb=final_set_tb, warn_split=False)
        self.d = float(split)                 # p-q, held fixed
        self.halflife = float(halflife)
        self._dec = 0.0 if self.halflife <= 0 else 0.5 ** (1.0 / self.halflife)
        self._ew = 0.0                        # weighted sum of implied p+q
        self._w = 0.0                         # total weight
        self.obs = []                         # (state, price) - kept for parity/inspection
        self.split_violation = False

    # ------------------------------------------------------------------ parameters
    @property
    def s(self):
        return None if self._w <= 0 else self._ew / self._w

    @property
    def p(self):
        s = self.s
        return None if s is None else (s + self.d) / 2.0

    @property
    def q(self):
        s = self.s
        return None if s is None else (s - self.d) / 2.0

    # ------------------------------------------------------------------ mechanics
    def _price_at(self, s, state):
        return _model((s + self.d) / 2.0, (s - self.d) / 2.0,
                      self._state(state), self.best_of, self.final_set_tb)

    def _state(self, state):
        st = {k: v for k, v in state.items() if not k.startswith("_")}
        st.setdefault("points_me", 0)
        st.setdefault("points_opp", 0)
        return self._norm._state(**st)

    def implied_sum(self, market_prob, **state):
        """Invert one price for the p+q it implies, with p-q held. None if unreachable.

        Bisection, not Newton: the map s -> price is monotone but flattens hard near 0 and
        1, where a derivative step overshoots the bracket and returns nonsense.
        """
        lo, hi = S_LO, S_HI
        if not (0.0 < market_prob < 1.0):
            return None
        target = float(market_prob)
        if self._price_at(lo, state) > target or self._price_at(hi, state) < target:
            return None                       # price outside what any p+q can produce here
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if self._price_at(mid, state) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def observe(self, market_prob, weight=1.0, **state):
        """Fold one boundary in. Unreachable prices are SKIPPED, not clamped — clamping
        would silently peg p+q to a bound and freeze the tracker there."""
        isum = self.implied_sum(market_prob, **state)
        if isum is None:
            return self
        if state.get("server") is not None:
            n = self._state(state)
            self._norm._register_first_server(n["sets_me"], n["sets_opp"],
                                              n["games_me"], n["games_opp"], n["i_serve"])
        self._ew = self._ew * self._dec + isum * weight
        self._w = self._w * self._dec + weight
        self.obs.append((self._state(state), float(market_prob)))
        return self

    def price(self, **state):
        """(model probability for 'me', half-width). Half-width is 0.0: this estimator has
        no covariance to propagate — its uncertainty lives in the one-point bracket."""
        s = self.s
        if s is None:
            raise ValueError("SumTracker has no observations yet")
        return self._price_at(s, state), 0.0


def tour_split(prior):
    """The split to pin. Same value poll_tennis already resolves per tour."""
    return float(prior)
