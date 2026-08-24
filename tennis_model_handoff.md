# Tennis in-play model — model, code, and results from the Sherif vs Oliynykova test

Self-contained handoff. Paste this into a new chat to pick up where we left off.

---

## 1. The model

Assumption: every point is independent. A player wins a point **on their own serve** with constant
probability `p` and **on return** with constant probability `q`. No fatigue, no momentum. Two
parameters for the whole match.

### Game (from a point score)

With `w = p` if serving, `q` if returning, and points as integers (3–3 = deuce):

```
k_w = w² / (w² + (1-w)²)                    # P(win from deuce)

g_w(x,y) = 1                                 if x≥4 and x−y≥2
         = 0                                 if y≥4 and y−x≥2
         = k_w                                if x = y ≥ 3          (deuce)
         = w + (1−w)·k_w                      if x = y+1 ≥ 4        (my advantage)
         = w·k_w                              if y = x+1 ≥ 4        (their advantage)
         = w·g_w(x+1,y) + (1−w)·g_w(x,y+1)   otherwise
```

From 0–0 this has the closed form

```
G(w) = w⁴ · (15 − 4w − 10w² / (1 − 2w + 2w²))
```

Equivalently `G(w) = P(Bin(6,w) ≥ 4) + P(Bin(6,w) = 3)·k_w`. Checks: G(0.5)=0.5, G(0.6)≈0.736,
G(0.65)≈0.830. Note `q` never enters a service game; it drives return games, where `G(q)` is the
break probability.

### Tiebreak

Trick that removes the serving pattern: over the first 12 points each player serves exactly 6
regardless of who serves first. So with `X = Bin(6,p) + Bin(6,q)`:

```
D = pq / (pq + (1−p)(1−q))                  # P(win from 6–6, 7–7, 8–8, … — all equal)
TB(p,q) = P(X ≥ 7) + P(X = 6)·D
```

**Serving first in a tiebreak makes no difference**: `TB(p,q) = 1 − TB(1−q, 1−p)` exactly (verified
symbolically and against brute-force enumeration of every point sequence). Only one tiebreak
function is needed. For a mid-tiebreak point score, recurse as in the game, flipping the server
after every odd-numbered point, with `D` as the 6–6 value.

### Set (from a game score)

With `s` = server of the next game:

```
S(a,b,s) = 1                                             if a≥6 and a−b≥2
         = 0                                             if b≥6 and b−a≥2
         = TB(p,q)                                       if a = b = 6
         = G_s·S(a+1,b,s̄) + (1−G_s)·S(a,b+1,s̄)          otherwise
```

Closed form from 0–0, with `Y = Bin(5,G(p)) + Bin(5,G(q))`:

```
F = G(p)G(q) + [G(p)(1−G(q)) + (1−G(p))G(q)]·TB(p,q)     # value of 5–5
P(set) = P(Y ≥ 6) + P(Y = 5)·F
```

### Any mid-game state

The current game is a weighted coin flip between the two game scores it can produce:

```
P = g_w(x,y)·S(a+1,b,s̄) + (1 − g_w(x,y))·S(a,b+1,s̄)
```

### Facts worth remembering

- **Next server often doesn't matter.** At a game score with `a+b` even, both players serve the same
  number of the remaining games, so the price is identical either way (3–1 gives 0.898 whoever serves
  next). It only matters when `a+b` is odd (3–0: 0.971 if you serve next, 0.922 if they do).
- **Match prices are almost a pure function of `p+q`.** At every state checked, the gradient of the
  match price w.r.t. `(p,q)` points within a few degrees of the (1,1) direction. Consequence: the
  quality gap `p+q` is pinned almost immediately (±0.002 from two prices), while the split `p−q` is
  loose (±0.15). This is the Klaassen–Magnus result. It means match-winner prices are a blunt
  instrument for the split — but also that you don't need the split to price the match.
- **A hold's value depends on the hold rate.** From a pre-match 0.70, a hold is worth +5.4 cents to a
  50%-holder and +0.5 to a 98%-holder; the jump and the drop are in the ratio (1−G(p)) : G(p).
  Only the *average* of the two branches is fixed by the pre-match price:
  `M(0–0) = G(p)·M(1–0) + (1−G(p))·M(0–1)`.
- **Two holds tell you nothing.** The price at 1–1, 2–2, … returns to the 0–0 value regardless of the
  split; two games have just been consumed.
- **Uncertainty propagates weakly to prices.** With prices known to ±1 cent, `p` is only pinned to
  about ±0.03 after five games — but the induced uncertainty on any *match* price is ±0.5 cent. The
  split matters a lot for **game**-level markets (P(hold from 15–40) ranges 0.28–0.38 over the same
  interval) and for tiebreaks, which are pure point-level quantities.

---

## 2. Code

Four files, kept together:

| file | what it does |
|---|---|
| `tennis_model.py` | `game_win_prob`, `G`, `tiebreak_win_prob`, `set_from_games`, `set_win_prob`, `match_win_prob` — any state, memoised |
| `market_implied.py` | invert observed prices → `(p,q)`: `implied_pq`, `fit_uncertainty`, `model_price_ci`, `invert_G`, `normalise` (strip vig) |
| `implied_model.py` | `ImpliedModel` — incremental fitter; `observe()` each price, `price()`, `report()`, `residuals()` |
| `live.py` | CLI logger for one match; writes `live_log.json`, prints prediction-vs-market table each time |

Usage:

```python
m = ImpliedModel(best_of=3, first_server='me', split_prior=0.14)  # 0.28 men, 0.14 women
m.observe(0.465)                                    # pre-match
m.observe(0.365, games_me=0, games_opp=1)           # server derived from alternation
m.set_first_server(0, 1, 'opp')                     # who serves game 1 of set 2
m.observe(0.195, sets_me=0, sets_opp=1, games_me=0, games_opp=0)
m.price(games_me=3, games_opp=4, points_me='15', points_opp='40')   # → (prob, ±95%)
m.report(current=dict(games_me=3, games_opp=4))
```

```bash
python3 live.py init --first-server me --pregame 0.465 --best-of 3 --split-prior 0.14
python3 live.py add 0 1 0.365                  # my_games opp_games my_price
python3 live.py set-server 0 1 opp             # first server of set 2
python3 live.py add 0 0 0.195 --sets-me 0 --sets-opp 1
python3 live.py report
```

Notes: `split_prior` is a weak prior on `p−q` (SD 0.30) that only bites when the observed prices are
near-parallel (e.g. pre-match + 1–1) and would otherwise leave the split unidentified; pass
`split_prior_sd=5` for a pure prior-free solve. Use vig-free midpoints. Feed **game-boundary** prices
to the fit; mid-game prices are for comparison, not fitting (they're the noisiest).

---

## 3. Test: Sherif vs Oliynykova, WTA Monterrey R32, hard, best of 3, 23 Aug 2026

Kalshi match-winner market, 1-cent resolution. "Sherif" = the tracked player; scores are
Sherif–Oliynykova. Sherif served game 1 of set 1; Oliynykova served game 1 of set 2.

Pre-match price wandered 0.50 → 0.48 → 0.55 → 0.45 → **0.465** with no tennis played. Logged at 0.465.

**Clean line** = fixed model fitted on the first in-play price alone: **p = 0.599, q = 0.394**
(hold 73.4%, break 25.2%, tiebreak 48.9%). **Pred** = running fit on all *earlier* prices.
All errors in cents.

| state | market | pred | err | clean | vs clean | mkt move | model move |
|---|---|---|---|---|---|---|---|
| pre-match | 0.465 | – | – | 0.464 | +0.1 | – | – |
| 0–1 | 0.365 | 0.380 | −1.5 | 0.364 | +0.1 | −10.0 | −10.0 |
| 0–2 | 0.305 | 0.331 | −2.6 | 0.330 | −2.5 | −6.0 | −3.4 |
| 1–2 | 0.375 | 0.344 | +3.1 | 0.355 | +2.0 | +7.0 | +2.4 |
| 1–3 | 0.320 | 0.316 | +0.4 | 0.317 | +0.3 | −5.5 | −3.7 |
| 2–3 | 0.365 | 0.350 | +1.5 | 0.341 | +2.4 | +4.5 | +2.4 |
| 2–4 | 0.325 | 0.300 | +2.5 | 0.299 | +2.6 | −4.0 | −4.2 |
| 3–4 | 0.355 | 0.341 | +1.4 | 0.321 | +3.4 | +3.0 | +2.2 |
| 3–5 | 0.285 | 0.275 | +1.0 | 0.271 | +1.4 | −7.0 | −5.0 |
| **set 2** 0–0 | 0.195 | 0.221 | −2.6 | 0.227 | −3.2 | −9.0 | −4.5 |
| 1–0 | 0.335 | 0.286 | +4.9 | 0.326 | +0.9 | +14.0 | +9.9 |
| 1–1 | 0.235 | 0.228 | +0.7 | 0.227 | +0.8 | −10.0 | −9.8 |
| 1–2 | 0.185 | 0.185 | −0.0 | 0.191 | −0.6 | −5.0 | −3.7 |
| 2–2 | 0.245 | 0.229 | +1.6 | 0.228 | +1.7 | +6.0 | +3.8 |
| 2–3 | 0.185 | 0.181 | +0.4 | 0.187 | −0.2 | −6.0 | −4.2 |
| 3–3 | 0.295 | 0.231 | +6.4 | 0.229 | +6.6 | +11.0 | +4.3 |
| 3–4 | 0.215 | 0.179 | +3.6 | 0.180 | +3.5 | −8.0 | −4.9 |
| 4–4 | 0.285 | 0.237 | +4.8 | 0.230 | +5.5 | +7.0 | +5.0 |
| 4–5 | 0.195 | 0.176 | +1.9 | 0.170 | +2.5 | −9.0 | −6.1 |

**Final: Oliynykova won 6–3, 6–4.** Set 1 finished 3–6. In set 2 Sherif served at 4–5 to stay in the
match, reached deuce (market 0.175 vs model 0.160–0.169, one of the closest prints of the match), and
was broken. The match therefore ended at the state after the last logged price, so 0.195 at 4–5 was the
final tradeable quote; the model's 0.176 for that state was 1.9c low, in line with the persistent
positive bias.

Mid-game prints observed (not fitted), all roughly **one point ahead of the scoreboard**:

| state | market | model | model's price one point later |
|---|---|---|---|
| 2–3 set 1, Oliy serving, 0–15 | 0.405 | 0.383 | 0.412 (at 0–30) |
| 4–3 set 1, Oliy serving, 30–15 | 0.335 | 0.300–0.318 | 0.330 (at 30–30) |
| 2–2 set 2, Oliy serving, 0–15 | 0.295 | 0.256 | 0.288 (at 0–30) |
| 4–5 set 2, Sherif serving, deuce | 0.175 | 0.160–0.169 | — (within ~1c; fine) |

### Findings

1. **The two-parameter model tracks a real market to ~3 cents.** Mean error +1.4c, rms 2.7c over 19
   prices spanning two sets, from a `(p,q)` fitted on a *single* game.
2. **Every move was amplified ≈1.5×.** Median |market move| / |model move| = 1.47 (1.59 on games
   Sherif won, 1.44 on games she lost). Most robust finding in the log; reproduced at point level.
3. **Overshoots largely reverted.** Regressing change-in-error on error gives slope **−0.77**: about
   three quarters of each deviation came back by the next boundary. Signature of over-reaction rather
   than re-rating → the tradeable version of "market is wrong" (fade the post-game move).
4. **The result is consistent with the fixed model, not the market's late enthusiasm.** The market
   re-rated Sherif upward through set 2 (see below) and she lost the set anyway. That is one sample and
   proves nothing on its own, but it is the direction the reversion finding predicts.
5. **A genuine level shift late in set 2.** Bias grew from +1.2c (set 1) to +1.8c (set 2); last four
   prints ran +3.5 to +6.6c. Rolling four-price fits imply a fresh-match price for Sherif rising
   0.45 → 0.53 → 0.56 by 3–3 and 4–4. So the market did re-rate her upward on top of the noise.
6. **Underdog premium (Kalshi overpricing underdogs): partly supported, one clear counterexample.**
   14 of 18 prints above the clean line, and a constant 7% compression toward 0.5 fits the residuals
   about as well as anything. **But** the deepest-underdog print of the match (0.195, immediately
   after losing set 1) came in 2.6–5c *below* every model reading — the opposite of a premium. Two
   surviving readings: a premium that operates in-set but not at set boundaries, or no premium and
   just the set-2 re-rating. One match cannot separate them.
7. **Caveat.** The pre-match price moved 7 cents on no tennis. Against that noise floor, a 1–3 cent
   in-play bias is small: the fixed-probability model was arguably more internally consistent than the
   market it was measuring.

### Correction to keep in the record

At 3–4 in set 1 I described game 8 as "Oliynykova serving for the set". Wrong — a hold made it 3–5,
not 3–6. The predicted price for that state (0.275) was right; the label and the set-end premium
comparison (which belonged to game 9) were not.

---

## 4. Protocol for the next match

1. Tell me: the two players, **best of 3 or 5**, **who serves game 1**, tour (men/women, for the split
   prior), and the **pre-match vig-free mid**.
2. Then one line per game: score and price, e.g. `1-0 0.55`. State scores consistently from one
   player's point of view (say which) — mid-match ambiguity cost us a re-label in this run.
3. At each set boundary say **who serves game 1 of the next set** (the player who did *not* serve the
   last game of the previous set; after a tiebreak, whoever received its first point).
4. Optional and useful: mid-game prints (`0-30 0.29`) and how each game went (to 15, deuce, etc.).
   Mid-game prints are where the over-reaction is most visible.
5. What gets reported back each time: market vs the model's prediction for that state (fitted only on
   earlier prices), the error in cents, the refitted `(p,q)`, and the two branches for the next game.

### Hypotheses to test next, in priority order

- **Move amplification ≈1.5×.** Prediction: |market move| / |model move| has median >1.3 again.
  Strongest and simplest claim from this match.
- **Mean reversion of overshoots.** Prediction: regression slope of change-in-error on error is
  around −0.5 to −1. If it holds across matches, fading post-game moves is the trade.
- **Underdog premium.** Needs a match where the underdog price goes low *and* crosses set boundaries.
  Watch specifically whether the bias flips sign at set boundaries as it did here.
- **Re-rating vs fixed `(p,q)`.** Diagnostic: rolling-window fits. If the implied fresh-match level
  drifts monotonically, the market is learning about the players and no fixed `(p,q)` will hold — that
  is the real limit of this model, and it showed up in set 2 here.
- **Sharper identification of the split.** Match-winner prices barely identify `p−q`. If a
  game-winner or total-games market is quoted, `invert_G(price)` gives `p` or `q` directly and pins
  the split in one observation instead of twenty.
