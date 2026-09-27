"""Independent verification of the Tennis-Model math (no imports from the repo except the model)."""
import sys, random, math
sys.path.insert(0, '/Users/bradleyguan/Documents/Coding/Tennis-Model')
from tennis_model import game_win_prob, G, tiebreak_win_prob, set_from_games, set_win_prob, match_win_prob

random.seed(12345)
fails = []
def chk(name, a, b, tol):
    ok = abs(a - b) <= tol
    print(f"{'OK ' if ok else 'FAIL'} {name:52s} {a:.6f} vs {b:.6f}  (d={a-b:+.2e})")
    if not ok: fails.append(name)

print("=== 1. G closed form vs recursion ===")
for w in (0.3, 0.5, 0.6, 0.65, 0.75, 0.9):
    chk(f"G({w}) == game_win_prob({w},0,0)", G(w), game_win_prob(w, 0, 0), 1e-12)

print("\n=== 2. G vs binomial identity  P(Bin(6,w)>=4) + P(Bin(6,w)=3)*k ===")
def binom(n, k): return math.comb(n, k)
for w in (0.4, 0.6, 0.65):
    k = w*w/(w*w + (1-w)**2)
    alt = sum(binom(6,i)*w**i*(1-w)**(6-i) for i in range(4,7)) + binom(6,3)*w**3*(1-w)**3 * k
    chk(f"G({w}) binomial identity", G(w), alt, 1e-12)

print("\n=== 3. Tiebreak: closed form vs recursion ===")
for p, q in ((0.6,0.4),(0.65,0.35),(0.5,0.5),(0.7,0.3)):
    D = p*q/(p*q + (1-p)*(1-q))
    # X = Bin(6,p) + Bin(6,q)
    dist = [0.0]*13
    for i in range(7):
        for j in range(7):
            dist[i+j] += binom(6,i)*p**i*(1-p)**(6-i) * binom(6,j)*q**j*(1-q)**(6-j)
    alt = sum(dist[7:]) + dist[6]*D
    chk(f"TB({p},{q}) closed form", tiebreak_win_prob(p,q,0,0,True), alt, 1e-10)

print("\n=== 4. Tiebreak serve-order symmetry: TB(p,q) == 1 - TB(1-q,1-p) ===")
for p,q in ((0.6,0.4),(0.72,0.31),(0.55,0.48)):
    chk(f"serve-first irrelevant ({p},{q})",
        tiebreak_win_prob(p,q,0,0,True), tiebreak_win_prob(p,q,0,0,False), 1e-12)
    chk(f"TB(p,q)==1-TB(1-q,1-p) ({p},{q})",
        tiebreak_win_prob(p,q,0,0,True), 1-tiebreak_win_prob(1-q,1-p,0,0,True), 1e-12)

print("\n=== 5. Set closed form from 0-0 ===")
for p,q in ((0.65,0.40),(0.60,0.39),(0.5,0.5)):
    gp, gq = G(p), G(q)
    TB = tiebreak_win_prob(p,q)
    F = gp*gq + (gp*(1-gq) + (1-gp)*gq)*TB
    dist = [0.0]*11
    for i in range(6):
        for j in range(6):
            dist[i+j] += binom(5,i)*gp**i*(1-gp)**(5-i) * binom(5,j)*gq**j*(1-gq)**(5-j)
    alt = sum(dist[6:]) + dist[5]*F
    chk(f"set 0-0 ({p},{q})", set_from_games(p,q,0,0,True), alt, 1e-10)
    chk(f"set 0-0 server-irrelevant ({p},{q})",
        set_from_games(p,q,0,0,True), set_from_games(p,q,0,0,False), 1e-12)

print("\n=== 6. Zero-sum: P_A(state) + P_B(mirror state) == 1 ===")
p,q = 0.62, 0.41
# B's params: serves with prob 1-q (A returns with q), returns with 1-p
pb, qb = 1-q, 1-p
for (a,b,srv) in [(0,0,True),(3,1,True),(3,1,False),(5,5,True),(2,4,False),(6,5,True)]:
    chk(f"set zero-sum a={a} b={b} srv={srv}",
        set_from_games(p,q,a,b,srv) + set_from_games(pb,qb,b,a,not srv), 1.0, 1e-12)
for bo in (3,5):
    chk(f"match zero-sum bo{bo} 1-0 g3-2",
        match_win_prob(p,q,1,0,bo,games_me=3,games_opp=2,i_serve=True)
        + match_win_prob(pb,qb,0,1,bo,games_me=2,games_opp=3,i_serve=False), 1.0, 1e-12)

print("\n=== 7. Monte Carlo: full match simulation vs match_win_prob ===")
def sim_game(w):
    x=y=0
    while True:
        if random.random() < w: x+=1
        else: y+=1
        if x>=4 and x-y>=2: return True
        if y>=4 and y-x>=2: return False

def sim_tb(p,q,i_serve):
    x=y=0; n=0
    while True:
        n+=1
        w = p if i_serve else q
        if random.random() < w: x+=1
        else: y+=1
        if n % 2 == 1: i_serve = not i_serve
        if x>=7 and x-y>=2: return True
        if y>=7 and y-x>=2: return False

def sim_set(p,q,i_serve):
    a=b=0
    while True:
        if a==6 and b==6: return sim_tb(p,q,i_serve)
        if sim_game(p if i_serve else q): a+=1
        else: b+=1
        i_serve = not i_serve
        if a>=6 and a-b>=2: return True
        if b>=6 and b-a>=2: return False

def sim_match(p,q,best_of):
    need = best_of//2+1; sm=so=0
    while sm<need and so<need:
        if sim_set(p,q,True): sm+=1
        else: so+=1
    return sm>=need

for (p,q,bo) in ((0.65,0.40,3),(0.60,0.39,3),(0.58,0.45,5)):
    N=40000
    wins = sum(sim_match(p,q,bo) for _ in range(N))
    mc = wins/N
    se = math.sqrt(mc*(1-mc)/N)
    model = match_win_prob(p,q,0,0,bo)
    chk(f"MC match p={p} q={q} bo{bo} (2se={2*se:.4f})", mc, model, 3*se)

print("\n=== 8. Monte Carlo: mid-state set price ===")
p,q = 0.62, 0.41
for (a,b,srv,px,py) in [(3,4,False,1,3),(5,5,True,3,3),(2,1,True,2,2)]:
    def sim_from():
        # play out current game from px,py
        x,y = px,py; w = p if srv else q
        while not ((x>=4 and x-y>=2) or (y>=4 and y-x>=2)):
            if random.random()<w: x+=1
            else: y+=1
        won = x>y
        aa,bb = (a+1,b) if won else (a,b+1)
        s = not srv
        while True:
            if aa>=6 and aa-bb>=2: return True
            if bb>=6 and bb-aa>=2: return False
            if aa==6 and bb==6: return sim_tb(p,q,s)
            if sim_game(p if s else q): aa+=1
            else: bb+=1
            s = not s
    N=40000
    mc = sum(sim_from() for _ in range(N))/N
    se = math.sqrt(mc*(1-mc)/N)
    model = set_win_prob(p,q,a,b,srv,px,py)
    chk(f"MC set {a}-{b} srv={srv} pts {px}-{py} (2se={2*se:.4f})", mc, model, 3*se)

print("\n=== 9. Fitter round-trip: recover known (p,q) from exact prices ===")
sys.path.insert(0, '/Users/bradleyguan/Documents/Coding/Tennis-Model')
from market_implied import implied_pq, model_prob, invert_G
from implied_model import ImpliedModel
for truth in ((0.65,0.40),(0.60,0.39),(0.70,0.30)):
    obs = [(dict(games_me=0,games_opp=0,i_serve=True), model_prob(*truth, dict(games_me=0,games_opp=0,i_serve=True),3)),
           (dict(games_me=1,games_opp=0,i_serve=False), model_prob(*truth, dict(games_me=1,games_opp=0,i_serve=False),3)),
           (dict(games_me=3,games_opp=2,i_serve=False,points_me='40',points_opp='30'),
            model_prob(*truth, dict(games_me=3,games_opp=2,i_serve=False,points_me='40',points_opp='30'),3))]
    pf,qf,rmse = implied_pq(obs,3)
    chk(f"implied_pq recovers p={truth[0]}", pf, truth[0], 2e-3)
    chk(f"implied_pq recovers q={truth[1]}", qf, truth[1], 2e-3)

for w in (0.55,0.65,0.75):
    chk(f"invert_G(G({w}))", invert_G(G(w)), w, 1e-9)

print("\n=== 10. ImpliedModel round-trip (prior-free) ===")
truth=(0.65,0.40)
m = ImpliedModel(best_of=3, first_server='me', split_prior_sd=5.0)
for (a,b) in [(0,0),(1,0),(1,1),(2,1),(3,1),(3,2),(4,2)]:
    st = m._state(games_me=a, games_opp=b)
    m.observe(model_prob(*truth, st, 3), games_me=a, games_opp=b)
chk("ImpliedModel recovers p", m.p, truth[0], 5e-3)
chk("ImpliedModel recovers q", m.q, truth[1], 5e-3)

print("\n=== 11. Server alternation logic ===")
m2 = ImpliedModel(first_server='me')
for (a,b,exp) in [(0,0,True),(1,0,False),(1,1,True),(2,1,False),(3,3,True)]:
    got = m2._serving(0,0,a,b)
    print(f"{'OK ' if got==exp else 'FAIL'} serving at {a}-{b}: {got} (expect {exp})")
    if got!=exp: fails.append(f"serving {a}-{b}")

print("\n=== 12. set_number_win_prob: numbered-set (SETWINNER) pricing ===")
from tennis_model import set_number_win_prob
p, q = 0.62, 0.41
pb, qb = 1 - q, 1 - p
st = dict(games_me=3, games_opp=2, i_serve=True, points_me=2, points_opp=1)
mirror = dict(games_me=2, games_opp=3, i_serve=False, points_me=1, points_opp=2)
chk("set_no==current == set_win_prob",
    set_number_win_prob(p, q, 1, **st), set_win_prob(p, q, tb_target=7, **st), 1e-12)
chk("decider tb=10 honored (1-1, set 3)",
    set_number_win_prob(p, q, 3, sets_me=1, sets_opp=1, best_of=3, final_set_tb=10, **st),
    set_win_prob(p, q, tb_target=10, **st), 1e-12)
chk("current-set zero-sum",
    set_number_win_prob(p, q, 1, **st) + set_number_win_prob(pb, qb, 1, **mirror), 1.0, 1e-12)
# set 2 of a bo3 is ALWAYS played -> its two sides ARE complements even priced from set 1
chk("set-2 zero-sum (always played)",
    set_number_win_prob(p, q, 2, **st) + set_number_win_prob(pb, qb, 2, **mirror), 1.0, 1e-12)
# martingale over the next point, for a FUTURE set market ('me' is serving -> point prob p)
w_st = dict(st, points_me=st["points_me"] + 1)
l_st = dict(st, points_opp=st["points_opp"] + 1)
chk("future-set (3) martingale over next point",
    p * set_number_win_prob(p, q, 3, **w_st) + (1 - p) * set_number_win_prob(p, q, 3, **l_st),
    set_number_win_prob(p, q, 3, **st), 1e-12)

print("\n=== 13. Monte Carlo: set-3 winner from mid-set-1 state (unplayed = neither) ===")
def sim_set_from_state(a, b, srv, x, y):
    while not ((x >= 4 and x - y >= 2) or (y >= 4 and y - x >= 2)):
        if random.random() < (p if srv else q): x += 1
        else: y += 1
    if x > y: a += 1
    else: b += 1
    s = not srv
    while True:
        if a >= 6 and a - b >= 2: return True
        if b >= 6 and b - a >= 2: return False
        if a == 6 and b == 6: return sim_tb(p, q, s)
        if sim_game(p if s else q): a += 1
        else: b += 1
        s = not s
N = 40000
res = {"me": 0, "opp": 0, "unplayed": 0}
for _ in range(N):
    sm = so = 0
    if sim_set_from_state(st["games_me"], st["games_opp"], st["i_serve"],
                          st["points_me"], st["points_opp"]): sm += 1
    else: so += 1
    if sim_set(p, q, True): sm += 1
    else: so += 1
    if sm == 1 and so == 1:
        res["me" if sim_set(p, q, True) else "opp"] += 1
    else:
        res["unplayed"] += 1
m_me, m_opp = set_number_win_prob(p, q, 3, **st), set_number_win_prob(pb, qb, 3, **mirror)
for label, mc_n, model in (("me wins set 3", res["me"], m_me),
                           ("opp wins set 3", res["opp"], m_opp),
                           ("set 3 unplayed", res["unplayed"], 1.0 - m_me - m_opp)):
    mc = mc_n / N
    se = math.sqrt(max(mc * (1 - mc), 1e-9) / N)
    chk(f"MC {label} (2se={2*se:.4f})", mc, model, 3 * se)

print("\n" + "="*60)
print(f"{len(fails)} FAILURES" if fails else "ALL CHECKS PASSED")
for f in fails: print("  -", f)
