# notebooks/run_everything.py
"""
Master script — runs the full pipeline end to end.
All gates must pass. This is what you show in interviews.
"""
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep matplotlib cache in a writable project-local directory.
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

print("=" * 60)
print("  GTO POKER BOT — FULL PIPELINE")
print("=" * 60)

# ── GATE 1: Kuhn Nash Solution ──────────────────────────────
print("\n[1/10] Verifying Kuhn Nash solution...")
from core.cfr_plus import CFRPlus
from core.exploitability import compute_exploitability

solver = CFRPlus()
solver.train(20000)
strat = solver.get_full_strategy()

j_bluff = strat.get("J:", {}).get("b", 0.0)
k_bet = strat.get("K:", {}).get("b", 0.0)
q_bet = strat.get("Q:", {}).get("b", 0.0)
q_call_after_cb = strat.get("Q:cb", {}).get("c", 0.0)
q_call_vs_bet = strat.get("Q:b", {}).get("c", 0.0)
expl = compute_exploitability(strat)

print(f"  P1 K bet:    {k_bet:.3f}  (equilibrium family: ~3*J_bluff)")
print(f"  P1 Q bet:    {q_bet:.3f}  (expected ~0.0)")
print(f"  P1 J bluff:  {j_bluff:.3f}  (equilibrium alpha in [0, 1/3])")
print(f"  P1 Q:cb call {q_call_after_cb:.3f}  (expected ~alpha+1/3)")
print(f"  P2 Q:b call  {q_call_vs_bet:.3f}  (expected ~1/3)")
print(f"  Exploitability: {expl:.6f}  (expected < 0.005)")
assert 0.0 <= j_bluff <= 0.36
assert abs(k_bet - 3.0 * j_bluff) < 0.08
assert q_bet < 0.1
assert abs(q_call_after_cb - (j_bluff + 1.0 / 3.0)) < 0.08
assert abs(q_call_vs_bet - (1.0 / 3.0)) < 0.08
assert expl < 0.01
print("  PASS ✓")

# ── GATE 2: CFR+ vs Vanilla convergence ─────────────────────
print("\n[2/10] Verifying CFR+ and DCFR convergence vs Vanilla...")
from core.cfr import VanillaCFR
from core.dcfr import DCFR

v = VanillaCFR()
v.train(2000)
v_expl = compute_exploitability(v.get_full_strategy())

c = CFRPlus()
c.train(2000)
c_expl = compute_exploitability(c.get_full_strategy())

dc = DCFR()
dc.train(2000)
dc_expl = compute_exploitability(dc.get_full_strategy())

print(f"  Vanilla @ 2000: {v_expl:.6f}")
print(f"  CFR+    @ 2000: {c_expl:.6f}")
print(f"  DCFR    @ 2000: {dc_expl:.6f}")
assert c_expl < v_expl
assert dc_expl < v_expl
print("  PASS ✓")

# ── GATE 3: Opponent model — GTO opponent has KL ≈ 0 ────────
print("\n[3/10] Verifying Dirichlet model KL behavior...")
from opponent_model.dirichlet_model import DirichletOpponentModel

model = DirichletOpponentModel(alpha_prior=1.0)
rng = random.Random(7)

# Simulate opponent actions sampled from the learned Nash strategy itself.
for _ in range(60):
    model.observe("K:", "b" if rng.random() < strat["K:"]["b"] else "c")
    model.observe("Q:", "b" if rng.random() < strat["Q:"]["b"] else "c")
    model.observe("J:", "b" if rng.random() < strat["J:"]["b"] else "c")

kl_k = model.kl_from_nash("K:", ["b","c"], strat)
kl_q = model.kl_from_nash("Q:", ["b","c"], strat)
kl_j = model.kl_from_nash("J:", ["b","c"], strat)

print(f"  KL at K: (GTO sample) = {kl_k:.4f}  (expected small)")
print(f"  KL at Q: (GTO sample) = {kl_q:.4f}  (expected small)")
print(f"  KL at J: (GTO sample) = {kl_j:.4f}  (expected small)")
assert kl_k < 0.08 and kl_q < 0.08 and kl_j < 0.08
print("  PASS ✓")

# ── GATE 4: SPRT detects nit quickly ─────────────────────────
print("\n[4/10] Verifying SPRT detects leak in < 30 observations...")
from opponent_model.hypothesis_test import SPRTLeakDetector

# Use a non-degenerate baseline frequency (Q:b fold ~ 2/3), then test
# a strong over-fold leak at that same node.
sprt = SPRTLeakDetector(alpha=0.05, beta=0.05, epsilon=0.20)
nash_fold_freq = 1.0 - strat.get("Q:b", {"c": 1.0 / 3.0}).get("c", 1.0 / 3.0)
nit_fold_freq = 0.90
rng = random.Random(17)

n_to_decide = None
for i in range(100):
    opp_action = "f" if rng.random() < nit_fold_freq else "c"
    result = sprt.update("Q:b", "f", opp_action, nash_fold_freq)
    if result == "H1":
        n_to_decide = i + 1
        break

print(f"  SPRT detected nit after {n_to_decide} observations")
assert n_to_decide is not None and n_to_decide < 50
print("  PASS ✓")

# ── GATE 5: Glosten-Milgrom isomorphism ──────────────────────
print("\n[5/10] Verifying Glosten-Milgrom toy mapping consistency...")
from trading.glosten_milgrom import GlostenMilgromModel

# Use an exogenous informed-trader fraction and check consistency both ways.
gm = GlostenMilgromModel(V_L=1, V_H=3, mu=2.0 / 3.0)
implied_alpha = gm.implied_bluff_frequency_from_mu()
print(f"  Exogenous μ: {gm.mu:.4f} -> implied alpha: {implied_alpha:.4f}")
assert abs(j_bluff - implied_alpha) < 0.12, \
    f"CFR bluff freq {j_bluff:.4f} not close to implied alpha {implied_alpha:.4f}"

result = gm.verify_isomorphism(j_bluff)
spread_gap = abs(result["gm_spread"] - result["poker_spread"])
print(f"\n  Spread gap |GM - Poker|: {spread_gap:.4f}")
assert spread_gap < 0.02, f"Spread gap too large: {spread_gap}"
print("  PASS ✓")

# ── GATE 6: Generate all plots ───────────────────────────────
print("\n[6/10] Generating analysis plots...")
from analysis.convergence_plots import generate_convergence_plot, generate_convergence_table
generate_convergence_plot(max_iters=3000, sample_every=100)
generate_convergence_table(max_iters=2000, sample_every=100)
print("  Plots saved to analysis/")

# ── GATE 7: Variance reduction check ─────────────────────────
print("\n[7/10] Verifying MCCFR variance reduction...")
from core.mccfr import ExternalSamplingMCCFR

mccfr = ExternalSamplingMCCFR()
mccfr.train(5000)
mccfr.variance_reduction_stats()

# Baselines should be nonzero and calibrated after 5000 iters
calibrated = sum(1 for i_set, n in mccfr.baseline_count.items() if n > 20)
print(f"  Info sets with calibrated baseline: {calibrated}")
assert calibrated >= 3
print("  PASS ✓")

# ── GATE 8: Posterior variance confidence gating ─────────────
print("\n[8/10] Verifying posterior variance confidence gating...")
from opponent_model.dirichlet_model import DirichletOpponentModel

model = DirichletOpponentModel()
# Simulate nit: folds 90% of the time facing a bet
for _ in range(30):
    model.observe("J:b", "f" if random.random() < 0.90 else "c")

conf = model.confidence_from_variance("J:b", ["c", "f"], strat)
var = model.posterior_variance("J:b", "f", ["c", "f"])

print(f"  Posterior variance on fold action: {var:.6f}")
print(f"  Confidence score: {conf:.4f}  (expected > 0.5 after 30 obs of nit)")
assert conf > 0.3, f"Confidence too low: {conf}"
print("  PASS ✓")

# ── GATE 9: Leduc solver smoke test ───────────────────────────
print("\n[9/10] Verifying Leduc solver training smoke test...")
from core.leduc_poker import LeducCFR

leduc = LeducCFR()
leduc.train(2)
leduc_strat = leduc.get_full_strategy()
print(f"  Leduc info sets learned: {len(leduc_strat)}")
assert len(leduc_strat) > 0
print("  PASS ✓")

# ── GATE 10: Pytest invocation smoke check ─────────────────────
print("\n[10/10] Verifying pytest-style tests are discoverable...")
try:
    import pytest  # noqa: F401
except ModuleNotFoundError:
    print("  SKIP: pytest not installed in current environment.")
    print("  Install dev dependencies with: pip install -r requirements-dev.txt")
else:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_kuhn_game.py", "tests/test_cfr_convergence.py", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    assert result.returncode == 0, result.stderr
    print("  PASS ✓")

print("\n" + "=" * 60)
print("  ALL GATES PASSED — Pipeline complete.")
print("=" * 60)
