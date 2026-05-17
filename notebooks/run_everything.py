# notebooks/run_everything.py
"""
Master script - runs the full pipeline end to end.
All gates must pass. This validates every component of the system.
"""
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

print("=" * 60)
print("  GTO POKER BOT - FULL PIPELINE")
print("=" * 60)

# ── GATE 1: Kuhn Nash Solution - EXACT values ──────────────────
print("\n[1/12] Verifying Kuhn Nash solution (exact analytical values)...")
from core.cfr_plus import CFRPlus
from core.exploitability import compute_exploitability, compute_exploitability_bruteforce

solver = CFRPlus()
solver.train(50000)
strat = solver.get_full_strategy()

j_bluff = strat.get("J:", {}).get("b", 0.0)
k_bet = strat.get("K:", {}).get("b", 0.0)
q_bet = strat.get("Q:", {}).get("b", 0.0)
q_call_after_cb = strat.get("Q:cb", {}).get("c", 0.0)
q_call_vs_bet = strat.get("Q:b", {}).get("c", 0.0)
j_bet_p2 = strat.get("J:c", {}).get("b", 0.0)
k_bet_p2 = strat.get("K:c", {}).get("b", 0.0)
q_bet_p2 = strat.get("Q:c", {}).get("b", 0.0)
expl = compute_exploitability(strat)

print(f"  P0 K: bet    {k_bet:.4f}  (expected: 3×α)")
print(f"  P0 Q: bet    {q_bet:.4f}  (expected: 0)")
print(f"  P0 J: bluff  {j_bluff:.4f}  (equilibrium α ∈ [0, 1/3])")
print(f"  P0 Q:cb call {q_call_after_cb:.4f}  (expected: α + 1/3)")
print(f"  P1 Q:b call  {q_call_vs_bet:.4f}  (expected: exactly 1/3)")
print(f"  P1 J:c bet   {j_bet_p2:.4f}  (expected: 1/3)")
print(f"  P1 K:c bet   {k_bet_p2:.4f}  (expected: 1)")
print(f"  P1 Q:c bet   {q_bet_p2:.4f}  (expected: 0)")
print(f"  Exploitability: {expl:.6f}")

# Tight tolerances (0.01 instead of old 0.08)
assert 0.0 <= j_bluff <= 0.34
assert abs(k_bet - 3.0 * j_bluff) < 0.01
assert q_bet < 0.005
assert abs(q_call_after_cb - (j_bluff + 1.0 / 3.0)) < 0.01
assert abs(q_call_vs_bet - 1.0 / 3.0) < 0.01
assert abs(j_bet_p2 - 1.0 / 3.0) < 0.01
assert k_bet_p2 > 0.99
assert q_bet_p2 < 0.01
assert expl < 0.001
print("  PASS ✓")

# ── GATE 2: Cross-validate exploitability methods ────────────
print("\n[2/12] Cross-validating tree-traversal vs brute-force exploitability...")
tree_expl = compute_exploitability(strat)
brute_expl = compute_exploitability_bruteforce(strat)
print(f"  Tree traversal: {tree_expl:.6f}")
print(f"  Brute force:    {brute_expl:.6f}")
print(f"  Difference:     {abs(tree_expl - brute_expl):.6f}")
assert abs(tree_expl - brute_expl) < 0.001, \
    f"Methods disagree: tree={tree_expl}, brute={brute_expl}"
print("  PASS ✓")

# ── GATE 3: CFR+ vs Vanilla vs DCFR convergence ─────────────
print("\n[3/12] Verifying convergence of all CFR variants...")
from core.cfr import VanillaCFR
from core.dcfr import DCFR

v = VanillaCFR()
v.train(3000)
v_expl = compute_exploitability(v.get_full_strategy())

c = CFRPlus()
c.train(3000)
c_expl = compute_exploitability(c.get_full_strategy())

dc = DCFR()
dc.train(3000)
dc_expl = compute_exploitability(dc.get_full_strategy())

print(f"  Vanilla @ 3000: {v_expl:.6f}")
print(f"  CFR+    @ 3000: {c_expl:.6f}")
print(f"  DCFR    @ 3000: {dc_expl:.6f}")
# All variants should converge well at 3000 iterations.
# With alternating updates, vanilla CFR converges correctly and may
# match or beat CFR+ at moderate iteration counts.
assert v_expl < 0.01, f"Vanilla exploitability {v_expl} too high"
assert c_expl < 0.01, f"CFR+ exploitability {c_expl} too high"
assert dc_expl < 0.01, f"DCFR exploitability {dc_expl} too high"
assert dc_expl < v_expl, f"DCFR should beat Vanilla"
print("  PASS ✓")

# ── GATE 4: Opponent model - GTO opponent has KL ≈ 0 ────────
print("\n[4/12] Verifying Dirichlet model KL behavior...")
from opponent_model.dirichlet_model import DirichletOpponentModel

model = DirichletOpponentModel(alpha_prior=1.0)
rng = random.Random(7)

for _ in range(200):
    model.observe("K:", "b" if rng.random() < strat["K:"]["b"] else "c")
    model.observe("Q:", "b" if rng.random() < strat["Q:"]["b"] else "c")
    model.observe("J:", "b" if rng.random() < strat["J:"]["b"] else "c")

kl_k = model.kl_from_nash("K:", ["b", "c"], strat)
kl_q = model.kl_from_nash("Q:", ["b", "c"], strat)
kl_j = model.kl_from_nash("J:", ["b", "c"], strat)

print(f"  KL at K: = {kl_k:.4f}")
print(f"  KL at Q: = {kl_q:.4f}")
print(f"  KL at J: = {kl_j:.4f}")
assert kl_k < 0.03 and kl_q < 0.03 and kl_j < 0.03
print("  PASS ✓")

# ── GATE 5: SPRT detects nit quickly ─────────────────────────
print("\n[5/12] Verifying SPRT leak detection speed...")
from opponent_model.hypothesis_test import SPRTLeakDetector

sprt = SPRTLeakDetector(alpha=0.05, beta=0.05, epsilon=0.20)
nash_fold_freq = 1.0 - strat.get("Q:b", {}).get("c", 1.0 / 3.0)
rng = random.Random(17)

n_to_decide = None
for i in range(100):
    opp_action = "f" if rng.random() < 0.90 else "c"
    result = sprt.update("Q:b", "f", opp_action, nash_fold_freq)
    if result == "H1":
        n_to_decide = i + 1
        break

print(f"  SPRT detected nit after {n_to_decide} observations")
assert n_to_decide is not None and n_to_decide < 50
print("  PASS ✓")

# ── GATE 6: Glosten-Milgrom isomorphism ──────────────────────
print("\n[6/12] Verifying Glosten-Milgrom mapping + trading signals...")
from trading.glosten_milgrom import GlostenMilgromModel, TradingSignalGenerator

implied_mu = GlostenMilgromModel(V_L=1, V_H=3).implied_mu_from_bluff_frequency(j_bluff)
gm = GlostenMilgromModel(V_L=1, V_H=3, mu=implied_mu)
result = gm.verify_isomorphism(j_bluff)
spread_gap = abs(result["gm_spread"] - result["poker_spread"])
assert spread_gap < 0.01

# Test functional trading signals
signal_gen = TradingSignalGenerator(V_L=1, V_H=3)
signals = signal_gen.from_cfr_solver(strat)
assert signals["signal"] == "HOLD"  # at Nash, no edge
assert signals["solver_converged"] is True
signal_gen.report(signals)
print("  PASS ✓")

# ── GATE 7: Generate analysis plots ──────────────────────────
print("\n[7/12] Generating analysis plots...")
from analysis.convergence_plots import generate_convergence_plot, generate_convergence_table
generate_convergence_plot(max_iters=3000, sample_every=100)
generate_convergence_table(max_iters=2000, sample_every=100)
print("  PASS ✓")

# ── GATE 8: MCCFR variance reduction ─────────────────────────
print("\n[8/12] Verifying MCCFR baseline calibration...")
from core.mccfr import ExternalSamplingMCCFR

mccfr = ExternalSamplingMCCFR()
mccfr.train(10000)
mccfr.variance_reduction_stats()

calibrated = sum(1 for n in mccfr.baseline_count.values() if n > 20)
print(f"  Info sets with calibrated baseline: {calibrated}")
assert calibrated >= 3
print("  PASS ✓")

# ── GATE 9: Posterior variance confidence gating ─────────────
print("\n[9/12] Verifying posterior variance confidence gating...")
model = DirichletOpponentModel()
for _ in range(50):
    model.observe("J:b", "f" if random.random() < 0.90 else "c")

conf = model.confidence_from_variance("J:b", ["c", "f"], strat)
var = model.posterior_variance("J:b", "f", ["c", "f"])
print(f"  Posterior variance on fold action: {var:.6f}")
print(f"  Confidence score: {conf:.4f}")
assert conf > 0.3
print("  PASS ✓")

# ── GATE 10: Leduc solver convergence ─────────────────────────
print("\n[10/12] Verifying Leduc Poker solver convergence...")
from core.leduc_poker import LeducCFR, LeducGameInterface
from core.exploitability import compute_exploitability_generic

leduc = LeducCFR()
leduc.train(30)
leduc_strat = leduc.get_full_strategy()
n_info_sets = len(leduc_strat)
print(f"  Leduc info sets learned: {n_info_sets}")
assert n_info_sets > 50

# Verify all strategies are valid distributions
for i_set, s in leduc_strat.items():
    total = sum(s.values())
    assert abs(total - 1.0) < 1e-6, f"Bad distribution at {i_set}: {s}"

# Verify exploitability decreases with training
game_if = LeducGameInterface()
expl_10 = compute_exploitability_generic(leduc_strat, game_if)

leduc2 = LeducCFR()
leduc2.train(80)
expl_80 = compute_exploitability_generic(leduc2.get_full_strategy(), game_if)
print(f"  Exploitability @ 30 iters: {expl_10:.4f}")
print(f"  Exploitability @ 80 iters: {expl_80:.4f}")
assert expl_80 < expl_10, "Exploitability should decrease with more training"
print("  PASS ✓")

# ── GATE 11: Game engine smoke test ──────────────────────────
print("\n[11/12] Verifying game engine integration...")
from core.game_engine import KuhnPokerBot

bot = KuhnPokerBot(training_iters=5000, exploit_mode=True)
assert bot.nash_strategy is not None
assert len(bot.nash_strategy) >= 12  # 12 Kuhn info sets
assert bot.exploitability < 0.01
print(f"  Bot initialized. Exploitability: {bot.exploitability:.6f}")
print(f"  Opponent model ready: {bot.opponent_model is not None}")
print(f"  Strategy mixer ready: {bot.strategy_mixer is not None}")
print(f"  SPRT detector ready:  {bot.leak_detector is not None}")
print("  PASS ✓")

# ── GATE 12: Pytest discovers and passes ALL tests ───────────
print("\n[12/12] Running full pytest suite...")
try:
    import pytest  # noqa: F401
except ModuleNotFoundError:
    print("  SKIP: pytest not installed.")
else:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
    assert result.returncode == 0, f"Tests failed:\n{result.stderr}"
    print("  PASS ✓")

print("\n" + "=" * 60)
print("  ALL GATES PASSED - Pipeline complete.")
print("=" * 60)
