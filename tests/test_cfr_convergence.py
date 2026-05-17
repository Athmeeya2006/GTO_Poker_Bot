# tests/test_cfr_convergence.py
import sys
sys.path.insert(0, '.')
from core.cfr import VanillaCFR
from core.cfr_plus import CFRPlus
from core.exploitability import compute_exploitability

def test_kuhn_nash_solution():
    """
    Two-player Kuhn Poker has:
      - a continuum of equilibria for Player 1 parameterized by alpha in [0, 1/3]
      - a unique equilibrium policy for Player 2

    Equilibrium relations used in this test:
      P1(J:) bet = alpha
      P1(K:) bet = 3*alpha
      P1(Q:cb) call = alpha + 1/3
      P2(K:b) call = 1
      P2(Q:b) call = 1/3
      P2(J:b) call = 0
    """
    solver = VanillaCFR()
    solver.train(10000)
    strat = solver.get_full_strategy()

    print("\n=== Kuhn Nash Equilibrium Verification ===")
    print("(After 10,000 iterations)\n")

    # Print all info sets
    for key in sorted(strat.keys()):
        probs = strat[key]
        print(f"  {key:12s}  {probs}")

    # Check key values.
    # Kuhn has a continuum of equilibria for P1:
    #   alpha = P1(J bet) in [0, 1/3]
    #   P1(K bet) = 3*alpha
    #   P1(Q after check-bet, call) = alpha + 1/3
    k_bet = strat.get("K:", {}).get("b", 0)
    j_bet = strat.get("J:", {}).get("b", 0)
    q_bet = strat.get("Q:", {}).get("b", 0)
    q_call_after_cb = strat.get("Q:cb", {}).get("c", 0)

    print(f"\nP1 K: bet prob = {k_bet:.4f}  (expected ~1.0)")
    print(f"P1 Q: bet prob = {q_bet:.4f}  (expected ~0.0)")
    print(f"P1 J: bet prob = {j_bet:.4f}  (expected ~0.33)")

    # P2 facing a bet: K calls, J folds, and Q calls ~1/3 in the unique P2 equilibrium.
    k_call = strat.get("K:b", {}).get("c", 0)
    q_call = strat.get("Q:b", {}).get("c", 0)
    j_call = strat.get("J:b", {}).get("c", 0)
    print(f"\nP2 K vs bet: call prob = {k_call:.4f}  (expected ~1.0)")
    print(f"P2 Q vs bet: call prob = {q_call:.4f}  (expected ~0.33)")
    print(f"P2 J vs bet: call prob = {j_call:.4f}  (expected ~0.0)")

    # Exploitability should be very low
    expl = compute_exploitability(strat)
    print(f"\nExploitability = {expl:.6f}  (expected < 0.01)")

    assert 0.0 <= j_bet <= 0.36, f"P1 J bluff frequency should be in equilibrium range, got {j_bet}"
    assert abs(k_bet - 3.0 * j_bet) < 0.08, f"P1 K bet should track 3*alpha, got K={k_bet}, J={j_bet}"
    assert q_bet < 0.1,  f"P1 Q should not bet, got {q_bet}"
    assert abs(q_call_after_cb - (j_bet + 1.0 / 3.0)) < 0.08, \
        f"P1 Q:cb call should be alpha+1/3, got {q_call_after_cb} vs {j_bet + 1.0/3.0}"
    assert k_call > 0.9, f"P2 K should call, got {k_call}"
    assert abs(q_call - 1.0 / 3.0) < 0.08, f"P2 Q should call ~1/3, got {q_call}"
    assert j_call < 0.1, f"P2 J should fold, got {j_call}"
    assert expl < 0.01,  f"Exploitability too high: {expl}"

    print("\nPASS: Kuhn Nash solution verified.")

def test_cfr_plus_faster():
    """CFR+ should reach lower exploitability in fewer iterations."""
    N = 2000

    vanilla = VanillaCFR()
    vanilla.train(N)
    vanilla_expl = compute_exploitability(vanilla.get_full_strategy())

    cfrplus = CFRPlus()
    cfrplus.train(N)
    plus_expl = compute_exploitability(cfrplus.get_full_strategy())

    print(f"\nVanilla CFR exploitability @ {N} iters: {vanilla_expl:.6f}")
    print(f"CFR+       exploitability @ {N} iters: {plus_expl:.6f}")

    assert plus_expl < vanilla_expl, \
        f"CFR+ should converge faster. CFR+={plus_expl}, Vanilla={vanilla_expl}"
    print("PASS: CFR+ converges faster than Vanilla CFR.")

if __name__ == "__main__":
    test_kuhn_nash_solution()
    test_cfr_plus_faster()
