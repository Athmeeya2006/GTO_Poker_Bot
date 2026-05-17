# analysis/convergence_plots.py
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

mpl_cache_dir = REPO_ROOT / ".mplconfig"
mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib
import matplotlib.pyplot as plt
from core.cfr import VanillaCFR
from core.cfr_plus import CFRPlus
from core.exploitability import compute_exploitability

def generate_convergence_plot(max_iters=5000, sample_every=100):
    """
    Trains both solvers and plots exploitability over iterations.
    This is your main proof that CFR works.
    """
    vanilla = VanillaCFR()
    cfrplus = CFRPlus()

    checkpoints = list(range(sample_every, max_iters + 1, sample_every))
    vanilla_exploits = []
    plus_exploits    = []

    print("Training Vanilla CFR...")
    for checkpoint in checkpoints:
        vanilla.train(sample_every)
        e = compute_exploitability(vanilla.get_full_strategy())
        vanilla_exploits.append(e)
        if checkpoint % 500 == 0:
            print(f"  iter {checkpoint}: exploitability = {e:.6f}")

    print("Training CFR+...")
    for checkpoint in checkpoints:
        cfrplus.train(sample_every)
        e = compute_exploitability(cfrplus.get_full_strategy())
        plus_exploits.append(e)
        if checkpoint % 500 == 0:
            print(f"  iter {checkpoint}: exploitability = {e:.6f}")

    # Theoretical O(1/sqrt(T)) bound for vanilla CFR
    C = vanilla_exploits[0] * (checkpoints[0] ** 0.5)
    theoretical = [C / (t ** 0.5) for t in checkpoints]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CFR Convergence - Kuhn Poker", fontsize=14, fontweight='bold')

    # Linear scale
    ax1.plot(checkpoints, vanilla_exploits, label='Vanilla CFR', color='steelblue', linewidth=2)
    ax1.plot(checkpoints, plus_exploits,    label='CFR+',        color='darkorange', linewidth=2)
    ax1.plot(checkpoints, theoretical,      label='O(1/√T) bound', color='gray',
             linestyle='--', linewidth=1.5)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Exploitability (chips/game)")
    ax1.set_title("Linear Scale")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale (more informative)
    ax2.semilogy(checkpoints, vanilla_exploits, label='Vanilla CFR', color='steelblue', linewidth=2)
    ax2.semilogy(checkpoints, plus_exploits,    label='CFR+',        color='darkorange', linewidth=2)
    ax2.semilogy(checkpoints, theoretical,      label='O(1/√T) bound', color='gray',
                 linestyle='--', linewidth=1.5)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Exploitability (log scale)")
    ax2.set_title("Log Scale")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("analysis/convergence_kuhn.png", dpi=150, bbox_inches='tight')
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    print("Saved: analysis/convergence_kuhn.png")

def generate_threeway_comparison(max_iters=5000, sample_every=50):
    """Vanilla CFR vs CFR+ vs MCCFR on Kuhn Poker."""
    from core.mccfr import ExternalSamplingMCCFR

    vanilla = VanillaCFR()
    cfrplus = CFRPlus()
    mccfr   = ExternalSamplingMCCFR()

    checkpoints = list(range(sample_every, max_iters + 1, sample_every))
    results = {"Vanilla CFR": [], "CFR+": [], "MCCFR (ext. sampling)": []}

    print("Running 3-way comparison...")
    for _ in checkpoints:
        vanilla.train(sample_every)
        cfrplus.train(sample_every)
        mccfr.train(sample_every * 6)  # MCCFR needs more iters (sampling variance)

        results["Vanilla CFR"].append(compute_exploitability(vanilla.get_full_strategy()))
        results["CFR+"].append(compute_exploitability(cfrplus.get_full_strategy()))
        results["MCCFR (ext. sampling)"].append(compute_exploitability(mccfr.get_full_strategy()))

    colors = {"Vanilla CFR": "steelblue", "CFR+": "darkorange", "MCCFR (ext. sampling)": "green"}

    plt.figure(figsize=(9, 5))
    for name, values in results.items():
        plt.semilogy(checkpoints, values, label=name, color=colors[name], linewidth=2)

    plt.xlabel("Iterations (vanilla/CFR+) | ×6 for MCCFR")
    plt.ylabel("Exploitability (log scale)")
    plt.title("Three-Way CFR Convergence Comparison — Kuhn Poker")
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig("analysis/threeway_convergence.png", dpi=150, bbox_inches='tight')
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    print("Saved: analysis/threeway_convergence.png")

if __name__ == "__main__":
    generate_convergence_plot()
    generate_threeway_comparison()
