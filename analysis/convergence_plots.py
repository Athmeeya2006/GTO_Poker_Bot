# analysis/convergence_plots.py
import os
import sys
from pathlib import Path
import csv
import math

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
from core.mccfr import ExternalSamplingMCCFR
from core.dcfr import DCFR
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


def _log_slope(xs, ys):
    """
    Compute local slope of log(y) vs log(x) using adjacent points.
    Returns list with first value as None.
    """
    slopes = [None]
    for i in range(1, len(xs)):
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = max(ys[i - 1], 1e-12), max(ys[i], 1e-12)
        slopes.append((math.log(y1) - math.log(y0)) /
                      (math.log(x1) - math.log(x0)))
    return slopes


def generate_convergence_table(max_iters=3000, sample_every=100, mccfr_scale=6,
                               output_csv="analysis/convergence_metrics.csv"):
    """
    Generate a quantitative convergence table for Vanilla CFR, CFR+, DCFR, and MCCFR.
    Saves CSV with exploitability and local log-log slopes.
    """
    checkpoints = list(range(sample_every, max_iters + 1, sample_every))
    vanilla = VanillaCFR()
    cfrplus = CFRPlus()
    dcfr = DCFR()
    mccfr = ExternalSamplingMCCFR()

    metrics = {
        "vanilla": [],
        "cfr_plus": [],
        "dcfr": [],
        "mccfr": [],
    }

    for _ in checkpoints:
        vanilla.train(sample_every)
        cfrplus.train(sample_every)
        dcfr.train(sample_every)
        mccfr.train(sample_every * mccfr_scale)

        metrics["vanilla"].append(compute_exploitability(vanilla.get_full_strategy()))
        metrics["cfr_plus"].append(compute_exploitability(cfrplus.get_full_strategy()))
        metrics["dcfr"].append(compute_exploitability(dcfr.get_full_strategy()))
        metrics["mccfr"].append(compute_exploitability(mccfr.get_full_strategy()))

    slopes = {k: _log_slope(checkpoints, v) for k, v in metrics.items()}

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration",
            "vanilla_exploitability",
            "cfr_plus_exploitability",
            "dcfr_exploitability",
            "mccfr_exploitability",
            "vanilla_loglog_slope",
            "cfr_plus_loglog_slope",
            "dcfr_loglog_slope",
            "mccfr_loglog_slope",
        ])
        for i, t in enumerate(checkpoints):
            writer.writerow([
                t,
                metrics["vanilla"][i],
                metrics["cfr_plus"][i],
                metrics["dcfr"][i],
                metrics["mccfr"][i],
                slopes["vanilla"][i],
                slopes["cfr_plus"][i],
                slopes["dcfr"][i],
                slopes["mccfr"][i],
            ])

    final = len(checkpoints) - 1
    print("\n=== Quantitative Convergence Summary ===")
    print(f"Final @ {checkpoints[final]} iterations:")
    print(f"  Vanilla: {metrics['vanilla'][final]:.6f}")
    print(f"  CFR+   : {metrics['cfr_plus'][final]:.6f}")
    print(f"  DCFR   : {metrics['dcfr'][final]:.6f}")
    print(f"  MCCFR  : {metrics['mccfr'][final]:.6f}")
    print(f"Saved table: {output_csv}")

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
