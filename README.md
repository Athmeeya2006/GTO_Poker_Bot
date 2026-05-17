# GTO Poker Bot

A Python framework for computing, validating, and exploiting approximate Nash equilibria in imperfect-information games - with a formal bridge to market microstructure theory.

---

## What This Does

Most CFR implementations stop at "solver converges, strategy printed." This project goes three layers deeper:

1. **Solve** - four CFR-family algorithms compute Nash-approximate strategies with measurable convergence guarantees.
2. **Validate** - exploitability is computed under strict imperfect-information constraints (no card-peeking in best-response evaluation), with automated convergence checks.
3. **Adapt** - a Bayesian opponent model detects when an opponent leaks exploitable patterns and gates strategy mixing based on statistical confidence, not raw deviation.

The theoretical through-line: bluff frequencies computed by CFR are formally isomorphic to the adverse-selection spread in the Glosten-Milgrom (1985) market-making model. The solver output *is* the zero-profit spread condition.

---

## Algorithms Implemented

| Solver | Notes |
|---|---|
| Vanilla CFR | Baseline. Full tree traversal, regret matching. |
| CFR+ | Regret clipping + linear averaging. Faster empirical convergence. |
| DCFR | Discounted CFR. Exponential down-weighting of early regrets. |
| External Sampling MCCFR | Sampled traversal with per-infoset baseline tracking for variance diagnostics. |

All four solvers are benchmarked against each other. Convergence metrics (exploitability vs. iterations, log-log slope) are exported to `analysis/convergence_metrics.csv`.

---

## Exploitability

Exploitability is computed using **information-set-consistent best responses** - the best-response player cannot observe the opponent's private card during computation. This matters: naive implementations inflate exploitability by solving a full-information problem instead of the correct imperfect-information one.

At Nash equilibrium, both players' best-response gains sum to zero. This implementation verifies that bound.

---

## Opponent Modeling

Three components work together:

**Dirichlet Posterior Model** - maintains a per-infoset Bayesian posterior over opponent action frequencies. Reports posterior mean, credible intervals, KL divergence from Nash, and posterior variance. The posterior variance feeds directly into the confidence gate.

**SPRT Leak Detector** - applies a Sequential Probability Ratio Test to detect statistically significant deviations from GTO play. Triggers exploitation only when the evidence meets a calibrated threshold, not on noise.

**Confidence-Gated Strategy Mixer** - blends GTO and exploitative strategies as a function of the SPRT signal and posterior precision. The mixer does not increase exploitation weight unless deviation is both present and statistically reliable.

---

## Glosten-Milgrom Isomorphism

The `trading/` module formalizes the connection between Kuhn Poker equilibria and the GM (1985) model of informed trading:

| Poker | Market |
|---|---|
| Player 1 with private card | Informed trader who knows asset value |
| Player 2 with no private info | Market maker |
| Bluffing with Jack (weak hand) | Noise trader submitting buy order |
| Always betting King (strong hand) | Informed trader buying at V_H |
| Nash bluff frequency α | Adverse selection component of bid-ask spread |

At Kuhn Nash equilibrium, Player 1 bluffs with probability α = 1/3. Running through Bayes' theorem: P(informed | bet) = 1/(1 + α) = 3/4. The GM zero-profit spread condition yields the same posterior. The CFR solution and the GM spread are the same object derived two ways.

---

## Repository Structure

```
core/
  kuhn_poker.py       - game engine, terminal conditions, infoset encoding
  cfr.py              - Vanilla CFR
  cfr_plus.py         - CFR+ (regret clipping, linear averaging)
  dcfr.py             - Discounted CFR
  mccfr.py            - External Sampling MCCFR + variance diagnostics
  exploitability.py   - imperfect-information best-response exploitability
  leduc_poker.py      - Leduc game engine and solver scaffold

opponent_model/
  dirichlet_model.py  - posterior mean, credible intervals, KL, variance score
  hypothesis_test.py  - SPRT-based leak detector
  strategy_mixer.py   - confidence-gated GTO/exploit blending
  range_model.py      - Bayesian hand-range updater

analysis/
  convergence_plots.py       - generates plots and metrics table
  convergence_kuhn.png       - solver convergence curves
  convergence_metrics.csv    - exploitability vs. iterations for all four solvers

trading/
  glosten_milgrom.py  - formal Kuhn <> GM isomorphism and spread derivation

tests/
  test_kuhn_game.py         - game mechanics
  test_cfr_convergence.py   - equilibrium-family checks, exploitability bound
  test_leduc_game.py        - Leduc smoke tests

notebooks/
  run_everything.py   - 10-gate validation pipeline
  run_everything.ipynb
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

---

## Running

### Full validation pipeline

```bash
python notebooks/run_everything.py
```

Runs 10 gates:

1. Kuhn Nash-family strategy constraints + exploitability bound
2. CFR+ and DCFR convergence speed vs. Vanilla CFR
3. Dirichlet KL behavior on near-Nash samples
4. SPRT leak detection sensitivity
5. GM / Kuhn spread consistency
6. Convergence plot and metrics export
7. MCCFR baseline calibration diagnostics
8. Posterior-variance confidence gating
9. Leduc smoke test
10. pytest discoverability

### Tests only

```bash
pytest
```

### Trading module

```bash
python trading/glosten_milgrom.py
```

---

## Technical Notes

**Kuhn equilibrium family.** Kuhn Poker admits a continuum of Player-1 equilibria parameterized by bluff frequency α ∈ [0, 1/3]. Validation uses the equilibrium relationship P1(K:bet) ≈ 3 × P1(J:bet) rather than asserting a single fixed point - this correctly tests membership in the equilibrium family rather than convergence to one specific member.

**MCCFR variance.** `ExternalSamplingMCCFR` tracks per-infoset baseline statistics and exposes `variance_reduction_stats()`. This makes variance reduction visible rather than implicit, and enables calibration checks in gate 7.

**Import note.** Run all scripts from repository root. If running from a subdirectory:

```bash
python -m notebooks.run_everything
python -m trading.glosten_milgrom
```

---

## Dependencies

Runtime: `numpy`, `scipy`, `matplotlib`  
Dev: `pytest` (see `requirements-dev.txt`)  
CI: GitHub Actions (`.github/workflows/ci.yml`)