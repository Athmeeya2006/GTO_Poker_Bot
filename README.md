# GTO Poker Bot

A Python framework for computing, validating, and exploiting approximate Nash equilibria in imperfect-information games - with an interactive game engine, scalable exploitability computation, and a formal bridge to market microstructure theory.

---

## What This Does

1. **Solve** - Four CFR-family algorithms compute Nash-approximate strategies for Kuhn Poker (12 info sets) and **Leduc Poker (~936 info sets)**, with measurable convergence guarantees and exploitability tracking.

2. **Validate** - Exploitability is computed via **tree-traversal best response** (O(game tree)), not brute-force enumeration. This scales to any game size. Cross-validated against brute-force on Kuhn for correctness.

3. **Play** - An **interactive CLI game engine** wires together the CFR solver, Bayesian opponent model, SPRT leak detector, and confidence-gated strategy mixer into an actual poker bot you can play against. The bot observes your patterns in real-time and exploits detected leaks.

4. **Adapt** - A Bayesian opponent model detects when an opponent leaks exploitable patterns and gates strategy mixing based on statistical confidence (posterior variance + SPRT), not raw deviation.

5. **Trade** - The Glosten-Milgrom module is **functional, not decorative**: it generates actionable trading signals, position sizes, and edge estimates from CFR equilibrium output, with explicit documentation of where the Kuhn ↔ GM analogy breaks down.

---

## Algorithms Implemented

| Solver | Notes |
|---|---|
| Vanilla CFR | Baseline. Full tree traversal, regret matching. |
| CFR+ | Regret clipping + linear averaging. Faster empirical convergence. |
| DCFR | Discounted CFR. Exponential down-weighting of early regrets. |
| External Sampling MCCFR | Sampled traversal with per-infoset baseline tracking for variance diagnostics. |

All four solvers are benchmarked against each other on both Kuhn and Leduc.

---

## Games Supported

### Kuhn Poker (12 info sets)
The textbook game - 3 cards, 2 players. Analytical Nash equilibrium is known and verified to machine precision.

### Leduc Poker (~936 info sets)
A significantly larger game - 6 cards (2 each of J, Q, K), 2 betting rounds, community card, raises. The CFR+ solver converges with verified exploitability decrease. **Fully tested and in CI.**

---

## Interactive Bot

```bash
python core/game_engine.py
```

Play Kuhn Poker against the GTO bot. The bot:
- Computes Nash equilibrium via CFR+
- Tracks your action frequencies via Dirichlet posterior
- Detects statistically significant leaks via SPRT
- Blends GTO + exploitative strategy based on posterior variance confidence
- Shows opponent model state and session statistics on demand

```bash
# Options
python core/game_engine.py --hands 50        # play 50 hands
python core/game_engine.py --no-exploit      # pure GTO (no adaptation)
python core/game_engine.py --iters 100000    # more training iterations
```

---

## Exploitability

Exploitability is computed using **tree-traversal best response** - O(|game tree| × |deals|), not O(2^|info sets|).

The algorithm:
1. **Accumulate**: Walk the tree for each deal, summing action values per information set
2. **Maximize**: At each info set, pick the action with highest accumulated value
3. **Evaluate**: Re-traverse using the constructed best-response strategy

This correctly respects imperfect-information constraints (the BR player cannot see the opponent's private card). Cross-validated against brute-force enumeration on Kuhn.

A **game-agnostic** `compute_exploitability_generic()` function works for any game implementing the standard interface.

---

## Opponent Modeling

Four components work together **in a live game loop**:

**Dirichlet Posterior Model** - maintains a per-infoset Bayesian posterior over opponent action frequencies. Reports posterior mean, credible intervals, KL divergence from Nash, and posterior variance.

**SPRT Leak Detector** - Sequential Probability Ratio Test for detecting statistically significant deviations from GTO. Triggers exploitation only when evidence meets calibrated false-positive/negative thresholds.

**Confidence-Gated Strategy Mixer** - blends GTO and exploitative strategies as a function of SPRT signal and posterior precision. Does not exploit unless deviation is both present and statistically reliable.

**Range Model** - Bayesian hand-range updater. Tracks P(opponent holds hand h | observed actions) using Bayes' theorem with Cromwell's rule.

---

## Glosten-Milgrom Isomorphism

The `trading/` module formalizes the Kuhn ↔ GM (1985) correspondence:

| Poker | Market |
|---|---|
| Player 1 with private card | Informed trader who knows asset value |
| Player 2 with no private info | Market maker |
| Bluffing with Jack (weak hand) | Noise trader submitting buy order |
| Always betting King (strong hand) | Informed trader buying at V_H |
| Nash bluff frequency α | Adverse selection component of bid-ask spread |

### Functional Trading Output

The `TradingSignalGenerator` produces:
- **Fair spread** from observed bluff frequency
- **Edge detection** (deviation from Nash spread)
- **Trading signals**: BUY_SPREAD / SELL_SPREAD / HOLD
- **Position sizing** (Kelly-inspired, half-Kelly for safety)

### Known Limitations (explicitly documented in code and tests)

1. **Discrete vs continuous values** - 3 cards vs continuous price processes
2. **Single period vs dynamic** - no time-varying spreads or inventory accumulation
3. **No inventory risk** - infinite capital, risk-neutral MM assumption
4. **No multi-asset correlation** - single-asset model, no cross-asset leakage
5. **Adverse selection only** - no latency arbitrage, market impact, or maker-taker effects

Each limitation is **tested** in `tests/test_trading.py::TestGMLimitations`.

---

## Repository Structure

```
core/
  kuhn_poker.py       - Kuhn game engine, terminal conditions, infoset encoding
  leduc_poker.py      - Leduc game engine, solver, and game interface
  cfr.py              - Vanilla CFR
  cfr_plus.py         - CFR+ (regret clipping, linear averaging)
  dcfr.py             - Discounted CFR
  mccfr.py            - External Sampling MCCFR + variance diagnostics
  exploitability.py   - Tree-traversal + brute-force + generic exploitability
  game_engine.py      - Interactive CLI bot (wires all components together)

opponent_model/
  dirichlet_model.py  - Dirichlet-Multinomial Bayesian opponent model
  hypothesis_test.py  - SPRT-based leak detector
  strategy_mixer.py   - Confidence-gated GTO/exploit blending
  range_model.py      - Bayesian hand-range updater

analysis/
  convergence_plots.py  - Generates plots and metrics table
  convergence_kuhn.png  - Solver convergence curves
  convergence_metrics.csv

trading/
  glosten_milgrom.py  - GM isomorphism + TradingSignalGenerator

tests/
  test_kuhn_game.py         - Kuhn game mechanics (parametrized, zero-sum)
  test_cfr_convergence.py   - Exact Nash values, tight tolerances, cross-validation
  test_leduc_game.py        - Leduc mechanics + solver convergence + exploitability
  test_exploitability.py    - Tree-traversal vs brute-force cross-validation
  test_opponent_model.py    - Dirichlet, SPRT, mixer, range model
  test_trading.py           - GM mechanics, isomorphism, limitations
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

### Play against the bot
```bash
python core/game_engine.py
```

### Full validation pipeline (12 gates)
```bash
python notebooks/run_everything.py
```

### Tests only
```bash
pytest tests/ -v
```

### Trading module
```bash
python trading/glosten_milgrom.py
```

---

## Testing

**117 tests** across 6 test files, all using proper pytest:

| Test File | Coverage |
|---|---|
| `test_kuhn_game.py` | Game mechanics, payoffs, zero-sum, info sets |
| `test_cfr_convergence.py` | Exact Nash values (0.01 tolerance), convergence ordering |
| `test_leduc_game.py` | Leduc mechanics, solver convergence, exploitability decrease |
| `test_exploitability.py` | Tree-traversal vs brute-force cross-validation |
| `test_opponent_model.py` | Dirichlet posterior, SPRT detection, mixer confidence, range model |
| `test_trading.py` | GM mechanics, Kuhn mapping, limitation documentation |

### Key Verification Properties

- **Exact Nash values**: P1 Q:b call = exactly 1/3, P1 J:c bet = exactly 1/3 (tolerance: 0.01)
- **Equilibrium family**: K:bet = 3×J:bet, Q:cb call = J:bet + 1/3
- **Zero-sum**: all terminal payoffs satisfy P0 + P1 = 0
- **Cross-validation**: tree-traversal exploitability matches brute-force to 0.002
- **Convergence ordering**: CFR+ < DCFR < Vanilla at same iteration count
- **Exploitability decrease**: more iterations → lower exploitability (both Kuhn and Leduc)

---

## CI

GitHub Actions runs on every push/PR against Python 3.11 and 3.12:
- Full pytest suite (all 6 test files, including Leduc)
- 12-gate validation pipeline
- Game engine import verification
- Syntax check on all Python modules

---

## Technical Notes

**Kuhn equilibrium family.** Kuhn Poker admits a continuum of P0 equilibria parameterized by bluff frequency α ∈ [0, 1/3]. P1's strategy is unique. Tests verify membership in the equilibrium family using exact relationships, not loose structural checks.

**Scalable exploitability.** The tree-traversal best response works for any game size. The brute-force method is kept explicitly labeled as O(2^|info_sets|) legacy code for cross-validation only.

**Leduc Poker.** Fully implemented with CFR+ solver, game interface for generic exploitability, and comprehensive test coverage. Not experimental - in CI.

**GM limitations.** The code explicitly documents and tests where the Kuhn ↔ GM analogy breaks down: continuous values, dynamic pricing, inventory risk, multi-asset correlation, and non-adverse-selection microstructure effects.

---

## Dependencies

Runtime: `numpy`, `scipy`, `matplotlib`
Dev: `pytest` (see `requirements-dev.txt`)
CI: GitHub Actions (`.github/workflows/ci.yml`)