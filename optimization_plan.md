# EA Operator Optimization – Catan Settlement Placement

## Problem Summary

The EA evolves a **13-dimensional weight vector** that greedily scores board vertices to maximize pip sum for settlements against a random opponent.

Current operators:
| Operator | Current Implementation | Issue |
|---|---|---|
| Recombination | Uniform crossover (50/50 per gene) | Destroys promising weight *combinations*; high variance child |
| Mutation | Gaussian with fixed σ=0.15 | Too aggressive in late gens (slows convergence); can't self-tune |
| Selection | 3-way tournament | Fine as-is |
| Elitism | 1 best carried over | Fine as-is |

**Goal**: Faster convergence to a higher pip-sum fitness, with explainable operator choices.

---

## Proposed Changes

### Component: `Interactive_random_board.py`

#### [MODIFY] [Interactive_random_board.py](file:///c:/Users/Yikai/OneDrive%20-%20Queen's%20University/Desktop/455%20GP/Interactive_random_board.py)

**1. Replace `recombine` — Uniform → BLX-α (Blend Crossover)**

> **Why?**  
> Uniform crossover treats each weight independently and is equivalent to random selection from one of the two parents per gene. For a real-valued vector like this weight vector, **BLX-α** samples the child from the interval `[min(a,b) − α·range, max(a,b) + α·range]`. This:
> - Preserves *relative ordering* and *scale* of weights between parents (exploitation)
> - Still introduces controlled exploration via α (α=0.3 is standard)
> - Empirically outperforms uniform crossover for real-coded EAs (Eshelman & Schaffer 1993)

```python
def recombine(pa, pb, alpha=0.3):
    """BLX-α blend crossover for real-valued weight vectors."""
    child = []
    for a, b in zip(pa, pb):
        lo, hi = min(a, b), max(a, b)
        spread = (hi - lo) * alpha
        child.append(random.uniform(lo - spread, hi + spread))
    return [max(0.0, w) for w in child]  # clip to non-negative
```

**2. Replace `mutate` — Fixed σ → Self-Adaptive (1/5-succuss rule approximation)**

> **Why?**  
> A fixed σ of 0.15 is blind to the current state of convergence. Early on it may be too small (slow exploration); late on it may be too large (disrupts near-optimal solutions). **Self-adaptive σ** per individual (each individual carries its own σ) is the canonical ES approach. We use a simplified version: each individual is a `(weights, sigma)` pair, and σ is mutated log-normally before the weights.

> - Early gens: large σ enables wide exploration  
> - Late gens: σ shrinks naturally as individuals converge → fine-grained exploitation  
> - Explains *why* we converge faster: the operator self-adjusts to the fitness landscape

```python
SIGMA_INIT   = 0.20   # starting step size
SIGMA_MIN    = 0.005  # lower bound
SIGMA_TAU    = 1.0 / math.sqrt(NUM_WEIGHTS)  # log-normal learning rate

def mutate(weights, sigma=None):
    """Self-adaptive Gaussian mutation with per-individual step size."""
    if sigma is None:
        sigma = SIGMA_INIT
    # Mutate sigma first (log-normal)
    sigma_new = max(SIGMA_MIN, sigma * math.exp(SIGMA_TAU * random.gauss(0, 1)))
    # Mutate weights
    new_weights = [max(0.0, w + random.gauss(0, sigma_new)) for w in weights]
    return new_weights, sigma_new
```

> The population representation changes from `list[float]` to `(list[float], float)` — weights + σ. The `run_ea` loop is updated to unpack/repack these pairs.

**3. Update `run_ea` to carry σ per individual**

The population changes from `[[w0..w12], ...]` to `[([w0..w12], sigma), ...]`. The fitness evaluation, selection, crossover, and mutation calls are updated accordingly. The diversity tracking now also tracks mean σ across the population.

**4. Fitness mode: use `"absolute"` for pure pip-sum maximization**

> Since your goal is to **maximize pip sum** (not beat the opponent), the `"absolute"` fitness mode is the correct choice for training. `"difference"` adds noise from the random opponent, which slows convergence. We set `_fitness_mode[0] = 2` (index of `"absolute"`) at startup.

**5. Minor: increase `EA_NUM_BOARDS` slightly for stability**

Change `EA_NUM_BOARDS = 30` → `EA_NUM_BOARDS = 40` to reduce noise in fitness estimation (better signal for σ self-adaptation).

---

### Component: `run_ea_only.py`

#### [MODIFY] [run_ea_only.py](file:///c:/Users/Yikai/OneDrive%20-%20Queen's%20University/Desktop/455%20GP/run_ea_only.py)

No functional change needed. The file calls `run_ea()` from the main module — all improvements are automatically available once the main file is updated.

---

## Verification Plan

### Automated Test (terminal)

Run the non-interactive script and compare `best fitness` at generation 50 vs the baseline:

```
cd "c:\Users\Yikai\OneDrive - Queen's University\Desktop\455 GP"
python run_ea_only.py
```

Expected output: `ea_results.png` file saved locally. Check:
- Panel 1 (Convergence): best fitness curve should reach its plateau **faster** (fewer gens) vs baseline
- Panel 4 (Mutation Effect): mutation delta should start high and decay → σ is shrinking
- Panel 6 (Best Weights): weights should cluster around interpretable high values for `pip_sum`, `max_pip`, `count_high`

### Manual Comparison

To visually compare baseline vs optimized:
1. In `Interactive_random_board.py`, temporarily revert operators to originals and run 'e'  → note best fitness
2. Apply optimized operators, run 'e' again → compare best fitness printed in terminal

> Since there are no existing automated unit tests in the repo, verification is done via the built-in stats tracking (convergence plot + boxplot) already present in the codebase.
