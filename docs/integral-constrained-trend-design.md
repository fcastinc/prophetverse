# Integral-Constrained Trend: Design Notes

**Date:** 2026-04-01
**Status:** Design — softmax allocation approach selected, ready for trial implementation

## Problem

DualIntegralTrend has the integral and rate loosely coupled via error correction (κ) and a soft integral observation constraint. In practice:

- κ lands small (0.02-0.03) — EC barely corrects
- The integral observation is a soft Laplace penalty — tolerates drift
- The model integral consistently overshoots actual integral (visible in plots)
- The rate fits OK week-to-week, but small errors accumulate in the cumsum

The fundamental issue: the rate and integral can diverge because they're not structurally linked. The rate model "wins" because there are more weekly observations pulling it, and the integral constraint is just one more term competing in the loss.

## Key Insight

The integral (cumulative demand) is easy to forecast — even a piecewise linear model gets R² > 0.95 for mature products. The weekly rate is noisy and hard. But the rate must sum to the integral over any forecast horizon.

**Separate pattern from level:**

1. **Level** (how much total?) — easy. Piecewise linear on cumsum. R² > 0.95.
2. **Pattern** (which weeks get how much?) — hard, but only needs to get relative shape right, not absolute level.

The rate model captures the **pattern** (spikes, seasonality, promotions). The integral model captures the **level** (total demand). Softmax bridges them — keeps the pattern, scales to the level.

The rate model can be "wrong" on the level and it doesn't matter — the softmax fixes it. It just needs to get the shape right (week 6 is higher than week 7, there's a spike in week 3, etc.).

## Architecture: Softmax Budget Allocation

### Level 1: Integral model (budget)
- Piecewise linear with damped changepoints on the cumulative
- No regressors needed — the cumsum is smooth enough
- Reuses Layer 1 math from DualIntegralTrend
- Outputs: S(t) for all t → budget for any window = S(t_end) - S(t_start)

### Level 2: Rate model (logits)
- Produces unconstrained **logit scores** for each timestep
- Logits driven by: regressors, seasonality, changepoints, trend dynamics
- These capture the allocation **pattern** — which weeks are high/low, where spikes are
- Logits do NOT need to be at the right absolute level

### Softmax bridge
```
budget = S(t_end) - S(t_start)         # from integral model
weights = softmax(logits / temperature) # from rate model, sums to 1
rates = budget × weights               # allocated rates, sum exactly to budget
```

- `temperature` controls how uniform vs spiky the allocation is
  - High temp → near-uniform (each week ≈ budget/H)
  - Low temp → logits dominate (spiky weeks get more)
  - Searchable via Optuna

### How it handles spikes
The logits capture spike patterns via regressors (sale codes, promotions) and seasonality. A promotion week gets a high logit → high softmax weight → more of the budget. But the total is still exactly the budget. The spike is "paid for" by other weeks getting slightly less.

## Constraint mechanism: why softmax works

| Property | DualIntegralTrend (current) | Softmax allocation (proposed) |
|---|---|---|
| Constraint type | Soft (EC + Laplace obs) | Exact (softmax sums to 1) |
| NUTS geometry | OK but conflicting gradients | Clean — NUTS operates on unconstrained logits |
| Pattern vs level | Coupled (rate must get both right) | Decoupled (integral=level, logits=pattern) |
| Correlations | Unconstrained | Flexible (logistic-normal, not forced negative like Dirichlet) |
| Regressor support | Effects added to rate | Regressors drive logits |

## Approaches evaluated and rejected

1. **numpyro.factor() penalty** — already tried, conflicts with EC, can't get tight without divergences
2. **Dirichlet allocation** — forces negative correlations between weeks, can't model adjacent-week similarity
3. **Scan with running budget** — sequential dependencies create funnel geometries, ordering-dependent
4. **Conditional (last rate deterministic)** — asymmetric treatment, last period absorbs all error
5. **Post-hoc reconciliation** — model-agnostic but model doesn't learn the constraint structure

## PV integration

The trend `_predict()` method outputs `rates.reshape((-1, 1))`. PV effects are additive on top. The softmax constraint must apply to the **total** (trend + effects), not just the trend.

Two options:
- **(A) Softmax inside trend only** — trend outputs budget-constrained base rates. Effects are additive deviations that can break the constraint. Simpler but constraint is approximate.
- **(B) Softmax on full model output** — similar to current `_model.py` integral obs block. After all effects are summed, apply the softmax normalization. Constraint is exact but requires `_model.py` changes.

For the trial: start with **(A)**. The trend produces budget-constrained base rates. If effects are small relative to the trend (which they should be for mature products), the constraint is approximately maintained. Can move to (B) if needed.

## Implementation plan

### Trial trend: `IntegralBudgetTrend`

Location: `pv-internal/src/prophetverse/effects/trend/integral_budget.py`

```python
class IntegralBudgetTrend(PiecewiseLinearTrend):
    """Budget-constrained trend: integral sets level, softmax allocates.
    
    Level 1: Piecewise linear integral path (damped changepoints)
    Level 2: Logit-based rate allocation (softmax × budget)
    """
    def __init__(
        self,
        # Integral changepoints (same as DualIntegralTrend Layer 1)
        changepoint_interval=26,
        changepoint_range=0.9,
        changepoint_prior_scale=0.01,
        integral_damping_beta_a=100.0,
        integral_damping_beta_b=1.0,
        mu_prior_scale_frac=0.2,
        # Rate logit model
        rate_cp_interval=8,
        rate_cp_range=0.95,
        rate_cp_prior_scale=0.005,
        # Softmax temperature
        temperature=1.0,
        temperature_prior_scale=1.0,  # if learned
        learn_temperature=False,
    ):
```

**`_predict` flow:**
1. Sample integral path parameters (mu, delta_S, phi_S) — same as DualIntegralTrend Layer 1
2. Compute `S(t)` analytically — the cumulative path
3. Compute per-timestep budgets: `budget(t) = S(t) - S(t-1)` — the "expected rate"
4. Sample rate logit parameters (delta_R, maybe rate damping)
5. Compute logits from rate changepoints + expected rate as base
6. Apply `softmax(logits / temperature)` → weights
7. For each segment, `rates = segment_budget × weights`
8. Output rates

**No EC, no scan, no soft integral observation.** The constraint is structural.

### What to test in the notebook
- Does the integral track tightly? (It should — it's the primary model)
- Do the rates capture spikes via regressors?
- How sensitive is the temperature parameter?
- Compare forecast accuracy (nRMSE) vs DualIntegralTrend

## Reusable from DualIntegralTrend

- Layer 1 math (integral path, damped changepoints, `S(t)` computation) — copy directly
- `_fit` method (mu loc/scale from data, rate CP grid construction)
- `_transform` method (changepoint matrices, selection index)
- Constructor parameters for integral changepoints

## Not reusable

- Layer 3 (EC scan) — replaced by softmax
- `_model.py` integral obs block — no longer needed (constraint is structural)
- kappa parameter — eliminated

## Tuning run evidence

Best Optuna trials show:
- kappa_prior_scale: 0.02-0.03 (EC is weak — model doesn't want it)
- integral_noise_prior: 1.4-3.9 (best trial has tightest constraint at 1.4)
- The model prefers the integral observation doing the work over EC
- But even the integral observation isn't tight enough — model integral consistently overshoots actual

Products examined: KR Philly Light Cream, Borden Singles, KR American Singles — all show the same pattern of integral overshoot. Borden Singles has a "kick" (slope change ~1994) that piecewise linear handles well.
