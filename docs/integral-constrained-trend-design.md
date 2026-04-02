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

## Experiment Results (2026-04-01)

### NB 10a: Integral-only fit (trend + seasonality, no regressors)
- Train R² > 0.999 for all series
- Endpoint error: 0.3-3.4% (total demand over 26-week forecast)
- **Proves: integral is easy to forecast, viable as budget**

### NB 10c: Integral fit WITH regressors
- Adding regressors (price, sales, macro) to cumulative fit makes it WORSE
- Rate from integral diff shows wild oscillations and negative values
- **Proves: regressors don't belong in the integral model**

### NB 06: DualIntegralTrend (rate model with regressors)
- Rate model captures spikes well (sale codes, promotions work)
- But integral drifts (overshoots consistently)
- **Proves: rate model gets pattern right but level wrong**

### Conclusion
- Integral model (trend-only) → reliable budget, 1-3% error
- Rate model (with regressors) → correct spike pattern, wrong level
- Neither alone is sufficient. Need joint model:
  - Integral level: sets the budget (no regressors)
  - Rate level: allocates budget across weeks (with regressors)
  - Constraint: rates sum to budget per window

### Key design revision
- Softmax over full time range doesn't work (washes out to ~uniform)
- Need **windowed** constraint: rates in each window sum to that window's budget
- Window size = forecast horizon (e.g., 13 weeks)
- Joint fit, not two-stage: one model, one posterior, shared uncertainty
- Regressors + effects operate on logits (rate level only)
- Integral parameters and rate parameters fit simultaneously

## Implementation Attempt Results (2026-04-01, late session)

### What we tried
Single-model approach: IntegralBudgetTrend computes S(t) internally,
outputs base rate, PV effects add on top, _model.py does softmax
per window + integral obs penalty.

### What happened
- Integral obs (soft penalty) doesn't track tightly enough — same problem as DualIntegralTrend
- Without fitting directly on cumsum, the integral parameters float free
- Softmax constraint allocates to wrong budget when integral is loose
- Proportional scaling of effects blew up (ratio near zero)

### What we actually need
**Two-stage pipeline, NOT one model:**

1. **Stage 1: Fit integral**
   - y = cumsum of weekly demand
   - Trend: DampedPiecewiseLinearTrendV3 + seasonality (no regressors)
   - Likelihood: Normal on cumsum
   - Output: S(t) for train + forecast periods
   - This works (NB 10a proved it — 0.3-3.4% endpoint error)

2. **Stage 2: Fit constrained rate**
   - y = weekly demand
   - S(t) from stage 1 is FIXED INPUT (not a parameter)
   - Base rate = diff1(S(t)) — from the integral, not sampled
   - Effects: regressors, seasonality, holidays (standard PV)
   - _model.py: softmax(trend + effects) × window_budget = constrained rates
   - Likelihood: Normal/NegBinomial on constrained rates vs y

### Key insight
The integral observation (soft penalty on cumsum) is NOT the same as
fitting directly on the cumsum. Direct fit gives R² > 0.999. Soft
penalty gives drift and overshoot. The integral must be fit as a
primary target, not an auxiliary constraint.

## Two-Stage Post-Hoc Experiment Results (2026-04-02)

### What we tried
Two-stage pipeline:
- Stage 1: Fit integral (cumsum) with DampedPiecewiseLinearTrendV3 + seasonality via full pipeline
- Stage 2: Fit rate model with full NB 06 pipeline (DampedPiecewiseLinearTrendV3 + regressors + holidays + sales)
- Post-hoc constraint: softmax or proportional scaling using stage 1 budgets

### Softmax results
- Optimal temperatures: 7000-10000 (basically uniform allocation)
- Raw model predictions are too flat OOS — softmax has no variation to preserve
- Budget errors: 9-37% depending on window alignment

### Proportional scaling results
- Preserves raw model pattern, adjusts level
- Worse than softmax — amplifies noise and wrong spikes

### Rolling window
- Tried rolling windows (every position, not just non-overlapping blocks)
- Same result — temperatures still very high, allocation still uniform

### Root cause discovered
The raw rate model (stage 2) doesn't capture spikes OOS even though:
- Sale codes ARE in X_test (nonzero values confirmed)
- The model fits spikes perfectly in-sample
- The sale coefficients are learned but tiny

**Why:** Without a constraint during training, the trend absorbs the spikes
via changepoints. The sale effect coefficients stay small because the trend
does all the work. At forecast time, the changepoints project forward
smoothly — no spikes. The sale coefficients are too small to matter.

**Comparison:** DualIntegralTrend (NB 06) captures OOS spikes because the
integral observation constraint prevents the trend from absorbing everything.
The constraint forces the model to use the sale effects, so the coefficients
are larger and produce spikes OOS.

### Key insight
**The constraint must be inside the model during training, not post-hoc.**

Post-hoc constraint can fix the level but cannot fix the pattern. The pattern
(which weeks get spikes) depends on the effect coefficients, which are
determined during training. If the training is unconstrained, the trend
absorbs the spikes and the effects atrophy.

The integral constraint serves TWO purposes:
1. Level correction (total demand) — can be done post-hoc
2. Effect amplification (forces model to use regressors) — MUST be during training

### Possible paths forward

1. **Tighter integral obs during training** — use the existing `_model.py`
   integral obs block with much lower noise scale. Force the cumsum to match
   tightly. The model has no choice but to use effects for spikes. This is
   DualIntegralTrend with a tighter constraint.

2. **DualIntegralTrend + post-hoc level correction** — the current model
   captures spikes OOS. Apply proportional scaling post-hoc to fix the
   integral drift. Pattern is right, level is adjusted.

3. **Two-stage with constrained training** — fit stage 2 with the integral
   obs block enabled (budget from stage 1 as tight prior). Effects learn
   large coefficients because the trend can't absorb spikes.

### Infrastructure improvements made
- `panels.py`: `y_col` and `y_fillna` parameters for cumsum data
- NB 10a rebuilt clean: single data load, two stages, tune controls
- `load_demand_data` import path fixed
- Backward-compatible `apply_best_params` (.get with defaults)
- `clustering.py` label type fix
