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

## Log-Softmax Budget Constraint (2026-04-02, late session)

### The breakthrough: log-transform before softmax

Regular softmax on raw predictions (values ~10,000) needs a temperature
parameter to avoid saturation. The optimizer finds temperature ≈ 7,000-10,000
which makes the allocation nearly uniform — defeating the purpose.

**Log-transform fixes this.** Take log of predictions before softmax:

```python
log_pred = log(max(pred, 1.0))   # 10,000 → 9.21, 50,000 → 10.82
weights = softmax(log_pred)       # balanced, no temperature needed
constrained = budget × weights    # sum exactly equals budget
```

Key properties:
- **Ratios preserved exactly**: if pred_A / pred_B = 5, constrained_A / constrained_B = 5
- **Sum constrained exactly**: window sum = budget, always
- **No temperature parameter**: log compression handles scale naturally
- **Gradients flow**: log + softmax + multiply are all differentiable
- **JAX-compatible**: works inside numpyro model, NUTS can trace through it

### Why this works when regular softmax didn't

Regular softmax(pred / T):
- T too low → saturates (one week gets everything)
- T too high → uniform (all weeks equal)
- No good middle ground because pred values span narrow range (10K-50K)

Log-softmax(log(pred)):
- log(10K) = 9.21, log(50K) = 10.82 — difference of 1.6
- Softmax on [9.21, 10.82] gives balanced but differentiated weights
- A 5x spike in demand → 5x more budget share, naturally

### Implementation: inside _model.py during training

```python
# After all effects are summed, before likelihood
if getattr(trend_model, 'budget_constraint_enabled', False):
    total = sum(predicted_effects that aren't latent)
    total_flat = total.flatten()
    
    # Get integral path from trend
    integral = predicted_effects["latent/trend_integral"]
    
    # Compute budgets per window
    # ... (windowed integral differences)
    
    # Log-softmax per window
    log_total = jnp.log(jnp.maximum(total_flat[:usable], 1.0))
    windowed = log_total.reshape(n_windows, window_size)
    weights = jax.nn.softmax(windowed, axis=1)
    constrained = (weights * budgets[:, None]).flatten()
    
    # Replace predicted_effects with constrained rates
    # Scale each effect proportionally
    ratio = constrained / (total_flat[:usable] + 1e-10)
    ...
```

### Why it must be during training, not post-hoc

Post-hoc constraint can fix the level but the model doesn't learn:
- Without constraint during training, the trend absorbs spikes via changepoints
- Effect coefficients (sales, holidays) stay small because the trend does all the work
- At forecast time, changepoints project smooth → no spikes
- Post-hoc softmax has no variation to redistribute

With constraint during training:
- The integral is locked (constraint forces cumsum to match)
- The trend CAN'T absorb spikes because the integral holds it
- Effects MUST explain the spikes → larger coefficients
- At forecast time, effects produce spikes → softmax redistributes correctly

### What needs to happen

1. Implement log-softmax constraint in `_model.py` as a new block
   - Activated by `budget_constraint_enabled` flag
   - Uses trend's `latent/trend_integral` for budgets
   - Non-overlapping windows (size = forecast horizon)
   - Shapes must be static for JAX tracing
   
2. The integral path (Layer 1 of DualIntegralTrend) provides the budget
   - No separate stage 1 needed — integral params fit jointly
   - The log-softmax constraint effectively makes the integral authoritative
   - Because rates must sum to S(t) per window, the integral params learn
     to match the actual cumsum

3. Temperature is eliminated — log handles it naturally

4. Test in NB 06 with DualIntegralTrend + budget_constraint_enabled

## ConstrainedIntegralTrend Implementation Attempt (2026-04-02)

### What we built
- `ConstrainedIntegralTrend`: receives fixed S(t) from stage 1, outputs base_rate + rate changepoints
- `_model.py` log-softmax block: after trend + effects summed, constrains per window
- Two-stage pipeline in NB 10e: stage 1 integral fit → stage 2 constrained rate model

### The scaling problem
PV normalizes y internally: `_scale = max(abs(y_raw))`, trend sees `y / _scale`.
The integral path is in raw units (cumsum of raw y). The constraint needs step
budgets in the SAME normalized space as the model output.

The core difficulty: **the trend doesn't have access to PV's `_scale`**.

What we tried:
1. **Pass y_max from notebook** — wrong because y_max was max across ALL UPCs, not per-series. PV computes _scale per series.
2. **max(diff(integral))** as proxy — underestimates because integral is smoothed (no spikes). Stage 1 predicted max_rate ≠ actual max_rate.
3. **Infer scale from mean(raw_steps) / mean(normalized_y)** — WRONG: this made step_budgets ≈ model output, turning the constraint into a no-op.
4. **data_scale from group_data** — correct value but needs verification that it matches PV's _scale exactly.

### Why scaling is so hard
- NegBinomial: PV passes `y / _scale` to trend with `scale=1`. Trend doesn't know _scale.
- Normal: PV passes `y` with `scale=_scale`. Trend knows scale.
- InverseGaussian: same as Normal (non-discrete).
- The trend can't use a one-size-fits-all approach because PV's behavior varies by likelihood.

### Key findings
1. **Log-softmax works mechanically** — gradients flow, ratios preserved, no temperature needed
2. **The constraint must modify the output** — if budget = model prediction, constraint is no-op
3. **The budget must come from the integral (stage 1)** not from the model's own prediction
4. **Scale mismatch between integral and model is the blocking issue**
5. **The model DOES learn meaningful effect coefficients** — sale_b, sale_s, price, holidays all have real values. The effects are there, the constraint just isn't applying them.
6. **Component decomposition showed mean ≈ budget_constrained_total** — confirmed no-op when scales match

### Options for next session
1. **Use InverseGaussian or Normal likelihood** — PV passes raw y with real scale. No NegBinomial scale gymnastics.
2. **Store _scale on the trend during PV's fit pipeline** — modify PV's base.py to expose _scale to the trend. Clean but invasive.
3. **Compute data_scale correctly per series** from group_data in the notebook — we were close, just need to verify the value matches PV's computation exactly.
4. **Skip PV's internal scaling entirely** — set `scale=None` or `scale=1` explicitly on Prophetverse. Priors would need retuning.

### Infrastructure built (reusable)
- `ConstrainedIntegralTrend` class with integral_path, data_scale, rate changepoints
- `_model.py` log-softmax constraint block (step budgets, windowed, non-overlapping)
- NB 10e two-stage pipeline (stage 1 integral → stage 2 constrained rate)
- `panels.py` y_col and y_fillna parameters
- Per-series integral path extraction from stage 1 predictions

## Improvement Ideas for DualIntegralTrend (2026-04-02)

Parking the ConstrainedIntegralTrend scaling issues. Returning to improving
the existing DualIntegralTrend which works and produces spikes.

### Ideas to explore

1. **Better priors on holidays/sales** — all effects currently get Normal(0, 1.4).
   Per-effect priors could help. Sales might need wider priors than macro regressors.
   Low risk, easy to test.

2. **NegBinomial with broadcast_mode="effect"** — fixed the bug for this in
   pv-internal. Enables hierarchical pooling across series. Could improve
   coefficient estimation for rare events.

3. **Effect mode refinement** — Optuna chose additive for sales but might be
   stuck in a local minimum. Multiplicative sales = spikes proportional to base
   level. Worth testing manually.

4. **Time-varying betas (random walk)** — sale coefficients that change over
   time. Products respond differently to promotions as they mature. Would need
   a custom PV effect or scan-based implementation.

5. **Structural improvements** — something we're overlooking? The EC mechanism
   is weak, the integral obs is soft. Maybe a different coupling between
   integral and rate layers.

6. **Post-processing: component-error-weighted allocation** — ★ EXPLORING THIS ★
   Run MCMC, get posterior samples. The integral error tells you how much total
   adjustment is needed. The component posteriors tell you WHERE to allocate it.
   Weight corrections by posterior uncertainty — tight components stay, uncertain
   components absorb the correction. Uses MCMC uncertainty without changing the model.

## Post-hoc Log-Softmax Results (2026-04-02, final)

### What works
DualIntegralTrend (existing, unchanged) + post-hoc log-softmax:
- Use DualIntegralTrend as-is — it captures spike patterns via sale/holiday effects
- Post-hoc: log(predictions) → softmax → weights × budget = corrected rates
- Temperature optimized per series on IS data (lands 0.6-0.9)
- Budget from stage 1 integral forecast (within 2% of actual)

### Results (using actual OOS sum as budget — stage 1 would give ~2% error)
| Series | Endpoint err (orig → corr) | RMSE (orig → corr) |
|--------|---------------------------|---------------------|
| KR Phila Cream | 49% → 0% | 30,626 → 24,292 (↓21%) |
| KR American Singles | 41% → 0% | 25,135 → 22,713 (↓10%) |
| Borden Singles | 37% → 0% | 11,373 → 13,227 (↑16%) |
| KR Philly Light | 35% → 0% | 8,367 → 8,655 (↑3%) |
| Dom Cream Cheese | 7.5% → 0% | 6,026 → 6,166 (↑2%) |

### Why it works
- DualIntegralTrend learns large sale effect coefficients because the integral
  obs constraint prevents the trend from absorbing everything
- The spike PATTERN is correct — just the LEVEL (cumsum total) drifts
- Log-softmax preserves the pattern while fixing the level
- No model changes needed — pure post-processing

### What needs refinement
- Series that were already close (Dom, KR Philly) get slightly worse RMSE
- The correction is too aggressive on weeks where the model was already right
- Need: error budget (allow ±X% slack), rolling correction, per-component weighting
- Temperature optimizer could be improved (currently IS-only, global per series)

### Implementation path
1. Add `correct_forecast(predictions, integral_budgets)` function to mature_product
2. Accepts stage 1 integral budgets per series
3. Applies log-softmax with per-series optimized temperature
4. Returns corrected predictions in same format
5. Wrap stage 1 + stage 2 + correction in one pipeline function
