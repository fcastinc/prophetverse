# Integral-Constrained Trend: Design Notes

**Date:** 2026-04-01
**Status:** Early design — captures thinking, not a spec yet

## Problem

DualIntegralTrend has the integral and rate loosely coupled via error correction (κ) and a soft integral observation constraint. In practice:

- κ lands small (0.02-0.03) — EC barely corrects
- The integral observation is a soft Laplace penalty — tolerates drift
- The model integral consistently overshoots actual integral (visible in plots)
- The rate fits OK week-to-week, but small errors accumulate in the cumsum

The fundamental issue: the rate and integral can diverge because they're not structurally linked. The rate model "wins" because there are more weekly observations pulling it, and the integral constraint is just one more term competing in the loss.

## Key Insight

The integral (cumulative demand) is easy to forecast — even a piecewise linear model gets R² > 0.95 for mature products. The weekly rate is noisy and hard. But the rate must sum to the integral over any forecast horizon.

**The integral should be the budget, not the constraint.** It tells the rate model "you have X units to distribute over the next 13 weeks." The rate model handles the allocation (seasonality, promotions, spikes), but the total must land on the integral target.

## Architecture

Two levels, hard-linked:

### Level 1: Integral model
- Piecewise linear with damped changepoints on the cumulative
- No regressors needed — the cumsum is smooth enough
- This is essentially Layer 1 of DualIntegralTrend (expected integral path)
- High confidence, easy to fit
- Outputs: S(t) for all t (cumulative path)

### Level 2: Rate allocation model
- Receives the integral budget: "total demand over forecast horizon = S(T) - S(t_now)"
- Allocates across weeks using:
  - Regressors (price, promotions, sale codes)
  - Seasonality (Fourier terms)
  - Its own changepoints or dynamics for week-to-week variation
- Must satisfy: Σ rate(t) = integral budget (hard or near-hard constraint)
- Handles spikes — the integral doesn't see them, the rate model does

### The constraint
- Not a soft penalty (current approach — doesn't work)
- Not EC pulling the rate (current approach — too weak)
- A **summation constraint**: predicted rates over the period must sum to the integral's prediction
- Could be implemented as:
  - Normalized allocation (rates as fractions of the total, multiplied by budget)
  - A Lagrange multiplier / hard constraint in the numpyro model
  - Post-hoc rescaling (fit rates freely, then scale to match integral — loses uncertainty)

## What this is NOT

- Not a replacement for effects/regressors — those still handle promotions, price, seasonality at the rate level
- Not a two-model pipeline — it's one model with two structural levels
- Not GenLogistic — that assumes an S-curve functional form. This is piecewise linear on the integral, fully flexible on the rate

## Open questions

1. **How to implement the summation constraint in numpyro?** Options:
   - `numpyro.factor()` penalty on `|Σ rate - integral_budget|`
   - Simplex allocation: rates = integral_budget × softmax(raw_weights)
   - Scan that tracks running sum and adjusts remaining budget
   
2. **Where do spikes come from?** If the integral is smooth, all spikes come from effects. Are the current regressors enough to explain all spikes, or does the rate model need its own spike-handling capability (changepoints, impulse effects)?

3. **Forecast horizon matters.** The integral budget for 13 weeks is well-determined. But within those 13 weeks, how is the budget allocated? The allocation pattern depends on seasonality and effects, which may be uncertain even if the total is known.

4. **Training vs forecast.** During training, we observe both the integral and the rate. During forecast, we only have the integral prediction. How does the rate model learn the allocation pattern from training data?

5. **How does this interact with PV's architecture?** The trend outputs rate(t). Effects add to it. The likelihood fits on (trend + effects). If the trend already includes the integral constraint, and effects are additive on top, do the effects break the constraint? Possibly: trend outputs the base allocation, effects are additive deviations, and the total (trend + effects) is what gets constrained to sum to the integral.

## Relationship to existing code

- **DualIntegralTrend Layer 1** — the integral model. This math is reusable.
- **DualIntegralTrend Layer 2** — rate changepoints. May still be useful for the allocation model.
- **DualIntegralTrend Layer 3 (EC)** — replaced by the hard summation constraint.
- **`_model.py` integral obs block** — replaced by the structural constraint inside the trend.
- **GenLogisticTrend** — inspiration for "model the cumulative, derive the rate" approach, but different functional form.

## Tuning run evidence

Best Optuna trials show:
- kappa_prior_scale: 0.02-0.03 (EC is weak — model doesn't want it)
- integral_noise_prior: 1.4-3.9 (best trial has tightest constraint at 1.4)
- The model prefers the integral observation doing the work over EC
- But even the integral observation isn't tight enough — model integral consistently overshoots actual

Products examined: KR Philly Light Cream, Borden Singles, KR American Singles — all show the same pattern of integral overshoot.
