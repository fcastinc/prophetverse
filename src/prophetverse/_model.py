import numpyro
import numpyro.distributions as dist
from prophetverse.effects.base import BaseEffect
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
from prophetverse.utils.numpyro import CacheMessenger


def wrap_with_cache_messenger(model):

    def wrapped(*args, **kwargs):
        with CacheMessenger():
            return model(*args, **kwargs)

    return wrapped


def model(
    y,
    trend_model: BaseEffect,
    trend_data: Dict[str, jnp.ndarray],
    target_model: BaseEffect,
    target_data: Dict[str, jnp.ndarray],
    data: Optional[Dict[str, jnp.ndarray]] = None,
    exogenous_effects: Optional[Dict[str, BaseEffect]] = None,
    **kwargs,
):
    """
    Define the Prophet-like model for univariate timeseries.

    Parameters
    ----------
        y (jnp.ndarray): Array of time series data.
        trend_model (BaseEffect): Trend model.
        trend_data (dict): Dictionary containing the data needed for the trend model.
        data (dict): Dictionary containing the exogenous data.
        exogenous_effects (dict): Dictionary containing the exogenous effects.
        noise_scale (float): Noise scale.
    """

    with CacheMessenger():
        predicted_effects: Dict[str, jnp.ndarray] = {}

        with numpyro.handlers.scope(prefix="trend"):
            trend = trend_model(data=trend_data, predicted_effects=predicted_effects)

        predicted_effects["trend"] = numpyro.deterministic("trend", trend)

        # Exogenous effects
        if exogenous_effects is not None:
            for exog_effect_name, exog_effect in exogenous_effects.items():
                transformed_data = data[exog_effect_name]  # type: ignore[index]

                with numpyro.handlers.scope(prefix=exog_effect_name):
                    effect = exog_effect(transformed_data, predicted_effects)

                effect = numpyro.deterministic(exog_effect_name, effect)
                predicted_effects[exog_effect_name] = effect

        # Auxiliary integral observation.
        # If trend exposes its internal integral path (latent/trend_integral),
        # constrain THAT against cumsum(y). This is tighter because the
        # trend integral is smooth — effects don't disturb it.
        # Otherwise falls back to cumsum(full_model) ≈ cumsum(y).
        if y is not None and getattr(trend_model, 'integral_obs_enabled', False):
            obs_dist_name = getattr(
                trend_model, 'integral_obs_distribution', 'laplace')
            noise_prior_scale = getattr(
                trend_model, 'integral_obs_noise_scale', 1.0)
            stride = getattr(
                trend_model, 'integral_obs_subsample_stride', 4)

            _INTEGRAL_OBS_DISTRIBUTIONS = {
                "laplace": dist.Laplace,
                "normal": dist.Normal,
            }
            obs_dist_cls = _INTEGRAL_OBS_DISTRIBUTIONS.get(
                obs_dist_name, dist.Laplace)

            y_flat = y.flatten()
            obs_cumsum = jnp.cumsum(y_flat)

            # Use trend's internal integral if available (smooth, no effects)
            trend_integral = predicted_effects.get("latent/trend_integral")
            if trend_integral is not None:
                model_cumsum = trend_integral.flatten()
            else:
                # Fallback: cumsum of full model output
                full_mean = sum(
                    eff for name, eff in predicted_effects.items()
                    if not name.startswith("latent/")
                )
                model_cumsum = jnp.cumsum(full_mean.flatten())

            # Subsample every Nth point
            n_obs = len(obs_cumsum)
            sub = jnp.arange(stride - 1, n_obs, stride)

            mean_rate = obs_cumsum[-1] / n_obs
            fixed_scale = getattr(
                trend_model, 'integral_obs_fixed_scale', False)
            if fixed_scale:
                # Fixed scale — model can't learn to loosen the constraint
                scale = noise_prior_scale * mean_rate + 1e-6
            else:
                # Sampled scale (original behavior)
                integral_noise = numpyro.sample(
                    "integral_noise_scale",
                    dist.HalfNormal(noise_prior_scale),
                )
                scale = integral_noise * mean_rate + 1e-6
            numpyro.sample(
                "integral_obs",
                obs_dist_cls(model_cumsum[sub], scale),
                obs=obs_cumsum[sub],
            )

        # Log-softmax budget constraint.
        # After all effects are summed, apply log-softmax per window so
        # rates sum to the integral budget. Log-transform preserves ratios
        # without needing a temperature parameter.
        # Activated by budget_constraint_enabled on the trend.
        if getattr(trend_model, 'budget_constraint_enabled', False):
            window_size = getattr(trend_model, 'budget_window_size', 26)
            step_budgets = predicted_effects.get("latent/step_budgets")

            if step_budgets is not None:
                # Compute total model output (trend + all effects)
                total = sum(
                    eff for name, eff in predicted_effects.items()
                    if not name.startswith("latent/")
                )
                total_flat = total.flatten()
                T = len(total_flat)

                # Normalize step budgets to model's internal space.
                # Step budgets are diff1(integral) in raw units.
                # Model operates in y/max(y) space. Use y to get scale.
                step_flat = step_budgets.flatten()
                if y is not None:
                    data_scale = jnp.max(jnp.abs(y)) + 1e-10
                    step_flat = step_flat / data_scale

                # Adjust window size to divide evenly
                n_windows = max(1, round(T / window_size))
                actual_win = T // n_windows
                usable_T = n_windows * actual_win

                # Log-softmax per window
                total_positive = jnp.maximum(total_flat[:usable_T], 1e-6)
                log_total = jnp.log(total_positive)
                windowed = log_total.reshape(n_windows, actual_win)
                weights = jax.nn.softmax(windowed, axis=1)

                # Budget per window = sum of step budgets
                budgets = step_flat[:usable_T].reshape(
                    n_windows, actual_win).sum(axis=1)

                constrained = (weights * budgets[:, None]).flatten()

                # Append remainder
                if usable_T < T:
                    constrained = jnp.concatenate([
                        constrained, total_flat[usable_T:]])

                # Replace effects with constrained total
                constrained = constrained.reshape(total.shape)
                for name in list(predicted_effects.keys()):
                    if not name.startswith("latent/"):
                        predicted_effects[name] = jnp.zeros_like(
                            predicted_effects[name])
                predicted_effects["trend"] = numpyro.deterministic(
                    "budget_constrained_total", constrained)

        target_model.predict(target_data, predicted_effects)
