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

        # Auxiliary integral observation — on the FULL model output.
        # Constrains cumsum(full_model) ≈ cumsum(y_observed).
        # Only active during training (y is not None).
        # Only active if the trend requests it via integral_obs_enabled.
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

            # Compute full model mean (same as likelihood does)
            full_mean = sum(
                eff for name, eff in predicted_effects.items()
                if not name.startswith("latent/")
            )
            full_mean_flat = full_mean.flatten()
            y_flat = y.flatten()

            model_cumsum = jnp.cumsum(full_mean_flat)
            obs_cumsum = jnp.cumsum(y_flat)

            # Subsample every Nth point
            n_obs = len(obs_cumsum)
            sub = jnp.arange(stride - 1, n_obs, stride)

            integral_noise = numpyro.sample(
                "integral_noise_scale",
                dist.HalfNormal(noise_prior_scale),
            )
            mean_rate = obs_cumsum[-1] / n_obs
            scale = integral_noise * mean_rate + 1e-6
            numpyro.sample(
                "integral_obs",
                obs_dist_cls(model_cumsum[sub], scale),
                obs=obs_cumsum[sub],
            )

        # Windowed budget constraint — softmax renormalization.
        # After all effects are summed, renormalize per window so rates
        # sum to the integral budget. Activated by budget_constraint_enabled.
        if y is not None and getattr(trend_model, 'budget_constraint_enabled', False):
            window_size = getattr(trend_model, 'budget_window_size', 13)
            integral = predicted_effects.get("latent/expected_integral")
            selection_ix = predicted_effects.get("latent/selection_ix")

            # Compute total model output (trend + all effects)
            total = sum(
                eff for name, eff in predicted_effects.items()
                if not name.startswith("latent/")
            )
            total_flat = total.flatten()
            T = len(total_flat)

            # Integral at selected indices
            integral_selected = integral[selection_ix]

            # Compute per-window budgets from integral differences
            # Budget for window [start, end) = S(end) - S(start)
            # Use S values with prepended 0 for first window
            integral_with_zero = jnp.concatenate([jnp.array([0.0]), integral_selected])

            # Adjust window size to divide evenly into T.
            # E.g., T=374, target=26 → 14.38 windows → 14 windows
            # → actual window = 374//14 = 26 (remainder 10 distributed)
            n_windows = max(1, round(T / window_size))
            window_size = T // n_windows
            usable_T = n_windows * window_size

            # Reshape usable portion into (n_windows, window_size)
            logits_windowed = total_flat[:usable_T].reshape(n_windows, window_size)

            # Budget per window
            window_starts = jnp.arange(0, usable_T, window_size)
            window_ends = window_starts + window_size
            budgets = integral_with_zero[window_ends] - integral_with_zero[window_starts]

            # Softmax per window, scale by budget
            weights = jax.nn.softmax(logits_windowed, axis=1)  # (n_windows, window_size)
            constrained_windowed = weights * budgets[:, None]  # broadcast budget per window

            # Flatten back
            constrained = constrained_windowed.flatten()

            # Append remainder (unconstrained pass-through)
            if usable_T < T:
                constrained = jnp.concatenate([
                    constrained, total_flat[usable_T:]
                ])

            # Scale all effects proportionally to preserve decomposition.
            # ratio = constrained / total. Each effect gets multiplied by
            # the ratio so their sum equals the constrained total.
            constrained = constrained.reshape(total.shape)
            ratio = constrained / (total + 1e-10)
            for name in list(predicted_effects.keys()):
                if not name.startswith("latent/"):
                    predicted_effects[name] = (
                        predicted_effects[name] * ratio
                    )
            numpyro.deterministic(
                "budget_constrained_total", constrained
            )

        target_model.predict(target_data, predicted_effects)
