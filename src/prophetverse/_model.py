import numpyro
import numpyro.distributions as dist
from prophetverse.effects.base import BaseEffect
from typing import Any, Dict, Optional
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

        target_model.predict(target_data, predicted_effects)
