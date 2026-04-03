"""Integral-coupled trend that lets the full model own the cumulative constraint.

This is a clean rebuild of the DualIntegralTrend idea:

* keep the same integral changepoint layer, rate changepoint layer, and
  error-correction scan for the trend itself
* do NOT expose a separate ``latent/trend_integral`` to ``_model.py``
* instead let ``_model.py`` constrain ``cumsum(trend + effects)`` directly

Why this exists
---------------
``DualIntegralTrend`` exposes ``expected_integral`` as ``latent/trend_integral``.
That causes the auxiliary integral observation in ``_model.py`` to constrain the
smooth latent path instead of the actual cumulative implied by the realized rate
and exogenous effects. For count likelihoods this also creates a unit mismatch:
the trend lives in scaled model space while ``y`` in the target likelihood is raw.

This trend opts out of that override so the integral observation can operate on
the full model output after all effects are summed.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from .piecewise import PiecewiseLinearTrend

__all__ = ["FullModelIntegralTrend"]


class FullModelIntegralTrend(PiecewiseLinearTrend):
    """Dual-layer integral trend that defers cumulative coupling to the full model.

    The trend keeps the same internal structure as ``DualIntegralTrend``:

    1. sparse integral changepoints for slow structural drift
    2. dense rate changepoints for medium-frequency dynamics
    3. an error-correction scan that ties the realized rate back to the
       expected integral path

    Unlike ``DualIntegralTrend``, it does not expose ``latent/trend_integral``.
    That forces ``prophetverse._model`` to use ``cumsum(trend + effects)`` for the
    integral observation, which is the object we actually care about constraining.
    """

    def __init__(
        self,
        changepoint_interval: int = 26,
        changepoint_range: float = 0.9,
        changepoint_prior_scale: float = 0.01,
        rate_cp_interval: int = 8,
        rate_cp_range: float = 0.95,
        rate_cp_prior_scale: float = 0.005,
        mu_prior_scale_frac: float = 0.2,
        integral_damping_beta_a: float = 100.0,
        integral_damping_beta_b: float = 1.0,
        rate_damping_beta_a: float = 20.0,
        rate_damping_beta_b: float = 1.0,
        kappa_prior_scale: float = 0.1,
        integral_obs_enabled: bool = True,
        integral_obs_distribution: str = "laplace",
        integral_obs_noise_scale: float = 1.0,
        integral_obs_subsample_stride: int = 4,
        integral_obs_fixed_scale: bool = False,
    ):
        self.rate_cp_interval = rate_cp_interval
        self.rate_cp_range = rate_cp_range
        self.rate_cp_prior_scale = rate_cp_prior_scale
        self.mu_prior_scale_frac = mu_prior_scale_frac
        self.integral_damping_beta_a = integral_damping_beta_a
        self.integral_damping_beta_b = integral_damping_beta_b
        self.rate_damping_beta_a = rate_damping_beta_a
        self.rate_damping_beta_b = rate_damping_beta_b
        self.kappa_prior_scale = kappa_prior_scale

        # Read by prophetverse._model.
        self.integral_obs_enabled = integral_obs_enabled
        self.integral_obs_distribution = integral_obs_distribution
        self.integral_obs_noise_scale = integral_obs_noise_scale
        self.integral_obs_subsample_stride = integral_obs_subsample_stride
        self.integral_obs_fixed_scale = integral_obs_fixed_scale

        # Fresh behavior: constrain cumsum(full_model), not a separate latent path.
        self.integral_obs_use_trend_integral = False
        # For discrete likelihoods, y reaches _model.py in raw units while the
        # model mean is still in scaled space. Ask _model.py to normalize obs.
        self.integral_obs_scale_mode = "divide_observed_by_data_scale"

        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
        )

    def _fit(self, y, X, scale=1.0):
        super()._fit(y, X, scale)
        y_mean = float(y.values.flatten().mean())
        self._mu_loc = y_mean
        self._mu_scale = max(abs(y_mean) * self.mu_prior_scale_frac, 1.0)
        self._train_fh = y.index.get_level_values(-1).unique().sort_values()

        t_scaled_train = self._index_to_scaled_timearray(self._train_fh)
        n_train = len(t_scaled_train)
        rate_cp_range_idx = int(n_train * self.rate_cp_range)
        cp_obs_indices = np.arange(
            self.rate_cp_interval,
            rate_cp_range_idx,
            self.rate_cp_interval,
        )
        self._rate_cp_ts = jnp.array(
            [float(t_scaled_train[i]) for i in cp_obs_indices],
            dtype=jnp.float32,
        )
        self._n_rate_cps = len(self._rate_cp_ts)

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict:
        requested_periods = fh
        all_periods = self._train_fh.append(
            requested_periods.difference(self._train_fh)
        ).sort_values()

        t_scaled = self._index_to_scaled_timearray(all_periods)
        integral_cp_matrix = self.get_changepoint_matrix(all_periods)

        t_arr = jnp.array(t_scaled, dtype=jnp.float32)
        rate_cp_matrix = jnp.maximum(
            0, t_arr[:, None] - self._rate_cp_ts[None, :]
        )

        selection_ix = np.array(
            [all_periods.get_loc(p) for p in requested_periods],
            dtype=np.int32,
        )

        return {
            "integral_cp_matrix": integral_cp_matrix,
            "rate_cp_matrix": rate_cp_matrix,
            "selection_ix": jnp.array(selection_ix),
            "n_total": len(all_periods),
        }

    def _predict(
        self,
        data: Dict,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        integral_cp_matrix = data["integral_cp_matrix"]
        rate_cp_matrix = data["rate_cp_matrix"]
        selection_ix = data["selection_ix"]
        T = data["n_total"]

        mu = numpyro.sample(
            "mean_rate",
            dist.Normal(self._mu_loc, self._mu_scale),
        )

        delta_S = numpyro.sample(
            "integral_changepoints",
            dist.Laplace(
                self._changepoint_prior_loc,
                self._changepoint_prior_scale,
            ),
        )

        phi_S = numpyro.sample(
            "integral_damping",
            dist.Beta(self.integral_damping_beta_a, self.integral_damping_beta_b),
        )

        delta_R = numpyro.sample(
            "rate_changepoints",
            dist.Laplace(
                jnp.zeros(self._n_rate_cps),
                jnp.full(self._n_rate_cps, self.rate_cp_prior_scale),
            ),
        )

        psi = numpyro.sample(
            "rate_damping",
            dist.Beta(self.rate_damping_beta_a, self.rate_damping_beta_b),
        )

        kappa = numpyro.sample(
            "kappa",
            dist.HalfNormal(self.kappa_prior_scale),
        )

        phi_S_safe = jnp.clip(phi_S, 1e-6, 1.0 - 1e-6)
        C_S = phi_S_safe * (-jnp.log(phi_S_safe)) / (1.0 - phi_S_safe)

        h_S = integral_cp_matrix
        log_phi_S = jnp.log(phi_S_safe)
        log_decay_S = jnp.clip(h_S * log_phi_S, -50.0, 0.0)
        phi_S_h = jnp.exp(log_decay_S)
        decay_S = jnp.where(integral_cp_matrix > 0, phi_S_h, 0.0)
        expected_rate = mu + C_S * (decay_S @ delta_S)

        damped_ramp_S = jnp.where(
            integral_cp_matrix > 0,
            phi_S_safe * (1.0 - phi_S_h) / (1.0 - phi_S_safe),
            0.0,
        )
        t_indices = jnp.arange(T, dtype=jnp.float32)
        expected_integral = mu * t_indices + damped_ramp_S @ delta_S

        psi_safe = jnp.clip(psi, 1e-6, 1.0 - 1e-6)
        h_R = rate_cp_matrix
        log_psi = jnp.log(psi_safe)
        log_decay_R = jnp.clip(h_R * log_psi, -50.0, 0.0)
        decay_R = jnp.where(rate_cp_matrix > 0, jnp.exp(log_decay_R), 0.0)
        rate_correction = decay_R @ delta_R

        def ec_step(S_actual, t_idx):
            S_expected_t = expected_integral[t_idx]
            base_rate_t = expected_rate[t_idx] + rate_correction[t_idx]
            deviation = S_actual - S_expected_t
            rate_t = base_rate_t - kappa * deviation
            S_actual_new = S_actual + rate_t
            return S_actual_new, (rate_t, S_actual_new)

        init_S = jnp.array(0.0, dtype=jnp.float32)
        t_steps = jnp.arange(T, dtype=jnp.int32)
        _, (all_rates, actual_integral) = jax.lax.scan(ec_step, init_S, t_steps)

        rates = all_rates[selection_ix]

        numpyro.deterministic("mean_rate_value", mu)
        numpyro.deterministic("integral_damping_value", phi_S)
        numpyro.deterministic("rate_damping_value", psi)
        numpyro.deterministic("kappa_value", kappa)
        numpyro.deterministic(
            "expected_integral",
            expected_integral[selection_ix],
        )
        numpyro.deterministic(
            "actual_integral",
            actual_integral[selection_ix],
        )

        # Diagnostics only; the integral observation should operate on the full model.
        predicted_effects["latent/expected_integral_scaled"] = expected_integral[
            selection_ix
        ]
        predicted_effects["latent/actual_integral_scaled"] = actual_integral[
            selection_ix
        ]

        return rates.reshape((-1, 1))
