"""Flat-rate integral-coupled trend.

Variant of ``JointDPWIntegralTrend`` that keeps the integral changepoint layer
and error-correction scan, but removes the dense rate changepoint layer.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .joint_dpw_integral import JointDPWIntegralTrend

__all__ = ["JointFlatIntegralTrend"]


class JointFlatIntegralTrend(JointDPWIntegralTrend):
    """Integral-coupled trend with a flat rate layer.

    This keeps the sparse integral changepoints and error-correction scan from
    ``JointDPWIntegralTrend`` but removes the medium-frequency rate
    changepoint/damping layer entirely.
    """

    _tags = {"capability:panel": True}

    def __init__(
        self,
        integral_cp_interval: int = 26,
        integral_cp_range: float = 0.9,
        integral_cp_prior_scale: float = 0.01,
        mu_prior_scale_frac: float = 0.2,
        integral_damping_beta_a: float = 100.0,
        integral_damping_beta_b: float = 1.0,
        kappa_prior_scale: float = 0.1,
        integral_obs_enabled: bool = True,
        integral_obs_distribution: str = "laplace",
        integral_obs_noise_scale: float = 1.0,
        integral_obs_subsample_stride: int = 4,
        integral_obs_fixed_scale: bool = False,
    ):
        super().__init__(
            integral_cp_interval=integral_cp_interval,
            integral_cp_range=integral_cp_range,
            integral_cp_prior_scale=integral_cp_prior_scale,
            rate_cp_interval=1,
            rate_cp_range=0.0,
            rate_cp_prior_scale=0.0,
            mu_prior_scale_frac=mu_prior_scale_frac,
            integral_damping_beta_a=integral_damping_beta_a,
            integral_damping_beta_b=integral_damping_beta_b,
            rate_damping_beta_a=1.0,
            rate_damping_beta_b=1.0,
            kappa_prior_scale=kappa_prior_scale,
            integral_obs_enabled=integral_obs_enabled,
            integral_obs_distribution=integral_obs_distribution,
            integral_obs_noise_scale=integral_obs_noise_scale,
            integral_obs_subsample_stride=integral_obs_subsample_stride,
            integral_obs_fixed_scale=integral_obs_fixed_scale,
        )

    def _fit(self, y, X, scale=1.0):
        super()._fit(y, X, scale)
        self._rate_cp_ts = jnp.array([], dtype=jnp.float32)
        self._n_rate_cps = 0

    def _transform(self, X, fh) -> Dict:
        transformed = super()._transform(X, fh)
        transformed.pop("rate_cp_matrix", None)
        return transformed

    def _predict(
        self,
        data: Dict,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        integral_cp_matrix = data["integral_cp_matrix"]
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

        def ec_step(S_actual, t_idx):
            S_expected_t = expected_integral[t_idx]
            base_rate_t = expected_rate[t_idx]
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
        numpyro.deterministic("kappa_value", kappa)
        numpyro.deterministic(
            "expected_integral",
            expected_integral[selection_ix],
        )
        numpyro.deterministic(
            "actual_integral",
            actual_integral[selection_ix],
        )

        predicted_effects["latent/expected_integral_scaled"] = expected_integral[
            selection_ix
        ]
        predicted_effects["latent/actual_integral_scaled"] = actual_integral[
            selection_ix
        ]

        n_series = getattr(self, "_n_series", 1)
        if n_series > 1:
            return jnp.tile(rates.reshape((1, -1, 1)), (n_series, 1, 1))
        return rates.reshape((-1, 1))
