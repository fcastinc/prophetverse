"""Dual integral trend — integral changepoints + rate changepoints + EC.

Two independent changepoint layers operating at different scales:

1. **Integral changepoints** (sparse, slow) — model structural drift in
   the cumulative. Defines the expected integral path S_expected(t).

2. **Rate changepoints** (dense, fast-damping) — model medium-frequency
   rate dynamics on top of the integral-derived rate. Captures sustained
   level shifts, seasonal amplitude changes, etc.

3. **Error correction** — OU process on the integral residual. Keeps the
   actual cumulative on the expected path.

## Math

### Layer 1: Expected integral path (analytical)

    S_expected(t) = μ·t + Σⱼ δⱼ^S · f_S(hⱼ)
    expected_rate(t) = μ + C(φ_S) · Σⱼ δⱼ^S · φ_S^hⱼ

    where f_S(h) = φ_S·(1 - φ_S^h) / (1 - φ_S)
    Sparse changepoints, slow damping (φ_S close to 1)

### Layer 2: Rate correction (analytical)

    rate_correction(t) = Σₖ δₖ^R · ψ^hₖ

    Dense changepoints, fast damping (ψ < φ_S).
    These are direct impulses on the rate, NOT on the integral.
    They capture dynamics the integral derivative can't.

### Layer 3: Error correction (sequential, via scan)

    deviation(t) = S_actual(t) - S_expected(t)
    rate(t) = expected_rate(t) + rate_correction(t) - κ · deviation(t)
    S_actual(t+1) = S_actual(t) + rate(t)

## Why two changepoint layers?

A single δⱼ on the integral has to simultaneously explain:
- Slow integral drift (needs large δ, slow decay)
- Sharp rate dynamics (needs small δ, fast decay)

These are different parameter scales. Coupling them through the derivative
forces a tradeoff. Two layers let each operate at its natural scale.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from .piecewise import PiecewiseLinearTrend

__all__ = ["DualIntegralTrend"]


class DualIntegralTrend(PiecewiseLinearTrend):
    """Dual-layer trend: integral changepoints + rate changepoints + EC.

    Subclasses PiecewiseLinearTrend for the integral changepoint grid.
    Builds a second, denser changepoint grid for the rate layer internally.

    Parameters
    ----------
    changepoint_interval : int
        Interval for integral changepoints (sparse).
    changepoint_range : float
        Range for integral changepoints.
    changepoint_prior_scale : float
        Prior scale for integral changepoint magnitudes.
    rate_cp_interval : int
        Interval for rate changepoints (dense).
    rate_cp_range : float
        Range for rate changepoints.
    rate_cp_prior_scale : float
        Prior scale for rate changepoint magnitudes.
    mu_prior_scale_frac : float
        Prior scale on μ as fraction of |mean(y)|.
    integral_damping_beta_a, integral_damping_beta_b : float
        Beta prior on integral damping φ_S (slow, near 1).
    rate_damping_beta_a, rate_damping_beta_b : float
        Beta prior on rate damping ψ (faster).
    kappa_prior_scale : float
        HalfNormal scale for EC speed κ.
    integral_obs_enabled : bool
        Enable auxiliary integral observation constraint in _model.py.
    integral_obs_distribution : str
        Distribution for integral observation: 'laplace' or 'normal'.
    integral_obs_noise_scale : float
        HalfNormal prior scale for integral observation noise.
    integral_obs_subsample_stride : int
        Subsample every Nth point for integral observation.
    """

    def __init__(
        self,
        # Integral changepoints (sparse, slow)
        changepoint_interval: int = 26,
        changepoint_range: float = 0.9,
        changepoint_prior_scale: float = 0.01,
        # Rate changepoints (dense, fast)
        rate_cp_interval: int = 8,
        rate_cp_range: float = 0.95,
        rate_cp_prior_scale: float = 0.005,
        # Mean rate
        mu_prior_scale_frac: float = 0.2,
        # Integral damping (slow — near 1)
        integral_damping_beta_a: float = 100.0,
        integral_damping_beta_b: float = 1.0,
        # Rate damping (faster)
        rate_damping_beta_a: float = 20.0,
        rate_damping_beta_b: float = 1.0,
        # Error correction
        kappa_prior_scale: float = 0.1,
        # Integral observation config (read by _model.py)
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
        # Integral observation config (read by _model.py)
        self.integral_obs_enabled = integral_obs_enabled
        self.integral_obs_distribution = integral_obs_distribution
        self.integral_obs_noise_scale = integral_obs_noise_scale
        self.integral_obs_subsample_stride = integral_obs_subsample_stride
        self.integral_obs_fixed_scale = integral_obs_fixed_scale
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


        # Build rate changepoint grid in the SAME scaled-time space as
        # the parent's integral changepoints (days-since-epoch / t_scale).
        # Place CPs at regular observation intervals along the training range.
        t_scaled_train = self._index_to_scaled_timearray(self._train_fh)
        n_train = len(t_scaled_train)
        rate_cp_range_idx = int(n_train * self.rate_cp_range)
        cp_obs_indices = np.arange(
            self.rate_cp_interval,
            rate_cp_range_idx,
            self.rate_cp_interval,
        )
        # Map observation indices to scaled time values
        self._rate_cp_ts = jnp.array(
            [float(t_scaled_train[i]) for i in cp_obs_indices],
            dtype=jnp.float32,
        )
        self._n_rate_cps = len(self._rate_cp_ts)

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict:
        """Build full changepoint matrices (integral + rate) and selection index.

        Both CP matrices use the same scaled-time axis (days-since-epoch / t_scale)
        so damping operates in consistent units.
        """
        requested_periods = fh
        all_periods = self._train_fh.append(
            requested_periods.difference(self._train_fh)
        ).sort_values()

        # Scaled time for full range
        t_scaled = self._index_to_scaled_timearray(all_periods)

        # Integral changepoint matrix (from parent)
        integral_cp_matrix = self.get_changepoint_matrix(all_periods)

        # Rate changepoint matrix (same time axis)
        t_arr = jnp.array(t_scaled, dtype=jnp.float32)
        rate_cp_matrix = jnp.maximum(
            0, t_arr[:, None] - self._rate_cp_ts[None, :]
        )

        # Selection index
        selection_ix = np.array([
            all_periods.get_loc(p) for p in requested_periods
        ], dtype=np.int32)

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
        """Dual-layer trend: integral path + rate dynamics + EC."""
        integral_cp_matrix = data["integral_cp_matrix"]
        rate_cp_matrix = data["rate_cp_matrix"]
        selection_ix = data["selection_ix"]
        T = data["n_total"]

        # === Sample parameters ===

        mu = numpyro.sample(
            "mean_rate",
            dist.Normal(self._mu_loc, self._mu_scale),
        )

        # Integral changepoints (sparse, slow)
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

        # Rate changepoints (dense, fast)
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

        # Error correction
        kappa = numpyro.sample(
            "kappa",
            dist.HalfNormal(self.kappa_prior_scale),
        )


        # === Layer 1: Expected integral path ===

        phi_S_safe = jnp.clip(phi_S, 1e-6, 1.0 - 1e-6)
        C_S = phi_S_safe * (-jnp.log(phi_S_safe)) / (1.0 - phi_S_safe)

        # h in period units (weeks) — interpretable priors
        h_S = integral_cp_matrix
        log_phi_S = jnp.log(phi_S_safe)
        log_decay_S = jnp.clip(h_S * log_phi_S, -50.0, 0.0)
        phi_S_h = jnp.exp(log_decay_S)
        decay_S = jnp.where(integral_cp_matrix > 0, phi_S_h, 0.0)

        expected_rate = mu + C_S * (decay_S @ delta_S)

        # Expected integral: μ·t + Σ δⱼ · f(hⱼ)
        damped_ramp_S = jnp.where(
            integral_cp_matrix > 0,
            phi_S_safe * (1.0 - phi_S_h) / (1.0 - phi_S_safe),
            0.0,
        )
        t_indices = jnp.arange(T, dtype=jnp.float32)
        expected_integral = mu * t_indices + damped_ramp_S @ delta_S

        # === Layer 2: Rate correction (damped impulses) ===

        psi_safe = jnp.clip(psi, 1e-6, 1.0 - 1e-6)
        h_R = rate_cp_matrix  # period units (weeks)
        log_psi = jnp.log(psi_safe)
        log_decay_R = jnp.clip(h_R * log_psi, -50.0, 0.0)
        decay_R = jnp.where(rate_cp_matrix > 0, jnp.exp(log_decay_R), 0.0)

        rate_correction = decay_R @ delta_R

        # === Layer 3: Error correction via scan ===

        def ec_step(S_actual, t_idx):
            S_expected_t = expected_integral[t_idx]
            base_rate_t = expected_rate[t_idx] + rate_correction[t_idx]

            deviation = S_actual - S_expected_t
            rate_t = base_rate_t - kappa * deviation

            S_actual_new = S_actual + rate_t
            return S_actual_new, rate_t

        init_S = jnp.array(0.0, dtype=jnp.float32)
        t_steps = jnp.arange(T, dtype=jnp.int32)
        final_S, all_rates = jax.lax.scan(ec_step, init_S, t_steps)

        # Self-consistency between S_actual and S_expected is handled
        # by the EC term (κ) in the scan. An additional numpyro.factor
        # penalty was tested but made results worse — it conflicts with
        # EC and over-constrains the model. See integral_trend_analysis.md.

        # Auxiliary integral observation on full model output is handled
        # in PV's _model.py. Enabled via self._integral_obs_enabled flag.

        # Select requested rows
        rates = all_rates[selection_ix]

        # === Diagnostics + integral output ===
        numpyro.deterministic("mean_rate_value", mu)
        numpyro.deterministic("integral_damping_value", phi_S)
        numpyro.deterministic("rate_damping_value", psi)
        numpyro.deterministic("kappa_value", kappa)
        numpyro.deterministic(
            "expected_integral",
            expected_integral[selection_ix],
        )

        return rates.reshape((-1, 1))
