"""Integral budget trend — integral sets level, softmax allocates rates.

Separates demand forecasting into two decoupled problems:

1. **Level** (how much total demand?) — piecewise linear integral path
   with damped changepoints. Easy to forecast, R² > 0.95 for mature products.

2. **Pattern** (which weeks get how much?) — logit-based allocation model
   driven by rate changepoints. Captures spikes, seasonality, level shifts.

The softmax bridge connects them:

    budget(t) = S(t) - S(t-1)           # per-step budget from integral
    weights = softmax(logits / temp)    # allocation pattern from rate model
    rates = total_budget × weights      # allocated rates, sum exactly to budget

The rate model doesn't need to get the absolute level right — only the
relative pattern. The integral model provides the level. This structural
decoupling eliminates the integral drift problem in DualIntegralTrend.

## Why not DualIntegralTrend?

DualIntegralTrend couples integral and rate via EC (κ) + soft Laplace
observation. In practice κ lands small (0.02-0.03), the integral drifts,
and the model consistently overshoots the actual cumulative. The rate
"wins" because there are more weekly observations pulling it.

This trend makes the constraint structural (softmax sums to 1) rather
than penalized (soft observation that can be violated).
"""

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from .piecewise import PiecewiseLinearTrend

__all__ = ["IntegralBudgetTrend"]


class IntegralBudgetTrend(PiecewiseLinearTrend):
    """Budget-constrained trend: integral sets level, softmax allocates.

    Parameters
    ----------
    changepoint_interval : int
        Interval for integral changepoints (sparse).
    changepoint_range : float
        Range for integral changepoints.
    changepoint_prior_scale : float
        Prior scale for integral changepoint magnitudes.
    mu_prior_scale_frac : float
        Prior scale on mean rate μ as fraction of |mean(y)|.
    integral_damping_beta_a, integral_damping_beta_b : float
        Beta prior on integral damping φ_S.
    rate_cp_interval : int
        Interval for rate changepoints (for logit model).
    rate_cp_range : float
        Range for rate changepoints.
    rate_cp_prior_scale : float
        Prior scale for rate logit changepoint magnitudes.
    temperature : float
        Softmax temperature. Higher = more uniform allocation.
        Lower = logits dominate (spikier). Default 1.0.
    learn_temperature : bool
        If True, sample temperature from HalfNormal(temperature_prior_scale).
    temperature_prior_scale : float
        Prior scale for learned temperature.
    """

    def __init__(
        self,
        # Integral changepoints (sparse, slow)
        changepoint_interval: int = 26,
        changepoint_range: float = 0.9,
        changepoint_prior_scale: float = 0.01,
        # Mean rate
        mu_prior_scale_frac: float = 0.2,
        # Integral damping
        integral_damping_beta_a: float = 100.0,
        integral_damping_beta_b: float = 1.0,
        # Rate logit model
        rate_cp_interval: int = 8,
        rate_cp_range: float = 0.95,
        rate_cp_prior_scale: float = 0.005,
        rate_damping_beta_a: float = 20.0,
        rate_damping_beta_b: float = 1.0,
        # Softmax temperature
        temperature: float = 1.0,
        learn_temperature: bool = False,
        temperature_prior_scale: float = 1.0,
    ):
        self.rate_cp_interval = rate_cp_interval
        self.rate_cp_range = rate_cp_range
        self.rate_cp_prior_scale = rate_cp_prior_scale
        self.mu_prior_scale_frac = mu_prior_scale_frac
        self.integral_damping_beta_a = integral_damping_beta_a
        self.integral_damping_beta_b = integral_damping_beta_b
        self.rate_damping_beta_a = rate_damping_beta_a
        self.rate_damping_beta_b = rate_damping_beta_b
        self.temperature = temperature
        self.learn_temperature = learn_temperature
        self.temperature_prior_scale = temperature_prior_scale
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

        # Build rate changepoint grid (same as DualIntegralTrend)
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
        """Build changepoint matrices and selection index."""
        requested_periods = fh
        all_periods = self._train_fh.append(
            requested_periods.difference(self._train_fh)
        ).sort_values()

        t_scaled = self._index_to_scaled_timearray(all_periods)

        # Integral changepoint matrix (from parent)
        integral_cp_matrix = self.get_changepoint_matrix(all_periods)

        # Rate changepoint matrix
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
        """Integral budget trend: integral path → softmax allocation."""
        integral_cp_matrix = data["integral_cp_matrix"]
        rate_cp_matrix = data["rate_cp_matrix"]
        selection_ix = data["selection_ix"]
        T = data["n_total"]

        # === Level 1: Integral path (same math as DualIntegralTrend) ===

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

        phi_S_safe = jnp.clip(phi_S, 1e-6, 1.0 - 1e-6)

        # Damped ramp for integral: S(t) = μ·t + Σ δⱼ · f(hⱼ)
        h_S = integral_cp_matrix
        log_phi_S = jnp.log(phi_S_safe)
        log_decay_S = jnp.clip(h_S * log_phi_S, -50.0, 0.0)
        phi_S_h = jnp.exp(log_decay_S)

        damped_ramp_S = jnp.where(
            integral_cp_matrix > 0,
            phi_S_safe * (1.0 - phi_S_h) / (1.0 - phi_S_safe),
            0.0,
        )
        t_indices = jnp.arange(T, dtype=jnp.float32)
        expected_integral = mu * t_indices + damped_ramp_S @ delta_S

        # Per-step budget: how much demand each timestep should contribute
        # S(t) - S(t-1), with S(-1) = 0
        step_budgets = jnp.diff(expected_integral, prepend=0.0)

        # === Level 2: Rate logits (allocation pattern) ===

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

        psi_safe = jnp.clip(psi, 1e-6, 1.0 - 1e-6)
        h_R = rate_cp_matrix
        log_psi = jnp.log(psi_safe)
        log_decay_R = jnp.clip(h_R * log_psi, -50.0, 0.0)
        decay_R = jnp.where(rate_cp_matrix > 0, jnp.exp(log_decay_R), 0.0)

        # Logits = base rate (from integral derivative) + rate corrections
        C_S = phi_S_safe * (-jnp.log(phi_S_safe)) / (1.0 - phi_S_safe)
        decay_S = jnp.where(integral_cp_matrix > 0, phi_S_h, 0.0)
        base_rate = mu + C_S * (decay_S @ delta_S)

        logits = base_rate + decay_R @ delta_R

        # === Softmax bridge ===

        if self.learn_temperature:
            temp = numpyro.sample(
                "temperature",
                dist.HalfNormal(self.temperature_prior_scale),
            )
        else:
            temp = self.temperature

        # Apply softmax over the FULL time range to get allocation weights
        weights = jax.nn.softmax(logits / temp)

        # Total budget = integral value at end of time range
        total_budget = expected_integral[-1]

        # Allocated rates: pattern × level
        rates = total_budget * weights

        # Select requested rows
        rates = rates[selection_ix]

        # === Diagnostics ===
        numpyro.deterministic("mean_rate_value", mu)
        numpyro.deterministic("integral_damping_value", phi_S)
        numpyro.deterministic("rate_damping_value", psi)
        numpyro.deterministic(
            "expected_integral",
            expected_integral[selection_ix],
        )
        numpyro.deterministic(
            "step_budgets",
            step_budgets[selection_ix],
        )

        return rates.reshape((-1, 1))
