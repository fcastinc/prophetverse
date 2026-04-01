"""Integral budget trend — piecewise linear integral, base rate output.

Computes a piecewise linear integral path S(t) with damped changepoints.
Outputs the derivative (base rate) as the trend. Exposes S(t) so that
_model.py can apply a windowed softmax budget constraint after effects
are added.

The constraint is NOT in this trend — it's in _model.py. This trend
just provides the integral path and the base rate. PV effects (regressors,
seasonality, holidays) add on top as normal. Then _model.py renormalizes
the total (trend + effects) per window so rates sum to the integral budget.

## Flow

    trend._predict() → base_rate (derivative of S(t))
    effects._predict() → regressor/seasonality contributions
    total = base_rate + effects              ← PV does this
    _model.py: per-window softmax(total) × budget  ← constraint
    likelihood(constrained_total, y)

## Relationship to DualIntegralTrend

Uses the same Layer 1 math (integral path with damped changepoints).
Does NOT have Layer 2 (rate changepoints), Layer 3 (EC scan), or
integral observation. The constraint mechanism is fundamentally
different — structural (softmax) vs penalized (Laplace obs + EC).
"""

from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from .piecewise import PiecewiseLinearTrend

__all__ = ["IntegralBudgetTrend"]


class IntegralBudgetTrend(PiecewiseLinearTrend):
    """Integral path trend with budget constraint support.

    Outputs the integral derivative as the base rate. Exposes
    the integral path S(t) via numpyro deterministic so _model.py
    can compute window budgets and apply softmax constraint.

    Parameters
    ----------
    changepoint_interval : int
        Interval for integral changepoints.
    changepoint_range : float
        Range for integral changepoints.
    changepoint_prior_scale : float
        Prior scale for integral changepoint magnitudes.
    mu_prior_scale_frac : float
        Prior scale on mean rate as fraction of |mean(y)|.
    integral_damping_beta_a, integral_damping_beta_b : float
        Beta prior on integral damping phi.
    budget_constraint_enabled : bool
        If True, _model.py applies windowed softmax constraint.
    budget_window_size : int
        Number of periods per constraint window (= forecast horizon).
    """

    def __init__(
        self,
        changepoint_interval: int = 26,
        changepoint_range: float = 0.9,
        changepoint_prior_scale: float = 0.01,
        mu_prior_scale_frac: float = 0.2,
        integral_damping_beta_a: float = 100.0,
        integral_damping_beta_b: float = 1.0,
        budget_constraint_enabled: bool = True,
        budget_window_size: int = 13,
    ):
        self.mu_prior_scale_frac = mu_prior_scale_frac
        self.integral_damping_beta_a = integral_damping_beta_a
        self.integral_damping_beta_b = integral_damping_beta_b
        self.budget_constraint_enabled = budget_constraint_enabled
        self.budget_window_size = budget_window_size
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

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Dict:
        """Build changepoint matrix and selection index."""
        requested_periods = fh
        all_periods = self._train_fh.append(
            requested_periods.difference(self._train_fh)
        ).sort_values()

        t_scaled = self._index_to_scaled_timearray(all_periods)
        integral_cp_matrix = self.get_changepoint_matrix(all_periods)

        selection_ix = np.array([
            all_periods.get_loc(p) for p in requested_periods
        ], dtype=np.int32)

        return {
            "integral_cp_matrix": integral_cp_matrix,
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
        """Compute integral path and output base rate (derivative)."""
        integral_cp_matrix = data["integral_cp_matrix"]
        selection_ix = data["selection_ix"]
        T = data["n_total"]

        # === Sample integral parameters ===

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

        # === Compute integral path S(t) ===

        phi_S_safe = jnp.clip(phi_S, 1e-6, 1.0 - 1e-6)

        h_S = integral_cp_matrix
        log_phi_S = jnp.log(phi_S_safe)
        log_decay_S = jnp.clip(h_S * log_phi_S, -50.0, 0.0)
        phi_S_h = jnp.exp(log_decay_S)

        # Integral: S(t) = mu*t + sum_j delta_j * f(h_j)
        damped_ramp_S = jnp.where(
            integral_cp_matrix > 0,
            phi_S_safe * (1.0 - phi_S_h) / (1.0 - phi_S_safe),
            0.0,
        )
        t_indices = jnp.arange(T, dtype=jnp.float32)
        expected_integral = mu * t_indices + damped_ramp_S @ delta_S

        # === Base rate = derivative of integral ===
        # C(phi) normalization for analytical derivative
        C_S = phi_S_safe * (-jnp.log(phi_S_safe)) / (1.0 - phi_S_safe)
        decay_S = jnp.where(integral_cp_matrix > 0, phi_S_h, 0.0)
        base_rate = mu + C_S * (decay_S @ delta_S)

        # Select requested rows
        rates = base_rate[selection_ix]

        # === Expose integral for _model.py budget constraint ===
        # Pass through predicted_effects so _model.py can read it.
        # Using latent/ prefix so it's not included in the model sum.
        predicted_effects["latent/expected_integral"] = numpyro.deterministic(
            "expected_integral_full", expected_integral
        )
        predicted_effects["latent/selection_ix"] = selection_ix

        # Diagnostics
        numpyro.deterministic("mean_rate_value", mu)
        numpyro.deterministic("integral_damping_value", phi_S)
        numpyro.deterministic(
            "expected_integral",
            expected_integral[selection_ix],
        )

        return rates.reshape((-1, 1))
