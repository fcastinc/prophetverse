"""Constrained integral trend — receives fixed S(t), log-softmax in _model.py.

Takes a pre-fit integral path S(t) as fixed input. Outputs a base rate
(derivative of S(t)). PV effects add on top. Then _model.py applies
log-softmax per window so rates sum to S(t)'s budget.

The integral is NOT fit in this model — it comes from a separate stage 1
fit on cumulative demand. This ensures the integral is accurate (R² > 0.999)
while the rate model learns effect coefficients under the budget constraint.

## Flow

    Stage 1 (separate): fit cumsum → S(t) for all t

    Stage 2 (this model):
        trend._predict() → base_rate (diff of fixed S(t)) + rate changepoints
        effects._predict() → regressors, seasonality, holidays
        total = base_rate + effects              ← PV does this
        _model.py: log-softmax(total) × budget   ← constraint per window
        likelihood(constrained_total, y)          ← fits on weekly demand
"""

from typing import Dict

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from .piecewise import PiecewiseLinearTrend

__all__ = ["ConstrainedIntegralTrend"]


class ConstrainedIntegralTrend(PiecewiseLinearTrend):
    """Trend with fixed integral constraint and rate changepoints.

    Receives S(t) from a stage 1 fit. Outputs base rate + rate changepoints.
    The _model.py log-softmax block constrains total output per window.

    Parameters
    ----------
    integral_path : array-like
        Pre-computed integral path S(t) from stage 1. Length = n_train + n_forecast.
        Set after construction via set_integral_path().
    rate_cp_interval : int
        Interval for rate changepoints.
    rate_cp_range : float
        Range for rate changepoints.
    rate_cp_prior_scale : float
        Prior scale for rate changepoint magnitudes.
    rate_damping_beta_a, rate_damping_beta_b : float
        Beta prior on rate damping.
    budget_constraint_enabled : bool
        Activates log-softmax constraint in _model.py.
    budget_window_size : int
        Window size for constraint (= forecast horizon).
    """

    def __init__(
        self,
        # Rate changepoints
        rate_cp_interval: int = 8,
        rate_cp_range: float = 0.95,
        rate_cp_prior_scale: float = 0.005,
        rate_damping_beta_a: float = 20.0,
        rate_damping_beta_b: float = 1.0,
        # Inherited but not used for integral (integral is fixed)
        changepoint_interval: int = 26,
        changepoint_range: float = 0.9,
        changepoint_prior_scale: float = 0.01,
        # Budget constraint
        budget_constraint_enabled: bool = True,
        budget_window_size: int = 26,
        # Fixed integral path from stage 1
        integral_path=None,
        # Max of raw y — for normalizing step budgets to match PV's space
        y_max=None,
    ):
        self.rate_cp_interval = rate_cp_interval
        self.rate_cp_range = rate_cp_range
        self.rate_cp_prior_scale = rate_cp_prior_scale
        self.rate_damping_beta_a = rate_damping_beta_a
        self.rate_damping_beta_b = rate_damping_beta_b
        self.budget_constraint_enabled = budget_constraint_enabled
        self.budget_window_size = budget_window_size
        self.integral_path = integral_path
        self.y_max = y_max
        self._integral_path = (
            jnp.array(integral_path, dtype=jnp.float32)
            if integral_path is not None else None
        )
        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
        )

    def set_integral_path(self, integral_path):
        """Set the fixed integral path from stage 1.

        Parameters
        ----------
        integral_path : array-like
            S(t) values for the full time range (train + forecast).
        """
        self.integral_path = list(integral_path)  # serializable for clone
        self._integral_path = jnp.array(integral_path, dtype=jnp.float32)

    def _fit(self, y, X, scale=1.0):
        super()._fit(y, X, scale)
        self._train_fh = y.index.get_level_values(-1).unique().sort_values()

        # Compute step budgets from integral — keep in raw units.
        # _model.py normalizes using raw y (which it has access to).
        # The integral path stays raw — base_rate will be in raw units
        # but the log-softmax only uses relative values so scale doesn't matter.
        if self._integral_path is not None:
            self._step_budgets = jnp.diff(
                self._integral_path, prepend=0.0)
        else:
            self._step_budgets = None

        # Build rate changepoint grid
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
        """Build rate changepoint matrix and selection index."""
        requested_periods = fh
        all_periods = self._train_fh.append(
            requested_periods.difference(self._train_fh)
        ).sort_values()

        t_scaled = self._index_to_scaled_timearray(all_periods)

        # Rate changepoint matrix
        t_arr = jnp.array(t_scaled, dtype=jnp.float32)
        rate_cp_matrix = jnp.maximum(
            0, t_arr[:, None] - self._rate_cp_ts[None, :]
        )

        selection_ix = np.array([
            all_periods.get_loc(p) for p in requested_periods
        ], dtype=np.int32)

        return {
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
        """Base rate from fixed integral + sampled rate changepoints."""
        rate_cp_matrix = data["rate_cp_matrix"]
        selection_ix = data["selection_ix"]
        T = data["n_total"]

        # === Base rate from fixed integral ===
        # S(t) is fixed data — not sampled
        integral = self._integral_path
        # Base rate = S(t) - S(t-1)
        base_rate = jnp.diff(integral[:T], prepend=0.0)

        # === Rate changepoints (sampled) ===
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

        rate_correction = decay_R @ delta_R

        # Total rate = base (from integral) + corrections
        rates = base_rate + rate_correction

        # Select requested rows
        rates = rates[selection_ix]

        # Expose step budgets for _model.py constraint
        # Already normalized in _fit — same scale as model output
        predicted_effects["latent/step_budgets"] = self._step_budgets[:T][selection_ix]

        # Diagnostics
        numpyro.deterministic("rate_damping_value", psi)
        numpyro.deterministic(
            "expected_integral",
            integral[:T][selection_ix],
        )
        numpyro.deterministic(
            "base_rate",
            base_rate[selection_ix],
        )

        return rates.reshape((-1, 1))
