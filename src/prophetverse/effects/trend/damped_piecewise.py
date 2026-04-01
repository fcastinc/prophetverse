"""Damped piecewise linear trend v3 — damping applied in-sample.

Fixes the v2 limitation where damping only applied at forecast time.
In v3, each changepoint's slope contribution decays geometrically
from the moment it occurs:

    contribution_j(t) = delta_j * phi * (1 - phi^h_j) / (1 - phi)

where h_j = max(0, t - cp_j) is the time since changepoint j.

As phi→1: contribution → delta_j * h_j (undamped, same as v2 in-sample)
As phi→0: contribution → 0 (instant decay)

This means:
- Recent changepoints have near-full effect
- Old changepoints gradually decay toward zero
- The trend "forgets" old regime changes
- phi is identifiable from in-sample data (learn_damping=True works)
- In-sample and forecast dynamics are consistent
- MCMC posterior over phi + changepoints is correct

The standard piecewise linear uses A @ delta where A is the changepoint
matrix. v3 replaces A with A_damped where each element is the geometric
series instead of the linear ramp. This is element-wise ops, still
JAX-differentiable but not a simple matmul.
"""

from typing import Dict, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .piecewise import PiecewiseLinearTrend

__all__ = ["DampedPiecewiseLinearTrendV3"]


class DampedPiecewiseLinearTrendV3(PiecewiseLinearTrend):
    """Piecewise linear trend with in-sample damping.

    Each changepoint's slope decays geometrically from when it occurs.
    Damping is applied BOTH in-sample and at forecast time, making
    the model consistent for MCMC/VI inference.

    Parameters
    ----------
    changepoint_interval : int
        Interval between potential changepoints.
    changepoint_range : float
        Fraction of training data to place changepoints in.
    changepoint_prior_scale : float
        Laplace prior scale on changepoint coefficients.
    offset_prior_scale : float
        Prior scale for the trend offset.
    damping_factor : float
        Per-period damping factor phi in (0, 1). Used as fixed value
        when learn_damping=False.
    damping_beta_a, damping_beta_b : float
        Beta prior params when learn_damping=True.
    learn_damping : bool
        If True (default), phi is sampled from Beta(a, b).
        If False, fixed at damping_factor.
    """

    def __init__(
        self,
        changepoint_interval: int = 25,
        changepoint_range: float = 0.8,
        changepoint_prior_scale: float = 0.001,
        offset_prior_scale: float = 0.1,
        damping_factor: float = 0.998,
        damping_beta_a: float = 30.0,
        damping_beta_b: float = 1.0,
        learn_damping: bool = True,
        remove_seasonality_before_suggesting_initial_vals: bool = True,
        global_rate_prior_loc: Optional[float] = None,
        offset_prior_loc: Optional[float] = None,
    ):
        self.damping_factor = damping_factor
        self.damping_beta_a = damping_beta_a
        self.damping_beta_b = damping_beta_b
        self.learn_damping = learn_damping
        super().__init__(
            changepoint_interval=changepoint_interval,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
            offset_prior_scale=offset_prior_scale,
            remove_seasonality_before_suggesting_initial_vals=(
                remove_seasonality_before_suggesting_initial_vals
            ),
            global_rate_prior_loc=global_rate_prior_loc,
            offset_prior_loc=offset_prior_loc,
        )

    def _predict(
        self,
        data: jnp.ndarray,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute trend with in-sample damping.

        Replaces the standard A @ delta with a damped version where
        each changepoint's contribution decays geometrically.
        """
        changepoint_matrix = data  # A[i,j] = max(0, t[i] - cp[j])

        # Sample parameters
        offset = numpyro.sample(
            "offset",
            dist.Normal(self._offset_prior_loc, self._offset_prior_scale),
        )
        changepoint_coefficients = numpyro.sample(
            "changepoint_coefficients",
            dist.Laplace(
                self._changepoint_prior_loc,
                self._changepoint_prior_scale,
            ),
        )

        # Sample or fix damping factor
        if self.learn_damping:
            phi = numpyro.sample(
                "damping_factor",
                dist.Beta(self.damping_beta_a, self.damping_beta_b),
            )
        else:
            phi = self.damping_factor

        # Transform changepoint matrix: linear ramps → damped ramps.
        #
        # Original: A[i,j] = max(0, t[i] - cp[j]) in scaled time
        # Damped:   A_d[i,j] = phi*(1 - phi^h) / (1-phi) in real periods
        #
        # phi is a scalar — this is element-wise on the matrix, then
        # matmul with delta works exactly as before.
        h_real = changepoint_matrix * self.t_scale
        eps = 1e-10

        A_damped = jnp.where(
            changepoint_matrix > 0,
            phi * (1 - jnp.power(phi, h_real)) / (1 - phi + eps),
            0.0,
        )

        # Back to scaled-time units (delta is per-scaled-time)
        A_damped_scaled = A_damped / self.t_scale

        # Standard matmul — same as parent, just with damped A
        trend = A_damped_scaled @ changepoint_coefficients + offset

        return trend.reshape((-1, 1))
