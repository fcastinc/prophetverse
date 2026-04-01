"""Inverse Gaussian (Wald) distribution for numpyro.

Continuous, strictly positive, Var = mu^3/lam — variance grows as cube of mean.
Heavier-tailed than Gamma (Var = mu^2/shape), suitable for demand data with
large promotional spikes.

Reparameterized version maps (loc, scale) for compatibility with
Prophetverse's TargetLikelihood interface.

    mu = loc
    lam = loc / scale^2
    Var = mu^3/lam = loc^2 * scale^2
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes


class InverseGaussian(dist.Distribution):
    """Inverse Gaussian (Wald) distribution.

    Parameters
    ----------
    loc : mean mu > 0
    concentration : shape lam > 0

    PDF: f(x; mu, lam) = sqrt(lam / (2pi x^3)) * exp(-lam(x-mu)^2 / (2mu^2 x))
    Mean = mu
    Var  = mu^3 / lam
    """

    arg_constraints = {
        "loc": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["loc", "concentration"]

    def __init__(self, loc, concentration, *, validate_args=None):
        self.loc, self.concentration = promote_shapes(loc, concentration)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.loc), jnp.shape(self.concentration),
        )
        super().__init__(
            batch_shape=batch_shape, validate_args=validate_args,
        )

    def log_prob(self, value):
        mu = self.loc
        lam = self.concentration
        return (
            0.5 * jnp.log(lam)
            - 0.5 * jnp.log(2 * jnp.pi)
            - 1.5 * jnp.log(value)
            - lam * (value - mu) ** 2 / (2 * mu**2 * value)
        )

    def sample(self, key, sample_shape=()):
        import jax.random as random

        shape = sample_shape + self.batch_shape
        mu = jnp.broadcast_to(self.loc, shape)
        lam = jnp.broadcast_to(self.concentration, shape)

        key1, key2 = random.split(key)
        v = random.normal(key1, shape)
        y = v**2

        x1 = mu + (mu**2 * y) / (2 * lam) - (
            mu / (2 * lam)
        ) * jnp.sqrt(4 * mu * lam * y + mu**2 * y**2)

        u = random.uniform(key2, shape)
        x2 = mu**2 / x1
        return jnp.where(u <= mu / (mu + x1), x1, x2)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.loc**3 / self.concentration


class InverseGaussianReparametrized(InverseGaussian):
    """InverseGaussian reparameterized as (loc, scale) for Prophetverse.

    Prophetverse passes (mean, noise_scale) to the likelihood.
    Maps: mu = loc, lam = loc / scale^2
    So Var = loc^2 * scale^2 — scale controls CV-like behavior.
    """

    arg_constraints = {
        "loc": constraints.positive,
        "scale": constraints.positive,
    }
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale=1.0, *, validate_args=None):
        self.scale = scale
        concentration = loc / (scale**2)
        super().__init__(
            loc=loc, concentration=concentration,
            validate_args=validate_args,
        )
