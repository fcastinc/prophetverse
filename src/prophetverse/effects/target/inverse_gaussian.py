"""InverseGaussian target likelihood for Prophetverse."""
from prophetverse.distributions.inverse_gaussian import InverseGaussianReparametrized
from prophetverse.effects.target.univariate import (
    TargetLikelihood,
    _PositiveSmoothClipper,
)


class InverseGaussianTargetLikelihood(TargetLikelihood):
    """Prophetverse target likelihood using InverseGaussian.

    Uses _PositiveSmoothClipper (picklable) for the link function.
    """

    def __init__(self, noise_scale=0.05, epsilon=1e-5):
        self.epsilon = epsilon
        link_function = _PositiveSmoothClipper(epsilon)
        super().__init__(
            noise_scale,
            link_function=link_function,
            likelihood_func=InverseGaussianReparametrized,
        )
