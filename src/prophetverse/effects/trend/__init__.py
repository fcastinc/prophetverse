"""Module for trend models in prophetverse."""

from .base import TrendEffectMixin
from .flat import FlatTrend
from .piecewise import PiecewiseLinearTrend, PiecewiseLogisticTrend
from .damped_piecewise import DampedPiecewiseLinearTrendV3
from .dual_integral import DualIntegralTrend
from .integral_budget import IntegralBudgetTrend
from .constrained_integral import ConstrainedIntegralTrend

__all__ = [
    "TrendEffectMixin",
    "FlatTrend",
    "PiecewiseLinearTrend",
    "PiecewiseLogisticTrend",
    "DampedPiecewiseLinearTrendV3",
    "DualIntegralTrend",
    "IntegralBudgetTrend",
    "ConstrainedIntegralTrend",
]
