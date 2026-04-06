"""Module for trend models in prophetverse."""

from .base import TrendEffectMixin
from .flat import FlatTrend
from .piecewise import PiecewiseLinearTrend, PiecewiseLogisticTrend
from .damped_piecewise import DampedPiecewiseLinearTrend
from .dual_integral import DualIntegralTrend
from .joint_dpw_integral import JointDPWIntegralTrend
from .joint_flat_integral import JointFlatIntegralTrend
from .full_model_integral import FullModelIntegralTrend
from .integral_budget import IntegralBudgetTrend
from .constrained_integral import ConstrainedIntegralTrend

__all__ = [
    "TrendEffectMixin",
    "FlatTrend",
    "PiecewiseLinearTrend",
    "PiecewiseLogisticTrend",
    "DampedPiecewiseLinearTrend",
    "DualIntegralTrend",
    "JointDPWIntegralTrend",
    "JointFlatIntegralTrend",
    "FullModelIntegralTrend",
    "IntegralBudgetTrend",
    "ConstrainedIntegralTrend",
]
