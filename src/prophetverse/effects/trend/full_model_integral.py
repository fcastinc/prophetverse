"""Backward-compatible wrapper for the renamed joint DPW integral trend module."""

from .joint_dpw_integral import JointDPWIntegralTrend

FullModelIntegralTrend = JointDPWIntegralTrend

__all__ = ["FullModelIntegralTrend", "JointDPWIntegralTrend"]
