"""
Rotation ETF Payout Strategy

A custom ETF rotation strategy with systematic withdrawal/payout feature.
Based on the standard Rotation_ETF strategy with enhanced withdrawal logic.
"""

from .backtester import RotationETFPayoutBacktester

__all__ = ['RotationETFPayoutBacktester']
