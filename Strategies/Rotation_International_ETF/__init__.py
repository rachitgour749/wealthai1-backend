"""
International ETF Rotation Strategy

A momentum-based rotation strategy for international ETFs using 52-week high/low signals.
Uses NYSE trading calendar and US Eastern Time timezone.
Zero transaction costs for international ETF trading.
"""

from .services.backtester import InternationalETFRotationBacktester

__all__ = ['InternationalETFRotationBacktester']
