"""
Strategies/utilities/logging_config.py — COMPATIBILITY SHIM

This module now re-exports StrategyLogger from the new strategyLogger.py,
which is backed by Python's logging module (not print()).

All existing code that does:
    from Strategies.utilities.logging_config import StrategyLogger
continues to work without modification.

Logging behaviour is now controlled by config/logging_config.json (single file).
"""

# Re-export — this is the ONLY line needed.
from Strategies.utilities.strategyLogger import StrategyLogger  # noqa: F401

__all__ = ["StrategyLogger"]
