"""
Strategies/Rotation_International_ETF/services/backtester.py

COMPATIBILITY SHIM — This file used to contain a 3,051-line duplicate of
ETFRotationBacktester. It is now a thin alias that delegates entirely to
ETFRotationBacktester(market='US').

All International ETF Rotation logic is handled by ETFRotationBacktester,
which is market-aware (resolves data_table + benchmark from marketConfig).

DO NOT add new logic here. Use ETFRotationBacktester or RotationHandler directly.
"""

import warnings
from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester


class InternationalETFRotationBacktester(ETFRotationBacktester):
    """
    Deprecated: Use ETFRotationBacktester(market='US') directly.

    Kept as a compatibility alias so existing code that imports
    InternationalETFRotationBacktester continues to work without changes.
    """

    def __init__(self, market: str = "US", db_path: str = None):
        warnings.warn(
            "InternationalETFRotationBacktester is deprecated. "
            "Use ETFRotationBacktester(market='US') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Always force market=US regardless of what caller passes
        super().__init__(market="US", db_path=db_path)
