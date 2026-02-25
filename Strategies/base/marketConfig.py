"""
marketConfig.py — Single Source of Truth for Market & Asset Configuration

Every strategy, handler and backtester resolves DB tables and benchmark
symbols via getMarketConfig(). No more hardcoded table names scattered
across the codebase.

Usage:
    from Strategies.base.marketConfig import getMarketConfig

    config = getMarketConfig("INDIA", "ETF")
    table  = config["dataTable"]        # "etf_market"
    index  = config["indexTable"]       # "nifty_50_index_market"
    bench  = config["benchmark"]        # "NIFTY_50"
    syms   = config["benchmarkSymbols"] # ["NIFTY_50", "^NSEI", ...]

Supported combinations:
    (INDIA, ETF)   (INDIA, STOCK)
    (US, ETF)      (US, STOCK)
"""

from typing import Dict, Any, List


# ──────────────────────────────────────────────────────────────────
# Core Config Table
# ──────────────────────────────────────────────────────────────────

MARKET_CONFIG: Dict[tuple, Dict[str, Any]] = {
    ("INDIA", "ETF"): {
        "dataTable":        "etf_market",
        "indexTable":       "nifty_50_index_market",
        "benchmark":        "NIFTY_50",
        # All known symbol variants stored in DB for this index
        "benchmarkSymbols": ["NIFTY_50", "^NSEI", "NSEI", "NIFTY50", "^NIFTY50"],
        "currency":         "INR",
        "timezone":         "Asia/Kolkata",
        "calendar":         "NSE",
    },
    ("INDIA", "STOCK"): {
        "dataTable":        "stock_market",
        "indexTable":       "nifty_50_index_market",
        "benchmark":        "NIFTY_50",
        "benchmarkSymbols": ["NIFTY_50", "^NSEI", "NSEI", "NIFTY50", "^NIFTY50"],
        "currency":         "INR",
        "timezone":         "Asia/Kolkata",
        "calendar":         "NSE",
    },
    ("US", "ETF"): {
        "dataTable":        "us_etf_market",
        "indexTable":       "s_p_500_index_market",
        "benchmark":        "S&P_500",
        "benchmarkSymbols": ["S&P_500", "^GSPC", "SPY", "SP500"],
        "currency":         "USD",
        "timezone":         "America/New_York",
        "calendar":         "NYSE",
    },
    ("US", "STOCK"): {
        "dataTable":        "us_stock_market",
        "indexTable":       "s_p_500_index_market",
        "benchmark":        "S&P_500",
        "benchmarkSymbols": ["S&P_500", "^GSPC", "SPY", "SP500"],
        "currency":         "USD",
        "timezone":         "America/New_York",
        "calendar":         "NYSE",
    },
}


# ──────────────────────────────────────────────────────────────────
# Strategy-type → default (market, assetType) mapping
# Used when caller only has strategy_type but no explicit market/asset
# ──────────────────────────────────────────────────────────────────

STRATEGY_DEFAULTS: Dict[str, Dict[str, str]] = {
    "ETF_Rotation":                 {"market": "INDIA", "assetType": "ETF"},
    "International_ETF_Rotation":   {"market": "US",    "assetType": "ETF"},  # legacy alias
    "Stock_Rotation":               {"market": "INDIA", "assetType": "STOCK"},
    "ETF_Payout":                   {"market": "INDIA", "assetType": "ETF"},
    "ETF_Buy_on_Dip":               {"market": "INDIA", "assetType": "ETF"},
    "ETF_Swing_Strategy":           {"market": "INDIA", "assetType": "ETF"},
    "US_ETF_Swing_Strategy":        {"market": "US",    "assetType": "ETF"},
    "RS_ETF_Rotation":              {"market": "INDIA", "assetType": "ETF"},
    "RS_Stocks":                    {"market": "INDIA", "assetType": "STOCK"},
    "SuperTrend":                   {"market": "INDIA", "assetType": "STOCK"},
}


# ──────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────

def getMarketConfig(market: str, assetType: str) -> Dict[str, Any]:
    """
    Return the config dict for the given market + assetType pair.

    Args:
        market:    "INDIA" or "US" (case-insensitive)
        assetType: "ETF" or "STOCK" (case-insensitive)

    Returns:
        Dict with keys: dataTable, indexTable, benchmark,
                        benchmarkSymbols, currency, timezone, calendar

    Raises:
        ValueError if the combination is not supported.
    """
    key = (market.upper(), assetType.upper())
    config = MARKET_CONFIG.get(key)
    if config is None:
        supported = [f"{m}/{a}" for m, a in MARKET_CONFIG]
        raise ValueError(
            f"Unsupported market/assetType: '{market}/{assetType}'. "
            f"Supported: {supported}"
        )
    return config


def getDefaultsForStrategy(strategyType: str) -> Dict[str, str]:
    """
    Return the default market + assetType for a given strategy type.

    Useful when a request does not explicitly carry market/assetType.
    """
    defaults = STRATEGY_DEFAULTS.get(strategyType)
    if defaults is None:
        return {"market": "INDIA", "assetType": "ETF"}
    return defaults


def resolveMarketConfig(
    strategyType: str,
    market: str | None = None,
    assetType: str | None = None,
) -> Dict[str, Any]:
    """
    Combined helper: resolve market + assetType (using strategy defaults
    as fallback) then return the full config dict.

    Args:
        strategyType: e.g. "ETF_Rotation"
        market:       explicit override ("INDIA" / "US"), or None
        assetType:    explicit override ("ETF" / "STOCK"), or None
    """
    defaults = getDefaultsForStrategy(strategyType)
    resolvedMarket    = (market    or defaults["market"]).upper()
    resolvedAssetType = (assetType or defaults["assetType"]).upper()
    return getMarketConfig(resolvedMarket, resolvedAssetType)


def getBenchmarkSymbolsClause(market: str, assetType: str) -> str:
    """
    Return a SQL-safe IN clause string for benchmark symbols.

    Example: "'NIFTY_50', '^NSEI', 'NSEI', 'NIFTY50', '^NIFTY50'"
    """
    config = getMarketConfig(market, assetType)
    symbols = config["benchmarkSymbols"]
    return ", ".join(f"'{s}'" for s in symbols)


def getSupportedMarkets() -> List[str]:
    """Return all unique market names."""
    return list({m for m, _ in MARKET_CONFIG})


def getSupportedAssetTypes() -> List[str]:
    """Return all unique asset types."""
    return list({a for _, a in MARKET_CONFIG})
