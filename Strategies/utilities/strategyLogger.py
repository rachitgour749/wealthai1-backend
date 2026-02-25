"""
strategyLogger.py — Thin wrapper over Python's logging module.

Replaces the old print()-based logging_config.py. All strategies create
a StrategyLogger instance and call the same familiar API:

    logger = StrategyLogger("Rotation")
    logger.info("Loading data")
    logger.debug("Detail info")
    logger.trade("BUY NIFTYBEES @ 200")
    logger.performance("Sharpe: 1.42")
    logger.error("Something went wrong")

Log levels and module-level verbosity are controlled centrally via
config/logging_config.json and the setupLogging() bootstrap.

Category → level mapping:
    debug       → logging.DEBUG
    info        → logging.INFO
    progress    → logging.INFO   (with ⏳ prefix)
    trade       → logging.INFO   (with 💹 prefix)
    performance → logging.INFO   (with 📊 prefix)
    warning     → logging.WARNING
    error       → logging.ERROR
"""

import json
import logging
import os
from typing import Dict, Any, Optional


# ──────────────────────────────────────────────────────────────────
# Load the global logging categories config once at module import
# ──────────────────────────────────────────────────────────────────

def _loadCategoryConfig() -> Dict[str, bool]:
    """Read the categories block from config/logging_config.json."""
    try:
        projectRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        configPath  = os.path.join(projectRoot, "config", "logging_config.json")
        with open(configPath, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("categories", {})
    except Exception:
        return {
            "debug": False, "info": True, "progress": True,
            "trade": True,  "performance": True, "warning": True, "error": True,
        }


def _loadPurchaseLimitConfig() -> Dict[str, Any]:
    """Read purchaseLimitConfig from config/logging_config.json."""
    try:
        projectRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        configPath  = os.path.join(projectRoot, "config", "logging_config.json")
        with open(configPath, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("purchaseLimitConfig", {"enabled": True, "allocationMultiplier": 1.5})
    except Exception:
        return {"enabled": True, "allocationMultiplier": 1.5}


_CATEGORY_CONFIG     = _loadCategoryConfig()
_PURCHASE_LIMIT_CONFIG = _loadPurchaseLimitConfig()


# ──────────────────────────────────────────────────────────────────
# StrategyLogger
# ──────────────────────────────────────────────────────────────────

class StrategyLogger:
    """
    Thin wrapper over logging.Logger with domain-specific helpers.

    Keeps the same API as the old print()-based StrategyLogger so that
    no strategy code needs to change method call sites.
    """

    def __init__(self, moduleName: str, categoryOverrides: Optional[Dict[str, bool]] = None):
        """
        Args:
            moduleName:        Python logger name (e.g. "Rotation", "RS_ETF").
                               Log level is controlled by config/logging_config.json.
            categoryOverrides: Optional per-instance overrides for categories,
                               e.g. {"debug": True} during development.
        """
        self.logger       = logging.getLogger(moduleName)
        self.moduleName   = moduleName
        # Merge global categories with any per-instance overrides
        self._categories  = {**_CATEGORY_CONFIG, **(categoryOverrides or {})}
        # Purchase limit config (used by rotation backtesters)
        self.purchaseLimitConfig = _PURCHASE_LIMIT_CONFIG

    # ── Category checks ───────────────────────────────────────────

    def _enabled(self, category: str) -> bool:
        return self._categories.get(category, True)

    # ── Public API (matches old StrategyLogger exactly) ───────────

    def debug(self, message: str) -> None:
        if self._enabled("debug"):
            self.logger.debug(message)

    def info(self, message: str) -> None:
        if self._enabled("info"):
            self.logger.info(message)

    def progress(self, message: str) -> None:
        if self._enabled("progress"):
            self.logger.info(f"⏳ {message}")

    def warning(self, message: str) -> None:
        if self._enabled("warning"):
            self.logger.warning(message)

    def error(self, message: str) -> None:
        # errors are always logged regardless of category config
        self.logger.error(message)

    def trade(self, message: str) -> None:
        if self._enabled("trade"):
            self.logger.info(f"💹 {message}")

    def performance(self, message: str) -> None:
        if self._enabled("performance"):
            self.logger.info(f"📊 {message}")

    # ── Limit category (rotation-specific) ───────────────────────

    def limit(self, message: str) -> None:
        """Log purchase-limit related messages (maps to INFO)."""
        if self._categories.get("debug", False):   # show limit logs only in debug mode
            self.logger.info(f"🔒 {message}")

    # ── Dynamic config helpers ────────────────────────────────────

    def setCategory(self, category: str, enabled: bool) -> None:
        """Enable or disable a log category at runtime."""
        self._categories[category] = enabled

    def enableAll(self) -> None:
        """Enable all log categories."""
        for key in self._categories:
            self._categories[key] = True

    def disableAll(self) -> None:
        """Disable all categories except errors."""
        for key in self._categories:
            if key != "error":
                self._categories[key] = False

    def getConfig(self) -> Dict[str, Any]:
        """Return the current runtime category config (for debug inspection)."""
        return {
            "moduleName": self.moduleName,
            "categories": self._categories.copy(),
            "level":      logging.getLevelName(self.logger.getEffectiveLevel()),
        }

    # ── Backward-compat: _log(category, message) ─────────────────

    def _log(self, category: str, message: str, prefix: str = "") -> None:
        """
        Legacy method kept for backward compatibility with backtesters
        that call self.logger._log(category, message).
        """
        if not self._enabled(category):
            return
        full = f"{prefix} {message}".strip() if prefix else message
        if category == "error":
            self.logger.error(full)
        elif category == "debug":
            self.logger.debug(full)
        else:
            self.logger.info(full)
