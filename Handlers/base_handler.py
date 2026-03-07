"""
Base Handler Interface for Strategy Backtests
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from sqlalchemy.orm import Session
from APIs.unified_schemas import UnifiedBacktestRequest, UnifiedBacktestResponse


class BaseStrategyHandler(ABC):
    """
    Abstract base class for strategy handlers.
    Each strategy implements this interface.
    """
    
    def __init__(self, db: Session):
        """
        Initialize handler with database session
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    @abstractmethod
    async def run_backtest(self, request: UnifiedBacktestRequest) -> UnifiedBacktestResponse:
        """
        Run backtest for the strategy
        
        Args:
            request: Unified backtest request
            
        Returns:
            UnifiedBacktestResponse with results
        """
        pass
    
    @abstractmethod
    def validate_request(self, request: UnifiedBacktestRequest) -> None:
        """
        Validate strategy-specific parameters
        
        Args:
            request: Unified backtest request
            
        Raises:
            ValueError: If validation fails
        """
        pass
    
    def _clean_tickers(self, tickers: Any) -> Any:
        """
        Clean tickers: splits comma-separated strings, removes .NS suffix.
        Accepts: a list of strings, a single string, or a comma-separated string.
        Returns: a list of cleaned ticker strings.
        """
        if tickers is None:
            return []

        # Normalize to a flat list first
        if isinstance(tickers, str):
            # Could be "GOLDBEES, ITBEES, MON100" — split on comma
            tickers = [t.strip() for t in tickers.split(',') if t.strip()]
        elif isinstance(tickers, list):
            # Each item itself could be a comma-separated string
            expanded = []
            for t in tickers:
                if isinstance(t, str) and ',' in t:
                    expanded.extend([x.strip() for x in t.split(',') if x.strip()])
                elif isinstance(t, str):
                    expanded.append(t.strip())
                else:
                    expanded.append(t)
            tickers = expanded

        # Remove .NS suffix and filter blanks
        return [t.replace('.NS', '').strip() for t in tickers if t and isinstance(t, str)]

    def _normalize_benchmark_metrics(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize benchmark metrics keys to a consistent format.
        
        Standard keys:
        - total_return_pct
        - cagr_pct
        - annualized_return_pct
        - max_drawdown_pct
        - sharpe_ratio
        - volatility_pct
        - win_rate_pct
        - xirr_pct
        - final_value
        - total_investment
        - calmar_ratio
        
        Args:
            raw_metrics: Metrics dictionary from various backtesters
            
        Returns:
            Normalized metrics dictionary
        """
        if not raw_metrics:
            return {}
            
        normalized = {}
        
        # Helper to extract value from multiple possible keys
        def get_val(keys, default=0.0):
            for k in keys:
                if k in raw_metrics:
                    val = raw_metrics[k]
                    # Handle string percentages like "12.34%"
                    if isinstance(val, str) and '%' in val:
                        try:
                            # Use regex to extract numeric part gracefully
                            import re
                            match = re.search(r"([-+]?\d*\.?\d+)", val.replace(',', ''))
                            if match:
                                return float(match.group(1))
                        except:
                            pass
                    # Handle currency strings like "₹1,234"
                    if isinstance(val, str) and ('₹' in val or ',' in val):
                        try:
                            import re
                            match = re.search(r"([-+]?\d*\.?\d+)", val.replace('₹', '').replace(',', ''))
                            if match:
                                return float(match.group(1))
                        except:
                            pass
                    return val
            return default

        normalized['total_return_pct'] = get_val(['total_return_pct', 'Total Return', 'total_return'])
        normalized['cagr_pct'] = get_val(['cagr_pct', 'CAGR', 'cagr'])
        normalized['annualized_return_pct'] = get_val(['annualized_return_pct', 'Annualized Return', 'annualized_return'])
        normalized['max_drawdown_pct'] = get_val(['max_drawdown_pct', 'Max Drawdown', 'max_drawdown'])
        normalized['sharpe_ratio'] = get_val(['sharpe_ratio', 'Sharpe Ratio', 'sharpe'])
        normalized['volatility_pct'] = get_val(['volatility_pct', 'Volatility', 'volatility'])
        normalized['win_rate_pct'] = get_val(['win_rate_pct', 'Win Rate', 'win_rate'])
        normalized['xirr_pct'] = get_val(['xirr_pct', 'XIRR', 'xirr'])
        normalized['final_value'] = get_val(['final_value', 'Final Value', 'final_capital'])
        normalized['total_investment'] = get_val(['total_investment', 'Total Investment', 'total_capital'])
        normalized['calmar_ratio'] = get_val(['calmar_ratio', 'Calmar Ratio', 'calmar'])
        
        # Ensure all values are sanitized
        return self._sanitize_data(normalized)

    def _sanitize_data(self, obj: Any) -> Any:
        """
        Recursively convert NaN/inf to 0 and numpy types to standard Python types
        for JSON serialization.
        
        Args:
            obj: Object to sanitize
            
        Returns:
            Sanitized object
        """
        import math
        import numpy as np
        
        if isinstance(obj, dict):
            return {str(k): self._sanitize_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_data(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_data(item) for item in obj)
        elif isinstance(obj, (int, float)):
            if math.isnan(obj) or math.isinf(obj):
                return 0
            return obj
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return 0
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize_data(obj.tolist())
        elif hasattr(obj, 'to_dict'): # For Pydantic models or similar
            try:
                return self._sanitize_data(obj.to_dict())
            except:
                return str(obj)
        elif hasattr(obj, 'isoformat'): # For datetime objects
            return obj.isoformat()
        else:
            return obj
