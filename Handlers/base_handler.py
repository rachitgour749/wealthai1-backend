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
    
    def _sanitize_data(self, obj: Any) -> Any:
        """
        Recursively convert NaN/inf to 0 in nested structures for JSON serialization
        
        Args:
            obj: Object to sanitize
            
        Returns:
            Sanitized object
        """
        import math
        
        if isinstance(obj, dict):
            return {k: self._sanitize_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_data(item) for item in obj]
        elif isinstance(obj, (int, float)):
            if math.isnan(obj) or math.isinf(obj):
                return 0
            return obj
        else:
            return obj
