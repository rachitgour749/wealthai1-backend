from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import List, Dict, Any, Optional

class ExchangePolicy(ABC):
    """
    Interface for market-specific rules and exchange logic.
    """
    
    @property
    @abstractmethod
    def market_name(self) -> str:
        pass

    @property
    @abstractmethod
    def currency_symbol(self) -> str:
        pass

    @property
    @abstractmethod
    def market_start_time(self) -> time:
        pass

    @property
    @abstractmethod
    def market_end_time(self) -> time:
        pass

    @abstractmethod
    def is_trading_day(self, date_obj: datetime.date) -> bool:
        pass

    @abstractmethod
    def calculate_transaction_costs(self, action: str, asset_type: str, amount: float, brokerage_percent: float) -> Dict[str, float]:
        pass

    @abstractmethod
    def format_currency(self, amount: float) -> str:
        pass
