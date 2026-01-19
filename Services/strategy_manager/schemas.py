"""Pydantic schemas for centralized strategy management system"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SaveStrategyRequest(BaseModel):
    """Request schema for saving a strategy"""
    user_id: str
    strategy_type: str
    strategy_name: Optional[str] = None # Support from payload
    tickers: Optional[Any] = None # Support List[str] or str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strategies_parameters: Optional[Dict[str, Any]] = None
    use_custom_date: bool = False

class DeleteStrategyClientRequest(BaseModel):
    """Request schema for removing clients from a strategy"""
    run_id: str
    clients: List[str]

class DeployStrategyRequest(BaseModel):
    """Request schema for deploying a strategy"""
    run_id: str
    client_info: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    reference_capital: Optional[float] = None
    email_notification: bool = False
    telegram_notification: bool = False
    user_code: Optional[str] = None

class StrategyResponse(BaseModel):
    """Response schema for strategy operations"""
    success: bool
    message: str
    run_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class StrategyInstanceSchema(BaseModel):
    """Schema for a strategy instance entry"""
    id: int
    user_id: str
    strategy_name: Optional[str] = None
    strategy_type: str
    tickers: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strategies_parameters: Optional[Dict[str, Any]] = None
    use_custom_date: bool = False
    run_id: str
    client_info: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    status: str
    reference_capital: Optional[float] = None
    last_execution_date: Optional[datetime] = None
    next_execution_date: Optional[datetime] = None
    email_notification: bool = False
    telegram_notification: bool = False
    user_code: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
