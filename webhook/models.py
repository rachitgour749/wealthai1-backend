from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class StrategyStatus(str, Enum):
    """Strategy status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    FAILED = "failed"

class StrategyCreate(BaseModel):
    """Request model for creating a strategy"""
    strategy_name: str = Field(..., min_length=1, max_length=255, description="Name of the strategy")
    user_email: Optional[str] = Field(None, description="User email address")
    webhook: str = Field(..., min_length=1, description="Webhook URL")
    strategy_data: Dict[str, Any] = Field(..., description="Strategy configuration data")
    is_active: bool = Field(default=True, description="Whether the strategy is active")
    
    @validator('strategy_name')
    def validate_strategy_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Strategy name cannot be empty')
        return v.strip()
    
    @validator('webhook')
    def validate_webhook(cls, v):
        if not v or not v.strip():
            raise ValueError('Webhook URL cannot be empty')
        return v.strip()

class StrategyUpdate(BaseModel):
    """Request model for updating a strategy"""
    strategy_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Name of the strategy")
    user_email: Optional[str] = Field(None, description="User email address")
    webhook: Optional[str] = Field(None, min_length=1, description="Webhook URL")
    strategy_data: Optional[Dict[str, Any]] = Field(None, description="Strategy configuration data")
    is_active: Optional[bool] = Field(None, description="Whether the strategy is active")
    
    @validator('strategy_name')
    def validate_strategy_name(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Strategy name cannot be empty')
        return v.strip() if v else v
    
    @validator('webhook')
    def validate_webhook(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Webhook URL cannot be empty')
        return v.strip() if v else v

class StrategyStatusUpdate(BaseModel):
    """Request model for updating strategy status"""
    is_active: bool = Field(..., description="Whether the strategy is active")

class StrategyResponse(BaseModel):
    """Response model for strategy data"""
    id: int
    strategy_name: str
    user_email: Optional[str]
    webhook: str
    strategy_data: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class JsonGenerate(BaseModel):
    """Request model for generating JSON data"""
    client_ids: List[str] = Field(..., min_items=1, description="List of client IDs")
    capitals: List[float] = Field(..., min_items=1, description="List of capital amounts")
    strategy_type: str = Field(..., description="Type of strategy")
    
    @validator('client_ids')
    def validate_client_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one client ID is required')
        return v
    
    @validator('capitals')
    def validate_capitals(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one capital amount is required')
        if any(capital <= 0 for capital in v):
            raise ValueError('All capital amounts must be positive')
        return v

class JsonSave(BaseModel):
    """Request model for saving JSON data"""
    user_email: str = Field(..., description="User email address")
    json_data: Dict[str, Any] = Field(..., description="JSON data to save")
    strategy_name: Optional[str] = Field(None, description="Name of the strategy")
    
    @validator('user_email')
    def validate_user_email(cls, v):
        if not v or not v.strip():
            raise ValueError('User email cannot be empty')
        return v.strip()

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: datetime
    database_connected: bool
    total_strategies: int
    active_strategies: int
    
    class Config:
        from_attributes = True

class WebhookNotification(BaseModel):
    """Model for webhook notifications"""
    strategy_id: int
    strategy_name: str
    user_email: Optional[str]
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    
    class Config:
        from_attributes = True

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    
    class Config:
        from_attributes = True

class SuccessResponse(BaseModel):
    """Response model for successful operations"""
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime
    
    class Config:
        from_attributes = True
