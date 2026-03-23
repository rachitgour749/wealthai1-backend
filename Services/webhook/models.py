from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
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
    user_email: str = Field(..., alias="userEmail", description="User email address")
    json_data: Dict[str, Any] = Field(..., alias="jsonData", description="JSON data to save")
    strategy_name: Optional[str] = Field(None, alias="strategyName", description="Name of the strategy")
    
    @validator('user_email')
    def validate_user_email(cls, v):
        if not v or not v.strip():
            raise ValueError('User email cannot be empty')
        return v.strip()
    
    class Config:
        allow_population_by_field_name = True

class DeployRequest(BaseModel):
    """Request model for deploy functionality - generates and saves JSON data"""
    user_email: str = Field(..., description="User email address")
    client_ids: List[str] = Field(..., min_items=1, description="List of client IDs")
    capitals: List[float] = Field(..., min_items=1, description="List of capital amounts")
    strategy_name: Optional[str] = Field(None, description="Name of the strategy")
    
    @validator('user_email')
    def validate_user_email(cls, v):
        if not v or not v.strip():
            raise ValueError('User email cannot be empty')
        return v.strip()
    
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

class WebhookCreateRequest(BaseModel):
    """Request model for creating a webhook"""
    user_id: str = Field(..., description="User identifier")
    RA_code: Optional[str] = Field(None, description="RA code (valid only if source is RA)")
    source: str = Field(..., description="INDIVIDUAL or RA")
    category: str = Field(..., description="FNO or EQUITY")
    strategy_type: str = Field(..., description="Type of strategy")
    name: str = Field(..., description="Strategy name")
    ra_code: Optional[str] = Field(None, description="RA code (same as RA_code, for consistency)")
    client_info: Dict[str, Any] = Field(..., description="Client mapping/info")

    @validator('source')
    def validate_source(cls, v):
        if v not in ['INDIVIDUAL', 'RA']:
            raise ValueError('Source must be INDIVIDUAL or RA')
        return v

    @validator('category')
    def validate_category(cls, v):
        if v not in ['FNO', 'EQUITY']:
            raise ValueError('Category must be FNO or EQUITY')
        return v

class WebhookCreateResponse(BaseModel):
    """Response model for creating a webhook"""
    status: str
    source: str
    run_id: Optional[str] = None
    secret_key: Optional[str] = None
    message: Optional[str] = None

# RA CRUD Schemas
class RACreateRequest(BaseModel):
    ra_email: str = Field(..., description="Email of the Research Analyst")
    strategy_type: str = Field(..., description="Strategy type (e.g., RS_Stocks, ETF_Rotation)")

class RAUpdateRequest(BaseModel):
    secret_key: Optional[str] = None
    is_active: Optional[bool] = None

class RAResponse(BaseModel):
    id: int
    ra_email: Optional[str]
    ra_code: str
    strategy_type: str
    secret_key: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

# Execution Engine Schemas
class TradeExecuteRequest(BaseModel):
    strategy_type: str = Field(..., description="Strategy name/type")
    symbol: str = Field(..., description="Trading symbol (e.g., RELIANCE)")
    exchnge: str = Field(..., description="Exchange name (NSE/BSE)")
    order_side: str = Field(..., description="BUY or SELL")
    authorized_email: List[str] = Field(..., description="List of user emails to execute for")
    ra_code: str = Field(..., description="RA identifier")

class UserExecutionDetail(BaseModel):
    email: str
    status: str
    message: str
    order_id: Optional[str] = None

class TradeExecuteIndividualRequest(BaseModel):
    run_id: str
    strategy_type: str
    symbol: str
    exchnge: str
    order_side: str
    authorized_email: str # Single email for individual mode

class UnifiedTradeExecuteRequest(BaseModel):
    ra_code: Optional[str] = None
    run_id: Optional[str] = None
    strategy_type: str
    symbol: str
    exchnge: str
    order_side: str
    authorized_email: Union[List[str], str]

class TradeExecuteResponse(BaseModel):
    status: str
    processed: int
    executed: int
    failures: int
    details: List[UserExecutionDetail]

    class Config:
        from_attributes = True

class WebhookDetailResponse(BaseModel):
    run_id: str
    name: Optional[str] = None
    strategy_type: str
    client_info: Dict[str, Any]
    status: str
    category: str
    source: str
    ra_code: Optional[str] = None
    secret_key: Optional[str] = None

class WebhookListResponse(BaseModel):
    user_id: str
    webhooks: List[WebhookDetailResponse]

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
