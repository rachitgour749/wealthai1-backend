from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from Services.subscription.subscription_schemas import SubscriptionStatus, SubscriptionType, ProductCode

class ClientListItem(BaseModel):
    """Summary information for client list"""
    user_email: str
    user_name: Optional[str] = None
    phone_no: Optional[str] = None
    status: str
    role: str
    created_at: datetime

class ProductSummary(BaseModel):
    """Summary of a product subscription"""
    product_code: str
    subscription_type: str
    status: str
    subscription_start_date: Optional[datetime] = None
    subscription_end_date: Optional[datetime] = None
    plan_name: Optional[str] = None
    remaining_tokens: Optional[int] = None

class ClientDetailResponse(BaseModel):
    """Detailed client information"""
    user_email: str
    user_name: Optional[str] = None
    phone_no: Optional[str] = None
    status: str
    role: str
    created_at: datetime
    updated_at: datetime
    products: List[ProductSummary] = []

class UpdateClientRequest(BaseModel):
    """Request to update basic client info"""
    user_name: Optional[str] = None
    phone_no: Optional[str] = None
    status: Optional[str] = None # TRIAL, PAID, etc.
    role: Optional[str] = None # CLIENT, ADMIN, RM, etc.

class UpdatePlanRequest(BaseModel):
    """Request to update client plan"""
    plan_code: int = Field(..., description="Plan code from plan_master")

class UpdateSubscriptionDatesRequest(BaseModel):
    """Request to update subscription dates for a specific product"""
    product_code: ProductCode
    subscription_start_date: Optional[datetime] = None
    subscription_end_date: Optional[datetime] = None

class UpdateCreditsRequest(BaseModel):
    """Request to increase or decrease user credits (tokens)"""
    product_code: ProductCode = Field(default=ProductCode.CHATAI)
    amount: int = Field(..., gt=0, description="Amount of tokens to add or subtract")
    operation: Literal["increase", "decrease"] = Field(..., description="Operation type")

class AdminOperationResponse(BaseModel):
    """Standard response for admin operations"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
