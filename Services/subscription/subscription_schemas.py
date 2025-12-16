# Services/Subscription/subscription_schemas.py
"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration"""
    TRIAL = "trial"
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class SubscriptionPlan(str, Enum):
    """Subscription plan enumeration"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ProductCode(str, Enum):
    """Product code enumeration"""
    MARKETAI = "MARKETAI"
    CHATAI = "CHATAI"
    TRADAI = "TRADEAI"
    AUTOMATIONAI = "AUTOMATIONAI"


class SubscriptionType(str, Enum):
    """Subscription type enumeration"""
    TRIAL = "trial"
    PAID = "paid"
    BUNDLE = "bundle"


class SubscriptionRequest(BaseModel):
    """Request model for creating subscription"""
    user_email: str = Field(..., description="User email address")
    user_name: Optional[str] = Field(None, description="User name")
    plan: SubscriptionPlan = Field(default=SubscriptionPlan.FREE, description="Subscription plan")
    
    @validator('user_email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class SubscriptionResponse(BaseModel):
    """Response model for subscription"""
    id: str
    user_email: str
    user_name: Optional[str]
    plan: SubscriptionPlan
    status: SubscriptionStatus
    trial_start_date: Optional[datetime] = None
    trial_end_date: Optional[datetime] = None
    subscription_start_date: Optional[datetime] = None
    subscription_end_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    is_trial_active: bool
    days_remaining: int


class SubscriptionStatusResponse(BaseModel):
    """Response model for subscription status check"""
    user_email: str
    has_subscription: bool = True
    status: SubscriptionStatus
    plan: SubscriptionPlan
    plan_code: Optional[str] = None
    is_trial_active: bool
    trial_end_date: Optional[datetime] = None
    subscription_end_date: Optional[datetime] = None
    days_remaining: int
    can_access_premium: bool


class TrialExtensionRequest(BaseModel):
    """Request model for extending trial period"""
    user_email: str = Field(..., description="User email address")
    extension_days: int = Field(..., description="Number of days to extend trial", ge=1, le=30)
    
    @validator('user_email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class SubscriptionUpgradeRequest(BaseModel):
    """Request model for upgrading subscription"""
    user_email: str = Field(..., description="User email address")
    new_plan: SubscriptionPlan = Field(..., description="New subscription plan")
    
    @validator('user_email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class SubscriptionAnalytics(BaseModel):
    """Subscription analytics model"""
    total_users: int
    trial_users: int
    active_subscribers: int
    expired_subscriptions: int
    trial_conversion_rate: float
    average_trial_duration: float
    plan_distribution: Dict[str, int]


# Product-based subscription models
class ProductSubscriptionRequest(BaseModel):
    """Request model for creating product subscription"""
    user_email: str = Field(..., description="User email address")
    product_code: ProductCode = Field(..., description="Product code (MARKETAI, CHATAI, TRADAI)")
    subscription_type: SubscriptionType = Field(default=SubscriptionType.TRIAL, description="Subscription type")
    plan_code: Optional[str] = Field(default="FREE", description="Plan code for paid subscriptions")
    
    @validator('user_email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class ProductAccessResponse(BaseModel):
    """Response model for product access check"""
    user_email: str
    product_code: ProductCode
    has_access: bool
    access_type: str  # trial, paid, bundle
    days_remaining: int
    trial_end_date: Optional[datetime] = None
    paid_end_date: Optional[datetime] = None
    status: SubscriptionStatus
    plan: Optional[str] = None


class ProductSubscriptionResponse(BaseModel):
    """Response model for product subscription"""
    id: str
    user_email: str
    product_code: ProductCode
    subscription_type: SubscriptionType
    status: SubscriptionStatus
    plan_code: str
    trial_start_date: Optional[datetime] = None
    trial_end_date: Optional[datetime] = None
    paid_start_date: Optional[datetime] = None
    paid_end_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    is_trial_active: bool
    is_paid_active: bool
    days_remaining: int
    is_bundle_subscription: bool = False
    # ChatAI Token Tracking Fields
    chatai_key: Optional[str] = None
    total_tokens: int = 0
    used_tokens: int = 0


class BundleSubscriptionRequest(BaseModel):
    """Request model for bundle subscription"""
    user_email: str = Field(..., description="User email address")
    product_codes: List[ProductCode] = Field(..., description="List of product codes for bundle")
    plan_code: str = Field(default="BUNDLE", description="Bundle plan code")
    
    @validator('user_email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('product_codes')
    def validate_product_codes(cls, v):
        if len(v) < 2:
            raise ValueError('Bundle must include at least 2 products')
        return v


class ProductUpgradeRequest(BaseModel):
    """Request model for upgrading product subscription"""
    user_email: str = Field(..., description="User email address")
    product_code: ProductCode = Field(..., description="Product code to upgrade")
    plan_code: str = Field(..., description="New plan code")
    
    @validator('user_email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class AllProductsStatusResponse(BaseModel):
    """Response model for all products status"""
    user_email: str
    products: Dict[str, ProductAccessResponse]
    total_active_products: int
    has_bundle: bool
    bundle_products: List[ProductCode] = []


# New models for Product Trial Activation API
class ProductTrialActivationRequest(BaseModel):
    """Request model for activating product trial with plan_code"""
    user_email: str = Field(..., description="User email address")
    user_name: str = Field(..., description="User name")
    plan_code: int = Field(..., description="Plan code from plan_master table", gt=0)
    
    @validator('user_email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('plan_code')
    def validate_plan_code(cls, v):
        if v <= 0:
            raise ValueError('Plan code must be a positive number')
        return v


class ProductTrialActivationResponse(BaseModel):
    """Response model for product trial activation"""
    user_email: str
    startDate: datetime
    endDate: datetime
    planCode: int
    planName: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserDetailsResponse(BaseModel):
    """Response model for user details"""
    user_email: str
    user_name: Optional[str]
    phone_no: Optional[str] = None
    status: str  # TRIAL/PAID
    created_at: datetime
    updated_at: datetime


class ProductManagerRecord(BaseModel):
    """Response model for product entries stored in product_manager"""
    id: str
    user_email: str
    product_code: ProductCode
    product_code_label: str
    subscription_type: SubscriptionType
    status: SubscriptionStatus
    subscription_start_date: Optional[datetime]
    subscription_end_date: Optional[datetime]
    plan_name: Optional[str] = None
    chatai_key: Optional[str] = None
    total_token: Optional[str] = None
    used_token: Optional[str] = None
    remaining_token: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ZohoPaymentCallbackRequest(BaseModel):
    """Request model for Zoho payment callback"""
    EmailID: str = Field(..., description="User email address")
    SubscriptionID: str = Field(..., description="Zoho subscription ID")
    SubscriptionName: str = Field(..., description="Name of the subscription")
    PlanName: str = Field(..., description="Name of the plan purchased")
    
    @validator('EmailID')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class PaymentSuccessParams(BaseModel):
    """
    Model for validating GET parameters if needed.
    (Optional usage since arguments are flattened in the endpoint,
     but good for documentation/internal passing).
    """
    emailID: str
    planname: str
    subscriptionID: str
    nextBillingDate: Optional[str] = None
    transactionID: Optional[str] = None
    invoiceNumber: Optional[str] = None

