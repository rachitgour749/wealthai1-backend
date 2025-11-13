from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class PaymentStatus(str, Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"

class PaymentMethod(str, Enum):
    """Payment method enumeration"""
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"
    CARD_LESS_EMI = "cardless_emi"

class OrderRequest(BaseModel):
    """Request model for creating payment order"""
    amount: int = Field(..., description="Amount in paise (e.g., 50000 for ₹500)")
    currency: str = Field(default="INR", description="Currency code")
    receipt: Optional[str] = Field(None, description="Receipt ID")
    notes: Optional[Dict[str, str]] = Field(None, description="Additional notes")
    customer: Optional[Dict[str, str]] = Field(None, description="Customer information")
    prefill: Optional[Dict[str, str]] = Field(None, description="Prefill customer details")
    callback_url: Optional[str] = Field(None, description="Callback URL after payment")
    cancel_url: Optional[str] = Field(None, description="Cancel URL")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be greater than 0')
        if v < 100:  # Minimum ₹1
            raise ValueError('Amount must be at least ₹1 (100 paise)')
        return v

class OrderResponse(BaseModel):
    """Response model for payment order"""
    id: str
    entity: str
    amount: int
    amount_paid: int
    amount_due: int
    currency: str
    receipt: str
    offer_id: Optional[str] = None
    status: str
    attempts: int
    notes: Dict[str, str]
    created_at: int
    updated_at: Optional[int] = None

class PaymentVerificationRequest(BaseModel):
    """Request model for payment verification"""
    razorpay_order_id: str = Field(..., description="Razorpay order ID")
    razorpay_payment_id: str = Field(..., description="Razorpay payment ID")
    razorpay_signature: str = Field(..., description="Payment signature for verification")

class PaymentVerificationResponse(BaseModel):
    """Response model for payment verification"""
    verified: bool
    order_id: str
    payment_id: str
    amount: int
    currency: str
    status: str
    message: str

class RefundRequest(BaseModel):
    """Request model for payment refund"""
    payment_id: str = Field(..., description="Payment ID to refund")
    amount: Optional[int] = Field(None, description="Amount to refund in paise (full refund if not specified)")
    speed: str = Field(default="normal", description="Refund speed: normal, instant")
    notes: Optional[Dict[str, str]] = Field(None, description="Refund notes")
    receipt: Optional[str] = Field(None, description="Refund receipt")

class RefundResponse(BaseModel):
    """Response model for payment refund"""
    id: str
    entity: str
    amount: int
    currency: str
    payment_id: str
    notes: Dict[str, str]
    receipt: str
    status: str
    speed_processed: str
    speed_requested: str

class PaymentWebhookData(BaseModel):
    """Model for webhook data from Razorpay"""
    entity: str
    account_id: str
    event: str
    contains: List[str]
    payload: Dict[str, Any]
    created_at: int

class CustomerInfo(BaseModel):
    """Customer information model"""
    name: str
    email: str
    contact: str
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    country: Optional[str] = None

class PaymentPlan(BaseModel):
    """Payment plan/subscription model"""
    name: str
    description: str
    amount: int
    currency: str = "INR"
    interval: str = "month"  # month, year, week, day
    interval_count: int = 1
    trial_period_days: Optional[int] = None
    notes: Optional[Dict[str, str]] = None

class PaymentHistoryItem(BaseModel):
    """Payment history item model"""
    order_id: str
    payment_id: str
    amount: int
    currency: str
    status: PaymentStatus
    method: PaymentMethod
    created_at: datetime
    updated_at: datetime
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None

class PaymentAnalytics(BaseModel):
    """Payment analytics model"""
    total_transactions: int
    total_amount: int
    successful_transactions: int
    failed_transactions: int
    pending_transactions: int
    average_transaction_amount: float
    currency: str
    period: str  # daily, weekly, monthly, yearly
