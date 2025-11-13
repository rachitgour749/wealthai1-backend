from pydantic import BaseModel, EmailStr, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

class CustomStrategyRequest(BaseModel):
    user_email: str
    user_phone: Optional[str] = None
    strategy_description: str
    
    @validator('user_phone')
    def validate_phone(cls, v):
        if not v:
            raise ValueError('Phone number is required')
        # Remove all non-digit characters
        phone_digits = ''.join(filter(str.isdigit, v))
        
        # Handle international numbers with country codes
        if len(phone_digits) == 12 and phone_digits.startswith('91'):
            # Remove country code 91
            phone_digits = phone_digits[2:]
        elif len(phone_digits) == 13 and phone_digits.startswith('91'):
            # Remove country code 91
            phone_digits = phone_digits[2:]
        
        if len(phone_digits) != 10:
            raise ValueError('Phone number must be exactly 10 digits')
        return phone_digits

class CustomStrategyAnalysis(BaseModel):
    strategy_rating: int
    trading_instruments: Optional[str] = "ANY INSTRUMENT"
    time_frame: str
    trading_rules: List[str]
    position_sizing: str
    risk_management: str
    strategy_logic: str

class CustomStrategyResponse(BaseModel):
    success: bool
    strategy_id: Optional[int] = None
    analysis: Optional[CustomStrategyAnalysis] = None
    message: str
    error: Optional[str] = None

class CustomStrategySaveRequest(BaseModel):
    user_email: str
    user_phone: str
    strategy_description: str
    analysis: CustomStrategyAnalysis
    
    @validator('user_phone')
    def validate_phone(cls, v):
        if not v:
            raise ValueError('Phone number is required')
        # Remove all non-digit characters
        phone_digits = ''.join(filter(str.isdigit, v))
        
        # Handle international numbers with country codes
        if len(phone_digits) == 12 and phone_digits.startswith('91'):
            # Remove country code 91
            phone_digits = phone_digits[2:]
        elif len(phone_digits) == 13 and phone_digits.startswith('91'):
            # Remove country code 91
            phone_digits = phone_digits[2:]
        
        if len(phone_digits) != 10:
            raise ValueError('Phone number must be exactly 10 digits')
        return phone_digits

class CustomStrategyListResponse(BaseModel):
    success: bool
    strategies: List[Dict[str, Any]]
    count: int

class EmailNotificationRequest(BaseModel):
    strategy_id: int
    user_email: str
    user_phone: Optional[str] = None
    strategy_description: str
    analysis: CustomStrategyAnalysis
    
    @validator('user_phone')
    def validate_phone(cls, v):
        if not v:
            raise ValueError('Phone number is required')
        # Remove all non-digit characters
        phone_digits = ''.join(filter(str.isdigit, v))
        
        # Handle international numbers with country codes
        if len(phone_digits) == 12 and phone_digits.startswith('91'):
            # Remove country code 91
            phone_digits = phone_digits[2:]
        elif len(phone_digits) == 13 and phone_digits.startswith('91'):
            # Remove country code 91
            phone_digits = phone_digits[2:]
        
        if len(phone_digits) != 10:
            raise ValueError('Phone number must be exactly 10 digits')
        return phone_digits
