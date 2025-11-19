# app/schemas.py
"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from enum import Enum


class UserRole(str, Enum):
    """User role enumeration"""
    ADVISOR = "advisor"
    STAFF = "staff"
    CLIENT = "client"


class Channel(str, Enum):
    """Communication channel enumeration"""
    WEB = "web"
    WHATSAPP = "whatsapp"
    PORTAL = "portal"
    OTHER = "other"


class ChatRequest(BaseModel):
    """Incoming chat request schema"""
    user_email: EmailStr = Field(..., description="User's Gmail ID")
    message: str = Field(..., min_length=1, description="Latest user message")
    conversation_id: Optional[str] = Field(None, description="Session continuity ID")
    user_role: Optional[UserRole] = Field(None, description="User's role")
    channel: Optional[Channel] = Field(None, description="Communication channel")


class DomainRelevance(str, Enum):
    """Domain relevance classification"""
    FINANCE_INTERMEDIARY = "finance_intermediary_domain"
    OUT_OF_SCOPE = "out_of_scope"


class Category(str, Enum):
    """Financial product categories"""
    MUTUAL_FUNDS = "Mutual Funds"
    INSURANCE = "Insurance"
    STOCK_MARKETS = "Stock Markets"


class ThirdLevelIntent(str, Enum):
    """Third-level intent classification"""
    EDUCATIONAL_EXPLANATION = "educational_explanation"
    REGULATION_OR_COMPLIANCE = "regulation_or_compliance"
    PRODUCT_SELECTION_OR_COMPARISON = "product_selection_or_comparison"
    CLIENT_CASE_PLANNING_OR_SUITABILITY = "client_case_planning_or_suitability"
    PORTFOLIO_OR_POLICY_REVIEW = "portfolio_or_policy_review"
    OPERATIONS_OR_TRANSACTION_SUPPORT = "operations_or_transaction_support"
    SALES_OR_MARKETING_COMMUNICATION = "sales_or_marketing_communication"
    TOOLS_OR_WORKFLOW_OR_AUTOMATION = "tools_or_workflow_or_automation"
    OTHER_IN_DOMAIN = "other_in_domain"


class Audience(str, Enum):
    """Target audience for response"""
    INTERMEDIARY = "intermediary"
    END_CLIENT = "end_client"


class ZohoCRMDataStatus(str, Enum):
    """Status of Zoho CRM data availability"""
    NOT_REQUIRED = "not_required"
    AVAILABLE = "available"
    MISSING = "missing"
    UNKNOWN = "unknown"


class RouterOutput(BaseModel):
    """Router LLM output schema"""
    domain_relevance: DomainRelevance
    primary_category: Optional[Category] = None
    additional_categories: List[Category] = Field(default_factory=list)
    is_multi_category: bool = False
    third_level_intent: Optional[ThirdLevelIntent] = None
    audience: Optional[Audience] = None
    use_zoho_crm_data: bool = False
    use_common_kb: bool = False
    zoho_crm_data_status: ZohoCRMDataStatus = ZohoCRMDataStatus.NOT_REQUIRED


class ChatResponse(BaseModel):
    """Outgoing chat response schema"""
    reply: str = Field(..., description="Model's final answer to user")
    router_metadata: Dict[str, Any] = Field(..., description="Parsed router JSON")
    used_user_context: bool = Field(..., description="Whether user-specific context was used")
    used_common_kb: bool = Field(..., description="Whether common KB was used")
    conversation_id: str = Field(..., description="Session ID for continuity")


class Message(BaseModel):
    """Single conversation message"""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class SessionState(BaseModel):
    """Session state model"""
    conversation_id: str
    messages: List[Message] = Field(default_factory=list)
    summary: Optional[str] = None
    session_metadata: Dict[str, Any] = Field(default_factory=dict)