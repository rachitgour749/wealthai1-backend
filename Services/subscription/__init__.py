# Services/Subscription/__init__.py
"""Subscription module for WealthAI1 - Refactored to match ChatAI1 structure"""

# Export main router (using relative imports for case-insensitivity)
from .api.subscription import subscription_router

# Export commonly used schemas
from .subscription_schemas import (
    ProductCode,
    SubscriptionStatus,
    SubscriptionPlan,
    SubscriptionType,
    ProductTrialActivationRequest,
    ProductTrialActivationResponse,
)

# Export primary service
from .services.subscription_service import subscription_service

__all__ = [
    "subscription_router",
    "ProductCode",
    "SubscriptionStatus",
    "SubscriptionPlan",
    "SubscriptionType",
    "ProductTrialActivationRequest",
    "ProductTrialActivationResponse",
    "subscription_service",
]
