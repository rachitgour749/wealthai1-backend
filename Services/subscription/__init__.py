# Services/Subscription/__init__.py
"""Subscription module for WealthAI1 - Refactored to match ChatAI1 structure"""

# Export main router
from Services.Subscription.api.subscription import subscription_router

# Export commonly used schemas
from Services.Subscription.subscription_schemas import (
    ProductCode,
    SubscriptionStatus,
    SubscriptionPlan,
    SubscriptionType,
    ProductTrialActivationRequest,
    ProductTrialActivationResponse,
)

# Export primary service
from Services.Subscription.services.subscription_service import subscription_service

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
