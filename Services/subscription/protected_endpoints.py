"""
Example of how to protect existing endpoints with subscription middleware
This shows how to integrate subscription checks into your existing API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

from .middleware import require_premium_access, require_trial_access, require_basic_access
from .auth_integration import get_user_with_subscription

# Create router for protected endpoints
protected_router = APIRouter(prefix="/api/protected", tags=["protected-endpoints"])

logger = logging.getLogger(__name__)

@protected_router.get("/premium-feature")
async def premium_feature(
    access_info: Dict[str, Any] = Depends(require_premium_access)
):
    """
    Example endpoint that requires premium access (trial or paid subscription)
    """
    try:
        user_email = access_info["user_email"]
        logger.info(f"Premium feature accessed by user: {user_email}")
        
        # Your premium feature logic here
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Premium feature accessed successfully",
                "user_email": user_email,
                "subscription_status": access_info["subscription_status"],
                "days_remaining": access_info["days_remaining"],
                "feature_data": {
                    "premium_content": "This is premium content",
                    "access_level": "premium"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in premium feature: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to access premium feature: {str(e)}")

@protected_router.get("/trial-only-feature")
async def trial_only_feature(
    access_info: Dict[str, Any] = Depends(require_trial_access)
):
    """
    Example endpoint that requires active trial access
    """
    try:
        user_email = access_info["user_email"]
        logger.info(f"Trial-only feature accessed by user: {user_email}")
        
        # Your trial-only feature logic here
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Trial-only feature accessed successfully",
                "user_email": user_email,
                "is_trial_active": access_info["is_trial_active"],
                "days_remaining": access_info["days_remaining"],
                "feature_data": {
                    "trial_content": "This is trial-only content",
                    "access_level": "trial"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in trial-only feature: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to access trial-only feature: {str(e)}")

@protected_router.get("/basic-feature")
async def basic_feature(
    access_info: Dict[str, Any] = Depends(require_basic_access)
):
    """
    Example endpoint that requires basic access (always available)
    """
    try:
        user_email = access_info["user_email"]
        logger.info(f"Basic feature accessed by user: {user_email}")
        
        # Your basic feature logic here
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Basic feature accessed successfully",
                "user_email": user_email,
                "subscription_status": access_info.get("subscription_status", "unknown"),
                "feature_data": {
                    "basic_content": "This is basic content available to all users",
                    "access_level": "basic"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in basic feature: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to access basic feature: {str(e)}")

@protected_router.get("/user-dashboard")
async def user_dashboard(
    user_info: Dict[str, Any] = Depends(get_user_with_subscription)
):
    """
    Example endpoint that shows user dashboard with subscription info
    """
    try:
        user_email = user_info["user_email"]
        subscription_status = user_info["subscription_status"]
        
        logger.info(f"Dashboard accessed by user: {user_email}")
        
        # Determine dashboard content based on subscription status
        dashboard_content = {
            "basic_features": [
                "View basic data",
                "Basic analytics",
                "Limited reports"
            ]
        }
        
        if subscription_status.can_access_premium:
            dashboard_content["premium_features"] = [
                "Advanced analytics",
                "Premium reports",
                "Custom dashboards",
                "API access",
                "Priority support"
            ]
        
        if subscription_status.is_trial_active:
            dashboard_content["trial_info"] = {
                "days_remaining": subscription_status.days_remaining,
                "trial_end_date": subscription_status.trial_end_date.isoformat() if subscription_status.trial_end_date else None,
                "upgrade_prompt": "Upgrade now to continue after trial ends!"
            }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Dashboard loaded successfully",
                "user": {
                    "email": user_email,
                    "name": user_info["user_name"]
                },
                "subscription": {
                    "status": subscription_status.status.value,
                    "plan": subscription_status.plan.value,
                    "is_trial_active": subscription_status.is_trial_active,
                    "can_access_premium": subscription_status.can_access_premium,
                    "days_remaining": subscription_status.days_remaining
                },
                "dashboard_content": dashboard_content
            }
        )
        
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard: {str(e)}")

@protected_router.get("/subscription-status")
async def get_subscription_status_endpoint(
    user_info: Dict[str, Any] = Depends(get_user_with_subscription)
):
    """
    Get detailed subscription status for the authenticated user
    """
    try:
        user_email = user_info["user_email"]
        subscription_status = user_info["subscription_status"]
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "user_email": user_email,
                "subscription": {
                    "status": subscription_status.status.value,
                    "plan": subscription_status.plan.value,
                    "is_trial_active": subscription_status.is_trial_active,
                    "can_access_premium": subscription_status.can_access_premium,
                    "days_remaining": subscription_status.days_remaining,
                    "trial_start_date": subscription_status.trial_start_date.isoformat() if subscription_status.trial_start_date else None,
                    "trial_end_date": subscription_status.trial_end_date.isoformat() if subscription_status.trial_end_date else None,
                    "subscription_start_date": subscription_status.subscription_start_date.isoformat() if subscription_status.subscription_start_date else None,
                    "subscription_end_date": subscription_status.subscription_end_date.isoformat() if subscription_status.subscription_end_date else None
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get subscription status: {str(e)}")

# Example of how to modify existing endpoints
def add_subscription_protection_to_existing_endpoint(original_endpoint):
    """
    Decorator to add subscription protection to existing endpoints
    
    Usage:
    @add_subscription_protection_to_existing_endpoint
    async def your_existing_endpoint():
        # Your existing logic
        pass
    """
    async def protected_wrapper(*args, **kwargs):
        # Add subscription check here
        # This is a simplified example - you'd need to adapt based on your specific needs
        return await original_endpoint(*args, **kwargs)
    
    return protected_wrapper

