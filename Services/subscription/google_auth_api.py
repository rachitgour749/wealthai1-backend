"""
Google Authentication API endpoints for subscription system
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging

from subscription.auth_integration import google_auth_integration
from subscription.service import subscription_service
from subscription.models import SubscriptionStatusResponse

# Create router
google_auth_router = APIRouter(prefix="/api/auth", tags=["google-auth"])

# Configure logging
logger = logging.getLogger(__name__)

@google_auth_router.post("/google-login")
async def google_login(
    authorization: str = Header(..., description="Google OAuth token")
):
    """
    Handle Google OAuth login and automatically create/check subscription
    This endpoint should be called when a user successfully logs in via Google OAuth
    """
    try:
        logger.info("Processing Google login request")
        
        result = await google_auth_integration.handle_google_login(authorization)
        
        logger.info(f"Google login processed successfully for user: {result['user_email']}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": result["message"],
                "user": {
                    "email": result["user_email"],
                    "name": result["user_name"]
                },
                "subscription": {
                    "status": result["subscription_status"].status.value,
                    "plan": result["subscription_status"].plan.value,
                    "is_trial_active": result["subscription_status"].is_trial_active,
                    "can_access_premium": result["subscription_status"].can_access_premium,
                    "days_remaining": result["subscription_status"].days_remaining,
                    "trial_end_date": result["subscription_status"].trial_end_date.isoformat() if result["subscription_status"].trial_end_date else None,
                    "subscription_end_date": result["subscription_status"].subscription_end_date.isoformat() if result["subscription_status"].subscription_end_date else None
                },
                "trial_created": result["trial_created"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Google login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process Google login: {str(e)}")

@google_auth_router.get("/user-info")
async def get_user_info(
    authorization: str = Header(..., description="Google OAuth token")
):
    """
    Get current user information and subscription status
    """
    try:
        logger.info("Getting user information")
        
        user_info = await google_auth_integration.get_user_subscription_info(authorization)
        
        logger.info(f"User information retrieved for: {user_info['user_email']}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "user": {
                    "email": user_info["user_email"],
                    "name": user_info["user_name"]
                },
                "subscription": {
                    "status": user_info["subscription_status"].status.value,
                    "plan": user_info["subscription_status"].plan.value,
                    "is_trial_active": user_info["subscription_status"].is_trial_active,
                    "can_access_premium": user_info["subscription_status"].can_access_premium,
                    "days_remaining": user_info["subscription_status"].days_remaining,
                    "trial_end_date": user_info["subscription_status"].trial_end_date.isoformat() if user_info["subscription_status"].trial_end_date else None,
                    "subscription_end_date": user_info["subscription_status"].subscription_end_date.isoformat() if user_info["subscription_status"].subscription_end_date else None
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user information: {str(e)}")

@google_auth_router.get("/check-access")
async def check_access(
    authorization: str = Header(..., description="Google OAuth token")
):
    """
    Check if user has access to premium features
    """
    try:
        logger.info("Checking user access")
        
        user_info = await google_auth_integration.get_user_subscription_info(authorization)
        
        logger.info(f"Access check completed for: {user_info['user_email']}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "has_access": user_info["can_access_premium"],
                "is_trial_active": user_info["is_trial_active"],
                "days_remaining": user_info["days_remaining"],
                "message": "Access granted" if user_info["can_access_premium"] else "Subscription required"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check access: {str(e)}")

@google_auth_router.post("/start-trial")
async def start_trial(
    authorization: str = Header(..., description="Google OAuth token")
):
    """
    Manually start a trial for the authenticated user
    """
    try:
        logger.info("Starting trial for user")
        
        user_info = await google_auth_integration.get_user_subscription_info(authorization)
        user_email = user_info["user_email"]
        user_name = user_info["user_name"]
        
        # Check if user already has an active trial or subscription
        if user_info["can_access_premium"]:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "User already has an active trial or subscription",
                    "subscription_status": user_info["subscription_status"].status.value,
                    "days_remaining": user_info["days_remaining"]
                }
            )
        
        # Create new trial
        from subscription.models import SubscriptionRequest, SubscriptionPlan
        
        subscription_request = SubscriptionRequest(
            user_email=user_email,
            user_name=user_name,
            plan=SubscriptionPlan.FREE
        )
        
        new_subscription = await subscription_service.create_subscription(subscription_request)
        
        logger.info(f"Trial started successfully for user: {user_email}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "30-day free trial started successfully",
                "subscription": {
                    "status": new_subscription.status.value,
                    "plan": new_subscription.plan.value,
                    "is_trial_active": new_subscription.is_trial_active,
                    "days_remaining": new_subscription.days_remaining,
                    "trial_end_date": new_subscription.trial_end_date.isoformat()
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting trial: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start trial: {str(e)}")

@google_auth_router.get("/health")
async def health_check():
    """Health check endpoint for Google auth integration"""
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "google-auth-integration",
                "message": "Google authentication integration is running"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "service": "google-auth-integration",
                "error": str(e)
            }
        )
