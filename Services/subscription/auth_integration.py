"""
Google Authentication Integration for Subscription System
Handles automatic trial creation when users login via Google OAuth
"""

from fastapi import HTTPException, Depends, Header
from typing import Optional, Dict, Any
import jwt
import logging
from datetime import datetime, timedelta

from .service import subscription_service
from .models import SubscriptionRequest, SubscriptionPlan, SubscriptionStatus

logger = logging.getLogger(__name__)

class GoogleAuthIntegration:
    """Integration class for Google OAuth with subscription system"""
    
    def __init__(self):
        self.service = subscription_service
    
    def extract_user_from_google_token(self, authorization: str) -> Dict[str, str]:
        """
        Extract user information from Google OAuth token
        In production, you should verify the token with Google's API
        """
        try:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")
            
            token = authorization.replace("Bearer ", "")
            
            # For demo purposes, we'll assume the token contains user info
            # In production, you should:
            # 1. Verify the token with Google's OAuth2 API
            # 2. Extract user information from the verified token
            
            # Mock user extraction - replace with actual Google token verification
            # This is a simplified approach for demonstration
            user_info = {
                "email": token,  # Assuming token is email for demo
                "name": "Google User",  # Extract from actual token
                "google_id": "google_123"  # Extract from actual token
            }
            
            return user_info
            
        except Exception as e:
            logger.error(f"Error extracting user from Google token: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    async def handle_google_login(self, authorization: str) -> Dict[str, Any]:
        """
        Handle Google login and automatically create/check subscription
        This should be called when a user successfully logs in via Google OAuth
        """
        try:
            # Extract user information from Google token
            user_info = self.extract_user_from_google_token(authorization)
            user_email = user_info["email"]
            user_name = user_info["name"]
            
            logger.info(f"Handling Google login for user: {user_email}")
            
            # Check if user already has a subscription
            existing_status = await self.service.get_subscription_status(user_email)
            
            if existing_status.status == SubscriptionStatus.TRIAL and existing_status.is_trial_active:
                # User already has an active trial
                logger.info(f"User {user_email} already has an active trial")
                return {
                    "user_email": user_email,
                    "user_name": user_name,
                    "subscription_status": existing_status,
                    "trial_created": False,
                    "message": "Welcome back! Your trial is still active."
                }
            
            elif existing_status.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.EXPIRED]:
                # User has a subscription (active or expired)
                logger.info(f"User {user_email} has existing subscription: {existing_status.status}")
                return {
                    "user_email": user_email,
                    "user_name": user_name,
                    "subscription_status": existing_status,
                    "trial_created": False,
                    "message": f"Welcome back! Your subscription status: {existing_status.status}"
                }
            
            else:
                # Create new subscription for first-time user (not trial)
                logger.info(f"Creating subscription for new user (first-time sign-in): {user_email}")
                
                subscription_request = SubscriptionRequest(
                    user_email=user_email,
                    user_name=user_name,
                    plan=SubscriptionPlan.FREE
                )
                
                new_subscription = await self.service.create_subscription(subscription_request)
                
                return {
                    "user_email": user_email,
                    "user_name": user_name,
                    "subscription_status": new_subscription,
                    "trial_created": False,  # No trial created on first sign-in
                    "message": "Welcome! Your subscription is now active."
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling Google login: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to handle Google login: {str(e)}")
    
    async def get_user_subscription_info(self, authorization: str) -> Dict[str, Any]:
        """Get user's subscription information from Google token"""
        try:
            user_info = self.extract_user_from_google_token(authorization)
            user_email = user_info["email"]
            
            subscription_status = await self.service.get_subscription_status(user_email)
            
            return {
                "user_email": user_email,
                "user_name": user_info["name"],
                "subscription_status": subscription_status,
                "can_access_premium": subscription_status.can_access_premium,
                "is_trial_active": subscription_status.is_trial_active,
                "days_remaining": subscription_status.days_remaining
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting user subscription info: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get subscription info: {str(e)}")

# Global instance
google_auth_integration = GoogleAuthIntegration()

# Dependency functions for FastAPI
async def get_current_user_from_google_token(
    authorization: Optional[str] = Header(None)
) -> Dict[str, str]:
    """FastAPI dependency to get current user from Google token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    return google_auth_integration.extract_user_from_google_token(authorization)

async def get_user_with_subscription(
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """FastAPI dependency to get user with subscription info"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    return await google_auth_integration.get_user_subscription_info(authorization)

async def require_active_subscription(
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """FastAPI dependency to require active subscription (trial or paid)"""
    user_info = await get_user_with_subscription(authorization)
    
    if not user_info["can_access_premium"]:
        raise HTTPException(
            status_code=403,
            detail="Active subscription required. Please start a trial or upgrade your plan."
        )
    
    return user_info

async def require_trial_or_subscription(
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """FastAPI dependency to require trial or active subscription"""
    user_info = await get_user_with_subscription(authorization)
    
    if not (user_info["is_trial_active"] or user_info["can_access_premium"]):
        raise HTTPException(
            status_code=403,
            detail="Trial or subscription required. Please start a trial or upgrade your plan."
        )
    
    return user_info

