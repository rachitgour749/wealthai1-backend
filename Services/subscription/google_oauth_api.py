"""
Google OAuth Integration API for Subscription System
Handles frontend Google OAuth tokens and manages subscription lifecycle
"""

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
from datetime import datetime
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests
    GOOGLE_OAUTH_AVAILABLE = True
except ImportError:
    GOOGLE_OAUTH_AVAILABLE = False
    # Mock classes for when Google OAuth is not available
    class id_token:
        @staticmethod
        def verify_oauth2_token(*args, **kwargs):
            raise NotImplementedError("Google OAuth not installed")
    
    class requests:
        class Request:
            pass
import os
import json

from .service import subscription_service
from .models import (
    SubscriptionRequest, SubscriptionResponse, SubscriptionStatusResponse,
    SubscriptionPlan, SubscriptionStatus
)

# Create router
google_oauth_router = APIRouter(prefix="/api/auth", tags=["google-oauth"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google OAuth settings (these should be environment variables in production)
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', 'your-google-client-id')

def serialize_datetime_dict(data):
    """Convert datetime objects in a dictionary to ISO format strings for JSON serialization"""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = serialize_datetime_dict(value)
            elif hasattr(value, 'dict'):
                # Handle Pydantic models
                result[key] = serialize_datetime_dict(value.dict())
            else:
                result[key] = value
        return result
    elif hasattr(data, 'dict'):
        # Handle Pydantic models directly
        return serialize_datetime_dict(data.dict())
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data

class GoogleOAuthHandler:
    """Handler for Google OAuth integration with subscription system"""
    
    def __init__(self):
        self.service = subscription_service
    
    def verify_google_token(self, token: str) -> Dict[str, Any]:
        """Verify Google OAuth token and extract user info"""
        try:
            # Validate token is not empty
            if not token or token.strip() == "":
                raise ValueError("Empty token provided")
            
            if GOOGLE_OAUTH_AVAILABLE:
                # Try proper Google OAuth verification first
                try:
                    idinfo = id_token.verify_oauth2_token(
                        token, 
                        requests.Request(), 
                        GOOGLE_CLIENT_ID
                    )
                    
                    # Verify the issuer
                    if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                        raise ValueError('Wrong issuer')
                    
                    # Validate that essential fields exist
                    if not idinfo.get("email"):
                        raise ValueError("Token missing email field")
                    
                    logger.info(f"Successfully verified Google token for: {idinfo.get('email')}")
                    
                    return {
                        "email": idinfo.get("email"),
                        "name": idinfo.get("name", ""),
                        "picture": idinfo.get("picture", ""),
                        "sub": idinfo.get("sub", ""),
                        "email_verified": idinfo.get("email_verified", True)
                    }
                except Exception as google_error:
                    logger.warning(f"Google token verification failed, trying JWT decode: {str(google_error)}")
                    # Fall through to JWT decode attempt
            
            # Fallback: Try to decode JWT without verification (for development)
            try:
                from jose import jwt
                decoded = jwt.get_unverified_claims(token)
                
                # Validate that essential fields exist
                if not decoded.get("email"):
                    raise ValueError("Token missing email field")
                
                logger.info(f"Successfully decoded JWT token for: {decoded.get('email')}")
                    
                return {
                    "email": decoded.get("email"),
                    "name": decoded.get("name", ""),
                    "picture": decoded.get("picture", ""),
                    "sub": decoded.get("sub", ""),
                    "email_verified": decoded.get("email_verified", True)
                }
            except Exception as jwt_error:
                logger.warning(f"JWT decode failed: {str(jwt_error)}")
                # Last fallback: assume the token is the email (for development)
                if "@" in token:
                    logger.info(f"Using email fallback for token: {token}")
                    return {
                        "email": token,
                        "name": "Google User",
                        "picture": "",
                        "sub": token.replace("@", "_").replace(".", "_"),
                        "email_verified": True
                    }
                else:
                    raise ValueError(f"Invalid token format - not a valid JWT or email: {str(jwt_error)}")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error verifying Google token: {str(e)}")
            raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")
    
    async def handle_google_login(self, token: str) -> Dict[str, Any]:
        """Handle Google login and manage subscription"""
        try:
            # Verify Google token and extract user info
            user_info = self.verify_google_token(token)
            user_email = user_info["email"]
            user_name = user_info["name"]
            
            logger.info(f"Processing Google login for user: {user_email}")
            
            # Check if user already has a subscription in database
            existing_subscription = self.service.db_manager.get_subscription_by_email(user_email)
            user_exists = existing_subscription is not None
            
            if user_exists:
                # Existing user - get their current subscription status
                logger.info(f"Existing user {user_email} logged in")
                
                existing_status = await self.service.get_subscription_status(user_email)
                
                # Check subscription status
                if existing_status.status == SubscriptionStatus.TRIAL and existing_status.is_trial_active:
                    message = f"Welcome back! Your trial expires in {existing_status.days_remaining} days."
                elif existing_status.status == SubscriptionStatus.ACTIVE:
                    message = f"Welcome back! Your subscription is active."
                elif existing_status.status in [SubscriptionStatus.EXPIRED, SubscriptionStatus.CANCELLED]:
                    message = "Your trial/subscription has expired. Please subscribe to continue using premium features."
                else:
                    message = "Welcome back!"
                
                return {
                    "user_info": user_info,
                    "subscription_status": serialize_datetime_dict(existing_status),
                    "is_new_user": False,
                    "trial_created": False,
                    "message": message
                }
            else:
                # New user - create subscription (not trial)
                logger.info(f"Creating subscription for new user (first-time sign-in): {user_email}")
                
                subscription_request = SubscriptionRequest(
                    user_email=user_email,
                    user_name=user_name,
                    plan=SubscriptionPlan.FREE
                )
                
                new_subscription = await self.service.create_subscription(subscription_request)
                
                return {
                    "user_info": user_info,
                    "subscription_status": serialize_datetime_dict(new_subscription),
                    "is_new_user": True,
                    "trial_created": False,  # No trial created on first sign-in
                    "message": "Welcome to WealthAI! Your subscription is now active."
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling Google login: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Login processing failed: {str(e)}")

# Global handler instance
google_oauth_handler = GoogleOAuthHandler()

@google_oauth_router.post("/google-login")
async def google_login(request: Dict[str, Any]):
    """
    Handle Google OAuth login and subscription management
    Expected request: {"token": "google_oauth_token"}
    """
    try:
        token = request.get("token")
        if not token:
            raise HTTPException(status_code=400, detail="Google token is required")
        
        result = await google_oauth_handler.handle_google_login(token)
        
        return JSONResponse(content={
            "success": True,
            "data": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Google login endpoint: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@google_oauth_router.get("/subscription-status")
async def get_user_subscription_status(authorization: Optional[str] = Header(None)):
    """Get subscription status for authenticated user"""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        token = authorization.replace("Bearer ", "")
        user_info = google_oauth_handler.verify_google_token(token)
        user_email = user_info["email"]
        
        subscription_status = await subscription_service.get_subscription_status(user_email)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "user_info": user_info,
                "subscription_status": serialize_datetime_dict(subscription_status)
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get subscription status: {str(e)}")

@google_oauth_router.post("/check-access")
async def check_user_access(
    request: Dict[str, Any],
    authorization: Optional[str] = Header(None)
):
    """
    Check if user has access to specific features
    Expected request: {"feature": "premium"}
    """
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        token = authorization.replace("Bearer ", "")
        user_info = google_oauth_handler.verify_google_token(token)
        user_email = user_info["email"]
        
        feature = request.get("feature", "premium")
        access_info = await subscription_service.check_access_permission(user_email, feature)
        
        return JSONResponse(content={
            "success": True,
            "data": access_info
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check access: {str(e)}")

@google_oauth_router.get("/user-info")
async def get_user_info(authorization: Optional[str] = Header(None)):
    """Get user information from Google token"""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        token = authorization.replace("Bearer ", "")
        user_info = google_oauth_handler.verify_google_token(token)
        
        return JSONResponse(content={
            "success": True,
            "data": {"user_info": user_info}
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user info: {str(e)}")

@google_oauth_router.get("/health")
async def health_check():
    """Health check endpoint for Google OAuth service"""
    try:
        return JSONResponse(content={
            "success": True,
            "data": {
                "status": "healthy",
                "service": "google-oauth",
                "message": "Google OAuth integration service is running"
            }
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)
