"""
Google OAuth Integration API for Subscription System
Handles frontend Google OAuth tokens and manages subscription lifecycle
"""

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
import hashlib # Added for token hashing
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

from ..services.subscription_service import subscription_service
from ..database import subscription_manager
from ..subscription_schemas import (
    SubscriptionRequest,
    SubscriptionPlan,
    SubscriptionStatus,
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
    
    async def handle_google_login(self, token: str, phone_no: str = None) -> Dict[str, Any]:
        """Handle Google login and manage subscription"""
        try:
            # Verify Google token and extract user info
            user_info = self.verify_google_token(token)
            user_email = user_info["email"]
            user_name = user_info["name"]
            
            logger.info(f"Processing Google login for user: {user_email}")
            
            # STORE RAW TOKEN: As requested by user
            try:
                # token_hash = hashlib.sha256(token.encode()).hexdigest() # OLD: Hashed
                subscription_manager.update_user_token_hash(user_email, token) # NEW: Raw Token
                logger.info(f"Updated token for {user_email}")
            except Exception as e:
                logger.error(f"Failed to update token for {user_email}: {e}")
                # Proceeding anyway, but auth might fail later depending on middleware strictness
            
            # Check if user already has a subscription in database
            # We use get_user_details to check for existence in user_details table
            existing_user = await self.service.get_user_details(user_email)
            
            if existing_user:
                # Existing user - return details
                logger.info(f"Existing user {user_email} logged in")
                
                return {
                    "user_email": existing_user.user_email,
                    "user_name": existing_user.user_name,
                    "created_at": existing_user.created_at.isoformat() if existing_user.created_at else None,
                    "updated_at": existing_user.updated_at.isoformat() if existing_user.updated_at else None,
                    "status": existing_user.status,
                    "phone_no": existing_user.phone_no,
                    "is_new_user": False,
                    "token": token, # Return token to frontend
                    "message": "Welcome back!"
                }
            else:
                # New user - create subscription (trial)
                logger.info(f"Creating new user: {user_email}")
                
                subscription_request = SubscriptionRequest(
                    user_email=user_email,
                    user_name=user_name,
                    plan=SubscriptionPlan.FREE
                )
                
                # create_subscription now handles creating UserDetails and setting status/phone_no
                # We pass phone_no to it
                new_user = await self.service.create_subscription(subscription_request, phone_no=phone_no)
                
                return {
                    "user_email": new_user.user_email,
                    "user_name": new_user.user_name,
                    "created_at": new_user.created_at.isoformat() if new_user.created_at else None,
                    "updated_at": new_user.updated_at.isoformat() if new_user.updated_at else None,
                    "status": new_user.status,
                    "phone_no": new_user.phone_no,
                    "is_new_user": True,
                    "token": token, # Return token to frontend
                    "message": "Welcome to WealthAI! Your account has been created."
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
    Expected request: {"token": "google_oauth_token", "phone_no": "optional_phone_number"}
    """
    try:
        token = request.get("token")
        phone_no = request.get("phone_no")
        
        if not token:
            raise HTTPException(status_code=400, detail="Google token is required")
        
        result = await google_oauth_handler.handle_google_login(token, phone_no)
        
        return JSONResponse(content={
            "success": True,
            "data": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Google login endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@google_oauth_router.get("/user-info")
async def get_user_info(authorization: Optional[str] = Header(None)):
    """Get user information from Google token"""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        token = authorization.replace("Bearer ", "")
        user_info = google_oauth_handler.verify_google_token(token)
        user_email = user_info["email"]
        
        # Get details from database
        user_details = await subscription_service.get_user_details(user_email)
        
        if not user_details:
             raise HTTPException(status_code=404, detail="User not found")

        return JSONResponse(content={
            "success": True,
            "data": {
                "user_email": user_details.user_email,
                "user_name": user_details.user_name,
                "created_at": user_details.created_at.isoformat() if user_details.created_at else None,
                "updated_at": user_details.updated_at.isoformat() if user_details.updated_at else None,
                "status": user_details.status,
                "phone_no": user_details.phone_no
            }
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

