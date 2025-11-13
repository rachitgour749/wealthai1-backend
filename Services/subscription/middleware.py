"""
Subscription Middleware for Route Protection
Protects routes based on subscription status and trial period
"""

from fastapi import HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Callable
import logging
from datetime import datetime

from .auth_integration import google_auth_integration
from .service import subscription_service

logger = logging.getLogger(__name__)

class SubscriptionMiddleware:
    """Middleware class for subscription-based route protection"""
    
    def __init__(self):
        self.service = subscription_service
    
    async def check_subscription_access(
        self, 
        user_email: str, 
        feature: str = "premium",
        require_trial: bool = False
    ) -> Dict[str, Any]:
        """
        Check if user has access to a specific feature
        
        Args:
            user_email: User's email address
            feature: Feature to check access for
            require_trial: Whether to require an active trial specifically
        
        Returns:
            Dict containing access information
        """
        try:
            # Get user's subscription status
            subscription_status = await self.service.get_subscription_status(user_email)
            
            # Check access based on feature requirements
            if feature == "premium":
                has_access = subscription_status.can_access_premium
            elif feature == "trial":
                has_access = subscription_status.is_trial_active
            elif feature == "basic":
                has_access = True  # Basic features are always available
            else:
                has_access = subscription_status.can_access_premium
            
            # Additional check for trial requirement
            if require_trial and not subscription_status.is_trial_active:
                has_access = False
            
            return {
                "has_access": has_access,
                "user_email": user_email,
                "feature": feature,
                "subscription_status": subscription_status.status.value,
                "plan": subscription_status.plan.value,
                "is_trial_active": subscription_status.is_trial_active,
                "can_access_premium": subscription_status.can_access_premium,
                "days_remaining": subscription_status.days_remaining,
                "trial_end_date": subscription_status.trial_end_date,
                "subscription_end_date": subscription_status.subscription_end_date
            }
            
        except Exception as e:
            logger.error(f"Error checking subscription access: {str(e)}")
            return {
                "has_access": False,
                "error": str(e),
                "user_email": user_email,
                "feature": feature
            }
    
    def create_access_denied_response(self, access_info: Dict[str, Any]) -> JSONResponse:
        """Create a standardized access denied response"""
        
        if access_info.get("error"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Subscription check failed",
                    "message": "Unable to verify subscription status",
                    "details": access_info["error"]
                }
            )
        
        # Determine the appropriate message based on subscription status
        if not access_info["is_trial_active"] and not access_info["can_access_premium"]:
            message = "Your trial has expired. Please upgrade to continue using premium features."
        elif access_info["subscription_status"] == "trial" and access_info["days_remaining"] <= 0:
            message = "Your trial has expired. Please upgrade to continue using premium features."
        elif access_info["subscription_status"] == "expired":
            message = "Your subscription has expired. Please renew to continue using premium features."
        else:
            message = "Premium subscription required for this feature."
        
        return JSONResponse(
            status_code=403,
            content={
                "error": "Access denied",
                "message": message,
                "subscription_status": access_info["subscription_status"],
                "plan": access_info["plan"],
                "is_trial_active": access_info["is_trial_active"],
                "days_remaining": access_info["days_remaining"],
                "trial_end_date": access_info["trial_end_date"].isoformat() if access_info["trial_end_date"] else None,
                "upgrade_required": True
            }
        )

# Global middleware instance
subscription_middleware = SubscriptionMiddleware()

# FastAPI Dependency Functions
async def require_premium_access(
    authorization: str = Depends(lambda x: x.headers.get("Authorization"))
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires premium access (trial or paid subscription)
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Extract user from Google token
        user_info = google_auth_integration.extract_user_from_google_token(authorization)
        user_email = user_info["email"]
        
        # Check subscription access
        access_info = await subscription_middleware.check_subscription_access(
            user_email, 
            feature="premium"
        )
        
        if not access_info["has_access"]:
            raise HTTPException(
                status_code=403,
                detail=subscription_middleware.create_access_denied_response(access_info).body.decode()
            )
        
        return access_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in require_premium_access: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify subscription access")

async def require_trial_access(
    authorization: str = Depends(lambda x: x.headers.get("Authorization"))
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires active trial access
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Extract user from Google token
        user_info = google_auth_integration.extract_user_from_google_token(authorization)
        user_email = user_info["email"]
        
        # Check trial access
        access_info = await subscription_middleware.check_subscription_access(
            user_email, 
            feature="trial",
            require_trial=True
        )
        
        if not access_info["has_access"]:
            raise HTTPException(
                status_code=403,
                detail="Active trial required for this feature"
            )
        
        return access_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in require_trial_access: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify trial access")

async def require_basic_access(
    authorization: str = Depends(lambda x: x.headers.get("Authorization"))
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires basic access (always available)
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Extract user from Google token
        user_info = google_auth_integration.extract_user_from_google_token(authorization)
        user_email = user_info["email"]
        
        # Check basic access (always available)
        access_info = await subscription_middleware.check_subscription_access(
            user_email, 
            feature="basic"
        )
        
        return access_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in require_basic_access: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify basic access")

# Utility function for manual access checking
async def check_user_access(
    user_email: str, 
    feature: str = "premium"
) -> Dict[str, Any]:
    """
    Utility function to check user access without FastAPI dependencies
    """
    return await subscription_middleware.check_subscription_access(user_email, feature)

# Route protection decorator
def protect_route(feature: str = "premium", require_trial: bool = False):
    """
    Decorator to protect routes based on subscription status
    
    Usage:
    @protect_route("premium")
    async def premium_endpoint():
        pass
    
    @protect_route("trial", require_trial=True)
    async def trial_only_endpoint():
        pass
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # This would need to be integrated with your specific route handling
            # For now, it's a placeholder for the decorator pattern
            return await func(*args, **kwargs)
        return wrapper
    return decorator

