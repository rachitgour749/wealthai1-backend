from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import logging

from .service import subscription_service
from .product_service import product_subscription_service
from .models import (
    SubscriptionRequest, SubscriptionResponse, SubscriptionStatusResponse,
    TrialExtensionRequest, SubscriptionUpgradeRequest,
    ProductSubscriptionRequest, ProductAccessResponse, ProductSubscriptionResponse,
    AllProductsStatusResponse, ProductCode,
    ProductTrialActivationRequest, ProductTrialActivationResponse
)

# Create router
subscription_router = APIRouter(prefix="/api/subscription", tags=["subscription"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_email_from_header(authorization: Optional[str] = Header(None)) -> str:
    """Extract user email from authorization header (for Google OAuth)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if authorization.startswith("Bearer "):
        user_email = authorization.replace("Bearer ", "")
        
        if '@' not in user_email:
            raise HTTPException(status_code=401, detail="Invalid email format in token")
        
        return user_email.lower()
    
    raise HTTPException(status_code=401, detail="Invalid authorization format")

# =============================================================================
# ESSENTIAL SUBSCRIPTION ENDPOINTS (KEEP THESE)
# =============================================================================

@subscription_router.get("/status-simple/{email}", response_model=SubscriptionStatusResponse)
async def get_subscription_status_simple(email: str):
    """Get subscription status by email (used by frontend)"""
    try:
        logger.info(f"Getting subscription status for email: {email}")
        
        status = await subscription_service.get_subscription_status(email)
        
        logger.info(f"Subscription status retrieved for email: {email}")
        return status
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get subscription status: {str(e)}")

@subscription_router.post("/create-simple", response_model=SubscriptionResponse)
async def create_subscription_simple(request: SubscriptionRequest):
    """Create a new subscription with 30-day trial (used by frontend)"""
    try:
        logger.info(f"Creating subscription for user: {request.user_email}")
        
        subscription = await subscription_service.create_subscription(request)
        
        logger.info(f"Subscription created successfully for user: {request.user_email}")
        return subscription
        
    except Exception as e:
        logger.error(f"Error creating subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create subscription: {str(e)}")

@subscription_router.post("/activate-product-trial", response_model=ProductTrialActivationResponse)
async def activate_product_trial(request: ProductTrialActivationRequest):
    """
    Activate product trial with plan_code validation
    
    Request:
    - user_email: User's email address
    - user_name: User's name
    - plan_code: Plan code from plan_master table (validated against database)
    
    Response:
    - user_email: User's email
    - startDate: Trial start date
    - endDate: Trial end date (calculated from plan_master.extension_days)
    - planCode: Plan code used
    - planName: Plan name from plan_master table
    """
    try:
        logger.info(f"Activating product trial for user: {request.user_email}, plan_code: {request.plan_code}")
        
        # Validate plan_code
        plan_validation = subscription_service.db_manager.validate_plan_code(request.plan_code)
        if not plan_validation["valid"]:
            logger.warning(f"Invalid plan_code {request.plan_code} for user: {request.user_email}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid plan code: {plan_validation['message']}"
            )
        
        # Activate trial
        response = await subscription_service.activate_product_trial(request)
        
        logger.info(f"Product trial activated successfully for user: {request.user_email}, plan: {response.planName}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error activating product trial: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to activate product trial: {str(e)}")

# =============================================================================
# PRODUCT-SPECIFIC SUBSCRIPTION ENDPOINTS (KEEP THESE)
# =============================================================================

@subscription_router.get("/product/status-simple/{email}", response_model=AllProductsStatusResponse)
async def get_all_products_status_simple(email: str):
    """Get access status for all products (used by frontend)"""
    try:
        logger.info(f"Getting all products status for user: {email}")
        
        status = product_subscription_service.get_all_products_status(email)
        
        logger.info(f"All products status retrieved for user: {email}")
        return status
        
    except Exception as e:
        logger.error(f"Error getting all products status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get all products status: {str(e)}")

@subscription_router.post("/product/enable-trial-simple/{email}/{product_code}", response_model=ProductSubscriptionResponse)
async def enable_product_trial_simple(email: str, product_code: str):
    """Enable trial for a specific product (used by frontend)"""
    try:
        logger.info(f"Enabling product trial for user: {email}, product: {product_code}")
        
        # Validate product code
        try:
            product_enum = ProductCode(product_code.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid product code: {product_code}")
        
        subscription = product_subscription_service.enable_product_trial(
            user_email=email,
            product_code=product_enum
        )
        
        logger.info(f"Product trial enabled successfully for user: {email}")
        return subscription
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling product trial: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enable product trial: {str(e)}")

@subscription_router.get("/product/access-simple/{email}/{product_code}", response_model=ProductAccessResponse)
async def check_product_access_simple(email: str, product_code: str, request: Request = None):
    """Check if user has access to a specific product (used by frontend)"""
    try:
        logger.info(f"Checking product access for user: {email}, product: {product_code}")
        
        # Validate product code
        try:
            product_enum = ProductCode(product_code.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid product code: {product_code}")
        
        access_info = product_subscription_service.check_product_access(
            user_email=email,
            product_code=product_enum,
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None
        )
        
        logger.info(f"Product access check completed for user: {email}")
        return access_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking product access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check product access: {str(e)}")

# =============================================================================
# HEALTH CHECK (KEEP THIS)
# =============================================================================

@subscription_router.get("/health")
async def health_check():
    """Health check endpoint for subscription service"""
    try:
        return {
            "status": "healthy",
            "service": "subscription",
            "message": "Subscription service is running"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "subscription",
            "error": str(e)
        }

# =============================================================================
# MIDDLEWARE FUNCTIONS (KEEP THESE)
# =============================================================================

async def check_subscription_access(user_email: str, feature: str = "premium") -> bool:
    """Middleware function to check if user has subscription access"""
    try:
        access_info = await subscription_service.check_access_permission(user_email, feature)
        return access_info["has_access"]
    except Exception as e:
        logger.error(f"Error checking subscription access: {str(e)}")
        return False

async def require_subscription_access(
    feature: str = "premium",
    user_email: str = Depends(get_user_email_from_header)
):
    """Dependency to require subscription access for protected routes"""
    has_access = await check_subscription_access(user_email, feature)
    
    if not has_access:
        raise HTTPException(
            status_code=403, 
            detail=f"Subscription required for {feature} features. Please upgrade your plan or start a trial."
        )
    
    return user_email
