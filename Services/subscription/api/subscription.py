# Services/Subscription/api/subscription.py
"""Subscription API endpoints"""
from fastapi import APIRouter, HTTPException
from typing import List
import logging

from ..services.subscription_service import subscription_service
from ..subscription_schemas import (
    ProductTrialActivationRequest,
    ProductTrialActivationResponse,
    ProductManagerRecord,
    UserDetailsResponse,
)

# Create router
subscription_router = APIRouter(prefix="/api/subscription", tags=["subscription"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SIMPLIFIED API - ONLY 3 ENDPOINTS
# =============================================================================

@subscription_router.get("/user/{email}", response_model=UserDetailsResponse)
async def get_user_details(email: str):
    """Get user details by email"""
    try:
        logger.info(f"Getting user details for: {email}")
        
        user = await subscription_service.get_user_details(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user details: {str(e)}")

@subscription_router.post("/activate-trial", response_model=ProductTrialActivationResponse)
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
        
        logger.info(
            f"Product trial activated successfully for user: {request.user_email}, "
            f"plan: {response.planName}"
        )
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
# PRODUCT LISTING
# =============================================================================

@subscription_router.get("/products/{email}", response_model=List[ProductManagerRecord])
async def list_user_products(email: str):
    try:
        products = await subscription_service.get_user_products(email)
        return products
    except Exception as e:
        logger.error(f"Error fetching products for {email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {str(e)}")

# =============================================================================
# HEALTH CHECK
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
