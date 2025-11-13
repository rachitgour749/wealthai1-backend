#!/usr/bin/env python3
"""
Product Access Control Middleware
Middleware for controlling access to different products based on subscription status
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Callable, Dict, Any
import logging

from .product_service import product_subscription_service
from .models import ProductCode

logger = logging.getLogger(__name__)

class ProductAccessMiddleware:
    """Middleware for product access control"""
    
    def __init__(self):
        self.service = product_subscription_service
    
    def require_product_access(self, product_code: ProductCode):
        """Decorator to require access to a specific product"""
        def decorator(func: Callable) -> Callable:
            async def wrapper(request: Request, *args, **kwargs):
                # Extract user email from request (assuming it's in headers or JWT)
                user_email = self._extract_user_email(request)
                
                if not user_email:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User email not found in request"
                    )
                
                # Check product access
                access_info = self.service.check_product_access(
                    user_email=user_email,
                    product_code=product_code,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent")
                )
                
                if not access_info.has_access:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "error": "Product access denied",
                            "product": product_code.value,
                            "message": f"Access to {product_code.value} is not available. Please subscribe to access this product.",
                            "access_type": access_info.access_type,
                            "days_remaining": access_info.days_remaining
                        }
                    )
                
                # Add access info to request state for use in the endpoint
                request.state.product_access = access_info
                
                return await func(request, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def require_trial_or_paid_access(self, product_code: ProductCode):
        """Decorator to require trial or paid access to a specific product"""
        def decorator(func: Callable) -> Callable:
            async def wrapper(request: Request, *args, **kwargs):
                user_email = self._extract_user_email(request)
                
                if not user_email:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User email not found in request"
                    )
                
                # Check product access
                access_info = self.service.check_product_access(
                    user_email=user_email,
                    product_code=product_code,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent")
                )
                
                if not access_info.has_access:
                    # If no access, try to enable trial
                    try:
                        trial_response = self.service.enable_product_trial(
                            user_email=user_email,
                            product_code=product_code
                        )
                        
                        # Re-check access after enabling trial
                        access_info = self.service.check_product_access(
                            user_email=user_email,
                            product_code=product_code
                        )
                        
                        if not access_info.has_access:
                            return JSONResponse(
                                status_code=status.HTTP_403_FORBIDDEN,
                                content={
                                    "error": "Product access denied",
                                    "product": product_code.value,
                                    "message": f"Unable to enable trial for {product_code.value}. Please contact support.",
                                    "access_type": access_info.access_type
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error enabling trial for {user_email}, {product_code}: {e}")
                        return JSONResponse(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={
                                "error": "Internal server error",
                                "message": "Unable to process product access request"
                            }
                        )
                
                # Add access info to request state
                request.state.product_access = access_info
                
                return await func(request, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def _extract_user_email(self, request: Request) -> str:
        """Extract user email from request headers or JWT token"""
        # Try to get from Authorization header (JWT token)
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you would decode the JWT token here
            # For now, we'll assume the email is in a custom header
            pass
        
        # Try to get from custom header
        user_email = request.headers.get("x-user-email")
        if user_email:
            return user_email
        
        # Try to get from query parameters (for testing)
        user_email = request.query_params.get("user_email")
        if user_email:
            return user_email
        
        return None

# Global middleware instance
product_access_middleware = ProductAccessMiddleware()

# Convenience decorators for each product
require_marketai_access = product_access_middleware.require_product_access(ProductCode.MARKETAI)
require_chatai_access = product_access_middleware.require_product_access(ProductCode.CHATAI)
require_tradai_access = product_access_middleware.require_product_access(ProductCode.TRADAI)

require_marketai_trial_or_paid = product_access_middleware.require_trial_or_paid_access(ProductCode.MARKETAI)
require_chatai_trial_or_paid = product_access_middleware.require_trial_or_paid_access(ProductCode.CHATAI)
require_tradai_trial_or_paid = product_access_middleware.require_trial_or_paid_access(ProductCode.TRADAI)
