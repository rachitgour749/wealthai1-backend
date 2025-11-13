#!/usr/bin/env python3
"""
Product Subscription Service
Handles product-specific subscription operations
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import logging

from .database import product_subscription_manager
from .models import (
    ProductCode, SubscriptionType, SubscriptionStatus,
    ProductSubscriptionRequest, ProductAccessResponse, 
    ProductSubscriptionResponse, BundleSubscriptionRequest,
    ProductUpgradeRequest, AllProductsStatusResponse
)

logger = logging.getLogger(__name__)

class ProductSubscriptionService:
    """Service for managing product-specific subscriptions"""
    
    def __init__(self):
        self.db_manager = product_subscription_manager
    
    def create_product_trial(self, user_email: str, product_code: ProductCode, 
                           trial_days: int = 7) -> ProductSubscriptionResponse:
        """Create a trial subscription for a specific product"""
        try:
            subscription = self.db_manager.create_product_subscription(
                user_email=user_email,
                product_code=product_code,
                subscription_type=SubscriptionType.TRIAL,
                plan_code="FREE",
                trial_days=trial_days
            )
            
            return self._convert_to_response(subscription)
            
        except Exception as e:
            logger.error(f"Error creating product trial for {user_email}, {product_code}: {e}")
            raise e
    
    def create_bundle_subscription(self, user_email: str, product_codes: List[ProductCode], 
                                 plan_code: str = "BUNDLE") -> List[ProductSubscriptionResponse]:
        """Create a bundle subscription for multiple products"""
        try:
            subscriptions = self.db_manager.create_bundle_subscription(
                user_email=user_email,
                product_codes=product_codes,
                plan_code=plan_code
            )
            
            return [self._convert_to_response(sub) for sub in subscriptions]
            
        except Exception as e:
            logger.error(f"Error creating bundle subscription for {user_email}: {e}")
            raise e
    
    def check_product_access(self, user_email: str, product_code: ProductCode, 
                           ip_address: str = None, user_agent: str = None) -> ProductAccessResponse:
        """Check if user has access to a specific product"""
        try:
            access_info = self.db_manager.check_product_access(user_email, product_code)
            
            # Log the access attempt
            self.db_manager.log_product_access(
                user_email=user_email,
                product_code=product_code,
                access_granted=access_info["has_access"],
                access_type=access_info["access_type"],
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Handle the case where status might be "no_subscription"
            status_value = access_info["status"]
            if status_value == "no_subscription":
                status_enum = SubscriptionStatus.EXPIRED  # Use EXPIRED as default for no subscription
            else:
                status_enum = SubscriptionStatus(status_value)
            
            return ProductAccessResponse(
                user_email=user_email,
                product_code=product_code,
                has_access=access_info["has_access"],
                access_type=access_info["access_type"],
                days_remaining=access_info["days_remaining"],
                trial_end_date=access_info.get("trial_end_date"),
                paid_end_date=access_info.get("paid_end_date"),
                status=status_enum,
                plan=access_info.get("plan")
            )
            
        except Exception as e:
            logger.error(f"Error checking product access for {user_email}, {product_code}: {e}")
            raise e
    
    def get_all_products_status(self, user_email: str) -> AllProductsStatusResponse:
        """Get access status for all products for a user"""
        try:
            products_info = self.db_manager.get_all_user_products(user_email)
            
            products = {}
            total_active = 0
            bundle_products = []
            
            for product_code in ProductCode:
                access_info = products_info[product_code.value]
                
                # Handle the case where status might be "no_subscription"
                status_value = access_info["status"]
                if status_value == "no_subscription":
                    status_enum = SubscriptionStatus.EXPIRED  # Use EXPIRED as default for no subscription
                else:
                    status_enum = SubscriptionStatus(status_value)
                
                products[product_code.value] = ProductAccessResponse(
                    user_email=user_email,
                    product_code=product_code,
                    has_access=access_info["has_access"],
                    access_type=access_info["access_type"],
                    days_remaining=access_info["days_remaining"],
                    trial_end_date=access_info.get("trial_end_date"),
                    paid_end_date=access_info.get("paid_end_date"),
                    status=status_enum,
                    plan=access_info.get("plan")
                )
                
                if access_info["has_access"]:
                    total_active += 1
                    if access_info.get("is_bundle", False):
                        bundle_products.append(product_code)
            
            return AllProductsStatusResponse(
                user_email=user_email,
                products=products,
                total_active_products=total_active,
                has_bundle=len(bundle_products) > 0,
                bundle_products=bundle_products
            )
            
        except Exception as e:
            logger.error(f"Error getting all products status for {user_email}: {e}")
            raise e
    
    def upgrade_product_subscription(self, user_email: str, product_code: ProductCode, 
                                   plan_code: str) -> bool:
        """Upgrade a product subscription to paid plan"""
        try:
            return self.db_manager.upgrade_product_subscription(
                user_email=user_email,
                product_code=product_code,
                plan_code=plan_code
            )
            
        except Exception as e:
            logger.error(f"Error upgrading product subscription for {user_email}, {product_code}: {e}")
            raise e
    
    def enable_product_trial(self, user_email: str, product_code: ProductCode) -> ProductSubscriptionResponse:
        """Enable trial for a specific product (used when user first accesses a product)"""
        try:
            # Check if user already has access
            access_info = self.db_manager.check_product_access(user_email, product_code)
            
            if access_info["has_access"]:
                # User already has access, return existing subscription
                subscription = self.db_manager.get_session().query(
                    self.db_manager.ProductSubscription
                ).filter(
                    self.db_manager.ProductSubscription.user_email == user_email.lower(),
                    self.db_manager.ProductSubscription.product_code == product_code
                ).first()
                
                if subscription:
                    return self._convert_to_response(subscription)
            
            # Create new trial subscription
            return self.create_product_trial(user_email, product_code)
            
        except Exception as e:
            logger.error(f"Error enabling product trial for {user_email}, {product_code}: {e}")
            raise e
    
    def _convert_to_response(self, subscription) -> ProductSubscriptionResponse:
        """Convert database subscription to response model"""
        now = datetime.now(timezone.utc)
        
        # Calculate if trial is active
        is_trial_active = (
            subscription.status == SubscriptionStatus.TRIAL and 
            subscription.trial_end_date and
            subscription.trial_end_date > now
        )
        
        # Calculate if paid subscription is active
        is_paid_active = (
            subscription.status == SubscriptionStatus.ACTIVE and
            subscription.paid_end_date and
            subscription.paid_end_date > now
        )
        
        # Calculate days remaining
        if is_trial_active and subscription.trial_end_date:
            days_remaining = (subscription.trial_end_date - now).days
        elif is_paid_active and subscription.paid_end_date:
            days_remaining = (subscription.paid_end_date - now).days
        else:
            days_remaining = 0
        
        return ProductSubscriptionResponse(
            id=subscription.id,
            user_email=subscription.user_email,
            product_code=subscription.product_code,
            subscription_type=subscription.subscription_type,
            status=subscription.status,
            plan_code=subscription.plan_code,
            trial_start_date=subscription.trial_start_date,
            trial_end_date=subscription.trial_end_date,
            paid_start_date=subscription.paid_start_date,
            paid_end_date=subscription.paid_end_date,
            created_at=subscription.created_at,
            updated_at=subscription.updated_at,
            is_trial_active=is_trial_active,
            is_paid_active=is_paid_active,
            days_remaining=days_remaining,
            is_bundle_subscription=subscription.is_bundle_subscription
        )

# Global service instance
product_subscription_service = ProductSubscriptionService()
