from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import uuid

from .database import subscription_manager
from .models import (
    SubscriptionRequest, SubscriptionResponse, SubscriptionStatusResponse,
    TrialExtensionRequest, SubscriptionUpgradeRequest, SubscriptionAnalytics,
    SubscriptionStatus, SubscriptionPlan,
    ProductTrialActivationRequest, ProductTrialActivationResponse
)

class SubscriptionService:
    """Service class for subscription management"""
    
    def __init__(self):
        self.db_manager = subscription_manager
    
    async def create_subscription(self, request: SubscriptionRequest) -> SubscriptionResponse:
        """Create a new subscription for first-time sign-in (is_trial=False)"""
        try:
            # Check if user already has a subscription
            existing = self.db_manager.get_subscription_by_email(request.user_email)
            
            if existing:
                # Return existing subscription
                return self._subscription_to_response(existing)
            
            # Create new subscription
            subscription = self.db_manager.create_subscription(
                user_email=request.user_email,
                user_name=request.user_name,
                plan=request.plan
            )
            
            return self._subscription_to_response(subscription)
            
        except Exception as e:
            raise Exception(f"Failed to create subscription: {str(e)}")
    
    async def get_subscription_status(self, user_email: str) -> SubscriptionStatusResponse:
        """Get subscription status for a user"""
        try:
            status_info = self.db_manager.check_trial_status(user_email)
            
            if not status_info["has_subscription"]:
                # User has no subscription in database
                return SubscriptionStatusResponse(
                    user_email=user_email,
                    has_subscription=False,
                    status=SubscriptionStatus.TRIAL,  # Default status when no subscription
                    plan=SubscriptionPlan.FREE,  # Default plan when no subscription
                    plan_code=None,  # No plan code when no subscription
                    is_trial_active=False,
                    trial_end_date=None,
                    subscription_end_date=None,
                    days_remaining=0,
                    can_access_premium=False
                )
            
            # User has subscription in database
            subscription = self.db_manager.get_subscription_by_email(user_email)
            
            # Convert plan_code to string (handle None and numeric values)
            plan_code_str = None
            if subscription.plan_code is not None:
                plan_code_str = str(int(subscription.plan_code)) if subscription.plan_code else None
            
            return SubscriptionStatusResponse(
                user_email=user_email,
                has_subscription=True,
                status=SubscriptionStatus(status_info["status"]),
                plan=SubscriptionPlan(status_info["plan"]),
                plan_code=plan_code_str,  # Include plan_code as string
                is_trial_active=status_info["is_trial_active"],
                trial_end_date=status_info.get("trial_end_date"),
                subscription_end_date=status_info.get("subscription_end_date"),
                days_remaining=status_info["days_remaining"],
                can_access_premium=status_info["can_access_premium"]
            )
            
        except Exception as e:
            raise Exception(f"Failed to get subscription status: {str(e)}")
    
    async def extend_trial(self, request: TrialExtensionRequest) -> Dict[str, Any]:
        """Extend trial period for a user"""
        try:
            success = self.db_manager.extend_trial(request.user_email, request.extension_days)
            
            if not success:
                return {
                    "success": False,
                    "message": "Failed to extend trial. User may not have an active trial."
                }
            
            # Get updated subscription info
            status_info = self.db_manager.check_trial_status(request.user_email)
            
            return {
                "success": True,
                "message": f"Trial extended by {request.extension_days} days",
                "new_trial_end_date": status_info.get("trial_end_date"),
                "days_remaining": status_info["days_remaining"]
            }
            
        except Exception as e:
            raise Exception(f"Failed to extend trial: {str(e)}")
    
    async def upgrade_subscription(self, request: SubscriptionUpgradeRequest) -> Dict[str, Any]:
        """Upgrade user subscription plan"""
        try:
            success = self.db_manager.upgrade_subscription(request.user_email, request.new_plan)
            
            if not success:
                return {
                    "success": False,
                    "message": "Failed to upgrade subscription. User may not exist."
                }
            
            # Get updated subscription info
            status_info = self.db_manager.check_trial_status(request.user_email)
            
            return {
                "success": True,
                "message": f"Subscription upgraded to {request.new_plan.value}",
                "new_plan": request.new_plan.value,
                "status": status_info["status"],
                "subscription_end_date": status_info.get("subscription_end_date")
            }
            
        except Exception as e:
            raise Exception(f"Failed to upgrade subscription: {str(e)}")
    
    async def check_access_permission(self, user_email: str, feature: str = "premium") -> Dict[str, Any]:
        """Check if user has access to premium features"""
        try:
            status_info = self.db_manager.check_trial_status(user_email)
            
            # Define feature access rules
            access_rules = {
                "premium": status_info["can_access_premium"],
                "basic": True,  # Basic features are always available
                "trial": status_info["is_trial_active"]
            }
            
            has_access = access_rules.get(feature, False)
            
            return {
                "has_access": has_access,
                "user_email": user_email,
                "feature": feature,
                "status": status_info["status"],
                "plan": status_info["plan"],
                "is_trial_active": status_info["is_trial_active"],
                "days_remaining": status_info["days_remaining"]
            }
            
        except Exception as e:
            raise Exception(f"Failed to check access permission: {str(e)}")
    
    async def get_subscription_analytics(self) -> SubscriptionAnalytics:
        """Get subscription analytics"""
        try:
            analytics_data = self.db_manager.get_subscription_analytics()
            
            return SubscriptionAnalytics(
                total_users=analytics_data["total_users"],
                trial_users=analytics_data["trial_users"],
                active_subscribers=analytics_data["active_subscribers"],
                expired_subscriptions=analytics_data["expired_subscriptions"],
                trial_conversion_rate=analytics_data["trial_conversion_rate"],
                average_trial_duration=30.0,  # Default 30 days
                plan_distribution=analytics_data["plan_distribution"]
            )
            
        except Exception as e:
            raise Exception(f"Failed to get subscription analytics: {str(e)}")
    
    async def cleanup_expired_trials(self) -> Dict[str, Any]:
        """Clean up expired trials"""
        try:
            expired_count = self.db_manager.cleanup_expired_trials()
            
            return {
                "success": True,
                "expired_trials_updated": expired_count,
                "message": f"Updated {expired_count} expired trials"
            }
            
        except Exception as e:
            raise Exception(f"Failed to cleanup expired trials: {str(e)}")
    
    def _subscription_to_response(self, subscription) -> SubscriptionResponse:
        """Convert database subscription to response model"""
        now = datetime.now(timezone.utc)
        
        # Map plan_code to plan enum
        plan_code_map = {
            0: SubscriptionPlan.FREE,
            1: SubscriptionPlan.BASIC,
            2: SubscriptionPlan.PREMIUM,
            3: SubscriptionPlan.ENTERPRISE
        }
        plan_code_value = int(subscription.plan_code) if subscription.plan_code else 0
        plan = plan_code_map.get(plan_code_value, SubscriptionPlan.FREE)
        
        # Determine status based on is_trial and dates
        if subscription.is_trial:
            is_trial_active = (
                subscription.subscription_end_date and 
                subscription.subscription_end_date > now
            )
            status = SubscriptionStatus.TRIAL if is_trial_active else SubscriptionStatus.EXPIRED
        else:
            is_trial_active = False
            is_active = (
                subscription.subscription_end_date and 
                subscription.subscription_end_date > now
            )
            status = SubscriptionStatus.ACTIVE if is_active else SubscriptionStatus.EXPIRED
        
        # Calculate days remaining
        if subscription.subscription_end_date:
            days_remaining = max(0, (subscription.subscription_end_date - now).days)
        else:
            days_remaining = 0
        
        # Trial dates - use subscription_start_date and subscription_end_date for trials
        trial_start_date = subscription.subscription_start_date if subscription.is_trial else None
        trial_end_date = subscription.subscription_end_date if subscription.is_trial else None
        
        return SubscriptionResponse(
            id=subscription.user_email,  # Use user_email as ID since it's the PK
            user_email=subscription.user_email,
            user_name=subscription.user_name,
            plan=plan,
            status=status,
            trial_start_date=trial_start_date,
            trial_end_date=trial_end_date,
            subscription_start_date=subscription.subscription_start_date,
            subscription_end_date=subscription.subscription_end_date,
            created_at=subscription.created_at,
            updated_at=subscription.updated_at,
            is_trial_active=is_trial_active,
            days_remaining=days_remaining
        )

    async def activate_product_trial(self, request: ProductTrialActivationRequest) -> ProductTrialActivationResponse:
        """Activate product trial with plan_code validation"""
        try:
            # Validate plan_code exists in plan_master
            plan_validation = self.db_manager.validate_plan_code(request.plan_code)
            if not plan_validation["valid"]:
                raise ValueError(plan_validation["message"])
            
            # Create subscription with validated plan_code
            subscription = self.db_manager.create_subscription_with_plan_code(
                user_email=request.user_email,
                user_name=request.user_name,
                plan_code=request.plan_code
            )
            
            # Return response with plan details
            return ProductTrialActivationResponse(
                user_email=subscription.user_email,
                startDate=subscription.subscription_start_date,
                endDate=subscription.subscription_end_date,
                planCode=int(subscription.plan_code),
                planName=plan_validation["plan_name"]
            )
            
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise Exception(f"Failed to activate product trial: {str(e)}")
    
    async def activate_paid_subscription(self, user_email: str, plan_code: int, 
                                        payment_id: str = None, payment_status: str = None) -> SubscriptionResponse:
        """Activate paid subscription and update product_subscriptions accordingly"""
        try:
            # Activate paid subscription
            subscription = self.db_manager.activate_paid_subscription(
                user_email=user_email,
                plan_code=plan_code,
                payment_id=payment_id,
                payment_status=payment_status
            )
            
            # Return response
            return self._subscription_to_response(subscription)
            
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise Exception(f"Failed to activate paid subscription: {str(e)}")

# Global subscription service instance
subscription_service = SubscriptionService()

