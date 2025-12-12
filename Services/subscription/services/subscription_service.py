from datetime import datetime, timezone
from typing import List, Optional

from ..database import subscription_manager, normalize_status
from ..subscription_schemas import (
    SubscriptionRequest,
    SubscriptionStatusResponse,
    SubscriptionStatus,
    SubscriptionPlan,
    SubscriptionType,
    ProductTrialActivationRequest,
    ProductTrialActivationResponse,
    UserDetailsResponse,
    ProductManagerRecord,
    ZohoPaymentCallbackRequest,
    PaymentSuccessParams,
)


class SubscriptionService:
    """Business logic around user_details and product_manager."""

    def __init__(self):
        self.db_manager = subscription_manager

    async def create_subscription(
        self, request: SubscriptionRequest, phone_no: Optional[str] = None
    ) -> UserDetailsResponse:
        """Ensure the user exists inside user_details (no product rows are created here)."""
        try:
            user = self.db_manager.create_or_get_user(
                user_email=request.user_email,
                user_name=request.user_name,
                phone_no=phone_no,
            )
            return UserDetailsResponse(
                user_email=user.user_email,
                user_name=user.user_name,
                phone_no=user.phone_no,
                status=normalize_status(user.status),
                created_at=user.created_at,
                updated_at=user.updated_at,
            )
        except Exception as exc:
            raise Exception(f"Failed to upsert user: {exc}") from exc

    async def get_user_details(self, user_email: str) -> Optional[UserDetailsResponse]:
        user = self.db_manager.get_user_details(user_email)
        if not user:
            return None
        return UserDetailsResponse(
            user_email=user.user_email,
            user_name=user.user_name,
            phone_no=user.phone_no,
            status=normalize_status(user.status),
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    async def get_subscription_status(self, user_email: str) -> SubscriptionStatusResponse:
        user = self.db_manager.get_user_details(user_email)
        if not user:
            return SubscriptionStatusResponse(
                user_email=user_email,
                has_subscription=False,
                status=SubscriptionStatus.TRIAL,
                plan=SubscriptionPlan.FREE,
                plan_code=None,
                is_trial_active=False,
                trial_end_date=None,
                subscription_end_date=None,
                days_remaining=0,
                can_access_premium=False,
            )

        products = self.db_manager.get_user_products(user_email)
        if not products:
            return SubscriptionStatusResponse(
                user_email=user_email,
                has_subscription=False,
                status=SubscriptionStatus.EXPIRED,
                plan=SubscriptionPlan.FREE,
                plan_code=None,
                is_trial_active=False,
                trial_end_date=None,
                subscription_end_date=None,
                days_remaining=0,
                can_access_premium=False,
            )

        now = datetime.now(timezone.utc)
        active_trials = [
            p for p in products
            if p.subscription_type == SubscriptionType.TRIAL
            and p.subscription_end_date
            and p.subscription_end_date > now
        ]
        active_paid = [
            p for p in products
            if p.subscription_type == SubscriptionType.PAID
            and p.subscription_end_date
            and p.subscription_end_date > now
        ]

        all_end_dates = [p.subscription_end_date for p in products if p.subscription_end_date]
        latest_end = max(all_end_dates) if all_end_dates else None
        days_remaining = max(0, (latest_end - now).days) if latest_end else 0

        is_trial_active = len(active_trials) > 0
        can_access_premium = is_trial_active or len(active_paid) > 0

        status = SubscriptionStatus.EXPIRED
        if len(active_paid) > 0:
            status = SubscriptionStatus.ACTIVE
        elif is_trial_active:
            status = SubscriptionStatus.TRIAL

        trial_end = max((p.subscription_end_date for p in active_trials), default=None)

        plan = SubscriptionPlan.PREMIUM if status == SubscriptionStatus.ACTIVE else SubscriptionPlan.FREE

        return SubscriptionStatusResponse(
            user_email=user_email,
            has_subscription=True,
            status=status,
            plan=plan,
            plan_code=None,
            is_trial_active=is_trial_active,
            trial_end_date=trial_end,
            subscription_end_date=latest_end,
            days_remaining=days_remaining,
            can_access_premium=can_access_premium,
        )

    async def activate_product_trial(
        self, request: ProductTrialActivationRequest
    ) -> ProductTrialActivationResponse:
        """Create product_manager rows for all products for the user."""
        try:
            window = self.db_manager.create_product_trials(
                user_email=request.user_email,
                user_name=request.user_name,
                plan_code=request.plan_code,
                subscription_type=SubscriptionType.TRIAL,
            )
            return ProductTrialActivationResponse(
                user_email=request.user_email.lower(),
                startDate=window["start"],
                endDate=window["end"],
                planCode=request.plan_code,
                planName=window["plan_name"] or "Trial Plan",
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        except Exception as exc:
            raise Exception(f"Failed to activate trial: {exc}") from exc

    async def get_user_products(self, user_email: str) -> List[ProductManagerRecord]:
        products = self.db_manager.get_user_products(user_email)
        return [self.db_manager.serialize_product(p) for p in products]

    async def process_payment_callback(self, email: str, plan_name: str, subscription_id: str) -> dict:
        """Process Zoho payment callback"""
        try:
            return self.db_manager.activate_paid_subscription(
                user_email=email,
                plan_name=plan_name,
                subscription_id=subscription_id,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        except Exception as exc:
            raise Exception(f"Failed to process callback: {exc}") from exc



subscription_service = SubscriptionService()
