import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from Databases.app_data_db_connection import create_connection, get_session

# Case-insensitive import for Subscription (handles 'subscription' vs 'Subscription')
import os
import sys
import importlib

# Find the actual subscription directory name
_services_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
_subscription_dir = None
if os.path.exists(_services_dir):
    for item in os.listdir(_services_dir):
        if item.lower() == 'subscription':
            _subscription_dir = item
            break

if _subscription_dir:
    _subscription_models = importlib.import_module(f'Services.{_subscription_dir}.subscription_models')
    _subscription_schemas = importlib.import_module(f'Services.{_subscription_dir}.subscription_schemas')
    ProductManager = _subscription_models.ProductManager
    Subscription = _subscription_models.Subscription
    ProductCode = _subscription_schemas.ProductCode
    SubscriptionStatus = _subscription_schemas.SubscriptionStatus
else:
    # Fallback to direct import (will fail if directory doesn't exist)
    from Services.Subscription.subscription_models import ProductManager, Subscription
    from Services.Subscription.subscription_schemas import ProductCode, SubscriptionStatus

logger = logging.getLogger(__name__)

# Ensure the Neon connection is initialized
try:
    create_connection()
except Exception as exc:
    logger.error("Failed to initialize Neon DB connection for Single Sign On service: %s", exc)


router = APIRouter(prefix="/single-sign-on", tags=["Single Sign On"])


def _format_date_dd_mm_yyyy(value: datetime | None) -> str | None:
    """Format datetime to DD-MM-YYYY format."""
    if value is None:
        return None
    try:
        # Handle both timezone-aware and timezone-naive datetimes
        if isinstance(value, datetime):
            return value.strftime("%d-%m-%Y")
        return None
    except Exception as exc:
        logger.warning("Error formatting date %s: %s", value, exc)
        return None


def _get_plan_name(plan_code: str | None, session: Session) -> str:
    """Get plan name from plan_master table using plan_code."""
    if not plan_code:
        return "free"
    
    try:
        plan_code_int = int(plan_code)
        result = session.execute(
            text("SELECT plan_name FROM plan_master WHERE plan_code = :plan_code"),
            {"plan_code": plan_code_int}
        )
        plan = result.fetchone()
        if plan and plan[0]:
            # Convert plan_name to lowercase for consistency
            plan_name = plan[0].lower()
            return plan_name
    except (ValueError, Exception) as exc:
        logger.warning("Error fetching plan name for plan_code %s: %s", plan_code, exc)
    
    # Default fallback based on plan_code
    plan_code_lower = str(plan_code).lower()
    if "basic" in plan_code_lower:
        return "basic"
    elif "premium" in plan_code_lower:
        return "premium"
    elif "enterprise" in plan_code_lower:
        return "enterprise"
    return "free"


def _is_subscription_active(subscription) -> bool:
    """Check if subscription is currently active based on status and dates using simplified date fields."""
    try:
        now = datetime.now(timezone.utc)  # Use timezone-aware datetime
        
        # Check if subscription is active using simplified date fields
        if subscription.subscription_start_date and subscription.subscription_end_date:
            # Ensure both datetimes are timezone-aware for comparison
            start_date = subscription.subscription_start_date
            end_date = subscription.subscription_end_date
            
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            
            # Check if current time is within subscription period
            is_active = start_date <= now <= end_date
            
            # Also check status is TRIAL or ACTIVE
            # Handle both enum and string status values
            if hasattr(subscription.status, 'value'):
                status_value = subscription.status.value
            elif isinstance(subscription.status, str):
                status_value = subscription.status
            else:
                status_value = str(subscription.status)
            
            return is_active and (status_value == SubscriptionStatus.TRIAL.value or 
                                 status_value == SubscriptionStatus.ACTIVE.value)
        
        return False
    except Exception as exc:
        logger.warning("Error checking subscription active status: %s", exc)
        # Default to False if there's an error
        return False


@router.get("/{userid}")
async def single_sign_on(userid: str) -> Dict[str, Any]:
    """
    Validate user access for TRADEAI product and return subscription details in the requested format.
    """
    session: Session | None = None

    try:
        session = get_session()
        if not session:
            raise HTTPException(
                status_code=500,
                detail="Failed to establish database connection"
            )
        user_email = userid.lower()

        # ------------------------------------------------------------------
        # Validate TRADEAI access using product_manager table
        # ------------------------------------------------------------------
        trad_ai_subscription = (
            session.query(ProductManager)
            .filter(
                ProductManager.user_email == user_email,
                ProductManager.product_code == ProductCode.TRADAI.value,  # Compare with string value
            )
            .order_by(ProductManager.updated_at.desc())
            .first()
        )

        if not trad_ai_subscription:
            raise HTTPException(
                status_code=404,
                detail="TRADEAI product subscription not found for the provided user",
            )

        # Get user name from Subscription table
        main_subscription = (
            session.query(Subscription)
            .filter(Subscription.user_email == user_email)
            .first()
        )
        
        user_name = main_subscription.user_name if main_subscription and main_subscription.user_name else None
        
        # If user_name is not found, try to extract from email or use a default
        if not user_name:
            # Try to get name from email (e.g., "upendra.singh@pravisht.com" -> "Upendra Singh")
            email_parts = user_email.split("@")[0].split(".")
            user_name = " ".join(word.capitalize() for word in email_parts)
        
        # For product_manager-based flow, plan name is not stored; use a simple label
        plan_name = "tradeai"
        
        # Get subscription dates from ProductManager
        subscription_start_date = trad_ai_subscription.subscription_start_date
        subscription_end_date = trad_ai_subscription.subscription_end_date
        
        # Check if subscription is active
        is_active = _is_subscription_active(trad_ai_subscription)
        
        # product_manager does not carry product_metadata; keep a fixed default
        allowed_accounts = 10  # Default value
        
        # Build response in the requested format
        response_payload = {
            "status": "success",
            "message": "User information fetched successfully",
            "data": {
                "name": user_name,
                "email": user_email,
                "allowed_accounts": allowed_accounts,
                "is_active": is_active,
                "subscribed_plan": plan_name,
                "subscription_start_date": _format_date_dd_mm_yyyy(subscription_start_date),
                "subscription_end_date": _format_date_dd_mm_yyyy(subscription_end_date),
            }
        }

        return response_payload

    except HTTPException:
        raise
    except Exception as exc:
        error_message = str(exc)
        logger.exception("Error processing Single Sign On request for user %s: %s", userid, exc)
        # Include more details in development, but keep generic in production
        detail_message = f"Failed to process Single Sign On request: {error_message}"
        raise HTTPException(status_code=500, detail=detail_message) from exc
    finally:
        if session:
            session.close()

