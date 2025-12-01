from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

from Databases.app_data_db_connection import (
    create_connection,
    get_session as get_neon_session,
    Base as NeonBase,
    init_database as init_neon_database,
)
from Services.Subscription.subscription_models import UserDetails, ProductManager
from Services.Subscription.subscription_schemas import (
    SubscriptionStatus,
    SubscriptionType,
    ProductCode,
    ProductManagerRecord,
)
from Services.Subscription.subscription_config import settings

Base = NeonBase

PRODUCT_CODE_SEQUENCE: List[ProductCode] = [
    ProductCode.MARKETAI,
    ProductCode.TRADAI,
    ProductCode.CHATAI,
    ProductCode.AUTOMATIONAI,
]

PRODUCT_CODE_SHORT_MAP: Dict[ProductCode, str] = {
    ProductCode.MARKETAI: "M",
    ProductCode.TRADAI: "T",
    ProductCode.CHATAI: "A",
    ProductCode.AUTOMATIONAI: "A",
}

# Reverse mapping: from database string values to ProductCode enum
# Handles both short codes (A, M, T, C) and full enum values (CHATAI, MARKETAI, etc.)
PRODUCT_CODE_FROM_DB: Dict[str, ProductCode] = {
    "M": ProductCode.MARKETAI,
    "MARKETAI": ProductCode.MARKETAI,
    "T": ProductCode.TRADAI,
    "TRADAI": ProductCode.TRADAI,
    "TRADEAI": ProductCode.TRADAI,  # Handle both spellings
    "A": ProductCode.CHATAI,  # Default to CHATAI for 'A' (ambiguous with AUTOMATIONAI)
    "C": ProductCode.CHATAI,  # Old short code for CHATAI
    "CHATAI": ProductCode.CHATAI,
    "AUTOMATIONAI": ProductCode.AUTOMATIONAI,
}

CHATAI_DEFAULT_TOKENS = settings.CHATAI_DEFAULT_TOKENS


def db_string_to_product_code(db_value: str) -> ProductCode:
    """Convert database string value to ProductCode enum."""
    if not db_value:
        raise ValueError("Empty product_code value")
    # Normalize to uppercase
    normalized = db_value.upper().strip()
    if normalized in PRODUCT_CODE_FROM_DB:
        return PRODUCT_CODE_FROM_DB[normalized]
    # Try to match enum value directly
    try:
        return ProductCode(normalized)
    except ValueError:
        # Provide helpful error message with known values
        known_values = ", ".join(sorted(set(PRODUCT_CODE_FROM_DB.keys())))
        raise ValueError(
            f"Unknown product_code value: '{db_value}'. "
            f"Known values: {known_values}. "
            f"Please update PRODUCT_CODE_FROM_DB mapping if this is a valid product code."
        )


# Status value mapping: from database short codes to full status strings
STATUS_FROM_DB: Dict[str, str] = {
    "T": "TRIAL",
    "TRIAL": "TRIAL",
    "F": "FALSE",
    "FALSE": "FALSE",
    "P": "PAID",
    "PAID": "PAID",
    "E": "EXPIRED",
    "EXPIRED": "EXPIRED",
    "C": "CANCELLED",
    "CANCELLED": "CANCELLED",
    "S": "SUSPENDED",
    "SUSPENDED": "SUSPENDED",
}


def normalize_status(db_status: str) -> str:
    """Convert database status value (single char or full string) to full status string."""
    if not db_status:
        return "FALSE"  # Default status
    # Normalize to uppercase
    normalized = db_status.upper().strip()
    # Return mapped value or the original if it's already a full status
    return STATUS_FROM_DB.get(normalized, normalized if len(normalized) > 1 else "FALSE")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SubscriptionManager:
    """Data access layer built directly on user_details and product_manager tables."""
    
    def __init__(self):
        if not create_connection():
            raise RuntimeError("Failed to connect to Neon database")
        self.init_database()
    
    def init_database(self):
        try:
            init_neon_database()
        except Exception as exc:  # pragma: no cover - just a safeguard
            print(f"Warning: could not initialize subscription tables: {exc}")
    
    def get_session(self) -> Session:
        return get_neon_session()
    
    def close_session(self, session: Session):
        session.close()
    
    # -------------------------------------------------------------------------
    # user_details helpers
    # -------------------------------------------------------------------------
    def create_or_get_user(
        self,
        user_email: str,
        user_name: Optional[str] = None,
        phone_no: Optional[str] = None,
        status: str = "FALSE",
    ) -> UserDetails:
        session = self.get_session()
        try:
            email = user_email.lower()
            user = session.query(UserDetails).filter(UserDetails.user_email == email).first()
            now = utcnow()

            if user:
                updated = False
                if user_name and user.user_name != user_name:
                    user.user_name = user_name
                    updated = True
                if phone_no and user.phone_no != phone_no:
                    user.phone_no = phone_no
                    updated = True
                if updated:
                    user.updated_at = now
                    session.commit()
                return user

            user = UserDetails(
                user_email=email,
                user_name=user_name,
                phone_no=phone_no,
                status=status,
                created_at=now,
                updated_at=now,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        except Exception:
            session.rollback()
            raise
        finally:
            self.close_session(session)

    def get_user_details(self, user_email: str) -> Optional[UserDetails]:
        session = self.get_session()
        try:
            return (
                session.query(UserDetails)
                .filter(UserDetails.user_email == user_email.lower())
                .first()
            )
        finally:
            self.close_session(session)
    
    # -------------------------------------------------------------------------
    # product_manager helpers
    # -------------------------------------------------------------------------
    def user_has_products(self, user_email: str) -> bool:
        session = self.get_session()
        try:
            return (
                session.query(ProductManager)
                .filter(ProductManager.user_email == user_email.lower())
                .first()
                is not None
            )
        finally:
            self.close_session(session)
    
    def get_product_entry(self, user_email: str, product_code: ProductCode) -> Optional[ProductManager]:
        session = self.get_session()
        try:
            # Compare with string value since product_code column is now String
            return (
                session.query(ProductManager)
                .filter(
                    ProductManager.user_email == user_email.lower(),
                    ProductManager.product_code == product_code.value,
                )
                .first()
            )
        finally:
            self.close_session(session)
    
    def get_user_products(self, user_email: str) -> List[ProductManager]:
        session = self.get_session()
        try:
            return (
                session.query(ProductManager)
                .filter(ProductManager.user_email == user_email.lower())
                .order_by(ProductManager.created_at.asc())
                .all()
            )
        finally:
            self.close_session(session)
    
    def create_product_trials(
        self,
        user_email: str,
        user_name: str,
        plan_code: int,
        subscription_type: SubscriptionType = SubscriptionType.TRIAL,
    ) -> Dict[str, datetime]:
        session = self.get_session()
        try:
            email = user_email.lower()
            if (
                session.query(ProductManager)
                .filter(ProductManager.user_email == email)
                .first()
            ):
                raise ValueError(f"User {user_email} already has product entries")

            plan_info = self.validate_plan_code(plan_code)
            if not plan_info["valid"]:
                raise ValueError(plan_info["message"])
            
            # Ensure user exists
            user = (
                session.query(UserDetails)
                .filter(UserDetails.user_email == email)
                .first()
            )
            now = utcnow()
            if not user:
                user = UserDetails(
                    user_email=email,
                    user_name=user_name,
                    status="TRIAL",
                    created_at=now,
                    updated_at=now,
                )
                session.add(user)
                session.flush()
            else:
                # Existing user successfully activating trial – flip status to TRIAL
                if user.status != "TRIAL":
                    user.status = "TRIAL"
                    user.updated_at = now

            end_date = now + timedelta(days=plan_info["extension_days"] or settings.DEFAULT_TRIAL_DAYS)

            for product in PRODUCT_CODE_SEQUENCE:
                chatai_key = (
                    f"chatai_{email.replace('@', '_').replace('.', '_')}_{int(now.timestamp())}"
                    if product == ProductCode.CHATAI
                    else "N/A"
                )
                total_tokens = CHATAI_DEFAULT_TOKENS if product == ProductCode.CHATAI else 0

                entry = ProductManager(
                    id=str(uuid.uuid4()),
                    user_email=email,
                    product_code=product.value,  # Store as string (full enum value)
                    subscription_type=subscription_type,
                    status=SubscriptionStatus.ACTIVE,
                    subscription_start_date=now,
                    subscription_end_date=end_date,
                    chatai_key=chatai_key,
                    total_token=total_tokens,
                    used_token=0,
                    remaining_token=total_tokens,
                    created_at=now,
                    updated_at=now,
                )
                session.add(entry)
            
            session.commit()
            return {"start": now, "end": end_date, "plan_name": plan_info["plan_name"]}
        except Exception:
            session.rollback()
            raise
        finally:
            self.close_session(session)
    
    def serialize_product(self, product: ProductManager) -> ProductManagerRecord:
        # Convert database string to ProductCode enum
        product_code_enum = db_string_to_product_code(product.product_code)
        product_label = PRODUCT_CODE_SHORT_MAP.get(product_code_enum, product_code_enum.value)
        if product_code_enum == ProductCode.CHATAI:
            total_token = str(product.total_token or 0)
            used_token = str(product.used_token or 0)
            remaining_token = str(product.remaining_token or 0)
        else:
            total_token = used_token = remaining_token = "N/A"
        return ProductManagerRecord(
            id=product.id,
            user_email=product.user_email,
            product_code=product_code_enum,
            product_code_label=product_label,
            subscription_type=product.subscription_type,
            status=product.status,
            subscription_start_date=product.subscription_start_date,
            subscription_end_date=product.subscription_end_date,
            chatai_key=product.chatai_key,
            total_token=total_token,
            used_token=used_token,
            remaining_token=remaining_token,
            created_at=product.created_at,
            updated_at=product.updated_at,
        )

    # -------------------------------------------------------------------------
    # Plan helpers (reuse legacy tables)
    # -------------------------------------------------------------------------
    def validate_plan_code(self, plan_code: int) -> Dict[str, Optional[str]]:
        session = self.get_session()
        try:
            result = session.execute(
                text(
                    """
                    SELECT plan_code, plan_name, extension_days
                    FROM plan_master
                    WHERE plan_code = :plan_code
                    """
                ),
                {"plan_code": plan_code},
            )
            row = result.fetchone()
            if not row:
                return {"valid": False, "message": f"Plan code {plan_code} not found", "plan_name": None, "extension_days": None}
            return {
                "valid": True,
                "plan_name": row[1],
                "extension_days": int(row[2]) if row[2] else settings.DEFAULT_TRIAL_DAYS,
            }
        finally:
            self.close_session(session)
    

# Shared manager instance (product service reuses same object)
subscription_manager = SubscriptionManager()
product_subscription_manager = subscription_manager

