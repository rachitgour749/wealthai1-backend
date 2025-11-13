from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, Enum as SQLEnum, JSON, Numeric, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import os
import sys
import uuid

# Import Neon database connection
# Get the project root directory (two levels up from Services/subscription)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
databases_path = os.path.join(project_root, 'Databases')

# Add Databases directory to path if not already there
if databases_path not in sys.path:
    sys.path.insert(0, databases_path)

try:
    from neon_db_connection import create_connection, get_session as get_neon_session, get_engine, Base as NeonBase, init_database as init_neon_database
except ImportError as e:
    print(f"Warning: Could not import neon_db_connection: {e}")
    # Fallback to local base if import fails
    from sqlalchemy.orm import declarative_base as _declarative_base
    NeonBase = _declarative_base()
    def create_connection():
        return False
    def get_neon_session():
        raise RuntimeError("Neon database not configured")
    def init_neon_database():
        pass

from .models import SubscriptionStatus, SubscriptionPlan, ProductCode, SubscriptionType

# Use Base from Neon connection
Base = NeonBase

class Subscription(Base):
    """Database model for user subscriptions - matches new Neon PostgreSQL schema"""
    __tablename__ = "subscription"
    
    # Primary key is user_email
    user_email = Column(String, primary_key=True)
    user_name = Column(String)
    
    # Subscription dates
    subscription_start_date = Column(DateTime(timezone=True))
    subscription_end_date = Column(DateTime(timezone=True))
    
    # Metadata timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Plan and trial info
    plan_code = Column(Numeric, nullable=True)  # Numeric plan code
    is_trial = Column(Boolean, default=False)  # Boolean trial flag (False for first-time sign-in)

class ProductSubscription(Base):
    """Database model for product-specific subscriptions - using Neon PostgreSQL"""
    __tablename__ = "product_subscriptions"
    
    id = Column(String, primary_key=True)
    user_email = Column(String(255), nullable=False, index=True)
    product_code = Column(SQLEnum(ProductCode), nullable=False)
    subscription_type = Column(SQLEnum(SubscriptionType), nullable=False, default=SubscriptionType.TRIAL)
    status = Column(SQLEnum(SubscriptionStatus), nullable=False, default=SubscriptionStatus.TRIAL)
    plan_code = Column(String(50), nullable=False, default="FREE")
    
    # Trial period fields
    trial_start_date = Column(DateTime(timezone=True), nullable=True)
    trial_end_date = Column(DateTime(timezone=True), nullable=True)
    trial_duration_days = Column(Integer, default=7)  # Default 7 days trial
    
    # Paid subscription fields
    paid_start_date = Column(DateTime(timezone=True), nullable=True)
    paid_end_date = Column(DateTime(timezone=True), nullable=True)
    payment_id = Column(String(100), nullable=True)
    payment_status = Column(String(50), nullable=True)
    
    # Bundle information
    bundle_id = Column(String(36), nullable=True)  # Links products in same bundle
    is_bundle_subscription = Column(Boolean, default=False)
    
    # ChatAI Token Tracking Fields
    chatai_key = Column(String(100), nullable=True)  # Unique key for ChatAI subscription
    total_tokens = Column(Integer, default=0)  # Total tokens allocated for this subscription
    used_tokens = Column(Integer, default=0)  # Tokens used so far
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    product_metadata = Column(JSON, nullable=True)  # Store additional product-specific data

class ProductAccessLog(Base):
    """Database model for tracking product access attempts - using Neon PostgreSQL"""
    __tablename__ = "product_access_logs"
    
    id = Column(String, primary_key=True)
    user_email = Column(String(255), nullable=False, index=True)
    product_code = Column(SQLEnum(ProductCode), nullable=False)
    access_attempted_at = Column(DateTime(timezone=True), default=func.now())
    access_granted = Column(Boolean, nullable=False)
    access_type = Column(String(20), nullable=True)  # trial, paid, bundle
    subscription_id = Column(String(36), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

class SubscriptionManager:
    """Database manager for subscription operations using Neon PostgreSQL"""
    
    def __init__(self):
        # Initialize Neon database connection
        if not create_connection():
            raise RuntimeError("Failed to connect to Neon database")
        
        # Create tables if they don't exist
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            init_neon_database()
        except Exception as e:
            print(f"Warning: Could not initialize all tables: {e}")
    
    def get_session(self) -> Session:
        """Get database session from Neon connection"""
        return get_neon_session()
    
    def close_session(self, session: Session):
        """Close database session"""
        session.close()
    
    def create_subscription(self, user_email: str, user_name: str = None, 
                          plan: SubscriptionPlan = SubscriptionPlan.FREE) -> Subscription:
        """Create a new subscription for first-time sign-in (is_trial=False)"""
        session = self.get_session()
        try:
            # Check if user already has a subscription
            existing = session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
            
            if existing:
                return existing
            
            # Calculate subscription dates (30-day period)
            now = datetime.now(timezone.utc)  # Use timezone-aware datetime
            start_date = now
            end_date = now + timedelta(days=7)
            
            # Map plan to plan_code (numeric)
            plan_code_map = {
                SubscriptionPlan.FREE: 0,
                SubscriptionPlan.BASIC: 1,
                SubscriptionPlan.PREMIUM: 2,
                SubscriptionPlan.ENTERPRISE: 3
            }
            plan_code = plan_code_map.get(plan, 0)
            
            subscription = Subscription(
                user_email=user_email.lower(),
                user_name=user_name,
                subscription_start_date=start_date,
                subscription_end_date=end_date,
                plan_code=plan_code,
                is_trial=False,  # Set to False for first-time sign-in
                created_at=now,
                updated_at=now
            )
            
            session.add(subscription)
            session.commit()
            session.refresh(subscription)
            return subscription
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_subscription_by_email(self, user_email: str) -> Optional[Subscription]:
        """Get subscription by user email"""
        session = self.get_session()
        try:
            return session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
        finally:
            self.close_session(session)
    
    def update_subscription_status(self, user_email: str, status: SubscriptionStatus) -> bool:
        """Update subscription status (maps to is_trial boolean)"""
        session = self.get_session()
        try:
            subscription = session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
            
            if subscription:
                # Map status to is_trial
                subscription.is_trial = (status == SubscriptionStatus.TRIAL)
                subscription.updated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def extend_trial(self, user_email: str, extension_days: int) -> bool:
        """Extend trial period for a user"""
        session = self.get_session()
        try:
            subscription = session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
            
            if subscription and subscription.is_trial:
                # Extend the subscription_end_date
                if subscription.subscription_end_date:
                    subscription.subscription_end_date += timedelta(days=extension_days)
                else:
                    subscription.subscription_end_date = datetime.now(timezone.utc) + timedelta(days=extension_days)
                subscription.updated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def upgrade_subscription(self, user_email: str, new_plan: SubscriptionPlan) -> bool:
        """Upgrade user subscription plan"""
        session = self.get_session()
        try:
            subscription = session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
            
            if subscription:
                # Map plan to plan_code
                plan_code_map = {
                    SubscriptionPlan.FREE: 0,
                    SubscriptionPlan.BASIC: 1,
                    SubscriptionPlan.PREMIUM: 2,
                    SubscriptionPlan.ENTERPRISE: 3
                }
                subscription.plan_code = plan_code_map.get(new_plan, 0)
                subscription.is_trial = False  # Upgrade means no longer trial
                subscription.subscription_start_date = datetime.now(timezone.utc)
                subscription.subscription_end_date = datetime.now(timezone.utc) + timedelta(days=365)  # 1 year
                subscription.updated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def check_trial_status(self, user_email: str) -> Dict[str, Any]:
        """Check if user's trial is still active"""
        subscription = self.get_subscription_by_email(user_email)
        
        if not subscription:
            return {
                "has_subscription": False,
                "is_trial_active": False,
                "can_access_premium": False,
                "status": "no_subscription"
            }
        
        now = datetime.now(timezone.utc)  # Use timezone-aware datetime
        is_trial_active = (
            subscription.is_trial and 
            subscription.subscription_end_date and
            subscription.subscription_end_date > now
        )
        
        is_subscription_active = (
            not subscription.is_trial and
            subscription.subscription_end_date and
            subscription.subscription_end_date > now
        )
        
        can_access_premium = is_trial_active or is_subscription_active
        
        # Calculate days remaining
        if subscription.subscription_end_date:
            days_remaining = max(0, (subscription.subscription_end_date - now).days)
        else:
            days_remaining = 0
        
        # Map plan_code to plan enum
        plan_code_map = {
            0: SubscriptionPlan.FREE.value,
            1: SubscriptionPlan.BASIC.value,
            2: SubscriptionPlan.PREMIUM.value,
            3: SubscriptionPlan.ENTERPRISE.value
        }
        plan_value = plan_code_map.get(int(subscription.plan_code) if subscription.plan_code else 0, SubscriptionPlan.FREE.value)
        
        # Determine status
        if subscription.is_trial:
            status_value = SubscriptionStatus.TRIAL.value if is_trial_active else SubscriptionStatus.EXPIRED.value
        else:
            status_value = SubscriptionStatus.ACTIVE.value if is_subscription_active else SubscriptionStatus.EXPIRED.value
        
        return {
            "has_subscription": True,
            "is_trial_active": is_trial_active,
            "is_subscription_active": is_subscription_active,
            "can_access_premium": can_access_premium,
            "status": status_value,
            "plan": plan_value,
            "trial_end_date": subscription.subscription_end_date if subscription.is_trial else None,
            "subscription_end_date": subscription.subscription_end_date,
            "days_remaining": days_remaining
        }
    
    def get_all_subscriptions(self, limit: int = 100, offset: int = 0) -> List[Subscription]:
        """Get all subscriptions with pagination"""
        session = self.get_session()
        try:
            return session.query(Subscription).order_by(
                Subscription.created_at.desc()
            ).offset(offset).limit(limit).all()
        finally:
            self.close_session(session)
    
    def get_subscription_analytics(self) -> Dict[str, Any]:
        """Get subscription analytics"""
        session = self.get_session()
        try:
            now = datetime.now(timezone.utc)
            total_users = session.query(Subscription).count()
            
            # Count trial users (is_trial=True and end_date > now)
            trial_users = session.query(Subscription).filter(
                Subscription.is_trial == True,
                Subscription.subscription_end_date > now
            ).count()
            
            # Count active subscribers (is_trial=False and end_date > now)
            active_subscribers = session.query(Subscription).filter(
                Subscription.is_trial == False,
                Subscription.subscription_end_date > now
            ).count()
            
            # Count expired subscriptions (end_date < now)
            expired_subscriptions = session.query(Subscription).filter(
                Subscription.subscription_end_date < now
            ).count()
            
            # Calculate trial conversion rate
            total_trials = session.query(Subscription).filter(
                Subscription.is_trial == True
            ).count()
            
            conversion_rate = (active_subscribers / total_trials * 100) if total_trials > 0 else 0
            
            # Plan distribution by plan_code
            plan_distribution = {
                "free": session.query(Subscription).filter(Subscription.plan_code == 0).count(),
                "basic": session.query(Subscription).filter(Subscription.plan_code == 1).count(),
                "premium": session.query(Subscription).filter(Subscription.plan_code == 2).count(),
                "enterprise": session.query(Subscription).filter(Subscription.plan_code == 3).count()
            }
            
            return {
                "total_users": total_users,
                "trial_users": trial_users,
                "active_subscribers": active_subscribers,
                "expired_subscriptions": expired_subscriptions,
                "trial_conversion_rate": round(conversion_rate, 2),
                "plan_distribution": plan_distribution
            }
            
        finally:
            self.close_session(session)
    
    def cleanup_expired_trials(self) -> int:
        """Mark expired trials as expired (update is_trial if needed)"""
        session = self.get_session()
        try:
            now = datetime.now(timezone.utc)
            # Update subscriptions where trial has expired but is_trial is still True
            expired_subscriptions = session.query(Subscription).filter(
                Subscription.is_trial == True,
                Subscription.subscription_end_date < now
            ).all()
            
            expired_count = 0
            for subscription in expired_subscriptions:
                # Optionally, you could set is_trial to False, but the schema doesn't require it
                subscription.updated_at = now
                expired_count += 1
            
            session.commit()
            return expired_count
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def validate_plan_code(self, plan_code: int) -> Dict[str, Any]:
        """Validate plan_code exists in plan_master table and return plan details"""
        session = self.get_session()
        try:
            result = session.execute(
                text("SELECT plan_code, plan_name, extension_days, description FROM plan_master WHERE plan_code = :plan_code"),
                {"plan_code": plan_code}
            )
            plan = result.fetchone()
            
            if not plan:
                return {
                    "valid": False,
                    "message": f"Plan code {plan_code} does not exist in plan_master"
                }
            
            return {
                "valid": True,
                "plan_code": int(plan[0]),
                "plan_name": plan[1],
                "extension_days": int(plan[2]),
                "description": plan[3]
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating plan code: {str(e)}"
            }
        finally:
            self.close_session(session)
    
    def get_products_for_plan_code(self, plan_code: int) -> List[ProductCode]:
        """Get list of ProductCode enums for products included in a plan_code"""
        session = self.get_session()
        try:
            # First, try to query plan_prod_mapping joined with prod_master
            # Note: plan_prod_mapping uses prod_code (numeric), prod_master uses prod_id (integer)
            result = session.execute(
                text("""
                    SELECT pm.prod_code 
                    FROM plan_prod_mapping ppm
                    LEFT JOIN prod_master pm ON ppm.prod_code = pm.prod_id
                    WHERE ppm.plan_code = :plan_code
                """),
                {"plan_code": plan_code}
            )
            rows = result.fetchall()
            
            products = []
            
            # If prod_master has data, use it
            if rows and rows[0][0] is not None:
                # prod_master has data, use prod_code strings
                for row in rows:
                    product_code_str = row[0].upper() if row[0] else None
                    if product_code_str:
                        try:
                            products.append(ProductCode(product_code_str))
                        except ValueError:
                            print(f"Warning: Invalid product code '{product_code_str}' found in prod_master for plan_code {plan_code}")
                            continue
            else:
                # prod_master is empty or join failed, use numeric prod_code mapping
                # Get numeric prod_codes from plan_prod_mapping
                result = session.execute(
                    text("""
                        SELECT prod_code 
                        FROM plan_prod_mapping
                        WHERE plan_code = :plan_code
                    """),
                    {"plan_code": plan_code}
                )
                prod_codes = result.fetchall()
                
                # Map numeric prod_code to ProductCode enum
                # Based on common mapping: 1=TRADAI, 2=MARKETAI, 3=CHATAI, 4=AUTOMATIONAI
                prod_code_mapping = {
                    1: ProductCode.TRADAI,
                    2: ProductCode.MARKETAI,
                    3: ProductCode.CHATAI,
                    4: ProductCode.AUTOMATIONAI
                }
                
                for row in prod_codes:
                    numeric_code = int(row[0]) if row[0] else None
                    if numeric_code and numeric_code in prod_code_mapping:
                        products.append(prod_code_mapping[numeric_code])
                    else:
                        print(f"Warning: Unknown prod_code {numeric_code} for plan_code {plan_code}")
            
            # Debug: Log if no products found
            if not products:
                print(f"Warning: No products found for plan_code {plan_code}")
            
            return products
        except Exception as e:
            print(f"Error getting products for plan_code {plan_code}: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            self.close_session(session)
    
    def create_subscription_with_plan_code(self, user_email: str, user_name: str, plan_code: int, 
                                          is_trial: bool = True) -> Subscription:
        """Create or update subscription using plan_code from plan_master and create/update product_subscriptions"""
        session = self.get_session()
        try:
            # Validate plan_code first
            plan_info = self.validate_plan_code(plan_code)
            if not plan_info["valid"]:
                raise ValueError(plan_info["message"])
            
            # Get products for this plan_code
            product_codes = self.get_products_for_plan_code(plan_code)
            is_bundle = len(product_codes) > 1
            
            # Check if user already has a subscription
            existing = session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
            
            extension_days = plan_info["extension_days"]
            now = datetime.now(timezone.utc)
            start_date = now
            end_date = now + timedelta(days=extension_days)
            
            if existing:
                # Update existing subscription with new plan
                existing.user_name = user_name
                existing.plan_code = plan_code
                existing.subscription_start_date = start_date
                existing.subscription_end_date = end_date
                existing.is_trial = is_trial
                existing.updated_at = now
                subscription = existing
            else:
                # Create new subscription
                subscription = Subscription(
                    user_email=user_email.lower(),
                    user_name=user_name,
                    subscription_start_date=start_date,
                    subscription_end_date=end_date,
                    plan_code=plan_code,
                    is_trial=is_trial,
                    created_at=now,
                    updated_at=now
                )
                session.add(subscription)
            
            # Create or update product_subscriptions for each product in the plan
            for product_code in product_codes:
                # Check if product_subscription exists
                existing_product_sub = session.query(ProductSubscription).filter(
                    ProductSubscription.user_email == user_email.lower(),
                    ProductSubscription.product_code == product_code
                ).first()
                
                if existing_product_sub:
                    # Update existing product subscription
                    existing_product_sub.plan_code = str(plan_code)
                    existing_product_sub.subscription_type = SubscriptionType.TRIAL if is_trial else SubscriptionType.PAID
                    existing_product_sub.status = SubscriptionStatus.TRIAL if is_trial else SubscriptionStatus.ACTIVE
                    existing_product_sub.trial_start_date = start_date if is_trial else existing_product_sub.trial_start_date
                    existing_product_sub.trial_end_date = end_date if is_trial else existing_product_sub.trial_end_date
                    existing_product_sub.trial_duration_days = extension_days if is_trial else existing_product_sub.trial_duration_days
                    existing_product_sub.is_bundle_subscription = is_bundle
                    existing_product_sub.updated_at = now
                    
                    # For paid subscriptions, update paid dates
                    if not is_trial:
                        existing_product_sub.paid_start_date = start_date
                        existing_product_sub.paid_end_date = end_date
                else:
                    # Create new product subscription
                    chatai_key = None
                    if product_code == ProductCode.CHATAI:
                        chatai_key = f"chatai_{user_email.lower().replace('@', '_').replace('.', '_')}_{int(now.timestamp())}"
                    
                    product_sub = ProductSubscription(
                        id=f"ps_{user_email.lower().replace('@', '_').replace('.', '_')}_{product_code.value}_{int(now.timestamp())}",
                        user_email=user_email.lower(),
                        product_code=product_code,
                        subscription_type=SubscriptionType.TRIAL if is_trial else SubscriptionType.PAID,
                        status=SubscriptionStatus.TRIAL if is_trial else SubscriptionStatus.ACTIVE,
                        plan_code=str(plan_code),
                        trial_start_date=start_date if is_trial else None,
                        trial_end_date=end_date if is_trial else None,
                        trial_duration_days=extension_days if is_trial else 0,
                        paid_start_date=start_date if not is_trial else None,
                        paid_end_date=end_date if not is_trial else None,
                        is_bundle_subscription=is_bundle,
                        chatai_key=chatai_key,
                        total_tokens=0,
                        used_tokens=0,
                        created_at=now,
                        updated_at=now
                    )
                    session.add(product_sub)
            
            session.commit()
            session.refresh(subscription)
            return subscription
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def activate_paid_subscription(self, user_email: str, plan_code: int, 
                                   payment_id: str = None, payment_status: str = None) -> Subscription:
        """Activate paid subscription and update product_subscriptions accordingly"""
        session = self.get_session()
        try:
            # Validate plan_code
            plan_info = self.validate_plan_code(plan_code)
            if not plan_info["valid"]:
                raise ValueError(plan_info["message"])
            
            # Get products for this plan_code
            plan_product_codes = self.get_products_for_plan_code(plan_code)
            is_bundle = len(plan_product_codes) > 1
            
            # Update main subscription
            subscription = session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
            
            extension_days = plan_info["extension_days"]
            now = datetime.now(timezone.utc)
            start_date = now
            end_date = now + timedelta(days=extension_days)
            
            if subscription:
                subscription.plan_code = plan_code
                subscription.subscription_start_date = start_date
                subscription.subscription_end_date = end_date
                subscription.is_trial = False
                subscription.updated_at = now
            else:
                # Create new subscription if doesn't exist
                subscription = Subscription(
                    user_email=user_email.lower(),
                    plan_code=plan_code,
                    subscription_start_date=start_date,
                    subscription_end_date=end_date,
                    is_trial=False,
                    created_at=now,
                    updated_at=now
                )
                session.add(subscription)
            
            # Get all existing product subscriptions for this user
            all_user_products = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower()
            ).all()
            
            # Update products IN the plan to PAID/ACTIVE
            for product_code in plan_product_codes:
                existing_product_sub = session.query(ProductSubscription).filter(
                    ProductSubscription.user_email == user_email.lower(),
                    ProductSubscription.product_code == product_code
                ).first()
                
                if existing_product_sub:
                    # Update to PAID
                    # Preserve trial_start_date and trial_end_date (historical data) - don't overwrite
                    existing_product_sub.subscription_type = SubscriptionType.PAID
                    existing_product_sub.status = SubscriptionStatus.ACTIVE
                    existing_product_sub.plan_code = str(plan_code)
                    existing_product_sub.paid_start_date = start_date
                    existing_product_sub.paid_end_date = end_date
                    existing_product_sub.is_bundle_subscription = is_bundle
                    existing_product_sub.payment_id = payment_id
                    existing_product_sub.payment_status = payment_status
                    existing_product_sub.updated_at = now
                    # Note: trial_start_date and trial_end_date are preserved (not overwritten)
                else:
                    # Create new PAID subscription
                    chatai_key = None
                    if product_code == ProductCode.CHATAI:
                        chatai_key = f"chatai_{user_email.lower().replace('@', '_').replace('.', '_')}_{int(now.timestamp())}"
                    
                    product_sub = ProductSubscription(
                        id=f"ps_{user_email.lower().replace('@', '_').replace('.', '_')}_{product_code.value}_{int(now.timestamp())}",
                        user_email=user_email.lower(),
                        product_code=product_code,
                        subscription_type=SubscriptionType.PAID,
                        status=SubscriptionStatus.ACTIVE,
                        plan_code=str(plan_code),
                        paid_start_date=start_date,
                        paid_end_date=end_date,
                        is_bundle_subscription=is_bundle,
                        payment_id=payment_id,
                        payment_status=payment_status,
                        chatai_key=chatai_key,
                        total_tokens=0,
                        used_tokens=0,
                        created_at=now,
                        updated_at=now
                    )
                    session.add(product_sub)
            
            # For products NOT in the plan, check and update status if needed
            plan_product_code_set = {pc.value for pc in plan_product_codes}
            for existing_product_sub in all_user_products:
                if existing_product_sub.product_code.value not in plan_product_code_set:
                    # Product not in new plan
                    # If it's ACTIVE and paid_end_date has passed, mark as EXPIRED
                    if (existing_product_sub.status == SubscriptionStatus.ACTIVE and 
                        existing_product_sub.paid_end_date and 
                        existing_product_sub.paid_end_date < now):
                        existing_product_sub.status = SubscriptionStatus.EXPIRED
                        existing_product_sub.updated_at = now
                    # Keep TRIAL records as-is (they expire naturally)
            
            session.commit()
            session.refresh(subscription)
            return subscription
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def init_database(self):
        """Initialize database tables"""
        try:
            init_neon_database()
            print("Subscription database tables initialized successfully")
        except Exception as e:
            print(f"Error initializing subscription database: {e}")

class ProductSubscriptionManager:
    """Database manager for product-specific subscription operations using Neon PostgreSQL"""
    
    def __init__(self):
        # Initialize Neon database connection
        if not create_connection():
            raise RuntimeError("Failed to connect to Neon database")
        
        # Create tables if they don't exist
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            init_neon_database()
        except Exception as e:
            print(f"Warning: Could not initialize all tables: {e}")
    
    def get_session(self) -> Session:
        """Get database session from Neon connection"""
        return get_neon_session()
    
    def close_session(self, session: Session):
        """Close database session"""
        session.close()
    
    def create_product_subscription(self, user_email: str, product_code: ProductCode, 
                                  subscription_type: SubscriptionType = SubscriptionType.TRIAL,
                                  plan_code: str = "FREE", trial_days: int = 7, 
                                  total_tokens: int = 0) -> ProductSubscription:
        """Create a new product subscription"""
        session = self.get_session()
        try:
            # Check if user already has a subscription for this product
            existing = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower(),
                ProductSubscription.product_code == product_code
            ).first()
            
            if existing:
                return existing
            
            # Calculate dates
            now = datetime.now(timezone.utc)
            trial_start = now if subscription_type == SubscriptionType.TRIAL else None
            trial_end = now + timedelta(days=trial_days) if subscription_type == SubscriptionType.TRIAL else None
            
            # Generate ChatAI key for CHATAI product
            chatai_key = None
            if product_code == ProductCode.CHATAI:
                chatai_key = f"chatai_{user_email.lower().replace('@', '_').replace('.', '_')}_{int(now.timestamp())}"
            
            subscription = ProductSubscription(
                id=f"ps_{user_email.lower().replace('@', '_').replace('.', '_')}_{product_code.value}_{int(now.timestamp())}",
                user_email=user_email.lower(),
                product_code=product_code,
                subscription_type=subscription_type,
                status=SubscriptionStatus.TRIAL if subscription_type == SubscriptionType.TRIAL else SubscriptionStatus.ACTIVE,
                plan_code=plan_code,
                trial_start_date=trial_start,
                trial_end_date=trial_end,
                trial_duration_days=trial_days,
                is_bundle_subscription=False,
                chatai_key=chatai_key,
                total_tokens=total_tokens,
                used_tokens=0
            )
            
            session.add(subscription)
            session.commit()
            session.refresh(subscription)
            return subscription
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def create_bundle_subscription(self, user_email: str, product_codes: List[ProductCode], 
                                 plan_code: str = "BUNDLE") -> List[ProductSubscription]:
        """Create a bundle subscription for multiple products"""
        session = self.get_session()
        try:
            bundle_id = str(uuid.uuid4())
            subscriptions = []
            now = datetime.now(timezone.utc)
            
            for product_code in product_codes:
                # Check if user already has a subscription for this product
                existing = session.query(ProductSubscription).filter(
                    ProductSubscription.user_email == user_email.lower(),
                    ProductSubscription.product_code == product_code
                ).first()
                
                if existing:
                    # Update existing subscription to be part of bundle
                    existing.bundle_id = bundle_id
                    existing.is_bundle_subscription = True
                    existing.plan_code = plan_code
                    subscriptions.append(existing)
                else:
                    subscription = ProductSubscription(
                        id=f"ps_{user_email.lower().replace('@', '_').replace('.', '_')}_{product_code.value}_{int(now.timestamp())}",
                        user_email=user_email.lower(),
                        product_code=product_code,
                        subscription_type=SubscriptionType.BUNDLE,
                        status=SubscriptionStatus.ACTIVE,
                        plan_code=plan_code,
                        paid_start_date=now,
                        paid_end_date=now + timedelta(days=365),  # 1 year
                        bundle_id=bundle_id,
                        is_bundle_subscription=True
                    )
                    session.add(subscription)
                    subscriptions.append(subscription)
            
            session.commit()
            for sub in subscriptions:
                session.refresh(sub)
            return subscriptions
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def check_product_access(self, user_email: str, product_code: ProductCode) -> Dict[str, Any]:
        """Check if user has access to a specific product"""
        session = self.get_session()
        try:
            # First, check product_subscriptions table
            product_subscription = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower(),
                ProductSubscription.product_code == product_code
            ).first()
            
            if product_subscription:
                # Product subscription exists, use it
                now = datetime.now(timezone.utc)
                is_trial_active = (
                    product_subscription.status == SubscriptionStatus.TRIAL and 
                    product_subscription.trial_end_date and
                    product_subscription.trial_end_date > now
                )
                
                is_paid_active = (
                    product_subscription.status == SubscriptionStatus.ACTIVE and
                    product_subscription.paid_end_date and
                    product_subscription.paid_end_date > now
                )
                
                has_access = is_trial_active or is_paid_active
                
                # Calculate days remaining
                if is_trial_active:
                    days_remaining = (product_subscription.trial_end_date - now).days
                    access_type = "trial"
                elif is_paid_active:
                    days_remaining = (product_subscription.paid_end_date - now).days
                    access_type = "paid"
                else:
                    days_remaining = 0
                    access_type = "expired"
                
                return {
                    "has_access": has_access,
                    "access_type": access_type,
                    "days_remaining": days_remaining,
                    "trial_end_date": product_subscription.trial_end_date,
                    "paid_end_date": product_subscription.paid_end_date,
                    "status": product_subscription.status.value,
                    "plan": product_subscription.plan_code,
                    "is_bundle": product_subscription.is_bundle_subscription
                }
            
            # No product_subscription found, check main subscription table
            main_subscription = session.query(Subscription).filter(
                Subscription.user_email == user_email.lower()
            ).first()
            
            if not main_subscription or not main_subscription.plan_code:
                return {
                    "has_access": False,
                    "access_type": "none",
                    "days_remaining": 0,
                    "status": "no_subscription",
                    "trial_end_date": None,
                    "paid_end_date": None,
                    "plan": None,
                    "is_bundle": False
                }
            
            # Check if the plan_code includes this product
            # Use SubscriptionManager to get products for plan_code
            subscription_mgr = SubscriptionManager()
            plan_products = subscription_mgr.get_products_for_plan_code(int(main_subscription.plan_code))
            plan_product_codes = {pc.value for pc in plan_products}
            
            # Check subscription dates first
            now = datetime.now(timezone.utc)
            is_active = (
                main_subscription.subscription_end_date and
                main_subscription.subscription_end_date > now
            )
            
            # If no products found in plan_prod_mapping, but subscription exists and is active,
            # grant access (fallback for cases where mapping might be missing)
            if not plan_products:
                # No products found in mapping - this shouldn't happen but handle gracefully
                # If subscription is active, grant access
                if is_active:
                    days_remaining = (main_subscription.subscription_end_date - now).days
                    access_type = "trial" if main_subscription.is_trial else "paid"
                    return {
                        "has_access": True,
                        "access_type": access_type,
                        "days_remaining": days_remaining,
                        "trial_end_date": main_subscription.subscription_end_date if main_subscription.is_trial else None,
                        "paid_end_date": main_subscription.subscription_end_date if not main_subscription.is_trial else None,
                        "status": SubscriptionStatus.TRIAL.value if main_subscription.is_trial else SubscriptionStatus.ACTIVE.value,
                        "plan": str(main_subscription.plan_code),
                        "is_bundle": False
                    }
                else:
                    return {
                        "has_access": False,
                        "access_type": "expired",
                        "days_remaining": 0,
                        "status": "expired",
                        "trial_end_date": main_subscription.subscription_end_date if main_subscription.is_trial else None,
                        "paid_end_date": main_subscription.subscription_end_date if not main_subscription.is_trial else None,
                        "plan": str(main_subscription.plan_code),
                        "is_bundle": False
                    }
            
            # Check if product is in the plan
            if product_code.value not in plan_product_codes:
                # Product not in the plan
                return {
                    "has_access": False,
                    "access_type": "none",
                    "days_remaining": 0,
                    "status": "expired",
                    "trial_end_date": None,
                    "paid_end_date": None,
                    "plan": str(main_subscription.plan_code),
                    "is_bundle": len(plan_products) > 1
                }
            
            # Product is in the plan, check subscription dates
            if not is_active:
                return {
                    "has_access": False,
                    "access_type": "expired",
                    "days_remaining": 0,
                    "status": "expired",
                    "trial_end_date": main_subscription.subscription_end_date if main_subscription.is_trial else None,
                    "paid_end_date": main_subscription.subscription_end_date if not main_subscription.is_trial else None,
                    "plan": str(main_subscription.plan_code),
                    "is_bundle": len(plan_products) > 1
                }
            
            # Subscription is active and product is in plan, calculate access
            days_remaining = max(0, (main_subscription.subscription_end_date - now).days)
            access_type = "trial" if main_subscription.is_trial else "paid"
            
            return {
                "has_access": True,
                "access_type": access_type,
                "days_remaining": days_remaining,
                "trial_end_date": main_subscription.subscription_end_date if main_subscription.is_trial else None,
                "paid_end_date": main_subscription.subscription_end_date if not main_subscription.is_trial else None,
                "status": SubscriptionStatus.TRIAL.value if main_subscription.is_trial else SubscriptionStatus.ACTIVE.value,
                "plan": str(main_subscription.plan_code),
                "is_bundle": len(plan_products) > 1
            }
            
        finally:
            self.close_session(session)
    
    def get_all_user_products(self, user_email: str) -> Dict[str, Dict[str, Any]]:
        """Get access status for all products for a user"""
        products = {}
        for product_code in ProductCode:
            access_info = self.check_product_access(user_email, product_code)
            products[product_code.value] = access_info
        
        return products
    
    def upgrade_product_subscription(self, user_email: str, product_code: ProductCode, 
                                   plan_code: str) -> bool:
        """Upgrade a product subscription to paid plan"""
        session = self.get_session()
        try:
            subscription = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower(),
                ProductSubscription.product_code == product_code
            ).first()
            
            if subscription:
                now = datetime.now(timezone.utc)
                subscription.subscription_type = SubscriptionType.PAID
                subscription.status = SubscriptionStatus.ACTIVE
                subscription.plan_code = plan_code
                subscription.paid_start_date = now
                subscription.paid_end_date = now + timedelta(days=365)  # 1 year
                subscription.updated_at = now
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def log_product_access(self, user_email: str, product_code: ProductCode, 
                          access_granted: bool, access_type: str = None, 
                          subscription_id: str = None, ip_address: str = None, 
                          user_agent: str = None) -> None:
        """Log product access attempt"""
        session = self.get_session()
        try:
            log_entry = ProductAccessLog(
                id=str(uuid.uuid4()),
                user_email=user_email.lower(),
                product_code=product_code,
                access_granted=access_granted,
                access_type=access_type,
                subscription_id=subscription_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def update_token_usage(self, user_email: str, product_code: ProductCode, tokens_used: int) -> bool:
        """Update token usage for a product subscription"""
        session = self.get_session()
        try:
            subscription = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower(),
                ProductSubscription.product_code == product_code
            ).first()
            
            if subscription:
                subscription.used_tokens += tokens_used
                subscription.updated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def set_total_tokens(self, user_email: str, product_code: ProductCode, total_tokens: int) -> bool:
        """Set total tokens for a product subscription"""
        session = self.get_session()
        try:
            subscription = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower(),
                ProductSubscription.product_code == product_code
            ).first()
            
            if subscription:
                subscription.total_tokens = total_tokens
                subscription.updated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_token_usage(self, user_email: str, product_code: ProductCode) -> Dict[str, Any]:
        """Get token usage information for a product subscription"""
        session = self.get_session()
        try:
            subscription = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower(),
                ProductSubscription.product_code == product_code
            ).first()
            
            if not subscription:
                return {
                    "has_subscription": False,
                    "total_tokens": 0,
                    "used_tokens": 0,
                    "remaining_tokens": 0,
                    "chatai_key": None
                }
            
            remaining_tokens = max(0, subscription.total_tokens - subscription.used_tokens)
            
            return {
                "has_subscription": True,
                "total_tokens": subscription.total_tokens,
                "used_tokens": subscription.used_tokens,
                "remaining_tokens": remaining_tokens,
                "chatai_key": subscription.chatai_key,
                "status": subscription.status.value,
                "plan_code": subscription.plan_code
            }
            
        finally:
            self.close_session(session)
    
    def reset_token_usage(self, user_email: str, product_code: ProductCode) -> bool:
        """Reset token usage for a product subscription"""
        session = self.get_session()
        try:
            subscription = session.query(ProductSubscription).filter(
                ProductSubscription.user_email == user_email.lower(),
                ProductSubscription.product_code == product_code
            ).first()
            
            if subscription:
                subscription.used_tokens = 0
                subscription.updated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)

# Global subscription manager instances
subscription_manager = SubscriptionManager()
product_subscription_manager = ProductSubscriptionManager()

