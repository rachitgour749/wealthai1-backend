import sys
import os
import asyncio
from unittest.mock import MagicMock
from datetime import datetime, timedelta, timezone
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Add current directory to path
sys.path.append(os.getcwd())

# --- MOCKING DATABASE CONNECTION ---
# We mock the database connection module to use an in-memory SQLite database
# This ensures we don't need a real Neon DB connection and tests are isolated

# Create a real Base for SQLAlchemy to use with SQLite
TestBase = declarative_base()

# Mock the module structure
mock_db_module = MagicMock()
mock_db_module.Base = TestBase
mock_db_module.create_connection = lambda: True
mock_db_module.get_session = MagicMock()
mock_db_module.init_database = MagicMock()

# Inject mocks into sys.modules
sys.modules["Databases"] = MagicMock()
sys.modules["Databases.app_data_db_connection"] = mock_db_module
sys.modules["app_data_db_connection"] = mock_db_module

# --- IMPORT SERVICES ---
# Now we can import the services which will use our mocked module
from Services.subscription import database
from Services.subscription.models import (
    ProductCode, SubscriptionType, SubscriptionStatus, 
    SubscriptionRequest, SubscriptionPlan
)
from Services.subscription.service import SubscriptionService

# --- SETUP SQLITE ---
engine = create_engine("sqlite:///:memory:")
# Create tables
database.Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Patch get_session methods to return our SQLite session
def mock_get_session():
    return Session()

# Patch the methods on the instances or classes
database.get_neon_session = mock_get_session
database.SubscriptionManager.get_session = lambda self: Session()

# Also need to populate plan_master for validation
def populate_plan_master():
    session = Session()
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS plan_master (
            plan_code INTEGER PRIMARY KEY,
            plan_name TEXT,
            extension_days INTEGER,
            description TEXT
        )
    """))
    session.execute(text("""
        INSERT INTO plan_master (plan_code, plan_name, extension_days, description) VALUES 
        (6, 'Premium Plan', 30, 'Premium Monthly Subscription')
    """))
    
    # Also need plan_prod_mapping
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS plan_prod_mapping (
            plan_code INTEGER,
            prod_code INTEGER
        )
    """))
    # Map plan 6 to TRADAI (1) and MARKETAI (2)
    session.execute(text("INSERT INTO plan_prod_mapping (plan_code, prod_code) VALUES (6, 1)"))
    session.execute(text("INSERT INTO plan_prod_mapping (plan_code, prod_code) VALUES (6, 2)"))
    
    session.commit()
    session.close()

populate_plan_master()

# --- VERIFICATION TESTS ---

async def verify_new_user_flow():
    print("\n--- Verifying New User Flow ---")
    service = SubscriptionService()
    
    email = "newuser@example.com"
    name = "New User"
    
    # 1. Create Subscription (New User Trial)
    print(f"Creating subscription for {email}...")
    try:
        response = await service.create_subscription(SubscriptionRequest(
            user_email=email,
            user_name=name
        ))
        print("Success: Subscription created.")
        print(f"Response Status: {response.status}")
        print(f"Trial Active: {response.is_trial_active}")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # 2. Verify Database State
    session = Session()
    
    # Check UserDetails
    user = session.query(database.UserDetails).filter_by(user_email=email).first()
    if user:
        print(f"Verified: User {user.user_email} exists in users_details.")
    else:
        print("FAILED: User not found in users_details.")
        
    # Check ProductManager (Should have 4 products)
    products = session.query(database.ProductManager).filter_by(user_email=email).all()
    print(f"Found {len(products)} products for user.")
    
    expected_products = {ProductCode.MARKETAI, ProductCode.CHATAI, ProductCode.TRADAI, ProductCode.AUTOMATIONAI}
    found_products = {p.product_code for p in products}
    
    if expected_products == found_products:
        print("Verified: All 4 products created.")
    else:
        print(f"FAILED: Expected {expected_products}, found {found_products}")
        
    # Check Status (Should be TRIAL)
    all_trial = all(p.subscription_type == SubscriptionType.TRIAL for p in products)
    if all_trial:
        print("Verified: All products are in TRIAL mode.")
    else:
        print("FAILED: Some products are not in TRIAL mode.")
        
    session.close()

async def verify_payment_flow():
    print("\n--- Verifying Payment Flow (Existing User) ---")
    service = SubscriptionService()
    email = "newuser@example.com" # Use same user
    plan_code = 6 # Premium Plan (mapped to TRADAI and MARKETAI)
    
    # 1. Process Payment
    print(f"Processing payment for {email} with plan_code {plan_code}...")
    try:
        response = await service.activate_paid_subscription(
            user_email=email,
            plan_code=plan_code
        )
        print("Success: Payment processed.")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # 2. Verify Database State
    session = Session()
    products = session.query(database.ProductManager).filter_by(user_email=email).all()
    
    # Check TRADAI and MARKETAI should be PAID
    paid_products = [p for p in products if p.subscription_type == SubscriptionType.PAID]
    trial_products = [p for p in products if p.subscription_type == SubscriptionType.TRIAL]
    
    paid_codes = {p.product_code for p in paid_products}
    trial_codes = {p.product_code for p in trial_products}
    
    print(f"Paid Products: {paid_codes}")
    print(f"Trial Products: {trial_codes}")
    
    if ProductCode.TRADAI in paid_codes and ProductCode.MARKETAI in paid_codes:
        print("Verified: TRADAI and MARKETAI updated to PAID.")
    else:
        print("FAILED: TRADAI and MARKETAI not updated correctly.")
        
    if ProductCode.CHATAI in trial_codes and ProductCode.AUTOMATIONAI in trial_codes:
        print("Verified: CHATAI and AUTOMATIONAI remain in TRIAL.")
    else:
        print("FAILED: Other products affected incorrectly.")
        
    session.close()

async def main():
    await verify_new_user_flow()
    await verify_payment_flow()

if __name__ == "__main__":
    asyncio.run(main())
