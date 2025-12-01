#!/usr/bin/env python3
"""
Backfill script to create product_subscriptions for users who have subscriptions
but no product_subscriptions records
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
databases_path = os.path.join(project_root, 'Databases')
sys.path.insert(0, databases_path)
sys.path.insert(0, project_root)

from neon_db_connection import create_connection, get_engine
from sqlalchemy import text
from datetime import datetime, timezone

def backfill_product_subscriptions():
    """Backfill product_subscriptions for existing subscriptions"""
    print("=" * 70)
    print("BACKFILLING product_subscriptions FOR EXISTING SUBSCRIPTIONS")
    print("=" * 70)
    
    if not create_connection():
        print("[ERROR] Failed to connect to database!")
        sys.exit(1)
    
    engine = get_engine()
    
    with engine.connect() as conn:
        # Get all subscriptions with plan_code
        result = conn.execute(
            text("""
                SELECT user_email, user_name, plan_code, subscription_start_date, 
                       subscription_end_date, is_trial
                FROM subscription
                WHERE plan_code IS NOT NULL
            """)
        )
        subscriptions = result.fetchall()
        
        print(f"\nFound {len(subscriptions)} subscriptions with plan_code")
        
        if not subscriptions:
            print("[INFO] No subscriptions found to backfill")
            return
        
        # Process each subscription
        for sub in subscriptions:
            user_email = sub[0]
            user_name = sub[1]
            plan_code = int(sub[2]) if sub[2] else None
            start_date = sub[3]
            end_date = sub[4]
            is_trial = sub[5]
            
            if not plan_code:
                continue
            
            print(f"\nProcessing: {user_email}, plan_code: {plan_code}")
            
            # Get products for this plan_code
            result = conn.execute(
                text("""
                    SELECT prod_code 
                    FROM plan_prod_mapping
                    WHERE plan_code = :plan_code
                """),
                {"plan_code": plan_code}
            )
            prod_codes = result.fetchall()
            
            # Map numeric prod_code to ProductCode string
            prod_code_mapping = {
                1: "TRADEAI",
                2: "MARKETAI",
                3: "CHATAI",
                4: "AUTOMATIONAI"
            }
            
            products_to_create = []
            for row in prod_codes:
                numeric_code = int(row[0]) if row[0] else None
                if numeric_code and numeric_code in prod_code_mapping:
                    products_to_create.append(prod_code_mapping[numeric_code])
            
            print(f"  Products to create: {products_to_create}")
            
            if not products_to_create:
                print(f"  [WARNING] No products found for plan_code {plan_code}")
                continue
            
            # Check existing product_subscriptions
            result = conn.execute(
                text("""
                    SELECT product_code 
                    FROM product_subscriptions
                    WHERE user_email = :user_email
                """),
                {"user_email": user_email.lower()}
            )
            existing_products = {row[0] for row in result.fetchall()}
            
            # Create missing product_subscriptions
            now = datetime.now(timezone.utc)
            is_bundle = len(products_to_create) > 1
            
            for product_code in products_to_create:
                if product_code in existing_products:
                    print(f"  [SKIP] {product_code} already exists")
                    continue
                
                # Generate ID
                product_sub_id = f"ps_{user_email.lower().replace('@', '_').replace('.', '_')}_{product_code}_{int(now.timestamp())}"
                
                # Generate chatai_key if needed
                chatai_key = None
                if product_code == "CHATAI":
                    chatai_key = f"chatai_{user_email.lower().replace('@', '_').replace('.', '_')}_{int(now.timestamp())}"
                
                # Determine subscription type and status
                subscription_type = "TRIAL" if is_trial else "PAID"
                status = "TRIAL" if is_trial else "ACTIVE"
                
                # Insert product_subscription
                try:
                    conn.execute(
                        text("""
                            INSERT INTO product_subscriptions 
                            (id, user_email, product_code, subscription_type, status, plan_code,
                             trial_start_date, trial_end_date, trial_duration_days,
                             paid_start_date, paid_end_date, is_bundle_subscription,
                             chatai_key, total_tokens, used_tokens, created_at, updated_at)
                            VALUES 
                            (:id, :user_email, :product_code, :subscription_type, :status, :plan_code,
                             :trial_start_date, :trial_end_date, :trial_duration_days,
                             :paid_start_date, :paid_end_date, :is_bundle_subscription,
                             :chatai_key, :total_tokens, :used_tokens, :created_at, :updated_at)
                        """),
                        {
                            "id": product_sub_id,
                            "user_email": user_email.lower(),
                            "product_code": product_code,
                            "subscription_type": subscription_type,
                            "status": status,
                            "plan_code": str(plan_code),
                            "trial_start_date": start_date if is_trial else None,
                            "trial_end_date": end_date if is_trial else None,
                            "trial_duration_days": (end_date - start_date).days if is_trial and start_date and end_date else 0,
                            "paid_start_date": start_date if not is_trial else None,
                            "paid_end_date": end_date if not is_trial else None,
                            "is_bundle_subscription": is_bundle,
                            "chatai_key": chatai_key,
                            "total_tokens": 0,
                            "used_tokens": 0,
                            "created_at": now,
                            "updated_at": now
                        }
                    )
                    conn.commit()
                    print(f"  [SUCCESS] Created {product_code}")
                except Exception as e:
                    print(f"  [ERROR] Failed to create {product_code}: {e}")
                    conn.rollback()
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Backfill completed!")
    print("=" * 70)

if __name__ == "__main__":
    backfill_product_subscriptions()

