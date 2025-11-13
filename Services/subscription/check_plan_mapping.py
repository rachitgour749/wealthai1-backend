#!/usr/bin/env python3
"""
Script to check plan_prod_mapping data for plan_code 8
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
databases_path = os.path.join(project_root, 'Databases')
sys.path.insert(0, databases_path)
sys.path.insert(0, project_root)

from neon_db_connection import create_connection, get_engine
from sqlalchemy import text

def check_plan_mapping():
    """Check plan_prod_mapping data"""
    print("=" * 70)
    print("CHECKING plan_prod_mapping FOR plan_code 8")
    print("=" * 70)
    
    if not create_connection():
        print("[ERROR] Failed to connect to database!")
        sys.exit(1)
    
    engine = get_engine()
    
    with engine.connect() as conn:
        # First, check the structure of plan_prod_mapping table
        print("\nChecking plan_prod_mapping table structure...")
        result = conn.execute(
            text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'plan_prod_mapping'
                ORDER BY ordinal_position
            """)
        )
        columns = result.fetchall()
        print("Columns in plan_prod_mapping:")
        for col in columns:
            print(f"  - {col[0]}: {col[1]}")
        
        # Check plan_prod_mapping for plan_code 8 (using actual column names)
        print("\n" + "-" * 70)
        print("Checking data for plan_code 8...")
        result = conn.execute(
            text("SELECT * FROM plan_prod_mapping WHERE plan_code = 8")
        )
        rows = result.fetchall()
        
        print(f"\nFound {len(rows)} entries for plan_code 8:")
        print("-" * 70)
        if rows:
            for row in rows:
                print(f"  plan_code: {row[0]}, prod_code: {row[1]}")
        else:
            print("  [WARNING] No entries found for plan_code 8!")
        
        # Check all plan_prod_mapping entries
        print("\n" + "-" * 70)
        print("All plan_prod_mapping entries:")
        print("-" * 70)
        result = conn.execute(
            text("SELECT * FROM plan_prod_mapping ORDER BY plan_code")
        )
        all_rows = result.fetchall()
        if all_rows:
            for row in all_rows:
                print(f"  plan_code: {row[0]}, prod_code: {row[1]}")
        else:
            print("  [WARNING] plan_prod_mapping table is empty!")
        
        # Check prod_master structure
        print("\n" + "-" * 70)
        print("prod_master table structure:")
        print("-" * 70)
        result = conn.execute(
            text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'prod_master'
                ORDER BY ordinal_position
            """)
        )
        prod_cols = result.fetchall()
        for col in prod_cols:
            print(f"  - {col[0]}: {col[1]}")
        
        # Check prod_master for available products
        print("\n" + "-" * 70)
        print("Available products in prod_master:")
        print("-" * 70)
        result = conn.execute(
            text("SELECT * FROM prod_master ORDER BY prod_id")
        )
        products = result.fetchall()
        if products:
            for prod in products:
                print(f"  product_id: {prod[0]}, product_code: {prod[1]}, product_name: {prod[2]}")
        else:
            print("  [WARNING] prod_master table is empty!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_plan_mapping()

