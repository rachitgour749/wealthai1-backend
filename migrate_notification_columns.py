"""
Initialize/Migrate database tables with notification columns
"""

import sys
sys.path.insert(0, '.')

from Services.Deployments_helper.deployment_helper import init_rs_etf_instance_table
from Databases.app_data_db_connection import create_connection, init_database

def migrate_tables():
    print("=" * 60)
    print("Migrating Database Tables")
    print("=" * 60)
    
    # Step 1: Connect to database
    print("\n1. Connecting to PostgreSQL database...")
    if not create_connection():
        print("   ❌ Failed to connect to database")
        return False
    print("   ✅ Connected successfully")
    
    # Step 2: Initialize base tables (includes etf_saved_strategy from SQLAlchemy models)
    print("\n2. Initializing base tables from SQLAlchemy models...")
    if not init_database():
        print("   ❌ Failed to initialize database")
        return False
    print("   ✅ Base tables initialized (etf_saved_strategy updated)")
    
    # Step 3: Initialize/migrate rs_etf_instance table
    print("\n3. Initializing/migrating rs_etf_instance table...")
    if not init_rs_etf_instance_table():
        print("   ❌ Failed to initialize rs_etf_instance table")
        return False
    print("   ✅ rs_etf_instance table initialized/migrated")
    
    print("\n" + "=" * 60)
    print("🎉 Migration Complete!")
    print("=" * 60)
    print("\nNotification columns added to:")
    print("  • etf_saved_strategy")
    print("  • rs_etf_instance")
    print("\nRun test_notification_columns.py to verify")
    
    return True

if __name__ == "__main__":
    success = migrate_tables()
    exit(0 if success else 1)
