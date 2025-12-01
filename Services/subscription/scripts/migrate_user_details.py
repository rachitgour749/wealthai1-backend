"""
Direct migration script to add phone_no and status columns to user_details table
This script will show detailed output and handle errors properly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from Databases.app_data_db_connection import create_connection, get_engine
from sqlalchemy import text

def migrate_user_details_table():
    """Add phone_no and status columns to user_details table"""
    
    print("\n" + "="*80)
    print("MIGRATION: Adding columns to user_details table")
    print("="*80 + "\n")
    
    # Create database connection
    print("Step 1: Connecting to database...")
    if not create_connection():
        print("❌ ERROR: Failed to connect to database")
        return False
    print("✅ Database connected successfully\n")
    
    engine = get_engine()
    
    try:
        with engine.begin() as connection:
            # First, check current schema
            print("Step 2: Checking current table schema...")
            result = connection.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='user_details'
                ORDER BY ordinal_position
            """))
            
            existing_columns = [row[0] for row in result.fetchall()]
            print(f"   Current columns: {', '.join(existing_columns)}\n")
            
            # Check if columns already exist
            phone_no_exists = 'phone_no' in existing_columns
            status_exists = 'status' in existing_columns
            
            # Add phone_no column if it doesn't exist
            print("Step 3: Adding phone_no column...")
            if not phone_no_exists:
                connection.execute(text("""
                    ALTER TABLE user_details 
                    ADD COLUMN phone_no VARCHAR(50)
                """))
                connection.commit()
                print("   ✅ phone_no column added successfully")
            else:
                print("   ℹ️  phone_no column already exists (skipping)")
            
            # Add status column if it doesn't exist
            print("\nStep 4: Adding status column...")
            if not status_exists:
                connection.execute(text("""
                    ALTER TABLE user_details 
                    ADD COLUMN status VARCHAR(20) DEFAULT 'TRIAL'
                """))
                connection.commit()
                print("   ✅ status column added successfully")
            else:
                print("   ℹ️  status column already exists (skipping)")
            
            # Verify the changes
            print("\nStep 5: Verifying changes...")
            result = connection.execute(text("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name='user_details'
                ORDER BY ordinal_position
            """))
            
            print("   Final table schema:")
            for row in result.fetchall():
                col_name, data_type, max_len = row
                length_str = f"({max_len})" if max_len else ""
                print(f"     - {col_name}: {data_type}{length_str}")
            
            print("\n" + "="*80)
            print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
            print("="*80 + "\n")
            return True
                
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED!")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80 + "\n")
        return False

if __name__ == "__main__":
    success = migrate_user_details_table()
    sys.exit(0 if success else 1)
