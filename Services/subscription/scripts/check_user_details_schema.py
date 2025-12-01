"""
Check current user_details table schema
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from Databases.app_data_db_connection import create_connection, get_engine
from sqlalchemy import text

def check_table_schema():
    """Check the current schema of user_details table"""
    
    print("Checking user_details table schema...")
    
    # Create database connection
    if not create_connection():
        print("ERROR: Failed to connect to database")
        return False
    
    engine = get_engine()
    
    try:
        with engine.connect() as connection:
            # Get all columns from user_details table
            result = connection.execute(text("""
                SELECT column_name, data_type, character_maximum_length, column_default
                FROM information_schema.columns 
                WHERE table_name='user_details'
                ORDER BY ordinal_position
            """))
            
            columns = result.fetchall()
            
            if not columns:
                print("WARNING: user_details table does not exist!")
                return False
            
            print("\nCurrent columns in user_details table:")
            print("-" * 80)
            for col in columns:
                col_name, data_type, max_length, default = col
                length_str = f"({max_length})" if max_length else ""
                default_str = f" DEFAULT {default}" if default else ""
                print(f"  {col_name:<20} {data_type}{length_str:<20} {default_str}")
            print("-" * 80)
            
            # Check if our columns exist
            column_names = [col[0] for col in columns]
            phone_no_exists = 'phone_no' in column_names
            status_exists = 'status' in column_names
            
            print(f"\nphone_no exists: {phone_no_exists}")
            print(f"status exists: {status_exists}")
            
            return True
                
    except Exception as e:
        print(f"\nError checking schema: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*80)
    print("User Details Table Schema Check")
    print("="*80)
    print()
    
    check_table_schema()
    
    print()
    print("="*80)
