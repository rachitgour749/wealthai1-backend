"""Script to normalize roles to uppercase in user_details table"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import text
from Databases.app_data_db_connection import get_db

def normalize_roles():
    print("Normalizing roles in user_details table...")
    from Databases.app_data_db_connection import create_connection
    if not create_connection():
        print("Failed to connect to database")
        return
    db = next(get_db())
    try:
        # Update all roles to uppercase
        result = db.execute(text("""
            UPDATE user_details 
            SET role = UPPER(role) 
            WHERE role IS NOT NULL AND role != UPPER(role)
        """))
        db.commit()
        print(f"Successfully normalized {result.rowcount} rows.")
        
        # Also update any 'user' roles to 'CLIENT'
        result = db.execute(text("""
            UPDATE user_details 
            SET role = 'CLIENT' 
            WHERE role = 'USER' OR role IS NULL
        """))
        db.commit()
        print(f"Successfully updated {result.rowcount} undefined/user roles to CLIENT.")
        
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    normalize_roles()
