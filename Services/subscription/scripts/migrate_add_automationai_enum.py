#!/usr/bin/env python3
"""
Migration script to add AUTOMATIONAI to the productcode enum in PostgreSQL
This script adds 'AUTOMATIONAI' to the existing productcode enum type.
"""

import sys
import os

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
databases_path = os.path.join(project_root, 'Databases')
sys.path.insert(0, databases_path)
sys.path.insert(0, project_root)

from neon_db_connection import create_connection, get_engine
from sqlalchemy import text

def check_enum_value_exists(engine, enum_name, value):
    """Check if a value exists in an enum type"""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM pg_enum 
                    WHERE enumlabel = :value 
                    AND enumtypid = (
                        SELECT oid 
                        FROM pg_type 
                        WHERE typname = :enum_name
                    )
                )
            """),
            {"value": value, "enum_name": enum_name}
        )
        return result.fetchone()[0]

def add_enum_value(engine, enum_name, value):
    """Add a value to an existing enum type"""
    with engine.connect() as conn:
        try:
            # Check if value already exists
            if check_enum_value_exists(engine, enum_name, value):
                print(f"[OK] Value '{value}' already exists in enum '{enum_name}'")
                return True
            
            # Add the new value to the enum
            conn.execute(
                text(f"ALTER TYPE {enum_name} ADD VALUE IF NOT EXISTS '{value}'")
            )
            conn.commit()
            print(f"[SUCCESS] Successfully added '{value}' to enum '{enum_name}'")
            return True
        except Exception as e:
            print(f"[ERROR] Error adding '{value}' to enum '{enum_name}': {e}")
            conn.rollback()
            return False

def get_enum_values(engine, enum_name):
    """Get all values in an enum type"""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT enumlabel 
                FROM pg_enum 
                WHERE enumtypid = (
                    SELECT oid 
                    FROM pg_type 
                    WHERE typname = :enum_name
                )
                ORDER BY enumsortorder
            """),
            {"enum_name": enum_name}
        )
        return [row[0] for row in result.fetchall()]

def main():
    """Main migration function"""
    print("=" * 70)
    print("MIGRATION: Add AUTOMATIONAI to productcode enum")
    print("=" * 70)
    
    # Initialize connection
    if not create_connection():
        print("[ERROR] Failed to connect to database!")
        sys.exit(1)
    
    engine = get_engine()
    print("\n[OK] Connected to database")
    
    # Check current enum values
    print("\n" + "-" * 70)
    print("Current productcode enum values:")
    print("-" * 70)
    try:
        current_values = get_enum_values(engine, 'productcode')
        for value in current_values:
            print(f"  - {value}")
    except Exception as e:
        print(f"[WARNING] Could not retrieve enum values: {e}")
        print("   This might mean the enum doesn't exist yet or has a different name")
    
    # Add AUTOMATIONAI to the enum
    print("\n" + "-" * 70)
    print("Adding AUTOMATIONAI to productcode enum...")
    print("-" * 70)
    
    success = add_enum_value(engine, 'productcode', 'AUTOMATIONAI')
    
    if success:
        # Verify the addition
        print("\n" + "-" * 70)
        print("Updated productcode enum values:")
        print("-" * 70)
        try:
            updated_values = get_enum_values(engine, 'productcode')
            for value in updated_values:
                marker = "  [OK]" if value == 'AUTOMATIONAI' else "  -"
                print(f"{marker} {value}")
        except Exception as e:
            print(f"[WARNING] Could not verify enum values: {e}")
    
    print("\n" + "=" * 70)
    if success:
        print("[SUCCESS] Migration completed successfully!")
        print("\nThe productcode enum now includes:")
        print("  - MARKETAI")
        print("  - CHATAI")
        print("  - TRADAI")
        print("  - AUTOMATIONAI")
    else:
        print("[ERROR] Migration failed!")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    main()

