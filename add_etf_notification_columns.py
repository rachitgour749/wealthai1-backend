"""
Manually add notification columns to etf_saved_strategy table
"""

from Databases.app_data_db_connection import create_connection, get_session
from sqlalchemy import text

def add_columns_to_etf_table():
    print("=" * 60)
    print("Adding Notification Columns to etf_saved_strategy")
    print("=" * 60)
    
    if not create_connection():
        print("❌ Failed to connect to database")
        return False
    
    session = get_session()
    
    try:
        # Check if columns already exist
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns
            WHERE table_name = 'etf_saved_strategy'
            AND column_name IN ('email_notification', 'telegram_notification')
        """))
        
        existing_columns = [row[0] for row in result.fetchall()]
        
        columns_to_add = []
        if 'email_notification' not in existing_columns:
            columns_to_add.append('email_notification')
        if 'telegram_notification' not in existing_columns:
            columns_to_add.append('telegram_notification')
        
        if not columns_to_add:
            print("\n✅ All columns already exist!")
            return True
        
        print(f"\n📝 Adding columns: {', '.join(columns_to_add)}")
        
        # Add missing columns
        for column_name in columns_to_add:
            try:
                session.execute(text(f"""
                    ALTER TABLE etf_saved_strategy 
                    ADD COLUMN {column_name} BOOLEAN DEFAULT FALSE NOT NULL
                """))
                session.commit()
                print(f"   ✅ Added {column_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"   ℹ️  {column_name} already exists")
                else:
                    print(f"   ❌ Error adding {column_name}: {e}")
                    session.rollback()
                    return False
        
        print("\n" + "=" * 60)
        print("🎉 Columns added successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    success = add_columns_to_etf_table()
    exit(0 if success else 1)
