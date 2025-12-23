"""
Test script to verify notification columns were added to database tables
"""

from Databases.app_data_db_connection import create_connection, get_session
from sqlalchemy import text

def test_notification_columns():
    print("=" * 60)
    print("Testing Notification Columns Implementation")
    print("=" * 60)
    
    # Connect to database
    if not create_connection():
        print("❌ Failed to connect to database")
        return False
    
    session = get_session()
    
    try:
        # Check for notification columns in both tables
        result = session.execute(text("""
            SELECT 
                table_name, 
                column_name, 
                data_type, 
                column_default,
                is_nullable
            FROM information_schema.columns 
            WHERE table_name IN ('etf_saved_strategy', 'rs_etf_instance') 
            AND column_name IN ('email_notification', 'telegram_notification')
            ORDER BY table_name, column_name
        """))
        
        rows = result.fetchall()
        
        if not rows:
            print("\n❌ No notification columns found!")
            print("   Run the deployment helper to create/migrate tables")
            return False
        
        print("\n✅ Notification Columns Found:\n")
        print(f"{'Table':<25} {'Column':<25} {'Type':<15} {'Default':<15} {'Nullable'}")
        print("-" * 95)
        
        for row in rows:
            table, column, dtype, default, nullable = row
            print(f"{table:<25} {column:<25} {dtype:<15} {str(default):<15} {nullable}")
        
        # Count columns per table
        etf_count = sum(1 for row in rows if row[0] == 'etf_saved_strategy')
        rs_count = sum(1 for row in rows if row[0] == 'rs_etf_instance')
        
        print("\n" + "=" * 60)
        print(f"✅ etf_saved_strategy: {etf_count}/2 columns found")
        print(f"✅ rs_etf_instance: {rs_count}/2 columns found")
        
        if etf_count == 2 and rs_count == 2:
            print("\n🎉 All notification columns successfully added!")
            return True
        else:
            print("\n⚠️  Some columns are missing. Check table migration.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    success = test_notification_columns()
    exit(0 if success else 1)
