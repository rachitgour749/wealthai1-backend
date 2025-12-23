"""
Quick verification script to check notification values in database
"""

from Databases.app_data_db_connection import get_session
from sqlalchemy import text

def check_notification_values():
    session = get_session()
    
    print("\n" + "="*80)
    print("NOTIFICATION PREFERENCES IN DATABASE")
    print("="*80)
    
    # Check RS strategies
    print("\nRS ETF Strategies:")
    print("-"*80)
    print(f"{'Strategy Name':<35} {'Email':<10} {'Telegram':<10} {'Status':<10}")
    print("-"*80)
    
    result = session.execute(text("""
        SELECT strategy_name, email_notification, telegram_notification, status
        FROM rs_etf_instance 
        WHERE strategy_name LIKE 'Test%'
        ORDER BY strategy_name
    """))
    
    rs_count = 0
    for row in result.fetchall():
        rs_count += 1
        email = "✅ YES" if row[1] else "❌ NO"
        telegram = "✅ YES" if row[2] else "❌ NO"
        print(f"{row[0]:<35} {email:<10} {telegram:<10} {row[3]:<10}")
    
    if rs_count == 0:
        print("(No test strategies found)")
    
    # Check Rotation ETF strategies
    print("\nRotation ETF Strategies:")
    print("-"*80)
    print(f"{'Strategy Name':<35} {'Email':<10} {'Telegram':<10} {'Status':<10}")
    print("-"*80)
    
    result = session.execute(text("""
        SELECT strategy_name, email_notification, telegram_notification, status
        FROM etf_saved_strategy 
        WHERE strategy_name LIKE 'Test%'
        ORDER BY strategy_name
    """))
    
    etf_count = 0
    for row in result.fetchall():
        etf_count += 1
        email = "✅ YES" if row[1] else "❌ NO"
        telegram = "✅ YES" if row[2] else "❌ NO"
        print(f"{row[0]:<35} {email:<10} {telegram:<10} {row[3]:<10}")
    
    if etf_count == 0:
        print("(No test strategies found)")
    
    session.close()
    
    print("\n" + "="*80)
    print(f"Total RS Strategies: {rs_count}")
    print(f"Total Rotation ETF Strategies: {etf_count}")
    print("="*80 + "\n")

if __name__ == "__main__":
    check_notification_values()
