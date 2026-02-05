
import sys
import os
import logging
from datetime import datetime
from sqlalchemy import text

# Add project root to path
sys.path.append(os.getcwd())

from Services.scheduler.generators.etf_rotation_generator import generate_etf_rotation_signals
from Services.scheduler.generators.rotation_stocks_generator import generate_stock_rotation_signals
from Databases.app_data_db_connection import get_session, create_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_generated_signals(run_time):
    """Check database for signals generated after run_time"""
    session = get_session()
    try:
        query = text("""
            SELECT id, strategy_name, symbol_name, order_side, execution_status, created_at 
            FROM trading_signals 
            WHERE created_at >= :run_time
            ORDER BY created_at DESC
        """)
        
        result = session.execute(query, {'run_time': run_time}).fetchall()
        
        print("\n" + "="*60)
        print(f"VERIFICATION RESULTS (Signals generated after {run_time.strftime('%H:%M:%S')})")
        print("="*60)
        
        if result:
            print(f"[OK] Found {len(result)} new signals:")
            for row in result:
                print(f"  ID: {row[0]}, Strategy: {row[1]}, Symbol: {row[2]}, Side: {row[3]}, Status: {row[4]}")
        else:
            print("[FAIL] No new signals found.")
            print("  - Check if strategies are 'running' in saved_instances")
            print("  - Check if today is a valid trading day")
            print("  - Check logs for errors")
            
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error verifying signals: {e}")
    finally:
        session.close()

def run_test():
    """Run manual signal generation test"""
    if not create_connection():
        logger.error("Failed to connect to database")
        return

    start_time = datetime.utcnow()
    print(f"Starting test at {start_time}")
    
    # 1. Test ETF Rotation
    print("\n--- Testing ETF Rotation Generation ---")
    try:
        # Pass a specific date if needed, or None for today
        # generate_etf_rotation_signals(datetime(2026, 2, 4)) 
        generate_etf_rotation_signals()
    except Exception as e:
        logger.error(f"ETF Rotation failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test Stock Rotation
    print("\n--- Testing Stock Rotation Generation ---")
    try:
        generate_stock_rotation_signals()
    except Exception as e:
        logger.error(f"Stock Rotation failed: {e}")

    # 3. Verify
    check_generated_signals(start_time)

if __name__ == "__main__":
    run_test()
