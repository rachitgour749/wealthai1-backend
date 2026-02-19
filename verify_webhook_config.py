import sys
import os
import requests
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from Databases.app_data_db_connection import get_session, create_connection, engine, Base
from Services.strategy_manager.models import SavedInstance
from Services.strategy_manager.webhook_manager import create_webhook_strategy
from sqlalchemy import inspect

def verify_webhook_creation():
    print("Verifying External Strategy Creation...")
    
    # Initialize DB
    create_connection()
    
    # Try creating table (simplest way for test)
    try:
        SavedInstance.__table__.create(bind=engine)
        print("Created saved_instances table.")
    except Exception as e:
        print(f"Table might already exist or error: {e}")
    
    # Test Data
    user_id = "test_user_ext@example.com"
    strategy_type = "External_Strategy"
    strategy_name = "TradingView Alpha"
    reference_capital = 500000.0
    client_info = {"TEST_CLIENT_001": 1.5, "TEST_CLIENT_002": 0.8}
    webhook_val = "marketai_signal_source"
    
    db = get_session()
    try:
        # 1. Call logic directly (Integrated Test)
        print("1. Calling create_webhook_strategy service...")
        result = create_webhook_strategy(
            user_id=user_id,
            strategy_type=strategy_type,
            strategy_name=strategy_name,
            reference_capital=reference_capital,
            client_info=client_info,
            webhook=webhook_val,
            db=db
        )
        
        run_id = result['run_id']
        print(f"   -> Service returned Run ID: {run_id}")
        
        # 2. Verify in Database
        print("2. Verifying database record...")
        instance = db.query(SavedInstance).filter(SavedInstance.run_id == run_id).first()
        
        if not instance:
            print("❌ FAILED: SavedInstance not found in DB.")
            return

        print(f"   -> Found Instance ID: {instance.id}")
        print(f"   -> Source: {instance.source}")
        print(f"   -> Status: {instance.status}")
        print(f"   -> Webhook/Source: {instance.webhook_url}")
        print(f"   -> Run ID Prefix: {instance.run_id.split('_')[0]}")
        
        # Assertions
        assert instance.source == 'other', f"Expected source='other', got '{instance.source}'"
        assert instance.status == 'running', f"Expected status='running', got '{instance.status}'"
        assert instance.webhook_url == webhook_val, f"Expected webhook='{webhook_val}', got '{instance.webhook_url}'"
        assert instance.run_id.startswith("EXT_"), f"Expected run_id to start with 'EXT_', got '{instance.run_id}'"
        assert instance.client_info == client_info, "Client Info mismatch"
        
        print("\n✅ SUCCESS: External Strategy created and verified in DB.")
        
        # Cleanup
        print("3. Cleaning up test data...")
        db.delete(instance)
        db.commit()
        print("   -> Test record deleted.")
        
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    verify_webhook_creation()
