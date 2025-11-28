import sys
import os
import asyncio

# Add project root to path
# File is in Strategies/Rotation_Stocks/verify_stock_refactor_v2.py
# Root is ../../ (wealthai1-backend)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

async def verify_refactoring():
    print("🔍 Verifying Stock Rotation Refactoring...")
    
    try:
        # 1. Verify Imports
        print("\n1. Verifying Imports...")
        
        from Strategies.Rotation_Stocks.api.stock_routes import stock_router, initialize_stock_backtester
        print("   ✅ Successfully imported stock_router and initialize_stock_backtester from api.stock_routes")
        
        from Strategies.Rotation_Stocks.services.backtester import StockRotationBacktester
        print("   ✅ Successfully imported StockRotationBacktester from services.backtester")
        
        from Strategies.Rotation_Stocks.services.signal_generator import LiveStockSignalGenerator
        print("   ✅ Successfully imported LiveStockSignalGenerator from services.signal_generator")
        
        from Strategies.Rotation_Stocks.stock_schemas import BacktestRequest
        print("   ✅ Successfully imported BacktestRequest from stock_schemas")
        
        # 2. Verify Backtester Initialization
        print("\n2. Verifying Backtester Initialization...")
        success = initialize_stock_backtester()
        if success:
            print("   ✅ Stock Backtester initialized successfully")
        else:
            print("   ❌ Failed to initialize Stock Backtester")
            return False
            
        # 3. Verify Signal Generator Initialization
        print("\n3. Verifying Signal Generator Initialization...")
        try:
            generator = LiveStockSignalGenerator()
            print("   ✅ LiveStockSignalGenerator initialized successfully")
        except Exception as e:
            print(f"   ❌ Failed to initialize LiveStockSignalGenerator: {e}")
            return False
            
        print("\n✨ Refactoring Verification Passed!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(verify_refactoring())
