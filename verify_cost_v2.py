import asyncio
import json
from Databases.app_data_db_connection import create_connection
from Handlers.etf_swing_handler import ETFSwingHandler
from APIs.unified_schemas import UnifiedBacktestRequest

async def verify():
    create_connection()
    handler = ETFSwingHandler(None)
    req = UnifiedBacktestRequest(
        strategy_type='ETF_Swing_Strategy', 
        start_date='2024-02-27', 
        end_date='2025-02-18',
        tickers=['NIFTYBEES'], 
        initial_capital=1000000, 
        sma_lookback=20,
        brokerage_percent=0.1
    )
    try:
        res = await handler.run_backtest(req)
        if res.success:
            print("SUCCESS: Backtest completed")
            print(f"Cost Breakdown: {json.dumps(res.cost_breakdown, indent=2)}")
            
            if res.transaction_log:
                first_trade = res.transaction_log[0]
                costs = first_trade.get('costs', {})
                print(f"Keys in costs: {list(costs.keys())}")
                print(f"Costs Dict: {json.dumps(costs, indent=2)}")
            
            # Check if breakdown has expected structure
            if res.cost_breakdown:
                first_year = list(res.cost_breakdown.keys())[0]
                year_data = res.cost_breakdown[first_year]
                print(f"Year {first_year} data: {year_data}")
        else:
            print(f"FAILED: {res.error}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    asyncio.run(verify())
