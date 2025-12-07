import sys
sys.path.insert(0, r'c:\Users\Lenovo\Desktop\WEALTHAI_PROD\wealthai1-backend')

from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester

print("Initializing ETF Rotation Backtester...")
try:
    backtester = ETFRotationBacktester()
    print("✅ Backtester initialized successfully")
    
    print("\nRunning backtest...")
    result = backtester.run_backtest(
        tickers=['NIFTYBEES', 'BANKBEES', 'JUNIORBEES'],
        start_date='2021-07-19',
        end_date='2025-12-01',
        capital_per_week=50000,
        accumulation_weeks=52,
        brokerage_percent=0.03,
        compounding_enabled=True
    )
    
    print(f"\n✅ Backtest completed!")
    print(f"Result: {result}")
    print(f"Weekly NAV DF length: {len(backtester.weekly_nav_df)}")
    print(f"Benchmark DF length: {len(backtester.nifty50_df)}")
    
    if len(backtester.weekly_nav_df) > 0 and len(backtester.nifty50_df) > 0:
        print("\n✅ Both dataframes have data!")
        print(f"Alignment: {len(backtester.weekly_nav_df) == len(backtester.nifty50_df)}")
    else:
        print("\n❌ One or both dataframes are empty!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
