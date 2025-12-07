import sys
sys.path.insert(0, r'c:\Users\Lenovo\Desktop\WEALTHAI_PROD\wealthai1-backend')

from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester
import json

print("Initializing ETF Rotation Backtester...")
backtester = ETFRotationBacktester()
backtester.set_verbose(True)  # Enable verbose mode

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

print("\n" + "="*60)
print("CALCULATING METRICS")
print("="*60)

etf_metrics = backtester.calculate_metrics(50000, 52, 8.0)
benchmark_metrics = backtester.calculate_benchmark_metrics(2600000, 8.0)

print('\n=== ETF Strategy Metrics ===')
for key, value in etf_metrics.items():
    print(f'  {key}: {value}')

print('\n=== Benchmark Metrics ===')
for key, value in benchmark_metrics.items():
    print(f'  {key}: {value}')

print(f'\n=== Data Alignment Check ===')
print(f'Weekly strategy data points: {len(backtester.weekly_nav_df)}')
print(f'Benchmark data points: {len(backtester.nifty50_df)}')
print(f'Alignment match: {len(backtester.weekly_nav_df) == len(backtester.nifty50_df)}')

if not backtester.nifty50_df.empty:
    print(f'\nBenchmark DataFrame columns: {list(backtester.nifty50_df.columns)}')
    print(f'First 3 benchmark dates: {backtester.nifty50_df["date"].head(3).tolist()}')
    print(f'Last 3 benchmark dates: {backtester.nifty50_df["date"].tail(3).tolist()}')

# Validation
if benchmark_metrics:
    volatility = float(benchmark_metrics.get('Volatility', '0%').replace('%', ''))
    max_dd = float(benchmark_metrics.get('Max Drawdown', '0%').replace('%', ''))
    total_days = benchmark_metrics.get('Total Days', 0)

    print(f'\n=== Validation Results ===')
    print(f'✓ Volatility > 0: {volatility > 0} (value: {volatility}%)')
    print(f'✓ Max Drawdown != 0: {max_dd != 0} (value: {max_dd}%)')
    print(f'✓ Total Days > 1: {total_days > 1} (value: {total_days})')
    print(f'✓ Data alignment: {len(backtester.weekly_nav_df) == len(backtester.nifty50_df)}')

    all_passed = (volatility > 0 and max_dd != 0 and total_days > 1 and 
                  len(backtester.weekly_nav_df) == len(backtester.nifty50_df))
    
    if all_passed:
        print('\n✅ ALL VALIDATIONS PASSED!')
    else:
        print('\n❌ SOME VALIDATIONS FAILED!')
else:
    print('\n❌ No benchmark metrics calculated!')
