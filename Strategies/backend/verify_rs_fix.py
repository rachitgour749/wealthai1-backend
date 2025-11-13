"""
Verification script to confirm RS calculation fix works properly
"""
import pandas as pd
from datetime import datetime, timedelta
from services.ingestion import load_stock_data, load_index_data
from backtester_core.rs_calculator import calculate_rs_score

def verify_rs_fix():
    """Verify that RS calculation works with proper data loading"""
    
    print("=" * 80)
    print("RS CALCULATION FIX VERIFICATION")
    print("=" * 80)
    
    # Backtest parameters
    backtest_start = '2024-10-09'
    backtest_end = '2025-10-07'
    
    # Calculate data load start (90 days before)
    backtest_start_dt = datetime.strptime(backtest_start, '%Y-%m-%d')
    data_load_start = (backtest_start_dt - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"\n1. Loading data with 90-day buffer:")
    print(f"   Data load start: {data_load_start}")
    print(f"   Backtest start: {backtest_start}")
    print(f"   Backtest end: {backtest_end}")
    
    # Load data
    stock_data = load_stock_data(start_date=data_load_start, end_date=backtest_end)
    index_data = load_index_data(start_date=data_load_start, end_date=backtest_end)
    
    print(f"\n2. Data loaded:")
    print(f"   Stock records: {len(stock_data)}")
    print(f"   Index records: {len(index_data)}")
    print(f"   Symbols: {stock_data['symbol'].nunique()}")
    
    # Test RS calculation for one symbol
    test_symbol = stock_data['symbol'].iloc[0]
    symbol_data = stock_data[stock_data['symbol'] == test_symbol].copy()
    symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
    
    print(f"\n3. Testing RS calculation for {test_symbol}:")
    print(f"   Total rows: {len(symbol_data)}")
    print(f"   Date range: {symbol_data['date'].min()} to {symbol_data['date'].max()}")
    
    # Calculate RS
    result = calculate_rs_score(symbol_data, index_data, windows=[5, 21, 63])
    
    # Analyze results
    print(f"\n4. RS Calculation Results:")
    print(f"   Total rows: {len(result)}")
    print(f"   Rows with NaN rs_score: {result['rs_score'].isna().sum()}")
    print(f"   Rows with valid rs_score: {result['rs_score'].notna().sum()}")
    
    # Filter to backtest period
    backtest_data = result[(result['date'] >= backtest_start) & (result['date'] <= backtest_end)]
    
    print(f"\n5. Backtest Period Analysis ({backtest_start} to {backtest_end}):")
    print(f"   Total rows in backtest period: {len(backtest_data)}")
    print(f"   Rows with NaN rs_score: {backtest_data['rs_score'].isna().sum()}")
    print(f"   Rows with valid rs_score: {backtest_data['rs_score'].notna().sum()}")
    
    # Show first few rows of backtest period
    print(f"\n6. First 10 rows of backtest period:")
    print(backtest_data[['date', 'adj_close', 'rs_5d', 'rs_21d', 'rs_63d', 'rs_score']].head(10).to_string())
    
    # Verification
    print(f"\n7. Verification Results:")
    nan_count = backtest_data['rs_score'].isna().sum()
    valid_count = backtest_data['rs_score'].notna().sum()
    
    if nan_count == 0 and valid_count > 0:
        print(f"   ✅ SUCCESS: All rows in backtest period have valid RS scores!")
        print(f"   ✅ RS calculation is working correctly")
        print(f"   ✅ No NaN values in backtest period")
    elif nan_count < 10 and valid_count > 200:
        print(f"   ⚠️  PARTIAL SUCCESS: {nan_count} NaN rows, {valid_count} valid rows")
        print(f"   ⚠️  Most rows have valid RS scores, only a few NaN at the start")
    else:
        print(f"   ❌ FAILED: Too many NaN values ({nan_count} NaN, {valid_count} valid)")
        print(f"   ❌ RS calculation still has issues")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    return nan_count == 0

if __name__ == "__main__":
    try:
        success = verify_rs_fix()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)





