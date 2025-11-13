# Scheduler Fixes Summary

## Issues Fixed

### 1. **Missing `etf_saved_strategy` Table**
- **Problem**: `ERROR:etf_signal_generator:Error getting running instances: no such table: etf_saved_strategy`
- **Fix**: Added table creation in `ETFSignalGenerator.create_tables()` method
- **Location**: `Strategies/etfstrategy/etf_signal_generator.py`

### 2. **Missing `stock_saved_strategy` Table**
- **Problem**: Same issue for stock strategy
- **Fix**: Added table creation in `StockSignalGenerator.create_tables()` method
- **Location**: `Strategies/stockstrategy/stock_signal_generator.py`

### 3. **Table Auto-Initialization**
- Both signal generators now automatically create their required instance tables on initialization
- Tables are created with proper schema matching the API definitions

## Changes Made

### ETF Signal Generator (`Strategies/etfstrategy/etf_signal_generator.py`)
1. ✅ Added `etf_saved_strategy` table creation in `create_tables()` method
2. ✅ Improved ticker parsing (handles JSON strings)
3. ✅ Improved table lookup logic (checks multiple possible table names)

### STOCK Signal Generator (`Strategies/stockstrategy/stock_signal_generator.py`)
1. ✅ Added `stock_saved_strategy` table creation in `create_tables()` method
2. ✅ Improved ticker parsing (handles JSON strings)
3. ✅ Improved table lookup logic (checks multiple possible table names)

## Table Schemas Created

### `etf_saved_strategy` Table
- All required columns from the API schema
- Index on `user_id` for faster queries
- Default `status = 'deploy'`

### `stock_saved_strategy` Table
- All required columns from the API schema
- Index on `user_id` for faster queries
- Default `status = 'deploy'`

## Expected Behavior Now

1. **On Signal Generator Initialization**:
   - ✅ Creates `live_signals` / `live_stock_signals` tables
   - ✅ Creates `etf_saved_strategy` / `stock_saved_strategy` tables
   - ✅ Creates necessary indexes

2. **On Signal Generation**:
   - ✅ Checks for running instances in the instance tables
   - ✅ If no running instances, returns informative error message
   - ✅ If running instances exist, generates signals for all of them

3. **Scheduler Execution**:
   - ✅ At 5:55 AM, ETF signals are generated
   - ✅ At 6:00 AM, STOCK signals are generated
   - ✅ At 6:05 AM, RS signals are generated
   - ✅ All use last Friday date automatically

## Testing

When you restart the server:
1. Tables will be auto-created on first signal generator initialization
2. Scheduler will check for running instances
3. If instances exist, signals will be generated
4. If no instances exist, you'll get a clear error message

## Next Steps

To test signal generation:
1. Ensure you have at least one running instance in:
   - `etf_saved_strategy` with `status='running'` for ETF
   - `stock_saved_strategy` with `status='running'` for STOCK
   - `saved_rs_strategies` with `status='running'` for RS

2. Restart the server to initialize tables

3. Wait for 5:55 AM IST or manually trigger the jobs

---

**Status**: ✅ Fixed - Tables will auto-create on initialization

