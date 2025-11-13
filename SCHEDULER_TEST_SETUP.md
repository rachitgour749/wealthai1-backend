# Scheduler Test Setup - Signal Generation at 5:55 AM

## Overview
Configured schedulers to generate signals for ETF, STOCK, and RS strategies at **5:55 AM IST** using the **last Friday date**.

## Scheduled Jobs

### **ETF Strategy**
- **Time**: 5:55 AM IST
- **Signal Date**: Last Friday (calculated dynamically)
- **Job ID**: `etf_one_time_test_5_55`
- **Method**: `generate_etf_signals_for_last_friday_job()`

### **STOCK Strategy**
- **Time**: 6:00 AM IST (5 minutes after ETF)
- **Signal Date**: Last Friday (calculated dynamically)
- **Job ID**: `stock_one_time_test_5_55`
- **Method**: `generate_stock_signals_for_last_friday_job()`

### **RS Strategy**
- **Time**: 6:05 AM IST (10 minutes after ETF)
- **Signal Date**: Last Friday (calculated dynamically)
- **Job ID**: `rs_one_time_test_5_55`
- **Method**: `generate_signals_for_last_friday_job()`

## How It Works

1. **Last Friday Calculation**:
   - If today is Friday → uses today
   - Otherwise → calculates days since last Friday and subtracts

2. **Signal Generation**:
   - ETF: Generates signals for all running instances in `etf_saved_strategy` table (status='running')
   - STOCK: Generates signals for all running instances in `stock_saved_strategy` table (status='running')
   - RS: Generates signals for all running instances in `saved_rs_strategies` table (status='running')

3. **Validation**:
   - Only generates for instances with `status = 'running'`
   - Applies quality filters (price, volume, turnover)
   - Uses dynamic limits from strategy config

## Startup

When you start `server.py`, the schedulers will:
1. Initialize automatically via `start_all_schedulers()`
2. Schedule the one-time jobs at 5:55 AM (ETF), 6:00 AM (STOCK), 6:05 AM (RS)
3. Log the scheduled times and target dates

## Testing

To test immediately without waiting:
1. Start the server: `python server.py`
2. Check logs: `Schedulers/logs/etf_scheduler.log` and `Schedulers/logs/rs_scheduler.log`
3. Wait for 5:55 AM IST or modify the time in scheduler code for immediate testing

## Logs to Check

- **ETF/STOCK**: `Schedulers/logs/etf_scheduler.log`
- **RS**: `Schedulers/logs/rs_scheduler.log`
- **Signal Generator**: `logs/etf_signal_generator.log`, `logs/stock_signal_generator.log`

## Expected Output

When jobs run, you'll see:
```
=== One-time ETF Signal Generation at 5:55 AM Started ===
   Target date (last Friday): 2025-10-30
✅ ETF signals generated successfully
   Generated 15 total signals
   Instances processed: 4
   Instances successful: 4

=== One-time Stock Signal Generation at 5:55 AM Started ===
   Target date (last Friday): 2025-10-30
✅ Stock signals generated successfully
   Generated 10 total signals
   Instances processed: 2
   Instances successful: 2

=== One-time RS Signal Generation at 5:55 AM Started ===
   Target date (last Friday): 2025-10-30
✅ RS Signal Generation Complete: 2/2 instances successful, 40 total signals
```

---

**Status**: ✅ Ready - Jobs scheduled for 5:55 AM IST (ETF), 6:00 AM IST (STOCK), 6:05 AM IST (RS)

