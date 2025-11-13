# One-Time Signal Generation Setup for 2025-10-30

## Summary
Configured the scheduler to automatically generate ETF and STOCK signals for the date **2025-10-30** at **5:25 AM** (IST) when the server starts.

## Changes Made

### 1. Scheduler Updated (`Schedulers/etf_stock_scheduler/scheduler.py`)
- ✅ Added `generate_etf_signals_for_date_job()` method
- ✅ Added `generate_stock_signals_for_date_job()` method
- ✅ Added one-time jobs scheduled for 5:25 AM IST (ETF) and 5:30 AM IST (STOCK)
- ✅ Jobs will run automatically when scheduler starts

### 2. Signal Generators
- ✅ Both ETF and STOCK signal generators already support `signal_date` parameter
- ✅ Will generate signals specifically for **2025-10-30**

### 3. Server Auto-Start
- ✅ `server.py` already has `@app.on_event("startup")` that calls `start_all_schedulers()`
- ✅ Scheduler starts automatically when you run `python server.py`

## Scheduled Jobs

| Job | Signal Date | Run Time | Status |
|-----|-------------|----------|--------|
| ETF Signals | 2025-10-30 | 5:25 AM IST | ✅ Scheduled |
| STOCK Signals | 2025-10-30 | 5:30 AM IST | ✅ Scheduled |

**Note**: If 5:25 AM has already passed today, the job will be scheduled for tomorrow at 5:25 AM.

## How It Works

1. **Server Start**: When you run `python server.py`, the startup event triggers
2. **Scheduler Initialization**: `start_all_schedulers()` is called automatically
3. **Job Registration**: One-time jobs for 2025-10-30 are registered
4. **Signal Generation**: At 5:25 AM (ETF) and 5:30 AM (STOCK), signals are generated
5. **Database Storage**: Signals are saved to:
   - ETF: `live_signals` table in `unified_etf_data.sqlite`
   - STOCK: `live_stock_signals` table in `unified_etf_data.sqlite`

## Verification

After the server starts, check the logs:
- **Scheduler logs**: `Schedulers/logs/etf_scheduler.log`
- **Signal generator logs**: `logs/etf_signal_generator.log` and `logs/stock_signal_generator.log`

You should see:
```
📅 Scheduling one-time signal generation for 2025-10-30
   ETF signals: 2025-11-01 05:25:00 IST
   Stock signals: 2025-11-01 05:30:00 IST
```

## Manual Trigger (Optional)

If you want to generate signals immediately without waiting for the scheduler:

```bash
# ETF Signals
curl http://localhost:8000/api/signals/generate

# STOCK Signals  
curl http://localhost:8000/api/stock/signals/generate
```

**Note**: These manual endpoints use the last Friday as the signal date. To generate for 2025-10-30 specifically, use the scheduler (which is already configured).

## Testing

1. Start the server:
   ```bash
   python server.py
   ```

2. Wait for scheduler logs showing the one-time jobs are scheduled

3. Wait until 5:25 AM (or check logs after the scheduled time)

4. Verify signals in database:
   ```sql
   -- ETF Signals
   SELECT * FROM live_signals WHERE signal_date = '2025-10-30';
   
   -- STOCK Signals
   SELECT * FROM live_stock_signals WHERE signal_date = '2025-10-30';
   ```

---

**Setup Date**: November 1, 2025  
**Status**: ✅ Ready - Scheduler will run automatically when server starts

