@echo off
echo ================================================================================
echo WealthAI Backend - Scheduler Setup for Windows
echo ================================================================================
echo.
echo This will set up schedulers to run automatically with the server
echo.
echo Press any key to continue...
pause
echo.

echo Step 1: Creating Schedulers directory structure...
if not exist "Schedulers" mkdir Schedulers
if not exist "Schedulers\logs" mkdir Schedulers\logs
echo ✅ Directory structure created

echo.
echo Step 2: Making sure all scheduler files are in place...
if exist "Schedulers\etf_stock_scheduler\scheduler.py" (
    echo ✅ ETF/Stock scheduler found
) else (
    echo ❌ ETF/Stock scheduler not found
    exit /b 1
)

if exist "Schedulers\rs_scheduler\rs_scheduler.py" (
    echo ✅ RS scheduler found
) else (
    echo ❌ RS scheduler not found
    exit /b 1
)

if exist "Schedulers\scheduler_manager.py" (
    echo ✅ Scheduler manager found
) else (
    echo ❌ Scheduler manager not found
    exit /b 1
)

echo.
echo Step 3: Testing scheduler imports...
python -c "from Schedulers.scheduler_manager import start_schedulers, stop_schedulers, get_scheduler_status; print('✅ Scheduler imports working')"
if errorlevel 1 (
    echo ❌ Scheduler imports failed
    exit /b 1
)

echo.
echo ================================================================================
echo ✅ Windows Scheduler Setup Complete!
echo ================================================================================
echo.
echo Your schedulers will now start automatically when you run:
echo   python server.py
echo.
echo Scheduler Schedule:
echo   - ETF/Stock EOD Data: Daily at 4:00 PM IST
echo   - ETF Signals: Friday at 4:30 PM IST
echo   - Stock Signals: Friday at 4:35 PM IST
echo   - RS EOD Data: Daily at 6:00 PM IST
echo   - RS Signals: Daily at 6:00 PM IST
echo.
echo To test: python server.py
echo To check logs: Schedulers\logs\
echo.
pause
