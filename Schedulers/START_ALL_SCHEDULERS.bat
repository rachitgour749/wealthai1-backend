@echo off
echo ================================================================================
echo Starting ALL WealthAI Schedulers
echo ================================================================================
echo.
echo This will start:
echo   1. ETF + Stock Scheduler (4:00 PM EOD, 4:30/4:35 PM Signals)
echo   2. RS Strategy Scheduler (6:00 PM EOD + Signals)
echo.
echo Each scheduler will open in a NEW window
echo KEEP ALL WINDOWS OPEN to keep schedulers running!
echo.
echo Press any key to start all schedulers...
pause
echo.

echo Starting ETF/Stock Scheduler...
start "ETF-Stock Scheduler" cmd /k "cd /d %~dp0cronjob && start_etf_scheduler.bat"

timeout /t 2 /nobreak > nul

echo Starting RS Strategy Scheduler...
start "RS Strategy Scheduler" cmd /k "cd /d %~dp0Strategies\rsStrategy && start_rs_scheduler.bat"

echo.
echo ================================================================================
echo ✅ All schedulers started successfully!
echo ================================================================================
echo.
echo You should see 2 new windows:
echo   - ETF-Stock Scheduler (12 ETFs + 49 Stocks)
echo   - RS Strategy Scheduler (Nifty 500)
echo.
echo To stop schedulers: Press Ctrl+C in each window
echo To monitor: Check logs in the 'logs' folder
echo.
echo Press any key to exit this window...
pause

