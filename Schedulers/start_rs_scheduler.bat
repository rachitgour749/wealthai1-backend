@echo off
echo ================================================================================
echo Starting RS Strategy Scheduler
echo ================================================================================
echo.
echo This scheduler will run automatically:
echo   - Daily EOD Data Fetch: 6:00 PM IST
echo   - Daily Signal Generation: 6:00 PM IST  
echo   - Weekly Cleanup: Sunday 2:00 AM IST
echo.
echo Press Ctrl+C to stop the scheduler
echo ================================================================================
echo.

cd /d "%~dp0"
python rs_scheduler.py

pause

