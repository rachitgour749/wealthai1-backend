@echo off
echo Starting ETF Data Scheduler...
echo This will run automatically at 4:00 PM IST daily
echo Press Ctrl+C to stop the scheduler
echo.

cd /d "%~dp0"
python scheduler.py --mode scheduled

pause
