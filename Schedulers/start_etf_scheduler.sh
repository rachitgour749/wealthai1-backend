#!/bin/bash
echo "================================================================================"
echo "Starting ETF/Stock Data Scheduler"
echo "================================================================================"
echo ""
echo "This scheduler will run automatically:"
echo "   - Daily EOD Data Fetch: 4:00 PM IST (ETFs + Stocks)"
echo "   - Friday ETF Signal Generation: 4:30 PM IST"
echo "   - Friday Stock Signal Generation: 4:35 PM IST"
echo "   - Monday Signal Execution: 9:20 AM IST"
echo ""
echo "Press Ctrl+C to stop the scheduler"
echo "================================================================================"
echo ""

cd "$(dirname "$0")/cronjob"
python3 scheduler.py --mode scheduled

