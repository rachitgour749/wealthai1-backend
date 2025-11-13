#!/bin/bash
echo "================================================================================"
echo "Starting RS Strategy Scheduler"
echo "================================================================================"
echo ""
echo "This scheduler will run automatically:"
echo "   - Daily EOD Data Fetch: 6:00 PM IST"
echo "   - Daily Signal Generation: 6:00 PM IST"
echo "   - Weekly Cleanup: Sunday 2:00 AM IST"
echo ""
echo "Press Ctrl+C to stop the scheduler"
echo "================================================================================"
echo ""

cd "$(dirname "$0")/Strategies/rsStrategy"
python3 rs_scheduler.py

