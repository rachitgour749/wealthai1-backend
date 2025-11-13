#!/bin/bash
echo "================================================================================"
echo "Starting ALL WealthAI Schedulers"
echo "================================================================================"
echo ""
echo "This will start:"
echo "   1. ETF + Stock Scheduler (4:00 PM EOD, 4:30/4:35 PM Signals)"
echo "   2. RS Strategy Scheduler (6:00 PM EOD + Signals)"
echo ""
echo "Both schedulers will run in the background"
echo "================================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start ETF/Stock Scheduler in background
echo "Starting ETF/Stock Scheduler..."
cd "$SCRIPT_DIR/cronjob"
nohup python3 scheduler.py --mode scheduled > ../logs/etf_scheduler.log 2>&1 &
ETF_PID=$!
echo "ETF/Stock Scheduler started with PID: $ETF_PID"

# Wait a bit before starting next scheduler
sleep 2

# Start RS Strategy Scheduler in background
echo "Starting RS Strategy Scheduler..."
cd "$SCRIPT_DIR/Strategies/rsStrategy"
nohup python3 rs_scheduler.py > ../../logs/rs_scheduler.log 2>&1 &
RS_PID=$!
echo "RS Strategy Scheduler started with PID: $RS_PID"

echo ""
echo "================================================================================"
echo "✅ All schedulers started successfully!"
echo "================================================================================"
echo ""
echo "ETF/Stock Scheduler PID: $ETF_PID"
echo "RS Strategy Scheduler PID: $RS_PID"
echo ""
echo "To check logs:"
echo "   tail -f $SCRIPT_DIR/logs/etf_scheduler.log"
echo "   tail -f $SCRIPT_DIR/logs/rs_scheduler.log"
echo ""
echo "To stop schedulers:"
echo "   kill $ETF_PID $RS_PID"
echo ""
echo "Or use: pkill -f 'scheduler.py' && pkill -f 'rs_scheduler.py'"
echo "================================================================================"

