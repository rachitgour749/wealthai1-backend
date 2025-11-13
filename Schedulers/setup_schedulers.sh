#!/bin/bash

# WealthAI Backend Scheduler Setup Script for Ubuntu
# This script sets up schedulers to run automatically on system boot

set -e

echo "================================================================================"
echo "WealthAI Backend - Scheduler Setup"
echo "================================================================================"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then 
    echo "⚠️  Please run this script as the ubuntu user, not as root."
    echo "   The script will ask for sudo password when needed."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Make shell scripts executable
echo "Step 1: Making shell scripts executable..."
chmod +x "$SCRIPT_DIR/start_etf_scheduler.sh"
chmod +x "$SCRIPT_DIR/start_rs_scheduler.sh"
chmod +x "$SCRIPT_DIR/start_all_schedulers.sh"
echo "✅ Shell scripts are now executable"
echo ""

# Step 2: Create logs directory if it doesn't exist
echo "Step 2: Ensuring logs directory exists..."
mkdir -p "$SCRIPT_DIR/logs"
chmod 755 "$SCRIPT_DIR/logs"
echo "✅ Logs directory ready"
echo ""

# Step 3: Install systemd service files
echo "Step 3: Installing systemd service files..."
echo "   (This requires sudo password)"
sudo cp "$SCRIPT_DIR/etf-scheduler.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/rs-scheduler.service" /etc/systemd/system/
echo "✅ Service files installed"
echo ""

# Step 4: Reload systemd
echo "Step 4: Reloading systemd daemon..."
sudo systemctl daemon-reload
echo "✅ Systemd reloaded"
echo ""

# Step 5: Enable services (auto-start on boot)
echo "Step 5: Enabling services to start on boot..."
sudo systemctl enable etf-scheduler.service
sudo systemctl enable rs-scheduler.service
echo "✅ Services enabled for auto-start"
echo ""

# Step 6: Start services
echo "Step 6: Starting scheduler services..."
sudo systemctl start etf-scheduler.service
sudo systemctl start rs-scheduler.service
echo "✅ Services started"
echo ""

# Wait a moment for services to initialize
sleep 3

# Step 7: Check service status
echo "================================================================================"
echo "📊 Service Status Check"
echo "================================================================================"
echo ""

echo "ETF/Stock Scheduler Status:"
sudo systemctl status etf-scheduler.service --no-pager -l
echo ""

echo "RS Strategy Scheduler Status:"
sudo systemctl status rs-scheduler.service --no-pager -l
echo ""

# Step 8: Show useful commands
echo "================================================================================"
echo "✅ Setup Complete!"
echo "================================================================================"
echo ""
echo "Your schedulers are now running and will auto-start on system boot!"
echo ""
echo "📋 Useful Commands:"
echo ""
echo "Check scheduler status:"
echo "   sudo systemctl status etf-scheduler"
echo "   sudo systemctl status rs-scheduler"
echo ""
echo "View live logs:"
echo "   tail -f $SCRIPT_DIR/logs/etf_scheduler.log"
echo "   tail -f $SCRIPT_DIR/logs/rs_scheduler.log"
echo ""
echo "View systemd logs:"
echo "   sudo journalctl -u etf-scheduler -f"
echo "   sudo journalctl -u rs-scheduler -f"
echo ""
echo "Restart schedulers:"
echo "   sudo systemctl restart etf-scheduler"
echo "   sudo systemctl restart rs-scheduler"
echo ""
echo "Stop schedulers:"
echo "   sudo systemctl stop etf-scheduler"
echo "   sudo systemctl stop rs-scheduler"
echo ""
echo "Disable auto-start:"
echo "   sudo systemctl disable etf-scheduler"
echo "   sudo systemctl disable rs-scheduler"
echo ""
echo "================================================================================"
echo ""
echo "📅 Scheduler Times (IST):"
echo "   - ETF/Stock EOD Data: Daily at 4:00 PM"
echo "   - ETF Signals: Friday at 4:30 PM"
echo "   - Stock Signals: Friday at 4:35 PM"
echo "   - Signal Execution: Monday at 9:20 AM"
echo "   - RS EOD Data: Daily at 6:00 PM"
echo "   - RS Signals: Daily at 6:00 PM"
echo "   - RS Cleanup: Sunday at 2:00 AM"
echo "================================================================================"

