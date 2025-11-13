#!/bin/bash

# WealthAI Backend Scheduler Setup Script for Ubuntu
# This script sets up schedulers to run automatically on system boot

set -e

echo "================================================================================"
echo "WealthAI Backend - Scheduler Setup for Ubuntu"
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
chmod +x "$SCRIPT_DIR/setup_schedulers_ubuntu.sh"
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

# Step 6: Test scheduler imports
echo "Step 6: Testing scheduler imports..."
python3 -c "from Schedulers.scheduler_manager import start_schedulers, stop_schedulers, get_scheduler_status; print('✅ Scheduler imports working')"
if [ $? -ne 0 ]; then
    echo "❌ Scheduler imports failed"
    exit 1
fi
echo "✅ Scheduler imports working"
echo ""

# Step 7: Show useful commands
echo "================================================================================"
echo "✅ Ubuntu Scheduler Setup Complete!"
echo "================================================================================"
echo ""
echo "Your schedulers will now start automatically when you run:"
echo "   python3 server.py"
echo ""
echo "Scheduler Schedule:"
echo "   - ETF/Stock EOD Data: Daily at 4:00 PM IST"
echo "   - ETF Signals: Friday at 4:30 PM IST"
echo "   - Stock Signals: Friday at 4:35 PM IST"
echo "   - RS EOD Data: Daily at 6:00 PM IST"
echo "   - RS Signals: Daily at 6:00 PM IST"
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
echo "To test: python3 server.py"
echo "To check logs: $SCRIPT_DIR/logs/"
echo "================================================================================"
