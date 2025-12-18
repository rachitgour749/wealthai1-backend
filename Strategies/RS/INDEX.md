# RS Strategy Configuration - Documentation Index

Welcome to the RS Strategy configuration documentation! This index will help you find what you need quickly.

## 📚 Documentation Files

### For Quick Start
1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** ⚡
   - Quick toggle instructions
   - Comparison table
   - Common commands
   - **Start here if you just want to change the mode!**

### For Understanding
2. **[FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md)** 📊
   - Visual flow diagrams
   - Daily vs Weekly mode comparison
   - Example scenarios
   - Code architecture
   - **Great for visual learners!**

### For Complete Guide
3. **[README.md](README.md)** 📖
   - Complete configuration guide
   - Detailed explanations
   - Usage examples
   - Troubleshooting
   - **Read this for comprehensive understanding**

### For Technical Details
4. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** 🔧
   - What was implemented
   - Code changes made
   - Technical implementation
   - Testing recommendations
   - **For developers and technical users**

## 🎯 Quick Navigation

### I want to...

#### Change stop loss mode
→ Go to [QUICK_REFERENCE.md](QUICK_REFERENCE.md#-quick-toggle)

#### Understand the difference between modes
→ Go to [QUICK_REFERENCE.md](QUICK_REFERENCE.md#-comparison)
→ Or [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md#daily-mode-flow)

#### See visual examples
→ Go to [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md#example-scenario)

#### Learn how configuration works
→ Go to [README.md](README.md#configuration-priority)

#### Override config via API
→ Go to [README.md](README.md#example-2-override-via-api)

#### Understand code changes
→ Go to [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#3-code-changes)

#### Troubleshoot issues
→ Go to [README.md](README.md#troubleshooting)

#### Test the configuration
→ Go to [QUICK_REFERENCE.md](QUICK_REFERENCE.md#-test-configuration)

## 📁 Configuration Files

### Main Files
- **`rs_config.json`** - Main configuration file (edit this to change mode)
- **`rs_config_loader.py`** - Configuration loader utility (run to test)

### Code Files (Modified)
- `../RS_ETF/rs_etf_backtester_core.py` - RS ETF strategy
- `../RS_Stocks/rs_backtester_core.py` - RS Stocks strategy

## 🚀 Quick Start Guide

### Step 1: Understand the Modes
Read: [QUICK_REFERENCE.md - Comparison Table](QUICK_REFERENCE.md#-comparison)

### Step 2: Choose Your Mode
- **Daily Mode** (Default): Better risk protection, more trades
- **Weekly Mode**: Pure weekly rebalancing, fewer trades

### Step 3: Change Configuration
Edit `rs_config.json`:
```json
{
  "stop_loss_settings": {
    "daily_stop_loss_check": true  // or false
  }
}
```

### Step 4: Test Configuration
```bash
python rs_config_loader.py
```

### Step 5: Run Backtest
Your strategy will automatically use the new configuration!

## 📊 File Purposes at a Glance

| File | Purpose | Audience | Length |
|------|---------|----------|--------|
| **QUICK_REFERENCE.md** | Quick commands & toggle | Everyone | 1 page |
| **FLOW_DIAGRAMS.md** | Visual explanations | Visual learners | 3 pages |
| **README.md** | Complete guide | Users | 5 pages |
| **IMPLEMENTATION_SUMMARY.md** | Technical details | Developers | 7 pages |
| **INDEX.md** | This file | Everyone | 1 page |

## 🎓 Learning Path

### Beginner Path
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Get the basics
2. [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md) - See it visually
3. Try changing the config and running a backtest

### Intermediate Path
1. [README.md](README.md) - Full understanding
2. [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md) - See examples
3. Experiment with both modes
4. Compare backtest results

### Advanced Path
1. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
2. Review code changes in backtester files
3. Consider custom modifications
4. Implement API overrides

## 💡 Common Questions

**Q: Which mode should I use?**
→ See [QUICK_REFERENCE.md - When to Use Each Mode](QUICK_REFERENCE.md#-when-to-use-each-mode)

**Q: How do I change the mode?**
→ See [QUICK_REFERENCE.md - Quick Toggle](QUICK_REFERENCE.md#-quick-toggle)

**Q: What's the difference between modes?**
→ See [FLOW_DIAGRAMS.md - Daily vs Weekly](FLOW_DIAGRAMS.md#daily-mode-flow)

**Q: Can I override per backtest?**
→ See [README.md - Override via API](README.md#example-2-override-via-api)

**Q: How do I test the config?**
→ See [QUICK_REFERENCE.md - Test Configuration](QUICK_REFERENCE.md#-test-configuration)

**Q: What files were changed?**
→ See [IMPLEMENTATION_SUMMARY.md - Files Summary](IMPLEMENTATION_SUMMARY.md#files-summary)

## 🔗 External Resources

### Related Strategies
- RS_ETF Strategy: `../RS_ETF/`
- RS_Stocks Strategy: `../RS_Stocks/`

### Related Documentation
- Benchmark Calculator: `../benchmark_calculator.py`
- Strategy Logger: `../utilities/logging_config.py`

## 📝 Version Information

- **Version:** 1.0
- **Date:** 2025-12-18
- **Author:** RS Strategy Team
- **Status:** Production Ready

## 🤝 Support

For issues or questions:
1. Check [README.md - Troubleshooting](README.md#troubleshooting)
2. Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
3. Contact development team

---

**Happy Trading! 📈**

*Remember: Always backtest both modes before using in production!*
