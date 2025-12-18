# RS Strategy Stop Loss - Quick Reference

## 🎯 Quick Toggle

### Switch to Weekly Stop Loss
Edit `Strategies/RS/rs_config.json`:
```json
{
  "stop_loss_settings": {
    "daily_stop_loss_check": false
  }
}
```

### Switch to Daily Stop Loss (Default)
```json
{
  "stop_loss_settings": {
    "daily_stop_loss_check": true
  }
}
```

## 📊 Comparison

| Feature | Daily Mode | Weekly Mode |
|---------|-----------|-------------|
| **Check Frequency** | Every day | Every day (accumulated) |
| **Execution** | Immediate | Monday (with signals) |
| **Risk Protection** | ✅ High | ⚠️ Moderate |
| **Trading Frequency** | Higher | Lower |
| **Transaction Costs** | Higher | Lower |
| **Alignment** | Defensive | Weekly rebalancing |

## 🔧 Configuration Locations

```
Strategies/
├── RS/
│   ├── rs_config.json          ← Edit this to change mode
│   ├── rs_config_loader.py     ← Loader utility
│   ├── README.md               ← Full documentation
│   └── IMPLEMENTATION_SUMMARY.md
├── RS_ETF/
│   └── rs_etf_backtester_core.py  ← Uses config
└── RS_Stocks/
    └── rs_backtester_core.py      ← Uses config
```

## 💡 When to Use Each Mode

### Use Daily Mode When:
- ✅ Risk management is top priority
- ✅ You want immediate loss protection
- ✅ Trading costs are acceptable
- ✅ You prefer tighter control

### Use Weekly Mode When:
- ✅ You want pure weekly rebalancing
- ✅ Minimizing trades is important
- ✅ You're comfortable with intra-week risk
- ✅ You want all trades on Monday

## 🧪 Test Configuration

```bash
cd Strategies/RS
python rs_config_loader.py
```

Expected output:
```
✓ RS Strategy configuration loaded
Daily Stop Loss Check: True  # or False
```

## 🔄 Override via API

```python
# Override for specific backtest
config_dict = {
    'total_capital': 1000000,
    'daily_stop_loss_check': False  # ← Override here
}

backtester = RSETFStrategyBacktester(db=db, config_dict=config_dict)
```

## 📝 Log Messages

### Daily Mode
```
Stop Loss Mode: Daily Check
⚠️ STOP LOSS HIT: RELIANCE - Current: ₹2,450.00 <= Stop Loss: ₹2,465.00
```

### Weekly Mode
```
Stop Loss Mode: Weekly Check (Signal Day Only)
📊 Weekly Stop Loss Summary: 3 position(s) to exit
  Combined exits: 5 total (RS signals + stop loss)
```

## ⚡ Quick Commands

```bash
# View config
cat Strategies/RS/rs_config.json

# Test config loader
python Strategies/RS/rs_config_loader.py

# Read documentation
cat Strategies/RS/README.md
```

## 🎓 Key Concepts

**Daily Mode:**
- Check → Execute immediately
- Mon-Fri: Independent stop loss exits
- Friday: Generate signals for Monday

**Weekly Mode:**
- Mon-Thu: Check → Accumulate
- Friday: Check → Combine with RS signals
- Monday: Execute everything together

## 🚀 Default Behavior

**Out of the box:** Daily stop loss check (existing behavior)
**To change:** Edit `rs_config.json` and set `daily_stop_loss_check: false`

---

For detailed documentation, see `README.md`
For implementation details, see `IMPLEMENTATION_SUMMARY.md`
