# RS Strategy Configuration

This directory contains centralized configuration for both **RS_ETF** and **RS_Stocks** strategies.

## Configuration Files

### `rs_config.json`
Main configuration file that controls strategy behavior for both RS_ETF and RS_Stocks.

### `rs_config_loader.py`
Python utility module that loads and manages the configuration settings.

## Stop Loss Configuration

The most important configuration is the **stop loss check frequency**:

### Daily Stop Loss Check (Default: `true`)
```json
{
  "stop_loss_settings": {
    "daily_stop_loss_check": true
  }
}
```

**Behavior:**
- Stop loss is checked **every trading day**
- Positions that hit stop loss are **sold immediately** (same day)
- Provides **better risk protection** but may increase trading frequency
- Trade reason: `"Stop Loss (Daily)"`

### Weekly Stop Loss Check (`false`)
```json
{
  "stop_loss_settings": {
    "daily_stop_loss_check": false
  }
}
```

**Behavior:**
- Stop loss is checked **daily** but exits are **accumulated**
- All stop loss exits are executed on **Monday** (execution day) along with RS signal exits
- Aligns with the strategy's **weekly rebalancing philosophy**
- **Higher risk** - positions that drop mid-week are held until Friday/Monday
- Trade reason: Combined with regular exits on Monday

## How It Works

### Daily Mode Flow:
```
Monday:    Check SL → Execute SL exits immediately
Tuesday:   Check SL → Execute SL exits immediately
Wednesday: Check SL → Execute SL exits immediately
Thursday:  Check SL → Execute SL exits immediately
Friday:    Check SL → Execute SL exits immediately
           Generate RS signals → Store for Monday
Monday:    Execute RS signals (entries + exits)
```

### Weekly Mode Flow:
```
Monday:    Check SL → Accumulate
Tuesday:   Check SL → Accumulate
Wednesday: Check SL → Accumulate
Thursday:  Check SL → Accumulate
Friday:    Check SL → Accumulate
           Generate RS signals
           Combine SL exits + RS exits → Store for Monday
Monday:    Execute ALL exits + entries together
```

## Configuration Priority

The configuration is loaded with the following priority:

1. **API/Config Dict Override** - If `daily_stop_loss_check` is passed in `config_dict`, it takes highest priority
2. **Centralized Config File** - If not overridden, loads from `rs_config.json`
3. **Default Value** - If config file is missing, defaults to `true` (daily check)

## Usage Examples

### Example 1: Change to Weekly Stop Loss
Edit `rs_config.json`:
```json
{
  "stop_loss_settings": {
    "daily_stop_loss_check": false
  }
}
```

### Example 2: Override via API
```python
config_dict = {
    'total_capital': 1000000,
    'max_positions': 20,
    'stop_loss_pct': 15.0,
    'daily_stop_loss_check': False  # Override to weekly mode
}

backtester = RSETFStrategyBacktester(db=db, config_dict=config_dict)
```

### Example 3: Test Configuration Loader
```bash
cd Strategies/RS
python rs_config_loader.py
```

Output:
```
=== RS Strategy Configuration ===
Daily Stop Loss Check: True
Stop Loss %: 15.0
Capital Reset Threshold %: 25.0
Buffer Capital %: 10.0
Max Positions: 20
Transaction Cost %: 0.1
```

## Strategy Defaults

The configuration file also contains default values for other strategy parameters:

```json
{
  "strategy_defaults": {
    "stop_loss_pct": 15.0,
    "capital_reset_threshold_pct": 25.0,
    "buffer_capital_pct": 10.0,
    "max_positions": 20,
    "transaction_cost_pct": 0.1
  }
}
```

These can be used as fallback values or reference values for the strategies.

## Logging

When a strategy initializes, it will log the stop loss mode:

```
Stop Loss Mode: Daily Check
```
or
```
Stop Loss Mode: Weekly Check (Signal Day Only)
```

## Recommendations

### Use Daily Check When:
- Risk management is the top priority
- You want immediate protection against losses
- Trading costs are not a major concern
- You prefer tighter risk control

### Use Weekly Check When:
- You want pure weekly rebalancing
- Minimizing trading frequency is important
- You're comfortable with higher intra-week risk
- You want all trades to execute together on Monday

## Impact on Backtest Results

Changing from daily to weekly stop loss will affect:
- **Total Trades**: Weekly mode will have fewer trades
- **Max Drawdown**: Weekly mode may have higher drawdowns
- **Transaction Costs**: Weekly mode will have lower costs
- **Win Rate**: May vary depending on market conditions
- **CAGR**: Could be higher or lower depending on the period

**Recommendation**: Backtest both modes and compare results before choosing one for live trading.

## Technical Details

### Files Modified:
- `RS_ETF/rs_etf_backtester_core.py` - Added conditional stop loss logic
- `RS_Stocks/rs_backtester_core.py` - Added conditional stop loss logic

### New Attributes:
- `self.daily_stop_loss_check` - Boolean flag
- `self.weekly_stop_loss_exits` - List to accumulate weekly stop loss exits

### Methods Affected:
- `__init__()` - Loads configuration
- `run_backtest()` - Implements conditional logic

## Troubleshooting

### Configuration Not Loading
If you see "Using default configuration", check:
1. File exists at `Strategies/RS/rs_config.json`
2. JSON syntax is valid
3. File permissions allow reading

### Stop Loss Mode Not Changing
1. Check the log message during initialization
2. Verify `rs_config.json` has correct value
3. Check if API is overriding with `config_dict`
4. Restart the application to reload config

## Version History

- **v1.0** (2025-12-18): Initial implementation with configurable daily/weekly stop loss
