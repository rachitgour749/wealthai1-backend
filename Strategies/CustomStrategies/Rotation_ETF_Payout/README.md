# Rotation ETF Payout Strategy

A custom ETF rotation strategy with systematic withdrawal/payout feature, based on the standard Rotation ETF strategy.

## Overview

This strategy extends the classic ETF rotation approach by adding a **systematic withdrawal feature** during the churning phase. It's designed for scenarios where you want to:

- Build a portfolio through regular investments (accumulation phase)
- Generate regular income through systematic withdrawals (churning phase)
- Track total withdrawals over the backtest period

## Key Concept

### Accumulation Phase
During the accumulation phase (first N weeks):
- **Buy** ETFs worth `accumulation_per_week` amount every week
- Select ETF with smallest distance from 52-week low (momentum-based selection)
- Build up the portfolio systematically

### Churning Phase
After accumulation is complete:
- **Sell** ETFs worth `accumulation_per_week + withdraw_amount` every week
- The `accumulation_per_week` portion is reinvested (standard rotation)
- The `withdraw_amount` portion is withdrawn from the portfolio
- Track cumulative withdrawals throughout the backtest

## Configuration Parameters

Edit `config.json` to customize the strategy:

```json
{
  "accumulation_weeks": 13,           // Number of weeks to accumulate
  "accumulation_per_week": 50000,     // Weekly investment during accumulation
  "withdraw_amount": 25000,           // Weekly withdrawal during churning
  "selected_etfs": [...],             // List of ETFs to trade
  "brokerage_percent": 0.03,          // Transaction cost percentage
  "compounding_enabled": true         // Enable dynamic churning amounts
}
```

### Parameter Details

| Parameter | Description | Example |
|-----------|-------------|---------|
| `accumulation_weeks` | Number of weeks to build the portfolio | 13, 26, 52 |
| `accumulation_per_week` | Amount to invest each week during accumulation | 50000 |
| `withdraw_amount` | Amount to withdraw each week during churning | 25000 |
| `selected_etfs` | List of ETF symbols to trade | ["NIFTYBEES", "BANKBEES", ...] |
| `brokerage_percent` | Transaction costs as percentage | 0.03 (0.03%) |
| `compounding_enabled` | Adjust churning based on portfolio performance | true/false |

## Example Scenario

**Configuration:**
- Accumulation weeks: 13
- Accumulation per week: ₹50,000
- Withdrawal amount: ₹25,000

**Behavior:**

**Weeks 1-13 (Accumulation):**
- Week 1: Buy ETFs worth ₹50,000
- Week 2: Buy ETFs worth ₹50,000
- ...
- Week 13: Buy ETFs worth ₹50,000
- **Total invested: ₹6,50,000**

**Week 14+ (Churning with Withdrawal):**
- Week 14: Sell ETFs worth ₹75,000 (₹50,000 reinvested + ₹25,000 withdrawn)
- Week 15: Sell ETFs worth ₹75,000 (₹50,000 reinvested + ₹25,000 withdrawn)
- ...
- **Cumulative withdrawal tracked and reported**

## Usage

### Python API

```python
from Strategies.CustomStrategies.Rotation_ETF_Payout import RotationETFPayoutBacktester

# Create backtester
backtester = RotationETFPayoutBacktester()

# Run backtest
results = backtester.run_backtest(
    start_date='2020-01-01',
    end_date='2024-12-31',
    selected_etfs=['NIFTYBEES', 'JUNIORBEES', 'BANKBEES', 'ITBEES', 'GOLDBEES'],
    accumulation_weeks=13,
    accumulation_per_week=50000,
    withdraw_amount=25000,
    brokerage_percent=0.03,
    compounding_enabled=True
)

# Access results
print(f"Total Withdrawn: ₹{results['results']['total_withdrawn']:,.0f}")
print(f"Final NAV: ₹{results['results']['final_nav']:,.0f}")
print(f"ROI: {results['results']['roi_percent']:.2f}%")
```

### REST API

**Endpoint:** `POST /api/rotation-etf-payout/backtest`

**Request:**
```json
{
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "selected_etfs": ["NIFTYBEES", "JUNIORBEES", "BANKBEES"],
  "accumulation_weeks": 13,
  "accumulation_per_week": 50000,
  "withdraw_amount": 25000,
  "brokerage_percent": 0.03,
  "compounding_enabled": true
}
```

**Response:**
```json
{
  "success": true,
  "strategy": "Rotation_ETF_Payout",
  "results": {
    "total_invested": 650000,
    "total_withdrawn": 1250000,
    "final_nav": 850000,
    "net_gain": 450000,
    "roi_percent": 69.23
  },
  "withdrawal_log": [
    {
      "week": 14,
      "date": "2020-04-06",
      "withdrawal_amount": 25000,
      "cumulative_withdrawn": 25000
    },
    ...
  ]
}
```

## Understanding Results

### Key Metrics

- **Total Invested**: Sum of all investments during accumulation phase
- **Total Withdrawn**: Sum of all withdrawals during churning phase ⭐ **NEW**
- **Final NAV**: Current portfolio value (cash + holdings)
- **Net Gain**: Final NAV - Total Invested + Total Withdrawn
- **ROI**: (Net Gain / Total Invested) × 100

### Withdrawal Log

The `withdrawal_log` array tracks each withdrawal:
```json
{
  "week": 14,
  "date": "2020-04-06",
  "withdrawal_amount": 25000,
  "cumulative_withdrawn": 25000
}
```

## Comparison with Standard Rotation ETF

| Feature | Rotation ETF | Rotation ETF Payout |
|---------|--------------|---------------------|
| Accumulation | ✅ Yes | ✅ Yes |
| Churning | ✅ Yes | ✅ Yes |
| Withdrawals | ❌ No | ✅ Yes |
| Withdrawal Tracking | ❌ No | ✅ Yes |
| Use Case | Growth | Income + Growth |

## Use Cases

1. **Retirement Planning**: Simulate systematic withdrawals from retirement portfolio
2. **Income Generation**: Test regular income generation strategies
3. **SWP Analysis**: Analyze Systematic Withdrawal Plans
4. **Pension Simulation**: Model pension-like regular payouts

## Files Structure

```
Rotation_ETF_Payout/
├── __init__.py          # Package initialization
├── config.json          # Configuration file
├── backtester.py        # Main backtester class
├── api_routes.py        # FastAPI routes
└── README.md           # This file
```

## Notes

- The withdrawal amount is added on top of the standard churning amount
- All transaction costs and taxes are calculated as per Indian market rules
- Compounding can be enabled to dynamically adjust churning amounts based on portfolio performance
- The strategy uses 52-week momentum for ETF selection (same as base Rotation ETF)

## Support

For issues or questions, refer to the main Rotation ETF strategy documentation or contact the development team.
