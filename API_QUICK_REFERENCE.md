# WealthAI API - Quick Reference Guide

## 🚀 Getting Started

### Base URL
```
http://localhost:8000
```

### Authentication
```
Authorization: Bearer <google_oauth_token>
```

---

## 📊 Most Used Endpoints

### 1. Run Backtest (Centralized)
```bash
POST /api/run_backtest
Content-Type: application/json

{
  "strategy_type": "ETF_Rotation",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "tickers": ["NIFTYBEES", "BANKBEES"],
  "capital_per_week": 50000,
  "accumulation_weeks": 52,
  "brokerage_percent": 0.1
}
```

### 2. Calculate Available Date Range
```bash
POST /api/rs-etf-strategy/date-range
Content-Type: application/json

{
  "main_index": "^NSEI",
  "etf_universe": "ALL_ETFS",
  "lookback_weeks": 5,
  "lookback_months": 20,
  "lookback_quarters": 60
}
```

### 3. Get Transaction Costs
```bash
GET /api/rs-etf-strategy/backtests/{backtest_id}/costs
```

### 4. Get Backtest Results
```bash
GET /api/rs-etf-strategy/backtests/{backtest_id}
```

### 5. Get Trade History
```bash
GET /api/rs-etf-strategy/backtests/{backtest_id}/trades
```

---

## 🔑 Strategy Types

| Type | Description |
|------|-------------|
| `ETF_Rotation` | Weekly SIP ETF rotation |
| `RS_ETF_Rotation` | Relative Strength ETF |
| `International_ETF_Rotation` | US ETF rotation |
| `Rotation_Stocks` | Stock rotation |
| `ETF_Payout` | ETF with withdrawals |
| `SuperTrend` | Technical indicator strategy |

---

## 📈 Common Metrics

All backtest responses include:

- `total_return_pct`: Total return percentage
- `cagr_pct`: Compound Annual Growth Rate
- `sharpe_ratio`: Risk-adjusted return
- `max_drawdown_pct`: Maximum loss from peak
- `win_rate_pct`: Percentage of winning trades
- `total_trades`: Number of trades executed

---

## 🛠️ Utility Endpoints

### Market Data
```bash
# List all symbols
GET /api/rs-etf-strategy/market-data/symbols

# Get ETF data
GET /api/rs-etf-strategy/market-data/etf/NIFTYBEES.NS

# Get index data
GET /api/rs-etf-strategy/market-data/index/^NSEI
```

### Health Checks
```bash
GET /api/health
GET /api/centralized/health
GET /api/rs-etf-strategy/health
```

---

## 🔐 Authentication Flow

```bash
# 1. Login with Google
POST /api/auth/google-login
{
  "token": "google_oauth_token",
  "phone_no": "+919876543210"
}

# 2. Get user info
GET /api/auth/user-info
Authorization: Bearer <token>
```

---

## 📋 Response Format

### Success
```json
{
  "success": true,
  "data": { ... }
}
```

### Error
```json
{
  "success": false,
  "error": "Error message",
  "detail": "Detailed information"
}
```

---

## 🎯 Common Workflows

### Workflow 1: Run Complete Backtest Analysis

```bash
# Step 1: Check data availability
POST /api/rs-etf-strategy/date-range

# Step 2: Run backtest
POST /api/run_backtest

# Step 3: Get detailed results
GET /api/rs-etf-strategy/backtests/{backtest_id}

# Step 4: Analyze trades
GET /api/rs-etf-strategy/backtests/{backtest_id}/trades

# Step 5: Check costs
GET /api/rs-etf-strategy/backtests/{backtest_id}/costs

# Step 6: View portfolio evolution
GET /api/rs-etf-strategy/backtests/{backtest_id}/portfolio
```

### Workflow 2: SuperTrend Strategy

```bash
# Step 1: Get current config
GET /api/config

# Step 2: Update config (optional)
PUT /api/config

# Step 3: Run backtest
POST /api/run/backtest

# Step 4: Get buy/sell candidates
GET /api/candidates

# Step 5: View current positions
GET /api/positions
```

---

## 🔢 Parameter Reference

### ETF Rotation Parameters
```json
{
  "tickers": ["NIFTYBEES", "BANKBEES"],
  "capital_per_week": 50000,
  "accumulation_weeks": 52,
  "brokerage_percent": 0.1,
  "compounding_enabled": false,
  "risk_free_rate": 8.0
}
```

### RS ETF Parameters
```json
{
  "total_capital": 1000000,
  "etf_universe": "ALL_ETFS",
  "max_positions": 20,
  "lookback_weeks": 5,
  "lookback_months": 20,
  "lookback_quarters": 60,
  "stop_loss_pct": 15.0,
  "buffer_capital_pct": 10.0
}
```

---

## 📚 Full Documentation

See [API_REFERENCE.md](./API_REFERENCE.md) for complete documentation.

---

## 💡 Tips

1. **Always check date range** before running backtests
2. **Use centralized API** (`/api/run_backtest`) for simplicity
3. **Monitor transaction costs** - they significantly impact returns
4. **Test with small date ranges** first
5. **Check health endpoints** if experiencing issues

---

## 🐛 Troubleshooting

### Common Issues

**404 Not Found**
- Check endpoint URL spelling
- Ensure server is running
- Verify backtest_id exists

**422 Validation Error**
- Check required parameters
- Verify parameter types
- See error details in response

**500 Internal Error**
- Check server logs
- Verify database connection
- Contact support

---

## 📞 Support

- **Swagger Docs**: http://localhost:8000/docs
- **Email**: support@wealthai.com
- **Version**: 1.0.0
