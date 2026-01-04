# Centralized Backtest API - Documentation

## Overview

The Centralized Backtest API provides a **single endpoint** (`/api/centralized/run_backtest`) to execute backtests for all strategy types. This eliminates the need to call different endpoints for different strategies.

## 📁 Folder Structure

```
d:\WEALTHAI_PROD\New folder\
├── APIs/                          # NEW: Centralized API endpoints
│   ├── __init__.py
│   ├── unified_schemas.py         # Unified request/response schemas
│   └── centralized_backtest.py    # Main centralized endpoint
│
├── Handlers/                      # NEW: Strategy-specific handlers
│   ├── __init__.py
│   ├── base_handler.py            # Base handler interface
│   ├── etf_rotation_handler.py    # ETF Rotation handler
│   ├── rs_etf_handler.py          # RS ETF handler
│   ├── international_etf_handler.py
│   ├── rotation_stocks_handler.py
│   ├── etf_payout_handler.py
│   └── supertrend_handler.py
│
└── Strategies/                    # EXISTING: Old APIs still work
    ├── Rotation_ETF/
    ├── RS_ETF/
    ├── Rotation_International_ETF/
    ├── Rotation_Stocks/
    ├── CustomStrategies/Rotation_ETF_Payout/
    └── SuperTrend/
```

## 🚀 API Endpoint

### Base URL
```
POST /api/centralized/run_backtest
```

### Authentication
Bearer Token (Google OAuth)

## 📝 Request Format

### Common Required Fields
```json
{
  "strategy_type": "ETF_Rotation | RS_ETF_Rotation | International_ETF_Rotation | Rotation_Stocks | ETF_Payout | SuperTrend",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD"
}
```

## 📊 Strategy-Specific Examples

### 1. ETF Rotation
```json
{
  "strategy_type": "ETF_Rotation",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "tickers": ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"],
  "capital_per_week": 50000,
  "accumulation_weeks": 52,
  "brokerage_percent": 0.1,
  "compounding_enabled": false,
  "risk_free_rate": 8.0
}
```

### 2. RS ETF Rotation
```json
{
  "strategy_type": "RS_ETF_Rotation",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "total_capital": 1000000,
  "etf_universe": "ALL_ETFS",
  "max_positions": 20,
  "lookback_weeks": 5,
  "lookback_months": 20,
  "lookback_quarters": 60,
  "risk_free_rate": 8.0
}
```

### 3. International ETF Rotation
```json
{
  "strategy_type": "International_ETF_Rotation",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "tickers": ["SPY", "QQQ", "IWM"],
  "capital_per_week": 50000,
  "accumulation_weeks": 52,
  "brokerage_percent": 0.1,
  "compounding_enabled": false,
  "risk_free_rate": 8.0
}
```

### 4. Rotation Stocks
```json
{
  "strategy_type": "Rotation_Stocks",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
  "capital_per_week": 50000,
  "accumulation_weeks": 52,
  "brokerage_percent": 0.1,
  "compounding_enabled": false,
  "risk_free_rate": 8.0
}
```

### 5. ETF Payout
```json
{
  "strategy_type": "ETF_Payout",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "tickers": ["NIFTYBEES.NS", "BANKBEES.NS"],
  "capital_per_week": 50000,
  "accumulation_weeks": 52,
  "brokerage_percent": 0.1,
  "withdraw_amount": 10000,
  "payout_start_week": 10,
  "compounding_enabled": false,
  "risk_free_rate": 8.0
}
```

### 6. SuperTrend
```json
{
  "strategy_type": "SuperTrend",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 1000000,
  "brokerage_pct": 0.1,
  "buffer_pct": 10.0,
  "ema_short": 10,
  "ema_long": 20,
  "max_holdings": 5
}
```

## 📤 Response Format

```json
{
  "success": true,
  "strategy_type": "ETF_Rotation",
  "metrics": {
    "total_return": 45.5,
    "cagr_pct": 12.3,
    "sharpe_ratio": 1.8,
    "max_drawdown": -15.2,
    "total_trades": 156,
    "win_rate_pct": 65.4,
    "final_capital": 1455000
  },
  "performance_data": {
    "dates": ["2020-01-01", "2020-01-08", ...],
    "etf_strategy": [1000000, 1050000, ...],
    "cumulative_investment": [50000, 100000, ...],
    "benchmark_buyhold": [1000000, 1020000, ...]
  },
  "transaction_log": [
    {
      "week": 1,
      "date": "2020-01-08",
      "action": "BUY",
      "ticker": "NIFTYBEES.NS",
      "units": 100,
      "price": 500,
      "amount": 50000,
      "transaction_costs": 50,
      "nav": 1000000
    }
  ]
}
```

## 🔍 Additional Endpoints

### Health Check
```
GET /api/centralized/health
```

Response:
```json
{
  "status": "healthy",
  "service": "Centralized Backtest API",
  "supported_strategies": [
    "ETF_Rotation",
    "RS_ETF_Rotation",
    "International_ETF_Rotation",
    "Rotation_Stocks",
    "ETF_Payout",
    "SuperTrend"
  ]
}
```

### List Strategies
```
GET /api/centralized/strategies
```

Returns detailed information about all supported strategies and their parameters.

## ⚙️ Parameter Reference

### Rotation Strategies (ETF_Rotation, International_ETF_Rotation, Rotation_Stocks)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tickers` | List[str] | ✅ | List of asset tickers |
| `start_date` | str | ✅ | Backtest start date |
| `end_date` | str | ✅ | Backtest end date |
| `capital_per_week` | float | ✅ | Weekly investment amount |
| `accumulation_weeks` | int | ✅ | Number of weeks to invest |
| `brokerage_percent` | float | ✅ | Brokerage percentage |
| `compounding_enabled` | bool | ❌ | Enable compounding (default: false) |
| `risk_free_rate` | float | ❌ | Risk-free rate (default: 8.0) |

### RS Strategies (RS_ETF_Rotation)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_date` | str | ✅ | Backtest start date |
| `end_date` | str | ✅ | Backtest end date |
| `total_capital` | float | ✅ | Total capital |
| `etf_universe` | str | ❌ | ETF universe (default: "ALL_ETFS") |
| `custom_etfs` | List[str] | ❌ | Custom ETF selection |
| `max_positions` | int | ❌ | Max positions (default: 20) |
| `lookback_weeks` | int | ❌ | RS lookback weeks (default: 5) |
| `lookback_months` | int | ❌ | RS lookback months (default: 20) |
| `lookback_quarters` | int | ❌ | RS lookback quarters (default: 60) |

### ETF Payout (Additional Parameters)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `withdraw_amount` | float | ❌ | Withdrawal amount per churning week |
| `payout_start_week` | int | ❌ | Week to start payouts (default: 1) |

### SuperTrend

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_date` | str | ✅ | Backtest start date |
| `end_date` | str | ✅ | Backtest end date |
| `initial_capital` | float | ✅ | Initial capital |
| `brokerage_pct` | float | ❌ | Brokerage percentage (default: 0.1) |
| `buffer_pct` | float | ❌ | Buffer percentage (default: 10.0) |
| `ema_short` | int | ❌ | Short EMA period (default: 10) |
| `ema_long` | int | ❌ | Long EMA period (default: 20) |
| `max_holdings` | int | ❌ | Max holdings (default: 5) |
| `symbols` | List[str] | ❌ | Custom symbol selection |

## 🔄 Backward Compatibility

**All existing APIs remain functional!** The old endpoints are still available:

- `/api/metrics` - ETF Rotation (old)
- `/api/rs-etf-strategy/backtests/run` - RS ETF (old)
- `/api/international-etf/metrics` - International ETF (old)
- `/api/stocks/metrics` - Rotation Stocks (old)
- `/api/etf-payout/metrics` - ETF Payout (old)
- `/api/supertrend/backtest` - SuperTrend (old)

## 🧪 Testing

### Using cURL
```bash
curl -X POST "http://localhost:8000/api/centralized/run_backtest" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "strategy_type": "ETF_Rotation",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "tickers": ["NIFTYBEES.NS", "BANKBEES.NS"],
    "capital_per_week": 50000,
    "accumulation_weeks": 52,
    "brokerage_percent": 0.1
  }'
```

### Using Python
```python
import requests

url = "http://localhost:8000/api/centralized/run_backtest"
headers = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}
data = {
    "strategy_type": "ETF_Rotation",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "tickers": ["NIFTYBEES.NS", "BANKBEES.NS"],
    "capital_per_week": 50000,
    "accumulation_weeks": 52,
    "brokerage_percent": 0.1
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## 🎯 Benefits

1. **Single Endpoint**: One API for all strategies
2. **Type Safety**: Pydantic validation ensures correct parameters
3. **Clear Documentation**: Auto-generated Swagger docs
4. **Backward Compatible**: Old APIs still work
5. **Easy to Extend**: Add new strategies by creating a handler
6. **Consistent Response**: Unified response format across all strategies

## 🐛 Error Handling

### Validation Errors (400)
```json
{
  "detail": [
    {
      "loc": ["body", "tickers"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Backtest Errors (200 with success: false)
```json
{
  "success": false,
  "strategy_type": "ETF_Rotation",
  "metrics": {},
  "error": "ETF Rotation backtest failed: No data available for selected tickers"
}
```

## 📚 Swagger Documentation

Access the interactive API documentation at:
```
http://localhost:8000/docs
```

Login credentials:
- Username: `wealthwisers@fintech.gmail.com`
- Password: `WW@fintech.2025`

## 🔧 Troubleshooting

### Import Errors
If you see import errors, ensure:
1. `APIs` and `Handlers` folders are in the root directory
2. Both folders have `__init__.py` files
3. Server is restarted after adding new files

### Strategy Not Found
Verify `strategy_type` matches exactly (case-sensitive):
- `ETF_Rotation` ✅
- `etf_rotation` ❌

### Missing Parameters
Check the parameter reference for your strategy type. Required parameters must be provided.
