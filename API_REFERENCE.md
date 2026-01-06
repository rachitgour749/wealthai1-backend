# WealthAI Backend - Complete API Reference

## Table of Contents
1. [Centralized Backtest API](#centralized-backtest-api)
2. [Date Range Calculation API](#date-range-calculation-api)
3. [Transaction Cost Analysis API](#transaction-cost-analysis-api)
4. [Market Data APIs](#market-data-apis)
5. [Backtest Management APIs](#backtest-management-apis)
6. [Strategy-Specific APIs](#strategy-specific-apis)
7. [Subscription & Auth APIs](#subscription--auth-apis)

---

## Centralized Backtest API

### POST `/api/run_backtest`
**Description**: Unified endpoint for running backtests across all strategy types.

**Request Body**:
```json
{
  "strategy_type": "ETF_Rotation | RS_ETF_Rotation | International_ETF_Rotation | Rotation_Stocks | ETF_Payout | SuperTrend",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  // ... strategy-specific parameters
}
```

**Response**:
```json
{
  "success": true,
  "strategy_type": "ETF_Rotation",
  "metrics": { /* performance metrics */ },
  "performance_data": { /* time series data */ },
  "transaction_log": [ /* trade history */ ]
}
```

**See**: [Centralized API README](./APIs/README.md) for detailed documentation.

---

## Date Range Calculation API

### POST `/api/rs-etf-strategy/date-range`
### POST `/api/rs-strategy/date-range`

**Description**: Calculate available date range for backtesting based on data availability.

**Request Body**:
```json
{
  "main_index": "^NSEI",
  "etf_universe": "ALL_ETFS",  // or stock_universe for stocks
  "custom_etfs": ["NIFTYBEES.NS", "BANKBEES.NS"],  // optional
  "lookback_weeks": 5,
  "lookback_months": 20,
  "lookback_quarters": 60
}
```

**Response**:
```json
{
  "success": true,
  "available_start_date": "2020-01-01",
  "available_end_date": "2024-12-31",
  "total_days": 1826,
  "data_coverage": {
    "main_index": {
      "start_date": "2020-01-01",
      "end_date": "2024-12-31",
      "total_records": 1200
    },
    "etfs": {
      "NIFTYBEES.NS": {
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "total_records": 1200
      }
    }
  },
  "warnings": []
}
```

**Use Case**: Call this before running a backtest to determine valid date ranges.

---

## Transaction Cost Analysis API

### GET `/api/rs-etf-strategy/backtests/{backtest_id}/costs`
### GET `/api/rs-strategy/backtests/{backtest_id}/costs`

**Description**: Get detailed transaction cost breakdown for a completed backtest.

**Path Parameters**:
- `backtest_id` (string): Unique identifier of the backtest

**Response**:
```json
{
  "success": true,
  "backtest_id": "abc123",
  "cost_summary": {
    "total_transaction_costs": 15000.50,
    "total_brokerage": 12000.00,
    "total_taxes": 3000.50,
    "cost_as_percentage_of_capital": 1.5,
    "cost_as_percentage_of_returns": 3.2
  },
  "cost_breakdown": {
    "buy_transactions": {
      "count": 45,
      "total_value": 500000,
      "total_costs": 7500
    },
    "sell_transactions": {
      "count": 40,
      "total_value": 480000,
      "total_costs": 7200
    }
  },
  "monthly_costs": [
    {
      "month": "2024-01",
      "total_costs": 1200,
      "transaction_count": 8
    }
  ],
  "cost_impact_analysis": {
    "returns_without_costs": 45000,
    "returns_with_costs": 30000,
    "cost_drag_percentage": 33.3
  }
}
```

**Use Case**: Analyze how transaction costs impact strategy performance.

---

## Market Data APIs

### GET `/api/rs-etf-strategy/market-data/symbols`
### GET `/api/rs-strategy/market-data/symbols`

**Description**: Get list of all available symbols in the database.

**Response**:
```json
{
  "success": true,
  "symbols": ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"],
  "total_count": 284,
  "last_updated": "2024-12-31T23:59:59Z"
}
```

---

### GET `/api/rs-etf-strategy/market-data/etf/{symbol}`
### GET `/api/rs-strategy/market-data/stock/{symbol}`

**Description**: Get historical price data for a specific symbol.

**Path Parameters**:
- `symbol` (string): ETF/Stock symbol (e.g., "NIFTYBEES.NS")

**Query Parameters**:
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)
- `limit` (optional): Maximum number of records (default: 1000)

**Response**:
```json
{
  "success": true,
  "symbol": "NIFTYBEES.NS",
  "data": [
    {
      "date": "2024-01-01",
      "open": 100.50,
      "high": 102.00,
      "low": 100.00,
      "close": 101.50,
      "volume": 1000000
    }
  ],
  "total_records": 250,
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-12-31"
  }
}
```

---

### GET `/api/rs-etf-strategy/market-data/index/{index_symbol}`

**Description**: Get historical index data.

**Path Parameters**:
- `index_symbol` (string): Index symbol (e.g., "^NSEI")

**Response**: Same format as ETF/Stock data above.

---

## Backtest Management APIs

### GET `/api/rs-etf-strategy/backtests`
### GET `/api/rs-strategy/backtests`

**Description**: Get list of all backtests for a user.

**Query Parameters**:
- `user_id` (required): User identifier
- `limit` (optional): Maximum results (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response**:
```json
{
  "success": true,
  "backtests": [
    {
      "backtest_id": "abc123",
      "user_id": "user@example.com",
      "strategy_type": "RS_ETF",
      "created_at": "2024-01-15T10:30:00Z",
      "status": "completed",
      "config": {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "total_capital": 1000000
      },
      "metrics": {
        "total_return": 45.5,
        "cagr": 12.3,
        "sharpe_ratio": 1.8
      }
    }
  ],
  "total_count": 15,
  "page": 1,
  "total_pages": 1
}
```

---

### GET `/api/rs-etf-strategy/backtests/{backtest_id}`
### GET `/api/rs-strategy/backtests/{backtest_id}`

**Description**: Get detailed results for a specific backtest.

**Path Parameters**:
- `backtest_id` (string): Unique backtest identifier

**Response**:
```json
{
  "success": true,
  "backtest_id": "abc123",
  "user_id": "user@example.com",
  "created_at": "2024-01-15T10:30:00Z",
  "status": "completed",
  "config": { /* backtest configuration */ },
  "metrics": {
    "total_return_pct": 45.5,
    "annualized_return_pct": 13.2,
    "cagr_pct": 12.3,
    "xirr_pct": 12.5,
    "max_drawdown_pct": -15.2,
    "sharpe_ratio": 1.8,
    "sortino_ratio": 2.1,
    "calmar_ratio": 0.8,
    "win_rate_pct": 65.4,
    "total_trades": 156,
    "avg_trade_return_pct": 2.3,
    "final_capital": 1455000,
    "total_invested": 1000000,
    "absolute_profit": 455000
  },
  "performance_data": {
    "dates": ["2020-01-01", "2020-01-08", ...],
    "portfolio_value": [1000000, 1020000, ...],
    "benchmark_value": [1000000, 1015000, ...]
  }
}
```

---

### GET `/api/rs-etf-strategy/backtests/{backtest_id}/trades`
### GET `/api/rs-strategy/backtests/{backtest_id}/trades`

**Description**: Get transaction log for a backtest.

**Response**:
```json
{
  "success": true,
  "backtest_id": "abc123",
  "trades": [
    {
      "trade_id": 1,
      "date": "2020-01-08",
      "action": "BUY",
      "symbol": "NIFTYBEES.NS",
      "quantity": 100,
      "price": 500.00,
      "total_value": 50000,
      "brokerage": 50,
      "taxes": 10,
      "total_cost": 50060,
      "portfolio_value_after": 1000000
    },
    {
      "trade_id": 2,
      "date": "2020-02-15",
      "action": "SELL",
      "symbol": "NIFTYBEES.NS",
      "quantity": 100,
      "price": 520.00,
      "total_value": 52000,
      "brokerage": 52,
      "taxes": 260,
      "net_proceeds": 51688,
      "profit_loss": 1628,
      "portfolio_value_after": 1001628
    }
  ],
  "total_trades": 156,
  "summary": {
    "total_buy_trades": 78,
    "total_sell_trades": 78,
    "total_buy_value": 5000000,
    "total_sell_value": 5455000,
    "total_brokerage": 5050,
    "total_taxes": 25000
  }
}
```

---

### GET `/api/rs-etf-strategy/backtests/{backtest_id}/portfolio`
### GET `/api/rs-strategy/backtests/{backtest_id}/portfolio`

**Description**: Get portfolio snapshots over time.

**Response**:
```json
{
  "success": true,
  "backtest_id": "abc123",
  "snapshots": [
    {
      "date": "2020-01-08",
      "total_value": 1000000,
      "cash": 950000,
      "invested": 50000,
      "positions": [
        {
          "symbol": "NIFTYBEES.NS",
          "quantity": 100,
          "avg_price": 500,
          "current_price": 500,
          "current_value": 50000,
          "unrealized_pnl": 0,
          "weight_pct": 5.0
        }
      ],
      "position_count": 1
    }
  ]
}
```

---

### DELETE `/api/rs-etf-strategy/backtests/{backtest_id}`
### DELETE `/api/rs-strategy/backtests/{backtest_id}`

**Description**: Delete a backtest.

**Path Parameters**:
- `backtest_id` (string): Backtest to delete

**Query Parameters**:
- `user_id` (required): User identifier for authorization

**Response**:
```json
{
  "success": true,
  "message": "Backtest deleted successfully",
  "backtest_id": "abc123"
}
```

---

## Strategy-Specific APIs

### ETF Universe API

#### GET `/api/rs-etf-strategy/etfs/universe/{universe}`

**Description**: Get list of ETFs in a specific universe.

**Path Parameters**:
- `universe` (string): Universe name (e.g., "ALL_ETFS", "NIFTY_ETFS")

**Response**:
```json
{
  "success": true,
  "universe": "ALL_ETFS",
  "etfs": [
    {
      "symbol": "NIFTYBEES.NS",
      "name": "Nippon India ETF Nifty BeES",
      "category": "Equity",
      "tracking_index": "NIFTY 50"
    }
  ],
  "total_count": 284
}
```

---

### Stock Universe API

#### GET `/api/rs-strategy/stocks/universe/{universe}`

**Description**: Get list of stocks in a specific universe.

**Path Parameters**:
- `universe` (string): "NIFTY50", "NIFTY500", "ALL_STOCKS"

**Response**:
```json
{
  "success": true,
  "universe": "NIFTY50",
  "stocks": [
    {
      "symbol": "RELIANCE.NS",
      "name": "Reliance Industries Ltd",
      "sector": "Energy",
      "market_cap": 1500000000000
    }
  ],
  "total_count": 50
}
```

---

### SuperTrend Strategy APIs

#### GET `/api/config`

**Description**: Get current SuperTrend strategy configuration.

**Response**:
```json
{
  "ema_short": 10,
  "ema_long": 20,
  "supertrend_period": 10,
  "supertrend_stop_pct": 10.0,
  "max_holdings": 5,
  "buffer_pct": 10.0,
  "price_floor": 50.0,
  "liquidity_cr": 100.0,
  "rs_window_1": 5,
  "rs_window_2": 20,
  "rs_window_3": 60,
  "benchmark": "^NSEI",
  "universe": "NIFTY500"
}
```

---

#### PUT `/api/config`

**Description**: Update SuperTrend strategy configuration.

**Request Body**: Same as GET response above.

**Response**:
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "config": { /* updated config */ }
}
```

---

#### POST `/api/run/backtest`

**Description**: Run SuperTrend backtest.

**Request Body**:
```json
{
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 1000000,
  "brokerage_pct": 0.1,
  "buffer_pct": 10.0
}
```

**Response**: Similar to other backtest responses.

---

#### GET `/api/candidates`

**Description**: Get current buy/sell candidates based on SuperTrend signals.

**Response**:
```json
{
  "success": true,
  "date": "2024-01-15",
  "buy_candidates": [
    {
      "symbol": "RELIANCE.NS",
      "current_price": 2500,
      "ema_short": 2480,
      "ema_long": 2450,
      "supertrend": 2400,
      "rs_score": 85.5,
      "signal_strength": "STRONG"
    }
  ],
  "sell_candidates": [],
  "total_buy": 5,
  "total_sell": 0
}
```

---

#### GET `/api/positions`

**Description**: Get current positions in SuperTrend strategy.

**Response**:
```json
{
  "success": true,
  "positions": [
    {
      "symbol": "RELIANCE.NS",
      "quantity": 40,
      "avg_price": 2400,
      "current_price": 2500,
      "current_value": 100000,
      "unrealized_pnl": 4000,
      "pnl_pct": 4.17,
      "holding_days": 15
    }
  ],
  "total_positions": 5,
  "total_invested": 480000,
  "current_value": 500000,
  "total_pnl": 20000
}
```

---

## Subscription & Auth APIs

### POST `/api/auth/google-login`

**Description**: Authenticate user with Google OAuth token.

**Request Body**:
```json
{
  "token": "google_oauth_token_here",
  "phone_no": "+919876543210"  // optional
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "user_email": "user@example.com",
    "user_name": "John Doe",
    "status": "active",
    "phone_no": "+919876543210",
    "is_new_user": false,
    "token": "google_oauth_token_here",
    "message": "Welcome back!"
  }
}
```

---

### GET `/api/auth/user-info`

**Description**: Get current user information.

**Headers**:
- `Authorization`: `Bearer <google_oauth_token>`

**Response**:
```json
{
  "success": true,
  "data": {
    "user_email": "user@example.com",
    "user_name": "John Doe",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "status": "active",
    "phone_no": "+919876543210"
  }
}
```

---

## Health Check APIs

### GET `/api/rs-etf-strategy/health`
### GET `/api/rs-strategy/health`
### GET `/api/health`
### GET `/api/centralized/health`

**Description**: Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "service": "RS ETF Strategy API",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

---

### GET `/api/rs-etf-strategy/health/database`
### GET `/api/rs-strategy/health/database`

**Description**: Check database connection health.

**Response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "total_records": {
    "etfs": 284,
    "index_data": 1200,
    "backtests": 150
  },
  "last_data_update": "2024-01-15T00:00:00Z"
}
```

---

## Common Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing token |
| 404 | Not Found - Resource doesn't exist |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error |

---

## Common Error Response Format

```json
{
  "success": false,
  "error": "Error message here",
  "detail": "Detailed error information",
  "code": "ERROR_CODE"
}
```

---

## Rate Limiting

Most endpoints support standard rate limiting:
- **Backtest APIs**: 10 requests per minute
- **Market Data APIs**: 100 requests per minute
- **Auth APIs**: 20 requests per minute

---

## Authentication

Most endpoints require authentication via Google OAuth token:

```
Authorization: Bearer <google_oauth_token>
```

Or via query parameter:
```
?user_id=user@example.com
```

---

## Pagination

List endpoints support pagination:

**Query Parameters**:
- `limit`: Number of results per page (default: 50, max: 100)
- `offset`: Number of results to skip (default: 0)

**Response includes**:
```json
{
  "total_count": 150,
  "page": 1,
  "total_pages": 3,
  "results": [...]
}
```

---

## Quick Reference

### Most Common Workflows

**1. Run a Backtest**:
```
1. POST /api/run_backtest (centralized)
   OR
   POST /api/rs-etf-strategy/backtests/run (strategy-specific)

2. GET /api/rs-etf-strategy/backtests/{backtest_id} (get results)

3. GET /api/rs-etf-strategy/backtests/{backtest_id}/trades (view trades)

4. GET /api/rs-etf-strategy/backtests/{backtest_id}/costs (analyze costs)
```

**2. Check Data Availability**:
```
1. POST /api/rs-etf-strategy/date-range (check available dates)

2. GET /api/rs-etf-strategy/market-data/symbols (list symbols)

3. GET /api/rs-etf-strategy/market-data/etf/{symbol} (get price data)
```

**3. User Authentication**:
```
1. POST /api/auth/google-login (login)

2. GET /api/auth/user-info (get user details)
```

---

## Support

For API support, contact: support@wealthai.com

**Swagger Documentation**: `http://localhost:8000/docs`

**API Version**: 1.0.0

**Last Updated**: 2026-01-04
