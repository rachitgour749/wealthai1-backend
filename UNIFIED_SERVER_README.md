# Unified Rotation Backtester API Server

## Overview

This unified server combines both **Stock Strategy** and **ETF Strategy** backtesting functionality into a single FastAPI application running on one port. The server maintains separate routing for each strategy while keeping the same API paths as the original separate servers.

## Architecture

### File Structure
```
wealthai1-backend-main/
├── unified_server.py          # Main unified server file
├── stockstrategy/
│   ├── stockserver.py         # Original stock server (now replaced)
│   ├── stockbacktester_core.py # Stock backtester core logic
│   └── ...
├── etf-strategy/
│   ├── server.py              # Original ETF server (now replaced)
│   ├── backtester_core.py     # ETF backtester core logic
│   └── ...
├── chatAI/
│   ├── chat_api.py            # ChatAI API router
│   ├── chat_core.py           # ChatAI core functionality
│   └── README.md              # ChatAI documentation
└── unified_etf_data.sqlite    # Database file
```

### Key Components

1. **Main FastAPI App**: `unified_server.py`
2. **Stock Router**: Handles all stock-related endpoints with prefix `/api/stocks`
3. **ETF Router**: Handles all ETF-related endpoints with prefix `/api/etfs`
4. **ChatAI Router**: Handles all ChatAI-related endpoints with prefix `/api`
5. **Shared Models**: Common Pydantic models for all strategies
6. **Health Check**: Unified health endpoint for all services

## API Endpoints

### Health & Status
- `GET /` - Root endpoint with server info
- `GET /health` - Health check for all services
- `GET /favicon.ico` - Favicon endpoint

### Stock Strategy Endpoints (`/api/stocks`)
All original stock endpoints are preserved with the same paths:

- `GET /api/stocks/` - List available stocks
- `GET /api/stocks/default` - Get default stock selection
- `POST /api/stocks/date-range` - Calculate common date range
- `POST /api/stocks/diagnose` - Diagnose stock data
- `GET /api/stocks/overview` - Get stock overview
- `POST /api/stocks/metrics` - Run stock backtest
- `GET /api/stocks/metrics/table` - Get metrics table
- `GET /api/stocks/transaction-costs/summary` - Transaction costs summary
- `GET /api/stocks/transaction-log` - Get transaction log
- `GET /api/stocks/transaction-costs` - Get transaction costs data
- `GET /api/stocks/skipped-trades` - Get skipped trades
- `GET /api/stocks/trade-execution-status` - Get execution status
- `GET /api/stocks/charts/equity-curve` - Get equity curve chart
- `GET /api/stocks/charts/transaction-costs` - Get costs chart
- `POST /api/stocks/cleanup` - Clean up resources
- `GET /api/stocks/costs/summary` - Get costs summary
- `GET /api/stocks/costs/analysis` - Get costs analysis
- `GET /api/stocks/costs/breakdown` - Get costs breakdown

### ETF Strategy Endpoints (`/api/etfs`)
All original ETF endpoints are preserved with the same paths:

- `GET /api/etfs/` - List available ETFs
- `GET /api/etfs/default` - Get default ETF selection
- `POST /api/etfs/date-range` - Calculate common date range
- `POST /api/etfs/diagnose` - Diagnose ETF data
- `GET /api/etfs/overview` - Get ETF overview
- `POST /api/etfs/metrics` - Run ETF backtest
- `GET /api/etfs/metrics/table` - Get metrics table
- `GET /api/etfs/transaction-costs/summary` - Transaction costs summary
- `GET /api/etfs/transaction-log` - Get transaction log
- `GET /api/etfs/transaction-costs` - Get transaction costs data
- `GET /api/etfs/skipped-trades` - Get skipped trades
- `GET /api/etfs/trade-execution-status` - Get execution status
- `GET /api/etfs/charts/equity-curve` - Get equity curve chart
- `GET /api/etfs/charts/transaction-costs` - Get costs chart
- `POST /api/etfs/cleanup` - Clean up resources
- `GET /api/etfs/costs/summary` - Get costs summary
- `GET /api/etfs/costs/analysis` - Get costs analysis
- `GET /api/etfs/costs/breakdown` - Get costs breakdown

### ChatAI Endpoints (`/api`)
All ChatAI endpoints for AI chat functionality:

- `POST /api/chat` - Chat with AI assistant
- `POST /api/rate` - Rate AI response
- `GET /api/health` - ChatAI service health check

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages (install from requirements.txt):
  ```bash
  pip install fastapi uvicorn pandas numpy sqlite3
  ```

### Database Setup
Ensure the database files are in the correct location:
- `unified_etf_data.sqlite` - Main database file

### Running the Server

1. **Navigate to the project directory**:
   ```bash
   cd wealthai1-backend-main
   ```

2. **Run the unified server**:
   ```bash
   python unified_server.py
   ```

3. **Server will start on**: `http://127.0.0.1:8000`

## Key Features

### 1. Unified Health Check
The `/health` endpoint provides status for both backtesters:
```json
{
  "api_status": "healthy",
  "stock_backtester_initialized": true,
  "etf_backtester_initialized": true,
  "stock_database_available": true,
  "etf_database_available": true,
  "stock_count": 50,
  "etf_count": 30
}
```

### 2. Separate Error Handling
Each strategy has its own error handling and initialization checks:
- Stock backtester errors are prefixed with "Stock backtester not initialized"
- ETF backtester errors are prefixed with "ETF backtester not initialized"

### 3. Resource Management
- Both backtesters are initialized at startup
- Proper cleanup on server shutdown
- Individual cleanup endpoints for each strategy

### 4. CORS Support
Full CORS support for frontend integration:
- All origins allowed for development
- All HTTP methods supported
- All headers allowed

## Migration from Separate Servers

### Before (Separate Servers)
```bash
# Stock server on port 8000
python stockstrategy/stockserver.py

# ETF server on port 8001  
python etf-strategy/server.py
```

### After (Unified Server)
```bash
# Single server on port 8000
python unified_server.py
```

### API Path Changes
**No changes required!** All API paths remain exactly the same:
- Stock APIs: `/api/stocks/*`
- ETF APIs: `/api/etfs/*`

## Benefits

1. **Single Port**: Only one port needed (8000)
2. **Unified Management**: Single server process to manage
3. **Shared Resources**: Common database connections and configurations
4. **Consistent Health Monitoring**: Single health endpoint for both strategies
5. **Easier Deployment**: One application to deploy and monitor
6. **Reduced Resource Usage**: Single FastAPI instance instead of two

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure you're in the correct directory
   - Check that `stockstrategy/` and `etf-strategy/` folders exist

2. **Database Connection Issues**:
   - Verify `unified_etf_data.sqlite` exists
   - Check file permissions

3. **Port Already in Use**:
   - Stop any existing servers on port 8000
   - Use `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (Linux/Mac)

### Health Check
Always check the `/health` endpoint first to verify both backtesters are initialized correctly.

## Development

### Adding New Endpoints
1. Add to appropriate router (`stock_router` or `etf_router`)
2. Use the correct prefix (`/api/stocks/` or `/api/etfs/`)
3. Include proper error handling for the respective backtester

### Testing
Test both strategies independently:
- Stock endpoints: `http://localhost:8000/api/stocks/`
- ETF endpoints: `http://localhost:8000/api/etfs/`

## API Documentation
Once the server is running, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

These will show all available endpoints for both stock and ETF strategies.
