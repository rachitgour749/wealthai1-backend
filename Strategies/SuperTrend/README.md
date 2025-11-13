# Backend API - Supertrend + EMA + RS Strategy

FastAPI backend for trading strategy platform.

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Initialize database**:
```bash
python -c "from api.database import init_database; init_database()"
```

3. **Run server**:
```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── api/
│   ├── routes.py          # API endpoints
│   ├── models.py          # Pydantic models
│   └── database.py        # Database connection
├── backtester_core/
│   ├── indicators.py      # Technical indicators
│   ├── rs_calculator.py   # Relative strength
│   ├── signal_logic.py    # Entry/exit signals
│   ├── portfolio_manager.py # Portfolio management
│   ├── backtest_engine.py # Backtest engine
│   └── utils.py           # Utility functions
└── services/
    ├── scheduler.py       # Scheduled jobs
    └── ingestion.py       # Data ingestion
```

## Key Modules

### Indicators (`backtester_core/indicators.py`)
- EMA calculation
- Supertrend with fixed 10% stop (no ATR)

### RS Calculator (`backtester_core/rs_calculator.py`)
- Relative strength vs NIFTY50
- Multi-window RS scoring (5D, 21D, 63D)

### Signal Logic (`backtester_core/signal_logic.py`)
- Eligibility filters
- Entry/exit signal generation
- Hygiene filters (price, liquidity)

### Portfolio Manager (`backtester_core/portfolio_manager.py`)
- Position management
- Equal-weight allocation
- Top-N selection by RS
- Rebalancing logic

### Backtest Engine (`backtester_core/backtest_engine.py`)
- Full backtest simulation
- Performance metrics calculation
- Trade execution simulation

## Database

Uses SQLite (`strategy_data.sqlite`) with tables:
- `stock_data` - Stock price data
- `index_data` - Benchmark index data
- `strategy_config` - Strategy parameters
- `candidates` - Current candidates
- `current_positions` - Active positions
- `backtest_results` - Backtest trades

## Environment Variables

Create `.env` file:
```
DATABASE_PATH=strategy_data.sqlite
LOG_LEVEL=INFO
```

## Testing

Run individual modules:
```bash
python -m backtester_core.indicators
python -m backtester_core.rs_calculator
python -m backtester_core.backtest_engine
```

## Production Deployment

1. Use production ASGI server (Gunicorn + Uvicorn workers)
2. Set proper CORS origins
3. Enable logging
4. Set up database backups
5. Configure scheduler for automated EOD runs

