# Database Configuration

This directory contains database connection modules for the WealthAI backend.

## Database Architecture

The backend uses **two types of databases**:

### 1. Neon PostgreSQL Database
- **Used for**: All general backend storage (subscriptions, payments, users, chat conversations, etc.)
- **Connection file**: `neon_db_connection.py`
- **Database URL**: Configured in `neon_db_connection.py`

### 2. SQLite Databases
- **Used for**: ETF, RS, Stock, backtest, and live signal generation
- **Locations**:
  - `unified_etf_data.sqlite` (root directory)
  - `Strategies/rsStrategy/nifty500_data_with_metadata.sqlite`
  - Other strategy-specific SQLite files

## Neon PostgreSQL Connection

### Testing Connection

To test the Neon database connection, run:

```bash
python Databases/neon_db_connection.py
```

This will:
- Connect to the Neon PostgreSQL database
- Test the connection
- Display connection status (success or error)

### Using the Connection

```python
from Databases.neon_db_connection import create_connection, get_session, Base

# Initialize connection (should be called once at application startup)
create_connection()

# Get a database session
session = get_session()
try:
    # Your database operations here
    # Example: result = session.query(YourModel).all()
    pass
finally:
    session.close()

# Initialize database tables (after defining models)
from Databases.neon_db_connection import init_database
init_database()
```

### Connection Status

- ✅ **Success**: Displays "Database connected successfully!"
- ❌ **Error**: Displays the error message

## Requirements

The Neon PostgreSQL connection requires:
- `sqlalchemy>=2.0.0`
- `psycopg2-binary>=2.9.0`

Both are included in `requirements.txt`.

## Database Migration Strategy

### Services Using Neon PostgreSQL (to be migrated):
- `Services/subscription/` - User subscriptions
- `Services/Payment/` - Payment transactions
- `Services/execution/` - Execution tracking
- `chatAI/` - Chat conversations and ratings
- `Services/webhook/` - Webhook configurations

### Services Using SQLite (to remain):
- `Strategies/etfstrategy/` - ETF backtesting
- `Strategies/rsStrategy/` - RS strategy and backtesting
- `Strategies/stockstrategy/` - Stock backtesting
- `Schedulers/` - Live signal generation (ETF, RS, Stock)

## Notes

- The Neon database connection uses connection pooling for better performance
- SSL is required for Neon database connections
- Connection timeouts are set to 10 seconds
- Connection pooling: 5 base connections, 10 max overflow

