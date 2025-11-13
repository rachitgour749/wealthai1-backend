# Migration from SQLite to PostgreSQL for Strategy and Signal Data

## Overview

This document describes the migration of strategy configurations, live signals, and execution tracking from SQLite (`unified_etf_data.sqlite`) to PostgreSQL (Neon database).

## Status: ✅ Partially Complete

### ✅ Completed

1. **Database Models Created** (`Databases/strategy_models.py`)
   - All SQLAlchemy models for strategy and signal tables
   - Models include: ETFSavedStrategy, StockSavedStrategy, RSEtFSavedStrategy, CustomStrategy
   - Live signal models: LiveSignal, LiveRun, LiveStockSignal, LiveStockRun
   - Execution models: ExecutedDetail, Strategy, SaveJson, DeployDetail

2. **Helper Functions Created** (`Databases/strategy_db_helpers.py`)
   - Helper functions for common database operations
   - Functions for saving/retrieving ETF and Stock strategies
   - Functions for saving/retrieving live signals
   - Functions for execution tracking

3. **Connection Module Updated** (`Databases/neon_db_connection.py`)
   - Updated to import and initialize strategy models
   - All tables are now created in PostgreSQL when `init_database()` is called

4. **ETF Strategy API Updated** (`Strategies/etfstrategy/etf_api.py`)
   - `init_saved_etf_strategies_table()` - Now uses PostgreSQL
   - `save_etf_strategy()` - Now saves to PostgreSQL
   - `get_saved_etf_strategies()` - Now reads from PostgreSQL

5. **ETF Signal Generator Updated** (`Strategies/etfstrategy/signal_generator.py`)
   - `create_tables()` - Now initializes PostgreSQL tables
   - `save_signals_to_database()` - Now saves signals to PostgreSQL

### ⚠️ Still Needs Migration

1. **Stock Strategy** (`Strategies/stockstrategy/`)
   - `stock_api.py` - Update all SQLite operations to PostgreSQL
   - `stock_signal_generator.py` - Update signal saving to PostgreSQL

2. **Custom Strategy** (`Strategies/customStrategy/database.py`)
   - Update CustomStrategyDatabase class to use PostgreSQL

3. **RS ETF Strategy** (`server.py`)
   - Update `init_rs_etf_saved_strategies_table()` function
   - Update all RS ETF strategy database operations

4. **Execution Service** (`Services/execution/`)
   - `database.py` - Update ExecutionDatabase class to use PostgreSQL
   - `execution_service.py` - Update all database queries

5. **Webhook Service** (`Services/webhook/`)
   - `webhook_logic.py` - Update database operations
   - Update `savejson` table operations

6. **Server Routes** (`server.py`)
   - Update deployment-related routes
   - Update all SQLite connections to PostgreSQL

7. **Signal Generator Functions**
   - `get_recent_signals()` - Update to read from PostgreSQL
   - `get_symbols_from_deployment()` - Update to read from PostgreSQL
   - Other database read operations

## Database Schema

All tables are now defined in PostgreSQL with the following structure:

### Strategy Tables
- `etf_saved_strategy` - ETF strategy configurations
- `stock_saved_strategy` - Stock strategy configurations
- `rs_etf_saved_strategies` - RS ETF strategy configurations
- `custom_strategies` - Custom user strategies

### Live Signal Tables
- `live_signals` - ETF trading signals
- `live_runs` - ETF signal generation runs
- `live_stock_signals` - Stock trading signals
- `live_stock_runs` - Stock signal generation runs

### Execution & Deployment Tables
- `executed_details` - Execution tracking records
- `strategies` - Webhook strategy configurations
- `savejson` - JSON data storage
- `deploy_details` - Deployment details (legacy/compatibility)

## Migration Steps for Remaining Code

### Pattern to Follow

1. **Replace SQLite connections:**
   ```python
   # OLD (SQLite)
   conn = sqlite3.connect("unified_etf_data.sqlite")
   cursor = conn.cursor()
   cursor.execute("SELECT ...")
   
   # NEW (PostgreSQL)
   from Databases.neon_db_connection import get_session
   from Databases.strategy_models import YourModel
   session = get_session()
   try:
       result = session.query(YourModel).filter(...).all()
   finally:
       session.close()
   ```

2. **Use helper functions when available:**
   ```python
   from Databases.strategy_db_helpers import save_etf_strategy, get_etf_strategies_by_user
   ```

3. **For INSERT/UPDATE operations, use SQLAlchemy:**
   ```python
   from sqlalchemy.dialects.postgresql import insert
   stmt = insert(YourModel).values(**data)
   stmt = stmt.on_conflict_do_update(...)  # For upserts
   session.execute(stmt)
   session.commit()
   ```

## Testing

After migration, test the following:

1. ✅ Database connection and table creation
2. ⚠️ Saving ETF strategies
3. ⚠️ Retrieving ETF strategies
4. ⚠️ Generating and saving live signals
5. ⚠️ Stock strategy operations
6. ⚠️ Execution tracking
7. ⚠️ Webhook operations

## Notes

- The `db_path` parameter is kept in function signatures for backward compatibility but is now ignored
- All database operations should use the Neon PostgreSQL connection
- Market data (ETF/Stock prices) was already migrated to PostgreSQL in a previous update
- The SQLite file (`unified_etf_data.sqlite`) may still exist but should no longer be used for new operations

## Next Steps

1. Complete migration of Stock strategy code
2. Complete migration of Custom strategy code
3. Complete migration of RS ETF strategy code
4. Complete migration of Execution service
5. Complete migration of Webhook service
6. Update all remaining server routes
7. Test all functionality end-to-end
8. Consider data migration script if existing SQLite data needs to be migrated

