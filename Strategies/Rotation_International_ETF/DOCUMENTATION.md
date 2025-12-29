# International ETF Rotation Strategy Documentation

## Overview

The **International ETF Rotation Strategy** is a momentum-based investment strategy designed for international Exchange-Traded Funds (ETFs). It is based on the existing `Rotation_ETF` strategy but adapted specifically for international markets with the following key differences:

- **Data Source**: Uses `international_etf_data` table instead of `etf_data`
- **Transaction Costs**: **Zero transaction costs** for all buy/sell operations
- **Holiday Calendar**: Uses **NYSE (New York Stock Exchange)** trading calendar
- **Timezone**: Operates in **US Eastern Time (ET)** timezone
- **Market**: Targets international ETFs (SPY, QQQ, VTI, EEM, EFA, etc.)

## Strategy Logic

### Core Momentum Strategy

The strategy employs a **52-week high/low momentum approach**:

1. **Signal Generation**: Calculates each ETF's distance from its 52-week high and 52-week low
2. **Ranking**: Ranks ETFs based on momentum indicators
3. **Selection**: Selects top-performing ETFs based on momentum
4. **Rebalancing**: Weekly rebalancing based on momentum changes

### Key Parameters

- **52-Week Period**: Uses exactly 252 trading days for momentum calculations
- **Signal Day**: Last trading day of the week (typically Friday)
- **Execution Day**: First trading day of the next week (typically Monday)
- **Rebalance Frequency**: Weekly

## International ETF Universe

The strategy supports the following international ETFs:

### US Market ETFs
- **SPY**: S&P 500 ETF - US Large Cap
- **QQQ**: NASDAQ 100 ETF - US Technology
- **VTI**: Total Stock Market ETF - US Broad Market

### International Market ETFs
- **EFA**: EAFE ETF - Developed Markets ex-US
- **EEM**: Emerging Markets ETF
- **VWO**: Emerging Markets ETF

### Sector ETFs
- **XLE**: Energy Sector ETF
- **XLF**: Financial Sector ETF
- **XLK**: Technology Sector ETF
- **XLV**: Healthcare Sector ETF

### Country-Specific ETFs
- **EWJ**: Japan ETF
- **EWG**: Germany ETF
- **EWQ**: France ETF

### Bond ETFs
- **AGG**: Aggregate Bond ETF - US Bonds

## NYSE Holiday Handling

### Trading Calendar

The strategy uses the **NYSE trading calendar** to determine trading days. This is different from the Indian ETF strategy which uses NSE holidays.

### Holiday Logic

- **Signal Generation**: Skipped on NYSE holidays
- **Execution**: Deferred to next available trading day if Monday is a holiday
- **Fallback**: If Friday is a holiday, signal generated on Thursday

### Major NYSE Holidays

The strategy automatically accounts for:
- New Year's Day
- Martin Luther King Jr. Day
- Presidents' Day
- Good Friday
- Memorial Day
- Independence Day
- Labor Day
- Thanksgiving Day
- Christmas Day

## US Timezone Implementation

### Timezone Handling

All date/time operations are performed in **US Eastern Time (ET)**:
- **Standard Time (EST)**: UTC-5 (November to March)
- **Daylight Time (EDT)**: UTC-4 (March to November)

### Market Hours

- **Signal Generation**: Based on market close (4:00 PM ET)
- **Execution**: Based on market open (9:30 AM ET)

### Implementation

The strategy uses Python's `pytz` library for timezone conversions:

```python
import pytz

eastern = pytz.timezone('US/Eastern')
# Convert datetime to US Eastern Time
eastern_time = eastern.localize(naive_datetime)
```

## Transaction Costs

### Zero Cost Model

**IMPORTANT**: The international ETF strategy has **ZERO transaction costs** for all operations.

This is a significant difference from the Indian ETF strategy which includes:
- Brokerage fees
- Securities Transaction Tax (STT)
- Exchange charges
- GST
- SEBI charges
- Stamp duty

### Cost Structure

```python
{
    'brokerage': 0.0,
    'stt': 0.0,
    'transaction_charges': 0.0,
    'gst': 0.0,
    'sebi_charges': 0.0,
    'stamp_duty': 0.0,
    'total_cost': 0.0
}
```

### Rationale

Zero transaction costs are applied because:
1. International ETF trading platforms often have different fee structures
2. Simplifies backtesting for international markets
3. Allows focus on pure strategy performance
4. Institutional investors may have negotiated zero-fee arrangements

## Database Schema

### Table: international_etf_data

```sql
CREATE TABLE international_etf_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume INTEGER,
    adjusted_close FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, date)
);
```

### Data Requirements

- **Minimum History**: 20 years of historical data preferred
- **Data Frequency**: Daily OHLCV data
- **Price Precision**: Rounded to 2 decimal places
- **Volume**: Integer values

## Differences from Indian ETF Strategy

| Feature | Indian ETF Strategy | International ETF Strategy |
|---------|-------------------|---------------------------|
| **Data Table** | `etf_data` | `international_etf_data` |
| **Transaction Costs** | Full Indian market costs | **Zero costs** |
| **Holiday Calendar** | NSE (India) | **NYSE (US)** |
| **Timezone** | IST (UTC+5:30) | **US Eastern (UTC-5/-4)** |
| **Currency** | INR (₹) | **USD ($)** |
| **Market Hours** | 9:15 AM - 3:30 PM IST | **9:30 AM - 4:00 PM ET** |
| **ETF Universe** | Indian ETFs | **International ETFs** |

## API Endpoints

The strategy provides comprehensive REST API endpoints for:

### Metadata & Configuration
- `GET /api/international-etf/metadata` - Get available international ETFs
- `GET /api/international-etf/date-range` - Calculate common date range
- `GET /api/international-etf/etf-info` - Get ETF descriptions

### Backtesting
- `POST /api/international-etf/backtest` - Run backtest
- `GET /api/international-etf/backtest-results/{strategy_id}` - Get results
- `GET /api/international-etf/portfolio-log/{strategy_id}` - Get portfolio log
- `GET /api/international-etf/transaction-log/{strategy_id}` - Get transaction log

### Strategy Management
- `POST /api/international-etf/save-strategy` - Save strategy configuration
- `GET /api/international-etf/strategies` - List saved strategies
- `GET /api/international-etf/strategy/{strategy_id}` - Get strategy details
- `DELETE /api/international-etf/strategy/{strategy_id}` - Delete strategy

### Deployment & Signals
- `POST /api/international-etf/deploy` - Deploy strategy
- `GET /api/international-etf/deployed-strategies` - List deployed strategies
- `GET /api/international-etf/signals/{deployment_id}` - Get trading signals
- `POST /api/international-etf/execute-signal` - Execute trading signal

### Analytics
- `GET /api/international-etf/performance/{strategy_id}` - Get performance metrics
- `GET /api/international-etf/charts/{strategy_id}` - Get chart data
- `GET /api/international-etf/drawdown/{strategy_id}` - Get drawdown analysis

## Usage Example

### Python Example

```python
from Strategies.Rotation_International_ETF.services.backtester import InternationalETFRotationBacktester

# Initialize backtester
bt = InternationalETFRotationBacktester()

# Get available ETFs
print(f"Available ETFs: {list(bt.etf_metadata.keys())}")

# Calculate common date range
etfs = ['SPY', 'QQQ', 'VTI']
start_date, end_date, years = bt.calculate_common_date_range(etfs)
print(f"Date range: {start_date} to {end_date} ({years:.1f} years)")

# Run backtest
results = bt.run_backtest(
    selected_etfs=etfs,
    start_date=start_date,
    end_date=end_date,
    initial_capital=100000,
    num_etfs_to_hold=2,
    rebalance_frequency='weekly'
)

# Verify zero transaction costs
costs = bt.calculate_transaction_costs('buy', 100000, 0.05)
assert costs['total_cost'] == 0.0
```

### API Example

```bash
# Run backtest via API
curl -X POST "http://localhost:8000/api/international-etf/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "selected_etfs": ["SPY", "QQQ", "VTI"],
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "num_etfs_to_hold": 2,
    "rebalance_frequency": "weekly"
  }'
```

## Performance Considerations

### Caching

The backtester implements intelligent caching:
- **Data Cache**: Caches loaded OHLCV data to avoid repeated database queries
- **Cache Key**: Based on tickers, start date, and end date
- **Buffer Data**: Includes 400-day historical buffer for momentum calculations

### Optimization

- **Vectorized Operations**: Uses pandas for efficient data processing
- **Batch Queries**: Loads all ETF data in single database query
- **Connection Pooling**: Uses SQLAlchemy connection pooling

## Testing

### Unit Tests

```python
# Test backtester initialization
bt = InternationalETFRotationBacktester()
assert len(bt.etf_metadata) > 0

# Test zero transaction costs
costs = bt.calculate_transaction_costs('buy', 100000, 0.05)
assert costs['total_cost'] == 0.0

# Test ETF descriptions
desc = bt.generate_asset_description('SPY')
assert 'S&P 500' in desc
```

### Integration Tests

- Test all API endpoints
- Verify database connectivity
- Test signal generation with NYSE holidays
- Validate timezone handling

## Logging

The strategy uses centralized logging via `StrategyLogger`:

```python
from Strategies.utilities.logging_config import StrategyLogger

logger = StrategyLogger('Rotation_International_ETF')
logger.info("Strategy initialized")
logger.progress("Running backtest...")
logger.error("Error occurred")
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **PROGRESS**: Progress updates during long operations
- **ERROR**: Error messages

## Future Enhancements

Potential future improvements:
1. **Advanced Holiday Handling**: Integration with `pandas_market_calendars` for more sophisticated NYSE holiday detection
2. **Timezone Automation**: Automatic timezone conversion for global users
3. **Multi-Currency Support**: Support for ETFs in different currencies
4. **Risk Management**: Position sizing based on volatility
5. **Tax Optimization**: Optional tax-loss harvesting for US investors

## Support

For issues or questions:
- Review the implementation plan
- Check the backtester source code
- Examine API route documentation
- Run test scripts for verification

## Version History

- **v1.0** (2025-12-26): Initial implementation
  - Zero transaction costs
  - NYSE holiday support
  - US Eastern Time timezone
  - International ETF universe
  - Full API compatibility with Rotation_ETF
