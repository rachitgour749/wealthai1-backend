# ETF Symbol Format Testing

## Problem
Both Zerodha and AngelOne are failing with "Failed to get symbol details" for ETFs (ITBEES, PHARMABEES) but working fine for stocks.

## Hypothesis
The symbol format expected by brokers for ETFs might be different from stocks.

## Common ETF Symbol Formats

### Zerodha (Kite)
- **Stocks**: `RELIANCE`, `SBIN`, `TCS`
- **ETFs**: May need exchange suffix like `ITBEES` or `NIFTYBEES`
- **Format**: Usually just the symbol name without any suffix

### AngelOne
- **Stocks**: `RELIANCE-EQ`, `SBIN-EQ` (with -EQ suffix for cash segment)
- **ETFs**: Need to verify - might not need -EQ suffix
- **Format**: Symbol + segment suffix

## Investigation Steps

1. **Check Server Logs**: Look for the debug output showing exact parameters sent to both brokers
2. **Verify Symbol Format**: Check if ETFs need any special formatting
3. **Test with Known Working Symbol**: Try with a stock first, then ETF

## Possible Issues

### Issue 1: Exchange Instrument ID (AngelOne)
- AngelOne requires `exchange_instrument_id` (symbol token)
- The tokens in `symbol_lookup.py` might be incorrect
- **Solution**: Get correct tokens from AngelOne symbol master file

### Issue 2: Symbol Formatting
- ETFs might not need `-EQ` suffix for AngelOne
- Zerodha might need exact symbol as listed on exchange
- **Solution**: Check broker documentation for ETF symbol format

### Issue 3: Exchange Segment
- ETFs might be in a different segment than stocks
- **Solution**: Verify the correct exchange segment for ETFs

## Next Steps

1. Run an order with debug logging enabled
2. Check the logs to see exact parameters being sent
3. Compare with broker API documentation
4. Fix symbol formatting based on findings

## Known Working Examples

### Stocks (Working)
```json
{
  "exchange": "NSE",
  "symbol": "YESBANK",
  "order_side": "BUY",
  "product_type": "DELIVERY",
  "clients": {"CLIENT_ID": "1"}
}
```

### ETFs (Failing)
```json
{
  "exchange": "NSE",
  "symbol": "ITBEES",
  "order_side": "BUY",
  "product_type": "DELIVERY",
  "clients": {"CLIENT_ID": "1"}
}
```

## References
- Zerodha Kite API: https://kite.trade/docs/connect/v3/
- AngelOne API: https://smartapi.angelbroking.com/docs/
