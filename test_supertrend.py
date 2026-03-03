import asyncio
from datetime import datetime
from Strategies.SuperTrend.services.backtester import SuperTrendBacktester

def test_supertrend():
    print("Initializing SuperTrend Backtester...")
    backtester = SuperTrendBacktester(market="INDIA", asset_type="STOCK")
    
    tickers = ["RELIANCE.NS"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000.0
    
    print(f"Running backtest for {tickers} from {start_date} to {end_date}...")
    
    # Custom config based on new rules
    config_params = {
        "atr_multiplier": 3.0,
        "atr_period": 10,
        "stop_loss_pct": 5.0,
        "number_of_slots": 2,
        "brokerage_percent": 0.05
    }
    
    result = backtester.run_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_free_rate=8.0,
        config_params=config_params
    )
    
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\n=== Backtest Completed Successfully ===")
        metrics = result.get("metrics", {})
        print("Metrics:", metrics)
        
        tx_log = result.get("transaction_log", [])
        if tx_log:
            print(f"\nSample Transaction Logs (First 3 of {len(tx_log)}):")
            for tx in tx_log[:3]:
                print(f"[{tx['date'].strftime('%Y-%m-%d')}] {tx['action']} {tx['symbol']} Qty: {tx['qty']} | Price: {tx['price']:.2f} | Amount: {tx['amount']:.2f}")
                if tx['action'] == 'SELL':
                    print(f"  Reason: {tx['reason']} | Net Proceeds: {tx['net_proceeds']}")
        else:
            print("\nNo transactions occurred in this period.")

if __name__ == "__main__":
    test_supertrend()
