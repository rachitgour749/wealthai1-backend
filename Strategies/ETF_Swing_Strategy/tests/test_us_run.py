import sys
import os
import json

# Add root project path for imports
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_path)

from Strategies.ETF_Swing_Strategy.services.backtester import ETFSwingBacktester
from Databases.app_data_db_connection import create_connection


def test_us_etf_swing_strategy():
    print("=" * 60)
    print("US ETF Swing Strategy - Backtest Test")
    print("=" * 60)

    # Initialize Database Connection
    if not create_connection():
        print("X Failed to connect to database. Check DATABASE_STRING in .env")
        return

    # Initialize backtester for US market
    backtester = ETFSwingBacktester(market="US", asset_type="ETF")

    # Sample US ETF tickers
    tickers = ["SPY", "QQQ", "GLD", "TLT", "IWM"]
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000.0  # USD

    print(f"\nMarket      : US")
    print(f"Asset Type  : ETF")
    print(f"Tickers     : {tickers}")
    print(f"Date Range  : {start_date} -> {end_date}")
    print(f"Capital     : ${initial_capital:,.0f}")
    print()

    try:
        results = backtester.run_backtest(tickers, start_date, end_date, initial_capital, risk_free_rate=4.5)

        if "error" in results:
            print(f"X Backtest failed: {results['error']}")
            return

        metrics = results.get("metrics", {})
        bench_metrics = metrics.get("benchmark_metrics", {})

        print("OK Backtest completed successfully!\n")
        print(f"{'Metric':<30} {'Strategy':>15} {'S&P 500 B&H':>15}")
        print("-" * 62)

        rows = [
            ("Total Return (%)",        "total_return_pct"),
            ("CAGR (%)",                "cagr"),
            ("Sharpe Ratio",            "sharpe_ratio"),
            ("Max Drawdown (%)",        "max_drawdown_pct"),
            ("Calmar Ratio",            "calmar_ratio"),
        ]
        for label, key in rows:
            s_val = metrics.get(key, "N/A")
            b_val = bench_metrics.get(key, "N/A")
            s_str = f"{s_val:.2f}" if isinstance(s_val, (int, float)) else str(s_val)
            b_str = f"{b_val:.2f}" if isinstance(b_val, (int, float)) else str(b_val)
            print(f"{label:<30} {s_str:>15} {b_str:>15}")

        total_trades = metrics.get("total_trades", 0)
        print(f"\nTotal Trades: {total_trades}")

        # Write results to file
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "us_results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(metrics, indent=2, default=str))
        print(f"\nFull metrics written to: {output_file}")

        # Show sample transactions
        tx_log = results.get("transaction_log", [])
        if tx_log:
            print(f"\nSample Trades (First 5):")
            for trade in tx_log[:5]:
                date_str = str(trade.get('date', 'N/A'))[:10]
                print(f"  {date_str} | {trade['action']} {trade['symbol']} @ ${trade['price']:.2f} (Qty: {trade['qty']})")
        else:
            print("\nNo trades executed during the period.")

    except Exception as e:
        print(f"X Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_us_etf_swing_strategy()
