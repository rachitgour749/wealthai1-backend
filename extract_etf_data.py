import sqlite3
import pandas as pd
from datetime import datetime

def extract_etf_data():
    """Extract daily high and low prices for NIFTYBEES, BANKBEES, and JUNIORBEES from Nov 11, 2022"""
    
    # Connect to database
    conn = sqlite3.connect('unified_etf_data.sqlite')
    
    # Check available tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Available tables: {tables}")
    
    # Try to find the right table with ETF data
    etf_table = None
    for table in ['etf_unified', 'etf_data', 'stock_data', 'ohlcv']:
        if any(table in str(t) for t in tables):
            etf_table = table
            break
    
    if not etf_table:
        print("No suitable ETF data table found")
        conn.close()
        return
    
    print(f"Using table: {etf_table}")
    
    # Check available symbols
    cursor.execute(f"SELECT DISTINCT symbol FROM {etf_table} WHERE symbol IN ('NIFTYBEES', 'BANKBEES', 'JUNIORBEES')")
    symbols = cursor.fetchall()
    print(f"Available ETF symbols: {symbols}")
    
    # Get data for November 11, 2022
    target_date = '2022-11-11'
    
    # Try different date formats and column names
    queries = [
        f"SELECT * FROM {etf_table} WHERE date = '{target_date}' AND symbol IN ('NIFTYBEES', 'BANKBEES', 'JUNIORBEES')",
        f"SELECT * FROM {etf_table} WHERE date LIKE '%2022-11-11%' AND symbol IN ('NIFTYBEES', 'BANKBEES', 'JUNIORBEES')",
        f"SELECT * FROM {etf_table} WHERE symbol IN ('NIFTYBEES', 'BANKBEES', 'JUNIORBEES') LIMIT 10"
    ]
    
    data_found = False
    for query in queries:
        try:
            cursor.execute(query)
            data = cursor.fetchall()
            if data:
                print(f"Query successful: {query}")
                print(f"Data found: {len(data)} rows")
                data_found = True
                break
        except Exception as e:
            print(f"Query failed: {query}")
            print(f"Error: {e}")
    
    if not data_found:
        print("No data found with any query")
        conn.close()
        return
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({etf_table})")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"Table columns: {columns}")
    
    # Get all data for the ETFs around the target date
    cursor.execute(f"""
        SELECT * FROM {etf_table} 
        WHERE symbol IN ('NIFTYBEES', 'BANKBEES', 'JUNIORBEES') 
        AND date >= '2022-11-01' AND date <= '2022-11-30'
        ORDER BY symbol, date
    """)
    
    all_data = cursor.fetchall()
    print(f"Total data points found: {len(all_data)}")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_data, columns=columns)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Find the target date data
    target_data = df[df['date'] == target_date]
    print(f"Data for {target_date}: {len(target_data)} rows")
    
    # Create the output text file
    with open('etf_daily_prices_nov11_2022.txt', 'w') as f:
        f.write("ETF Daily High and Low Prices - November 11, 2022\n")
        f.write("=" * 60 + "\n\n")
        
        if not target_data.empty:
            f.write(f"Data found for {target_date}:\n")
            f.write("-" * 40 + "\n")
            
            for _, row in target_data.iterrows():
                symbol = row['symbol']
                f.write(f"\n{symbol}:\n")
                
                # Try to find high/low columns
                high_col = None
                low_col = None
                close_col = None
                
                for col in columns:
                    if 'high' in col.lower():
                        high_col = col
                    elif 'low' in col.lower():
                        low_col = col
                    elif 'close' in col.lower():
                        close_col = col
                
                if high_col and low_col:
                    f.write(f"  High: Rs.{row[high_col]:.2f}\n")
                    f.write(f"  Low: Rs.{row[low_col]:.2f}\n")
                elif close_col:
                    f.write(f"  Close Price: Rs.{row[close_col]:.2f}\n")
                else:
                    # Show all available data
                    for col in columns:
                        if col not in ['symbol', 'date']:
                            f.write(f"  {col}: {row[col]}\n")
        else:
            f.write(f"No data found for {target_date}\n")
            f.write("Available dates in November 2022:\n")
            
            # Show available dates
            available_dates = df['date'].unique()
            for date in sorted(available_dates):
                f.write(f"  {date}\n")
        
        # Show sample data structure
        f.write(f"\n\nSample Data Structure:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Columns: {', '.join(columns)}\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
        
        # Show first few rows for each ETF
        for symbol in ['NIFTYBEES', 'BANKBEES', 'JUNIORBEES']:
            symbol_data = df[df['symbol'] == symbol]
            if not symbol_data.empty:
                f.write(f"\n{symbol} - First 5 rows:\n")
                for _, row in symbol_data.head().iterrows():
                    f.write(f"  {row['date']}: {row}\n")
    
    print(f"Data extracted and saved to 'etf_daily_prices_nov11_2022.txt'")
    conn.close()

if __name__ == "__main__":
    extract_etf_data()
