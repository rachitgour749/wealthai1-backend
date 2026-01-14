import sqlite3
import pandas as pd

def inspect_db():
    try:
        conn = sqlite3.connect("unified_etf_data.sqlite")
        
        print("--- ETF Data Summary ---")
        df_etf = pd.read_sql("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM etf_data GROUP BY symbol", conn)
        print(df_etf)
        
        print("\n--- Index Data Summary ---")
        try:
            df_index = pd.read_sql("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM index_data GROUP BY symbol", conn)
            print(df_index)
        except Exception as e:
            print(f"Index data error: {e}")
            
        print("\n--- Sample ETF Data (NIFTYBEES) ---")
        df_sample = pd.read_sql("SELECT * FROM etf_data WHERE symbol = 'NIFTYBEES' LIMIT 5", conn)
        print(df_sample)
        
        conn.close()
    except Exception as e:
        print(f"Error inspecting DB: {e}")

if __name__ == "__main__":
    inspect_db()
