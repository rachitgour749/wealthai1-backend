import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from Services.market_data_service import MarketDataService
from Databases.app_data_db_connection import create_connection
from datetime import datetime, timedelta

def check_sma_atr():
    if not create_connection():
        return

    ticker = "GOLDBEES"
    market = "INDIA"
    asset_type = "ETF"
    
    target_date = datetime(2022, 7, 15)
    start_date = target_date - timedelta(weeks=100)
    
    df = MarketDataService.fetch_ohlcv(ticker, market, asset_type, start_date, target_date)
    if df.empty:
        return

    df.index = pd.to_datetime(df.index)
    iso = df.index.isocalendar()
    df['year_week'] = iso.year.astype(str) + '_' + iso.week.astype(str).str.zfill(2)
    weekly = df.groupby('year_week').agg(
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last')
    )
    
    weekly['prev_close'] = weekly['close'].shift(1)
    weekly['tr'] = np.maximum(
        weekly['high'] - weekly['low'],
        np.maximum(
            abs(weekly['high'] - weekly['prev_close']),
            abs(weekly['low'] - weekly['prev_close'])
        )
    )
    weekly = weekly.dropna(subset=['tr'])
    
    # Simple Moving Average ATR
    weekly['atr_sma'] = weekly['tr'].rolling(window=10).mean()
    weekly['hl2'] = (weekly['high'] + weekly['low']) / 2
    weekly['b_lower_sma'] = weekly['hl2'] - (3 * weekly['atr_sma'])
    
    print("\nSMA-ATR Comparison for 2022-07-15:")
    res = weekly.iloc[-1]
    print(f"HL2:      {res['hl2']:.2f}")
    print(f"SMA ATR:  {res['atr_sma']:.4f}")
    print(f"B_Lower:  {res['b_lower_sma']:.2f}")

if __name__ == "__main__":
    check_sma_atr()
