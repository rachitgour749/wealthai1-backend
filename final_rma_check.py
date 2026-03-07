import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from Services.market_data_service import MarketDataService
from Databases.app_data_db_connection import create_connection
from datetime import datetime, timedelta

def final_rma_check():
    if not create_connection():
        return

    ticker = "GOLDBEES"
    market = "INDIA"
    asset_type = "ETF"
    
    target_date = datetime(2022, 7, 15)
    # Fetch exactly enough data for 52 weeks of weekly bars
    # 52 weeks * 7 days = 364 days
    start_date = target_date - timedelta(days=500) 
    
    df = MarketDataService.fetch_ohlcv(ticker, market, asset_type, start_date, target_date)
    if df.empty:
        return

    df.index = pd.to_datetime(df.index)
    iso = df.index.isocalendar()
    df['year_week'] = iso.year.astype(str) + '_' + iso.week.astype(str).str.zfill(2)
    
    # We need to take the LAST 52 weeks of weekly data
    weekly = df.groupby('year_week').agg(
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last')
    )
    
    # Take exactly 52 weekly rows
    weekly = weekly.iloc[-52:]
    
    weekly['prev_close'] = weekly['close'].shift(1)
    weekly['tr'] = np.maximum(
        weekly['high'] - weekly['low'],
        np.maximum(
            abs(weekly['high'] - weekly['prev_close']),
            abs(weekly['low'] - weekly['prev_close'])
        )
    )
    weekly = weekly.dropna(subset=['tr'])
    
    # RMA
    tr_values = weekly['tr'].values
    rma = np.zeros(len(tr_values))
    period = 10
    rma[period-1] = np.mean(tr_values[:period])
    for i in range(period, len(tr_values)):
        rma[i] = (rma[i-1] * (period - 1) + tr_values[i]) / period
    
    weekly['atr_rma'] = rma
    weekly['hl2'] = (weekly['high'] + weekly['low']) / 2
    weekly['b_lower_rma'] = weekly['hl2'] - (3 * weekly['atr_rma'])
    
    res = weekly.iloc[-1]
    print(f"\nFinal Result (52 Weekly Bars, RMA Initialization):")
    print(f"HL2:      {res['hl2']:.4f}")
    print(f"RMA ATR:  {res['atr_rma']:.4f}")
    print(f"B_Lower:  {res['b_lower_rma']:.4f}")

if __name__ == "__main__":
    final_rma_check()
