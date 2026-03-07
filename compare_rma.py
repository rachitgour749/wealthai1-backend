import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from Services.market_data_service import MarketDataService
from Databases.app_data_db_connection import create_connection
from datetime import datetime, timedelta

def compare_rma_ewm_raw():
    if not create_connection():
        return

    ticker = "GOLDBEES"
    market = "INDIA"
    asset_type = "ETF"
    
    target_date = datetime(2022, 7, 15)
    # Exactly 52 weeks of history before the target date
    start_date = target_date - timedelta(weeks=52)
    
    df = MarketDataService.fetch_ohlcv(ticker, market, asset_type, start_date, target_date)
    if df.empty:
        print("No data found!")
        return

    df.index = pd.to_datetime(df.index)
    iso = df.index.isocalendar()
    df['year_week'] = iso.year.astype(str) + '_' + iso.week.astype(str).str.zfill(2)
    weekly = df.groupby('year_week').agg(
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last')
    )
    
    # Calculate True Range
    weekly['prev_close'] = weekly['close'].shift(1)
    weekly['tr'] = np.maximum(
        weekly['high'] - weekly['low'],
        np.maximum(
            abs(weekly['high'] - weekly['prev_close']),
            abs(weekly['low'] - weekly['prev_close'])
        )
    )
    weekly = weekly.dropna(subset=['tr'])
    
    period = 10
    multiplier = 3.0
    
    # Method 1: Current EWM (Initialization with first value)
    weekly['atr_ewm'] = weekly['tr'].ewm(alpha=1/period, adjust=False).mean()
    
    # Method 2: Raw RMA (Wilder's - Initialization with SMA of first 10)
    tr_values = weekly['tr'].values
    rma = np.zeros(len(tr_values))
    if len(tr_values) >= period:
        rma[period-1] = np.mean(tr_values[:period])
        for i in range(period, len(tr_values)):
            rma[i] = (rma[i-1] * (period - 1) + tr_values[i]) / period
    
    rma_series = pd.Series(rma, index=weekly.index)
    rma_series[:period-1] = np.nan
    weekly['atr_rma'] = rma_series
    
    # HL2
    weekly['hl2'] = (weekly['high'] + weekly['low']) / 2
    
    # Basic Bands for both
    # Raw value usually refers to the band itself without the locking logic
    weekly['b_lower_ewm'] = weekly['hl2'] - (multiplier * weekly['atr_ewm'])
    weekly['b_lower_rma'] = weekly['hl2'] - (multiplier * weekly['atr_rma'])
    
    print("\nComparison at 2022-07-15 (GOLDBEES):")
    res = weekly.iloc[-1]
    print(f"Close:       {res['close']:.2f}")
    print(f"HL2:         {res['hl2']:.2f}")
    print("-" * 30)
    print(f"EWM ATR:     {res['atr_ewm']:.4f}")
    print(f"EWM " + "Raw" + " B_Lower: {res['b_lower_ewm']:.2f}")
    print("-" * 30)
    print(f"RMA ATR:     {res['atr_rma']:.4f}")
    print(f"RMA " + "Raw" + " B_Lower: {res['b_lower_rma']:.2f}")
    
    # Full ST with locking
    def calc_st(df, atr_col):
        st = [0.0] * len(df)
        direction = [1] * len(df)
        raw_lower = list(df['hl2'] - multiplier * df[atr_col])
        raw_upper = list(df['hl2'] + multiplier * df[atr_col])
        final_lower = list(raw_lower)
        close = list(df['close'])
        
        for i in range(1, len(df)):
            if pd.isna(final_lower[i-1]): continue
            
            # Locked Lowerband
            if final_lower[i] < final_lower[i-1] and close[i-1] > final_lower[i-1]:
                final_lower[i] = final_lower[i-1]
            
            # Direction Logic
            if close[i] > raw_upper[i]: direction[i] = 1
            elif close[i] < final_lower[i]: direction[i] = -1
            else: direction[i] = direction[i-1]
                
            if direction[i] == 1: st[i] = final_lower[i]
            else: st[i] = raw_upper[i]
        return st
    
    weekly['st_ewm'] = calc_st(weekly, 'atr_ewm')
    weekly['st_rma'] = calc_st(weekly, 'atr_rma')
    
    print("-" * 30)
    print(f"Final ST (EWM): {weekly['st_ewm'].iloc[-1]:.2f}")
    print(f"Final ST (RMA): {weekly['st_rma'].iloc[-1]:.2f}")

if __name__ == "__main__":
    compare_rma_ewm_raw()
