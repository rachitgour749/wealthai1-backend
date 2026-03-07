import pandas as pd
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from Services.market_data_service import MarketDataService
from Strategies.SuperTrend.strategy import SuperTrendStrategy
from Databases.app_data_db_connection import create_connection
from datetime import datetime

def explain_goldbees_st():
    # 0. Initialize DB
    if not create_connection():
        print("Failed to connect to database!")
        return

    # 1. Fetch data
    ticker = "GOLDBEES"
    market = "INDIA"
    asset_type = "ETF"
    
    # Needs enough history for ATR(10) Weekly (at least 15-20 weeks)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 7, 20)
    
    print(f"Fetching data for {ticker}...")
    df = MarketDataService.fetch_ohlcv(ticker, market, asset_type, start_date, end_date)
    
    if df.empty:
        print("No data found!")
        return
        
    print(f"Total daily records fetched: {len(df)}")
    
    # 2. Use Strategy logic to calculate
    strategy = SuperTrendStrategy(market=market, asset_type=asset_type)
    strategy.atr_period = 10
    strategy.atr_multiplier = 3.0
    
    # We call internal methods to see the intermediates
    df.index = pd.to_datetime(df.index)
    iso = df.index.isocalendar()
    df['year_week'] = iso.year.astype(str) + '_' + iso.week.astype(str).str.zfill(2)
    week_last_date = df.groupby('year_week').apply(lambda x: x.index[-1])
    weekly = df.groupby('year_week').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    )
    
    # Calculate SuperTrend on weekly
    st_res = strategy._calculate_supertrend(weekly, strategy.atr_period, strategy.atr_multiplier)
    weekly['st'] = st_res['supertrend']
    weekly['dir'] = st_res['st_direction']
    
    # Re-calculate ATR and bands for debugging
    high = weekly['high']
    low = weekly['low']
    close = weekly['close']
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/10, adjust=False).mean()
    hl2 = (high + low) / 2
    b_upper = hl2 + (3 * atr)
    b_lower = hl2 - (3 * atr)
    
    weekly['atr'] = atr
    weekly['hl2'] = hl2
    weekly['b_upper'] = b_upper
    weekly['b_lower'] = b_lower
    
    print("\nDetailed Weekly Breakdown for GOLDBEES:")
    print(f"{'Date':<12} | {'Close':<6} | {'HL2':<6} | {'ATR':<5} | {'B_Lower':<8} | {'Final_ST':<8}")
    print("-" * 65)
    
    last_weeks = weekly.tail(20)
    for date_str, row in last_weeks.iterrows():
        # Find the actual date for this year_week
        actual_date = week_last_date[date_str]
        print(f"{actual_date.strftime('%Y-%m-%d'):<12} | {row['close']:<6.2f} | {row['hl2']:<6.2f} | {row['atr']:<5.2f} | {row['b_lower']:<8.2f} | {row['st']:<8.2f}")

if __name__ == "__main__":
    explain_goldbees_st()
