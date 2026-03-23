import pandas as pd
from typing import List, Iterable, Optional
from stock_indicators import Quote, indicators
from datetime import datetime

class IndicatorHelper:
    """
    Utility class to bridge Pandas DataFrames and the stock-indicators library.
    """
    
    @staticmethod
    def df_to_quotes(df: pd.DataFrame) -> List[Quote]:
        """
        Convert a Pandas DataFrame to a list of Quote objects.
        Requires 'open', 'high', 'low', 'close' columns. 
        Uses the DataFrame index as the date.
        """
        if df.empty:
            return []
            
        quotes = []
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        for index, row in df.iterrows():
            quotes.append(Quote(
                date=index,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0))
            ))
        return quotes

    @staticmethod
    def supertrend_to_df(results: Iterable, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Convert SuperTrend results to a DataFrame aligned with the input index.
        """
        st_data = []
        for r in results:
            direction = None
            if r.lower_band is not None:
                direction = 1
            elif r.upper_band is not None:
                direction = -1
                
            st_data.append({
                'supertrend': float(r.super_trend) if r.super_trend is not None else None,
                'st_direction': direction
            })
            
        return pd.DataFrame(st_data, index=index)

        return pd.DataFrame(st_data, index=index)
