#!/usr/bin/env python
# Script to insert RS score calculation into rs_backtester_core.py

# Read the file
with open('Strategies/RS_Stocks/rs_backtester_core.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the insertion point
search_text = '# Run backtest\r\n            print(f"Processing {len(trading_dates)} trading dates...")'
alt_search_text = '# Run backtest\n            print(f"Processing {len(trading_dates)} trading dates...")'

# Code to insert
code_to_insert = '''
            # Calculate RS scores (Vectorized) - CRITICAL FIX for zero trades issue
            print("Calculating RS scores...")
            rs_scores_df = None  # Initialize
            try:
                stock_data_flat = stock_data.reset_index()
                
                # Prepare index data for vectorization
                if isinstance(index_data.index, pd.DatetimeIndex):
                    index_data_flat = index_data.reset_index()
                    if 'index' in index_data_flat.columns:
                        index_data_flat = index_data_flat.rename(columns={'index': 'date'})
                else:
                    index_data_flat = index_data
                
                rs_scores_df = self.calculate_rs_scores_vectorized(stock_data_flat, index_data_flat)
                print(f"✅ Pre-calculated RS scores for {rs_scores_df.shape[0]} dates and {rs_scores_df.shape[1]} stocks")
            except Exception as e:
                print(f"⚠️  Vectorized RS calculation failed: {e}")
                print("   Falling back to per-date calculation")
                rs_scores_df = None
            
'''

# Try to find and replace
if search_text in content:
    new_content = content.replace(search_text, code_to_insert + search_text)
    print("✅ Found with \\r\\n line endings")
elif alt_search_text in content:
    new_content = content.replace(alt_search_text, code_to_insert + alt_search_text)
    print("✅ Found with \\n line endings")
else:
    print("✗ Could not find insertion point")
    print("Searching for partial match...")
    if '# Run backtest' in content:
        print("Found '# Run backtest' comment")
    if 'Processing {len(trading_dates)} trading dates' in content:
        print("Found processing message")
    exit(1)

# Write back
with open('Strategies/RS_Stocks/rs_backtester_core.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ RS score calculation code inserted successfully")
