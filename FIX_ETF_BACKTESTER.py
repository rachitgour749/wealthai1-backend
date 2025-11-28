"""
MANUAL FIX INSTRUCTIONS
=======================

The ETF backtester __init__ method is missing critical initialization code.

PROBLEM:
--------
The __init__ method at line 38-52 is incomplete. It's missing:
1. init_market_data_database()
2. self.nifty50_df = pd.DataFrame()
3. self.etf_metadata = self.load_metadata()
4. Calculator initialization

SOLUTION:
---------
Manually edit the file:
C:\Users\Lenovo\Desktop\WEALTHAI_PROD\wealthai1-backend\Strategies\Rotation_ETF\services\backtester.py

Find the __init__ method (around line 38) and replace it with:

```python
    def __init__(self, db_path: str = None):
        \"\"\"
        Initialize ETF Rotation Backtester

        Args:
            db_path: Deprecated - kept for compatibility. Now uses PostgreSQL MarketData database.
        \"\"\"
        # Call base class constructor (initializes common attributes)
        super().__init__()
        
        # Initialize PostgreSQL connection
        if not create_market_data_connection():
            raise RuntimeError("Failed to connect to MarketData database")

        # Initialize database tables
        init_market_data_database()

        # ETF-specific attributes (not in base class)
        self.nifty50_df = pd.DataFrame()
        self.etf_metadata = self.load_metadata()  # Calls abstract method implementation
        
        # Initialize Calculator classes for cost and tax calculations
        self.cost_calculator = IndianMarketCostCalculator()
        self.tax_calculator = IndianCapitalGainsTaxCalculator()
```

STEPS:
------
1. Open the file in your code editor
2. Find the __init__ method (line 38)
3. Replace the entire method with the code above
4. Save the file
5. Restart the server: python server.py

VERIFICATION:
-------------
After fixing, test with:
```python
from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester
bt = ETFRotationBacktester()
print(hasattr(bt, 'etf_metadata'))  # Should print: True
```

WHY THIS HAPPENED:
------------------
The refactoring script had a bug that removed these lines instead of adding to them.
The backup was also broken because it was created AFTER the lines were already missing.

ALTERNATIVE:
------------
If you have Git, you can restore from version control:
```bash
git checkout Strategies/Rotation_ETF/services/backtester.py
```
"""

print(__doc__)
