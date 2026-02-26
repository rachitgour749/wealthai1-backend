import sys
import os
from datetime import datetime

# Add root project path for imports
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_path)

from Strategies.ETF_Swing_Strategy.strategy import ETFSwingStrategy

def test_logic_fix():
    print("Testing Slot Logic and Re-entry Fix...")
    
    # Initialize strategy
    strategy = ETFSwingStrategy(market="INDIA", asset_type="ETF")
    strategy.update_config({"number_of_slots": 4})
    strategy.initialize_portfolio(1000000.0)
    
    # Simulate March 24th scenario
    # Slot 0: OCCUPIED (some stock)
    # Slot 1: OCCUPIED (some stock)
    # Slot 2: FREE (previously AUTOBEES sold)
    # Slot 3: PENDING_FREE (PSUBNKBEES just sold)
    
    strategy.slots[0]["status"] = "OCCUPIED"
    strategy.slots[0]["data"] = {"symbol": "STOCK1", "qty": 100, "entry_price": 100}
    
    strategy.slots[1]["status"] = "OCCUPIED"
    strategy.slots[1]["data"] = {"symbol": "STOCK2", "qty": 100, "entry_price": 100}
    
    strategy.slots[2]["status"] = "FREE"
    strategy.slots[2]["data"] = {}
    
    # Mark slot 3 as PENDING_FREE (just sold PSUBNKBEES)
    strategy.slots[3]["status"] = "PENDING_FREE"
    strategy.slots[3]["last_symbol_info"] = {"symbol": "PSUBNKBEES"}
    strategy.slots[3]["data"] = {}
    
    # Set available cash (similar to user's log: ₹4,71,869.43)
    strategy.available_cash = 471869.43
    
    print(f"\nBefore Recalculation:")
    print(f"  Available Cash: {strategy.available_cash}")
    print(f"  Slots: {[s['status'] for s in strategy.slots]}")
    
    # 1. Test Capital Recalculation
    print("\n--- Testing Capital Recalculation ---")
    strategy._recalculate_slot_capital()
    
    # Verification: Divider should be 2 (FREE + PENDING_FREE)
    expected_divisor = 2
    expected_capital = strategy.available_cash / expected_divisor
    
    print(f"  Resulting Slot Capital: {strategy.slot_capital}")
    if abs(strategy.slot_capital - expected_capital) < 0.01:
        print("  [SUCCESS] Capital correctly divided among 2 slots (including pending).")
    else:
        print(f"  [FAILURE] Capital divided incorrectly. Expected {expected_capital}, got {strategy.slot_capital}")

    # 2. Test Same-Day Re-entry Prevention
    print("\n--- Testing Same-Day Re-entry Prevention ---")
    eligible_etfs = [
        {"symbol": "PSUBNKBEES", "close": 40.32, "distance": 6.42, "sma": 37.89},
        {"symbol": "GOLDBEES", "close": 50.36, "distance": 10.82, "sma": 45.44}
    ]
    
    entries = strategy.process_entries(eligible_etfs, datetime.now())
    
    symbols_bought = [e["symbol"] for e in entries]
    print(f"  Symbols Eligible: {[e['symbol'] for e in eligible_etfs]}")
    print(f"  Symbols Bought: {symbols_bought}")
    
    if "PSUBNKBEES" not in symbols_bought:
        print("  [SUCCESS] PSUBNKBEES correctly blocked from same-day re-entry.")
    else:
        print("  [FAILURE] PSUBNKBEES was bought again on the same day it was sold!")
        
    if "GOLDBEES" in symbols_bought:
        print("  [SUCCESS] GOLDBEES was bought in the free slot.")
    else:
        print("  [FAILURE] GOLDBEES was not bought despite a free slot being available.")

if __name__ == "__main__":
    test_logic_fix()
