"""
Quick test to verify the new transaction cost calculation logic.

This script demonstrates the new calculation flow:
1. Units = capital ÷ price
2. Transaction costs calculated on actual purchase amount
3. Total cost = purchase + costs
"""

# Test parameters
weekly_capital = 50000
price_per_unit = 64.55
brokerage_percent = 0.03

# Step 1: Calculate units directly
units = int(weekly_capital / price_per_unit)
print(f"Weekly Capital: ₹{weekly_capital:,.0f}")
print(f"Price per Unit: ₹{price_per_unit:.2f}")
print(f"Units to Buy: {units}")
print()

# Step 2: Calculate actual purchase amount
actual_purchase = units * price_per_unit
print(f"Actual Purchase Amount: ₹{actual_purchase:,.2f}")
print()

# Step 3: Calculate transaction costs on actual purchase
# Simplified calculation (actual implementation uses calculate_transaction_costs)
brokerage = actual_purchase * (brokerage_percent / 100)
exchange_charges = actual_purchase * 0.00325 / 100
stamp_duty = actual_purchase * 0.015 / 100
sebi_charges = actual_purchase * 0.0001 / 100
gst = (brokerage + exchange_charges) * 0.18
total_costs = brokerage + exchange_charges + stamp_duty + sebi_charges + gst

print(f"Transaction Cost Breakdown:")
print(f"  Brokerage (0.03%): ₹{brokerage:.2f}")
print(f"  Exchange Charges: ₹{exchange_charges:.2f}")
print(f"  Stamp Duty: ₹{stamp_duty:.2f}")
print(f"  SEBI Charges: ₹{sebi_charges:.2f}")
print(f"  GST (18%): ₹{gst:.2f}")
print(f"  Total Costs: ₹{total_costs:.2f}")
print()

# Step 4: Calculate total cost
total_cost = actual_purchase + total_costs
print(f"Total Cost (Purchase + Costs): ₹{total_cost:,.2f}")
print()

# Verify against expected values
print("=" * 50)
print("VERIFICATION:")
print(f"Expected Units: 774")
print(f"Actual Units: {units}")
print(f"Match: {'✅' if units == 774 else '❌'}")
print()
print(f"Expected Total Cost: ₹49,971")
print(f"Actual Total Cost: ₹{total_cost:,.0f}")
print(f"Match: {'✅' if abs(total_cost - 49971) < 1 else '❌'}")
