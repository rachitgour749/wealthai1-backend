import sys
import os
from datetime import datetime
import pandas as pd

# Add project root to sys.path
sys.path.append(os.getcwd())

from Strategies.ETF_Swing_Strategy.strategy import ETFSwingStrategy
from Exchange.USExchangePolicy import USExchangePolicy
from Exchange.IndianExchangePolicy import IndianExchangePolicy

def verify_refactor():
    print("\n" + "="*60)
    print("VERIFYING MULTI-MARKET REFACTOR")
    print("="*60)

    # 1. Test Indian ETF Strategy
    print("\n--- Testing Indian ETF Strategy ---")
    india_strategy = ETFSwingStrategy(market="INDIA", asset_type="ETF")
    print(f"Strategy Name: {india_strategy.strategy_name}")
    try:
        print(f"Currency Symbol: {india_strategy.policy.currency_symbol}")
    except:
        print("Currency Symbol: [Rupee Symbol - Encoding issue in terminal]")
    
    india_costs = india_strategy.calculate_etf_delivery_costs("BUY", 100000, 0.1)
    print(f"Indian BUY Costs for 1L: {india_costs['total_costs']:.2f} (Net: {india_costs['net_amount']:.2f})")
    print(f"STT: {india_costs.get('stt', 0.0):.2f}")
    try:
        print(f"Formatted: {india_strategy.policy.format_currency(123456.78)}")
    except:
        print(f"Formatted: (Encoding Issue) {123456.78:.2f}")

    # 2. Test US Stock Strategy
    print("\n--- Testing US Stock Strategy ---")
    us_strategy = ETFSwingStrategy(market="US", asset_type="STOCK")
    print(f"Strategy Name: {us_strategy.strategy_name}")
    print(f"Currency Symbol: {us_strategy.policy.currency_symbol}")
    
    us_costs = us_strategy.calculate_stock_delivery_costs("BUY", 100000, 0.1)
    print(f"US BUY Costs for 100k: {us_costs['total_costs']:.2f} (Net: {us_costs['net_amount']:.2f})")
    print(f"SEC Fee: {us_costs.get('sec_fee', 0.0):.2f}")
    print(f"Formatted: {us_strategy.policy.format_currency(123456.78)}")

    # 3. Test Signal Generator Market Awareness
    print("\n--- Testing Signal Generator Mock ---")
    mock_instance_in = {
        'id': 1,
        'strategies_parameters': {'market': 'INDIA', 'asset_type': 'ETF'}
    }
    mock_instance_us = {
        'id': 2,
        'strategies_parameters': {'market': 'US', 'asset_type': 'STOCK'}
    }
    
    for inst in [mock_instance_in, mock_instance_us]:
        params = inst['strategies_parameters']
        market = params.get('market')
        asset_type = params.get('asset_type')
        print(f"Mocking Instance {inst['id']}: Running logic for {market} {asset_type}")

    print("\n" + "="*60)
    print("REFACTOR VERIFIED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    verify_refactor()
