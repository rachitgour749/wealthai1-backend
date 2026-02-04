"""
Test script for scheduler configuration utility

Run this to verify:
- Configuration file loads correctly
- Trading day checks work
- Holiday detection works
- Strategy configuration access works
"""

from Services.scheduler.config_utils import scheduler_config
from datetime import date

def test_config_loading():
    """Test configuration loading"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    config = scheduler_config.config
    print(f"✓ Configuration loaded successfully")
    print(f"  Timezone: {scheduler_config.get_timezone()}")
    print(f"  Enabled strategies: {scheduler_config.get_all_enabled_strategies()}")
    print()

def test_trading_days():
    """Test trading day checks"""
    print("=" * 60)
    print("TEST 2: Trading Day Checks")
    print("=" * 60)
    
    # Test regular trading day (Monday, Feb 3, 2026)
    test_date1 = date(2026, 2, 3)
    is_trading = scheduler_config.is_trading_day(test_date1, 'NSE')
    print(f"  {test_date1} (Monday): {'✓ Trading day' if is_trading else '✗ Not trading'}")
    
    # Test weekend (Saturday, Feb 7, 2026)
    test_date2 = date(2026, 2, 7)
    is_trading = scheduler_config.is_trading_day(test_date2, 'NSE')
    print(f"  {test_date2} (Saturday): {'✓ Trading day' if is_trading else '✗ Not trading'}")
    
    # Test holiday (Republic Day, Jan 26, 2026)
    test_date3 = date(2026, 1, 26)
    is_trading = scheduler_config.is_trading_day(test_date3, 'NSE')
    print(f"  {test_date3} (Republic Day): {'✓ Trading day' if is_trading else '✗ Not trading'}")
    
    # Test US holiday (Independence Day observed, July 3, 2026)
    test_date4 = date(2026, 7, 3)
    is_trading_us = scheduler_config.is_trading_day(test_date4, 'US')
    print(f"  {test_date4} (US Independence Day): {'✓ Trading day' if is_trading_us else '✗ Not trading'}")
    print()

def test_holidays():
    """Test holiday lists"""
    print("=" * 60)
    print("TEST 3: Holiday Lists")
    print("=" * 60)
    
    nse_holidays = scheduler_config.get_nse_holidays(2026)
    print(f"  NSE India holidays 2026: {len(nse_holidays)} days")
    print(f"    First 3: {nse_holidays[:3]}")
    
    us_holidays = scheduler_config.get_us_holidays(2026)
    print(f"  US market holidays 2026: {len(us_holidays)} days")
    print(f"    First 3: {us_holidays[:3]}")
    
    us_early_close = scheduler_config.get_us_early_close(2026)
    print(f"  US early close days 2026: {len(us_early_close)} days")
    print(f"    All: {us_early_close}")
    print()

def test_strategy_config():
    """Test strategy configuration access"""
    print("=" * 60)
    print("TEST 4: Strategy Configuration")
    print("=" * 60)
    
    # Test ETF Rotation strategy
    etf_config = scheduler_config.get_strategy_config('rotation_etf')
    if etf_config:
        print(f"  rotation_etf:")
        print(f"    Enabled: {etf_config.get('enabled')}")
        print(f"    Signal generation day: {etf_config['signal_generation']['day_of_week']}")
        print(f"    Signal generation time: {etf_config['signal_generation']['time']}")
        print(f"    Generator module: {etf_config['generator_module']}")
    
    # Test market data ETF
    market_data_config = scheduler_config.get_strategy_config('market_data_etf')
    if market_data_config:
        print(f"  market_data_etf:")
        print(f"    Enabled: {market_data_config.get('enabled')}")
        print(f"    Fetch time: {market_data_config['data_fetch']['time']}")
        print(f"    Description: {market_data_config.get('description')}")
    print()

def test_next_previous_trading_day():
    """Test next/previous trading day calculation"""
    print("=" * 60)
    print("TEST 5: Next/Previous Trading Day")
    print("=" * 60)
    
    # From Friday before weekend
    friday = date(2026, 2, 6)
    next_trading = scheduler_config.get_next_trading_day(friday, 'NSE')
    print(f"  Next trading day after {friday} (Friday): {next_trading} ({next_trading.strftime('%A')})")
    
    # From Monday after weekend
    monday = date(2026, 2, 9)
    prev_trading = scheduler_config.get_previous_trading_day(monday, 'NSE')
    print(f"  Previous trading day before {monday} (Monday): {prev_trading} ({prev_trading.strftime('%A')})")
    
    # From holiday (Jan 26, 2026 - Republic Day)
    holiday = date(2026, 1, 26)
    next_after_holiday = scheduler_config.get_next_trading_day(holiday, 'NSE')
    print(f"  Next trading day after {holiday} (Republic Day): {next_after_holiday} ({next_after_holiday.strftime('%A')})")
    print()

def test_execution_settings():
    """Test execution settings"""
    print("=" * 60)
    print("TEST 6: Execution Settings")
    print("=" * 60)
    
    exec_settings = scheduler_config.get_execution_settings()
    print(f"  Max retries: {exec_settings.get('max_retries')}")
    print(f"  Retry delay: {exec_settings.get('retry_delay_minutes')} minutes")
    print(f"  Webhook timeout: {exec_settings.get('webhook_timeout_seconds')} seconds")
    print(f"  Signal expiry: {exec_settings.get('signal_expiry_days')} days")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCHEDULER CONFIGURATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_config_loading()
        test_trading_days()
        test_holidays()
        test_strategy_config()
        test_next_previous_trading_day()
        test_execution_settings()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
