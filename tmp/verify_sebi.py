import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

# Mock PriceService before importing order_utils
from Services.portfolio.price_service import PriceService
PriceService.get_current_price = MagicMock(return_value=100.0)

from helpers.order_utils import generate_random_mac, calculate_limit_price, validate_user_ip_creds
from Databases.broker_models import BrokerSession

def test_mac_generation():
    mac1 = generate_random_mac()
    mac2 = generate_random_mac()
    print(f"Generated Random MACs: {mac1}, {mac2}")
    assert mac1 != mac2
    assert len(mac1.split(':')) == 6
    print("✓ MAC Generation Test Passed")

def test_limit_price():
    # Buy side (+0.5%)
    buy_price, _ = calculate_limit_price("RELIANCE", "NSE", "BUY")
    print(f"Buy Limit Price for 100.0: {buy_price}")
    assert buy_price == 100.5
    
    # Sell side (-0.5%)
    sell_price, _ = calculate_limit_price("RELIANCE", "NSE", "SELL")
    print(f"Sell Limit Price for 100.0: {sell_price}")
    assert sell_price == 99.5
    print("✓ Limit Price Test Passed")

def test_validation():
    # Test Invalid Creds
    bad_record = BrokerSession(
        user_email="test@example.com",
        static_ip="1.2.3.4",
        static_ip_username="wrong",
        static_ip_password="wrong",
        static_ip_port="80"
    )
    is_valid, err = validate_user_ip_creds(bad_record)
    print(f"Validation Failure Test (Expected): {is_valid}, {err}")
    assert not is_valid
    
    # Test Valid Creds
    good_record = BrokerSession(
        user_email="test@example.com",
        static_ip="1.2.3.4",
        static_ip_username="anjanr",
        static_ip_password="vRDunhrENZR",
        static_ip_port="50100"
    )
    is_valid, err = validate_user_ip_creds(good_record)
    print(f"Validation Success Test: {is_valid}, {err}")
    assert is_valid
    print("✓ Credential Validation Test Passed")

if __name__ == "__main__":
    print("-" * 30)
    print("RUNNING VERIFICATION TESTS")
    print("-" * 30)
    try:
        test_mac_generation()
        test_limit_price()
        test_validation()
        print("-" * 30)
        print("ALL TESTS PASSED SUCCESSFULLY! ✅")
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        sys.exit(1)
