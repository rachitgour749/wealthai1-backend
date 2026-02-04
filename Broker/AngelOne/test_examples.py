"""
AngelOne Broker Integration - Test Examples

This file contains usage examples for testing the AngelOne broker integration.
"""

# Example 1: Login
def test_login():
    """Test AngelOne login functionality"""
    from Broker.AngelOne import AngelOneAuthenticator
    
    # Initialize authenticator
    authenticator = AngelOneAuthenticator(
        api_key="YOUR_API_KEY",
        api_secret=None,  # Not used for AngelOne
        client_code="YOUR_CLIENT_CODE"
    )
    
    # Perform login
    access_token, error = authenticator.login(
        client_code="YOUR_CLIENT_CODE",
        password="YOUR_PASSWORD",
        totp_secret="YOUR_TOTP_SECRET"
    )
    
    if access_token:
        print(f"✓ Login successful!")
        print(f"  Access Token: {access_token[:20]}...")
        print(f"  Refresh Token: {authenticator.refresh_token[:20]}...")
        print(f"  Feed Token: {authenticator.feed_token[:20]}...")
        return access_token, authenticator.refresh_token
    else:
        print(f"✗ Login failed: {error}")
        return None, None


# Example 2: Place Market Order (Cash Segment)
def test_place_market_order():
    """Test placing a market order in cash segment"""
    from Broker.AngelOne import place_order
    
    # First login to get access token
    access_token, refresh_token = test_login()
    
    if not access_token:
        print("Cannot place order without access token")
        return
    
    credentials = {
        'api_key': 'YOUR_API_KEY',
        'access_token': access_token,
        'refresh_token': refresh_token,
        'client_code': 'YOUR_CLIENT_CODE'
    }
    
    order_data = {
        'symbol': 'RELIANCE',  # Will be auto-formatted to RELIANCE-EQ
        'exchange': 'NSE',
        'order_side': 'BUY',
        'quantity': 1,
        'product_type': 'DELIVERY',
        'order_type': 'MARKET',
        'exchange_instrument_id': '2885',  # RELIANCE symboltoken
        'lotsize': 1
    }
    
    print("\nPlacing market order...")
    result = place_order(credentials, order_data)
    
    if result['status'] == 'success':
        print(f"✓ Order placed successfully!")
        print(f"  Order ID: {result['data'].get('orderid')}")
        print(f"  Unique Order ID: {result['data'].get('uniqueorderid')}")
    else:
        print(f"✗ Order failed: {result['message']}")
        print(f"  Error Type: {result.get('error_type')}")


# Example 3: Place Limit Order
def test_place_limit_order():
    """Test placing a limit order"""
    from Broker.AngelOne import place_order
    
    access_token, refresh_token = test_login()
    
    if not access_token:
        return
    
    credentials = {
        'api_key': 'YOUR_API_KEY',
        'access_token': access_token,
        'client_code': 'YOUR_CLIENT_CODE'
    }
    
    order_data = {
        'symbol': 'INFY',
        'exchange': 'NSE',
        'order_side': 'BUY',
        'quantity': 10,
        'product_type': 'INTRADAY',
        'order_type': 'LIMIT',
        'price': 1500.50,
        'exchange_instrument_id': '1594',  # INFY symboltoken
        'lotsize': 1
    }
    
    print("\nPlacing limit order...")
    result = place_order(credentials, order_data)
    
    if result['status'] == 'success':
        print(f"✓ Limit order placed!")
        print(f"  Order ID: {result['data'].get('orderid')}")
    else:
        print(f"✗ Order failed: {result['message']}")


# Example 4: Place Stop Loss Order
def test_place_stoploss_order():
    """Test placing a stop loss order"""
    from Broker.AngelOne import place_order
    
    access_token, refresh_token = test_login()
    
    if not access_token:
        return
    
    credentials = {
        'api_key': 'YOUR_API_KEY',
        'access_token': access_token,
        'client_code': 'YOUR_CLIENT_CODE'
    }
    
    order_data = {
        'symbol': 'TCS',
        'exchange': 'NSE',
        'order_side': 'SELL',
        'quantity': 5,
        'product_type': 'DELIVERY',
        'order_type': 'STOPLOSS_LIMIT',
        'price': 3500.00,
        'trigger_price': 3550.00,
        'exchange_instrument_id': '11536',  # TCS symboltoken
        'lotsize': 1
    }
    
    print("\nPlacing stop loss order...")
    result = place_order(credentials, order_data)
    
    if result['status'] == 'success':
        print(f"✓ Stop loss order placed!")
        print(f"  Order ID: {result['data'].get('orderid')}")
    else:
        print(f"✗ Order failed: {result['message']}")


# Example 5: Test Parameter Mapping
def test_parameter_mapping():
    """Test parameter mapping functions"""
    from Broker.AngelOne.Mapping import (
        map_exchange,
        map_product_type,
        map_order_type,
        map_order_side,
        map_variety,
        format_symbol_for_cash
    )
    
    print("\nTesting parameter mapping...")
    
    # Test exchange mapping
    assert map_exchange("NSECM") == "NSE"
    assert map_exchange("NSE") == "NSE"
    print("✓ Exchange mapping works")
    
    # Test product type mapping
    assert map_product_type("CNC") == "DELIVERY"
    assert map_product_type("MIS") == "INTRADAY"
    print("✓ Product type mapping works")
    
    # Test order type mapping
    assert map_order_type("MARKET") == "MARKET"
    assert map_order_type("SL") == "STOPLOSS_LIMIT"
    print("✓ Order type mapping works")
    
    # Test order side mapping
    assert map_order_side("BUY") == "BUY"
    assert map_order_side("SELL") == "SELL"
    print("✓ Order side mapping works")
    
    # Test variety mapping
    assert map_variety("MARKET") == "NORMAL"
    assert map_variety("STOPLOSS_LIMIT") == "STOPLOSS"
    print("✓ Variety mapping works")
    
    # Test symbol formatting
    assert format_symbol_for_cash("RELIANCE", "NSE") == "RELIANCE-EQ"
    assert format_symbol_for_cash("RELIANCE-EQ", "NSE") == "RELIANCE-EQ"
    print("✓ Symbol formatting works")
    
    print("\n✓ All parameter mapping tests passed!")


if __name__ == "__main__":
    print("="*60)
    print("ANGELONE BROKER INTEGRATION - TEST EXAMPLES")
    print("="*60)
    
    # Test parameter mapping (safe to run)
    test_parameter_mapping()
    
    # Uncomment to test login and order placement
    # WARNING: These will make real API calls
    # Make sure to use correct credentials and test account
    
    # test_login()
    # test_place_market_order()
    # test_place_limit_order()
    # test_place_stoploss_order()
    
    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)
