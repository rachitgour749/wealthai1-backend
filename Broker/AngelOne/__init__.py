"""
AngelOne Broker Integration

Provides standardized login and order placement functionality.

Usage:
    from Broker.AngelOne import AngelOneAuthenticator, place_order
    
    # Login
    authenticator = AngelOneAuthenticator(api_key, api_secret, client_code)
    access_token, error = authenticator.login(client_code, password, totp_secret)
    
    # Place Order
    credentials = {
        'api_key': api_key,
        'access_token': access_token,
        'client_code': client_code
    }
    order_data = {
        'symbol': 'RELIANCE',
        'exchange': 'NSE',
        'order_side': 'BUY',
        'quantity': 1,
        'product_type': 'DELIVERY',
        'order_type': 'MARKET',
        'exchange_instrument_id': '2885'
    }
    result = place_order(credentials, order_data)
"""

from Broker.AngelOne.ANGELONE import AngelOneAuthenticator, place_order

__all__ = ['AngelOneAuthenticator', 'place_order']
