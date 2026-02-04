"""
Test Script for Signal Generation

Manually trigger signal generation for testing purposes.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.scheduler.generators.etf_rotation_generator import generate_etf_rotation_signals
from Services.scheduler.generators.rotation_stocks_generator import generate_stock_rotation_signals
from Services.scheduler.generators.international_etf_generator import generate_international_etf_signals


def test_etf_rotation():
    """Test ETF Rotation signal generation"""
    print("\n" + "="*60)
    print("TESTING ETF ROTATION SIGNAL GENERATION")
    print("="*60)
    
    try:
        generate_etf_rotation_signals()
        print("\n✓ ETF Rotation signal generation completed")
    except Exception as e:
        print(f"\n✗ ETF Rotation signal generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_stock_rotation():
    """Test Stock Rotation signal generation"""
    print("\n" + "="*60)
    print("TESTING STOCK ROTATION SIGNAL GENERATION")
    print("="*60)
    
    try:
        generate_stock_rotation_signals()
        print("\n✓ Stock Rotation signal generation completed")
    except Exception as e:
        print(f"\n✗ Stock Rotation signal generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_international_etf():
    """Test International ETF signal generation"""
    print("\n" + "="*60)
    print("TESTING INTERNATIONAL ETF SIGNAL GENERATION")
    print("="*60)
    
    try:
        generate_international_etf_signals()
        print("\n✓ International ETF signal generation completed")
    except Exception as e:
        print(f"\n✗ International ETF signal generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_all():
    """Test all signal generators"""
    print("\n" + "="*60)
    print("TESTING ALL SIGNAL GENERATORS")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    test_etf_rotation()
    test_stock_rotation()
    test_international_etf()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test signal generation")
    parser.add_argument(
        '--strategy',
        choices=['etf', 'stocks', 'international', 'all'],
        default='all',
        help='Strategy to test (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.strategy == 'etf':
        test_etf_rotation()
    elif args.strategy == 'stocks':
        test_stock_rotation()
    elif args.strategy == 'international':
        test_international_etf()
    else:
        test_all()
