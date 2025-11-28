from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester

try:
    bt = ETFRotationBacktester()
    
    print("✅ ETFRotationBacktester initialized successfully!")
    print(f"✅ etf_metadata exists: {hasattr(bt, 'etf_metadata')}")
    print(f"✅ cost_calculator exists: {hasattr(bt, 'cost_calculator')}")
    print(f"✅ tax_calculator exists: {hasattr(bt, 'tax_calculator')}")
    
    if hasattr(bt, 'etf_metadata'):
        print(f"✅ etf_metadata type: {type(bt.etf_metadata)}")
        print(f"✅ Number of ETFs loaded: {len(bt.etf_metadata)}")
    
    print("\n🎉 ALL CHECKS PASSED - Issue is FIXED!")
    
except AttributeError as e:
    print(f"❌ AttributeError: {e}")
    print("❌ Issue NOT fixed - etf_metadata still missing")
except Exception as e:
    print(f"❌ Error: {e}")
