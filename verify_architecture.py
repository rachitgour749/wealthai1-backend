import sys
import os
import inspect

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from CoreLogic.WealthAIBase import WealthAIBase
    from Exchange.IndianExchange import IndianExchange
    from Segments.EquitySegment import EquitySegment
    from Segments.DerivativesSegment import DerivativesSegment
    from Strategies.Rotation.RotationStrategy import RotationStrategy
    from Strategies.RS.RSStrategy import RSStrategy
    from Strategies.Rotation_ETF.services.backtester import ETFRotationBacktester
    from Strategies.Rotation_Stocks.services.backtester import StockRotationBacktester
    from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester
    from Strategies.RS_Stocks.rs_backtester_core import RSStrategyBacktester

    print("✅ All modules imported successfully.")

    def check_inheritance(cls, expected_base, name):
        if issubclass(cls, expected_base):
            print(f"✅ {name} inherits from {expected_base.__name__}")
        else:
            print(f"❌ {name} DOES NOT inherit from {expected_base.__name__}")
            print(f"   MRO: {cls.mro()}")

    print("\n" + "="*60)
    print("VERIFYING CLASS HIERARCHY")
    print("="*60)
    
    print("\n📁 New Folder Structure:")
    print("  CoreLogic/WealthAIBase.py")
    print("  Exchange/IndianExchange.py")
    print("  Segments/EquitySegment.py")
    print("  Segments/DerivativesSegment.py")
    print("  Strategies/Rotation/RotationStrategy.py")
    print("  Strategies/RS/RSStrategy.py")
    
    print("\n🔗 Inheritance Chain:")
    check_inheritance(IndianExchange, WealthAIBase, "IndianExchange")
    check_inheritance(EquitySegment, IndianExchange, "EquitySegment")
    check_inheritance(DerivativesSegment, IndianExchange, "DerivativesSegment")
    check_inheritance(RotationStrategy, EquitySegment, "RotationStrategy")
    check_inheritance(RSStrategy, EquitySegment, "RSStrategy")
    check_inheritance(ETFRotationBacktester, RotationStrategy, "ETFRotationBacktester")
    check_inheritance(StockRotationBacktester, RotationStrategy, "StockRotationBacktester")
    check_inheritance(RSETFStrategyBacktester, RSStrategy, "RSETFStrategyBacktester")
    check_inheritance(RSStrategyBacktester, RSStrategy, "RSStrategyBacktester")
    
    print("\n" + "="*60)
    print("✅ ARCHITECTURE VERIFICATION COMPLETE")
    print("="*60)

except ImportError as e:
    print(f"❌ ImportError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
