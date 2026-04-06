import sys
import os

# Add project root to path
project_root = 'd:\\WEALTHAI_V2\\wealthai-backend-v2'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    print("Testing Strategy Generator Imports...")
    print("-" * 40)
    
    strategies = [
        ('etf_payout_generator', 'generate_etf_payout_signals'),
        ('etf_swing_generator', 'generate_etf_swing_signals'),
        ('etf_rotation_generator', 'generate_etf_rotation_signals'),
        ('international_etf_generator', 'generate_international_etf_signals'),
        ('rs_etf_strategy_generator', 'generate_rs_etf_signals'),
        ('supertrend_generator', 'generate_supertrend_signals')
    ]
    
    for mod_name, func_name in strategies:
        try:
            # Import from Services.scheduler.generators
            module = __import__(f'Services.scheduler.generators.{mod_name}', fromlist=[func_name])
            func = getattr(module, func_name)
            print(f"[OK] {mod_name}.{func_name} imported successfully")
        except Exception as e:
            print(f"[FAIL] {mod_name}.{func_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_imports()
