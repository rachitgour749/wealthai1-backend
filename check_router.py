
import sys
import os
sys.path.insert(0, os.path.abspath(os.curdir))

try:
    print("Attempting to import strategy_mgmt_router...")
    from APIs.strategy_management import strategy_mgmt_router
    print("Successfully imported strategy_mgmt_router")
    print(f"Number of routes: {len(strategy_mgmt_router.routes)}")
    for route in strategy_mgmt_router.routes:
        print(f" - {route.path} [{route.methods}]")
except Exception as e:
    print(f"Failed to import strategy_mgmt_router: {e}")
    import traceback
    traceback.print_exc()
