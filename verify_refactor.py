
import sys
import os

# Ensure we can import from Services (root current dir)
sys.path.insert(0, os.getcwd())

print("Attempting to import Services.ChatAI.api.main...")
try:
    from Services.ChatAI.api.main import router
    print("✅ Successfully imported router from Services.ChatAI.api.main")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Exception: {e}")
    sys.exit(1)

print("\nAttempting to import Services.ChatAI.api.admin_routes...")
try:
    from Services.ChatAI.api.admin_routes import CHATAI_ROOT, DATA_DIR
    print(f"✅ CHATAI_ROOT: {CHATAI_ROOT}")
    print(f"✅ DATA_DIR: {DATA_DIR}")
    
    # Verify path existence
    if not os.path.exists(CHATAI_ROOT):
        print(f"⚠️ Warning: CHATAI_ROOT does not exist on disk (expected for dry run if files moved)")
    else:
        print(f"✅ CHATAI_ROOT exists on disk")
        
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)

print("\nVerifying server.py imports...")
try:
    import server
    print("✅ server.py imported without error")
except Exception as e:
    print(f"❌ server.py import failed: {e}")
    sys.exit(1)
