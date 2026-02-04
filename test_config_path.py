"""
Test script to verify scheduler config path resolution
"""
from pathlib import Path
import sys

print("="*60)
print("Testing Scheduler Config Path Resolution")
print("="*60)

# Simulate the path resolution from config_utils.py
current_file = Path(__file__).absolute()
print(f"\n1. Current test file: {current_file}")

# Simulate being in Services/scheduler/config_utils.py
config_utils_path = Path("d:/WEALTHAI_V2/wealthai-backend-v2/Services/scheduler/config_utils.py")
print(f"\n2. Simulated config_utils.py location: {config_utils_path}")

# Calculate base_dir (go up 2 levels)
base_dir = config_utils_path.parent.parent.parent
print(f"\n3. Base directory (parent.parent.parent): {base_dir}")

# Calculate config path
config_path = base_dir / "config" / "scheduler_config.json"
print(f"\n4. Resolved config path: {config_path}")

# Check if file exists
exists = config_path.exists()
print(f"\n5. File exists: {exists}")

if exists:
    print("\n✅ SUCCESS: Config file found at correct location!")
    print(f"   Path: {config_path}")
else:
    print("\n❌ ERROR: Config file not found!")
    print(f"   Expected at: {config_path}")
    
    # Try to find the file
    print("\n   Searching for scheduler_config.json...")
    project_root = Path("d:/WEALTHAI_V2/wealthai-backend-v2")
    for found_file in project_root.rglob("scheduler_config.json"):
        print(f"   Found at: {found_file}")

print("\n" + "="*60)
