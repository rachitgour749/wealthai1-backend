import sqlite3
import json

# Check current database state
conn = sqlite3.connect('unified_etf_data.sqlite')
cursor = conn.cursor()

cursor.execute("SELECT * FROM savejson ORDER BY id DESC LIMIT 5")
saved_jsons = cursor.fetchall()

print("Current database entries:")
for i, saved_json in enumerate(saved_jsons):
    print(f"\nEntry {i+1}:")
    print(f"ID: {saved_json[0]}")
    json_data = json.loads(saved_json[1])
    print(f"User Email: {json_data.get('user_email')}")
    print(f"Strategy Name: {json_data.get('strategy_name')}")
    
    # Check the actual JSON content
    json_content = json_data.get('json_data', {})
    print(f"JSON Content Keys: {list(json_content.keys())}")
    
    # Check for problematic fields
    if 'strategy_type' in json_content:
        print(f"❌ Contains 'strategy_type': {json_content['strategy_type']}")
    if 'timestamp' in json_content:
        print(f"❌ Contains 'timestamp': {json_content['timestamp']}")
    
    # Show the full JSON content for debugging
    print(f"Full JSON: {json.dumps(json_content, indent=2)}")

conn.close()
