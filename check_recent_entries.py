import sqlite3
import json
from datetime import datetime

# Check most recent entries in database
conn = sqlite3.connect('unified_etf_data.sqlite')
cursor = conn.cursor()

cursor.execute("SELECT * FROM savejson ORDER BY id DESC LIMIT 3")
saved_jsons = cursor.fetchall()

print("Most recent entries in database:")
for i, saved_json in enumerate(saved_jsons):
    print(f"\nEntry {i+1}:")
    json_data = json.loads(saved_json[1])
    print(f"User Email: {json_data.get('user_email')}")
    print(f"Strategy Name: {json_data.get('strategy_name')}")
    print(f"Execution Date: {json_data.get('execution_date')}")
    print(f"Execution Time: {json_data.get('execution_time')}")
    print(f"Full Timestamp: {json_data.get('full_timestamp')}")
    print(f"ISO Timestamp: {json_data.get('iso_timestamp')}")
    print(f"All keys: {list(json_data.keys())}")

conn.close()
