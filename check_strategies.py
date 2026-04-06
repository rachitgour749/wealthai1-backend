import sqlite3
import os

db_path = 'd:\\WEALTHAI_V2\\wealthai-backend-v2\\Databases\\app_data.sqlite'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT strategy_name, strategy_type FROM saved_instances;")
    rows = cursor.fetchall()
    print("Strategy Name | Strategy Type")
    print("-" * 40)
    for row in rows:
        print(f"{row[0]} | {row[1]}")
    conn.close()
else:
    print(f"Database not found at {db_path}")
