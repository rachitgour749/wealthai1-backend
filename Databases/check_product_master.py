import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_data_db_connection import create_connection, get_engine
from sqlalchemy import text, inspect

create_connection()
engine = get_engine()
inspector = inspect(engine)

print("product_master columns:")
cols = inspector.get_columns('product_master')
for c in cols:
    print(f"  {c['name']}: {c['type']}")

with engine.connect() as conn:
    result = conn.execute(text('SELECT * FROM product_master LIMIT 5'))
    rows = result.fetchall()
    print('\nSample data:')
    for row in rows:
        print(row)

