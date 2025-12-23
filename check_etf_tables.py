import sys
sys.path.insert(0, 'Databases')
from market_data_db_connection import create_connection, get_session
from sqlalchemy import text

create_connection()
session = get_session()

# Get etf_info structure
print("ETF_INFO TABLE:")
result = session.execute(text("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'etf_info' 
    ORDER BY ordinal_position
"""))
for row in result:
    print(f"  {row[0]}: {row[1]}")

print("\nSample data:")
result = session.execute(text("SELECT * FROM etf_info LIMIT 3"))
for row in result:
    print(f"  {row}")

print("\n" + "="*80)
print("ETF_METADATA TABLE:")
result = session.execute(text("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'etf_metadata' 
    ORDER BY ordinal_position
"""))
for row in result:
    print(f"  {row[0]}: {row[1]}")

print("\nSample data:")
result = session.execute(text("SELECT * FROM etf_metadata LIMIT 3"))
for row in result:
    print(f"  {row}")

session.close()
