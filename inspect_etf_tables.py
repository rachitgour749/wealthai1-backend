"""
Inspect ETF tables structure
"""
from Databases.market_data_db_connection import create_connection, get_engine, get_session
from sqlalchemy import text, inspect

# Connect to database
create_connection()
engine = get_engine()
inspector = inspect(engine)
session = get_session()

print("="*80)
print("ETF_INFO TABLE STRUCTURE")
print("="*80)
cols = inspector.get_columns('etf_info')
for c in cols:
    nullable = "NULL" if c["nullable"] else "NOT NULL"
    print(f"{c['name']:25s} {str(c['type']):25s} {nullable}")

print("\n" + "="*80)
print("ETF_INFO SAMPLE DATA (5 rows)")
print("="*80)
result = session.execute(text('SELECT * FROM etf_info LIMIT 5'))
for row in result:
    print(row)

print("\n" + "="*80)
print("ETF_METADATA TABLE STRUCTURE")
print("="*80)
cols = inspector.get_columns('etf_metadata')
for c in cols:
    nullable = "NULL" if c["nullable"] else "NOT NULL"
    print(f"{c['name']:25s} {str(c['type']):25s} {nullable}")

print("\n" + "="*80)
print("ETF_METADATA SAMPLE DATA (5 rows)")
print("="*80)
result = session.execute(text('SELECT * FROM etf_metadata LIMIT 5'))
for row in result:
    print(row)

print("\n" + "="*80)
print("CHECKING FOR INTERNATIONAL ETFS")
print("="*80)

# Check if etf_info has market/country columns
result = session.execute(text("SELECT DISTINCT country FROM etf_info WHERE country IS NOT NULL LIMIT 10"))
countries = [row[0] for row in result]
print(f"Countries found in etf_info: {countries}")

# Count ETFs by country
result = session.execute(text("SELECT country, COUNT(*) as count FROM etf_info GROUP BY country ORDER BY count DESC"))
print("\nETF count by country:")
for row in result:
    print(f"  {row[0]}: {row[1]} ETFs")

session.close()
print("\n" + "="*80)
