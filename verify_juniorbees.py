from Databases.app_data_db_connection import get_session, create_connection
from sqlalchemy import text
import pandas as pd

create_connection()
session = get_session()

print(f"{'Date':<12} | {'Symbol':<10} | {'Open':<10} | {'Close':<10}")
print("-" * 50)

query = text("SELECT date, symbol, open, close FROM etf_market WHERE symbol='JUNIORBEES' AND date BETWEEN '2022-11-10' AND '2022-11-20' ORDER BY date")
rows = session.execute(query).fetchall()

for r in rows:
    d = r[0].strftime('%Y-%m-%d')
    s = r[1]
    o = f"{r[2]:.2f}"
    c = f"{r[3]:.2f}"
    print(f"{d:<12} | {s:<10} | {o:<10} | {c:<10}")

session.close()
