from database.sql_db import engine
from sqlalchemy import text

with engine.connect() as conn:
    # All tables
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
    print("=== TABLES ===")
    for row in result:
        print(row[0])

    # All columns of every table
    result2 = conn.execute(text("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name"))
    print("\n=== COLUMNS ===")
    for row in result2:
        print(f"{row[0]}  →  {row[1]}")