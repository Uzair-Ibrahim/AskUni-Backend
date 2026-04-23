from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# .env se password load karo
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Engine (Gari) banao jo database tak jayegi
try:
    engine = create_engine(DB_URL)
    # Ek dafa connect kar ke check karte hain
    with engine.connect() as connection:
        print("✅ Mashallah! PostgreSQL se connection successful ho gaya hai!")
except Exception as e:
    print(f"❌ Connection fail ho gaya. Error: {e}")