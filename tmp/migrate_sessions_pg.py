import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def migrate():
    load_dotenv()
    db_url = os.getenv("DATABASE_STRING")
    
    if not db_url:
        print("DATABASE_STRING environment variable is not set")
        return

    print(f"Connecting to database...")
    engine = create_engine(db_url)
    
    try:
        with engine.connect() as conn:
            # 1. Add static_ip_username
            print("Adding column static_ip_username...")
            try:
                conn.execute(text("ALTER TABLE broker_sessions ADD COLUMN IF NOT EXISTS static_ip_username VARCHAR;"))
                conn.commit()
            except Exception as e:
                print(f"Error adding static_ip_username: {e}")

            # 2. Add static_ip_password
            print("Adding column static_ip_password...")
            try:
                conn.execute(text("ALTER TABLE broker_sessions ADD COLUMN IF NOT EXISTS static_ip_password VARCHAR;"))
                conn.commit()
            except Exception as e:
                print(f"Error adding static_ip_password: {e}")

            # 3. Add static_ip_port
            print("Adding column static_ip_port...")
            try:
                conn.execute(text("ALTER TABLE broker_sessions ADD COLUMN IF NOT EXISTS static_ip_port VARCHAR;"))
                conn.commit()
            except Exception as e:
                print(f"Error adding static_ip_port: {e}")

            print("Migration successful.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    migrate()
