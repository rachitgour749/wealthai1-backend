from Databases.app_data_db_connection import create_connection, engine
from sqlalchemy import text

def check_schema():
    create_connection() # Initializes the global engine
    from Databases.app_data_db_connection import engine # Import AFTER initialization if needed, or use the one from module scope if it updates
    
    with engine.connect() as conn:
        try:
            # Check columns
            result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'saved_instances';"))
            columns = [row[0] for row in result]
            print("Columns in saved_instances:", columns)
            
            if 'source' in columns:
                print("✅ Source column exists.")
            else:
                print("❌ Source column MISSING. Attempting to add...")
                conn.execute(text("ALTER TABLE saved_instances ADD COLUMN source VARCHAR(50) DEFAULT 'internal';"))
                conn.commit()
                print("✅ Source column ADDED.")
                
        except Exception as e:
            print(f"Error checking schema: {e}")

if __name__ == "__main__":
    check_schema()
