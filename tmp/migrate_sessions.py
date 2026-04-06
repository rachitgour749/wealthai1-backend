import sqlite_utils
import os

def migrate():
    # Detect DB path (Assuming SQLite for now as it's common in these projects)
    # The project seems to use SQLite based on the general structure
    db_path = os.path.join(os.getcwd(), 'Databases', 'wealthai.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Please check the path.")
        return

    db = sqlite_utils.Database(db_path)
    table = db["broker_sessions"]
    
    # Adding columns if they don't exist
    cols = {
        "static_ip_username": str,
        "static_ip_password": str,
        "static_ip_port": str
    }
    
    for col_name, col_type in cols.items():
        if col_name not in table.columns_dict:
            print(f"Adding column {col_name} to broker_sessions...")
            table.add_column(col_name, col_type)
        else:
            print(f"Column {col_name} already exists.")

    print("Migration completed.")

if __name__ == "__main__":
    migrate()
