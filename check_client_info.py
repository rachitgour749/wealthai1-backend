from sqlalchemy import create_engine, text

# Database connection
DATABASE_URL = "postgresql://neondb_owner:npg_WgVhOYtnP12l@ep-solitary-silence-a1yoj91r.ap-southeast-1.aws.neon.tech/ApplicationData?sslmode=require"
engine = create_engine(DATABASE_URL)

run_id = "run_etfs_rotation_strategy_ETFs_Rotation_Payout___4_ETFs___2026_01_05_03_56_59_1768240968"

with engine.connect() as conn:
    # Check if run_id exists in saved_instances
    result = conn.execute(
        text("SELECT client_info FROM saved_instances WHERE run_id = :run_id"),
        {"run_id": run_id}
    )
    row = result.fetchone()
    
    if row:
        print(f"Found in saved_instances table")
        print(f"client_info: {row[0]}")
    else:
        print(f"NOT FOUND in saved_instances table")
        print(f"\nSearching in other strategy tables...")
        
        # Check etf_saved_strategy
        result = conn.execute(
            text("SELECT client_info FROM etf_saved_strategy WHERE run_id = :run_id"),
            {"run_id": run_id}
        )
        row = result.fetchone()
        if row:
            print(f"Found in etf_saved_strategy table")
            print(f"client_info: {row[0]}")
        else:
            print(f"NOT FOUND in etf_saved_strategy table")
