"""Database migration script for centralized strategy management system"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import create_engine, text

# Neon PostgreSQL Database URL
NEON_DATABASE_URL = "postgresql://neondb_owner:npg_WgVhOYtnP12l@ep-solitary-silence-a1yoj91r.ap-southeast-1.aws.neon.tech/ApplicationData?sslmode=require&channel_binding=require"

print("=" * 60)
print("Centralized Strategy Management - Database Migration")
print("=" * 60)
print(f"\nConnecting to Neon PostgreSQL database...")

# Create engine
engine = create_engine(NEON_DATABASE_URL)

try:
    with engine.connect() as conn:
        # Step 1: Create saved_instances table
        print("\nCreating saved_instances table...")
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS saved_instances (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                strategy_name VARCHAR(255),
                strategy_type VARCHAR(100) NOT NULL,
                tickers TEXT,
                start_date VARCHAR(50),
                end_date VARCHAR(50),
                strategies_parameters JSONB,
                use_custom_date BOOLEAN DEFAULT FALSE,
                run_id VARCHAR(100) UNIQUE,
                client_info JSONB,
                webhook_url TEXT,
                status VARCHAR(50) DEFAULT 'not deploy',
                reference_capital FLOAT,
                last_execution_date TIMESTAMP,
                next_execution_date TIMESTAMP,
                email_notification BOOLEAN DEFAULT FALSE,
                telegram_notification BOOLEAN DEFAULT FALSE,
                user_code VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """))
        
        # Step 2: Ensure strategy_name column exists (for existing tables)
        conn.execute(text("""
            ALTER TABLE saved_instances 
            ADD COLUMN IF NOT EXISTS strategy_name VARCHAR(255);
        """))
        
        # Step 3: Update default for status column and migrate existing labels
        conn.execute(text("""
            ALTER TABLE saved_instances 
            ALTER COLUMN status SET DEFAULT 'not deploy';
            
            UPDATE saved_instances 
            SET status = 'not deploy' 
            WHERE status = 'deploy';
        """))
        
        # Create indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_saved_instances_user ON saved_instances(user_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_saved_instances_run_id ON saved_instances(run_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_saved_instances_status ON saved_instances(status);"))
        
        conn.commit()
        
        print("\n" + "=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)
        print("\nDatabase changes:")
        print("  . saved_instances table created/verified")
        print("  . status column default set to 'not deploy'")
        print("  . existing 'deploy' statuses migrated to 'not deploy'")
        print("  . Indexes created for user_id, run_id, and status")
        print("=" * 60)
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
