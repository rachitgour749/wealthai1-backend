"""Simple database table creation script"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import create_engine, text

# Neon PostgreSQL Database URL
NEON_DATABASE_URL = "postgresql://neondb_owner:npg_WgVhOYtnP12l@ep-solitary-silence-a1yoj91r.ap-southeast-1.aws.neon.tech/ApplicationData?sslmode=require&channel_binding=require"

print(f"Connecting to Neon PostgreSQL database...")

# Create engine
engine = create_engine(NEON_DATABASE_URL)

try:
    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'portfolio_trades'
            );
        """))
        
        table_exists = result.scalar()
        
        if table_exists:
            print("✅ portfolio_trades table already exists")
        else:
            print("Creating portfolio_trades table...")
            
            # Create table
            conn.execute(text("""
                CREATE TABLE portfolio_trades (
                    id SERIAL PRIMARY KEY,
                    user_email VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) NOT NULL,
                    strategy_name VARCHAR(255) NOT NULL,
                    strategy_type VARCHAR(100) NOT NULL,
                    client_code VARCHAR(255) NOT NULL,
                    trade_date DATE NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    quantity INTEGER NOT NULL,
                    price NUMERIC(15, 4) NOT NULL,
                    brokerage NUMERIC(10, 2) DEFAULT 0,
                    taxes NUMERIC(10, 2) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """))
            
            # Create indexes
            conn.execute(text("CREATE INDEX idx_trades_user ON portfolio_trades(user_email);"))
            conn.execute(text("CREATE INDEX idx_trades_run ON portfolio_trades(run_id);"))
            conn.execute(text("CREATE INDEX idx_trades_client ON portfolio_trades(client_code);"))
            conn.execute(text("CREATE INDEX idx_trades_date ON portfolio_trades(trade_date);"))
            conn.execute(text("CREATE INDEX idx_trades_symbol ON portfolio_trades(symbol);"))
            conn.execute(text("CREATE UNIQUE INDEX idx_trades_composite ON portfolio_trades(run_id, client_code, symbol, trade_date, side);"))
            
            conn.commit()
            
            print("✅ Successfully created portfolio_trades table with all indexes")
    
    print("=" * 60)
    print("✅ Migration completed successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
