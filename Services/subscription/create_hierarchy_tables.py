"""Database migration script for hierarchical user access system"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import create_engine, text

# Neon PostgreSQL Database URL
NEON_DATABASE_URL = "postgresql://neondb_owner:npg_WgVhOYtnP12l@ep-solitary-silence-a1yoj91r.ap-southeast-1.aws.neon.tech/ApplicationData?sslmode=require&channel_binding=require"

print("=" * 60)
print("Hierarchical User Access System - Database Migration")
print("=" * 60)
print(f"\nConnecting to Neon PostgreSQL database...")

# Create engine
engine = create_engine(NEON_DATABASE_URL)

try:
    with engine.connect() as conn:
        # Step 1: Check if user_hierarchy table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'user_hierarchy'
            );
        """))
        
        table_exists = result.scalar()
        
        if table_exists:
            print("✅ user_hierarchy table already exists")
        else:
            print("\n📋 Creating user_hierarchy table...")
            
            # Create user_hierarchy table
            conn.execute(text("""
                CREATE TABLE user_hierarchy (
                    id SERIAL PRIMARY KEY,
                    user_email VARCHAR(255) NOT NULL,
                    parent_email VARCHAR(255),
                    hierarchy_level INTEGER NOT NULL DEFAULT 0,
                    created_by VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,
                    
                    UNIQUE(user_email),
                    CHECK (user_email != parent_email)
                );
            """))
            
            # Create indexes
            conn.execute(text("CREATE INDEX idx_user_hierarchy_parent ON user_hierarchy(parent_email);"))
            conn.execute(text("CREATE INDEX idx_user_hierarchy_user ON user_hierarchy(user_email);"))
            conn.execute(text("CREATE INDEX idx_user_hierarchy_level ON user_hierarchy(hierarchy_level);"))
            
            print("✅ user_hierarchy table created successfully")
        
        # Step 2: Add columns to user_details if they don't exist
        print("\n📋 Checking user_details table...")
        
        # Check if role column exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'user_details' 
                AND column_name = 'role'
            );
        """))
        
        role_exists = result.scalar()
        
        if not role_exists:
            print("   Adding 'role' column to user_details...")
            conn.execute(text("ALTER TABLE user_details ADD COLUMN role VARCHAR(50) DEFAULT 'CLIENT';"))
            print("   ✅ Added 'role' column")
        else:
            print("   ✅ 'role' column already exists")
        
        # Check if can_manage_users column exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'user_details' 
                AND column_name = 'can_manage_users'
            );
        """))
        
        manage_exists = result.scalar()
        
        if not manage_exists:
            print("   Adding 'can_manage_users' column to user_details...")
            conn.execute(text("ALTER TABLE user_details ADD COLUMN can_manage_users BOOLEAN DEFAULT FALSE;"))
            print("   ✅ Added 'can_manage_users' column")
        else:
            print("   ✅ 'can_manage_users' column already exists")
        
        conn.commit()
        
        print("\n" + "=" * 60)
        print("✅ Migration completed successfully!")
        print("=" * 60)
        print("\nDatabase changes:")
        print("  • user_hierarchy table created/verified")
        print("  • user_details.role column added/verified")
        print("  • user_details.can_manage_users column added/verified")
        print("  • All indexes created")
        print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
