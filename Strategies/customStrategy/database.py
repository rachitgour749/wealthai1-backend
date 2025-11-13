import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

class CustomStrategyDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use the unified database path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.db_path = os.path.join(current_dir, "unified_etf_data.sqlite")
        else:
            self.db_path = db_path
        
        self.init_database()
    
    def init_database(self):
        """Initialize the custom_strategies table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists and has old schema
            cursor.execute("PRAGMA table_info(custom_strategies)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if not columns:
                # Table doesn't exist, create new schema
                cursor.execute('''
                    CREATE TABLE custom_strategies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_email TEXT NOT NULL,
                        user_phone TEXT NOT NULL,
                        strategy_description TEXT NOT NULL,
                        ai_analysis_json TEXT NOT NULL,
                        strategy_rating INTEGER,
                        status TEXT DEFAULT 'pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            elif 'user_id' in columns and 'user_email' in columns:
                # Old schema exists, migrate to new schema
                print("🔄 Migrating custom_strategies table to new schema...")
                
                # Create new table with correct schema
                cursor.execute('''
                    CREATE TABLE custom_strategies_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_email TEXT NOT NULL,
                        user_phone TEXT NOT NULL,
                        strategy_description TEXT NOT NULL,
                        ai_analysis_json TEXT NOT NULL,
                        strategy_rating INTEGER,
                        status TEXT DEFAULT 'pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Copy data from old table to new table
                cursor.execute('''
                    INSERT INTO custom_strategies_new 
                    (id, user_email, user_phone, strategy_description, ai_analysis_json, strategy_rating, status, created_at, updated_at)
                    SELECT id, user_email, user_phone, strategy_description, ai_analysis_json, strategy_rating, status, created_at, updated_at
                    FROM custom_strategies
                ''')
                
                # Drop old table and rename new table
                cursor.execute('DROP TABLE custom_strategies')
                cursor.execute('ALTER TABLE custom_strategies_new RENAME TO custom_strategies')
                
                print("✅ Migration completed successfully")
            else:
                # Table exists with correct schema, just ensure it's up to date
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS custom_strategies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_email TEXT NOT NULL,
                        user_phone TEXT NOT NULL,
                        strategy_description TEXT NOT NULL,
                        ai_analysis_json TEXT NOT NULL,
                        strategy_rating INTEGER,
                        status TEXT DEFAULT 'pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            
            # Create index on user_email for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_custom_strategies_user_email 
                ON custom_strategies(user_email)
            ''')
            
            conn.commit()
            conn.close()
            print("[SUCCESS] Custom strategies table initialized successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Error initializing custom strategies table: {e}")
            return False
    
    def save_strategy(self, user_email: str, user_phone: str, 
                     strategy_description: str, analysis: Dict[str, Any]) -> int:
        """Save a custom strategy to the database"""
        conn = None
        try:
            # Use timeout to prevent database locking
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            
            # Convert analysis to JSON string
            analysis_json = json.dumps(analysis)
            strategy_rating = analysis.get('strategy_rating', 0)
            
            cursor.execute('''
                INSERT INTO custom_strategies 
                (user_email, user_phone, strategy_description, 
                 ai_analysis_json, strategy_rating, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_email, user_phone, strategy_description, 
                  analysis_json, strategy_rating, 'pending'))
            
            strategy_id = cursor.lastrowid
            conn.commit()
            
            print(f"[SUCCESS] Custom strategy saved with ID: {strategy_id}")
            return strategy_id
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"[WARNING] Database is locked, retrying in 1 second...")
                import time
                time.sleep(1)
                # Retry once
                return self.save_strategy(user_email, user_phone, strategy_description, analysis)
            else:
                print(f"[ERROR] Database error: {e}")
                raise e
        except Exception as e:
            print(f"[ERROR] Error saving custom strategy: {e}")
            raise e
        finally:
            if conn:
                conn.close()
    
    def get_strategies_by_email(self, user_email: str) -> List[Dict[str, Any]]:
        """Get all custom strategies for a user by email"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, user_email, user_phone, strategy_description,
                       ai_analysis_json, strategy_rating, status, created_at, updated_at
                FROM custom_strategies 
                WHERE user_email = ?
                ORDER BY created_at DESC
            ''', (user_email,))
            
            rows = cursor.fetchall()
            conn.close()
            
            strategies = []
            for row in rows:
                try:
                    analysis = json.loads(row[4]) if row[4] else {}
                    strategies.append({
                        "id": row[0],
                        "user_email": row[1],
                        "user_phone": row[2],
                        "strategy_description": row[3],
                        "analysis": analysis,
                        "strategy_rating": row[5],
                        "status": row[6],
                        "created_at": row[7],
                        "updated_at": row[8]
                    })
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse analysis for strategy ID {row[0]}: {e}")
                    continue
            
            return strategies
        except Exception as e:
            print(f"[ERROR] Error getting strategies for user {user_email}: {e}")
            return []
    
    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific custom strategy by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, user_email, user_phone, strategy_description,
                       ai_analysis_json, strategy_rating, status, created_at, updated_at
                FROM custom_strategies 
                WHERE id = ?
            ''', (strategy_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            try:
                analysis = json.loads(row[4]) if row[4] else {}
                return {
                    "id": row[0],
                    "user_email": row[1],
                    "user_phone": row[2],
                    "strategy_description": row[3],
                    "analysis": analysis,
                    "strategy_rating": row[5],
                    "status": row[6],
                    "created_at": row[7],
                    "updated_at": row[8]
                }
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse analysis for strategy ID {row[0]}: {e}")
                return None
        except Exception as e:
            print(f"[ERROR] Error getting strategy {strategy_id}: {e}")
            return None
    
    def update_strategy_status(self, strategy_id: int, status: str) -> bool:
        """Update strategy status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE custom_strategies 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, strategy_id))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            return success
        except Exception as e:
            print(f"[ERROR] Error updating strategy status: {e}")
            return False
