"""
Execution Database Module
Handles database operations for execution records
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ExecutionDatabase:
    """Database handler for execution records"""
    
    def __init__(self, db_path: str):
        """
        Initialize execution database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_table()
    
    def get_connection(self):
        """Get database connection with timeout"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_table(self):
        """Initialize executed_details table if it doesn't exist"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS executed_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    signal_symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    response_data TEXT,
                    webhook_url TEXT,
                    payload_sent TEXT,
                    execution_date TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check if execution_date column exists, add it if not (for existing tables)
            cursor.execute("PRAGMA table_info(executed_details)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'execution_date' not in columns:
                logger.info("Adding execution_date column to executed_details table...")
                cursor.execute('ALTER TABLE executed_details ADD COLUMN execution_date TEXT')
            
            # Create indexes for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_executed_details_user_id 
                ON executed_details(user_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_executed_details_run_id 
                ON executed_details(run_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_executed_details_execution_date 
                ON executed_details(execution_date)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Execution database table initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing execution database table: {e}")
            raise
    
    def insert_execution_record(
        self,
        user_id: str,
        run_id: str,
        signal_symbol: str,
        side: str,
        success: bool,
        error_message: Optional[str] = None,
        response_data: Optional[str] = None,
        webhook_url: Optional[str] = None,
        payload_sent: Optional[str] = None
    ) -> int:
        """
        Insert execution record into database
        
        Args:
            user_id: User email
            run_id: Deployment run ID
            signal_symbol: Trading symbol
            side: BUY or SELL
            success: Whether execution was successful
            error_message: Error message if failed
            response_data: Webhook response data (JSON string)
            webhook_url: Webhook URL used
            payload_sent: Payload sent to webhook (JSON string)
            
        Returns:
            ID of inserted record
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            execution_date = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute('''
                INSERT INTO executed_details 
                (user_id, run_id, signal_symbol, side, success, error_message, 
                 response_data, webhook_url, payload_sent, execution_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                run_id,
                signal_symbol,
                side,
                success,
                error_message,
                response_data,
                webhook_url,
                payload_sent,
                execution_date
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Execution record inserted: ID={record_id}, Symbol={signal_symbol}, Success={success}")
            return record_id
            
        except Exception as e:
            logger.error(f"Error inserting execution record: {e}")
            raise
    
    def clear_all_execution_data(self):
        """Clear all execution records from database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM executed_details')
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {deleted_count} execution records from database")
            
        except Exception as e:
            logger.error(f"Error clearing execution data: {e}")
            raise
    
    def get_execution_history(
        self,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        signal_symbol: Optional[str] = None,
        success: Optional[bool] = None,
        execution_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution history with optional filters
        
        Args:
            user_id: Filter by user ID
            run_id: Filter by run ID
            signal_symbol: Filter by symbol
            success: Filter by success status
            execution_date: Filter by execution date (YYYY-MM-DD)
            limit: Maximum number of records to return
            
        Returns:
            List of execution record dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM executed_details WHERE 1=1'
            params = []
            
            if user_id:
                query += ' AND user_id = ?'
                params.append(user_id)
            
            if run_id:
                query += ' AND run_id = ?'
                params.append(run_id)
            
            if signal_symbol:
                query += ' AND signal_symbol = ?'
                params.append(signal_symbol)
            
            if success is not None:
                query += ' AND success = ?'
                params.append(success)
            
            if execution_date:
                query += ' AND execution_date = ?'
                params.append(execution_date)
            
            query += ' ORDER BY created_at DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            records = []
            for row in rows:
                record = dict(row)
                # Parse JSON fields
                if record.get('response_data'):
                    try:
                        record['response_data'] = json.loads(record['response_data'])
                    except:
                        pass
                
                if record.get('payload_sent'):
                    try:
                        record['payload_sent'] = json.loads(record['payload_sent'])
                    except:
                        pass
                
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            raise
    
    def get_execution_summary(
        self,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get execution summary statistics
        
        Args:
            user_id: Filter by user ID
            run_id: Filter by run ID
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM executed_details WHERE 1=1'
            params = []
            
            if user_id:
                query += ' AND user_id = ?'
                params.append(user_id)
            
            if run_id:
                query += ' AND run_id = ?'
                params.append(run_id)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            total = len(rows)
            successful = sum(1 for row in rows if row['success'])
            failed = total - successful
            
            # Group by symbol
            symbol_counts = {}
            for row in rows:
                symbol = row['signal_symbol']
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            # Group by side
            buy_count = sum(1 for row in rows if row['side'] == 'BUY')
            sell_count = sum(1 for row in rows if row['side'] == 'SELL')
            
            summary = {
                'total': total,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / total * 100) if total > 0 else 0,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'symbol_counts': symbol_counts
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting execution summary: {e}")
            raise

