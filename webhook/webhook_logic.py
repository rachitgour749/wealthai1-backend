"""
Webhook logic implementation for the Strategy Management Backend
"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from webhook.config import config
from webhook.models import (
    StrategyCreate, StrategyUpdate, StrategyStatusUpdate,
    JsonGenerate, JsonSave, StrategyResponse, HealthResponse
)
from webhook.utils import (
    validate_strategy_data, generate_json_data, send_webhook_notification,
    log_strategy_operation, create_error_response, create_success_response,
    sanitize_input
)

# Get configuration
config_name = os.environ.get('FASTAPI_ENV', 'default')
app_config = config[config_name]

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
DATABASE = app_config.DATABASE

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create strategies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            user_email TEXT,
            webhook TEXT NOT NULL,
            reference_capital TEXT,
            client_ids TEXT,
            capitals TEXT,
            execution_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
    ''')
    
    # Create saved_json table for storing JSON data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_json (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            json_data TEXT NOT NULL,
            strategy_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

class WebhookLogic:
    """Webhook business logic implementation"""
    
    def __init__(self):
        """Initialize webhook logic"""
        self.database = DATABASE
    
    async def get_all_strategies(self) -> List[StrategyResponse]:
        """Get all strategies"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM strategies ORDER BY created_at DESC')
            strategies = cursor.fetchall()
            conn.close()
            
            result = []
            for strategy in strategies:
                result.append(StrategyResponse(
                    id=strategy['id'],
                    strategy_name=strategy['strategy_name'],
                    user_email=strategy['user_email'],
                    webhook=strategy['webhook'],
                    reference_capital=strategy['reference_capital'],
                    client_ids=json.loads(strategy['client_ids']) if strategy['client_ids'] else [],
                    capitals=json.loads(strategy['capitals']) if strategy['capitals'] else [],
                    execution_date=strategy['execution_date'],
                    created_at=strategy['created_at'],
                    status=strategy['status']
                ))
            
            return result
        except Exception as e:
            logger.error(f"Error getting strategies: {str(e)}")
            raise
    
    async def create_strategy(self, strategy: StrategyCreate) -> Dict[str, Any]:
        """Create a new strategy"""
        try:
            # Validate strategy data
            is_valid, validation_errors = validate_strategy_data(strategy.dict())
            if not is_valid:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail={"message": "Validation failed", "errors": validation_errors})
            
            # Sanitize input data
            strategy_name = sanitize_input(strategy.strategyName)
            user_email = sanitize_input(strategy.userEmail or "")
            webhook = sanitize_input(strategy.webhook)
            reference_capital = sanitize_input(strategy.referenceCapital or "")
            
            # Prepare client IDs and capitals data
            client_ids = [client.dict() for client in strategy.clientIds]
            capitals = [capital.dict() for capital in strategy.capitals]
            
            # Sanitize client IDs and capitals
            for client in client_ids:
                client['clientId'] = sanitize_input(client.get('clientId', ''))
            for capital in capitals:
                capital['capital'] = sanitize_input(capital.get('capital', ''))
            
            execution_date = datetime.now().strftime('%B %d, %Y')
            
            # Insert into database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO strategies (strategy_name, user_email, webhook, reference_capital, 
                                      client_ids, capitals, execution_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (strategy_name, user_email, webhook, reference_capital, 
                  json.dumps(client_ids), json.dumps(capitals), execution_date))
            
            strategy_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Log strategy creation
            log_strategy_operation("created", strategy_id, user_email or "anonymous", f"Strategy: {strategy_name}")
            
            # Send webhook notification if webhook URL is provided
            webhook_sent = False
            if webhook:
                try:
                    json_data = generate_json_data(client_ids, capitals)
                    webhook_data = {
                        "strategy_id": strategy_id,
                        "strategy_name": strategy_name,
                        "user_email": user_email or "anonymous",
                        "execution_date": execution_date,
                        "trading_data": json_data
                    }
                    webhook_sent = send_webhook_notification(webhook, webhook_data, app_config.MAX_RETRIES)
                except Exception as e:
                    logger.warning(f"Failed to send webhook notification: {str(e)}")
            
            return create_success_response(
                f"Strategy '{strategy_name}' deployed successfully!",
                {
                    "strategy_id": strategy_id,
                    "execution_date": execution_date,
                    "webhook_sent": webhook_sent
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            raise
    
    async def get_strategy_by_id(self, strategy_id: int) -> StrategyResponse:
        """Get a specific strategy by ID"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
            strategy = cursor.fetchone()
            conn.close()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            return StrategyResponse(
                id=strategy['id'],
                strategy_name=strategy['strategy_name'],
                user_email=strategy['user_email'],
                webhook=strategy['webhook'],
                reference_capital=strategy['reference_capital'],
                client_ids=json.loads(strategy['client_ids']) if strategy['client_ids'] else [],
                capitals=json.loads(strategy['capitals']) if strategy['capitals'] else [],
                execution_date=strategy['execution_date'],
                created_at=strategy['created_at'],
                status=strategy['status']
            )
        except Exception as e:
            logger.error(f"Error getting strategy: {str(e)}")
            raise
    
    async def update_strategy(self, strategy_id: int, strategy_update: StrategyUpdate) -> Dict[str, Any]:
        """Update a specific strategy"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if strategy exists
            cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
            strategy = cursor.fetchone()
            
            if not strategy:
                conn.close()
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Prepare update data
            update_data = {}
            if strategy_update.strategyName is not None:
                update_data['strategy_name'] = strategy_update.strategyName
            if strategy_update.userEmail is not None:
                update_data['user_email'] = strategy_update.userEmail
            if strategy_update.webhook is not None:
                update_data['webhook'] = strategy_update.webhook
            if strategy_update.referenceCapital is not None:
                update_data['reference_capital'] = strategy_update.referenceCapital
            if strategy_update.clientIds is not None:
                update_data['client_ids'] = json.dumps([client.dict() for client in strategy_update.clientIds])
            if strategy_update.capitals is not None:
                update_data['capitals'] = json.dumps([capital.dict() for capital in strategy_update.capitals])
            
            # Update strategy
            if update_data:
                set_clause = ', '.join([f"{key} = ?" for key in update_data.keys()])
                values = list(update_data.values()) + [strategy_id]
                cursor.execute(f'UPDATE strategies SET {set_clause} WHERE id = ?', values)
            
            conn.commit()
            conn.close()
            
            return {"message": "Strategy updated successfully"}
        except Exception as e:
            logger.error(f"Error updating strategy: {str(e)}")
            raise
    
    async def delete_strategy(self, strategy_id: int) -> Dict[str, Any]:
        """Delete a specific strategy"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if strategy exists
            cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
            strategy = cursor.fetchone()
            
            if not strategy:
                conn.close()
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Delete strategy
            cursor.execute('DELETE FROM strategies WHERE id = ?', (strategy_id,))
            conn.commit()
            conn.close()
            
            return {"message": "Strategy deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting strategy: {str(e)}")
            raise
    
    async def update_strategy_status(self, strategy_id: int, status_update: StrategyStatusUpdate) -> Dict[str, Any]:
        """Update strategy status (active/inactive)"""
        try:
            if status_update.status not in ['active', 'inactive']:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Status must be 'active' or 'inactive'")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('UPDATE strategies SET status = ? WHERE id = ?', (status_update.status, strategy_id))
            conn.commit()
            conn.close()
            
            return {"message": f"Strategy status updated to {status_update.status}"}
        except Exception as e:
            logger.error(f"Error updating strategy status: {str(e)}")
            raise
    
    async def health_check(self) -> HealthResponse:
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            database="connected" if os.path.exists(DATABASE) else "not_found",
            version="1.0.0"
        )
    
    async def generate_json_data(self, json_data: JsonGenerate) -> Dict[str, Any]:
        """Generate JSON data for trading orders based on client IDs and capitals"""
        try:
            client_ids = [client.dict() for client in json_data.clientIds]
            capitals = [capital.dict() for capital in json_data.capitals]
            
            if not client_ids or not capitals:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Client IDs and capitals are required")
            
            if len(client_ids) != len(capitals):
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Number of client IDs must match number of capital values")
            
            # Generate JSON data
            generated_json = generate_json_data(client_ids, capitals)
            
            logger.info(f"Generated JSON data for {len(client_ids)} clients")
            
            return create_success_response("JSON data generated successfully", {
                "json_data": generated_json,
                "client_count": len(client_ids)
            })
            
        except Exception as e:
            logger.error(f"Error generating JSON: {str(e)}")
            raise
    
    async def trigger_webhook(self, strategy_id: int) -> Dict[str, Any]:
        """Trigger webhook notification for a specific strategy"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
            strategy = cursor.fetchone()
            conn.close()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Generate JSON data for webhook
            client_ids = json.loads(strategy['client_ids']) if strategy['client_ids'] else []
            capitals = json.loads(strategy['capitals']) if strategy['capitals'] else []
            json_data = generate_json_data(client_ids, capitals)
            
            # Add strategy metadata
            webhook_data = {
                "strategy_id": strategy_id,
                "strategy_name": strategy['strategy_name'],
                "user_email": strategy['user_email'],
                "execution_date": strategy['execution_date'],
                "trading_data": json_data
            }
            
            # Send webhook notification
            webhook_sent = send_webhook_notification(
                strategy['webhook'], 
                webhook_data, 
                app_config.MAX_RETRIES
            )
            
            if webhook_sent:
                log_strategy_operation("webhook_triggered", strategy_id, strategy['user_email'])
                return create_success_response("Webhook notification sent successfully")
            else:
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail="Failed to send webhook notification")
                
        except Exception as e:
            logger.error(f"Error triggering webhook: {str(e)}")
            raise
    
    async def get_strategy_json(self, strategy_id: int) -> Dict[str, Any]:
        """Get JSON data for a specific strategy"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
            strategy = cursor.fetchone()
            conn.close()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Generate JSON data
            client_ids = json.loads(strategy['client_ids']) if strategy['client_ids'] else []
            capitals = json.loads(strategy['capitals']) if strategy['capitals'] else []
            json_data = generate_json_data(client_ids, capitals)
            
            return create_success_response("JSON data retrieved successfully", {
                "strategy_id": strategy_id,
                "strategy_name": strategy['strategy_name'],
                "json_data": json_data
            })
            
        except Exception as e:
            logger.error(f"Error getting strategy JSON: {str(e)}")
            raise
    
    async def save_json_data(self, json_save: JsonSave) -> Dict[str, Any]:
        """Save JSON data for a user"""
        try:
            # Insert into saved_json table
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO saved_json (user_email, json_data, strategy_name)
                VALUES (?, ?, ?)
            ''', (json_save.userEmail, json.dumps(json_save.jsonData), json_save.strategyName))
            
            saved_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"JSON data saved for user {json_save.userEmail} with ID {saved_id}")
            
            return create_success_response("JSON data saved successfully", {
                "saved_id": saved_id,
                "user_email": json_save.userEmail,
                "strategy_name": json_save.strategyName
            })
            
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")
            raise
    
    async def get_saved_json_data(self, user_email: str) -> Dict[str, Any]:
        """Get all saved JSON data for a user"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM saved_json 
                WHERE user_email = ? 
                ORDER BY created_at DESC
            ''', (user_email,))
            saved_jsons = cursor.fetchall()
            conn.close()
            
            result = []
            for saved_json in saved_jsons:
                result.append({
                    'id': saved_json['id'],
                    'user_email': saved_json['user_email'],
                    'json_data': json.loads(saved_json['json_data']),
                    'strategy_name': saved_json['strategy_name'],
                    'created_at': saved_json['created_at']
                })
            
            return create_success_response("Saved JSON data retrieved successfully", {
                "saved_jsons": result,
                "count": len(result)
            })
            
        except Exception as e:
            logger.error(f"Error getting saved JSON: {str(e)}")
            raise
    
    async def deploy_legacy(self, data: dict) -> Dict[str, Any]:
        """Legacy deploy endpoint"""
        try:
            print("Received data:", data)

            # Simulate processing
            strategy_name = data.get('strategyName')
            user_email = data.get('userEmail')
            webhook = data.get('webhook')
            reference_capital = data.get('referenceCapital')

            # Placeholder logic: validate and respond
            if not strategy_name or not webhook:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Missing required fields")

            # Simulate deployment success
            return {"message": f"Strategy '{strategy_name}' deployed successfully!"}
        except Exception as e:
            logger.error(f"Error in legacy deploy: {str(e)}")
            raise
