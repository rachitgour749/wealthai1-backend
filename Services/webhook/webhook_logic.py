"""
Webhook logic implementation for the Strategy Management Backend
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from .config import config
from .models import (
    StrategyCreate, StrategyUpdate, StrategyStatusUpdate,
    JsonGenerate, JsonSave, StrategyResponse, HealthResponse
)
from .utils import (
    validate_strategy_data, generate_json_data, send_webhook_notification,
    log_strategy_operation, create_error_response, create_success_response,
    sanitize_input
)
from Databases.app_data_db_connection import get_session, create_connection, init_database
from Databases.strategy_models import Strategy, SaveJson

# Get configuration
config_name = os.environ.get('FASTAPI_ENV', 'default')
app_config = config[config_name]

# Configure logging
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database with required tables"""
    try:
        # Ensure connection is established
        if not create_connection():
            logger.error("Failed to connect to PostgreSQL database")
            return False
        
        # Initialize all tables (including strategies and savejson)
        if not init_database():
            logger.error("Failed to initialize database tables")
            return False
        
        logger.info("Webhook database tables initialized successfully in PostgreSQL")
        return True
    except Exception as e:
        logger.error(f"Error initializing webhook database: {e}")
        import traceback
        traceback.print_exc()
        return False

class WebhookLogic:
    """Webhook business logic implementation"""
    
    def __init__(self):
        """Initialize webhook logic"""
        # Ensure database is initialized
        init_db()
    
    async def get_all_strategies(self) -> List[StrategyResponse]:
        """Get all strategies"""
        session = None
        try:
            session = get_session()
            
            strategies = session.query(Strategy).order_by(Strategy.created_at.desc()).all()
            
            result = []
            for strategy in strategies:
                client_ids = json.loads(strategy.client_ids) if strategy.client_ids else []
                capitals = json.loads(strategy.capitals) if strategy.capitals else []
                
                result.append(StrategyResponse(
                    id=strategy.id,
                    strategy_name=strategy.strategy_name,
                    user_email=strategy.user_email,
                    webhook=strategy.webhook,
                    reference_capital=strategy.reference_capital,
                    client_ids=client_ids,
                    capitals=capitals,
                    execution_date=strategy.execution_date,
                    created_at=strategy.created_at.isoformat() if strategy.created_at else None,
                    status=strategy.status
                ))
            
            return result
        except Exception as e:
            logger.error(f"Error getting strategies: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def create_strategy(self, strategy: StrategyCreate) -> Dict[str, Any]:
        """Create a new strategy"""
        session = None
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
            session = get_session()
            
            new_strategy = Strategy(
                strategy_name=strategy_name,
                user_email=user_email,
                webhook=webhook,
                reference_capital=reference_capital,
                client_ids=json.dumps(client_ids),
                capitals=json.dumps(capitals),
                execution_date=execution_date,
                status='active'
            )
            
            session.add(new_strategy)
            session.commit()
            session.refresh(new_strategy)
            
            strategy_id = new_strategy.id
            
            # Log strategy creation
            log_strategy_operation("created", strategy_id, user_email or "anonymous", f"Strategy: {strategy_name}")
            
            # Send webhook notification if webhook URL is provided
            webhook_sent = False
            if webhook:
                try:
                    json_data = generate_json_data(client_ids, capitals, "deploy")
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
            if session:
                session.rollback()
            logger.error(f"Error creating strategy: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def get_strategy_by_id(self, strategy_id: int) -> StrategyResponse:
        """Get a specific strategy by ID"""
        session = None
        try:
            session = get_session()
            
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            client_ids = json.loads(strategy.client_ids) if strategy.client_ids else []
            capitals = json.loads(strategy.capitals) if strategy.capitals else []
            
            return StrategyResponse(
                id=strategy.id,
                strategy_name=strategy.strategy_name,
                user_email=strategy.user_email,
                webhook=strategy.webhook,
                reference_capital=strategy.reference_capital,
                client_ids=client_ids,
                capitals=capitals,
                execution_date=strategy.execution_date,
                created_at=strategy.created_at.isoformat() if strategy.created_at else None,
                status=strategy.status
            )
        except Exception as e:
            logger.error(f"Error getting strategy: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def update_strategy(self, strategy_id: int, strategy_update: StrategyUpdate) -> Dict[str, Any]:
        """Update a specific strategy"""
        session = None
        try:
            session = get_session()
            
            # Check if strategy exists
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Update fields
            if strategy_update.strategyName is not None:
                strategy.strategy_name = strategy_update.strategyName
            if strategy_update.userEmail is not None:
                strategy.user_email = strategy_update.userEmail
            if strategy_update.webhook is not None:
                strategy.webhook = strategy_update.webhook
            if strategy_update.referenceCapital is not None:
                strategy.reference_capital = strategy_update.referenceCapital
            if strategy_update.clientIds is not None:
                strategy.client_ids = json.dumps([client.dict() for client in strategy_update.clientIds])
            if strategy_update.capitals is not None:
                strategy.capitals = json.dumps([capital.dict() for capital in strategy_update.capitals])
            
            session.commit()
            
            return {"message": "Strategy updated successfully"}
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error updating strategy: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def delete_strategy(self, strategy_id: int) -> Dict[str, Any]:
        """Delete a specific strategy"""
        session = None
        try:
            session = get_session()
            
            # Check if strategy exists
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Delete strategy
            session.delete(strategy)
            session.commit()
            
            return {"message": "Strategy deleted successfully"}
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error deleting strategy: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def update_strategy_status(self, strategy_id: int, status_update: StrategyStatusUpdate) -> Dict[str, Any]:
        """Update strategy status (active/inactive)"""
        session = None
        try:
            if status_update.status not in ['active', 'inactive']:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Status must be 'active' or 'inactive'")
            
            session = get_session()
            
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            strategy.status = status_update.status
            session.commit()
            
            return {"message": f"Strategy status updated to {status_update.status}"}
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error updating strategy status: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def health_check(self) -> HealthResponse:
        """Health check endpoint"""
        session = None
        try:
            # Test database connection
            from sqlalchemy import text
            session = get_session()
            session.execute(text("SELECT 1"))
            database_status = "connected"
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            database_status = "disconnected"
        finally:
            if session:
                session.close()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            database=database_status,
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
            generated_json = generate_json_data(client_ids, capitals, "deploy")
            
            logger.info(f"Generated JSON data for {len(client_ids)} clients")
            
            return create_success_response("JSON data generated successfully", {
                "json_data": generated_json,
                "client_count": len(client_ids)
            })
            
        except Exception as e:
            logger.error(f"Error generating JSON: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    async def trigger_webhook(self, strategy_id: int) -> Dict[str, Any]:
        """Trigger webhook notification for a specific strategy"""
        session = None
        try:
            session = get_session()
            
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Generate JSON data for webhook
            client_ids = json.loads(strategy.client_ids) if strategy.client_ids else []
            capitals = json.loads(strategy.capitals) if strategy.capitals else []
            json_data = generate_json_data(client_ids, capitals, "deploy")
            
            # Add strategy metadata
            webhook_data = {
                "strategy_id": strategy_id,
                "strategy_name": strategy.strategy_name,
                "user_email": strategy.user_email,
                "execution_date": strategy.execution_date,
                "trading_data": json_data
            }
            
            # Send webhook notification
            webhook_sent = send_webhook_notification(
                strategy.webhook, 
                webhook_data, 
                app_config.MAX_RETRIES
            )
            
            if webhook_sent:
                log_strategy_operation("webhook_triggered", strategy_id, strategy.user_email, {})
                return create_success_response("Webhook notification sent successfully")
            else:
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail="Failed to send webhook notification")
                
        except Exception as e:
            logger.error(f"Error triggering webhook: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def get_strategy_json(self, strategy_id: int) -> Dict[str, Any]:
        """Get JSON data for a specific strategy"""
        session = None
        try:
            session = get_session()
            
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            
            if not strategy:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Generate JSON data
            client_ids = json.loads(strategy.client_ids) if strategy.client_ids else []
            capitals = json.loads(strategy.capitals) if strategy.capitals else []
            json_data = generate_json_data(client_ids, capitals, "deploy")
            
            return create_success_response("JSON data retrieved successfully", {
                "strategy_id": strategy_id,
                "strategy_name": strategy.strategy_name,
                "json_data": json_data
            })
            
        except Exception as e:
            logger.error(f"Error getting strategy JSON: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def save_json_data(self, json_save: JsonSave) -> Dict[str, Any]:
        """Save JSON data for a user"""
        session = None
        try:
            session = get_session()
            
            # Get current execution date and time in local time with readable format
            current_time = datetime.now()
            execution_date = current_time.strftime("%B %d, %Y")
            execution_time = current_time.strftime("%I:%M:%S %p")
            full_timestamp = current_time.strftime("%B %d, %Y at %I:%M:%S %p")
            iso_timestamp = current_time.isoformat()  # JavaScript-compatible format
            
            save_data = {
                'user_email': json_save.user_email,
                'json_data': json_save.json_data,
                'strategy_name': json_save.strategy_name,
                'execution_date': execution_date,
                'execution_time': execution_time,
                'full_timestamp': full_timestamp,
                'iso_timestamp': iso_timestamp
            }
            
            new_save = SaveJson(
                user_email=json_save.user_email,
                json_data=json.dumps(save_data),
                strategy_name=json_save.strategy_name
            )
            
            session.add(new_save)
            session.commit()
            session.refresh(new_save)
            
            saved_id = new_save.id
            
            logger.info(f"JSON data saved for user {json_save.user_email} with ID {saved_id}")
            
            success_response = create_success_response("JSON data saved successfully", {
                "saved_id": saved_id,
                "user_email": json_save.user_email,
                "strategy_name": json_save.strategy_name
            })
            return {
                "message": success_response.message,
                "data": success_response.data,
                "timestamp": success_response.timestamp.isoformat()
            }
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error saving JSON: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def deploy_strategy(self, deploy_request) -> Dict[str, Any]:
        """Deploy strategy - generates JSON data and saves it to PostgreSQL"""
        session = None
        try:
            # First, generate the JSON data
            from .utils import generate_json_data
            
            if len(deploy_request.client_ids) != len(deploy_request.capitals):
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Number of client IDs must match number of capital values")
            
            # Generate JSON data
            generated_json = generate_json_data(deploy_request.client_ids, deploy_request.capitals, "deploy")
            
            session = get_session()
            
            # Get current execution date and time in local time with readable format
            current_time = datetime.now()
            execution_date = current_time.strftime("%B %d, %Y")
            execution_time = current_time.strftime("%I:%M:%S %p")
            full_timestamp = current_time.strftime("%B %d, %Y at %I:%M:%S %p")
            iso_timestamp = current_time.isoformat()  # JavaScript-compatible format
            
            # Prepare the data to save
            save_data = {
                'user_email': deploy_request.user_email,
                'json_data': generated_json,
                'strategy_name': deploy_request.strategy_name,
                'execution_date': execution_date,
                'execution_time': execution_time,
                'full_timestamp': full_timestamp,
                'iso_timestamp': iso_timestamp,
                'client_ids': deploy_request.client_ids,
                'capitals': deploy_request.capitals
            }
            
            # Insert into savejson table
            new_save = SaveJson(
                user_email=deploy_request.user_email,
                json_data=json.dumps(save_data),
                strategy_name=deploy_request.strategy_name
            )
            
            session.add(new_save)
            session.commit()
            session.refresh(new_save)
            
            saved_id = new_save.id
            
            logger.info(f"Strategy deployed for user {deploy_request.user_email} with ID {saved_id}")
            
            success_response = create_success_response("Strategy deployed successfully", {
                "saved_id": saved_id,
                "user_email": deploy_request.user_email,
                "strategy_name": deploy_request.strategy_name,
                "generated_json": generated_json,
                "client_count": len(deploy_request.client_ids)
            })
            return {
                "message": success_response.message,
                "data": success_response.data,
                "timestamp": success_response.timestamp.isoformat()
            }
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error deploying strategy: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def get_saved_json_data(self, user_email: str, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get saved JSON data for a user, optionally filtered by strategy name"""
        session = None
        try:
            session = get_session()
            
            query = session.query(SaveJson).filter(SaveJson.user_email == user_email)
            
            if strategy_name:
                query = query.filter(SaveJson.strategy_name == strategy_name)
            
            saved_jsons = query.order_by(SaveJson.id.desc()).all()
            
            result = []
            for saved_json in saved_jsons:
                json_data = json.loads(saved_json.json_data) if saved_json.json_data else {}
                
                result.append({
                    'id': saved_json.id,
                    'user_email': saved_json.user_email,
                    'json_data': json_data.get('json_data'),
                    'strategy_name': saved_json.strategy_name,
                    'execution_date': json_data.get('execution_date'),
                    'execution_time': json_data.get('execution_time'),
                    'full_timestamp': json_data.get('full_timestamp'),
                    'iso_timestamp': json_data.get('iso_timestamp'),
                    'created_at': saved_json.created_at.isoformat() if saved_json.created_at else None
                })
            
            success_response = create_success_response("Saved JSON data retrieved successfully", {
                "saved_jsons": result,
                "count": len(result)
            })
            return {
                "message": success_response.message,
                "data": success_response.data,
                "timestamp": success_response.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting saved JSON: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
    async def delete_saved_json_data(self, json_id: int) -> Dict[str, Any]:
        """Delete a specific saved JSON entry by ID"""
        session = None
        try:
            session = get_session()
            
            # Check if the JSON entry exists
            saved_json = session.query(SaveJson).filter(SaveJson.id == json_id).first()
            
            if not saved_json:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="JSON entry not found")
            
            # Delete the JSON entry
            session.delete(saved_json)
            session.commit()
            
            logger.info(f"JSON entry with ID {json_id} deleted successfully")
            
            success_response = create_success_response("JSON entry deleted successfully", {
                "deleted_id": json_id
            })
            return {
                "message": success_response.message,
                "data": success_response.data,
                "timestamp": success_response.timestamp.isoformat()
            }
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error deleting saved JSON: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()

    async def delete_saved_json_data_any(self, identifier: str) -> Dict[str, Any]:
        """Delete a saved JSON entry by numeric id or composite identifier.

        The identifier can be either:
        - a numeric userid (e.g., "5")
        - a composite key in the format "{iso_timestamp}_{user_email}_{strategy_name}"
        """
        session = None
        try:
            # Try numeric id first
            try:
                numeric_id = int(identifier)
            except ValueError:
                numeric_id = None
            if numeric_id is not None:
                return await self.delete_saved_json_data(numeric_id)

            # Otherwise, attempt composite match
            parts = identifier.split('_', 2)
            if len(parts) != 3:
                from fastapi import HTTPException
                raise HTTPException(status_code=422, detail="Invalid identifier format")
            iso_timestamp, user_email, strategy_name = parts

            session = get_session()
            
            saved_jsons = session.query(SaveJson).filter(
                SaveJson.user_email == user_email,
                SaveJson.strategy_name == strategy_name
            ).all()
            
            target_id = None
            for saved_json in saved_jsons:
                try:
                    payload = json.loads(saved_json.json_data) if saved_json.json_data else {}
                except Exception:
                    continue
                if payload.get('iso_timestamp') == iso_timestamp:
                    target_id = saved_json.id
                    break
            
            if target_id is None:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="JSON entry not found")

            saved_json = session.query(SaveJson).filter(SaveJson.id == target_id).first()
            if saved_json:
                session.delete(saved_json)
                session.commit()

            logger.info(f"JSON entry with composite identifier {identifier} deleted successfully (id={target_id})")

            success_response = create_success_response("JSON entry deleted successfully", {
                "deleted_id": target_id
            })
            return {
                "message": success_response.message,
                "data": success_response.data,
                "timestamp": success_response.timestamp.isoformat()
            }
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error deleting saved JSON (any): {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if session:
                session.close()
    
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
            import traceback
            traceback.print_exc()
            raise
