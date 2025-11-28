import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from Databases.app_data_db_connection import get_session, create_connection, init_database
from Databases.strategy_models import CustomStrategy

logger = logging.getLogger(__name__)

class CustomStrategyDatabase:
    def __init__(self, db_path: str = None):
        """
        Initialize CustomStrategyDatabase with PostgreSQL.
        db_path parameter is ignored (kept for compatibility) - always uses PostgreSQL ApplicationData database.
        """
        # Ensure database connection is established
        if not create_connection():
            logger.error("Failed to connect to PostgreSQL database")
            raise RuntimeError("Failed to connect to PostgreSQL database")
        
        # Initialize all tables (including custom_strategies)
        if not init_database():
            logger.error("Failed to initialize database tables")
            raise RuntimeError("Failed to initialize database tables")
        
        logger.info("CustomStrategyDatabase initialized with PostgreSQL")
    
    def init_database(self):
        """Initialize the custom_strategies table in PostgreSQL"""
        try:
            # Ensure connection is established
            if not create_connection():
                logger.error("Failed to connect to PostgreSQL database")
                return False
            
            # Initialize all tables (including custom_strategies)
            if not init_database():
                logger.error("Failed to initialize database tables")
                return False
            
            logger.info("Custom strategies table initialized successfully in PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error initializing custom strategies table: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_strategy(self, user_email: str, user_phone: str, 
                     strategy_description: str, analysis: Dict[str, Any]) -> int:
        """Save a custom strategy to the database"""
        session = None
        try:
            session = get_session()
            
            # Convert analysis to JSON string
            analysis_json = json.dumps(analysis)
            strategy_rating = analysis.get('strategy_rating', 0)
            
            # Create new strategy
            strategy = CustomStrategy(
                user_email=user_email,
                user_phone=user_phone,
                strategy_description=strategy_description,
                ai_analysis_json=analysis_json,
                strategy_rating=strategy_rating,
                status='pending'
            )
            
            session.add(strategy)
            session.commit()
            session.refresh(strategy)
            
            strategy_id = strategy.id
            logger.info(f"Custom strategy saved with ID: {strategy_id}")
            return strategy_id
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error saving custom strategy: {e}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            if session:
                session.close()
    
    def get_strategies_by_email(self, user_email: str) -> List[Dict[str, Any]]:
        """Get all custom strategies for a user by email"""
        session = None
        try:
            session = get_session()
            
            strategies = session.query(CustomStrategy).filter(
                CustomStrategy.user_email == user_email
            ).order_by(CustomStrategy.created_at.desc()).all()
            
            result = []
            for strategy in strategies:
                try:
                    analysis = json.loads(strategy.ai_analysis_json) if strategy.ai_analysis_json else {}
                    result.append({
                        "id": strategy.id,
                        "user_email": strategy.user_email,
                        "user_phone": strategy.user_phone,
                        "strategy_description": strategy.strategy_description,
                        "analysis": analysis,
                        "strategy_rating": strategy.strategy_rating,
                        "status": strategy.status,
                        "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
                        "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse analysis for strategy ID {strategy.id}: {e}")
                    continue
            
            return result
        except Exception as e:
            logger.error(f"Error getting strategies for user {user_email}: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if session:
                session.close()
    
    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific custom strategy by ID"""
        session = None
        try:
            session = get_session()
            
            strategy = session.query(CustomStrategy).filter(
                CustomStrategy.id == strategy_id
            ).first()
            
            if not strategy:
                return None
            
            try:
                analysis = json.loads(strategy.ai_analysis_json) if strategy.ai_analysis_json else {}
                return {
                    "id": strategy.id,
                    "user_email": strategy.user_email,
                    "user_phone": strategy.user_phone,
                    "strategy_description": strategy.strategy_description,
                    "analysis": analysis,
                    "strategy_rating": strategy.strategy_rating,
                    "status": strategy.status,
                    "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
                    "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None
                }
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse analysis for strategy ID {strategy.id}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error getting strategy {strategy_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if session:
                session.close()
    
    def update_strategy_status(self, strategy_id: int, status: str) -> bool:
        """Update strategy status"""
        session = None
        try:
            session = get_session()
            
            strategy = session.query(CustomStrategy).filter(
                CustomStrategy.id == strategy_id
            ).first()
            
            if not strategy:
                logger.warning(f"Strategy {strategy_id} not found")
                return False
            
            strategy.status = status
            session.commit()
            
            logger.info(f"Strategy {strategy_id} status updated to {status}")
            return True
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error updating strategy status: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if session:
                session.close()
