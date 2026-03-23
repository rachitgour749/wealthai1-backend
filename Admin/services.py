import logging
import importlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text

from Services.scheduler.config_utils import scheduler_config
from Services.subscription.database import subscription_manager
from Services.subscription.subscription_models import UserDetails, ProductManager
from Services.subscription.subscription_schemas import ProductCode, SubscriptionStatus, SubscriptionType
from Admin.schemas import (
    ClientListItem, ClientDetailResponse, ProductSummary,
    UpdateClientRequest, UpdatePlanRequest, UpdateSubscriptionDatesRequest,
    UpdateCreditsRequest, SystemTriggerRequest
)

logger = logging.getLogger(__name__)

class AdminService:
    def __init__(self):
        self.db_manager = subscription_manager

    def _get_strategy_metadata_by_type(self, strategy_type: str) -> Optional[Dict]:
        """Find strategy configuration by its strategy_type (strategy_filter.strategy_name)"""
        strategies = scheduler_config.config.get('strategies', {})
        for key, config in strategies.items():
            filter_config = config.get('strategy_filter', {})
            if filter_config.get('strategy_name') == strategy_type:
                return config
        return None

    def trigger_signal_generation(self, strategy_type: str) -> Dict[str, Any]:
        """Manually trigger signal generation for a strategy type"""
        config = self._get_strategy_metadata_by_type(strategy_type)
        if not config:
            return {"success": False, "message": f"No configuration found for strategy type: {strategy_type}"}
        
        if 'signal_generation' not in config:
            return {"success": False, "message": f"Signal generation not configured for: {strategy_type}"}

        try:
            module_path = config['generator_module']
            function_name = config['generator_function']
            
            logger.info(f"Manually triggering signal generator: {module_path}.{function_name}")
            
            module = importlib.import_module(module_path)
            generator_func = getattr(module, function_name)
            
            # Execute the generator
            # Most generators are synchronous or handle their own async loop if needed
            # We assume they can be called directly or are awaited if needed (though APScheduler usually handles sync)
            import inspect
            if inspect.iscoroutinefunction(generator_func):
                # If it's async, we might need an event loop, but APScheduler jobs are often sync wrappers
                return {"success": False, "message": "Async generators not directly supported via API yet without internal loop handling"}
            
            generator_func()
            return {"success": True, "message": f"Successfully triggered signal generation for {strategy_type}"}
        except Exception as e:
            logger.error(f"Failed to trigger signal generation for {strategy_type}: {e}")
            return {"success": False, "message": str(e)}

    def trigger_signal_execution(self, strategy_type: str) -> Dict[str, Any]:
        """Manually trigger signal execution for a strategy type"""
        config = self._get_strategy_metadata_by_type(strategy_type)
        if not config:
            return {"success": False, "message": f"No configuration found for strategy type: {strategy_type}"}
        
        if 'signal_execution' not in config:
            return {"success": False, "message": f"Signal execution not configured for: {strategy_type}"}

        try:
            module_path = config.get('executor_module', 'Services.scheduler.executors.signal_executor')
            function_name = config.get('executor_function', 'SignalExecutor().execute_signals')
            
            logger.info(f"Manually triggering signal executor: {module_path}.{function_name}")
            
            if '()' in function_name:
                class_name, method_name = function_name.split('().')
                module = importlib.import_module(module_path)
                executor_class = getattr(module, class_name)
                instance = executor_class()
                method = getattr(instance, method_name)
                method()
            else:
                module = importlib.import_module(module_path)
                exec_func = getattr(module, function_name)
                exec_func()
                
            return {"success": True, "message": f"Successfully triggered signal execution for {strategy_type}"}
        except Exception as e:
            logger.error(f"Failed to trigger signal execution for {strategy_type}: {e}")
            return {"success": False, "message": str(e)}

    def get_all_clients(self) -> List[ClientListItem]:
        """Fetch all users who are clients"""
        session = self.db_manager.get_session()
        try:
            users = session.query(UserDetails).filter(UserDetails.role == "CLIENT").all()
            return [
                ClientListItem(
                    user_email=u.user_email,
                    user_name=u.user_name,
                    phone_no=u.phone_no,
                    status=u.status,
                    role=u.role,
                    created_at=u.created_at
                ) for u in users
            ]
        finally:
            session.close()

    def get_client_details(self, email: str) -> Optional[ClientDetailResponse]:
        """Fetch full details for a client including products"""
        session = self.db_manager.get_session()
        try:
            user = session.query(UserDetails).filter(UserDetails.user_email == email.lower()).first()
            if not user:
                return None
            
            products = session.query(ProductManager).filter(ProductManager.user_email == email.lower()).all()
            
            product_summaries = [
                ProductSummary(
                    product_code=p.product_code,
                    subscription_type=p.subscription_type.value if hasattr(p.subscription_type, 'value') else str(p.subscription_type),
                    status=p.status.value if hasattr(p.status, 'value') else str(p.status),
                    subscription_start_date=p.subscription_start_date,
                    subscription_end_date=p.subscription_end_date,
                    plan_name=p.plan_name,
                    remaining_tokens=p.remaining_token
                ) for p in products
            ]
            
            return ClientDetailResponse(
                user_email=user.user_email,
                user_name=user.user_name,
                phone_no=user.phone_no,
                status=user.status,
                role=user.role,
                created_at=user.created_at,
                updated_at=user.updated_at,
                products=product_summaries
            )
        finally:
            session.close()

    def update_client_data(self, email: str, data: UpdateClientRequest) -> bool:
        """Update basic client information"""
        session = self.db_manager.get_session()
        try:
            user = session.query(UserDetails).filter(UserDetails.user_email == email.lower()).first()
            if not user:
                return False
            
            if data.user_name is not None:
                user.user_name = data.user_name
            if data.phone_no is not None:
                user.phone_no = data.phone_no
            if data.status is not None:
                user.status = data.status.upper()
            if data.role is not None:
                user.role = data.role.upper()
            
            user.updated_at = datetime.now(timezone.utc)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating client data: {e}")
            return False
        finally:
            session.close()

    def update_client_plan(self, email: str, plan_code: int) -> Dict[str, Any]:
        """Update client's plan using existing activation logic"""
        # We can reuse activate_paid_subscription or create a more direct one
        # but the request asks for "update client complete plan".
        # Let's use the logic that maps plan to products.
        try:
            # Need plan name from plan_code
            plan_info = self.db_manager.validate_plan_code(plan_code)
            if not plan_info["valid"]:
                return {"success": False, "message": plan_info["message"]}
            
            # Using activate_paid_subscription logic
            result = self.db_manager.activate_paid_subscription(
                user_email=email,
                plan_name=plan_info["plan_name"],
                subscription_id=f"ADMIN_UPDATE_{int(datetime.now().timestamp())}"
            )
            return result
        except Exception as e:
            logger.error(f"Error updating client plan: {e}")
            return {"success": False, "message": str(e)}

    def update_client_dates(self, email: str, date_info: UpdateSubscriptionDatesRequest) -> bool:
        """Update subscription dates for a specific product"""
        session = self.db_manager.get_session()
        try:
            product = session.query(ProductManager).filter(
                ProductManager.user_email == email.lower(),
                ProductManager.product_code == date_info.product_code.value
            ).first()
            
            if not product:
                return False
            
            if date_info.subscription_start_date:
                product.subscription_start_date = date_info.subscription_start_date
            if date_info.subscription_end_date:
                product.subscription_end_date = date_info.subscription_end_date
            
            product.updated_at = datetime.now(timezone.utc)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating client dates: {e}")
            return False
        finally:
            session.close()

    def update_user_credits(self, email: str, credit_info: UpdateCreditsRequest) -> Dict[str, Any]:
        """Increase or decrease user credits (tokens)"""
        session = self.db_manager.get_session()
        try:
            product = session.query(ProductManager).filter(
                ProductManager.user_email == email.lower(),
                ProductManager.product_code == credit_info.product_code.value
            ).first()
            
            if not product:
                return {"success": False, "message": f"Product {credit_info.product_code} not found for user"}
            
            current_credits = product.remaining_token or 0
            if credit_info.operation == "increase":
                new_credits = current_credits + credit_info.amount
                product.total_token = (product.total_token or 0) + credit_info.amount
            else:
                new_credits = max(0, current_credits - credit_info.amount)
                # used_tokens shouldn't necessarily change if we are just decreasing allocation, 
                # but let's just adjust remaining.
                
            product.remaining_token = new_credits
            product.updated_at = datetime.now(timezone.utc)
            session.commit()
            
            return {
                "success": True, 
                "message": f"Credits {credit_info.operation}d successfully",
                "data": {
                    "previous_credits": current_credits,
                    "new_credits": new_credits,
                    "product": credit_info.product_code
                }
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating user credits: {e}")
            return {"success": False, "message": str(e)}
        finally:
            session.close()

admin_service = AdminService()
