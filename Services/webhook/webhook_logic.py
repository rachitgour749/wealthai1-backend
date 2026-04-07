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
from Databases.webhook_models import WebhookConf, WebhookKey, RAConfig, WebhookExecutionLog

from Services.portfolio.price_service import PriceService
from helpers.broker_session_manager import get_broker_session, get_full_broker_session, save_broker_session
from Services.broker_manager import login_broker, dispatch_place_order
from Services.notification_service import WebhookNotifier
from Services.subscription.subscription_models import ProductManager
from Services.portfolio.portfolio_models import PortfolioTrade
from .models import TradeExecuteRequest, TradeExecuteResponse, UserExecutionDetail
from helpers.order_utils import generate_random_mac, calculate_limit_price, validate_user_ip_creds
from Databases.broker_models import BrokerSession

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
    
    async def create_webhook(self, request: WebhookCreateRequest) -> WebhookCreateResponse:
        """Create a new webhook configuration and managed keys"""
        session = None
        try:
            session = get_session()
            
            # 1. Validation: Prevent duplicates/updates ONLY for RA (Strict One-Per-Strategy Rule)
            if request.source == 'RA':
                actual_ra_code = request.ra_code or request.RA_code
                existing = session.query(WebhookConf).filter(
                    WebhookConf.user_id == request.user_id,
                    WebhookConf.ra_code == actual_ra_code,
                    WebhookConf.strategy_type == request.strategy_type,
                    WebhookConf.status == "active"
                ).first()
                
                if existing:
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=400, 
                        detail="webhook is already exist on this strategy_type on this RA"
                    )

            # 2. Generate run_id in format EXT_DDMMYYYYUUU
            current_date = datetime.now().strftime("%d%m%Y")
            uuu = f"{random.randint(0, 999):03d}"
            run_id = f"EXT_{current_date}{uuu}"

            # 1. Create entry in webhook_conf
            new_conf = WebhookConf(
                user_id=request.user_id,
                strategy_type=request.strategy_type,
                run_id=run_id,
                client_info=json.dumps(request.client_info),
                status="active",
                category=request.category,
                source=request.source,
                ra_code=request.ra_code or request.RA_code,
                name=request.name
            )
            session.add(new_conf)
            
            secret_key = None
            
            # 2. Logic based on source
            if request.source == 'INDIVIDUAL':
                # Create a new entry in webhook_key
                secret_key = f"whk_{secrets.token_hex(16)}"
                new_key = WebhookKey(
                    user_id=request.user_id,
                    secret_key=secret_key,
                    source=request.source
                )
                session.add(new_key)
                
            elif request.source == 'RA':
                # Verify RA existence and strategy match
                ra_code = request.ra_code or request.RA_code
                ra_entry = session.query(RAConfig).filter(
                    RAConfig.ra_code == ra_code,
                    RAConfig.strategy_type == request.strategy_type,
                    RAConfig.is_active == True
                ).first()
                
                if not ra_entry:
                    from fastapi import HTTPException
                    logger.warning(f"RA verification failed: {ra_code} for {request.strategy_type}")
                    raise HTTPException(status_code=404, detail=f"Invalid RA Code or Strategy mismatch for {ra_code}")
                
                # IMPORTANT: For RA source, do NOT return the secret_key to the user
                secret_key = None 
            
            session.commit()
            
            return WebhookCreateResponse(
                status="success",
                source=request.source,
                run_id=run_id,
                secret_key=secret_key,
                message="Webhook created successfully"
            )
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error creating webhook: {str(e)}")
            raise
        finally:
            if session:
                session.close()

    def _save_execution_log(self, session, email, signal, status, message, ra_email=None):
        """Helper to save execution log entry"""
        try:
            log = WebhookExecutionLog(
                user_email=email,
                strategy_name=signal.strategy_type,
                symbol=signal.symbol,
                side=signal.order_side.upper(),
                status=status,
                message=message,
                ra_email=ra_email
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to save execution log: {e}")
            if session: session.rollback()

    async def get_user_webhooks(self, user_id: str) -> List[WebhookDetailResponse]:
        """Fetch all webhooks for a given user"""
        session = get_session()
        try:
            # configs = session.query(WebhookConf).filter(WebhookConf.user_id == user_id).all()
            # To handle multiple records correctly, let's just get all
            configs = session.query(WebhookConf).filter(WebhookConf.user_id == user_id).all()
            
            # Fetch secret keys for INDIVIDUAL source
            individual_key = session.query(WebhookKey).filter(
                WebhookKey.user_id == user_id,
                WebhookKey.source == 'INDIVIDUAL'
            ).first()
            
            results = []
            for conf in configs:
                secret_key = None
                if conf.source == 'INDIVIDUAL' and individual_key:
                    secret_key = individual_key.secret_key
                
                results.append(WebhookDetailResponse(
                    run_id=conf.run_id,
                    name=conf.name,
                    strategy_type=conf.strategy_type,
                    client_info=json.loads(conf.client_info) if conf.client_info else {},
                    status=conf.status,
                    category=conf.category or "UNKNOWN",
                    source=conf.source or "UNKNOWN",
                    ra_code=conf.ra_code,
                    secret_key=secret_key
                ))
            return results
        finally:
            if session:
                session.close()

    async def update_webhook_status(self, run_id: str, status: str) -> bool:
        """Toggle active/inactive status using run_id"""
        session = get_session()
        try:
            conf = session.query(WebhookConf).filter(WebhookConf.run_id == run_id).first()
            if not conf:
                return False
            
            conf.status = status
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating status for {run_id}: {e}")
            if session: session.rollback()
            return False
        finally:
            if session:
                session.close()

    async def delete_webhook(self, run_id: str) -> bool:
        """Delete a webhook configuration using run_id"""
        session = get_session()
        try:
            conf = session.query(WebhookConf).filter(WebhookConf.run_id == run_id).first()
            if not conf:
                return False
            
            session.delete(conf)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting webhook {run_id}: {e}")
            if session: session.rollback()
            return False
        finally:
            if session:
                session.close()

    async def execute_trade_individual(self, request: TradeExecuteIndividualRequest, secret_header: str) -> TradeExecuteResponse:
        """Execute trade for an individual user based on run_id and secret key"""
        session = get_session()
        try:
            # 1. Authenticate via run_id and secret key
            # First find the config to get the user_id
            conf = session.query(WebhookConf).filter(
                WebhookConf.run_id == request.run_id,
                WebhookConf.status == "active"
            ).first()
            
            if not conf:
                logger.warning(f"Individual Trade Failed: run_id {request.run_id} not found or inactive")
                raise HTTPException(status_code=404, detail="Active configuration not found for this run_id")
            
            # Verify the authorized_email matches the config owner
            if conf.user_id != request.authorized_email:
                logger.warning(f"Individual Trade Failed: email mismatch {request.authorized_email} vs {conf.user_id}")
                raise HTTPException(status_code=403, detail="Authorized email does not match configuration owner")

            # Verify secret key for this user
            key_entry = session.query(WebhookKey).filter(
                WebhookKey.user_id == conf.user_id,
                WebhookKey.secret_key == secret_header,
                WebhookKey.source == 'INDIVIDUAL'
            ).first()
            
            if not key_entry:
                logger.warning(f"Individual Trade Failed: Invalid secret key for user {conf.user_id}")
                raise HTTPException(status_code=401, detail="Invalid webhook secret key")

            # 2. Price Fetching (Shared)
            shared_price = None
            try:
                shared_price = PriceService.get_current_price(request.symbol, request.exchnge)
            except Exception as e:
                logger.error(f"Failed to fetch shared price for INDIVIDUAL: {e}")

            # 3. Process the trade
            # Note: authorized_email is a string here, but we pass list to match response model
            detail = await self._process_single_user_trade(request.authorized_email, request, shared_price)
            
            executed_count = 1 if detail.status == 'executed' else 0
            failure_count = 1 if detail.status == 'error' else 0
            
            return TradeExecuteResponse(
                status="success",
                processed=1,
                executed=executed_count,
                failures=failure_count,
                details=[detail]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in individual trade execution: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            if session:
                session.close()

    async def execute_trade(self, request: TradeExecuteRequest, secret_header: str) -> TradeExecuteResponse:
        """Execute trade for all authorized emails in parallel"""
        session = get_session()
        try:
            # 1. RA Authentication
            ra_entry = session.query(RAConfig).filter(
                RAConfig.secret_key == secret_header,
                RAConfig.ra_code == request.ra_code,
                RAConfig.strategy_type == request.strategy_type,
                RAConfig.is_active == True
            ).first()

            if not ra_entry:
                from fastapi import HTTPException
                logger.warning(f"RA Auth failed for {request.ra_code} with secret {secret_header[:10]}...")
                raise HTTPException(status_code=401, detail="Invalid RA secret key or code/strategy mismatch")

            # 2. Batch Price Fetching (Optimization)
            shared_price = None
            if any(email in request.authorized_email for email in request.authorized_email): # Just a check
                # We can fetch the price once if it's an EQUITY trade
                # We'll check the first user's configuration to see if it's EQUITY
                # Actually, let's just fetch it if symbol is provided
                try:
                    shared_price = PriceService.get_current_price(request.symbol, request.exchnge)
                    logger.info(f"Fetched shared price for {request.symbol}: {shared_price}")
                except Exception as e:
                    logger.error(f"Failed to fetch shared price: {e}")

            # 3. Parallel Processing for all authorized emails
            tasks = [self._process_single_user_trade(email, request, shared_price, ra_entry.ra_email) for email in request.authorized_email]
            details = await asyncio.gather(*tasks)

            executed_count = sum(1 for d in details if d.status == 'executed')
            failure_count = sum(1 for d in details if d.status == 'error')

            # 4. RA Summary Email (Background)
            try:
                notifier = WebhookNotifier()
                asyncio.create_task(asyncio.to_thread(
                    notifier.send_ra_summary,
                    ra_email=ra_entry.ra_email,
                    strategy_name=request.strategy_type,
                    symbol=request.symbol,
                    side=request.order_side,
                    details=[d.model_dump() for d in details]
                ))
            except Exception as e:
                logger.error(f"Failed to trigger RA summary email: {e}")

            return TradeExecuteResponse(
                status="success",
                processed=len(request.authorized_email),
                executed=executed_count,
                failures=failure_count,
                details=details
            )

        except Exception as e:
            logger.error(f"Execution engine failure: {e}")
            raise
        finally:
            if session:
                session.close()

    def _check_subscription(self, session, email: str) -> bool:
        """Check if user has an active subscription for product M"""
        try:
            now = datetime.now()
            sub = session.query(ProductManager).filter(
                ProductManager.user_email == email,
                ProductManager.product_code == 'M',
                ProductManager.subscription_end_date > now
            ).first()
            return sub is not None
        except Exception as e:
            logger.error(f"Error checking subscription for {email}: {e}")
            return False

    async def unified_execute_trade(self, request: UnifiedTradeExecuteRequest, secret_header: str) -> TradeExecuteResponse:
        """Unified trade execution point for both RA and INDIVIDUAL signals"""
        # Determine mode
        if request.ra_code and request.ra_code != 'none':
            # RA Mode
            # Convert Unified to TradeExecuteRequest
            ra_req = TradeExecuteRequest(
                ra_code=request.ra_code,
                strategy_type=request.strategy_type,
                symbol=request.symbol,
                exchnge=request.exchnge,
                order_side=request.order_side,
                authorized_email=request.authorized_email if isinstance(request.authorized_email, list) else [request.authorized_email]
            )
            return await self.execute_trade(ra_req, secret_header)
        elif request.run_id:
            # Individual Mode
            ind_req = TradeExecuteIndividualRequest(
                run_id=request.run_id,
                strategy_type=request.strategy_type,
                symbol=request.symbol,
                exchnge=request.exchnge,
                order_side=request.order_side,
                authorized_email=request.authorized_email if isinstance(request.authorized_email, str) else request.authorized_email[0]
            )
            return await self.execute_trade_individual(ind_req, secret_header)
        else:
            raise HTTPException(status_code=400, detail="Either ra_code or run_id must be provided")

    async def _process_single_user_trade(self, email: str, signal: Union[TradeExecuteRequest, TradeExecuteIndividualRequest, UnifiedTradeExecuteRequest], shared_price: float = None, ra_email: str = None) -> UserExecutionDetail:
        """Helper to process a single user trade execution"""
        session = get_session()
        notifier = WebhookNotifier()
        status = "skipped"
        message = ""
        order_id = None
        
        # Helper for background notification
        def notify_background(cl_email, strat, sym, side, stat, msg):
            asyncio.create_task(asyncio.to_thread(
                notifier.send_client_notification, cl_email, strat, sym, side, stat, msg
            ))

        try:
            # 0. Subscription Validation
            if not self._check_subscription(session, email):
                status = "error"
                message = "user subscription plan is end"
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)

            # A. Fetch Configuration
            if isinstance(signal, TradeExecuteIndividualRequest):
                conf = session.query(WebhookConf).filter(
                    WebhookConf.run_id == signal.run_id,
                    WebhookConf.status == "active"
                ).first()
            else:
                conf = session.query(WebhookConf).filter(
                    WebhookConf.user_id == email,
                    # Optional: Include signal.name if you want to support specific RA alerts per name
                    # WebhookConf.name == signal.name, 
                    WebhookConf.ra_code == signal.ra_code,
                    WebhookConf.strategy_type == signal.strategy_type,
                    WebhookConf.status == "active"
                ).order_by(WebhookConf.created_at.desc()).first()

            if not conf:
                status = "error"
                message = f"No active configuration found for {signal.strategy_type} (RA:{signal.ra_code})"
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)

            # B. Broker Matching Check
            try:
                client_info = json.loads(conf.client_info) if conf.client_info else {}
                # The keys in client_info are client_ids
                expected_client_ids = list(client_info.keys())
                if not expected_client_ids:
                    raise Exception("No client_id found in client_info")
                
                target_client_id = expected_client_ids[0]
                client_value = client_info[target_client_id] 
            except Exception as e:
                status = "error"
                message = f"Invalid client_info format: {e}"
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)

            full_details = get_full_broker_session(email)
            stored_client_id = full_details.get("client_id") if full_details else None
            
            logger.info(f"Checking broker for {email}: Target={target_client_id}, Stored={stored_client_id}")
            
            # --- IP CREDENTIAL VALIDATION ---
            client_record = session.query(BrokerSession).filter_by(user_email=email).first()
            is_valid_ip, ip_err = validate_user_ip_creds(client_record)
            if not is_valid_ip:
                status = "error"
                message = f"IP Validation Failed: {ip_err}"
                logger.warning(f"Trade Rejected for {email}: {message}")
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)
            # -------------------------------

            if not full_details or stored_client_id != target_client_id:
                status = "error"
                message = f"broker is different (Target:{target_client_id}, Stored:{stored_client_id}, RA:{signal.ra_code})"
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)
            
            broker_name = full_details["broker_name"]
            api_key = full_details["api_key"]
            broker_credentials = full_details.get("broker_credentials")
            # We don't need access_token yet because Step D will fetch a fresh one.

            # C. Quantity Calculation
            try:
                if conf.category == 'EQUITY':
                    current_price = shared_price or PriceService.get_current_price(signal.symbol, signal.exchnge)
                    if not current_price or current_price <= 0:
                        raise Exception(f"Could not fetch price for {signal.symbol}")
                    qty = int(float(client_value) / current_price)
                else: # FNO
                    qty = int(client_value)
                
                if qty <= 0:
                    raise Exception(f"Calculated quantity is 0 for {client_value}")
            except Exception as e:
                status = "error"
                message = f"Quantity calculation failed: {e}"
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)

            # D. Smart Relogin
            try:
                if not full_details or not full_details.get("broker_credentials"):
                    raise Exception("No stored credentials for relogin")
                
                # Smart Check: If token is valid for > 30 mins, skip relogin
                # token_expire is a string ISO format or None in full_details
                is_token_valid = False
                if full_details.get("token_expire") and full_details.get("access_token"):
                    try:
                        expire_time = datetime.strptime(full_details["token_expire"], "%Y-%m-%d %H:%M:%S")
                        if (expire_time - datetime.utcnow()).total_seconds() > 1800: # 30 mins
                            is_token_valid = True
                            logger.info(f"Skipping relogin for {email}, token valid until {expire_time}")
                    except: pass

                if not is_token_valid:
                    creds = full_details["broker_credentials"]
                    login_result = login_broker(broker_name, creds)
                    
                    if login_result.get("status") == "error":
                        raise Exception(f"Relogin failed: {login_result.get('message')}")
                    
                    # Save fresh session
                    save_broker_session(email, broker_name, target_client_id, login_result, api_key, credentials=creds)
                    
                    # Update token for dispatch
                    access_token = login_result.get('access_token') or login_result.get('data', {}).get('access_token')
                else:
                    access_token = full_details["access_token"]
                    
            except Exception as e:
                status = "error"
                message = f"Broker Relogin failed: {e}"
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)

            # E. Order Dispatch
            # --- CALCULATE LIMIT PRICE (SEBI COMPLIANCE) ---
            limit_price, price_err = calculate_limit_price(signal.symbol, signal.exchnge, signal.order_side)
            if price_err:
                status = "error"
                message = f"Limit Price Calculation Failed: {price_err}"
                notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
                self._save_execution_log(session, email, signal, status, message, ra_email)
                return UserExecutionDetail(email=email, status=status, message=message)
            
            order_data = {
                'symbol': signal.symbol,
                'exchange': signal.exchnge,
                'order_side': signal.order_side.upper(),
                'product_type': 'DELIVERY' if conf.category == 'EQUITY' else 'MARGIN',
                'order_type': 'LIMIT',
                'price': limit_price,
                'quantity': qty,
                'validity': 'IOC', # Immediate Or Cancel
                'variety': 'regular',
                'static_ip': client_record.static_ip,
                'mac_address': generate_random_mac()
            }
            
            credentials = {
                'api_key': api_key,
                'access_token': access_token,
                'client_id': target_client_id
            }
            
            if broker_credentials:
                try:
                    b_creds = json.loads(broker_credentials)
                    if b_creds.get('sid'): credentials['sid'] = b_creds['sid']
                except: pass

            res = dispatch_place_order(broker_name, credentials, order_data)
            
            if res.get("status") == "success":
                status = "executed"
                order_id = res.get("data", {}).get("order_id")
                message = "Order placed successfully"
                
                # F. Portfolio Logging
                try:
                    price = float(res.get("data", {}).get("average_price", 0.0))
                    if price == 0: price = PriceService.get_current_price(signal.symbol, signal.exchnge)
                    
                    executed_at = datetime.now()  # Exact execution timestamp — makes every trade unique
                    trade_record = PortfolioTrade(
                        user_email=email,
                        run_id=conf.run_id,
                        strategy_name=signal.strategy_type,
                        strategy_type=signal.strategy_type,
                        client_code=target_client_id,
                        trade_date=executed_at.date(),
                        executed_at=executed_at,
                        symbol=signal.symbol,
                        side=signal.order_side.upper(),
                        quantity=qty,
                        price=price
                    )
                    session.add(trade_record)
                    session.commit()
                except Exception as p_err:
                    logger.error(f"Portfolio logging failed for {email}: {p_err}")
            else:
                status = "error"
                message = res.get("message", "Order placement failed")

            # G. Notification (Background) and Log
            notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
            self._save_execution_log(session, email, signal, status, message, ra_email)
            
        except Exception as e:
            logger.error(f"Error processing user {email}: {e}")
            status = "error"
            message = str(e)
            notify_background(email, signal.strategy_type, signal.symbol, signal.order_side, status, message)
            self._save_execution_log(session, email, signal, status, message, ra_email)
        finally:
            if session:
                session.close()

        return UserExecutionDetail(email=email, status=status, message=message, order_id=order_id)

    # RA CRUD Methods
    def create_ra(self, request: RACreateRequest) -> RAResponse:
        """Create a new RA configuration with automatic code and key generation"""
        session = get_session()
        try:
            # 1. Check if ra_email already has an assigned ra_code
            existing_ra = session.query(RAConfig).filter(
                RAConfig.ra_email == request.ra_email
            ).first()
            
            ra_code = None
            if existing_ra:
                ra_code = existing_ra.ra_code
            else:
                # 2. Generate new ra_code (RA_1001, RA_1002, etc.)
                all_codes = session.query(RAConfig.ra_code).distinct().all()
                max_num = 1000 # Start from 1001
                for (code,) in all_codes:
                    if code and code.startswith("RA_"):
                        try:
                            num = int(code.split("_")[1])
                            if num > max_num:
                                max_num = num
                        except (IndexError, ValueError):
                            continue
                ra_code = f"RA_{max_num + 1}"

            # 3. Check if strategy is already configured for this RA
            existing_strat = session.query(RAConfig).filter(
                RAConfig.ra_code == ra_code,
                RAConfig.strategy_type == request.strategy_type
            ).first()
            
            if existing_strat:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="strategy is already configure in this ra code")

            # 4. Generate random secret key
            secret_key = f"ras_{secrets.token_hex(16)}"

            new_ra = RAConfig(
                ra_email=request.ra_email,
                ra_code=ra_code,
                strategy_type=request.strategy_type,
                secret_key=secret_key,
                is_active=True
            )
            session.add(new_ra)
            session.commit()
            session.refresh(new_ra)
            return RAResponse.model_validate(new_ra)
        finally:
            session.close()

    def get_ras(self) -> List[RAResponse]:
        """Get all RA configurations"""
        session = get_session()
        try:
            ras = session.query(RAConfig).all()
            return [RAResponse.model_validate(ra) for ra in ras]
        finally:
            session.close()

    def get_ra_strategies(self, ra_code: str) -> List[str]:
        """Get all unique strategy types for a specific RA code"""
        session = get_session()
        try:
            strategies = session.query(RAConfig.strategy_type).filter(
                RAConfig.ra_code == ra_code,
                RAConfig.is_active == True
            ).distinct().all()
            return [s[0] for s in strategies]
        finally:
            session.close()

    def get_ra(self, ra_code: str, strategy_type: str) -> RAResponse:
        """Get a specific RA configuration"""
        session = get_session()
        try:
            ra = session.query(RAConfig).filter(
                RAConfig.ra_code == ra_code,
                RAConfig.strategy_type == strategy_type
            ).first()
            if not ra:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="RA configuration not found")
            return RAResponse.model_validate(ra)
        finally:
            session.close()

    def update_ra(self, ra_code: str, strategy_type: str, request: RAUpdateRequest) -> RAResponse:
        """Update an RA configuration"""
        session = get_session()
        try:
            ra = session.query(RAConfig).filter(
                RAConfig.ra_code == ra_code,
                RAConfig.strategy_type == strategy_type
            ).first()
            if not ra:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="RA configuration not found")
            
            if request.secret_key is not None:
                ra.secret_key = request.secret_key
            if request.is_active is not None:
                ra.is_active = request.is_active
                
            session.commit()
            session.refresh(ra)
            return RAResponse.model_validate(ra)
        finally:
            session.close()

    def delete_ra(self, ra_code: str, strategy_type: str) -> bool:
        """Delete an RA configuration"""
        session = get_session()
        try:
            ra = session.query(RAConfig).filter(
                RAConfig.ra_code == ra_code,
                RAConfig.strategy_type == strategy_type
            ).first()
            if not ra:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="RA configuration not found")
            
            session.delete(ra)
            session.commit()
            return True
        finally:
            session.close()

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
