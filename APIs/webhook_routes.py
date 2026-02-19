from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging

from Databases.app_data_db_connection import get_db
from Services.strategy_manager.webhook_manager import create_webhook_strategy

router = APIRouter()
logger = logging.getLogger(__name__)

class CreateWebhookRequest(BaseModel):
    user_id: str
    strategy_type: str = "External_Strategy"
    strategy_name: str
    reference_capital: float
    client_info: Dict[str, float]
    webhook: Optional[str] = None # Maps to webhook_url in DB (Source Identifier)

    class Config:
        extra = "allow"

@router.post(
    "/create_webhook",
    summary="Create External Strategy Configuration",
    description="""
    Create a new configuration for an external strategy (e.g., TradingView).
    - Generates a unique `run_id` with 'EXT' prefix.
    - Sets status to 'running' and source to 'other'.
    - Returns the `run_id` for use in external webhook calls.
    """,
    responses={
        200: {"description": "Strategy created successfully"},
        500: {"description": "Internal server error"}
    }
)
async def create_webhook_route(
    request: CreateWebhookRequest,
    db: Session = Depends(get_db)
):
    logger.info(f"Received create_webhook request from {request.user_id} for {request.strategy_name}")
    try:
        result = create_webhook_strategy(
            user_id=request.user_id,
            strategy_type=request.strategy_type,
            strategy_name=request.strategy_name,
            reference_capital=request.reference_capital,
            client_info=request.client_info,
            webhook=request.webhook,
            db=db
        )
        return result
        
    except Exception as e:
        logger.error(f"API Error creating webhook strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRADE EXECUTION WEBHOOK
# ============================================================================

from Services.subscription.subscription_models import ProductManager
from Services.subscription.subscription_schemas import SubscriptionStatus
from Services.broker_manager import dispatch_place_order
from helpers.broker_session_manager import get_broker_session
from Services.portfolio.portfolio_models import PortfolioTrade
from Services.portfolio.utils import calculate_brokerage, calculate_taxes
from Services.portfolio.price_service import PriceService
from Services.strategy_manager.models import SavedInstance
from datetime import datetime, timezone

class TradeExecuteRequest(BaseModel):
    signal_id: str
    strategy_name: str
    timestamp: str
    symbol: str
    exchange: str
    order_side: str
    authorized_emails: List[str]
    clients: Optional[Dict[str, str]] = None # "clients": {"ClientId": "Quantity"}
    
    class Config:
        extra = "allow"

@router.post(
    "/wealthai1.in/trade_execute",
    summary="Execute Trade for Authorized Users",
    description="""
    Execute a trade signal for a list of authorized users.
    - Checks subscription status (ACTIVE/TRIAL) and validity.
    - Checks if strategy is deployed ('running') and external ('other').
    - Placing orders via Broker API for authorized users.
    - Records successful trades in Portfolio.
    """,
    responses={
        200: {"description": "Execution processed"},
        400: {"description": "Validation failed"}
    }
)
async def trade_execute_webhook(
    request: TradeExecuteRequest,
    db: Session = Depends(get_db)
):
    logger.info(f"Received trade execution webhook: {request.signal_id} for {request.symbol}")
    
    results = {
        "processed": 0,
        "authorized": 0,
        "executed": 0,
        "failed": 0,
        "details": []
    }
    
    # 1. Parse Timestamp
    try:
        signal_time = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
    except:
        signal_time = datetime.utcnow()
        
    # 2. Iterate Authorized Emails
    for email in request.authorized_emails:
        results["processed"] += 1
        email_status = {"email": email, "status": "skipped", "message": ""}
        
        try:
            # A. Check Subscription
            # A. Check Subscription
            # We check the ProductManager table for this user specifically for product_code='M'
            # Get the LATEST subscription end date
            sub_record = db.query(ProductManager).filter(
                ProductManager.user_email == email,
                ProductManager.product_code == 'M'
            ).order_by(ProductManager.subscription_end_date.desc()).first()
            
            is_authorized = False
            
            if sub_record:
                # 2. Check Expiry
                # Use timezone-aware current time (UTC)
                current_time = datetime.now(timezone.utc)
                
                # DEBUG INFO in MESSAGE
                debug_msg = f" [End: {sub_record.subscription_end_date}, Now: {current_time}]"
                
                if sub_record.subscription_end_date and sub_record.subscription_end_date > current_time:
                   # 3. Check Status
                   if sub_record.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]:
                       is_authorized = True
                       email_status["message"] = f"Authorized. {debug_msg}" # Temporary Debug
                   else:
                       email_status["message"] = f"Subscription status is {sub_record.status}" + debug_msg
                else:
                     email_status["message"] = "Subscription expired" + debug_msg
            else:
                 email_status["message"] = "No 'M' series subscription found"
            
            if not is_authorized:
                logger.info(f"User {email} NOT authorized: {email_status['message']}")
                email_status["status"] = "unauthorized"
                results["details"].append(email_status)
                continue

            # B. Check Strategy Status (SavedInstance)
            # 1. Broadly find the strategy by name and user first
            instance_record = db.query(SavedInstance).filter(
                SavedInstance.user_id == email,
                SavedInstance.strategy_name == request.strategy_name
            ).first()

            if not instance_record:
                 email_status["status"] = "skipped"
                 email_status["message"] = f"No strategy found with name '{request.strategy_name}' for user {email}"
                 logger.info(f"User {email}: {email_status['message']}")
                 results["details"].append(email_status)
                 continue

            # 2. Check Source
            # We expect 'other' for external webhooks.
            # If it's 'internal' or None, we log that specifically.
            if instance_record.source != 'other':
                 email_status["status"] = "skipped"
                 email_status["message"] = f"Strategy found but source is '{instance_record.source}', expected 'other'. Please update strategy source."
                 logger.info(f"User {email}: {email_status['message']}")
                 results["details"].append(email_status)
                 continue

            # 3. Check Status
            if instance_record.status != 'running':
                 email_status["status"] = "skipped"
                 email_status["message"] = f"Strategy '{request.strategy_name}' is currently '{instance_record.status}', expected 'running'"
                 logger.info(f"User {email}: {email_status['message']}")
                 results["details"].append(email_status)
                 continue
                
            results["authorized"] += 1
            logger.info(f"User {email} authorized and strategy running. Proceeding to execution.")
            
            # C. Get Broker Session
            broker_name, client_id, access_token, api_key, broker_credentials = get_broker_session(email)
            
            if not broker_name or not access_token:
                email_status["status"] = "no_broker_session"
                email_status["message"] = "Broker session inactive or missing"
                results["details"].append(email_status)
                continue
                
            # D. Determine Quantity
            quantity = 0
            
            # 1. Try to get quantity from payload (clients map)
            if request.clients and client_id in request.clients:
                try:
                    quantity = int(request.clients[client_id])
                except:
                    quantity = 0
            
            # 2. If payload quantity is 0, try to calculate from reference_capital
            if quantity <= 0:
                try:
                    # Get Reference Capital from SavedInstance
                    capital = instance_record.reference_capital
                    
                    if capital and capital > 0:
                        # Fetch Current Price
                        current_price = PriceService.get_current_price(request.symbol, request.exchange)
                        
                        if current_price > 0:
                            quantity = int(capital / current_price)
                            logger.info(f"Calculated quantity for {email}: {capital} / {current_price} = {quantity}")
                        else:
                            email_status["message"] = f"Failed to fetch price for {request.symbol}, cannot calculate quantity."
                    else:
                        email_status["message"] = f"Reference capital is 0 or missing in strategy configuration."
                        
                except Exception as q_err:
                    logger.error(f"Error calculating quantity for {email}: {q_err}")
                    email_status["message"] = f"Error calculating quantity: {str(q_err)}"

            if quantity <= 0:
                 if not email_status["message"]:
                     email_status["message"] = f"No quantity defined for client {client_id} (Payload: 0, Calc failed)"
                 
                 email_status["status"] = "skipped"
                 results["details"].append(email_status)
                 continue
            
            # E. Place Order
            order_payload = {
                "symbol": request.symbol,
                "exchange": request.exchange,
                "order_side": request.order_side.upper(),
                "product_type": request.product_type if hasattr(request, 'product_type') else "DELIVERY", # Default
                "order_type": "MARKET",
                "quantity": quantity,
                "validity": "DAY",
                "variety": "regular" 
            }
            
            credentials = {
                'api_key': api_key,
                'access_token': access_token,
                'client_id': client_id
            }
            
            if broker_credentials:
                try:
                    creds_dict = json.loads(broker_credentials)
                    if creds_dict.get('sid'):
                        credentials['sid'] = creds_dict.get('sid')
                except:
                    pass

            logger.info(f"Placing order for {email} ({client_id}): {order_payload}")
            exec_result = dispatch_place_order(broker_name, credentials, order_payload)
            
            if exec_result.get("status") == "success":
                results["executed"] += 1
                email_status["status"] = "executed"
                email_status["order_id"] = exec_result.get("data", {}).get("order_id")
                
                # E. Record in Portfolio
                try:
                    # Fetch Price
                    price = 0.0
                    try:
                        price = float(exec_result.get("data", {}).get("average_price", 0.0))
                    except:
                        pass
                        
                    if price <= 0:
                        try:
                            # Use PriceService as fallback
                            price = PriceService.get_current_price(request.symbol, request.exchange)
                        except Exception as p_err:
                            logger.warning(f"Failed to fetch price for {request.symbol}: {p_err}")
                            price = 0.0
                    
                    brokerage = calculate_brokerage(quantity, price, request.order_side.upper())
                    taxes = calculate_taxes(quantity, price, request.order_side.upper())
                    
                    trade_record = PortfolioTrade(
                        user_email=email,
                        run_id=instance_record.run_id if instance_record else request.signal_id, # Use Strategy Run ID
                        strategy_name=request.strategy_name,
                        strategy_type="WEBHOOK",
                        client_code=client_id,
                        trade_date=signal_time.date(),
                        symbol=request.symbol,
                        side=request.order_side.upper(),
                        quantity=quantity,
                        price=price, 
                        brokerage=brokerage,
                        taxes=taxes
                    )
                    db.add(trade_record)
                    db.commit()
                    email_status["portfolio_logged"] = True
                except Exception as db_err:
                    logger.error(f"Failed to log portfolio trade for {email}: {db_err}")
                    email_status["portfolio_logged"] = False
                    
            else:
                results["failed"] += 1
                email_status["status"] = "failed"
                email_status["message"] = exec_result.get("message")
                
        except Exception as e:
            logger.error(f"Error processing user {email}: {e}")
            email_status["status"] = "error"
            email_status["message"] = str(e)
            
        results["details"].append(email_status)
        
    return results

