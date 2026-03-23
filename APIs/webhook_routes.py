from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
import json

from Databases.app_data_db_connection import get_db
from Services.strategy_manager.webhook_manager import create_webhook_strategy
from Databases.webhook_models import WebhookRA, WebhookExecutionLog

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMAS
# ============================================================================

class CreateWebhookRequest(BaseModel):
    user_id: str
    strategy_type: str = "External_Strategy"
    strategy_name: str
    reference_capital: float
    client_info: Dict[str, float]
    webhook: Optional[str] = None
    webhook_type: str = "individual"  # 'individual' or 'ra'

    class Config:
        extra = "allow"


class TradeExecuteRequest(BaseModel):
    signal_id: str
    strategy_name: str
    timestamp: str
    symbol: str
    exchange: str
    order_side: str
    authorized_emails: List[str]
    ra_email: Optional[str] = None  # NEW: For summary notifications
    clients: Optional[Dict[str, str]] = None

    class Config:
        extra = "allow"


# ============================================================================
# HELPERS
# ============================================================================

def get_webhook_config(db: Session) -> Optional[WebhookRA]:
    """Fetch the active global RA webhook config from DB."""
    return db.query(WebhookRA).order_by(WebhookRA.id.desc()).first()


def verify_security(request: Request, webhook_secret: Optional[str], config: Optional[WebhookRA]) -> bool:
    """
    Validate IP and master secret.
    Returns True if the request is from a valid master (RA mode).
    Raises HTTPException if IP is blocked.
    Returns False if no config or secret doesn't match master (per-user mode proceeds).
    """
    # 1. IP Whitelist check
    if config and config.is_ip_check_enabled:
        allowed_ips = config.get_allowed_ips()
        if allowed_ips:
            client_ip = request.client.host
            if client_ip not in allowed_ips:
                logger.warning(f"Unauthorized Webhook IP: {client_ip}")
                raise HTTPException(status_code=403, detail="Forbidden: IP not whitelisted")

    # 2. Secret must be present
    if not webhook_secret:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing X-Webhook-Secret header")

    # 3. Check if master (RA) mode
    if config and config.master_secret and webhook_secret == config.master_secret:
        return True  # RA master mode

    return False  # Individual user mode - caller must verify per-strategy key


# ============================================================================
# ENDPOINT: Create Webhook Strategy
# ============================================================================

@router.post(
    "/create_webhook",
    summary="Create External Strategy Configuration",
    description="""
    Create a new configuration for an external strategy (e.g., TradingView).
    - Generates a unique `run_id` with 'EXT' prefix.
    - Generates a unique `webhook_key` stored in the `webhook_keys` table.
    - Sets status to 'running' and source to 'other'.
    - Returns the `webhook_key` to use as the `X-Webhook-Secret` header.
    """,
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
            webhook_type=request.webhook_type,
            db=db
        )
        return result
    except Exception as e:
        logger.error(f"API Error creating webhook strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINT: Trade Execute Webhook
# ============================================================================

from Services.subscription.subscription_models import ProductManager
from Services.subscription.subscription_schemas import SubscriptionStatus
from Services.broker_manager import dispatch_place_order
from helpers.broker_session_manager import get_broker_session
from Services.portfolio.portfolio_models import PortfolioTrade
from Services.portfolio.utils import calculate_brokerage, calculate_taxes
from Services.portfolio.price_service import PriceService
from Services.strategy_manager.models import SavedInstance
from Services.notification_service import WebhookNotifier
from datetime import datetime, timezone


@router.post(
    "/wealthai1.in/trade_execute",
    summary="Execute Trade for Authorized Users",
    description="""
    Execute a trade signal for a list of authorized users.
    Security:
    - Validates source IP against DB whitelist (if enabled).
    - If X-Webhook-Secret matches MASTER secret (RA mode): all emails are trusted.
    - Otherwise, each user's webhook_key in webhook_keys table must match the secret.
    Validation:
    - Checks subscription (ACTIVE/TRIAL, product_code=M, not expired).
    - Checks SavedInstance source='other', status='running'.
    - Places orders via Broker API and records in Portfolio.
    """,
)
async def trade_execute_webhook(
    request_data: TradeExecuteRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    logger.info(f"Received trade execution webhook: {request_data.signal_id} for {request_data.symbol}")

    # --- 1. Load Webhook Config from DB ---
    config = get_webhook_config(db)

    # --- 2. Extract Secret Header ---
    webhook_secret = request.headers.get("X-Webhook-Secret")

    # --- 3. Run Security Check (IP + Master Secret) ---
    is_master_mode = False
    try:
        is_master_mode = verify_security(request, webhook_secret, config)
    except HTTPException as auth_err:
        # Log failure for all intended users before raising
        try:
            for email in request_data.authorized_emails:
                log_entry = WebhookExecutionLog(
                    user_email=email,
                    strategy_name=request_data.strategy_name,
                    symbol=request_data.symbol,
                    side=request_data.order_side.upper(),
                    status="error",
                    message=f"Security Error: {auth_err.detail}",
                    ra_email=request_data.ra_email
                )
                db.add(log_entry)
            db.commit()
        except Exception as log_err:
            logger.error(f"Failed to log security error: {log_err}")
            db.rollback()
        raise auth_err

    results = {
        "processed": 0,
        "authorized": 0,
        "executed": 0,
        "failed": 0,
        "details": []
    }

    # --- 4. Parse Timestamp ---
    try:
        signal_time = datetime.fromisoformat(request_data.timestamp.replace('Z', '+00:00'))
    except Exception:
        signal_time = datetime.now(timezone.utc)

    # --- 5. Iterate Authorized Emails ---
    for email in request_data.authorized_emails:
        results["processed"] += 1
        email_status = {
            "email": email,
            "status": "skipped",
            "message": "",
            "execution_mode": "ra" if is_master_mode else "individual"
        }
        
        # We'll use a local variable to capture instance_record for logging
        local_instance_record = None

        try:
            # A. Check Subscription
            sub_record = db.query(ProductManager).filter(
                ProductManager.user_email == email,
                ProductManager.product_code == 'M'
            ).order_by(ProductManager.subscription_end_date.desc()).first()

            is_authorized = False
            if sub_record:
                current_time = datetime.now(timezone.utc)
                if sub_record.subscription_end_date and sub_record.subscription_end_date > current_time:
                    if sub_record.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]:
                        is_authorized = True
                    else:
                        email_status["message"] = "you plan is expire please subscribe MarketsAi1 first"
                else:
                    email_status["message"] = "you plan is expire please subscribe MarketsAi1 first"
            else:
                email_status["message"] = "you plan is expire please subscribe MarketsAi1 first"

            if not is_authorized:
                logger.info(f"User {email} NOT authorized: {email_status['message']}")
                email_status["status"] = "unauthorized"
            else:
                # B. Find Strategy in SavedInstance
                local_instance_record = db.query(SavedInstance).filter(
                    SavedInstance.user_id == email,
                    SavedInstance.strategy_name == request_data.strategy_name
                ).first()

                if not local_instance_record:
                    email_status["status"] = "skipped"
                    email_status["message"] = f"No strategy found with name '{request_data.strategy_name}' for user {email}"
                else:
                    # C. Per-User Security Check (if not master mode)
                    is_user_security_ok = True
                    if not is_master_mode:
                        key_record = db.query(WebhookIndividual).filter(
                            WebhookIndividual.run_id == local_instance_record.run_id,
                            WebhookIndividual.user_email == email,
                            WebhookIndividual.is_active == True
                        ).first()

                        if not key_record or key_record.webhook_key != webhook_secret:
                            email_status["status"] = "unauthorized"
                            email_status["message"] = "Invalid X-Webhook-Secret for this user strategy"
                            logger.warning(f"Invalid webhook key used for user {email} / run_id {local_instance_record.run_id}")
                            is_user_security_ok = False

                    if is_user_security_ok:
                        # D. Check Strategy Source and Status
                        if local_instance_record.source != 'other':
                            email_status["status"] = "skipped"
                            email_status["message"] = f"Strategy source is '{local_instance_record.source}', expected 'other'"
                        elif local_instance_record.status != 'running':
                            email_status["status"] = "skipped"
                            email_status["message"] = f"Strategy '{request_data.strategy_name}' is '{local_instance_record.status}', expected 'running'"
                        else:
                            results["authorized"] += 1
                            logger.info(f"User {email} authorized and strategy running. Proceeding to execution.")

                            # E. Get Broker Session
                            broker_name, client_id, access_token, api_key, broker_credentials = get_broker_session(email)

                            if not broker_name or not access_token:
                                email_status["status"] = "no_broker_session"
                                email_status["message"] = "Account not found please configure there account through MarketsAi1"
                            else:
                                # F. Determine Quantity
                                quantity = 0
                                if request_data.clients and client_id in request_data.clients:
                                    try:
                                        quantity = int(request_data.clients[client_id])
                                    except Exception:
                                        quantity = 0

                                if quantity <= 0:
                                    try:
                                        capital = local_instance_record.reference_capital
                                        if capital and capital > 0:
                                            current_price = PriceService.get_current_price(request_data.symbol, request_data.exchange)
                                            if current_price > 0:
                                                quantity = int(capital / current_price)
                                                logger.info(f"Calculated quantity for {email}: {capital} / {current_price} = {quantity}")
                                            else:
                                                email_status["message"] = f"Failed to fetch price for {request_data.symbol}"
                                        else:
                                            email_status["message"] = "Reference capital is 0 or missing"
                                    except Exception as q_err:
                                        logger.error(f"Error calculating quantity for {email}: {q_err}")
                                        email_status["message"] = f"Error calculating quantity: {str(q_err)}"

                                if quantity <= 0:
                                    if not email_status["message"]:
                                        email_status["message"] = f"No quantity defined for client {client_id}"
                                    email_status["status"] = "skipped"
                                else:
                                    # G. Place Order
                                    order_payload = {
                                        "symbol": request_data.symbol,
                                        "exchange": request_data.exchange,
                                        "order_side": request_data.order_side.upper(),
                                        "product_type": getattr(request_data, 'product_type', 'DELIVERY'),
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
                                        except Exception:
                                            pass

                                    logger.info(f"Placing order for {email} ({client_id}): {order_payload}")
                                    exec_result = dispatch_place_order(broker_name, credentials, order_payload)

                                    if exec_result.get("status") == "success":
                                        results["executed"] += 1
                                        email_status["status"] = "executed"
                                        email_status["order_id"] = exec_result.get("data", {}).get("order_id")

                                        # H. Record in Portfolio
                                        try:
                                            price = 0.0
                                            try:
                                                price = float(exec_result.get("data", {}).get("average_price", 0.0))
                                            except Exception:
                                                pass

                                            if price <= 0:
                                                try:
                                                    price = PriceService.get_current_price(request_data.symbol, request_data.exchange)
                                                except Exception as p_err:
                                                    logger.warning(f"Failed to fetch price for {request_data.symbol}: {p_err}")
                                                    price = 0.0

                                            brokerage = calculate_brokerage(quantity, price, request_data.order_side.upper())
                                            taxes = calculate_taxes(quantity, price, request_data.order_side.upper())

                                            trade_record = PortfolioTrade(
                                                user_email=email,
                                                run_id=local_instance_record.run_id,
                                                strategy_name=request_data.strategy_name,
                                                strategy_type="WEBHOOK",
                                                client_code=client_id,
                                                trade_date=signal_time.date(),
                                                symbol=request_data.symbol,
                                                side=request_data.order_side.upper(),
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
                                        email_status["message"] = exec_result.get("message", "Order placement failed")

        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            # Map technical errors to user-friendly messages
            if "not enough values to unpack" in error_msg:
                friendly_msg = "System error: Session data mismatch. Please contact support."
            elif "Connection" in error_msg or "timeout" in error_msg.lower():
                friendly_msg = "Network error: Communication with broker failed."
            else:
                friendly_msg = f"Execution error: {error_msg}"
            
            logger.error(f"Error processing user {email}: {error_msg}")
            email_status["status"] = "error"
            email_status["message"] = friendly_msg

        results["details"].append(email_status)

        # --- I. Log to WebhookExecutionLog (For Daily Report) ---
        try:
            log_entry = WebhookExecutionLog(
                user_email=email,
                strategy_name=request_data.strategy_name,
                symbol=request_data.symbol,
                side=request_data.order_side.upper(),
                status=email_status["status"],
                message=email_status["message"],
                ra_email=request_data.ra_email,
                run_id=local_instance_record.run_id if local_instance_record else None
            )
            db.add(log_entry)
            db.commit()
        except Exception as log_err:
            logger.error(f"Failed to record webhook execution log for {email}: {log_err}")
            db.rollback()

    # --- 6. Send Email Notifications ---
    try:
        notifier = WebhookNotifier()
        if request_data.ra_email:
            # RA Mode: Send one summary to RA
            notifier.send_ra_summary(
                ra_email=request_data.ra_email,
                strategy_name=request_data.strategy_name,
                symbol=request_data.symbol,
                side=request_data.order_side,
                details=results["details"]
            )
            logger.info(f"RA Summary email dispatched to {request_data.ra_email}")
        else:
            # Individual Mode: Send to each client
            logger.info(f"Checking notifications for {len(results['details'])} users in individual mode")
            for detail in results["details"]:
                user_email = detail["email"]
                status = detail["status"]
                logger.info(f"User {user_email} status: {status}")
                
                # Send email for ALL statuses as requested by user
                logger.info(f"Attempting to send email to {user_email}...")
                sent = notifier.send_client_notification(
                    client_email=user_email,
                    strategy_name=request_data.strategy_name,
                    symbol=request_data.symbol,
                    side=request_data.order_side,
                    status=status,
                    message=detail.get("message", "")
                )
                logger.info(f"Email to {user_email} status: {'Sent' if sent else 'Failed'}")
            logger.info("Individual client notification emails dispatch process completed.")
    except Exception as notify_err:
        logger.error(f"Notification dispatch failed: {notify_err}")

    return results

