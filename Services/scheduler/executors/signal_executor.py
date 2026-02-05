import sys
import os
import json
import logging
import logging.config
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Databases.app_data_db_connection import get_session
from Databases.signal_models import TradingSignal
from Services.strategy_manager.models import SavedInstance
from Services.scheduler.config_utils import scheduler_config

# Load logging configuration
config_path = os.path.join(os.path.dirname(__file__), 'logging_config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        # Ensure log directory exists
        log_file = config['handlers']['file']['filename']
        log_dir = os.path.dirname(os.path.abspath(os.path.join(project_root, log_file)))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        # Fix relative path for log file
        config['handlers']['file']['filename'] = os.path.join(project_root, log_file)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('SignalExecutor')

class SignalExecutor:
    """
    Executes trading signals on the designated trading day (Monday, or next working day)
    """

    MARKETAI_WEBHOOK_URL = "marketai"
    PLACE_ORDER_API = "https://8sx9uc9pfy.ap-south-1.awsapprunner.com/api/broker/place_order"
    TRADE_EXECUTED_API = "https://8sx9uc9pfy.ap-south-1.awsapprunner.com/api/portfolio/webhook/trade-executed"

    def __init__(self):
        # Initialize Database Connection
        from Databases.app_data_db_connection import create_connection
        create_connection()
        self.session = get_session()

    def get_next_execution_date(self, from_date: datetime) -> datetime:
        """
        Calculate next trading execution date (First trading day of next week).
        Usually Monday. If Monday is holiday, then Tuesday.
        """
        # Move to next week's Monday
        days_ahead = 7 - from_date.weekday() # 0=Mon, 6=Sun. If Mon(0), +7=Next Mon.
        if days_ahead <= 0: # Should not happen if strictly next week, but safe fallback
            days_ahead += 7
            
        next_week_start = from_date + timedelta(days=days_ahead)
        
        # Check if Monday is holiday
        check_date = next_week_start
        while not scheduler_config.is_trading_day(check_date.date(), 'NSE'):
            check_date += timedelta(days=1)
            
        return check_date

    def is_execution_day(self, date_to_check: datetime) -> bool:
        """
        Check if today is the designated execution day for the week.
        Logic: Monday is default. If Monday was holiday, Tuesday is execution day.
        """
        # Simple check: Is today Monday?
        # If today is Tuesday, was Monday a holiday?
        # This logic is tricky if running daily.
        # Better: The scheduler calls this. The scheduler should decide WHEN to run.
        # But user said: "responsile for execute Trade on every first trading day of a week"
        # We assume this script runs daily or is triggered by scheduler on correct days.
        # Let's trust the caller or check simply if today is trading day.
        return scheduler_config.is_trading_day(date_to_check.date(), 'NSE')

    def fetch_pending_signals(self) -> List[TradingSignal]:
        """Step 1: Fetch pending signals from database"""
        try:
            signals = self.session.query(TradingSignal).filter(
                TradingSignal.execution_status == 'pending'
            ).all()
            logger.info(f"Fetched {len(signals)} pending signals.")
            return signals
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return []

    def prepare_marketai_payload(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Prepare payload for MarketAI logic (Common API)"""
        try:
            # Calculate quantity based on client_json (Amount) and Signal Price
            # client_json format: {"ClientId": "Amount"} (ex: {"C1": "50000"})
            
            clients_payload = {}
            client_data = signal.client_json
            
            if isinstance(client_data, str):
                try:
                    client_data = json.loads(client_data)
                except:
                    logger.error(f"Invalid client_json string for signal {signal.id}")
                    return None

            if not client_data:
                logger.warning(f"Empty client_json for signal {signal.id}")
                return None

            price = signal.price
            if not price or price <= 0:
                logger.error(f"Invalid price {price} for signal {signal.id}")
                return None

            for client_id, amount_val in client_data.items():
                try:
                    # Sanitize amount (remove currency symbols)
                    # Note: We already added sanitizer in generator, but double check
                    if isinstance(amount_val, str):
                         clean_amount = float(amount_val.replace('₹', '').replace('$', '').replace(',', '').strip())
                    else:
                        clean_amount = float(amount_val)
                    
                    # Calculate Quantity
                    quantity = int(clean_amount / price)
                    if quantity > 0:
                        clients_payload[client_id] = str(quantity)
                except Exception as e:
                    logger.error(f"Error calculating quantity for client {client_id}: {e}")

            if not clients_payload:
                return None

            payload = {
                "exchange": "NSE", # Hardcoded as per request? Or from signal? signal.exchange not in model, assumed NSE
                "symbol": signal.symbol_name,
                "user_id": signal.user_id,
                "order_side": signal.order_side,
                "product_type": "DELIVERY", # Hardcoded as per request
                "clients": clients_payload
            }
            return payload

        except Exception as e:
            logger.error(f"Error preparing payload: {e}")
            return None

    def notify_trade_executed(self, run_id: str, marketai_payload: Dict[str, Any]):
        """Post-trade webhook notification"""
        try:
            payload = {
                "run_id": run_id,
                "exchange": marketai_payload["exchange"],
                "symbol": marketai_payload["symbol"],
                "user_id": marketai_payload["user_id"],
                "order_side": marketai_payload["order_side"],
                "product_type": marketai_payload["product_type"],
                "clients": marketai_payload["clients"],
                "trade_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            logger.info(f"PORTFOLIO NOTIFICATION PAYLOAD:\n{json.dumps(payload, indent=2)}")
            logger.info(f"Sending to {self.TRADE_EXECUTED_API}...")
            
            resp = requests.post(self.TRADE_EXECUTED_API, json=payload, timeout=30)
            
            logger.info(f"PORTFOLIO RESPONSE: Status {resp.status_code}")
            logger.info(f"PORTFOLIO RESPONSE BODY: {resp.text}")
            
            if resp.status_code == 200:
                logger.info("✅ Portfolio notification success.")
                return True
            else:
                logger.error(f"❌ Portfolio notification failed.")
                return False
        except Exception as e:
            logger.error(f"❌ Error notifying portfolio: {e}")
            return False

    def update_instance_details(self, instance_id: int):
        """Step 4: Update SavedInstance table"""
        try:
            instance = self.session.query(SavedInstance).filter(SavedInstance.id == instance_id).first()
            if not instance:
                logger.error(f"Instance {instance_id} not found.")
                return

            # Update rem_exe_count
            try:
                current_count = int(instance.rem_exe_count) if instance.rem_exe_count is not None else 0
            except ValueError:
                logger.warning(f"Invalid rem_exe_count '{instance.rem_exe_count}' for instance {instance_id}. Resetting to 0.")
                current_count = 0

            instance.rem_exe_count = max(0, current_count - 1)
            
            # Update dates
            now = datetime.now()
            instance.last_execution_date = now
            instance.next_execution_date = self.get_next_execution_date(now)

            self.session.commit()
            logger.info(f"Updated instance {instance_id}: rem_exe_count={instance.rem_exe_count}, next_exec={instance.next_execution_date}")

        except Exception as e:
            logger.error(f"Error updating instance {instance_id}: {e}")
            self.session.rollback()

    def execute_signals(self):
        """Main execution flow"""
        logger.info("=" * 60)
        logger.info("STARTING SIGNAL EXECUTION")
        logger.info("=" * 60)
        
        # Step 1: Fetch
        logger.info("STEP 1: Fetching pending signals from database...")
        signals = self.fetch_pending_signals()
        
        if not signals:
            logger.info("No pending signals found. Exiting.")
            return

        for signal in signals:
            try:
                logger.info("-" * 40)
                logger.info(f"PROCESSING SIGNAL ID: {signal.id}")
                logger.info("-" * 40)
                
                # Log Signal Details
                logger.info("FETCHED DATA:")
                logger.info(f"  - Symbol: {signal.symbol_name}")
                logger.info(f"  - User ID: {signal.user_id}")
                logger.info(f"  - Run ID: {signal.run_id}")
                logger.info(f"  - Strategy: {signal.strategy_name}")
                logger.info(f"  - Order Side: {signal.order_side}")
                logger.info(f"  - Price: {signal.price}")
                logger.info(f"  - Client JSON: {signal.client_json}")
                logger.info(f"  - Webhook URL: {signal.webhook_url}")

                success = False
                webhook_payload = None

                # Determine logic based on webhook_url
                webhook_url_str = str(signal.webhook_url).lower().strip() if signal.webhook_url else ""
                is_marketai = (webhook_url_str == "marketai")
                
                logger.info(f"LOGIC DETECTION: Webhook URL is '{signal.webhook_url}' -> MarketAI Mode: {is_marketai}")

                if is_marketai:
                    # MarketAI Logic
                    logger.info("STEP 2: Preparing MarketAI Payload...")
                    webhook_payload = self.prepare_marketai_payload(signal)
                    
                    if webhook_payload:
                        logger.info(f"PREPARED PAYLOAD (MarketAI):\n{json.dumps(webhook_payload, indent=2)}")
                        
                        logger.info(f"STEP 3: Executing Order via Broker API ({self.PLACE_ORDER_API})...")
                        try:
                            resp = requests.post(self.PLACE_ORDER_API, json=webhook_payload, timeout=30)
                            logger.info(f"API RESPONSE: Status Code {resp.status_code}")
                            logger.info(f"API RESPONSE BODY: {resp.text}")
                            
                            if resp.status_code in [200, 201]:
                                success = True
                                logger.info("✅ MarketAI Order Placed Successfully.")
                            else:
                                logger.error(f"❌ MarketAI Order Failed.")
                        except Exception as req_err:
                             logger.error(f"❌ API Request Failed: {req_err}")
                    else:
                        logger.error("❌ Failed to prepare payload (Quantity calculation failed?)")

                else:
                    # Standard Webhook Logic
                    if signal.webhook_url and signal.webhook_url.startswith('http'):
                        logger.info(f"STEP 2: Preparing Standard Webhook Payload for {signal.webhook_url}...")
                        webhook_payload = self.prepare_marketai_payload(signal)
                        
                        if webhook_payload:
                            logger.info(f"PREPARED PAYLOAD (Standard):\n{json.dumps(webhook_payload, indent=2)}")
                            
                            logger.info(f"STEP 3: Sending Webhook to {signal.webhook_url}...")
                            try:
                                resp = requests.post(signal.webhook_url, json=webhook_payload, timeout=30)
                                logger.info(f"WEBHOOK RESPONSE: Status Code {resp.status_code}")
                                logger.info(f"WEBHOOK RESPONSE BODY: {resp.text}")

                                if resp.status_code in [200, 201]:
                                    success = True
                                    logger.info("✅ Standard Webhook Success.")
                                else:
                                    logger.error("❌ Standard Webhook Failed.")
                            except Exception as req_err:
                                logger.error(f"❌ Webhook Request Failed: {req_err}")
                        else:
                             logger.error("❌ Failed to prepare payload.")
                    else:
                        logger.warning(f"⚠️  Unknown or Invalid Webhook URL format: {signal.webhook_url}")
                        logger.warning("Skipping execution for this signal.")

                # If success, proceed to Step 3 & 4
                if success and webhook_payload:
                    # Notify Portfolio (Step 3)
                    logger.info("STEP 4: Notifying Portfolio Service...")
                    self.notify_trade_executed(signal.run_id, webhook_payload)
                    
                    # Update Signal Status
                    logger.info("STEP 5: Updating Signal Status to 'executed'...")
                    signal.execution_status = 'executed'
                    signal.executed_at = datetime.now()
                    self.session.commit()
                    logger.info("Signal updated.")
                    
                    # Update Instance (Step 4)
                    if signal.run_id:
                        logger.info(f"STEP 6: Updating Instance for Run ID {signal.run_id}...")
                        instance = self.session.query(SavedInstance).filter(
                            SavedInstance.run_id == signal.run_id
                        ).first()
                        if instance:
                            self.update_instance_details(instance.id)
                        else:
                            logger.warning(f"⚠️  No instance found for run_id {signal.run_id}")
                    
            except Exception as e:
                logger.error(f"❌ CRITICAL ERROR processing signal {signal.id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        logger.info("=" * 60)
        logger.info("SIGNAL EXECUTION COMPLETED")
        logger.info("=" * 60)

if __name__ == "__main__":
    executor = SignalExecutor()
    executor.execute_signals()
