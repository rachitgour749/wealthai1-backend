"""
Execution Service - Core Logic
Handles automatic signal execution on Monday
"""

import sqlite3
import json
import logging
import requests
import time
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from Services.execution.database import ExecutionDatabase
except ImportError:
    # Fallback to relative import
    from .database import ExecutionDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExecutionService:
    """Service for executing trading signals and tracking results"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize execution service"""
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.db_path = os.path.join(project_root, 'unified_etf_data.sqlite')
        else:
            self.db_path = db_path
        
        self.execution_db = ExecutionDatabase(self.db_path)
        logger.info(f"Execution service initialized with database: {self.db_path}")
    
    def get_connection(self):
        """Get database connection with timeout"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn
    
    def fetch_etf_signals(self, signal_date: str = None, side: str = None) -> List[Dict[str, Any]]:
        """
        Fetch ETF signals from live_signals table
        
        Args:
            signal_date: Date to fetch signals for (YYYY-MM-DD). Defaults to last Friday
            side: Filter by BUY or SELL. None = both
            
        Returns:
            List of ETF signal dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = '''
                SELECT 
                    user_id,
                    run_id,
                    signal_symbol,
                    side,
                    score,
                    reason,
                    payload_json,
                    signal_date,
                    created_at
                FROM live_signals
                WHERE signal_date = ?
            '''
            params = [signal_date]
            
            if side:
                query += " AND side = ?"
                params.append(side)
            
            query += " ORDER BY user_id, run_id, signal_symbol"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            signals = []
            for row in rows:
                signal = dict(row)
                signal['signal_type'] = 'ETF_rotation'  # Mark as ETF signal
                # Parse payload_json
                if signal['payload_json']:
                    try:
                        signal['payload'] = json.loads(signal['payload_json'])
                    except:
                        signal['payload'] = {}
                else:
                    signal['payload'] = {}
                
                signals.append(signal)
            
            logger.info(f"Found {len(signals)} ETF signals for execution on {signal_date}")
            return signals
            
        except Exception as e:
            logger.error(f"Error fetching ETF signals: {e}")
            return []

    def fetch_stock_signals(self, signal_date: str = None, side: str = None) -> List[Dict[str, Any]]:
        """
        Fetch Stock signals from live_stock_signals table
        
        Args:
            signal_date: Date to fetch signals for (YYYY-MM-DD). Defaults to last Friday
            side: Filter by BUY or SELL. None = both
            
        Returns:
            List of Stock signal dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = '''
                SELECT 
                    user_id,
                    run_id,
                    symbol as signal_symbol,
                    side,
                    score,
                    reason,
                    ltp as payload_json,
                    signal_date,
                    created_at
                FROM live_stock_signals
                WHERE signal_date = ?
            '''
            params = [signal_date]
            
            if side:
                query += " AND side = ?"
                params.append(side)
            
            query += " ORDER BY user_id, run_id, symbol"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            signals = []
            for row in rows:
                signal = dict(row)
                signal['signal_type'] = 'stock_rotation'  # Mark as STOCK signal
                
                # Convert ltp to payload format for consistency
                if signal['payload_json']:
                    try:
                        # ltp is stored as JSON with price and currency
                        ltp_data = json.loads(signal['payload_json'])
                        signal['payload'] = {
                            'price': ltp_data.get('price', 0),
                            'currency': ltp_data.get('currency', 'INR')
                        }
                    except (json.JSONDecodeError, TypeError):
                        # If ltp is not JSON, treat as price
                        signal['payload'] = {
                            'price': float(signal['payload_json']) if signal['payload_json'] else 0,
                            'currency': 'INR'
                        }
                else:
                    signal['payload'] = {'price': 0, 'currency': 'INR'}
                
                signals.append(signal)
            
            logger.info(f"Found {len(signals)} Stock signals for execution on {signal_date}")
            return signals
            
        except Exception as e:
            logger.error(f"Error fetching Stock signals: {e}")
            return []

    def fetch_signals_to_execute(self, signal_date: str = None, side: str = None, signal_type: str = 'auto') -> List[Dict[str, Any]]:
        """
        Fetch signals to execute from both ETF and Stock tables
        
        Args:
            signal_date: Date to fetch signals for (YYYY-MM-DD). Defaults to last Friday
            side: Filter by BUY or SELL. None = both
            signal_type: 'auto', 'etf', 'stock', or 'both'
            
        Returns:
            List of signal dictionaries (ETF and/or Stock)
        """
        try:
            # Calculate last Friday if no date provided
            if not signal_date:
                today = datetime.now()
                days_since_friday = (today.weekday() - 4) % 7
                last_friday = today - timedelta(days=days_since_friday)
                signal_date = last_friday.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching signals for {signal_date} (type: {signal_type})")
            
            all_signals = []
            
            if signal_type in ['auto', 'both', 'etf']:
                # Fetch ETF signals
                etf_signals = self.fetch_etf_signals(signal_date, side)
                all_signals.extend(etf_signals)
                logger.info(f"ETF signals: {len(etf_signals)}")
            
            if signal_type in ['auto', 'both', 'stock']:
                # Fetch Stock signals
                stock_signals = self.fetch_stock_signals(signal_date, side)
                all_signals.extend(stock_signals)
                logger.info(f"Stock signals: {len(stock_signals)}")
            
            # Sort by signal type, then by user_id, run_id, symbol
            # Handle None values by converting them to empty strings for sorting
            all_signals.sort(key=lambda x: (
                x.get('signal_type', '') or '',
                x.get('user_id', '') or '',
                x.get('run_id', '') or '',
                x.get('signal_symbol', '') or ''
            ))
            
            logger.info(f"Total signals found: {len(all_signals)} ({len([s for s in all_signals if s.get('signal_type') == 'ETF_rotation'])} ETF + {len([s for s in all_signals if s.get('signal_type') == 'stock_rotation'])} Stock)")
            return all_signals
            
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            raise
    
    def fetch_deployment_details(self, user_id: str, run_id: str, signal_type: str = 'ETF_rotation') -> Optional[Dict[str, Any]]:
        """
        Fetch client information, webhook URL, and accumulation_weeks from etf_saved_strategy or stock_saved_strategy table
        
        Args:
            user_id: User email
            run_id: Deployment run ID
            signal_type: 'ETF_rotation' or 'stock_rotation' - determines which table to query
            
        Returns:
            Dictionary with client_information, webhook_url, and accumulation_weeks
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Determine which table to query based on signal type
            if signal_type == 'stock_rotation':
                table_name = 'stock_saved_strategy'
            else:  # Default to ETF_rotation
                table_name = 'etf_saved_strategy'
            
            cursor.execute(f'''
                SELECT 
                    client_information_json,
                    webhook_url,
                    accumulation_weeks
                FROM {table_name}
                WHERE user_id = ? AND run_id = ?
            ''', (user_id, run_id))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                logger.warning(f"No deployment details found in {table_name} for user_id={user_id}, run_id={run_id}")
                return None
            
            result = {
                'client_information': json.loads(row['client_information_json']) if row['client_information_json'] else {},
                'webhook_url': row['webhook_url'],
                'accumulation_weeks': row['accumulation_weeks'] if row['accumulation_weeks'] is not None else 0
            }
            
            return result
            
        except Exception as e:
            table_name = 'etf_saved_strategy' if signal_type != 'stock_rotation' else 'stock_saved_strategy'
            logger.error(f"Error fetching deployment details from {table_name}: {e}")
            raise
    
    def calculate_quantities(self, signal_price: float, client_capitals: Dict[str, float]) -> Dict[str, int]:
        """
        Calculate quantity for each client based on capital and signal price
        Formula: quantity = floor(capital / signal_price)
        
        Args:
            signal_price: Current price of the stock/ETF
            client_capitals: Dictionary of {client_name: capital_amount}
            
        Returns:
            Dictionary of {client_name: quantity}
        """
        quantities = {}
        
        for client_name, capital in client_capitals.items():
            try:
                # Clean and convert capital to float
                if isinstance(capital, str):
                    # Remove currency symbols (₹, $, etc.), commas, and whitespace
                    capital_cleaned = capital.replace('₹', '').replace('$', '').replace(',', '').strip()
                    capital_float = float(capital_cleaned)
                else:
                    capital_float = float(capital)
                
                if capital_float > 0 and signal_price > 0:
                    # Calculate quantity (floor division to avoid over-allocation)
                    quantity = int(capital_float / signal_price)
                    quantities[client_name] = quantity
                    logger.info(f"✅ {client_name}: ₹{capital_float} ÷ ₹{signal_price} = {quantity} shares")
                else:
                    quantities[client_name] = 0
                    logger.warning(f"⚠️  {client_name}: Invalid values (capital={capital_float}, price={signal_price}), quantity=0")
                    
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Invalid capital value for {client_name}: '{capital}', error: {e}")
                quantities[client_name] = 0
        
        return quantities
    
    def prepare_execution_payload(self, signal: Dict[str, Any], quantities: Dict[str, int]) -> Dict[str, Any]:
        """
        Prepare JSON payload for webhook
        
        Args:
            signal: Signal dictionary from live_signals
            quantities: Calculated quantities for each client
            
        Returns:
            Webhook payload dictionary
        """
        payload = {
            "exchange": "NSE",
            "symbol": signal.get('signal_symbol', ''),
            "order_side": signal.get('side', ''),
            "product_type": "delivery",
            "clients": {name: str(qty) for name, qty in quantities.items()}
        }
        
        return payload
    
    def send_to_webhook(self, webhook_url: str, payload: Dict[str, Any], timeout: int = 30) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Send execution payload to webhook URL
        
        Args:
            webhook_url: Webhook URL
            payload: JSON payload to send
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            logger.info(f"Sending payload to webhook: {webhook_url}")
            
            headers = {
                'Content-Type': 'application/json'
                # Removed User-Agent to match Postman exactly
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            # Accept all 2xx status codes as success (200, 201, 202, 204, etc.)
            if 200 <= response.status_code < 300:
                logger.info(f"✅ Webhook success: {response.status_code}")
                try:
                    response_data = response.json()
                    logger.info(f"📥 Webhook Response: {json.dumps(response_data, indent=2)}")
                except:
                    response_data = {"status": "success", "message": response.text}
                    logger.info(f"📥 Webhook Response (text): {response.text}")
                
                return True, response_data, None
            else:
                error_msg = f"Webhook failed with status {response.status_code}: {response.text}"
                logger.error(f"❌ {error_msg}")
                return False, None, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = f"Webhook timeout after {timeout} seconds"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"Webhook error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
    
    def execute_single_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single signal
        
        Args:
            signal: Signal dictionary from live_signals
            
        Returns:
            Execution result dictionary
        """
        try:
            user_id = signal.get('user_id') or ''
            run_id = signal.get('run_id') or ''
            signal_symbol = signal.get('signal_symbol') or ''
            side = signal.get('side') or ''
            
            # Validate required fields
            if not user_id or not run_id or not signal_symbol or not side:
                error_msg = f"Missing required signal fields: user_id={user_id}, run_id={run_id}, symbol={signal_symbol}, side={side}"
                logger.error(f"❌ {error_msg}")
                return {
                    'success': False,
                    'signal': signal_symbol or 'unknown',
                    'error': error_msg
                }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Executing signal: {signal_symbol} {side} for {user_id} (run_id: {run_id})")
            logger.info(f"{'='*60}")
            
            # Print to console
            signal_type = signal.get('signal_type', 'ETF_rotation')  # Default to ETF_rotation for backward compatibility
            print(f"\n{'='*60}")
            print(f"🎯 EXECUTING {signal_type} SIGNAL")
            print(f"{'='*60}")
            print(f"📊 Symbol: {signal_symbol}")
            print(f"📈 Side: {side}")
            print(f"🏷️  Type: {signal_type}")
            print(f"👤 User: {user_id}")
            print(f"🆔 Run ID: {run_id}")
            
            # Step 1: Fetch deployment details
            deployment = self.fetch_deployment_details(user_id, run_id, signal_type)
            
            if not deployment:
                error_msg = f"No deployment details found for user_id={user_id}, run_id={run_id}"
                logger.error(f"❌ {error_msg}")
                
                # Save failed execution record
                self.execution_db.insert_execution_record(
                    user_id=user_id,
                    run_id=run_id,
                    signal_symbol=signal_symbol,
                    side=side,
                    success=False,
                    error_message=error_msg
                )
                
                return {
                    'success': False,
                    'signal': signal_symbol,
                    'error': error_msg
                }
            
            client_capitals = deployment['client_information']
            webhook_url = deployment['webhook_url']
            
            logger.info(f"📊 Found {len(client_capitals)} clients")
            
            accumulation_weeks = deployment.get('accumulation_weeks', 0)
            
            print(f"\n📋 STEP 1: Deployment Details")
            print(f"   Clients: {len(client_capitals)}")
            print(f"   Client Names: {list(client_capitals.keys())}")
            print(f"   Webhook: {webhook_url}")
            print(f"   Accumulation Weeks: {accumulation_weeks}")
            
            # Step 2: Get signal price from payload
            signal_price = signal.get('payload', {}).get('price', 0)
            
            if signal_price <= 0:
                error_msg = f"Invalid signal price: {signal_price}"
                logger.error(f"❌ {error_msg}")
                
                self.execution_db.insert_execution_record(
                    user_id=user_id,
                    run_id=run_id,
                    signal_symbol=signal_symbol,
                    side=side,
                    success=False,
                    error_message=error_msg
                )
                
                return {
                    'success': False,
                    'signal': signal_symbol,
                    'error': error_msg
                }
            
            logger.info(f"💰 Signal price: ₹{signal_price}")
            
            print(f"\n💰 STEP 2: Signal Price")
            print(f"   Price: ₹{signal_price}")
            
            # Step 3: Calculate quantities
            quantities = self.calculate_quantities(signal_price, client_capitals)
            logger.info(f"🧮 Calculated quantities: {quantities}")
            
            print(f"\n🧮 STEP 3: Quantity Calculation")
            for client, capital in client_capitals.items():
                qty = quantities.get(client, 0)
                # Clean capital value for display
                try:
                    if isinstance(capital, str):
                        capital_cleaned = capital.replace('₹', '').replace('$', '').replace(',', '').strip()
                        capital_float = float(capital_cleaned)
                    else:
                        capital_float = float(capital)
                except:
                    capital_float = 0
                print(f"   {client}: ₹{capital_float} ÷ ₹{signal_price} = {qty} shares")
            
            # Step 4: Prepare payload
            payload = self.prepare_execution_payload(signal, quantities)
            logger.info(f"📦 Prepared payload: {json.dumps(payload, indent=2)}")
            
            print(f"\n📦 STEP 4: JSON Payload Prepared")
            print(f"   Exchange: {payload['exchange']}")
            print(f"   Symbol: {payload['symbol']}")
            print(f"   Order Side: {payload['order_side']}")
            print(f"   Product Type: {payload['product_type']}")
            print(f"   Clients: {payload['clients']}")
            
            # Step 5: Send to webhook
            print(f"\n📤 SENDING TO WEBHOOK: {webhook_url}")
            print(f"📦 PAYLOAD:")
            print(json.dumps(payload, indent=2))
            print(f"\n⏳ Waiting for response...")
            
            success, response_data, error_message = self.send_to_webhook(webhook_url, payload)
            
            if success:
                print(f"\n✅ WEBHOOK SUCCESS!")
                print(f"📥 RESPONSE:")
                print(json.dumps(response_data, indent=2) if response_data else "No response data")
            else:
                print(f"\n❌ WEBHOOK FAILED!")
                print(f"⚠️  ERROR: {error_message}")
            
            # Step 6: Save execution record
            self.execution_db.insert_execution_record(
                user_id=user_id,
                run_id=run_id,
                signal_symbol=signal_symbol,
                side=side,
                success=success,
                response_data=json.dumps(response_data) if response_data else None,
                error_message=error_message,
                webhook_url=webhook_url,
                payload_sent=json.dumps(payload)
            )
            
            if success:
                logger.info(f"✅ Execution successful for {signal_symbol}")
            else:
                logger.error(f"❌ Execution failed for {signal_symbol}: {error_message}")
            
            return {
                'success': success,
                'signal': signal_symbol,
                'side': side,
                'user_id': user_id,
                'run_id': run_id,
                'response': response_data,
                'error': error_message
            }
            
        except Exception as e:
            error_msg = f"Unexpected error executing signal: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Save failed execution record
            try:
                self.execution_db.insert_execution_record(
                    user_id=signal.get('user_id', ''),
                    run_id=signal.get('run_id', ''),
                    signal_symbol=signal.get('signal_symbol', ''),
                    side=signal.get('side', ''),
                    success=False,
                    error_message=error_msg
                )
            except:
                pass
            
            return {
                'success': False,
                'signal': signal.get('signal_symbol', 'unknown'),
                'error': error_msg
            }
    
    def execute_all_signals(self, signal_date: str = None, side: str = None, signal_type: str = 'auto') -> Dict[str, Any]:
        """
        Execute all signals for a given date
        
        Args:
            signal_date: Date to execute signals for (YYYY-MM-DD). Defaults to last Friday
            side: Filter by BUY or SELL. None = both
            signal_type: 'auto', 'etf', 'stock', or 'both'
            
        Returns:
            Execution summary dictionary
        """
        try:
            start_time = datetime.now()
            logger.info("="*80)
            logger.info("🚀 STARTING AUTOMATIC SIGNAL EXECUTION")
            logger.info(f"📊 Signal Type: {signal_type.upper()}")
            logger.info("="*80)
            
            # Clear previous week's execution data before storing new data
            # This ensures we keep only the latest Monday's execution records
            logger.info("🗑️  Clearing previous Monday's execution data...")
            print("🗑️  Clearing previous Monday's execution data...")
            self.execution_db.clear_all_execution_data()
            logger.info("✅ Previous execution data cleared successfully")
            print("✅ Previous execution data cleared - ready to store new execution data")
            
            # Fetch signals
            signals = self.fetch_signals_to_execute(signal_date, side, signal_type)
            
            if not signals:
                logger.warning("⚠️  No signals found to execute")
                return {
                    'success': True,
                    'message': 'No signals to execute',
                    'total_signals': 0,
                    'successful': 0,
                    'failed': 0,
                    'results': []
                }
            
            logger.info(f"📊 Found {len(signals)} signals to execute")
            
            # Group signals by (user_id, run_id, signal_type) for accumulation_weeks logic
            signal_groups = {}
            for signal in signals:
                user_id = signal.get('user_id') or ''
                run_id = signal.get('run_id') or ''
                signal_type_key = signal.get('signal_type', 'ETF_rotation')
                group_key = (user_id, run_id, signal_type_key)
                
                if group_key not in signal_groups:
                    signal_groups[group_key] = []
                signal_groups[group_key].append(signal)
            
            logger.info(f"📦 Grouped signals into {len(signal_groups)} run_id groups")
            print(f"\n📦 Grouped signals into {len(signal_groups)} run_id groups")
            
            # Execute signals grouped by run_id
            results = []
            successful = 0
            failed = 0
            
            for group_idx, (group_key, group_signals) in enumerate(signal_groups.items(), 1):
                user_id, run_id, signal_type_key = group_key
                
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing Run ID Group {group_idx}/{len(signal_groups)}: {run_id} (User: {user_id}, Type: {signal_type_key})")
                logger.info(f"{'='*80}")
                
                print(f"\n{'='*80}")
                print(f"📦 RUN ID GROUP {group_idx}/{len(signal_groups)}")
                print(f"   User ID: {user_id}")
                print(f"   Run ID: {run_id}")
                print(f"   Signal Type: {signal_type_key}")
                print(f"   Total Signals: {len(group_signals)}")
                print(f"{'='*80}")
                
                # Fetch deployment details to get accumulation_weeks for this run_id
                deployment = self.fetch_deployment_details(user_id, run_id, signal_type_key)
                
                if not deployment:
                    logger.error(f"❌ No deployment details found for user_id={user_id}, run_id={run_id}")
                    # Process all signals but mark as failed
                    for signal in group_signals:
                        result = {
                            'success': False,
                            'signal': signal.get('signal_symbol', 'unknown'),
                            'error': f"No deployment details found for run_id={run_id}",
                            'user_id': user_id,
                            'run_id': run_id
                        }
                        results.append(result)
                        failed += 1
                    continue
                
                accumulation_weeks = deployment.get('accumulation_weeks', 0)
                logger.info(f"📊 Accumulation Weeks for run_id={run_id}: {accumulation_weeks}")
                print(f"📊 Accumulation Weeks: {accumulation_weeks}")
                
                # Filter signals based on accumulation_weeks
                if accumulation_weeks > 0:
                    # Execute only BUY signals
                    filtered_signals = [s for s in group_signals if s.get('side', '').upper() == 'BUY']
                    skipped_signals = [s for s in group_signals if s.get('side', '').upper() != 'BUY']
                    
                    logger.info(f"🔄 Accumulation phase active (weeks remaining: {accumulation_weeks})")
                    logger.info(f"   ✅ Will execute {len(filtered_signals)} BUY signals")
                    logger.info(f"   ⏭️  Will skip {len(skipped_signals)} SELL signals")
                    
                    print(f"\n🔄 ACCUMULATION PHASE ACTIVE")
                    print(f"   Weeks Remaining: {accumulation_weeks}")
                    print(f"   ✅ Will execute {len(filtered_signals)} BUY signals")
                    print(f"   ⏭️  Will skip {len(skipped_signals)} SELL signals")
                    
                    if skipped_signals:
                        for skipped in skipped_signals:
                            logger.info(f"   ⏭️  Skipping SELL signal: {skipped.get('signal_symbol')}")
                            print(f"   ⏭️  Skipping SELL signal: {skipped.get('signal_symbol')}")
                else:
                    # Execute all signals (BUY and SELL)
                    filtered_signals = group_signals
                    logger.info(f"✅ Accumulation complete - executing all signals (BUY + SELL)")
                    print(f"\n✅ ACCUMULATION COMPLETE")
                    print(f"   Will execute all {len(filtered_signals)} signals (BUY + SELL)")
                
                # Execute filtered signals for this run_id
                run_id_successful = 0
                run_id_failed = 0
                run_id_results = []
                
                for i, signal in enumerate(filtered_signals, 1):
                    logger.info(f"\n--- Processing signal {i}/{len(filtered_signals)} in run_id={run_id} ---")
                    result = self.execute_single_signal(signal)
                    run_id_results.append(result)
                    results.append(result)
                    
                    if result['success']:
                        run_id_successful += 1
                        successful += 1
                    else:
                        run_id_failed += 1
                        failed += 1
                    
                    # Add delay between requests to prevent rate limiting
                    if i < len(filtered_signals):  # Don't delay after the last signal
                        delay_seconds = 5
                        logger.info(f"⏳ Waiting {delay_seconds} seconds before next request...")
                        print(f"⏳ Waiting {delay_seconds} seconds before next request...")
                        time.sleep(delay_seconds)
                
                # After all signals for this run_id are executed, update accumulation_weeks if successful
                if accumulation_weeks > 0:
                    # Check if all executed signals were successful
                    all_successful = all(r['success'] for r in run_id_results)
                    
                    if all_successful and len(filtered_signals) > 0:
                        logger.info(f"✅ All signals executed successfully for run_id={run_id}")
                        print(f"\n✅ All signals executed successfully for run_id={run_id}")
                        
                        # Update accumulation_weeks
                        update_success = self.update_accumulation_weeks(user_id, run_id, signal_type_key)
                        if update_success:
                            logger.info(f"✅ Updated accumulation_weeks for run_id={run_id}")
                            print(f"✅ Updated accumulation_weeks for run_id={run_id}")
                        else:
                            logger.warning(f"⚠️  Failed to update accumulation_weeks for run_id={run_id}")
                            print(f"⚠️  Failed to update accumulation_weeks for run_id={run_id}")
                    else:
                        logger.warning(f"⚠️  Some signals failed for run_id={run_id} - accumulation_weeks not updated")
                        print(f"\n⚠️  Some signals failed for run_id={run_id}")
                        print(f"   Successful: {run_id_successful}, Failed: {run_id_failed}")
                        print(f"   accumulation_weeks NOT updated (will retry next execution)")
                
                # Summary for this run_id
                logger.info(f"\n📊 Run ID {run_id} Summary: {run_id_successful} successful, {run_id_failed} failed")
                print(f"\n📊 Run ID {run_id} Summary:")
                print(f"   ✅ Successful: {run_id_successful}")
                print(f"   ❌ Failed: {run_id_failed}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*80)
            logger.info("📊 EXECUTION SUMMARY")
            logger.info("="*80)
            logger.info(f"Total signals: {len(signals)}")
            logger.info(f"✅ Successful: {successful}")
            logger.info(f"❌ Failed: {failed}")
            logger.info(f"⏱️  Duration: {duration:.2f} seconds")
            logger.info("="*80)
            
            # Print summary to console
            print("\n" + "="*80)
            print("📊 FINAL EXECUTION SUMMARY")
            print("="*80)
            print(f"Total Signals: {len(signals)}")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print("💾 All execution details saved to database (previous data cleared)")
            print("="*80)
            
            return {
                'success': True,
                'message': 'Signal execution completed',
                'total_signals': len(signals),
                'successful': successful,
                'failed': failed,
                'duration_seconds': duration,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ Error in execute_all_signals: {e}")
            return {
                'success': False,
                'message': f'Execution failed: {str(e)}',
                'total_signals': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }
    
    def get_execution_history(self, **filters) -> List[Dict[str, Any]]:
        """Get execution history with filters"""
        return self.execution_db.get_execution_history(**filters)
    
    def get_execution_summary(self, user_id: str = None, run_id: str = None) -> Dict[str, Any]:
        """Get execution summary statistics"""
        return self.execution_db.get_execution_summary(user_id, run_id)
    
    def update_accumulation_weeks(self, user_id: str, run_id: str, signal_type: str = 'ETF_rotation') -> bool:
        """
        Decrement accumulation_weeks by 1 in etf_saved_strategy or stock_saved_strategy table
        
        Args:
            user_id: User email
            run_id: Deployment run ID
            signal_type: 'ETF_rotation' or 'stock_rotation' - determines which table to update
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Determine which table to update based on signal type
            if signal_type == 'stock_rotation':
                table_name = 'stock_saved_strategy'
            else:  # Default to ETF_rotation
                table_name = 'etf_saved_strategy'
            
            # Update accumulation_weeks only if it's greater than 0
            cursor.execute(f'''
                UPDATE {table_name}
                SET accumulation_weeks = accumulation_weeks - 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND run_id = ? AND accumulation_weeks > 0
            ''', (user_id, run_id))
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            if rows_affected > 0:
                logger.info(f"✅ Updated accumulation_weeks for user_id={user_id}, run_id={run_id} in {table_name}")
                return True
            else:
                logger.warning(f"⚠️  No rows updated for user_id={user_id}, run_id={run_id} in {table_name} (accumulation_weeks may already be 0 or less)")
                return False
                
        except Exception as e:
            table_name = 'etf_saved_strategy' if signal_type != 'stock_rotation' else 'stock_saved_strategy'
            logger.error(f"❌ Error updating accumulation_weeks in {table_name}: {e}")
            return False

