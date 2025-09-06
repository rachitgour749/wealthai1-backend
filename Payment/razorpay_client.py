import razorpay
import hmac
import hashlib
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from Payment.config import PaymentConfig
from Payment.models import PaymentStatus, PaymentMethod
from Payment.database import db_manager

class RazorpayClient:
    """Razorpay client wrapper for payment operations"""
    
    def __init__(self):
        """Initialize Razorpay client"""
        if not PaymentConfig.validate_config():
            raise ValueError("Invalid Razorpay configuration")
        
        self.client = razorpay.Client(auth=PaymentConfig.get_razorpay_auth())
        self.config = PaymentConfig
    
    def create_order(self, order_data: Dict[str, Any], customer_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new payment order
        
        Args:
            order_data: Order data including amount, currency, etc.
            customer_info: Customer information for database storage
            
        Returns:
            Order details from Razorpay
        """
        try:
            # Prepare order data - only valid parameters for Razorpay order creation
            order_params = {
                "amount": order_data["amount"],
                "currency": order_data.get("currency", self.config.DEFAULT_CURRENCY),
                "receipt": order_data.get("receipt", f"receipt_{int(time.time())}"),
                "payment_capture": self.config.PAYMENT_CAPTURE,
                "notes": order_data.get("notes", {})
            }
            
            # Note: prefill, callback_url, and cancel_url are not allowed in order creation
            # They are used in checkout options instead
            
            # Create order with Razorpay
            order = self.client.order.create(order_params)
            
            # Store order in database
            db_order_data = {
                "id": str(uuid.uuid4()),
                "razorpay_order_id": order["id"],
                "amount": order["amount"],
                "currency": order["currency"],
                "receipt": order["receipt"],
                "status": order["status"],
                "customer_name": customer_info.get("name") if customer_info else None,
                "customer_email": customer_info.get("email") if customer_info else None,
                "customer_contact": customer_info.get("contact") if customer_info else None,
                "notes": str(order_data.get("notes", {}))
            }
            
            db_manager.create_order(db_order_data)
            
            return order
            
        except Exception as e:
            raise Exception(f"Failed to create order: {str(e)}")
    
    def verify_payment_signature(self, order_id: str, payment_id: str, signature: str) -> bool:
        """
        Verify payment signature
        
        Args:
            order_id: Razorpay order ID
            payment_id: Razorpay payment ID
            signature: Payment signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Create signature body
            body = f"{order_id}|{payment_id}"
            
            # Generate expected signature
            expected_signature = hmac.new(
                self.config.RAZORPAY_KEY_SECRET.encode(),
                body.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(expected_signature, signature)
            
        except Exception as e:
            raise Exception(f"Failed to verify signature: {str(e)}")
    
    def capture_payment(self, payment_id: str, amount: Optional[int] = None) -> Dict[str, Any]:
        """
        Capture a payment
        
        Args:
            payment_id: Razorpay payment ID
            amount: Amount to capture (full amount if not specified)
            
        Returns:
            Capture response from Razorpay
        """
        try:
            capture_data = {"payment_id": payment_id}
            if amount:
                capture_data["amount"] = amount
            
            capture = self.client.payment.capture(**capture_data)
            
            # Update transaction status in database
            db_manager.update_transaction_status(payment_id, "captured")
            
            return capture
            
        except Exception as e:
            raise Exception(f"Failed to capture payment: {str(e)}")
    
    def get_payment_details(self, payment_id: str) -> Dict[str, Any]:
        """
        Get payment details
        
        Args:
            payment_id: Razorpay payment ID
            
        Returns:
            Payment details from Razorpay
        """
        try:
            payment = self.client.payment.fetch(payment_id)
            return payment
        except Exception as e:
            raise Exception(f"Failed to fetch payment details: {str(e)}")
    
    def get_order_details(self, order_id: str) -> Dict[str, Any]:
        """
        Get order details
        
        Args:
            order_id: Razorpay order ID
            
        Returns:
            Order details from Razorpay
        """
        try:
            order = self.client.order.fetch(order_id)
            return order
        except Exception as e:
            raise Exception(f"Failed to fetch order details: {str(e)}")
    
    def get_payment_methods(self) -> List[Dict[str, Any]]:
        """
        Get available payment methods
        
        Returns:
            List of available payment methods
        """
        try:
            # Razorpay doesn't have a direct payment_method.all() method
            # Return common payment methods that Razorpay supports
            common_methods = [
                {
                    "id": "card",
                    "name": "Credit/Debit Card",
                    "description": "Visa, MasterCard, RuPay, American Express",
                    "enabled": True
                },
                {
                    "id": "netbanking",
                    "name": "Net Banking",
                    "description": "All major Indian banks",
                    "enabled": True
                },
                {
                    "id": "upi",
                    "name": "UPI",
                    "description": "Unified Payments Interface",
                    "enabled": True
                },
                {
                    "id": "wallet",
                    "name": "Digital Wallets",
                    "description": "Paytm, PhonePe, Amazon Pay",
                    "enabled": True
                },
                {
                    "id": "emi",
                    "name": "EMI",
                    "description": "Credit Card EMI",
                    "enabled": True
                }
            ]
            return common_methods
        except Exception as e:
            raise Exception(f"Failed to fetch payment methods: {str(e)}")
    
    def create_refund(self, payment_id: str, refund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a refund
        
        Args:
            payment_id: Razorpay payment ID
            refund_data: Refund data including amount, notes, etc.
            
        Returns:
            Refund response from Razorpay
        """
        try:
            refund_params = {
                "payment_id": payment_id,
                "speed": refund_data.get("speed", "normal")
            }
            
            if "amount" in refund_data:
                refund_params["amount"] = refund_data["amount"]
            if "notes" in refund_data:
                refund_params["notes"] = refund_data["notes"]
            if "receipt" in refund_data:
                refund_params["receipt"] = refund_data["receipt"]
            
            refund = self.client.refund.create(refund_params)
            
            # Store refund in database
            db_refund_data = {
                "id": str(uuid.uuid4()),
                "payment_id": payment_id,
                "amount": refund["amount"],
                "currency": refund["currency"],
                "status": refund["status"],
                "speed": refund_data.get("speed", "normal"),
                "notes": str(refund_data.get("notes", {})),
                "receipt": refund.get("receipt", "")
            }
            
            db_manager.create_refund(db_refund_data)
            
            return refund
            
        except Exception as e:
            raise Exception(f"Failed to create refund: {str(e)}")
    
    def get_refund_details(self, refund_id: str) -> Dict[str, Any]:
        """
        Get refund details
        
        Args:
            refund_id: Razorpay refund ID
            
        Returns:
            Refund details from Razorpay
        """
        try:
            refund = self.client.refund.fetch(refund_id)
            return refund
        except Exception as e:
            raise Exception(f"Failed to fetch refund details: {str(e)}")
    
    def process_webhook(self, webhook_data: Dict[str, Any], signature: str) -> Dict[str, Any]:
        """
        Process webhook from Razorpay
        
        Args:
            webhook_data: Webhook payload
            signature: Webhook signature
            
        Returns:
            Processing result
        """
        try:
            # Verify webhook signature
            if not self.verify_webhook_signature(webhook_data, signature):
                raise Exception("Invalid webhook signature")
            
            event = webhook_data.get("event")
            payload = webhook_data.get("payload", {})
            
            if event == "payment.captured":
                return self._handle_payment_captured(payload)
            elif event == "payment.failed":
                return self._handle_payment_failed(payload)
            elif event == "refund.processed":
                return self._handle_refund_processed(payload)
            else:
                return {"status": "ignored", "event": event}
                
        except Exception as e:
            raise Exception(f"Failed to process webhook: {str(e)}")
    
    def verify_webhook_signature(self, webhook_data: Dict[str, Any], signature: str) -> bool:
        """
        Verify webhook signature
        
        Args:
            webhook_data: Webhook payload
            signature: Webhook signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            if not self.config.WEBHOOK_SECRET:
                return True  # Skip verification if no secret configured
            
            # Create signature body
            body = webhook_data.get("entity", "") + "|" + webhook_data.get("event", "")
            
            # Generate expected signature
            expected_signature = hmac.new(
                self.config.WEBHOOK_SECRET.encode(),
                body.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(expected_signature, signature)
            
        except Exception:
            return False
    
    def _handle_payment_captured(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment captured webhook"""
        try:
            payment = payload.get("payment", {})
            payment_id = payment.get("id")
            
            if payment_id:
                # Update transaction status
                db_manager.update_transaction_status(payment_id, "captured")
                
                # Store transaction in database if not exists
                if not db_manager.get_transaction_by_payment_id(payment_id):
                    transaction_data = {
                        "id": str(uuid.uuid4()),
                        "order_id": payment.get("order_id", ""),
                        "razorpay_payment_id": payment_id,
                        "amount": payment.get("amount", 0),
                        "currency": payment.get("currency", "INR"),
                        "status": "captured",
                        "method": payment.get("method"),
                        "bank": payment.get("bank"),
                        "wallet": payment.get("wallet"),
                        "card_id": payment.get("card_id"),
                        "vpa": payment.get("vpa"),
                        "email": payment.get("email"),
                        "contact": payment.get("contact"),
                        "fee": payment.get("fee"),
                        "tax": payment.get("tax")
                    }
                    
                    db_manager.create_transaction(transaction_data)
            
            return {"status": "success", "event": "payment.captured"}
            
        except Exception as e:
            return {"status": "error", "event": "payment.captured", "error": str(e)}
    
    def _handle_payment_failed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment failed webhook"""
        try:
            payment = payload.get("payment", {})
            payment_id = payment.get("id")
            
            if payment_id:
                # Update transaction status
                db_manager.update_transaction_status(payment_id, "failed")
                
                # Store transaction in database if not exists
                if not db_manager.get_transaction_by_payment_id(payment_id):
                    transaction_data = {
                        "id": str(uuid.uuid4()),
                        "order_id": payment.get("order_id", ""),
                        "razorpay_payment_id": payment_id,
                        "amount": payment.get("amount", 0),
                        "currency": payment.get("currency", "INR"),
                        "status": "failed",
                        "method": payment.get("method"),
                        "error_code": payment.get("error_code"),
                        "error_description": payment.get("error_description")
                    }
                    
                    db_manager.create_transaction(transaction_data)
            
            return {"status": "success", "event": "payment.failed"}
            
        except Exception as e:
            return {"status": "error", "event": "payment.failed", "error": str(e)}
    
    def _handle_refund_processed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle refund processed webhook"""
        try:
            refund = payload.get("refund", {})
            refund_id = refund.get("id")
            
            if refund_id:
                # Update refund status in database
                # This would require additional database methods
                pass
            
            return {"status": "success", "event": "refund.processed"}
            
        except Exception as e:
            return {"status": "error", "event": "refund.processed", "error": str(e)}
    
    def get_payment_link(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a payment link
        
        Args:
            payment_data: Payment data including amount, description, etc.
            
        Returns:
            Payment link details from Razorpay
        """
        try:
            link_params = {
                "amount": payment_data["amount"],
                "currency": payment_data.get("currency", self.config.DEFAULT_CURRENCY),
                "description": payment_data.get("description", "Payment"),
                "reference_id": payment_data.get("reference_id", f"ref_{int(time.time())}"),
                "callback_url": payment_data.get("callback_url"),
                "callback_method": "get"
            }
            
            if "customer" in payment_data:
                link_params["customer"] = payment_data["customer"]
            
            payment_link = self.client.payment_link.create(link_params)
            return payment_link
            
        except Exception as e:
            raise Exception(f"Failed to create payment link: {str(e)}")
    
    def get_settlements(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get settlements
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of settlements
        """
        try:
            params = {}
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date
            
            settlements = self.client.settlement.all(params)
            return settlements.get("items", [])
            
        except Exception as e:
            raise Exception(f"Failed to fetch settlements: {str(e)}")

# Global Razorpay client instance
razorpay_client = RazorpayClient()
