import re
import hashlib
import hmac
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from Payment.models import PaymentStatus, PaymentMethod

class PaymentUtils:
    """Utility functions for payment operations"""
    
    @staticmethod
    def validate_amount(amount: int, min_amount: int = 100, max_amount: int = 100000000) -> bool:
        """
        Validate payment amount
        
        Args:
            amount: Amount in paise
            min_amount: Minimum amount in paise (default: ₹1)
            max_amount: Maximum amount in paise (default: ₹10,00,000)
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(amount, int):
            return False
        
        if amount < min_amount or amount > max_amount:
            return False
        
        return True
    
    @staticmethod
    def validate_currency(currency: str) -> bool:
        """
        Validate currency code
        
        Args:
            currency: Currency code (e.g., 'INR', 'USD')
            
        Returns:
            True if valid, False otherwise
        """
        valid_currencies = ['INR', 'USD', 'EUR', 'GBP', 'SGD', 'AED']
        return currency.upper() in valid_currencies
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number (Indian format)
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Remove spaces and special characters
        phone = re.sub(r'[\s\-\(\)]', '', phone)
        
        # Check if it's a valid Indian mobile number
        pattern = r'^(\+91|91)?[6-9]\d{9}$'
        return re.match(pattern, phone) is not None
    
    @staticmethod
    def format_amount(amount: int, currency: str = "INR") -> str:
        """
        Format amount for display
        
        Args:
            amount: Amount in paise
            currency: Currency code
            
        Returns:
            Formatted amount string
        """
        if currency == "INR":
            rupees = amount / 100
            return f"₹{rupees:,.2f}"
        else:
            # For other currencies, assume amount is in smallest unit
            return f"{amount:,.2f} {currency}"
    
    @staticmethod
    def parse_amount(amount_str: str, currency: str = "INR") -> int:
        """
        Parse amount string to paise
        
        Args:
            amount_str: Amount string (e.g., "₹500.00", "500")
            currency: Currency code
            
        Returns:
            Amount in paise
        """
        try:
            # Remove currency symbols and commas
            clean_amount = re.sub(r'[₹$€£,]', '', amount_str).strip()
            
            # Parse as float and convert to paise
            amount_float = float(clean_amount)
            
            if currency == "INR":
                return int(amount_float * 100)
            else:
                # For other currencies, assume amount is in smallest unit
                return int(amount_float)
                
        except (ValueError, TypeError):
            raise ValueError(f"Invalid amount format: {amount_str}")
    
    @staticmethod
    def generate_receipt_id(prefix: str = "RCPT") -> str:
        """
        Generate unique receipt ID
        
        Args:
            prefix: Receipt ID prefix
            
        Returns:
            Unique receipt ID
        """
        timestamp = int(time.time())
        random_suffix = hashlib.md5(f"{timestamp}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    @staticmethod
    def generate_order_id(prefix: str = "ORD") -> str:
        """
        Generate unique order ID
        
        Args:
            prefix: Order ID prefix
            
        Returns:
            Unique order ID
        """
        timestamp = int(time.time())
        random_suffix = hashlib.md5(f"{timestamp}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    @staticmethod
    def sanitize_customer_data(customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize customer data for security
        
        Args:
            customer_data: Raw customer data
            
        Returns:
            Sanitized customer data
        """
        sanitized = {}
        
        # Sanitize name
        if "name" in customer_data:
            sanitized["name"] = str(customer_data["name"]).strip()[:100]
        
        # Sanitize email
        if "email" in customer_data:
            email = str(customer_data["email"]).strip().lower()
            if PaymentUtils.validate_email(email):
                sanitized["email"] = email
        
        # Sanitize phone
        if "contact" in customer_data:
            phone = str(customer_data["contact"]).strip()
            if PaymentUtils.validate_phone(phone):
                sanitized["contact"] = phone
        
        # Sanitize address fields
        address_fields = ["address", "city", "state", "zipcode", "country"]
        for field in address_fields:
            if field in customer_data:
                sanitized[field] = str(customer_data[field]).strip()[:200]
        
        return sanitized
    
    @staticmethod
    def validate_payment_method(method: str) -> bool:
        """
        Validate payment method
        
        Args:
            method: Payment method string
            
        Returns:
            True if valid, False otherwise
        """
        valid_methods = [method.value for method in PaymentMethod]
        return method in valid_methods
    
    @staticmethod
    def get_payment_method_display_name(method: str) -> str:
        """
        Get display name for payment method
        
        Args:
            method: Payment method string
            
        Returns:
            Display name for payment method
        """
        method_names = {
            "card": "Credit/Debit Card",
            "upi": "UPI",
            "netbanking": "Net Banking",
            "wallet": "Digital Wallet",
            "emi": "EMI",
            "cardless_emi": "Cardless EMI"
        }
        return method_names.get(method, method.title())
    
    @staticmethod
    def calculate_gst(amount: int, gst_rate: float = 18.0) -> int:
        """
        Calculate GST amount
        
        Args:
            amount: Base amount in paise
            gst_rate: GST rate percentage
            
        Returns:
            GST amount in paise
        """
        gst_amount = (amount * gst_rate) / 100
        return int(gst_amount)
    
    @staticmethod
    def calculate_total_with_gst(amount: int, gst_rate: float = 18.0) -> int:
        """
        Calculate total amount including GST
        
        Args:
            amount: Base amount in paise
            gst_rate: GST rate percentage
            
        Returns:
            Total amount including GST in paise
        """
        gst_amount = PaymentUtils.calculate_gst(amount, gst_rate)
        return amount + gst_amount
    
    @staticmethod
    def format_timestamp(timestamp: int) -> str:
        """
        Format Unix timestamp to readable date
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Formatted date string
        """
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def get_time_difference(timestamp1: int, timestamp2: int) -> str:
        """
        Get time difference between two timestamps
        
        Args:
            timestamp1: First timestamp
            timestamp2: Second timestamp
            
        Returns:
            Formatted time difference string
        """
        dt1 = datetime.fromtimestamp(timestamp1)
        dt2 = datetime.fromtimestamp(timestamp2)
        
        diff = abs(dt2 - dt1)
        
        if diff.days > 0:
            return f"{diff.days} days"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hours"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minutes"
        else:
            return f"{diff.seconds} seconds"
    
    @staticmethod
    def is_payment_expired(created_at: int, expiry_minutes: int = 30) -> bool:
        """
        Check if payment has expired
        
        Args:
            created_at: Payment creation timestamp
            expiry_minutes: Expiry time in minutes
            
        Returns:
            True if expired, False otherwise
        """
        current_time = int(time.time())
        expiry_time = created_at + (expiry_minutes * 60)
        return current_time > expiry_time
    
    @staticmethod
    def get_payment_status_color(status: str) -> str:
        """
        Get color code for payment status
        
        Args:
            status: Payment status
            
        Returns:
            Color code (hex)
        """
        status_colors = {
            "pending": "#FFA500",      # Orange
            "authorized": "#4169E1",   # Royal Blue
            "captured": "#32CD32",     # Lime Green
            "failed": "#DC143C",       # Crimson
            "cancelled": "#808080",    # Gray
            "refunded": "#FF6347",     # Tomato
            "partially_refunded": "#FF8C00"  # Dark Orange
        }
        return status_colors.get(status, "#000000")
    
    @staticmethod
    def generate_payment_summary(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate payment summary from transactions
        
        Args:
            transactions: List of payment transactions
            
        Returns:
            Payment summary statistics
        """
        if not transactions:
            return {
                "total_transactions": 0,
                "total_amount": 0,
                "successful_amount": 0,
                "failed_amount": 0,
                "pending_amount": 0,
                "success_rate": 0.0
            }
        
        total_transactions = len(transactions)
        total_amount = sum(t.get("amount", 0) for t in transactions)
        successful_amount = sum(t.get("amount", 0) for t in transactions if t.get("status") == "captured")
        failed_amount = sum(t.get("amount", 0) for t in transactions if t.get("status") == "failed")
        pending_amount = sum(t.get("amount", 0) for t in transactions if t.get("status") == "pending")
        
        success_rate = (successful_amount / total_amount * 100) if total_amount > 0 else 0.0
        
        return {
            "total_transactions": total_transactions,
            "total_amount": total_amount,
            "successful_amount": successful_amount,
            "failed_amount": failed_amount,
            "pending_amount": pending_amount,
            "success_rate": round(success_rate, 2)
        }
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = "*") -> str:
        """
        Mask sensitive data like card numbers, UPI IDs
        
        Args:
            data: Data to mask
            mask_char: Character to use for masking
            
        Returns:
            Masked data string
        """
        if not data or len(data) < 4:
            return data
        
        # For card numbers, show first 4 and last 4 digits
        if len(data) > 8:
            return data[:4] + mask_char * (len(data) - 8) + data[-4:]
        else:
            # For shorter data, mask middle part
            return data[0] + mask_char * (len(data) - 2) + data[-1]
    
    @staticmethod
    def validate_webhook_signature(payload: str, signature: str, secret: str) -> bool:
        """
        Validate webhook signature
        
        Args:
            payload: Webhook payload
            signature: Webhook signature
            secret: Webhook secret
            
        Returns:
            True if valid, False otherwise
        """
        try:
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, signature)
        except Exception:
            return False
    
    @staticmethod
    def create_webhook_signature(payload: str, secret: str) -> str:
        """
        Create webhook signature for testing
        
        Args:
            payload: Webhook payload
            secret: Webhook secret
            
        Returns:
            Generated signature
        """
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
