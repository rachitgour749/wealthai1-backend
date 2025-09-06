from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks

from typing import Dict, Any, Optional, List
import logging
import time

from Payment.models import (
    OrderRequest, OrderResponse, PaymentVerificationRequest, 
    PaymentVerificationResponse, RefundRequest, RefundResponse,
    CustomerInfo, PaymentPlan, PaymentHistoryItem, PaymentAnalytics
)
from Payment.razorpay_client import razorpay_client
from Payment.database import db_manager
from Payment.config import PaymentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/payment", tags=["Payment"])

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for payment service"""
    try:
        # Test Razorpay connection
        methods = razorpay_client.get_payment_methods()
        return {
            "status": "healthy",
            "service": "payment",
            "razorpay_connected": True,
            "environment": PaymentConfig.ENVIRONMENT
        }
    except Exception as e:
        logger.error(f"Payment service health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "payment",
            "razorpay_connected": False,
            "error": str(e)
        }

# Order management endpoints
@router.post("/order", response_model=OrderResponse)
async def create_payment_order(request: OrderRequest):
    """
    Create a new payment order
    
    This endpoint creates a payment order with Razorpay and returns
    the order details needed for frontend integration.
    """
    try:
        logger.info(f"Creating payment order for amount: {request.amount}")
        
        # Prepare order data - only include valid order parameters
        order_data = {
            "amount": request.amount,
            "currency": request.currency,
            "receipt": request.receipt,
            "notes": request.notes or {}
        }
        
        # Store customer info for database (not sent to Razorpay order creation)
        customer_info = request.customer if request.customer else {}
        
        # Create order with customer info passed separately
        order = razorpay_client.create_order(order_data, customer_info)
        
        # Ensure all required fields are present for the response model
        order_response = {
            "id": order.get("id"),
            "entity": order.get("entity", "order"),
            "amount": order.get("amount"),
            "amount_paid": order.get("amount_paid", 0),
            "amount_due": order.get("amount_due", order.get("amount", 0)),
            "currency": order.get("currency"),
            "receipt": order.get("receipt"),
            "offer_id": order.get("offer_id"),
            "status": order.get("status"),
            "attempts": order.get("attempts", 0),
            "notes": order.get("notes", {}),
            "created_at": order.get("created_at"),
            "updated_at": order.get("updated_at")
        }
        
        logger.info(f"Payment order created successfully: {order['id']}")
        return order_response
        
    except Exception as e:
        logger.error(f"Failed to create payment order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")

@router.get("/order/{order_id}")
async def get_order_details(order_id: str):
    """
    Get order details by order ID
    """
    try:
        order = razorpay_client.get_order_details(order_id)
        return order
    except Exception as e:
        logger.error(f"Failed to fetch order details: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Order not found: {str(e)}")

# Payment verification endpoints
@router.post("/verify", response_model=PaymentVerificationResponse)
async def verify_payment_signature(request: PaymentVerificationRequest):
    """
    Verify payment signature
    
    This endpoint verifies the payment signature returned by Razorpay
    to ensure the payment is legitimate.
    """
    try:
        logger.info(f"Verifying payment signature for order: {request.razorpay_order_id}")
        
        # Verify signature
        is_verified = razorpay_client.verify_payment_signature(
            request.razorpay_order_id,
            request.razorpay_payment_id,
            request.razorpay_signature
        )
        
        if is_verified:
            # Get payment details
            payment_details = razorpay_client.get_payment_details(request.razorpay_payment_id)
            
            logger.info(f"Payment verified successfully: {request.razorpay_payment_id}")
            
            return PaymentVerificationResponse(
                verified=True,
                order_id=request.razorpay_order_id,
                payment_id=request.razorpay_payment_id,
                amount=payment_details.get("amount", 0),
                currency=payment_details.get("currency", "INR"),
                status=payment_details.get("status", "unknown"),
                message="Payment verified successfully"
            )
        else:
            logger.warning(f"Payment signature verification failed: {request.razorpay_payment_id}")
            
            return PaymentVerificationResponse(
                verified=False,
                order_id=request.razorpay_order_id,
                payment_id=request.razorpay_payment_id,
                amount=0,
                currency="INR",
                status="failed",
                message="Payment signature verification failed"
            )
            
    except Exception as e:
        logger.error(f"Payment verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
# Payment capture endpoints
@router.post("/capture/{payment_id}")
async def capture_payment(payment_id: str, amount: Optional[int] = None):
    """
    Capture a payment
    
    This endpoint captures an authorized payment. If no amount is specified,
    the full authorized amount is captured.
    """
    try:
        logger.info(f"Capturing payment: {payment_id}, amount: {amount}")
        
        capture_result = razorpay_client.capture_payment(payment_id, amount)
        
        logger.info(f"Payment captured successfully: {payment_id}")
        return {
            "status": "success",
            "message": "Payment captured successfully",
            "capture_id": capture_result.get("id"),
            "amount": capture_result.get("amount"),
            "currency": capture_result.get("currency")
        }
        
    except Exception as e:
        logger.error(f"Payment capture failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Capture failed: {str(e)}")

# Refund endpoints
@router.post("/refund", response_model=RefundResponse)
async def create_refund(request: RefundRequest):
    """
    Create a refund for a payment
    
    This endpoint creates a refund for a captured payment.
    """
    try:
        logger.info(f"Creating refund for payment: {request.payment_id}")
        
        refund_data = {
            "amount": request.amount,
            "speed": request.speed,
            "notes": request.notes,
            "receipt": request.receipt
        }
        
        refund = razorpay_client.create_refund(request.payment_id, refund_data)
        
        logger.info(f"Refund created successfully: {refund['id']}")
        return refund
        
    except Exception as e:
        logger.error(f"Refund creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Refund failed: {str(e)}")

@router.get("/refund/{refund_id}")
async def get_refund_details(refund_id: str):
    """
    Get refund details by refund ID
    """
    try:
        refund = razorpay_client.get_refund_details(refund_id)
        return refund
    except Exception as e:
        logger.error(f"Failed to fetch refund details: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Refund not found: {str(e)}")

# Payment link endpoints
@router.post("/link")
async def create_payment_link(request: Dict[str, Any]):
    """
    Create a payment link
    
    This endpoint creates a payment link that can be shared with customers.
    """
    try:
        logger.info(f"Creating payment link for amount: {request.get('amount')}")
        
        payment_link = razorpay_client.get_payment_link(request)
        
        logger.info(f"Payment link created successfully: {payment_link['id']}")
        return payment_link
        
    except Exception as e:
        logger.error(f"Payment link creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Link creation failed: {str(e)}")

# Payment methods endpoint
@router.get("/methods")
async def get_payment_methods():
    """
    Get available payment methods
    
    This endpoint returns all available payment methods from Razorpay.
    """
    try:
        methods = razorpay_client.get_payment_methods()
        return {
            "payment_methods": methods,
            "count": len(methods)
        }
    except Exception as e:
        logger.error(f"Failed to fetch payment methods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch methods: {str(e)}")

# Payment history endpoints
@router.get("/history")
async def get_payment_history(limit: int = 100, offset: int = 0):
    """
    Get payment history
    
    This endpoint returns the payment transaction history.
    """
    try:
        history = db_manager.get_payment_history(limit, offset)
        return {
            "history": history,
            "count": len(history),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to fetch payment history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

# Payment analytics endpoints
@router.get("/analytics")
async def get_payment_analytics(period: str = "monthly"):
    """
    Get payment analytics
    
    This endpoint returns payment statistics and analytics.
    """
    try:
        analytics = db_manager.get_payment_analytics(period)
        return analytics
    except Exception as e:
        logger.error(f"Failed to fetch payment analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")

# Webhook endpoint
@router.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Razorpay webhooks
    
    This endpoint processes webhooks from Razorpay for payment status updates.
    """
    try:
        # Get webhook data
        webhook_data = await request.json()
        signature = request.headers.get("X-Razorpay-Signature", "")
        
        logger.info(f"Received webhook: {webhook_data.get('event')}")
        
        # Process webhook in background
        background_tasks.add_task(
            razorpay_client.process_webhook,
            webhook_data,
            signature
        )
        
        return {"status": "webhook_received"}
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Webhook processing failed: {str(e)}")

# Customer management endpoints
@router.post("/customer")
async def create_customer(customer_data: CustomerInfo):
    """
    Create a new customer
    
    This endpoint creates a customer record for payment tracking.
    """
    try:
        # This would integrate with your customer management system
        # For now, we'll just return success
        return {
            "status": "success",
            "message": "Customer created successfully",
            "customer_id": "cust_" + str(int(time.time()))
        }
    except Exception as e:
        logger.error(f"Customer creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Customer creation failed: {str(e)}")

# Payment plan endpoints
@router.post("/plan")
async def create_payment_plan(plan_data: PaymentPlan):
    """
    Create a payment plan
    
    This endpoint creates a recurring payment plan.
    """
    try:
        # This would integrate with your subscription management system
        return {
            "status": "success",
            "message": "Payment plan created successfully",
            "plan_id": "plan_" + str(int(time.time()))
        }
    except Exception as e:
        logger.error(f"Payment plan creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plan creation failed: {str(e)}")

# Settlement endpoints
@router.get("/settlements")
async def get_settlements(from_date: Optional[str] = None, to_date: Optional[str] = None):
    """
    Get settlements
    
    This endpoint returns settlement information from Razorpay.
    """
    try:
        settlements = razorpay_client.get_settlements(from_date, to_date)
        return {
            "settlements": settlements,
            "count": len(settlements),
            "from_date": from_date,
            "to_date": to_date
        }
    except Exception as e:
        logger.error(f"Failed to fetch settlements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch settlements: {str(e)}")

# Note: Exception handlers should be added to the main FastAPI app, not to APIRouter
# Global exception handling is handled by the main server.py file

# Initialize payment service
def init_payment_service():
    """Initialize payment service and database"""
    try:
        # Initialize database tables
        db_manager.init_database()
        logger.info("Payment service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize payment service: {str(e)}")
        return False

# Cleanup payment service
def cleanup_payment_service():
    """Cleanup payment service resources"""
    try:
        # Close database connections
        if hasattr(db_manager, 'engine'):
            db_manager.engine.dispose()
        logger.info("Payment service cleanup completed")
    except Exception as e:
        logger.error(f"Payment service cleanup failed: {str(e)}")

# Export router and initialization functions
payment_router = router

