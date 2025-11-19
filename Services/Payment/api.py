from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks

from typing import Dict, Any, Optional, List
import logging
import time

from .models import (
    OrderRequest, OrderResponse, PaymentVerificationRequest, 
    PaymentVerificationResponse, RefundRequest, RefundResponse,
    CustomerInfo, PaymentPlan, PaymentHistoryItem, PaymentAnalytics
)
from .database import db_manager
from .config import PaymentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/payment", tags=["Payment"])


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

