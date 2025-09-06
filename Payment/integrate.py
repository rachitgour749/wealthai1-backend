#!/usr/bin/env python3
"""
Simple integration file for Payment system
Use this to include payment API in your main server.py
"""

from Payment.api import router as payment_router

def include_payment_in_server(app):
    """
    Include payment system in your main FastAPI application
    
    Usage in server.py:
        from fastapi import FastAPI
        from Payment.integrate import include_payment_in_server
        
        app = FastAPI()
        include_payment_in_server(app)
    """
    # Initialize payment database
    try:
        from Payment.database import db_manager
        db_manager.init_database()
    except Exception as e:
        print(f"Warning: Could not initialize payment database: {e}")
    
    app.include_router(payment_router, prefix="/api/payment")
    
    # Add payment health check to main health endpoint
    @app.get("/payment-health")
    async def payment_health():
        try:
            from Payment.razorpay_client import razorpay_client
            # Test Razorpay connection
            razorpay_client.get_payment_methods()
            return {
                "status": "healthy",
                "service": "payment",
                "razorpay_connected": True,
                "environment": "test"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "payment",
                "razorpay_connected": False,
                "error": str(e)
            }
