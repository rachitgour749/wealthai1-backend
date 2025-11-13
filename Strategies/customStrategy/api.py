from fastapi import APIRouter, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from .models import (
    CustomStrategyRequest, 
    CustomStrategyResponse, 
    CustomStrategySaveRequest,
    CustomStrategyListResponse,
    CustomStrategyAnalysis
)
from .database import CustomStrategyDatabase
from .ai_service import AIService
from .email_service import EmailService
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
custom_strategy_router = APIRouter(prefix="/api/custom-strategy", tags=["Custom Strategy"])

# Initialize services
db_service = CustomStrategyDatabase()
ai_service = AIService()
email_service = EmailService()


@custom_strategy_router.post("/analyze", response_model=CustomStrategyResponse)
async def analyze_strategy(request: CustomStrategyRequest):
    """Analyze user's strategy description using AI"""
    try:
        logger.info(f"Analyzing strategy for user: {request.user_email}")
        
        # Analyze strategy using AI service
        analysis = ai_service.analyze_strategy(request.strategy_description)
        
        if not analysis:
            return CustomStrategyResponse(
                success=False,
                message="Failed to analyze strategy. Please try again.",
                error="AI analysis failed"
            )
        
        # Validate analysis
        if not ai_service.validate_analysis(analysis):
            return CustomStrategyResponse(
                success=False,
                message="Invalid analysis received. Please try again.",
                error="Invalid analysis format"
            )
        
        # Convert to CustomStrategyAnalysis model
        try:
            # Handle risk_management - convert dict to string if needed
            risk_mgmt = analysis.get('risk_management', '')
            if isinstance(risk_mgmt, dict):
                # Convert dict to formatted string
                risk_items = []
                for key, value in risk_mgmt.items():
                    if value and value.strip() and not value.strip() == '(assumed)':
                        risk_items.append(f"{key}: {value}")
                risk_mgmt = '; '.join(risk_items) if risk_items else 'Not specified'
            elif not risk_mgmt or risk_mgmt.strip() == '':
                risk_mgmt = 'Not specified'
            
            # Handle strategy_logic - convert list to string if needed
            strategy_logic = analysis.get('strategy_logic', '')
            if isinstance(strategy_logic, list):
                strategy_logic = '\n'.join(strategy_logic)
            elif not strategy_logic or strategy_logic.strip() == '':
                strategy_logic = 'Not specified'
            
            analysis_model = CustomStrategyAnalysis(
                strategy_rating=analysis['strategy_levels'],
                trading_instruments=analysis.get('trading_instruments', 'ANY INSTRUMENT'),
                time_frame=analysis['time_frame'],
                trading_rules=analysis['trading_rules'],
                position_sizing=analysis.get('position_sizing', 'Not specified'),
                risk_management=risk_mgmt,
                strategy_logic=strategy_logic
            )
        except Exception as e:
            logger.error(f"Error creating analysis model: {e}")
            return CustomStrategyResponse(
                success=False,
                message="Error processing analysis. Please try again.",
                error=str(e)
            )
        
        return CustomStrategyResponse(
            success=True,
            analysis=analysis_model,
            message="Strategy analyzed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_strategy: {e}")
        return CustomStrategyResponse(
            success=False,
            message="Internal server error. Please try again.",
            error=str(e)
        )

@custom_strategy_router.post("/save", response_model=CustomStrategyResponse)
async def save_strategy(
    request: CustomStrategySaveRequest, 
    background_tasks: BackgroundTasks
):
    """Save custom strategy and send email notifications"""
    try:
        logger.info(f"Saving strategy for user: {request.user_email}")
        
        # Convert analysis model to dict
        analysis_dict = request.analysis.dict()
        
        # Save strategy to database using email from request
        strategy_id = db_service.save_strategy(
            user_email=request.user_email,
            user_phone=request.user_phone,
            strategy_description=request.strategy_description,
            analysis=analysis_dict
        )
        
        # Get saved strategy data for email
        strategy_data = db_service.get_strategy_by_id(strategy_id)
        
        if not strategy_data:
            return CustomStrategyResponse(
                success=False,
                message="Strategy saved but could not retrieve data for notifications.",
                error="Database retrieval failed"
            )
        
        # Send email notifications in background
        background_tasks.add_task(send_notifications_background, strategy_data)
        
        return CustomStrategyResponse(
            success=True,
            strategy_id=strategy_id,
            message="Strategy saved successfully. Email notifications will be sent shortly."
        )
        
    except Exception as e:
        logger.error(f"Error in save_strategy: {e}")
        return CustomStrategyResponse(
            success=False,
            message="Failed to save strategy. Please try again.",
            error=str(e)
        )

@custom_strategy_router.get("/user/{user_email}", response_model=CustomStrategyListResponse)
async def get_user_strategies(user_email: str):
    """Get all custom strategies for a user by email"""
    try:
        logger.info(f"Getting strategies for user: {user_email}")
        
        strategies = db_service.get_strategies_by_email(user_email)
        
        return CustomStrategyListResponse(
            success=True,
            strategies=strategies,
            count=len(strategies)
        )
        
    except Exception as e:
        logger.error(f"Error in get_user_strategies: {e}")
        return CustomStrategyListResponse(
            success=False,
            strategies=[],
            count=0
        )

@custom_strategy_router.get("/{strategy_id}")
async def get_strategy_details(strategy_id: int):
    """Get specific strategy details"""
    try:
        logger.info(f"Getting strategy details for ID: {strategy_id}")
        
        strategy = db_service.get_strategy_by_id(strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "success": True,
            "strategy": strategy
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_strategy_details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@custom_strategy_router.put("/{strategy_id}/status")
async def update_strategy_status(strategy_id: int, status: str):
    """Update strategy status"""
    try:
        logger.info(f"Updating strategy {strategy_id} status to: {status}")
        
        valid_statuses = ['pending', 'reviewing', 'approved', 'rejected', 'implemented']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        success = db_service.update_strategy_status(strategy_id, status)
        
        if not success:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "success": True,
            "message": f"Strategy status updated to {status}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update_strategy_status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@custom_strategy_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "custom-strategy",
        "timestamp": "2024-01-01T00:00:00Z"
    }

async def send_notifications_background(strategy_data: Dict[str, Any]):
    """Background task to send email notifications"""
    try:
        logger.info(f"Sending notifications for strategy ID: {strategy_data.get('id')}")
        
        results = email_service.send_strategy_notifications(strategy_data)
        
        logger.info(f"Email notification results: {results}")
        
    except Exception as e:
        logger.error(f"Error sending notifications: {e}")
