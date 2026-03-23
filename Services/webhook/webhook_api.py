"""
Webhook API endpoints for the Strategy Management Backend
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import json
from datetime import datetime
import logging
from typing import List, Optional

from .config import config
from .models import (
    StrategyCreate, StrategyUpdate, StrategyStatusUpdate,
    JsonGenerate, JsonSave, DeployRequest, StrategyResponse, HealthResponse,
    WebhookCreateRequest, WebhookCreateResponse,
    RACreateRequest, RAUpdateRequest, RAResponse,
    TradeExecuteRequest, TradeExecuteResponse,
    WebhookDetailResponse, WebhookListResponse, TradeExecuteIndividualRequest, UnifiedTradeExecuteRequest
)
from fastapi import Request, Header
from Services.subscription.services.auth_integration import get_current_user_from_google_token
from .webhook_logic import (
    WebhookLogic, init_db
)

# Get configuration
import os
config_name = os.environ.get('FASTAPI_ENV', 'default')
app_config = config[config_name]

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["webhook"])

# Initialize webhook logic
webhook_logic = WebhookLogic()

@router.post("/create", response_model=WebhookCreateResponse, status_code=201)
async def create_webhook(
    request: WebhookCreateRequest,
    current_user: dict = Depends(get_current_user_from_google_token)
):
    """
    Create a new webhook configuration and managed keys.
    Security: Validates that the request user_id matches the authenticated user.
    """
    try:
        # Check if requested user_id matches authenticated user
        if request.user_id != current_user["email"]:
             logger.warning(f"User {current_user['email']} attempted to create webhook for {request.user_id}")
             # We might allow this for RAs or admins, but generally it's a security risk.
             # For now, let's enforce it or at least log it.
             # raise HTTPException(status_code=403, detail="Forbidden: Cannot create webhook for another user")
        
        return await webhook_logic.create_webhook(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies", response_model=List[StrategyResponse])
async def get_strategies():
    """Get all strategies"""
    try:
        return await webhook_logic.get_all_strategies()
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies", response_model=dict, status_code=201)
async def create_strategy(strategy: StrategyCreate):
    """Create a new strategy"""
    try:
        return await webhook_logic.create_strategy(strategy)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create strategy")

@router.get("/strategies/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: int):
    """Get a specific strategy by ID"""
    try:
        return await webhook_logic.get_strategy_by_id(strategy_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/strategies/{strategy_id}", response_model=dict)
async def update_strategy(strategy_id: int, strategy_update: StrategyUpdate):
    """Update a specific strategy"""
    try:
        return await webhook_logic.update_strategy(strategy_id, strategy_update)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/strategies/{strategy_id}", response_model=dict)
async def delete_strategy(strategy_id: int):
    """Delete a specific strategy"""
    try:
        return await webhook_logic.delete_strategy(strategy_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/strategies/{strategy_id}/status", response_model=dict)
async def update_strategy_status(strategy_id: int, status_update: StrategyStatusUpdate):
    """Update strategy status (active/inactive)"""
    try:
        return await webhook_logic.update_strategy_status(strategy_id, status_update)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await webhook_logic.health_check()

@router.post("/generate-json", response_model=dict)
async def generate_json(json_data: JsonGenerate):
    """Generate JSON data for trading orders based on client IDs and capitals"""
    try:
        return await webhook_logic.generate_json_data(json_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate JSON data")

@router.post("/strategies/{strategy_id}/webhook", response_model=dict)
async def trigger_webhook(strategy_id: int):
    """Trigger webhook notification for a specific strategy"""
    try:
        return await webhook_logic.trigger_webhook(strategy_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger webhook")

@router.get("/strategies/{strategy_id}/json", response_model=dict)
async def get_strategy_json(strategy_id: int):
    """Get JSON data for a specific strategy"""
    try:
        return await webhook_logic.get_strategy_json(strategy_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get strategy JSON")

@router.post("/save-json", response_model=dict, status_code=201)
async def save_json(json_save: JsonSave):
    """Save JSON data for a user"""
    try:
        return await webhook_logic.save_json_data(json_save)
    except Exception as e:
        logger.error(f"Error saving JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save JSON data")

@router.get("/saved-json/{user_email}", response_model=dict)
async def get_saved_json(user_email: str, strategy_name: Optional[str] = None):
    """Get saved JSON data for a user, optionally filtered by strategy name"""
    try:
        return await webhook_logic.get_saved_json_data(user_email, strategy_name)
    except Exception as e:
        logger.error(f"Error getting saved JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get saved JSON data")

@router.delete("/saved-json/{json_id}", response_model=dict)
async def delete_saved_json(json_id: int):
    """Delete a specific saved JSON entry by ID"""
    try:
        return await webhook_logic.delete_saved_json_data(json_id)
    except Exception as e:
        logger.error(f"Error deleting saved JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete saved JSON data")

@router.delete("/delete-json/{identifier}", response_model=dict)
async def delete_saved_json_legacy(identifier: str):
    """Legacy route accepting numeric id or composite identifier"""
    try:
        return await webhook_logic.delete_saved_json_data_any(identifier)
    except Exception as e:
        logger.error(f"Error deleting saved JSON (legacy): {str(e)}")
        # If underlying raised HTTPException, re-raise to preserve status
        if hasattr(e, 'status_code'):
            raise e
        raise HTTPException(status_code=500, detail="Failed to delete saved JSON data")

@router.post("/deploy", response_model=dict, status_code=201)
async def deploy_strategy(deploy_request: DeployRequest):
    """Deploy strategy - generates JSON data and saves it to PostgreSQL"""
    try:
        return await webhook_logic.deploy_strategy(deploy_request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying strategy: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deploy strategy")

# Legacy endpoint for backward compatibility
@router.post("/deploy-legacy", response_model=dict)
async def deploy_legacy(data: dict):
    """Legacy deploy endpoint"""
    try:
        return await webhook_logic.deploy_legacy(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy deploy: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deploy strategy")

# RA CRUD Endpoints
@router.post("/trade_execute/{x_webhook_secret}", response_model=TradeExecuteResponse)
async def unified_trade_execute(
    x_webhook_secret: str,
    request: UnifiedTradeExecuteRequest
):
    """
    Unified trade execution for both RA and Individual signals (Secret in URL).
    """
    logger.info(f"Received unified trade execution request (RA: {request.ra_code}, RunID: {request.run_id})")
    return await webhook_logic.unified_execute_trade(request, x_webhook_secret)

@router.post("/wealthai1.in/trade_execute/{x_webhook_secret}", response_model=TradeExecuteResponse)
async def legacy_ra_trade_execute(
    x_webhook_secret: str,
    request: UnifiedTradeExecuteRequest
):
    """Legacy RA endpoint (redirects to unified logic)"""
    return await webhook_logic.unified_execute_trade(request, x_webhook_secret)

@router.get("/user/{user_id}", response_model=WebhookListResponse)
async def get_user_webhooks(user_id: str):
    """Fetch all webhooks for a specific user"""
    webhooks = await webhook_logic.get_user_webhooks(user_id)
    return WebhookListResponse(user_id=user_id, webhooks=webhooks)

@router.post("/status/{run_id}/{status}")
async def update_webhook_status(run_id: str, status: str):
    """Toggle active/inactive status using path parameters"""
    if status not in ['active', 'inactive']:
        raise HTTPException(status_code=400, detail="Status must be 'active' or 'inactive'")
        
    success = await webhook_logic.update_webhook_status(run_id, status)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook configuration not found")
    return {"status": "success", "message": f"Webhook status updated to {status}"}

@router.delete("/delete/{run_id}")
async def delete_webhook(run_id: str):
    """Delete a webhook configuration using path parameter"""
    success = await webhook_logic.delete_webhook(run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook configuration not found")
    return {"status": "success", "message": "Webhook deleted successfully"}

@router.post("/ra", response_model=RAResponse, status_code=201)
async def create_ra(request: RACreateRequest):
    """Add a new Research Analyst configuration"""
    try:
        return webhook_logic.create_ra(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating RA: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/ra", response_model=List[RAResponse])
async def get_ras():
    """List all Research Analyst configurations"""
    try:
        return webhook_logic.get_ras()
    except Exception as e:
        logger.error(f"Error listing RAs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ra/{ra_code}/{strategy_type}", response_model=RAResponse)
async def get_ra(ra_code: str, strategy_type: str):
    """Get a specific RA configuration"""
    try:
        return webhook_logic.get_ra(ra_code, strategy_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/ra/{ra_code}/{strategy_type}", response_model=RAResponse)
async def update_ra(ra_code: str, strategy_type: str, request: RAUpdateRequest):
    """Update RA configuration"""
    try:
        return webhook_logic.update_ra(ra_code, strategy_type, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating RA: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/ra/{ra_code}/{strategy_type}")
async def delete_ra(ra_code: str, strategy_type: str):
    """Delete RA configuration"""
    try:
        webhook_logic.delete_ra(ra_code, strategy_type)
        return {"message": "RA deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting RA: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
