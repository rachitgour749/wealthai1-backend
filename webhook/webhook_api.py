"""
Webhook API endpoints for the Strategy Management Backend
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import sqlite3
import json
from datetime import datetime
import logging
from typing import List, Optional

from webhook.config import config
from webhook.models import (
    StrategyCreate, StrategyUpdate, StrategyStatusUpdate,
    JsonGenerate, JsonSave, StrategyResponse, HealthResponse
)
from webhook.webhook_logic import (
    WebhookLogic, get_db_connection, init_db
)

# Get configuration
import os
config_name = os.environ.get('FASTAPI_ENV', 'default')
app_config = config[config_name]

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["webhook"])

# Initialize webhook logic
webhook_logic = WebhookLogic()

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
async def get_saved_json(user_email: str):
    """Get all saved JSON data for a user"""
    try:
        return await webhook_logic.get_saved_json_data(user_email)
    except Exception as e:
        logger.error(f"Error getting saved JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get saved JSON data")

# Legacy endpoint for backward compatibility
@router.post("/deploy", response_model=dict)
async def deploy_legacy(data: dict):
    """Legacy deploy endpoint"""
    try:
        return await webhook_logic.deploy_legacy(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy deploy: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deploy strategy")
