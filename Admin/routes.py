from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List
from Admin.schemas import (
    ClientListItem, ClientDetailResponse, UpdateClientRequest,
    UpdatePlanRequest, UpdateSubscriptionDatesRequest,
    UpdateCreditsRequest, AdminOperationResponse
)
from Admin.services import admin_service

admin_router = APIRouter(prefix="/admin", tags=["Administration"])

@admin_router.get("/clients", response_model=List[ClientListItem])
async def list_clients():
    """List all registered clients"""
    return admin_service.get_all_clients()

@admin_router.get("/clients/{email}", response_model=ClientDetailResponse)
async def get_client(email: str):
    """Get full details for a specific client"""
    client = admin_service.get_client_details(email)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return client

@admin_router.put("/clients/{email}", response_model=AdminOperationResponse)
async def update_client(email: str, request: UpdateClientRequest):
    """Update basic client information (name, phone, status, role)"""
    success = admin_service.update_client_data(email, request)
    if not success:
        raise HTTPException(status_code=404, detail="Client not found or update failed")
    return AdminOperationResponse(success=True, message="Client updated successfully")

@admin_router.put("/clients/{email}/plan", response_model=AdminOperationResponse)
async def update_client_plan(email: str, request: UpdatePlanRequest):
    """Update complete subscription plan for a client"""
    result = admin_service.update_client_plan(email, request.plan_code)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "Plan update failed"))
    return AdminOperationResponse(success=True, message="Plan updated successfully", data=result)

@admin_router.put("/clients/{email}/dates", response_model=AdminOperationResponse)
async def update_client_dates(email: str, request: UpdateSubscriptionDatesRequest):
    """Update subscription start/end dates for a specific product"""
    success = admin_service.update_client_dates(email, request)
    if not success:
        raise HTTPException(status_code=404, detail="Client or product not found")
    return AdminOperationResponse(success=True, message="Subscription dates updated successfully")

@admin_router.put("/clients/{email}/credits", response_model=AdminOperationResponse)
async def update_client_credits(email: str, request: UpdateCreditsRequest):
    """Increase or decrease user tokens/credits"""
    result = admin_service.update_user_credits(email, request)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "Credit update failed"))
    return AdminOperationResponse(
        success=True, 
        message=result.get("message"), 
        data=result.get("data")
    )
