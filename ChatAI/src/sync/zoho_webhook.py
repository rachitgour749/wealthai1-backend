"""
Zoho Webhook Handler for Financial Advisor Chatbot

Receives real-time contact updates from Zoho CRM and syncs to File Search.
"""

import logging
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from src.core.error_handling import dlq, RetryHandler
from src.sync.transform import transform_contact_to_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks/zoho", tags=["Zoho Webhooks"])


class ZohoWebhookPayload(BaseModel):
    """Expected payload from Zoho webhook."""
    event_type: str  # create, update, delete
    module: str = "Contacts"
    data: dict = {}


@router.post("/contacts/{tenant_id}")
async def handle_zoho_contact_webhook(
    tenant_id: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle Zoho contact webhook.
    
    Zoho Webhook Configuration:
    - URL: https://your-domain.com/webhooks/zoho/contacts/{tenant_id}
    - Method: POST
    - Module: Contacts
    - Trigger: On Create, Edit, Delete
    
    Args:
        tenant_id: Tenant identifier
        request: FastAPI request
        background_tasks: Background task manager
    
    Returns:
        Processing status
    """
    try:
        payload = await request.json()
        logger.info(f"Received Zoho webhook for tenant {tenant_id}")
        
        # Attempt to process with retry
        await RetryHandler.with_retry(
            lambda: process_contact_webhook(tenant_id, payload),
            max_retries=2,
            base_delay=0.5
        )
        
        return {"status": "processed", "tenant_id": tenant_id}
    
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        
        # Queue for retry in background
        background_tasks.add_task(
            dlq.put,
            {
                "tenant_id": tenant_id,
                "payload": payload,
                "error": str(e)
            }
        )
        
        # Return 202 to acknowledge receipt even if processing failed
        raise HTTPException(202, detail="Queued for retry")


async def process_contact_webhook(tenant_id: str, payload: dict):
    """
    Process contact webhook payload.
    
    Args:
        tenant_id: Tenant identifier
        payload: Webhook payload from Zoho
    """
    from google import genai
    import os
    
    event_type = payload.get("event_type", "update")
    contact_data = payload.get("data", {})
    
    if not contact_data:
        logger.warning("Empty contact data in webhook payload")
        return
    
    # Transform to document format
    document = transform_contact_to_document(contact_data)
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Get store name
    from src.stores.tenant_manager import TenantStoreManager
    store_manager = TenantStoreManager(client)
    store_name = store_manager.get_client_store_name(tenant_id)
    
    if event_type == "delete":
        # Handle deletion - mark as inactive or remove
        logger.info(f"Contact deleted: {document['metadata'].get('client_id')}")
        # Note: File Search doesn't support deletion directly
        # Would need to track in metadata
    else:
        # Create or update
        await store_manager.upload_client_document(
            tenant_id=tenant_id,
            content=document["content"],
            metadata=document["metadata"]
        )
        logger.info(f"Synced contact: {document['metadata'].get('client_name')}")


@router.get("/status")
async def webhook_status():
    """Check webhook endpoint status and DLQ size."""
    return {
        "status": "active",
        "dlq_size": dlq.size(),
        "supported_events": ["create", "update", "delete"]
    }


@router.post("/retry-failed")
async def retry_failed_webhooks(background_tasks: BackgroundTasks):
    """Manually trigger retry of failed webhooks in DLQ."""
    if dlq.size() == 0:
        return {"status": "empty", "message": "No failed webhooks to retry"}
    
    background_tasks.add_task(
        dlq.process_retries,
        handler=lambda p: process_contact_webhook(p["tenant_id"], p["payload"])
    )
    
    return {"status": "processing", "queue_size": dlq.size()}
