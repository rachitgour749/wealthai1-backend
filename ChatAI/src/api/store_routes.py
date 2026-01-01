"""
Store Management API Endpoints

Provides endpoints to:
- Create File Search stores
- Upload files to stores
- Check store status
"""

import os
import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stores", tags=["Store Management"])

# Get client from environment
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY not configured")
    return genai.Client(api_key=api_key)


NAMESPACE = "finai_prod"


@router.post("/products/create")
async def create_products_store():
    """Create or verify the shared products File Search store."""
    client = get_gemini_client()
    store_name = f"{NAMESPACE}_shared_products"
    
    try:
        store = await client.aio.file_search_stores.create(
            name=store_name,
            display_name="Financial Products Knowledge Base"
        )
        return {"status": "created", "store_name": store_name}
    except Exception as e:
        if "already exists" in str(e).lower():
            return {"status": "exists", "store_name": store_name}
        logger.error(f"Store creation failed: {e}")
        raise HTTPException(500, f"Failed to create store: {str(e)}")


@router.post("/products/upload")
async def upload_to_products_store(
    file: UploadFile = File(...),
    display_name: str = Form(None)
):
    """Upload a file to the products File Search store."""
    client = get_gemini_client()
    store_name = f"{NAMESPACE}_shared_products"
    
    # Save uploaded file temporarily
    temp_path = None
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Upload to File Search
        await client.aio.files.upload_to_file_search_store(
            file_search_store=store_name,
            file_path=temp_path,
            display_name=display_name or file.filename
        )
        
        return {
            "status": "uploaded",
            "filename": file.filename,
            "store": store_name
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Failed to upload: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/tenants/{tenant_id}/create")
async def create_tenant_store(tenant_id: str):
    """Create or verify a tenant's client data store."""
    client = get_gemini_client()
    store_name = f"{NAMESPACE}_{tenant_id}_clients"
    
    try:
        await client.aio.file_search_stores.create(
            name=store_name,
            display_name=f"Client Data - {tenant_id}"
        )
        return {"status": "created", "store_name": store_name, "tenant_id": tenant_id}
    except Exception as e:
        if "already exists" in str(e).lower():
            return {"status": "exists", "store_name": store_name, "tenant_id": tenant_id}
        logger.error(f"Tenant store creation failed: {e}")
        raise HTTPException(500, f"Failed to create tenant store: {str(e)}")


@router.get("/status")
async def get_stores_status():
    """Get status of all known stores."""
    client = get_gemini_client()
    
    stores_to_check = [
        f"{NAMESPACE}_shared_products",
    ]
    
    status = {}
    for store_name in stores_to_check:
        try:
            store = await client.aio.file_search_stores.get(name=store_name)
            status[store_name] = {"exists": True, "display_name": store.display_name}
        except Exception as e:
            status[store_name] = {"exists": False, "error": str(e)}
    
    return status
