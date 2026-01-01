"""
Admin API Routes for Customer and Document Management

Provides endpoints for:
- Customer management (CRUD operations)
- Document management (upload, list, sync)
- Sync operations (product docs, Zoho data)

All endpoints require X-Admin-Key header for authentication.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============== Admin Authentication ==============

async def verify_admin_key(x_admin_key: str = Header(..., description="Admin API key for authentication")):
    """Verify admin API key from header."""
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        logger.warning("ADMIN_API_KEY not configured - admin routes are unprotected!")
        return True  # Allow access if key not configured (development mode)
    
    if x_admin_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return True


# Router with authentication dependency on all routes
router = APIRouter(
    prefix="/admin", 
    tags=["Admin"],
    dependencies=[Depends(verify_admin_key)]
)

# ============== Data Models ==============

class CustomerCreate(BaseModel):
    name: str
    email: str
    api_key: str  # Customer's Gemini API key
    zoho_org_id: Optional[str] = None
    zoho_client_id: Optional[str] = None
    zoho_client_secret: Optional[str] = None

class Customer(BaseModel):
    id: str
    name: str
    email: str
    status: str  # active, inactive, pending
    file_search_store: Optional[str] = None
    docs_synced: int = 0
    last_sync: Optional[str] = None
    created_at: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    category: str  # health_insurance, life_insurance, mutual_fund, etc.
    size_bytes: int
    uploaded_at: str

class SyncStatus(BaseModel):
    customer_id: str
    status: str  # pending, in_progress, completed, failed
    docs_synced: int
    total_docs: int
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

# ============== File Paths ==============

# ============== File Paths ==============

# Get the directory of the current file (admin_routes.py)
CURRENT_FILE_DIR = Path(__file__).resolve().parent
# Go up 1 level to reach ChatAI root (Services/ChatAI/api/ -> Services/ChatAI/)
CHATAI_ROOT = CURRENT_FILE_DIR.parent
DATA_DIR = CHATAI_ROOT / "data"

CUSTOMERS_FILE = DATA_DIR / "customers.json"
DOCUMENTS_DIR = DATA_DIR / "documents"
PRODUCTS_DIR = DATA_DIR / "products"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ============== Helper Functions ==============

def load_customers() -> dict:
    """Load customers from JSON file."""
    if CUSTOMERS_FILE.exists():
        with open(CUSTOMERS_FILE, 'r') as f:
            return json.load(f)
    return {"customers": []}

def save_customers(data: dict):
    """Save customers to JSON file."""
    CUSTOMERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOMERS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def generate_id(prefix: str) -> str:
    """Generate unique ID."""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# ============== Customer Endpoints ==============

@router.get("/customers", response_model=List[Customer])
async def list_customers():
    """List all registered customers."""
    data = load_customers()
    return data.get("customers", [])

@router.post("/customers", response_model=Customer)
async def create_customer(customer: CustomerCreate):
    """Register a new customer."""
    data = load_customers()
    
    # Check for duplicate email
    for c in data["customers"]:
        if c["email"] == customer.email:
            raise HTTPException(400, "Customer with this email already exists")
    
    new_customer = Customer(
        id=generate_id("cust"),
        name=customer.name,
        email=customer.email,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    # Store API key securely (in production, use Secret Manager)
    api_keys_file = DATA_DIR / "api_keys.json"
    api_keys = {}
    if api_keys_file.exists():
        with open(api_keys_file, 'r') as f:
            api_keys = json.load(f)
    
    api_keys[new_customer.id] = {
        "gemini_api_key": customer.api_key,
        "zoho_org_id": customer.zoho_org_id,
        "zoho_client_id": customer.zoho_client_id,
        "zoho_client_secret": customer.zoho_client_secret
    }
    
    with open(api_keys_file, 'w') as f:
        json.dump(api_keys, f, indent=2)
    
    data["customers"].append(new_customer.model_dump())
    save_customers(data)
    
    logger.info(f"Created customer: {new_customer.id} - {new_customer.name}")
    return new_customer

@router.get("/customers/{customer_id}", response_model=Customer)
async def get_customer(customer_id: str):
    """Get customer details."""
    data = load_customers()
    for c in data["customers"]:
        if c["id"] == customer_id:
            return c
    raise HTTPException(404, "Customer not found")

@router.delete("/customers/{customer_id}")
async def delete_customer(customer_id: str):
    """Delete a customer."""
    data = load_customers()
    data["customers"] = [c for c in data["customers"] if c["id"] != customer_id]
    save_customers(data)
    return {"message": "Customer deleted"}

# ============== Document Endpoints ==============

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(category: Optional[str] = None):
    """List all uploaded documents."""
    documents = []
    
    # Walk through products directory
    for file_path in PRODUCTS_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.md', '.txt']:
            # Determine category from path
            relative_path = file_path.relative_to(PRODUCTS_DIR)
            parts = relative_path.parts
            doc_category = parts[0] if len(parts) > 1 else "general"
            
            if category and doc_category.lower() != category.lower():
                continue
            
            doc = DocumentInfo(
                id=str(file_path),
                filename=file_path.name,
                category=doc_category,
                size_bytes=file_path.stat().st_size,
                uploaded_at=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            )
            documents.append(doc)
    
    return documents

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(...),
    auto_sync: bool = Form(default=True)
):
    """
    Upload a new document to the products repository.
    
    If auto_sync is True (default), also uploads to File Search store
    for immediate querying.
    """
    import time
    from google import genai
    
    # Create category directory if needed
    category_dir = PRODUCTS_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = category_dir / file.filename
    content = await file.read()
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    logger.info(f"Uploaded document: {file.filename} to {category}")
    
    result = {
        "message": "Document uploaded successfully",
        "path": str(file_path),
        "category": category,
        "size_bytes": len(content),
        "file_search_synced": False
    }
    
    # Auto-sync to File Search if enabled
    if auto_sync:
        try:
            # Read store name
            store_file = DATA_DIR.parent / ".store_name"
            if store_file.exists():
                with open(store_file, 'r') as f:
                    store_name = f.read().strip()
                
                # Initialize client with master API key
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    client = genai.Client(api_key=api_key)
                    
                    # Upload to File Search
                    display_name = f"{category} - {file.filename}"
                    operation = client.file_search_stores.upload_to_file_search_store(
                        file=str(file_path),
                        file_search_store_name=store_name,
                        config={'display_name': display_name[:100]}
                    )
                    
                    # Wait for completion (max 60 seconds)
                    wait_time = 0
                    while not operation.done and wait_time < 60:
                        time.sleep(2)
                        wait_time += 2
                        operation = client.operations.get(operation)
                    
                    if operation.done:
                        result["file_search_synced"] = True
                        result["message"] = "Document uploaded and synced to File Search"
                        logger.info(f"Synced to File Search: {file.filename}")
                    else:
                        logger.warning(f"File Search sync timeout for {file.filename}")
                else:
                    logger.warning("No GEMINI_API_KEY for auto-sync")
            else:
                logger.warning("No .store_name file found for auto-sync")
        except Exception as e:
            logger.error(f"Auto-sync failed: {e}")
            result["sync_error"] = str(e)
    
    return result


@router.delete("/documents/{document_path:path}")
async def delete_document(document_path: str):
    """Delete a document."""
    file_path = Path(document_path)
    if file_path.exists():
        file_path.unlink()
        return {"message": "Document deleted"}
    raise HTTPException(404, "Document not found")

@router.get("/documents/categories")
async def list_categories():
    """List all document categories."""
    categories = set()
    for item in PRODUCTS_DIR.iterdir():
        if item.is_dir():
            categories.add(item.name)
    return sorted(list(categories))

# ============== Sync Endpoints ==============

@router.post("/sync/customer/{customer_id}")
async def sync_customer(customer_id: str, background_tasks: BackgroundTasks):
    """Sync all product documents to a specific customer."""
    data = load_customers()
    customer = None
    for c in data["customers"]:
        if c["id"] == customer_id:
            customer = c
            break
    
    if not customer:
        raise HTTPException(404, "Customer not found")
    
    # Start background sync task
    background_tasks.add_task(sync_docs_to_customer, customer_id)
    
    return {
        "message": "Sync started",
        "customer_id": customer_id,
        "status": "in_progress"
    }

@router.post("/sync/all")
async def sync_all_customers(background_tasks: BackgroundTasks):
    """Sync product documents to all active customers."""
    data = load_customers()
    active_customers = [c for c in data["customers"] if c.get("status") == "active"]
    
    for customer in active_customers:
        background_tasks.add_task(sync_docs_to_customer, customer["id"])
    
    return {
        "message": "Sync started for all active customers",
        "customer_count": len(active_customers)
    }

@router.get("/sync/status/{customer_id}")
async def get_sync_status(customer_id: str):
    """Get sync status for a customer."""
    sync_file = DATA_DIR / "sync_status" / f"{customer_id}.json"
    if sync_file.exists():
        with open(sync_file, 'r') as f:
            return json.load(f)
    return {"status": "no_sync_history"}

# ============== Background Tasks ==============

async def sync_docs_to_customer(customer_id: str):
    """Background task to sync documents to a customer's File Search store."""
    from google import genai
    from google.genai import types
    
    logger.info(f"Starting sync for customer: {customer_id}")
    
    sync_dir = DATA_DIR / "sync_status"
    sync_dir.mkdir(parents=True, exist_ok=True)
    sync_file = sync_dir / f"{customer_id}.json"
    
    # Load customer API key
    api_keys_file = DATA_DIR / "api_keys.json"
    if not api_keys_file.exists():
        logger.error(f"API keys file not found for customer {customer_id}")
        return
    
    with open(api_keys_file, 'r') as f:
        api_keys = json.load(f)
    
    customer_keys = api_keys.get(customer_id)
    if not customer_keys:
        logger.error(f"No API key for customer {customer_id}")
        return
    
    # Get list of documents to sync
    docs = []
    for file_path in PRODUCTS_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.md', '.txt']:
            docs.append(file_path)
    
    # Update sync status
    status = {
        "customer_id": customer_id,
        "status": "in_progress",
        "docs_synced": 0,
        "total_docs": len(docs),
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None
    }
    with open(sync_file, 'w') as f:
        json.dump(status, f)
    
    try:
        # Initialize Gemini client with customer's API key
        client = genai.Client(api_key=customer_keys["gemini_api_key"])
        
        # Get or create File Search store for customer
        data = load_customers()
        customer = next((c for c in data["customers"] if c["id"] == customer_id), None)
        
        if not customer.get("file_search_store"):
            # Create new store
            store = client.file_search_stores.create(
                config={'display_name': f'Products - {customer["name"]}'}
            )
            customer["file_search_store"] = store.name
            save_customers(data)
        
        store_name = customer["file_search_store"]
        
        # Upload documents
        synced = 0
        for doc_path in docs:
            try:
                operation = client.file_search_stores.upload_to_file_search_store(
                    file=str(doc_path),
                    file_search_store_name=store_name,
                    config={'display_name': doc_path.name}
                )
                
                # Wait for upload
                import time
                wait_time = 0
                while not operation.done and wait_time < 60:
                    time.sleep(2)
                    wait_time += 2
                    operation = client.operations.get(operation)
                
                synced += 1
                status["docs_synced"] = synced
                with open(sync_file, 'w') as f:
                    json.dump(status, f)
                    
            except Exception as e:
                logger.warning(f"Failed to sync {doc_path.name}: {e}")
        
        # Update customer record
        customer["docs_synced"] = synced
        customer["last_sync"] = datetime.now().isoformat()
        customer["status"] = "active"
        save_customers(data)
        
        # Update final status
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat()
        with open(sync_file, 'w') as f:
            json.dump(status, f)
        
        logger.info(f"Sync completed for {customer_id}: {synced}/{len(docs)} docs")
        
    except Exception as e:
        logger.error(f"Sync failed for {customer_id}: {e}")
        status["status"] = "failed"
        status["error"] = str(e)
        with open(sync_file, 'w') as f:
            json.dump(status, f)


# ============== Zoho Sync Endpoints ==============

@router.post("/sync/zoho/{customer_id}")
async def sync_zoho_data(customer_id: str, background_tasks: BackgroundTasks):
    """Trigger Zoho data sync for a specific customer."""
    data = load_customers()
    customer = next((c for c in data["customers"] if c["id"] == customer_id), None)
    
    if not customer:
        raise HTTPException(404, "Customer not found")
    
    # Check if Zoho is configured
    api_keys_file = DATA_DIR / "api_keys.json"
    if api_keys_file.exists():
        with open(api_keys_file, 'r') as f:
            api_keys = json.load(f)
        
        customer_keys = api_keys.get(customer_id, {})
        if not customer_keys.get('zoho_org_id'):
            raise HTTPException(400, "Zoho not configured for this customer")
    
    # Start background Zoho sync
    background_tasks.add_task(run_zoho_sync_for_customer, customer_id)
    
    return {
        "message": "Zoho sync started",
        "customer_id": customer_id,
        "status": "in_progress"
    }


@router.post("/sync/zoho/all")
async def sync_all_zoho_data(background_tasks: BackgroundTasks):
    """Trigger Zoho data sync for all customers with Zoho configured."""
    from Services.ChatAI.sync.zoho_batch_sync import run_monthly_zoho_sync_for_all
    
    background_tasks.add_task(run_monthly_zoho_sync_for_all)
    
    return {
        "message": "Zoho sync started for all configured customers",
        "status": "in_progress"
    }


@router.get("/sync/zoho/status/{customer_id}")
async def get_zoho_sync_status(customer_id: str):
    """Get Zoho sync status for a customer."""
    sync_file = DATA_DIR / "sync_status" / f"{customer_id}_zoho.json"
    if sync_file.exists():
        with open(sync_file, 'r') as f:
            return json.load(f)
    return {"status": "no_zoho_sync_history"}


async def run_zoho_sync_for_customer(customer_id: str):
    """Background task to run Zoho sync."""
    from Services.ChatAI.sync.zoho_batch_sync import ZohoBatchSync
    
    try:
        sync = ZohoBatchSync(customer_id)
        await sync.run_full_sync()
    except Exception as e:
        logger.error(f"Zoho sync failed for {customer_id}: {e}")
        
        # Save error status
        sync_dir = DATA_DIR / "sync_status"
        sync_dir.mkdir(parents=True, exist_ok=True)
        
        with open(sync_dir / f"{customer_id}_zoho.json", 'w') as f:
            json.dump({
                "customer_id": customer_id,
                "status": "failed",
                "error": str(e)
            }, f)

