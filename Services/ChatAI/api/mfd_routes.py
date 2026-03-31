"""
MFD Self-Service API Routes

Allows Mutual Fund Distributors to:
- Create their own Google File Search Store
- Connect Zoho CRM via OAuth
- Sync contacts from Zoho CRM into their File Search Store
- Check sync status

These routes are authenticated by user email (x-user-email header),
NOT by admin key.
"""

import os
import json
import logging
import asyncio
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, BackgroundTasks, Request
from pydantic import BaseModel

import httpx
from google import genai

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

router = APIRouter(prefix="/api/mfd", tags=["MFD Self-Service"])


# ============== Models ==============

class CreateStoreRequest(BaseModel):
    display_name: str

class ZohoConnectRequest(BaseModel):
    client_id: str
    client_secret: str
    redirect_uri: Optional[str] = None
    zoho_domain: Optional[str] = "https://accounts.zoho.in"  # Default to IN

class ZohoCallbackRequest(BaseModel):
    code: str  # The authorization grant code from Zoho

class SyncRequest(BaseModel):
    module: Optional[str] = "Contacts"
    fields: Optional[str] = None  # Comma-separated field API names


# ============== Helpers ==============

def _load_customers():
    customers_file = DATA_DIR / "customers.json"
    if customers_file.exists():
        with open(customers_file, 'r') as f:
            return json.load(f)
    return {"customers": []}

def _save_customers(data):
    customers_file = DATA_DIR / "customers.json"
    customers_file.parent.mkdir(parents=True, exist_ok=True)
    with open(customers_file, 'w') as f:
        json.dump(data, f, indent=2)

def _load_api_keys():
    api_keys_file = DATA_DIR / "api_keys.json"
    if api_keys_file.exists():
        with open(api_keys_file, 'r') as f:
            return json.load(f)
    return {}

def _save_api_keys(data):
    api_keys_file = DATA_DIR / "api_keys.json"
    api_keys_file.parent.mkdir(parents=True, exist_ok=True)
    with open(api_keys_file, 'w') as f:
        json.dump(data, f, indent=2)

def _get_or_create_customer(email: str, name: str = ""):
    """Get customer by email, or create a new entry."""
    data = _load_customers()
    for customer in data.get("customers", []):
        if customer.get("email", "").lower() == email.lower():
            return customer, data
    
    # Create new customer
    import hashlib
    cust_id = f"cust_{hashlib.md5(email.encode()).hexdigest()[:8]}"
    customer = {
        "id": cust_id,
        "name": name or email.split("@")[0],
        "email": email,
        "status": "active",
        "file_search_store": None,
        "docs_synced": 0,
        "last_sync": None,
        "created_at": datetime.now().isoformat()
    }
    data["customers"].append(customer)
    _save_customers(data)
    return customer, data

def _get_gemini_client():
    """Get the platform Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback: read from api_keys.json
        api_keys = _load_api_keys()
        for cust_id, cust_keys in api_keys.items():
            if "gemini_api_key" in cust_keys:
                api_key = cust_keys["gemini_api_key"]
                break
    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY not configured on server")
    return genai.Client(api_key=api_key)


# ============== Profile Endpoint ==============

@router.get("/profile")
async def get_mfd_profile(x_user_email: str = Header(...)):
    """Get MFD's profile: store info, Zoho connection status, sync status."""
    email = x_user_email
    customer, _ = _get_or_create_customer(email)
    api_keys = _load_api_keys()
    cust_keys = api_keys.get(customer["id"], {})
    
    # Check Zoho connection
    zoho_connected = bool(cust_keys.get("zoho_refresh_token"))
    
    # Check sync status
    sync_file = DATA_DIR / "sync_status" / f"{customer['id']}_zoho.json"
    sync_status = None
    if sync_file.exists():
        with open(sync_file, 'r') as f:
            sync_status = json.load(f)
    
    # Use stored doc count from customer data (avoid slow API call)
    doc_count = customer.get("docs_synced", 0)
    store_display_name = None
    if customer.get("file_search_store"):
        try:
            client = _get_gemini_client()
            store = client.file_search_stores.get(name=customer["file_search_store"])
            store_display_name = getattr(store, "display_name", None)
        except Exception as e:
            logger.warning(f"Failed to get store info: {e}")
    
    return {
        "email": email,
        "customer_id": customer["id"],
        "name": customer["name"],
        "store": {
            "id": customer.get("file_search_store"),
            "display_name": store_display_name,
            "doc_count": doc_count,
        },
        "zoho": {
            "connected": zoho_connected,
            "domain": cust_keys.get("zoho_domain", "https://accounts.zoho.in"),
        },
        "last_sync": customer.get("last_sync"),
        "sync_status": sync_status,
    }


# ============== Store Management ==============

@router.post("/store/create")
async def create_store(body: CreateStoreRequest, x_user_email: str = Header(...)):
    """Create a new Google File Search Store for the MFD."""
    email = x_user_email
    customer, data = _get_or_create_customer(email)
    
    # Check if already has a store
    if customer.get("file_search_store"):
        return {
            "status": "exists",
            "store": customer["file_search_store"],
            "message": "You already have a store. Use the existing one."
        }
    
    try:
        client = _get_gemini_client()
        store = client.file_search_stores.create(
            config={"display_name": body.display_name}
        )
        
        # Update customer record
        for c in data["customers"]:
            if c["email"].lower() == email.lower():
                c["file_search_store"] = store.name
                break
        _save_customers(data)
        
        logger.info(f"Created store {store.name} for {email}")
        return {
            "status": "created",
            "store": store.name,
            "display_name": body.display_name,
            "message": "Store created successfully!"
        }
    except Exception as e:
        logger.error(f"Failed to create store for {email}: {e}")
        raise HTTPException(500, f"Failed to create store: {str(e)}")


# ============== Zoho OAuth ==============

@router.get("/zoho/auth-url")
async def get_zoho_auth_url(x_user_email: str = Header(...)):
    """Generate the Zoho OAuth authorization URL for the MFD to visit."""
    email = x_user_email
    customer, _ = _get_or_create_customer(email)
    api_keys = _load_api_keys()
    cust_keys = api_keys.get(customer["id"], {})
    
    client_id = cust_keys.get("zoho_client_id")
    if not client_id:
        raise HTTPException(400, "Zoho client_id not configured. Please connect Zoho first.")
    
    redirect_uri = cust_keys.get("zoho_redirect_uri", "http://localhost:5173/zoho-callback")
    zoho_domain = cust_keys.get("zoho_domain", "https://accounts.zoho.in")
    
    auth_url = (
        f"{zoho_domain}/oauth/v2/auth"
        f"?scope=ZohoCRM.modules.contacts.READ"
        f"&client_id={client_id}"
        f"&response_type=code"
        f"&access_type=offline"
        f"&redirect_uri={redirect_uri}"
        f"&prompt=consent"
    )
    
    return {"auth_url": auth_url, "redirect_uri": redirect_uri}


@router.post("/zoho/connect")
async def connect_zoho(body: ZohoConnectRequest, x_user_email: str = Header(...)):
    """Save Zoho OAuth credentials for the MFD."""
    email = x_user_email
    customer, _ = _get_or_create_customer(email)
    
    api_keys = _load_api_keys()
    cust_keys = api_keys.get(customer["id"], {})
    
    cust_keys["zoho_client_id"] = body.client_id
    cust_keys["zoho_client_secret"] = body.client_secret
    cust_keys["zoho_redirect_uri"] = body.redirect_uri or "http://localhost:5173/zoho-callback"
    cust_keys["zoho_domain"] = body.zoho_domain or "https://accounts.zoho.in"
    
    api_keys[customer["id"]] = cust_keys
    _save_api_keys(api_keys)
    
    # Generate auth URL
    redirect_uri = cust_keys["zoho_redirect_uri"]
    auth_url = (
        f"{cust_keys['zoho_domain']}/oauth/v2/auth"
        f"?scope=ZohoCRM.modules.contacts.READ"
        f"&client_id={body.client_id}"
        f"&response_type=code"
        f"&access_type=offline"
        f"&redirect_uri={redirect_uri}"
        f"&prompt=consent"
    )
    
    return {
        "status": "saved",
        "auth_url": auth_url,
        "message": "Zoho credentials saved. Visit the auth_url to authorize."
    }


@router.post("/zoho/callback")
async def zoho_callback(body: ZohoCallbackRequest, x_user_email: str = Header(...)):
    """Exchange Zoho authorization code for access & refresh tokens."""
    email = x_user_email
    customer, _ = _get_or_create_customer(email)
    
    api_keys = _load_api_keys()
    cust_keys = api_keys.get(customer["id"], {})
    
    client_id = cust_keys.get("zoho_client_id")
    client_secret = cust_keys.get("zoho_client_secret")
    redirect_uri = cust_keys.get("zoho_redirect_uri", "http://localhost:5173/zoho-callback")
    zoho_domain = cust_keys.get("zoho_domain", "https://accounts.zoho.in")
    
    if not client_id or not client_secret:
        raise HTTPException(400, "Zoho credentials not configured. Connect Zoho first.")
    
    # Exchange code for tokens
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{zoho_domain}/oauth/v2/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "code": body.code,
                },
                timeout=15.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(400, f"Zoho token exchange failed: {resp.text}")
            
            token_data = resp.json()
            
            if "error" in token_data:
                raise HTTPException(400, f"Zoho error: {token_data.get('error')}")
            
            # Save tokens
            cust_keys["zoho_access_token"] = token_data.get("access_token")
            cust_keys["zoho_refresh_token"] = token_data.get("refresh_token")
            cust_keys["zoho_api_domain"] = token_data.get("api_domain", "https://www.zohoapis.in")
            cust_keys["zoho_token_expires_at"] = (
                datetime.now().timestamp() + token_data.get("expires_in", 3600)
            )
            
            api_keys[customer["id"]] = cust_keys
            _save_api_keys(api_keys)
            
            return {
                "status": "connected",
                "message": "Zoho CRM connected successfully! You can now sync contacts."
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Zoho callback error: {e}")
        raise HTTPException(500, f"Failed to connect Zoho: {str(e)}")


async def _refresh_zoho_token(customer_id: str) -> Optional[str]:
    """Refresh Zoho access token using refresh token."""
    api_keys = _load_api_keys()
    cust_keys = api_keys.get(customer_id, {})
    
    refresh_token = cust_keys.get("zoho_refresh_token")
    if not refresh_token:
        return None
    
    zoho_domain = cust_keys.get("zoho_domain", "https://accounts.zoho.in")
    
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{zoho_domain}/oauth/v2/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": cust_keys["zoho_client_id"],
                    "client_secret": cust_keys["zoho_client_secret"],
                    "refresh_token": refresh_token,
                },
                timeout=15.0
            )
            
            if resp.status_code == 200:
                data = resp.json()
                cust_keys["zoho_access_token"] = data["access_token"]
                cust_keys["zoho_token_expires_at"] = (
                    datetime.now().timestamp() + data.get("expires_in", 3600)
                )
                api_keys[customer_id] = cust_keys
                _save_api_keys(api_keys)
                return data["access_token"]
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
    
    return None


async def _get_zoho_access_token(customer_id: str) -> str:
    """Get valid Zoho access token, refreshing if needed."""
    api_keys = _load_api_keys()
    cust_keys = api_keys.get(customer_id, {})
    
    access_token = cust_keys.get("zoho_access_token")
    expires_at = cust_keys.get("zoho_token_expires_at", 0)
    
    # Refresh if expired or about to expire (5 min buffer)
    if not access_token or datetime.now().timestamp() > expires_at - 300:
        access_token = await _refresh_zoho_token(customer_id)
        if not access_token:
            raise HTTPException(401, "Zoho token expired and refresh failed. Please reconnect Zoho.")
    
    return access_token


# ============== Sync Contacts ==============

# Track running syncs
_running_syncs = {}

@router.post("/zoho/sync")
async def trigger_zoho_sync(
    body: SyncRequest,
    background_tasks: BackgroundTasks,
    x_user_email: str = Header(...)
):
    """Trigger Zoho CRM contacts sync to File Search Store (runs in background)."""
    email = x_user_email
    customer, data = _get_or_create_customer(email)
    
    # Check prerequisites
    if not customer.get("file_search_store"):
        raise HTTPException(400, "No File Search Store created. Create one first.")
    
    api_keys = _load_api_keys()
    cust_keys = api_keys.get(customer["id"], {})
    
    if not cust_keys.get("zoho_refresh_token"):
        raise HTTPException(400, "Zoho CRM not connected. Connect Zoho first.")
    
    # Check if sync already running
    if _running_syncs.get(customer["id"]):
        return {"status": "already_running", "message": "A sync is already in progress."}
    
    # Start background sync
    _running_syncs[customer["id"]] = True
    background_tasks.add_task(
        _run_sync_task,
        customer["id"],
        customer.get("file_search_store"),
        body.module,
        body.fields
    )
    
    return {
        "status": "started",
        "message": f"Syncing {body.module} from Zoho CRM to your store. This may take a few minutes."
    }


async def _run_sync_task(customer_id: str, store_name: str, module: str, fields: Optional[str]):
    """Background task to sync Zoho contacts to File Search."""
    from Services.ChatAI.sync.transform import transform_contact_to_document
    
    sync_status = {
        "customer_id": customer_id,
        "started_at": datetime.now().isoformat(),
        "status": "in_progress",
        "module": module,
        "records_fetched": 0,
        "records_synced": 0,
        "records_failed": 0,
        "errors": []
    }
    
    sync_dir = DATA_DIR / "sync_status"
    sync_dir.mkdir(parents=True, exist_ok=True)
    sync_file = sync_dir / f"{customer_id}_zoho.json"
    
    def _save_status():
        with open(sync_file, 'w') as f:
            json.dump(sync_status, f, indent=2)
    
    _save_status()
    
    try:
        # Step 1: Get access token
        access_token = await _get_zoho_access_token(customer_id)
        api_keys = _load_api_keys()
        cust_keys = api_keys.get(customer_id, {})
        api_domain = cust_keys.get("zoho_api_domain", "https://www.zohoapis.in")
        
        # Step 2: Fetch records from Zoho CRM
        all_records = []
        page = 1
        per_page = 200
        
        # Default fields for Contacts module
        default_fields = (
            "Full_Name,First_Name,Last_Name,Email,Phone,Mobile,"
            "Account_Name,Title,Department,Description,"
            "Mailing_Street,Mailing_City,Mailing_State,Mailing_Zip,Mailing_Country,"
            "Created_Time,Modified_Time"
        )
        request_fields = fields or default_fields
        
        async with httpx.AsyncClient() as http:
            while True:
                try:
                    resp = await http.get(
                        f"{api_domain}/crm/v8/{module}",
                        headers={
                            "Authorization": f"Zoho-oauthtoken {access_token}",
                        },
                        params={
                            "fields": request_fields,
                            "page": page,
                            "per_page": per_page,
                            "sort_by": "Modified_Time",
                            "sort_order": "desc",
                        },
                        timeout=30.0
                    )
                    
                    if resp.status_code == 200:
                        resp_data = resp.json()
                        batch = resp_data.get("data", [])
                        all_records.extend(batch)
                        
                        info = resp_data.get("info", {})
                        if not info.get("more_records", False) or len(batch) < per_page:
                            break
                        page += 1
                    elif resp.status_code == 204:
                        # No records
                        break
                    else:
                        error_msg = f"Zoho API error {resp.status_code}: {resp.text[:200]}"
                        sync_status["errors"].append(error_msg)
                        logger.error(error_msg)
                        break
                        
                except Exception as e:
                    sync_status["errors"].append(f"Fetch error on page {page}: {str(e)}")
                    logger.error(f"Zoho fetch error: {e}")
                    break
        
        sync_status["records_fetched"] = len(all_records)
        _save_status()
        
        if not all_records:
            sync_status["status"] = "completed"
            sync_status["message"] = "No records found in Zoho CRM"
            sync_status["completed_at"] = datetime.now().isoformat()
            _save_status()
            _running_syncs.pop(customer_id, None)
            return
        
        # Step 3: Transform and upload to File Search Store
        gemini_client = _get_gemini_client()
        
        for i, record in enumerate(all_records):
            try:
                doc = transform_contact_to_document(record)
                
                # Write to temp file
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.txt', delete=False, encoding='utf-8'
                ) as f:
                    f.write(doc["content"])
                    temp_path = f.name
                
                # Upload to File Search Store
                display_name = doc["metadata"].get("client_name", f"Contact_{i+1}")
                operation = gemini_client.file_search_stores.upload_to_file_search_store(
                    file=temp_path,
                    file_search_store_name=store_name,
                    config={"display_name": display_name}
                )
                
                # Wait for completion
                wait_time = 0
                while not operation.done and wait_time < 60:
                    await asyncio.sleep(2)
                    wait_time += 2
                    operation = gemini_client.operations.get(operation)
                
                # Cleanup
                os.unlink(temp_path)
                
                if operation.done:
                    sync_status["records_synced"] += 1
                else:
                    sync_status["records_failed"] += 1
                    sync_status["errors"].append(f"Timeout uploading {display_name}")
                
                # Save progress every 10 records
                if (i + 1) % 10 == 0:
                    _save_status()
                    
            except Exception as e:
                sync_status["records_failed"] += 1
                sync_status["errors"].append(f"Upload error for record {i+1}: {str(e)[:100]}")
                logger.error(f"Sync upload error: {e}")
        
        # Step 4: Update customer record
        data = _load_customers()
        for c in data["customers"]:
            if c["id"] == customer_id:
                c["docs_synced"] = sync_status["records_synced"]
                c["last_sync"] = datetime.now().isoformat()
                break
        _save_customers(data)
        
        sync_status["status"] = "completed"
        sync_status["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        sync_status["status"] = "failed"
        sync_status["error"] = str(e)
        logger.error(f"Sync task failed: {e}")
    
    finally:
        _running_syncs.pop(customer_id, None)
        _save_status()


@router.get("/zoho/status")
async def get_sync_status(x_user_email: str = Header(...)):
    """Get current/last sync status."""
    email = x_user_email
    customer, _ = _get_or_create_customer(email)
    
    sync_file = DATA_DIR / "sync_status" / f"{customer['id']}_zoho.json"
    if not sync_file.exists():
        return {"status": "never_synced", "message": "No sync has been performed yet."}
    
    with open(sync_file, 'r') as f:
        status = json.load(f)
    
    # Add running flag
    status["is_running"] = bool(_running_syncs.get(customer["id"]))
    
    return status
