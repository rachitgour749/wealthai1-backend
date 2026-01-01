"""
Zoho Batch Sync for Financial Advisor Chatbot

Monthly batch sync of client data from Zoho CRM to File Search stores.
Pulls all contacts for a customer and syncs to their File Search store.
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import asyncio

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


class ZohoBatchSync:
    """
    Handles monthly batch sync of Zoho CRM contacts to File Search.
    
    Flow:
    1. Load customer's Zoho credentials
    2. Fetch all contacts from Zoho CRM API
    3. Transform to document format
    4. Upload to customer's File Search store
    """
    
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.zoho_credentials = self._load_zoho_credentials()
        
    def _load_zoho_credentials(self) -> dict:
        """Load Zoho credentials for customer."""
        api_keys_file = DATA_DIR / "api_keys.json"
        if not api_keys_file.exists():
            raise ValueError(f"No credentials found for customer {self.customer_id}")
        
        with open(api_keys_file, 'r') as f:
            all_keys = json.load(f)
        
        customer_keys = all_keys.get(self.customer_id, {})
        return {
            'org_id': customer_keys.get('zoho_org_id'),
            'client_id': customer_keys.get('zoho_client_id'),
            'client_secret': customer_keys.get('zoho_client_secret'),
            'gemini_api_key': customer_keys.get('gemini_api_key')
        }
    
    async def get_zoho_access_token(self) -> str:
        """
        Get Zoho access token using OAuth.
        
        Note: In production, you'd use refresh tokens stored securely.
        This is a simplified version for demo purposes.
        """
        import httpx
        
        if not self.zoho_credentials.get('client_id'):
            raise ValueError("Zoho client_id not configured")
        
        # In production: Use stored refresh token to get access token
        # For demo: Return placeholder (user would need to complete OAuth flow)
        logger.warning("Zoho OAuth not fully implemented - using placeholder")
        return "placeholder_token"
    
    async def fetch_contacts_from_zoho(self, access_token: str) -> List[dict]:
        """
        Fetch all contacts from Zoho CRM.
        
        Uses Zoho CRM API v2:
        GET https://www.zohoapis.com/crm/v2/Contacts
        
        Returns:
            List of contact records
        """
        import httpx
        
        contacts = []
        page = 1
        per_page = 200
        
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.get(
                        "https://www.zohoapis.com/crm/v2/Contacts",
                        headers={
                            "Authorization": f"Zoho-oauthtoken {access_token}",
                            "Content-Type": "application/json"
                        },
                        params={
                            "page": page,
                            "per_page": per_page
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        batch = data.get("data", [])
                        contacts.extend(batch)
                        
                        # Check if more pages
                        if len(batch) < per_page:
                            break
                        page += 1
                    else:
                        logger.error(f"Zoho API error: {response.status_code}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error fetching contacts: {e}")
                    break
        
        return contacts
    
    def transform_contact(self, contact: dict) -> dict:
        """Transform Zoho contact to document format."""
        from Services.ChatAI.sync.transform import transform_contact_to_document
        return transform_contact_to_document(contact)
    
    async def sync_to_file_search(
        self, 
        contacts: List[dict],
        progress_callback: Optional[callable] = None
    ) -> dict:
        """
        Sync contacts to customer's File Search store.
        
        Args:
            contacts: List of transformed contact documents
            progress_callback: Optional callback for progress updates
        
        Returns:
            Sync results summary
        """
        if not self.zoho_credentials.get('gemini_api_key'):
            raise ValueError("Customer Gemini API key not configured")
        
        client = genai.Client(api_key=self.zoho_credentials['gemini_api_key'])
        
        # Load customer data to get File Search store
        customers_file = DATA_DIR / "customers.json"
        with open(customers_file, 'r') as f:
            data = json.load(f)
        
        customer = next(
            (c for c in data['customers'] if c['id'] == self.customer_id), 
            None
        )
        
        if not customer:
            raise ValueError(f"Customer {self.customer_id} not found")
        
        store_name = customer.get('file_search_store')
        if not store_name:
            # Create client store for this customer
            store = client.file_search_stores.create(
                config={'display_name': f'Clients - {customer["name"]}'}
            )
            store_name = store.name
            customer['client_file_search_store'] = store_name
            
            with open(customers_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Sync contacts
        synced = 0
        failed = 0
        
        for i, contact in enumerate(contacts):
            try:
                # Create temporary file with contact data
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix='.txt', 
                    delete=False
                ) as f:
                    f.write(contact['content'])
                    temp_path = f.name
                
                # Upload to File Search
                operation = client.file_search_stores.upload_to_file_search_store(
                    file=temp_path,
                    file_search_store_name=store_name,
                    config={'display_name': contact['metadata'].get('client_name', f'Contact_{i}')}
                )
                
                # Wait for upload
                wait_time = 0
                while not operation.done and wait_time < 30:
                    time.sleep(1)
                    wait_time += 1
                    operation = client.operations.get(operation)
                
                # Cleanup temp file
                os.unlink(temp_path)
                
                if operation.done:
                    synced += 1
                else:
                    failed += 1
                
                if progress_callback:
                    progress_callback(i + 1, len(contacts))
                    
            except Exception as e:
                logger.error(f"Failed to sync contact: {e}")
                failed += 1
        
        return {
            'synced': synced,
            'failed': failed,
            'total': len(contacts)
        }
    
    async def run_full_sync(self) -> dict:
        """
        Run complete sync: Fetch from Zoho → Transform → Upload to File Search.
        """
        logger.info(f"Starting Zoho batch sync for customer {self.customer_id}")
        
        results = {
            'customer_id': self.customer_id,
            'started_at': datetime.now().isoformat(),
            'status': 'in_progress',
            'contacts_fetched': 0,
            'contacts_synced': 0,
            'errors': []
        }
        
        try:
            # Step 1: Get Zoho access token
            access_token = await self.get_zoho_access_token()
            
            # Step 2: Fetch contacts
            contacts = await self.fetch_contacts_from_zoho(access_token)
            results['contacts_fetched'] = len(contacts)
            
            if not contacts:
                results['status'] = 'completed'
                results['message'] = 'No contacts found in Zoho'
                return results
            
            # Step 3: Transform contacts
            documents = [self.transform_contact(c) for c in contacts]
            
            # Step 4: Sync to File Search
            sync_results = await self.sync_to_file_search(documents)
            results['contacts_synced'] = sync_results['synced']
            results['sync_failed'] = sync_results['failed']
            
            results['status'] = 'completed'
            results['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Batch sync failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        # Save sync results
        sync_dir = DATA_DIR / "sync_status"
        sync_dir.mkdir(parents=True, exist_ok=True)
        
        with open(sync_dir / f"{self.customer_id}_zoho.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


# ============== Scheduler Functions ==============

async def run_monthly_zoho_sync_for_all():
    """
    Run monthly Zoho sync for all customers with Zoho configured.
    
    Call this from a cron job or scheduled task.
    """
    logger.info("Starting monthly Zoho sync for all customers")
    
    customers_file = DATA_DIR / "customers.json"
    if not customers_file.exists():
        logger.warning("No customers found")
        return
    
    with open(customers_file, 'r') as f:
        data = json.load(f)
    
    api_keys_file = DATA_DIR / "api_keys.json"
    with open(api_keys_file, 'r') as f:
        api_keys = json.load(f)
    
    results = []
    
    for customer in data['customers']:
        customer_id = customer['id']
        customer_keys = api_keys.get(customer_id, {})
        
        # Skip customers without Zoho configured
        if not customer_keys.get('zoho_org_id'):
            logger.info(f"Skipping {customer_id} - no Zoho configured")
            continue
        
        try:
            sync = ZohoBatchSync(customer_id)
            result = await sync.run_full_sync()
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to sync {customer_id}: {e}")
            results.append({
                'customer_id': customer_id,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save overall results
    sync_summary = {
        'run_at': datetime.now().isoformat(),
        'customers_processed': len(results),
        'results': results
    }
    
    with open(DATA_DIR / "sync_status" / "monthly_sync_summary.json", 'w') as f:
        json.dump(sync_summary, f, indent=2)
    
    logger.info(f"Monthly sync completed: {len(results)} customers processed")
    return sync_summary


# ============== CLI Interface ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Zoho Batch Sync")
    parser.add_argument("--customer", help="Sync specific customer")
    parser.add_argument("--all", action="store_true", help="Sync all customers")
    
    args = parser.parse_args()
    
    if args.customer:
        sync = ZohoBatchSync(args.customer)
        result = asyncio.run(sync.run_full_sync())
        print(json.dumps(result, indent=2))
    elif args.all:
        result = asyncio.run(run_monthly_zoho_sync_for_all())
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python zoho_batch_sync.py --customer CUSTOMER_ID | --all")
