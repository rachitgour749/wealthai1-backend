"""
Tenant Store Manager for Financial Advisor Chatbot

Manages isolated File Search stores per tenant:
- Shared products store (MF schemes, insurance)
- Per-tenant client stores (portfolio data from Zoho)
"""

import logging
import os
import json
from typing import Optional, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


# Read actual store name from file if exists, otherwise use default
def get_product_store_name():
    store_file = os.path.join(os.path.dirname(__file__), "../../.store_name")
    if os.path.exists(store_file):
        with open(store_file) as f:
            return f.read().strip()
    return None  # Will fall back to general query


def load_customers() -> Dict[str, Any]:
    """Load customers from customers.json file."""
    customers_file = os.path.join(os.path.dirname(__file__), "../data/customers.json")
    if os.path.exists(customers_file):
        with open(customers_file, 'r') as f:
            return json.load(f)
    return {"customers": []}


def get_customer_by_id(tenant_id: str) -> Optional[Dict[str, Any]]:
    """Get customer data by tenant ID."""
    customers_data = load_customers()
    for customer in customers_data.get('customers', []):
        if customer.get('id') == tenant_id:
            return customer
    return None


class TenantStoreManager:
    """Manages File Search stores with multi-tenant isolation."""
    
    # Namespace prefix to prevent naming collisions
    NAMESPACE = "finai_prod"
    
    def __init__(self, client: genai.Client):
        self.client = client
        self._product_store = get_product_store_name()
    
    @property
    def product_store(self) -> Optional[str]:
        """Shared store name for financial products."""
        return self._product_store
    
    def get_client_store_name(self, tenant_id: str) -> Optional[str]:
        """Get store name for tenant's client data from customers.json.
        
        Falls back to Money Compound's store as default if tenant not found.
        """
        # First try to find the specific tenant
        customer = get_customer_by_id(tenant_id)
        if customer and customer.get('file_search_store'):
            store_id = customer['file_search_store']
            logger.info(f"Found FileSearchStore for {tenant_id}: {store_id}")
            return store_id
        
        # Fallback to Money Compound's store as default
        logger.info(f"Tenant {tenant_id} not found, using Money Compound store as default")
        money_compound = get_customer_by_id("money_compound")
        if money_compound and money_compound.get('file_search_store'):
            default_store = money_compound['file_search_store']
            logger.info(f"Using default Money Compound store: {default_store}")
            return default_store
        
        logger.warning(f"No FileSearchStore found for tenant {tenant_id} and no default store available")
        return None
    
    async def provision_tenant(self, tenant_id: str) -> str:
        """
        Create isolated client store for new tenant.
        
        Args:
            tenant_id: Unique tenant identifier
        
        Returns:
            Created store name
        """
        store_name = self.get_client_store_name(tenant_id)
        
        try:
            await self.client.aio.file_search_stores.create(
                config={'display_name': f"Client Data - {tenant_id}"}
            )
            logger.info(f"Created store for tenant: {tenant_id}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Store already exists for tenant: {tenant_id}")
            else:
                raise
        
        return store_name
    
    async def query_product_store(
        self, 
        query: str,
        model: str = "gemini-2.5-flash"
    ):
        """
        Query shared products File Search store.
        
        Args:
            query: User's question about products
            model: Gemini model to use
        
        Returns:
            Gemini response with product information
        """
        if not self.product_store:
            logger.warning("No product store configured, falling back to general query")
            return await self._fallback_query(query, "No product store configured")
        
        return await self._query_store(self.product_store, query, model)
    
    async def query_client_store(
        self,
        tenant_id: str,
        query: str,
        filter_client: Optional[str] = None,
        model: str = "gemini-2.5-flash"
    ):
        """
        Query tenant-specific client store.
        
        Args:
            tenant_id: Tenant identifier
            query: User's question about client
            filter_client: Optional client name to focus on
            model: Gemini model to use
        
        Returns:
            Gemini response with client information
        """
        store = self.get_client_store_name(tenant_id)
        
        if not store:
            logger.warning(f"No FileSearchStore configured for tenant {tenant_id}")
            return await self._fallback_query(query, f"No store configured for tenant {tenant_id}")
        
        if filter_client:
            query = f"Client: {filter_client}. {query}"
        
        return await self._query_store(store, query, model)
    
    async def query_with_google_search(
        self,
        query: str,
        model: str = "gemini-2.5-flash"
    ):
        """
        Query using Google Search for real-time information.
        
        Args:
            query: Question requiring current market data
            model: Gemini model to use
        
        Returns:
            Gemini response grounded in Google Search
        """
        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            return response
        except Exception as e:
            logger.error(f"Google Search query failed: {e}")
            return await self._fallback_query(query, str(e))
    
    async def _query_store(
        self, 
        store_name: str, 
        query: str,
        model: str = "gemini-2.5-flash"
    ):
        """Internal method to query a File Search store."""
        try:
            logger.info(f"Querying store: {store_name}")
            print("\n" + "="*50)
            print(f"DEBUG LOG - Store Query")
            print(f"Store Name: {store_name}")
            
            # Check if store_name is already a full resource path
            # Valid formats: corpora/xxx or fileSearchStores/xxx
            is_resource_path = store_name.startswith('corpora/') or store_name.startswith('fileSearchStores/')
            
            if not is_resource_path:
                logger.info(f"Store name '{store_name}' is not a resource path, searching for matching store...")
                # List all stores and find the one with matching display name
                try:
                    stores_pager = await self.client.aio.file_search_stores.list()
                    matching_store = None
                    
                    # AsyncPager needs to be iterated with async for
                    async for store in stores_pager:
                        # Check if display_name matches our store_name
                        if hasattr(store, 'display_name') and store.display_name == store_name:
                            matching_store = store.name
                            logger.info(f"Found matching store: {matching_store}")
                            break
                    
                    if not matching_store:
                        logger.warning(f"No store found with display_name '{store_name}', falling back")
                        return await self._fallback_query(query, f"Store '{store_name}' not found")
                    
                    store_name = matching_store
                except Exception as list_error:
                    logger.error(f"Failed to list stores: {list_error}")
                    return await self._fallback_query(query, f"Failed to find store: {list_error}")
            
            config = types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )]
            )
            print(f"Gemini Config Object: {config}")
            print("="*50 + "\n")

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=query,
                config=config
            )
            return response
        except Exception as e:
            logger.error(f"Store query failed for {store_name}: {e}")
            return await self._fallback_query(query, str(e))
    
    async def _fallback_query(self, query: str, error: str):
        """Fallback to plain Gemini when store query fails."""
        logger.warning(f"Using fallback for query: {query[:50]}...")
        
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""I couldn't retrieve specific documents due to: {error}

Please answer based on general knowledge: {query}

Note: This is a general response, not from your specific data."""
        )
        
        # Mark response as fallback
        response._is_fallback = True
        return response
    
    async def upload_client_document(
        self,
        tenant_id: str,
        content: str,
        metadata: dict
    ):
        """
        Upload client document to tenant's store.
        
        Args:
            tenant_id: Tenant identifier
            content: Document content
            metadata: Document metadata (client_id, client_name, etc.)
        """
        store_name = self.get_client_store_name(tenant_id)
        
        # Create temporary file and upload
        # File Search handles chunking and embedding
        await self.client.aio.files.upload_to_file_search_store(
            file_search_store=store_name,
            content=content,
            display_name=metadata.get("client_name", "client_doc"),
            metadata=metadata
        )
