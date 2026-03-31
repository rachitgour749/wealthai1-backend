"""
OPTIMIZED COMPLETE RE-SYNC: Zoho CRM ALL Modules -> File Search Store

Instead of 17 API calls PER contact (22,000+ calls), this:
1. Fetches ALL contacts in bulk (7 API calls)
2. Fetches ALL records from each product module in bulk (~17 calls)
3. Merges by contact ID
4. Uploads to File Search store

Total API calls: ~30 instead of 22,000+
"""
import httpx, json, os, sys, time, tempfile, io
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, ".")

def P(msg):
    print(msg, flush=True)

from dotenv import load_dotenv
load_dotenv()

from Services.ChatAI.sync.transform import transform_contact_to_document
from google import genai

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    P("ERROR: GEMINI_API_KEY not set in .env"); sys.exit(1)
CUSTOMER_ID = "cust_699df139"
STORE_A = "fileSearchStores/moneycompchattest-5r1aebaffgan"
BATCH_SIZE = 50

# Direct module API names and the field that links to Contact
# Format: (module_api_name, contact_link_field, display_name)
PRODUCT_MODULES = [
    ("Mutual_Fund_Transactions", "Contact_Name", "Mutual Fund Transactions"),
    ("Demat", "Contact_Name", "Demat"),
    ("Unlisted_Shares", "Contact_Name", "Unlisted Shares"),
    ("PMS", "Contact_Name", "PMS"),
    ("Bonds", "Contact_Name", "Bonds"),
    ("Life_Insurance", "Contact_Name", "Life Insurance"),
    ("Health_Insurance", "Contact_Name", "Health Insurance"),
    ("General_Insurance", "Contact_Name", "General Insurance"),
    ("NPS", "Contact_Name", "NPS"),
    ("Loan_Against_Mutual_Funds", "Contact", "Loan Against Mutual Funds"),
    ("Corporate_FD", "Contact_Name", "Corporate FD"),
    ("Invoice_Discounting", "Contact_Name", "Invoice Discounting"),
    ("P2P_Lending", "Contact_Name", "P2P Lending"),
    ("Financial_Planning", "Contact_Name", "Risk Profiling"),
    ("ISIF", "Contact_Name", "ISIF"),
]


def refresh_token():
    P("[STEP 1] Refreshing Zoho token...")
    with open("Services/ChatAI/data/api_keys.json", "r") as f:
        keys = json.load(f)
    k = keys[CUSTOMER_ID]
    resp = httpx.post(
        f"{k['zoho_domain']}/oauth/v2/token",
        data={
            "grant_type": "refresh_token",
            "client_id": k["zoho_client_id"],
            "client_secret": k["zoho_client_secret"],
            "refresh_token": k["zoho_refresh_token"],
        },
        timeout=15.0
    )
    data = resp.json()
    if "error" in data:
        P(f"  ERROR: {data}"); sys.exit(1)
    k["zoho_access_token"] = data["access_token"]
    keys[CUSTOMER_ID] = k
    with open("Services/ChatAI/data/api_keys.json", "w") as f:
        json.dump(keys, f, indent=2)
    P(f"  OK! Scope: {data.get('scope', 'unknown')}")
    return data["access_token"], k.get("zoho_api_domain", "https://www.zohoapis.in")


def fetch_all_from_module(module_name, token, api_domain):
    """Fetch ALL records from a Zoho module (paginated)."""
    headers = {"Authorization": f"Zoho-oauthtoken {token}"}
    all_records = []
    page = 1
    while True:
        try:
            resp = httpx.get(
                f"{api_domain}/crm/v2/{module_name}",
                headers=headers,
                params={"per_page": 200, "page": page},
                timeout=30.0
            )
            if resp.status_code == 200:
                data = resp.json()
                batch = data.get("data", [])
                all_records.extend(batch)
                info = data.get("info", {})
                if not info.get("more_records", False) or len(batch) < 200:
                    break
                page += 1
            elif resp.status_code == 204:
                break
            else:
                P(f"    Warning: {module_name} page {page}: {resp.status_code}")
                break
        except Exception as e:
            P(f"    Warning: {module_name} error: {str(e)[:80]}")
            break
    return all_records


def fetch_all_contacts(token, api_domain):
    P("\n[STEP 2] Fetching ALL contacts...")
    all_records = fetch_all_from_module("Contacts", token, api_domain)
    P(f"  Total: {len(all_records)} contacts")
    return all_records


def fetch_all_product_modules(token, api_domain):
    """Fetch ALL records from ALL product modules and index by contact ID."""
    P("\n[STEP 3] Fetching ALL product module data (bulk)...")
    
    # contact_id -> {module_name: [records]}
    contact_products = {}
    total_records = 0
    
    for module_api, contact_field, display_name in PRODUCT_MODULES:
        records = fetch_all_from_module(module_api, token, api_domain)
        
        if records:
            # Group by contact ID
            for record in records:
                contact_ref = record.get(contact_field)
                if isinstance(contact_ref, dict):
                    contact_id = contact_ref.get("id")
                else:
                    contact_id = None
                
                if contact_id:
                    if contact_id not in contact_products:
                        contact_products[contact_id] = {}
                    if display_name not in contact_products[contact_id]:
                        contact_products[contact_id][display_name] = []
                    contact_products[contact_id][display_name].append(record)
                    total_records += 1
            
            P(f"  {display_name:40s}: {len(records)} records")
        else:
            P(f"  {display_name:40s}: 0 records")
    
    P(f"\n  Total product records: {total_records}")
    P(f"  Contacts with product data: {len(contact_products)}")
    
    return contact_products


def format_related_record(record):
    """Format a related record into readable text."""
    lines = []
    skip_fields = {"id", "Created_Time", "Modified_Time", "Created_By", "Modified_By",
                   "Owner", "Contact_Name", "Contact"}
    
    for field, val in sorted(record.items()):
        if field.startswith("$") or field in skip_fields:
            continue
        if val is None or str(val).strip() in ("", "null", "None"):
            continue
        
        if isinstance(val, dict):
            val = val.get("name", str(val))
        elif isinstance(val, list):
            if val and isinstance(val[0], dict):
                val = ", ".join(v.get("name", str(v)) for v in val)
            else:
                val = ", ".join(str(v) for v in val)
        
        field_display = field.replace("_", " ").strip()
        lines.append(f"  - {field_display}: {val}")
    
    return lines


def transform_contact_with_products(contact, related_data):
    """Transform contact + product data into comprehensive document."""
    base_doc = transform_contact_to_document(contact)
    content = base_doc["content"]
    
    for module_name, records in related_data.items():
        if not records:
            continue
        
        section_lines = [f"\n## {module_name}"]
        
        for i, record in enumerate(records):
            if len(records) > 1:
                section_lines.append(f"\n### {module_name} #{i+1}")
            
            record_lines = format_related_record(record)
            if record_lines:
                section_lines.extend(record_lines)
        
        if len(section_lines) > 1:
            content += "\n" + "\n".join(section_lines)
    
    base_doc["content"] = content
    return base_doc


def upload_to_store(docs_batch, store_name, batch_num, total_batches):
    """Upload a batch of documents to the File Search store."""
    client = genai.Client(api_key=GEMINI_KEY)
    
    combined = "\n\n" + "=" * 80 + "\n\n"
    combined_content = combined.join(doc["content"] for doc in docs_batch)
    
    names = [d["metadata"].get("client_name", "?") for d in docs_batch]
    display_name = f"Full_{batch_num+1} ({names[0]} to {names[-1]})"
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(combined_content)
        tmp = f.name
    
    try:
        op = client.file_search_stores.upload_to_file_search_store(
            file=tmp, file_search_store_name=store_name,
            config={"display_name": display_name}
        )
        
        wait = 0
        while not op.done and wait < 120:
            time.sleep(3)
            wait += 3
            op = client.operations.get(op)
        
        os.unlink(tmp)
        
        if op.done:
            P(f"  Batch {batch_num+1}/{total_batches}: OK ({len(docs_batch)} contacts)")
            return len(docs_batch), 0
        else:
            P(f"  Batch {batch_num+1}/{total_batches}: TIMEOUT")
            return 0, len(docs_batch)
    except Exception as e:
        P(f"  Batch {batch_num+1}/{total_batches}: ERROR - {str(e)[:100]}")
        try: os.unlink(tmp)
        except: pass
        return 0, len(docs_batch)


def main():
    P("=" * 70)
    P("OPTIMIZED COMPLETE RE-SYNC: ALL Zoho Modules -> File Search Store")
    P(f"  Store: moneycompchattest")
    P(f"  Data: Contacts (109 fields) + 15 Product Modules (bulk fetch)")
    P("=" * 70)
    
    token, api_domain = refresh_token()
    
    # Step 2: Fetch all contacts
    contacts = fetch_all_contacts(token, api_domain)
    if not contacts:
        P("No contacts found!")
        return
    
    # Step 3: Fetch ALL product module data in bulk
    contact_products = fetch_all_product_modules(token, api_domain)
    
    # Step 4: Merge contacts with product data
    P("\n[STEP 4] Merging contacts with product data...")
    all_docs = []
    contacts_with_products = 0
    
    for contact in contacts:
        contact_id = contact["id"]
        related = contact_products.get(contact_id, {})
        
        if related:
            contacts_with_products += 1
        
        doc = transform_contact_with_products(contact, related)
        all_docs.append(doc)
    
    P(f"  {len(all_docs)} documents prepared")
    P(f"  {contacts_with_products} contacts have product data")
    
    # Step 5: Upload to store
    P(f"\n[STEP 5] Uploading to {STORE_A}...")
    total_synced = 0
    total_failed = 0
    total_batches = (len(all_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(all_docs))
        batch = all_docs[start:end]
        
        synced, failed = upload_to_store(batch, STORE_A, batch_num, total_batches)
        total_synced += synced
        total_failed += failed
    
    # Step 6: Update customer record
    P(f"\n[STEP 6] Updating customer record...")
    with open("Services/ChatAI/data/customers.json", "r") as f:
        data = json.load(f)
    
    now = datetime.now().isoformat()
    for c in data["customers"]:
        if c["id"] == "cust_699df139":
            c["docs_synced"] = total_synced
            c["last_sync"] = now
    
    with open("Services/ChatAI/data/customers.json", "w") as f:
        json.dump(data, f, indent=2)
    
    P(f"\n{'=' * 70}")
    P("COMPLETE!")
    P(f"  Contacts synced: {total_synced}")
    P(f"  Failed: {total_failed}")
    P(f"  Contacts with product data: {contacts_with_products}")
    P(f"  Modules included: Contacts + {len(PRODUCT_MODULES)} product modules")
    P(f"{'=' * 70}")


if __name__ == "__main__":
    main()
