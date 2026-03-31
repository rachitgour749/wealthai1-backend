"""Fetch sample contacts using v2 API which returns all fields by default."""
import httpx, json, sys, io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open("Services/ChatAI/data/api_keys.json", "r") as f:
    keys = json.load(f)

k = keys["cust_699df139"]
access_token = k["zoho_access_token"]
api_domain = k.get("zoho_api_domain", "https://www.zohoapis.in")

print("Fetching 2 sample contacts using v2 API...")
resp = httpx.get(
    f"{api_domain}/crm/v2/Contacts",
    headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
    params={"per_page": 2},
    timeout=30.0
)

if resp.status_code == 200:
    data = resp.json()
    contacts = data.get("data", [])
    
    if contacts:
        all_field_names = set()
        for contact in contacts:
            print(f"\n{'='*60}")
            name = contact.get('Full_Name', 'N/A')
            print(f"Contact: {name}")
            print(f"Total fields: {len(contact)}")
            print(f"{'='*60}")
            
            for key, value in sorted(contact.items()):
                all_field_names.add(key)
                if value and str(value).strip() and str(value) != "null" and str(value) != "None":
                    val_str = str(value)
                    if len(val_str) > 150:
                        val_str = val_str[:150] + "..."
                    try:
                        print(f"  {key:40s}: {val_str}")
                    except:
                        print(f"  {key:40s}: [encoding error]")
        
        print(f"\n\n{'='*60}")
        print(f"ALL FIELD NAMES ({len(all_field_names)} unique):")
        print(f"{'='*60}")
        for name in sorted(all_field_names):
            print(f"  - {name}")
    else:
        print("No contacts returned")
else:
    print(f"Error: {resp.status_code} - {resp.text[:500]}")
