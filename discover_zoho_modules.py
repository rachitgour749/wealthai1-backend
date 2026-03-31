"""Discover the correct API names for all product-related modules by testing related list names."""
import httpx, json, sys, io, os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open("Services/ChatAI/data/api_keys.json", "r") as f:
    keys = json.load(f)

k = keys["cust_699df139"]
access_token = k["zoho_access_token"]
api_domain = k.get("zoho_api_domain", "https://www.zohoapis.in")
headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}

# The related list API names from the settings response:
# api_name -> display_label -> module
related_lists = {
    "Contact_Name2": "Mutual Fund Transactions",
    "Demat": "Demat",
    "Contact": "Unlisted Shares",
    "Related_List_Label_4": "PMS",
    "Bonds": "Bonds",
    "Contact_Owner": "Life Insurance",
    "Health_Insurance": "Health Insurance",
    "General_Insurance": "General Insurance",
    "Related_List_Label_3": "NPS",
    "Related_List_Label_2": "Loan Against Mutual Funds",
    "Corporate_FD": "Corporate FD",
    "Invoice_Discounting": "Invoice Discounting",
    "P2P_Lending": "P2P Lending",
    "Related_List_Label": "Risk Profiling",
    "ISIF": "ISIF",
    "Family_Details": "Family Details",
    "Notes": "Notes",
}

# Find a contact with lots of data (Swati Jain)
print("Searching for Swati Jain...")
resp = httpx.get(
    f"{api_domain}/crm/v2/Contacts/search",
    headers=headers,
    params={"criteria": "(Full_Name:equals:Swati Jain)"},
    timeout=15
)
contacts = resp.json().get("data", []) if resp.status_code == 200 else []
if not contacts:
    resp = httpx.get(f"{api_domain}/crm/v2/Contacts/search", headers=headers,
                     params={"criteria": "(Last_Name:equals:Jain)"}, timeout=15)
    contacts = resp.json().get("data", []) if resp.status_code == 200 else []

# Also try a contact we know has MF data
print("Searching for contacts with most data...")
resp2 = httpx.get(f"{api_domain}/crm/v2/Contacts", headers=headers,
                  params={"per_page": 10, "sort_by": "Modified_Time", "sort_order": "desc"}, timeout=15)
test_contacts = resp2.json().get("data", []) if resp2.status_code == 200 else []

all_test = contacts + test_contacts
# Deduplicate
seen = set()
unique = []
for c in all_test:
    if c["id"] not in seen:
        seen.add(c["id"])
        unique.append(c)

print(f"Testing with {len(unique)} contacts")
print("=" * 80)

for related_api_name, display in related_lists.items():
    found_data = False
    for contact in unique[:10]:
        cid = contact["id"]
        name = contact.get("Full_Name", "?")
        try:
            r = httpx.get(
                f"{api_domain}/crm/v2/Contacts/{cid}/{related_api_name}",
                headers=headers, params={"per_page": 2}, timeout=10
            )
            if r.status_code == 200:
                records = r.json().get("data", [])
                if records:
                    print(f"\n{'='*80}")
                    print(f"  {display} (api: {related_api_name}) - Contact: {name}")
                    print(f"  Records: {len(records)}+")
                    print(f"  Fields:")
                    for field, val in sorted(records[0].items()):
                        if field.startswith("$"):
                            continue
                        if val and str(val).strip() not in ("", "null", "None"):
                            print(f"    {field:45s}: {str(val)[:120]}")
                    found_data = True
                    break
        except:
            pass
    
    if not found_data:
        print(f"  {display} (api: {related_api_name}): No data found in tested contacts")

# Also try fetching modules directly
print("\n" + "=" * 80)
print("DIRECT MODULE ACCESS TEST")
print("=" * 80)

direct_modules = [
    "Mutual_Fund_Transactions", "Life_Insurance", "Health_Insurance",
    "General_Insurance", "Demat", "Unlisted_Shares", "NPS", "Bonds",
    "PMS", "AIF", "ISIF", "Loan_Against_Mutual_Funds", "Corporate_FD",
    "Invoice_Discounting", "P2P_Lending", "Financial_Planning"
]

for mod in direct_modules:
    try:
        r = httpx.get(f"{api_domain}/crm/v2/{mod}", headers=headers,
                      params={"per_page": 2}, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            print(f"\n  {mod}: {len(data)}+ records")
            if data:
                for field, val in sorted(data[0].items()):
                    if field.startswith("$"):
                        continue
                    if val and str(val).strip() not in ("", "null", "None"):
                        print(f"    {field:45s}: {str(val)[:120]}")
        elif r.status_code == 204:
            print(f"  {mod}: Empty (204)")
        else:
            print(f"  {mod}: {r.status_code}")
    except Exception as e:
        print(f"  {mod}: Error - {str(e)[:80]}")
