 """Check what data is ACTUALLY in the contacts - find contacts with the most populated fields."""
import httpx, json, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open("Services/ChatAI/data/api_keys.json", "r") as f:
    keys = json.load(f)

k = keys["cust_699df139"]
access_token = k["zoho_access_token"]
api_domain = k.get("zoho_api_domain", "https://www.zohoapis.in")
headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}

# Fetch 50 contacts
resp = httpx.get(f"{api_domain}/crm/v2/Contacts", headers=headers, 
                 params={"per_page": 50, "sort_by": "Modified_Time", "sort_order": "desc"}, timeout=30)
contacts = resp.json().get("data", [])

print(f"Fetched {len(contacts)} contacts\n")

# Find contacts with the MOST non-empty fields
scored = []
for c in contacts:
    non_empty = 0
    for k2, v in c.items():
        if k2.startswith("$"):
            continue
        if v and str(v).strip() and str(v) != "null" and str(v) != "None":
            non_empty += 1
    scored.append((non_empty, c))

scored.sort(key=lambda x: -x[0])

# Show top 5 most populated contacts
print("=" * 70)
print("TOP 5 CONTACTS WITH MOST DATA")
print("=" * 70)

for rank, (count, contact) in enumerate(scored[:5], 1):
    name = contact.get("Full_Name", "Unknown")
    print(f"\n{'='*70}")
    print(f"#{rank} - {name} ({count} populated fields)")
    print(f"{'='*70}")
    for k2, v in sorted(contact.items()):
        if k2.startswith("$"):
            continue
        if v and str(v).strip() and str(v) != "null" and str(v) != "None":
            val = str(v)[:150]
            print(f"  {k2:40s}: {val}")

# Also show all unique field names across all contacts
print(f"\n\n{'='*70}")
print("ALL UNIQUE NON-EMPTY FIELDS ACROSS ALL CONTACTS")
print(f"{'='*70}")
field_counts = {}
for c in contacts:
    for k2, v in c.items():
        if k2.startswith("$"):
            continue
        if v and str(v).strip() and str(v) != "null" and str(v) != "None":
            field_counts[k2] = field_counts.get(k2, 0) + 1

for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
    pct = count * 100 // len(contacts)
    bar = "#" * (pct // 2)
    print(f"  {field:40s}: {count:4d}/{len(contacts)} ({pct:3d}%) {bar}")
