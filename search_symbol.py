import json

# Load the ScripMaster file
with open('Broker/AngelOne/resources/OpenAPIScripMaster.json', 'r') as f:
    data = json.load(f)

# Search for ITBEES
print("Searching for ITBEES...")
matches = [item for item in data if 'ITBEES' in item.get('name', '').upper() or 'ITBEES' in item.get('symbol', '').upper()]

print(f"\nFound {len(matches)} matches:\n")
for m in matches[:20]:
    print(f"Token: {m['token']:<15} Symbol: {m['symbol']:<25} Name: {m['name']:<25} Exchange: {m.get('exch_seg', 'N/A')}")
