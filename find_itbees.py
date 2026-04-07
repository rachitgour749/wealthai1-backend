import json

with open('Broker/AngelOne/resources/OpenAPIScripMaster.json', 'r') as f:
    data = json.load(f)

print("Searching for ITBEES...")
for item in data:
    if 'ITBEES' in str(item.get('symbol')).upper() or 'ITBEES' in str(item.get('name')).upper():
        print(f"Token: {item.get('token')}, Symbol: {item.get('symbol')}, Name: {item.get('name')}, Exch: {item.get('exch_seg')}")
