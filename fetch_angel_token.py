import requests
import json

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
print(f"Downloading {url}...")
try:
    r = requests.get(url)
    data = r.json()
    print(f"Downloaded {len(data)} instruments.")

    target_symbol = "PHARMABEES"
    found = False

    print(f"Searching for {target_symbol} on NSE...")
    for item in data:
        # Check for various symbol fields usually present
        sym = item.get('symbol')
        name = item.get('name')
        exch_seg = item.get('exch_seg')
        
        # Check mainly if name or symbol matches PHARMABEES
        if (sym == target_symbol or name == target_symbol) and exch_seg == 'NSE':
            print(f"FOUND: {item}")
            print(f"TOKEN: {item.get('token')}")
            found = True
            
    if not found:
        print("Not found on NSE. Checking other exchanges...")
        for item in data:
             if (item.get('symbol') == target_symbol or item.get('name') == target_symbol):
                 print(f"Found on {item.get('exch_seg')}: {item}")

except Exception as e:
    print(f"Error: {e}")
