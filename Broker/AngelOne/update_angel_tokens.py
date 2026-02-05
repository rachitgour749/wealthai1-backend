
import requests
import json
import os
import sys

# Constants
ANGEL_SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'resources')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'OpenAPIScripMaster.json')

def download_scrip_master():
    """Download the AngelOne Scrip Master JSON and save it locally."""
    print(f"Downloading Scrip Master from {ANGEL_SCRIP_MASTER_URL}...")
    try:
        response = requests.get(ANGEL_SCRIP_MASTER_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        with open(OUTPUT_FILE, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                print(".", end="", flush=True)
        
        print(f"\nSaved to {OUTPUT_FILE}")
        
        # Verify JSON validity
        print("Verifying JSON validity...")
        with open(OUTPUT_FILE, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded {len(data)} instruments.")
            
        print("Scrip Master download complete.")
        return True
    
    except Exception as e:
        print(f"Error downloading Scrip Master: {e}")
        return False

if __name__ == "__main__":
    # Ensure resources directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success = download_scrip_master()
    sys.exit(0 if success else 1)
