
import requests
import os
import sys

# Constants
DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
# Using the compact one first, if it doesn't have what we need, we'll switch to detailed.
# Actually, let's use the one that is most likely to have the mapping.
# The search result said the detailed one has Security ID.
DHAN_SCRIP_MASTER_DETAILED_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'resources')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'api-scrip-master.csv')

def download_scrip_master():
    """Download the Dhan Scrip Master CSV and save it locally."""
    print(f"Downloading Scrip Master from {DHAN_SCRIP_MASTER_URL}...")
    try:
        response = requests.get(DHAN_SCRIP_MASTER_URL, stream=True)
        response.raise_for_status()

        with open(OUTPUT_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    print(".", end="", flush=True)
        
        print(f"\nSaved to {OUTPUT_FILE}")
        
        # Check file size
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"Downloaded {file_size / 1024:.2f} KB")
        
        return True
    
    except Exception as e:
        print(f"Error downloading Scrip Master: {e}")
        return False

if __name__ == "__main__":
    # Ensure resources directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success = download_scrip_master()
    if success:
        print("Scrip Master download complete.")
    sys.exit(0 if success else 1)
