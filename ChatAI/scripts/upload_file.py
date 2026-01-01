"""
Upload Individual Files to File Search Store

Simple script to upload one or more files to the existing File Search store.

Usage:
    python scripts/upload_file.py path/to/file.pdf
    python scripts/upload_file.py file1.pdf file2.pdf --category "Health Insurance"
    python scripts/upload_file.py path/to/folder/ --recursive
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_store_name():
    """Get the current File Search store name."""
    store_file = Path(".store_name")
    if not store_file.exists():
        print("ERROR: No .store_name file found. Run setup_store.py first.")
        sys.exit(1)
    
    with open(store_file, 'r') as f:
        return f.read().strip()


def upload_file(client, store_name, file_path, display_name=None):
    """Upload a single file to File Search store."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"  ❌ File not found: {file_path}")
        return False
    
    if not display_name:
        display_name = file_path.name
    
    print(f"  📤 Uploading: {file_path.name}...", end=" ", flush=True)
    
    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file=str(file_path),
            file_search_store_name=store_name,
            config={'display_name': display_name[:100]}
        )
        
        # Wait for upload to complete
        wait_time = 0
        while not operation.done and wait_time < 120:
            time.sleep(2)
            wait_time += 2
            operation = client.operations.get(operation)
        
        if operation.done:
            print("✅")
            return True
        else:
            print("⏱️ Timeout")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        return False


def test_upload(client, store_name, query):
    """Test if the uploaded file is queryable."""
    from google.genai import types
    
    print(f"\n🔍 Testing query: \"{query}\"")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )
                ]
            )
        )
        print(f"✅ Response: {response.text[:300]}...")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload files to File Search store")
    parser.add_argument("files", nargs="+", help="Files or directories to upload")
    parser.add_argument("--category", "-c", help="Category/display name prefix")
    parser.add_argument("--recursive", "-r", action="store_true", 
                        help="Recursively upload directory contents")
    parser.add_argument("--test", "-t", help="Test query after upload")
    parser.add_argument("--store", "-s", help="Override store name (optional)")
    
    args = parser.parse_args()
    
    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    
    # Get store name
    store_name = args.store if args.store else get_store_name()
    print(f"\n📚 File Search Store: {store_name}\n")
    
    # Collect files to upload
    files_to_upload = []
    
    for path_str in args.files:
        path = Path(path_str)
        
        if path.is_file():
            files_to_upload.append(path)
        elif path.is_dir():
            if args.recursive:
                # Find all supported files
                for ext in ['*.pdf', '*.md', '*.txt']:
                    files_to_upload.extend(path.rglob(ext))
            else:
                # Only immediate children
                for ext in ['*.pdf', '*.md', '*.txt']:
                    files_to_upload.extend(path.glob(ext))
        else:
            print(f"⚠️ Path not found: {path}")
    
    if not files_to_upload:
        print("No files found to upload.")
        sys.exit(1)
    
    print(f"📁 Found {len(files_to_upload)} file(s) to upload:\n")
    
    # Upload files
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(files_to_upload, 1):
        display_name = file_path.name
        if args.category:
            display_name = f"{args.category} - {file_path.name}"
        
        print(f"[{i}/{len(files_to_upload)}]", end="")
        
        if upload_file(client, store_name, file_path, display_name):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"✅ Uploaded: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"{'='*50}")
    
    # Test if requested
    if args.test:
        test_upload(client, store_name, args.test)
    else:
        print("\n💡 Tip: Use --test \"your query\" to verify the upload")


if __name__ == "__main__":
    main()
