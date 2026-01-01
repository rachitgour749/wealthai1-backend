"""
File Search Store Setup and Ingestion Script

Creates the products store and uploads files for testing.
Handles nested folder structure for Insurance, Mutual Funds, etc.
"""

import time
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def main():
    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return
    
    client = genai.Client(api_key=api_key)
    
    print(f"=== File Search Store Setup ===\n")
    
    # Step 1: Create the store
    print(f"1. Creating File Search store...")
    try:
        store = client.file_search_stores.create(
            config={'display_name': 'Financial Products Knowledge Base - Full'}
        )
        store_name = store.name
        print(f"   ✅ Store created: {store_name}")
    except Exception as e:
        print(f"   ❌ Error creating store: {e}")
        return
    
    # Step 2: Find ALL PDF files recursively
    data_dir = Path("data/products")
    pdf_files = list(data_dir.rglob("*.pdf"))  # rglob = recursive glob
    
    # Also find markdown and text files
    md_files = list(data_dir.rglob("*.md"))
    txt_files = list(data_dir.rglob("*.txt"))
    
    all_files = pdf_files + md_files + txt_files
    
    print(f"\n2. Found {len(all_files)} files:")
    print(f"   - PDFs: {len(pdf_files)}")
    print(f"   - Markdown: {len(md_files)}")
    print(f"   - Text: {len(txt_files)}")
    
    # Group by folder for display
    folders = {}
    for f in all_files:
        folder = str(f.parent.relative_to(data_dir))
        folders[folder] = folders.get(folder, 0) + 1
    
    print(f"\n   By folder:")
    for folder, count in sorted(folders.items()):
        print(f"   - {folder}: {count} files")
    
    # Step 3: Upload files with progress
    print(f"\n3. Uploading files to store...")
    
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(all_files, 1):
        # Calculate relative path for display name
        relative_path = file_path.relative_to(data_dir)
        display_name = str(relative_path).replace("\\", "/").replace("/", " - ")
        
        print(f"   [{i}/{len(all_files)}] {relative_path.name}...", end=" ", flush=True)
        
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=store_name,
                config={
                    'display_name': display_name[:100]  # Truncate if too long
                }
            )
            
            # Wait for upload to complete (with timeout)
            wait_time = 0
            while not operation.done and wait_time < 60:
                time.sleep(2)
                wait_time += 2
                operation = client.operations.get(operation)
            
            if operation.done:
                print(f"✅")
                successful += 1
            else:
                print(f"⏱️ timeout")
                failed += 1
                
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            failed += 1
        
        # Progress update every 50 files
        if i % 50 == 0:
            print(f"\n   --- Progress: {successful} succeeded, {failed} failed ---\n")
    
    print(f"\n=== Upload Complete ===")
    print(f"Store Name: {store_name}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Step 4: Test query
    print(f"\n4. Testing query...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="List all the insurance and mutual fund products you have information about.",
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
        print(f"   Response preview: {response.text[:500]}...")
    except Exception as e:
        print(f"   ❌ Query error: {e}")
    
    # Save store name for later use
    with open(".store_name", "w") as f:
        f.write(store_name)
    print(f"\n   Store name saved to .store_name")


if __name__ == "__main__":
    main()
