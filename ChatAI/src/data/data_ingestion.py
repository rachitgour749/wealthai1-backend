"""
Data Ingestion Pipeline for Financial Advisor Chatbot

Uploads documents to Gemini File Search stores.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from google import genai

logger = logging.getLogger(__name__)


async def ingest_product_data(
    data_dir: Path,
    store_name: str,
    api_key: str = None,
    extensions: tuple = (".pdf", ".md", ".txt", ".docx")
) -> dict:
    """
    Upload financial product documents to File Search.
    
    Args:
        data_dir: Directory containing product documents
        store_name: Name of File Search store
        api_key: Gemini API key (uses env var if not provided)
        extensions: File extensions to process
    
    Returns:
        Summary of ingestion results
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required")
    
    client = genai.Client(api_key=api_key)
    data_dir = Path(data_dir)
    
    # Create or get store
    try:
        store = await client.aio.file_search_stores.create(
            name=store_name,
            display_name="Financial Products Knowledge Base"
        )
        logger.info(f"Created store: {store_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"Using existing store: {store_name}")
        else:
            raise
    
    # Find all files
    files = []
    for ext in extensions:
        files.extend(data_dir.glob(f"**/*{ext}"))
    
    logger.info(f"Found {len(files)} files to ingest")
    
    # Upload files
    results = {"success": 0, "failed": 0, "files": []}
    
    for file_path in files:
        try:
            await client.aio.files.upload_to_file_search_store(
                file_search_store=store_name,
                file_path=str(file_path),
                display_name=file_path.stem
            )
            results["success"] += 1
            results["files"].append({"path": str(file_path), "status": "success"})
            logger.info(f"Indexed: {file_path.name}")
        except Exception as e:
            results["failed"] += 1
            results["files"].append({"path": str(file_path), "status": "failed", "error": str(e)})
            logger.error(f"Failed to index {file_path.name}: {e}")
    
    results["total"] = len(files)
    results["timestamp"] = datetime.now().isoformat()
    
    return results


def check_data_freshness(doc_path: Path) -> dict:
    """
    Check if document data is stale based on freshness policy.
    
    Args:
        doc_path: Path to document
    
    Returns:
        Freshness status dict
    """
    FRESHNESS_POLICY = {
        "nav_returns": 30,      # Days
        "factsheet": 30,
        "expense_ratio": 90,
        "fund_manager": 180,
        "prospectus": 365,
        "default": 90
    }
    
    # Get file modification time
    mtime = datetime.fromtimestamp(doc_path.stat().st_mtime)
    age_days = (datetime.now() - mtime).days
    
    # Classify document type from filename
    filename_lower = doc_path.stem.lower()
    doc_type = "default"
    
    for key in FRESHNESS_POLICY:
        if key in filename_lower:
            doc_type = key
            break
    
    max_age = FRESHNESS_POLICY[doc_type]
    is_fresh = age_days <= max_age
    
    return {
        "path": str(doc_path),
        "doc_type": doc_type,
        "age_days": age_days,
        "max_age": max_age,
        "is_fresh": is_fresh,
        "status": "fresh" if is_fresh else "stale"
    }


def scan_for_stale_data(data_dir: Path) -> list[dict]:
    """
    Scan directory for stale documents.
    
    Args:
        data_dir: Directory to scan
    
    Returns:
        List of stale document info
    """
    stale_docs = []
    
    for file_path in Path(data_dir).glob("**/*"):
        if file_path.is_file() and file_path.suffix in [".pdf", ".md", ".txt"]:
            freshness = check_data_freshness(file_path)
            if not freshness["is_fresh"]:
                stale_docs.append(freshness)
    
    logger.info(f"Found {len(stale_docs)} stale documents in {data_dir}")
    return stale_docs


if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python data_ingestion.py <data_dir> <store_name>")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    store_name = sys.argv[2]
    
    result = asyncio.run(ingest_product_data(data_dir, store_name))
    print(f"Ingestion complete: {result['success']}/{result['total']} files")
