"""
Q&A Excel to Markdown Converter

Converts Q&A spreadsheets to optimized Markdown format for File Search RAG.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def convert_qa_excel_to_markdown(
    excel_path: str,
    output_dir: Path,
    column_map: dict = None,
    min_answer_length: int = 50
) -> list[Path]:
    """
    Convert Q&A Excel to Markdown files optimized for RAG.
    
    Args:
        excel_path: Path to Excel file
        output_dir: Directory to write Markdown files
        column_map: Map your column names, e.g. {"question": "Q", "answer": "A"}
        min_answer_length: Minimum answer length to include (quality filter)
    
    Returns:
        List of created file paths
    """
    # Default column mapping
    column_map = column_map or {
        "question": "Question",
        "answer": "Answer",
        "category": "Category"
    }
    
    # Read Excel
    df = pd.read_excel(excel_path)
    logger.info(f"Loaded {len(df)} rows from {excel_path}")
    
    # Rename columns to standard names
    reverse_map = {v: k for k, v in column_map.items()}
    df = df.rename(columns=reverse_map)
    
    # Validate required columns
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError(f"Excel must have question and answer columns. Found: {df.columns.tolist()}")
    
    # Quality filter: skip short/empty answers
    df["answer"] = df["answer"].astype(str)
    original_count = len(df)
    df = df[df["answer"].str.len() >= min_answer_length]
    filtered_count = original_count - len(df)
    
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} Q&A pairs with short answers")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    # Group by category if available
    if "category" in df.columns and df["category"].notna().any():
        for category, group in df.groupby("category"):
            safe_name = str(category).lower().replace(" ", "_").replace("/", "_")
            output_file = output_dir / f"qa_{safe_name}.md"
            _write_qa_file(output_file, str(category), group)
            created_files.append(output_file)
    else:
        output_file = output_dir / "qa_knowledge.md"
        _write_qa_file(output_file, "General", df)
        created_files.append(output_file)
    
    logger.info(f"Created {len(created_files)} Q&A files in {output_dir}")
    return created_files


def _write_qa_file(path: Path, category: str, df: pd.DataFrame):
    """Write Q&A Markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {category} - Q&A Knowledge Base\n\n")
        f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n")
        f.write(f"*Total Q&A pairs: {len(df)}*\n\n---\n\n")
        
        for _, row in df.iterrows():
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            
            f.write(f"## Q: {question}\n\n")
            f.write(f"**A:** {answer}\n\n")
            f.write("---\n\n")
    
    logger.info(f"Written {len(df)} Q&A pairs to {path.name}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python qa_converter.py <excel_path> <output_dir>")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    
    # Optional column mapping from command line
    column_map = None
    if len(sys.argv) >= 6:
        column_map = {
            "question": sys.argv[3],
            "answer": sys.argv[4],
            "category": sys.argv[5] if len(sys.argv) > 5 else "Category"
        }
    
    files = convert_qa_excel_to_markdown(excel_path, output_dir, column_map)
    print(f"Created: {[str(f) for f in files]}")
