# app/chatai1_prompts.py
"""System prompts for LLM calls - loaded from files or environment"""
import os
from pathlib import Path


# NOTE: In production, these should be loaded from environment variables or config files
# For now, we provide stub placeholders that can be replaced with actual prompt text

def load_prompt_from_file(filename: str) -> str:
    """Load prompt from file if exists, otherwise return placeholder"""
    prompt_path = Path(__file__).parent / "prompt_files" / filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    # Return environment variable if set
    env_var_name = filename.replace(".txt", "").upper().replace("-", "_")
    if env_var_name in os.environ:
        return os.environ[env_var_name]

    # Return placeholder
    return f"[PLACEHOLDER: {filename} - Load actual prompt content here]"


# Router System Prompt
ROUTER_SYSTEM_PROMPT = load_prompt_from_file("MASTER_INTENT_CLASSIFIER_AND_ROUTER_PROMPT.txt")

# Base ChatAI1 System Prompt
BASE_SYSTEM_PROMPT = load_prompt_from_file("SystemPrompt-Main.txt")

# Domain-specific prompts
MF_SYSTEM_PROMPT = load_prompt_from_file("SystemPrompt-MF.txt")
INSURANCE_SYSTEM_PROMPT = load_prompt_from_file("SystemPrompt-Insurance.txt")
STOCKS_SYSTEM_PROMPT = load_prompt_from_file("SystemPrompt-Stocks.txt")

# Alternative: Direct string constants (uncomment if loading from files doesn't work)
# Copy the actual prompt text from uploaded files into these constants

# ROUTER_SYSTEM_PROMPT = """
# You are the Unified Intent & Routing Analyzer for ChatAI1...
# [Full text from MASTER_INTENT_CLASSIFIER_AND_ROUTER_PROMPT.txt]
# """

# BASE_SYSTEM_PROMPT = """
# You are ChatAI1, an AI assistant designed specifically for...
# [Full text from SystemPrompt-Main.txt]
# """

# ... and so on for other prompts