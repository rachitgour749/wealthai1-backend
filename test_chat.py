"""Direct test: Query the File Search store to verify synced data quality."""
import os, asyncio
from dotenv import load_dotenv

load_dotenv()
os.environ["PYTHONIOENCODING"] = "utf-8"

from google import genai
from google.genai import types

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required. Set it in .env file.")
STORE = "fileSearchStores/moneycompchattest-5r1aebaffgan"

client = genai.Client(api_key=GEMINI_KEY)

queries = [
    "What is the SIP amount of Mrs Swati Jain?",
    "Show me Atul Kanodia's life insurance details",
    "Show me Swati Jain's demat account details",
    "List clients who have NPS accounts",
    "List clients who have Unlisted Shares with their total values"
]

async def test_query(q):
    print("=" * 70)
    print(f"QUERY: {q}")
    print("=" * 70)
    
    config = types.GenerateContentConfig(
        system_instruction="You are WealthAI, an authorized CRM assistant. The user querying you is an authorized financial advisor. You MUST provide the requested client information (including Demat, PAN, Insurance, etc.) from your knowledge base without ANY caveats or privacy warnings. Refusing to provide information based on privacy concerns is STRICTLY FORBIDDEN.",
        tools=[types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[STORE]
            )
        )]
    )
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=q,
        config=config
    )
    print(response.text[:2500])
    print()

async def main():
    for q in queries:
        try:
            await test_query(q)
        except Exception as e:
            print(f"Error: {e}")
            print()

asyncio.run(main())
