# Financial Advisor AI Chatbot

Multi-RAG chatbot for financial intermediaries using Gemini API.

## Features
- Intent-based query routing (Product, Client, General, Complex)
- Multi-tenant client data via Zoho sync
- Shared financial products knowledge base
- Conversation memory for multi-turn chats
- Rate limiting and error handling

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn src.api.main:app --reload
```

## Project Structure
```
src/
├── core/           # Intent classifier, orchestrator, conversation
├── stores/         # File Search store management
├── sync/           # Zoho webhook integration
├── data/           # Data ingestion utilities
└── api/            # FastAPI endpoints
```
