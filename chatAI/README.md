# ChatAI Integration

This directory contains the ChatAI integration for the unified WealthAI1 backend server.

## Structure

```
chatAI/
├── chat_api.py      # FastAPI router for ChatAI endpoints
├── chat_core.py     # Core ChatAI functionality and database operations
└── README.md        # This file
```

## Features

- **Chat Endpoint**: `/api/chat` - Generate AI responses
- **Rating Endpoint**: `/api/rate` - Store user ratings and feedback
- **Health Check**: `/api/health` - Check ChatAI service status
- **Database Storage**: SQLite database for conversations and ratings
- **Response History**: Track conversation history and user feedback

## API Endpoints

### POST /api/chat
Generate AI responses to user prompts.

**Request Body:**
```json
{
  "prompt": "What is a mutual fund?",
  "system_prompt": "Optional custom system prompt"
}
```

**Response:**
```json
{
  "response": "📚 DEFINITION: A mutual fund is...",
  "model_used": "mf-assistant:latest",
  "response_id": "uuid-string",
  "timestamp": 1756454819.0502355,
  "system_prompt_used": "System prompt used",
  "rating": null,
  "provider": "langchain_modal"
}
```

### POST /api/rate
Store user ratings and feedback for AI responses.

**Request Body:**
```json
{
  "trace_id": "response-uuid",
  "user_rating": 8,
  "feedback_comment": "Great explanation!"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Rating stored successfully",
  "trace_id": "response-uuid"
}
```

### GET /api/health
Check ChatAI service health status.

**Response:**
```json
{
  "service": "ChatAI",
  "status": "healthy",
  "initialized": true
}
```

## Database Schema

### conversations table
- `id`: Primary key
- `conversation_id`: Unique conversation identifier
- `user_prompt`: User's input message
- `ai_response`: AI's response
- `system_prompt`: System prompt used
- `model_used`: AI model identifier
- `provider`: AI provider name
- `timestamp`: Conversation timestamp
- `rating`: User rating (if provided)
- `feedback_comment`: User feedback (if provided)

### ratings table
- `id`: Primary key
- `trace_id`: Unique rating identifier
- `conversation_id`: Reference to conversation
- `user_rating`: User's rating (1-10)
- `feedback_comment`: User's feedback text
- `timestamp`: Rating timestamp

## Integration with Unified Server

The ChatAI module is integrated into the unified server following the same pattern as ETF and Stock strategies:

1. **Router Import**: `chat_router` is imported in `unified_server.py`
2. **Initialization**: `initialize_chat_ai()` is called during server startup
3. **Cleanup**: `cleanup_chat_ai()` is called during server shutdown
4. **Health Check**: ChatAI status is included in the main health check endpoint

## Customization

### AI Integration
To integrate with your actual AI backend, modify the `_generate_sample_response()` method in `chat_core.py` to call your AI service instead of returning sample responses.

### Database
The ChatAI uses SQLite by default. You can modify the database connection in `chat_core.py` to use other databases like PostgreSQL or MySQL.

### Response Format
The current implementation uses a structured format with emojis and sections. You can modify the response format by updating the system prompt or response processing logic.

## Usage

1. Start the unified server: `python unified_server.py`
2. The ChatAI endpoints will be available at `http://localhost:8000/api/`
3. Frontend can now use the unified server endpoints instead of separate ChatAI server

## Dependencies

- FastAPI
- Pydantic
- SQLite3 (built-in)
- UUID (built-in)
- Datetime (built-in)

