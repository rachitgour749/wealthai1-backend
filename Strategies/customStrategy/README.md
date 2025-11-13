# Custom Strategy Feature

This module provides a complete custom strategy building system that allows users to describe their trading strategies in natural language and receive AI-powered analysis and implementation guidance.

## Features

- **Natural Language Processing**: Users can describe strategies in English, Hindi, or Hinglish
- **AI Analysis**: Integration with OpenAI and Google Gemini for strategy analysis
- **Timeline UI**: Step-by-step process with progress tracking
- **Email Notifications**: Automatic notifications to users and team
- **Database Storage**: Secure storage of strategy data and analysis
- **Professional Integration**: Seamlessly integrated with existing WealthAI backend and frontend

## Architecture

### Backend Components

```
Strategies/customStrategy/
├── api.py              # FastAPI endpoints
├── models.py           # Pydantic data models
├── database.py         # Database operations
├── ai_service.py       # Google Gemini AI integration
├── email_service.py    # Email notifications
└── Strategy_Master_Prompt.txt  # Master prompt template
```

### Frontend Components

```
src/components/customStrategy/
├── CustomStrategyBuilder.jsx  # Main container component
├── DescriptionInput.jsx       # Step 1: Strategy description
├── AIResponseDisplay.jsx      # Step 2: AI analysis review
└── UserDetailsForm.jsx        # Step 3: Contact information
```

## API Endpoints

### 1. Analyze Strategy
```http
POST /api/custom-strategy/analyze
Content-Type: application/json

{
  "user_id": "string",
  "user_email": "string",
  "strategy_description": "string"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "strategy_rating": 2,
    "trading_instruments": "NIFTY50",
    "time_frame": "Daily",
    "trading_rules": ["RSI < 30", "Price > 20MA"],
    "position_sizing": "2% of portfolio",
    "risk_management": "5% stop loss",
    "strategy_logic": "Momentum strategy with RSI and MA filters"
  },
  "message": "Strategy analyzed successfully"
}
```

### 2. Save Strategy
```http
POST /api/custom-strategy/save
Content-Type: application/json

{
  "user_id": "string",
  "user_email": "string",
  "user_phone": "string",
  "strategy_description": "string",
  "analysis": { ... }
}
```

### 3. Get User Strategies
```http
GET /api/custom-strategy/user/{user_id}
```

### 4. Get Strategy Details
```http
GET /api/custom-strategy/{strategy_id}
```

## Database Schema

```sql
CREATE TABLE custom_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    user_email TEXT NOT NULL,
    user_phone TEXT,
    strategy_description TEXT NOT NULL,
    ai_analysis_json TEXT NOT NULL,
    strategy_rating INTEGER,
    status TEXT DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

### Environment Variables

```bash
# AI Service Configuration
GEMINI_API_KEY=your_gemini_api_key

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
TEAM_EMAIL=team@wealthai.com
```

### Master Prompt

The `Strategy_Master_Prompt.txt` file contains the master prompt that is combined with user descriptions to generate comprehensive strategy analysis. The prompt instructs the AI to:

1. Identify trading instruments
2. Determine timeframes
3. Extract trading rules
4. Analyze position sizing
5. Evaluate risk management
6. Generate strategy logic
7. Rate complexity (1-4 scale)

## Usage Flow

### 1. User Journey
1. User clicks "Build Your Custom Strategy" button
2. User describes their strategy in natural language
3. AI analyzes the description and generates structured analysis
4. User reviews and can edit the AI analysis
5. User provides contact information
6. Strategy is saved and email notifications are sent

### 2. Team Workflow
1. Team receives email notification with strategy details
2. Team reviews strategy complexity and requirements
3. Team contacts user within 24-48 hours
4. Team discusses implementation and next steps

## Testing

Run the test script to verify integration:

```bash
cd wealthai-backend2
python test_custom_strategy.py
```

## Integration Points

### Backend Integration
- Added to `server.py` with router inclusion
- Uses existing `unified_etf_data.sqlite` database
- Follows existing API patterns and error handling

### Frontend Integration
- Added route to `App.jsx`
- Updated "Build Your Custom Strategy" button in `MarketsAI1App.jsx`
- Uses existing UI components and styling patterns

## Error Handling

- Comprehensive error handling for AI service failures
- Fallback from Gemini to OpenAI if needed
- Input validation and sanitization
- Graceful degradation for email service failures

## Security

- Input validation and sanitization
- SQL injection prevention
- Email validation
- Secure API key management
- User data encryption

## Future Enhancements

- Strategy backtesting integration
- Real-time strategy monitoring
- Advanced AI model fine-tuning
- Multi-language support expansion
- Strategy performance tracking
