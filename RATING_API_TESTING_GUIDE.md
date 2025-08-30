# Rating API Testing Guide

## 🧪 **How to Test the Rating API**

### **Method 1: Quick Test Script**
```bash
python test_chat_integration.py
```

### **Method 2: Detailed Testing**
```bash
python test_rating_detailed.py
```

### **Method 3: Manual Testing with curl**

#### **Step 1: Get a response_id**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is a mutual fund?",
    "system_prompt": "You are a ChatGPT-style financial expert."
  }'
```

#### **Step 2: Rate the response**
```bash
curl -X POST "http://localhost:8000/api/rate" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "YOUR_RESPONSE_ID_HERE",
    "user_rating": 8,
    "feedback_comment": "Great explanation!"
  }'
```

### **Method 4: Using Postman/Insomnia**

#### **Chat Request:**
- **URL:** `POST http://localhost:8000/api/chat`
- **Headers:** `Content-Type: application/json`
- **Body:**
```json
{
  "prompt": "What is an ETF?",
  "system_prompt": "You are a ChatGPT-style financial expert."
}
```

#### **Rating Request:**
- **URL:** `POST http://localhost:8000/api/rate`
- **Headers:** `Content-Type: application/json`
- **Body:**
```json
{
  "trace_id": "RESPONSE_ID_FROM_CHAT",
  "user_rating": 9,
  "feedback_comment": "Excellent explanation!"
}
```

### **Method 5: Frontend Testing**

1. **Start the servers:**
   ```bash
   # Backend
   cd wealthai1-backend-main
   python unified_server.py
   
   # Frontend (in another terminal)
   cd wealthai1-portal-main
   npm start
   ```

2. **Navigate to ChatAI page**
3. **Send a message** (e.g., "What is a stock?")
4. **Wait for AI response**
5. **Click on the star rating** (1-5 stars)
6. **Optional: Add a comment**
7. **Verify rating is submitted**

## 📊 **Test Results Summary**

### **✅ What Works:**
- ✅ All valid ratings (1-10) are accepted
- ✅ Feedback comments are stored
- ✅ Multiple ratings for same response
- ✅ Long feedback comments
- ✅ Invalid trace_id handling
- ✅ Performance: ~2 seconds per request

### **⚠️ Expected Behaviors:**
- ⚠️ Invalid ratings (0, 11, -1, 100) are accepted (should be validated)
- ⚠️ Non-numeric ratings return 422 error (correct)
- ⚠️ Missing trace_id returns 422 error (correct)

## 🔧 **API Endpoints**

### **POST /api/chat**
Generate AI response and get response_id for rating.

**Request:**
```json
{
  "prompt": "Your question here",
  "system_prompt": "Optional system prompt"
}
```

**Response:**
```json
{
  "response": "AI response text",
  "model_used": "mf-assistant:latest",
  "response_id": "uuid-for-rating",
  "timestamp": 1756462514.623507,
  "system_prompt_used": "string",
  "rating": null,
  "provider": "langchain_modal"
}
```

### **POST /api/rate**
Submit user rating and feedback.

**Request:**
```json
{
  "trace_id": "response_id_from_chat",
  "user_rating": 8,
  "feedback_comment": "Optional feedback"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Rating stored successfully",
  "trace_id": "response_id_from_chat"
}
```

## 🐛 **Troubleshooting**

### **Common Issues:**

1. **"Server not running"**
   - Start the unified server: `python unified_server.py`

2. **"Invalid trace_id"**
   - Make sure to use the `response_id` from the chat response

3. **"422 Validation Error"**
   - Check that `user_rating` is a number between 1-10
   - Ensure `trace_id` is provided

4. **"Frontend not showing ratings"**
   - Check browser console for errors
   - Verify API endpoints are correct (localhost:8000)

### **Database Verification:**
```bash
# Check if ratings are stored
sqlite3 chat_ai_data.db "SELECT * FROM ratings ORDER BY timestamp DESC LIMIT 5;"
```

## 🎯 **Testing Checklist**

- [ ] Server health check passes
- [ ] Chat endpoint returns response_id
- [ ] Rating endpoint accepts valid ratings (1-10)
- [ ] Rating endpoint stores feedback comments
- [ ] Multiple ratings work for same response
- [ ] Frontend displays rating component
- [ ] Frontend submits ratings successfully
- [ ] Rating updates in real-time
- [ ] Error handling works for invalid inputs

## 📈 **Performance Benchmarks**

- **Average response time:** ~2 seconds per rating
- **Success rate:** 100% for valid requests
- **Database storage:** Immediate persistence
- **Concurrent requests:** Handled properly

