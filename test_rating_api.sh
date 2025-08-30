#!/bin/bash

# Test Rating API with curl commands
echo "🧪 Testing Rating API with curl"
echo "=================================="

# Base URL
BASE_URL="http://localhost:8000"

# Step 1: Test server health
echo "1. Testing server health..."
curl -X GET "$BASE_URL/health" -H "Content-Type: application/json"
echo -e "\n"

# Step 2: Send a chat message to get a response_id
echo "2. Sending chat message to get response_id..."
CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is a stock?",
    "system_prompt": "You are a ChatGPT-style financial expert."
  }')

echo "Chat Response:"
echo "$CHAT_RESPONSE"
echo -e "\n"

# Extract response_id from the chat response
RESPONSE_ID=$(echo "$CHAT_RESPONSE" | grep -o '"response_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$RESPONSE_ID" ]; then
    echo "❌ Failed to get response_id from chat response"
    exit 1
fi

echo "📝 Response ID: $RESPONSE_ID"
echo -e "\n"

# Step 3: Test rating API with the response_id
echo "3. Testing rating API..."
curl -X POST "$BASE_URL/api/rate" \
  -H "Content-Type: application/json" \
  -d "{
    \"trace_id\": \"$RESPONSE_ID\",
    \"user_rating\": 8,
    \"feedback_comment\": \"Great explanation! Very helpful.\"
  }"
echo -e "\n"

# Step 4: Test rating API with different ratings
echo "4. Testing different ratings..."
for rating in 1 3 5 7 10; do
    echo "Testing rating: $rating"
    curl -X POST "$BASE_URL/api/rate" \
      -H "Content-Type: application/json" \
      -d "{
        \"trace_id\": \"$RESPONSE_ID\",
        \"user_rating\": $rating,
        \"feedback_comment\": \"Test rating $rating\"
      }"
    echo -e "\n"
done

echo "✅ Rating API testing completed!"

