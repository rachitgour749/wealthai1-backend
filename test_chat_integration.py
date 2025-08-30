#!/usr/bin/env python3
"""
Test script for ChatAI integration with unified server
"""

import requests
import json
import time

# Server configuration
BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"
RATE_ENDPOINT = f"{BASE_URL}/api/rate"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

def test_server_health():
    """Test if the server is running and healthy"""
    print("🔍 Testing server health...")
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy!")
            print(f"   Stock Backtester: {data.get('stock_backtester_initialized', False)}")
            print(f"   ETF Backtester: {data.get('etf_backtester_initialized', False)}")
            print(f"   ChatAI: {data.get('chat_ai_initialized', False)}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n💬 Testing chat endpoint...")
    try:
        payload = {
            "prompt": "What is a mutual fund?",
            "system_prompt": "You are a ChatGPT-style financial expert. FORMAT: Start with 📚 DEFINITION (30 words max), then 💡 KEY POINTS (1 line each), add 🎯 EXAMPLE (1-2 lines), end with ✅ PRACTICAL TAKEAWAY (1 line). Use emojis, bullet points, and clear sections. Keep it concise but engaging like ChatGPT."
        }
        
        response = requests.post(CHAT_ENDPOINT, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat endpoint working!")
            print(f"   Response ID: {data.get('response_id', 'N/A')}")
            print(f"   Model Used: {data.get('model_used', 'N/A')}")
            print(f"   Provider: {data.get('provider', 'N/A')}")
            print(f"   Response Length: {len(data.get('response', ''))} characters")
            return data.get('response_id')
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error testing chat endpoint: {e}")
        return None

def test_rating_endpoint(response_id):
    """Test the rating endpoint"""
    if not response_id:
        print("\n⭐ Skipping rating test (no response ID)")
        return
    
    print(f"\n⭐ Testing rating endpoint with ID: {response_id}")
    try:
        payload = {
            "trace_id": response_id,
            "user_rating": 8,
            "feedback_comment": "Great explanation! Very helpful."
        }
        
        response = requests.post(RATE_ENDPOINT, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Rating endpoint working!")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Message: {data.get('message', 'N/A')}")
        else:
            print(f"❌ Rating endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing rating endpoint: {e}")

def test_chat_health():
    """Test ChatAI specific health endpoint"""
    print("\n🏥 Testing ChatAI health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ ChatAI health endpoint working!")
            print(f"   Service: {data.get('service', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Initialized: {data.get('initialized', False)}")
        else:
            print(f"❌ ChatAI health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing ChatAI health endpoint: {e}")

def main():
    """Run all tests"""
    print("🚀 Testing ChatAI Integration with Unified Server")
    print("=" * 50)
    
    # Test server health
    if not test_server_health():
        print("\n❌ Server is not running or unhealthy. Please start the server first.")
        return
    
    # Test ChatAI health
    test_chat_health()
    
    # Test chat endpoint
    response_id = test_chat_endpoint()
    
    # Test rating endpoint
    test_rating_endpoint(response_id)
    
    print("\n" + "=" * 50)
    print("✅ ChatAI integration test completed!")
    print("\nTo start the server, run:")
    print("   cd wealthai1-backend-main")
    print("   python unified_server.py")

if __name__ == "__main__":
    main()

