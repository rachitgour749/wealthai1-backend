#!/usr/bin/env python3
"""
Test specific responses from ChatAI
"""

import requests
import json

BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"

def test_specific_responses():
    """Test that ChatAI provides specific responses for different topics"""
    print("🧪 Testing Specific ChatAI Responses")
    print("=" * 50)
    
    # Test cases with expected keywords
    test_cases = [
        ("What is a bond?", "bond"),
        ("Tell me about ETFs", "etf"),
        ("How do mutual funds work?", "mutual fund"),
        ("What is diversification?", "diversification"),
        ("Explain compound interest", "compound"),
        ("What is a bull market?", "bull market"),
        ("How does retirement planning work?", "retirement"),
        ("What is market cap?", "market cap"),
        ("Explain asset allocation", "asset allocation"),
        ("What are dividends?", "dividend")
    ]
    
    for question, expected_keyword in test_cases:
        print(f"\n📝 Testing: '{question}'")
        try:
            response = requests.post(CHAT_ENDPOINT, json={
                "prompt": question,
                "system_prompt": "You are a ChatGPT-style financial expert."
            })
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '').lower()
                
                # Check if the response contains the expected keyword
                if expected_keyword in response_text:
                    print(f"✅ Found '{expected_keyword}' in response")
                else:
                    print(f"⚠️  Expected '{expected_keyword}' but not found")
                    print(f"   Response preview: {response_text[:100]}...")
                
                # Check response structure
                if "📚 DEFINITION:" in data.get('response', ''):
                    print("✅ Proper response structure with definition")
                else:
                    print("⚠️  Missing proper response structure")
                    
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test a generic question that should get a helpful response
    print(f"\n📝 Testing generic question: 'Hello'")
    try:
        response = requests.post(CHAT_ENDPOINT, json={
            "prompt": "Hello",
            "system_prompt": "You are a ChatGPT-style financial expert."
        })
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            
            if "I'm your AI financial assistant" in response_text:
                print("✅ Appropriate response for generic greeting")
            else:
                print("⚠️  Unexpected response for generic greeting")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Specific response testing completed!")

if __name__ == "__main__":
    test_specific_responses()

