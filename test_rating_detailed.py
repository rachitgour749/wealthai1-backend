#!/usr/bin/env python3
"""
Detailed Rating API Testing Script
"""

import requests
import json
import time
from datetime import datetime

# Server configuration
BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"
RATE_ENDPOINT = f"{BASE_URL}/api/rate"

def test_rating_api_detailed():
    """Comprehensive rating API testing"""
    print("🧪 Detailed Rating API Testing")
    print("=" * 50)
    
    # Test 1: Get a response_id first
    print("1. Getting response_id from chat...")
    try:
        chat_response = requests.post(CHAT_ENDPOINT, json={
            "prompt": "What is an ETF?",
            "system_prompt": "You are a ChatGPT-style financial expert."
        })
        
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            response_id = chat_data.get('response_id')
            print(f"✅ Got response_id: {response_id}")
        else:
            print(f"❌ Chat request failed: {chat_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting response_id: {e}")
        return
    
    # Test 2: Test valid ratings
    print("\n2. Testing valid ratings...")
    valid_ratings = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for rating in valid_ratings:
        try:
            response = requests.post(RATE_ENDPOINT, json={
                "trace_id": response_id,
                "user_rating": rating,
                "feedback_comment": f"Test rating {rating} - {datetime.now().strftime('%H:%M:%S')}"
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Rating {rating}: {data.get('success', False)} - {data.get('message', 'N/A')}")
            else:
                print(f"❌ Rating {rating} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing rating {rating}: {e}")
    
    # Test 3: Test invalid ratings
    print("\n3. Testing invalid ratings...")
    invalid_ratings = [0, 11, -1, 100, "invalid", None]
    
    for rating in invalid_ratings:
        try:
            response = requests.post(RATE_ENDPOINT, json={
                "trace_id": response_id,
                "user_rating": rating,
                "feedback_comment": f"Invalid rating test: {rating}"
            })
            
            print(f"Rating {rating}: Status {response.status_code}")
            if response.status_code != 200:
                print(f"   Expected error for invalid rating {rating}")
        except Exception as e:
            print(f"❌ Error testing invalid rating {rating}: {e}")
    
    # Test 4: Test without trace_id
    print("\n4. Testing without trace_id...")
    try:
        response = requests.post(RATE_ENDPOINT, json={
            "user_rating": 5,
            "feedback_comment": "Test without trace_id"
        })
        print(f"Without trace_id: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing without trace_id: {e}")
    
    # Test 5: Test with invalid trace_id
    print("\n5. Testing with invalid trace_id...")
    try:
        response = requests.post(RATE_ENDPOINT, json={
            "trace_id": "invalid-uuid-123",
            "user_rating": 5,
            "feedback_comment": "Test with invalid trace_id"
        })
        print(f"Invalid trace_id: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid trace_id: {e}")
    
    # Test 6: Test with long feedback comment
    print("\n6. Testing with long feedback comment...")
    long_comment = "This is a very long feedback comment " * 10
    try:
        response = requests.post(RATE_ENDPOINT, json={
            "trace_id": response_id,
            "user_rating": 7,
            "feedback_comment": long_comment
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Long comment: {data.get('success', False)}")
        else:
            print(f"❌ Long comment failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing long comment: {e}")
    
    # Test 7: Test multiple ratings for same response
    print("\n7. Testing multiple ratings for same response...")
    for i in range(3):
        try:
            response = requests.post(RATE_ENDPOINT, json={
                "trace_id": response_id,
                "user_rating": 8 + i,
                "feedback_comment": f"Multiple rating test {i+1}"
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Multiple rating {i+1}: {data.get('success', False)}")
            else:
                print(f"❌ Multiple rating {i+1} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing multiple rating {i+1}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Detailed rating API testing completed!")

def test_rating_performance():
    """Test rating API performance"""
    print("\n🚀 Performance Testing")
    print("=" * 30)
    
    # Get a response_id
    try:
        chat_response = requests.post(CHAT_ENDPOINT, json={
            "prompt": "What is a bond?",
            "system_prompt": "You are a ChatGPT-style financial expert."
        })
        response_id = chat_response.json().get('response_id')
    except:
        print("❌ Failed to get response_id for performance test")
        return
    
    # Test multiple rapid ratings
    start_time = time.time()
    success_count = 0
    
    for i in range(10):
        try:
            response = requests.post(RATE_ENDPOINT, json={
                "trace_id": response_id,
                "user_rating": (i % 10) + 1,
                "feedback_comment": f"Performance test {i+1}"
            })
            
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Performance test completed:")
    print(f"   Success rate: {success_count}/10")
    print(f"   Total time: {duration:.2f} seconds")
    print(f"   Average time per request: {duration/10:.3f} seconds")

if __name__ == "__main__":
    test_rating_api_detailed()
    test_rating_performance()

