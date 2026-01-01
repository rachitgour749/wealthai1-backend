/""
Intent Classification Test Suite

Tests all 4 intent types with various query patterns.
"""

import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"
HEADERS = {
    "X-Tenant-ID": "demo",
    "X-Session-ID": "intent_test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
}

# Test queries for each intent type
TEST_QUERIES = {
    "PRODUCT": [
        "What is HDFC Multi Asset Fund?",
        "Tell me about the features of Acko Personal Health Policy",
        "What is the expense ratio of HDFC Top 100 fund?",
        "Show me details of ICICI Bluechip Fund",
        "Explain the benefits of Arogya Sanjeevani policy"
    ],
    "CLIENT": [
        "What is Priya's current portfolio?",
        "Show me Rahul's holdings",
        "What is the NAV of Amit's mutual funds?",
        "How much has Sneha invested in HDFC funds?",
        "What are the details of client Rohan?"
    ],
    "GENERAL": [
        "What is ELSS?",
        "Explain term insurance",
        "What is NAV in mutual funds?",
        "How does health insurance work?",
        "What is the difference between equity and debt funds?"
    ],
    "COMPLEX": [
        "Compare HDFC Multi Asset Fund with SBI Balanced Fund",
        "Which is better: Acko health policy or Star Health Premier?",
        "Compare HDFC Top 100 with ICICI Bluechip",
        "Should I invest in ELSS or PPF for tax saving?",
        "Compare Priya's portfolio with HDFC Balanced Advantage Fund"
    ]
}

def test_query(query: str, expected_intent: str):
    """Test a single query and return results"""
    try:
        response = requests.post(
            f"{API_BASE}/api/query",
            headers=HEADERS,
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            actual_intent = data.get("intent", "unknown")
            is_correct = actual_intent.lower() == expected_intent.lower()
            
            return {
                "query": query,
                "expected": expected_intent,
                "actual": actual_intent,
                "correct": is_correct,
                "confidence": data.get("confidence", 0),
                "response_preview": data.get("response", "")[:100] + "..."
            }
        else:
            return {
                "query": query,
                "expected": expected_intent,
                "actual": "ERROR",
                "correct": False,
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "query": query,
            "expected": expected_intent,
            "actual": "ERROR",
            "correct": False,
            "error": str(e)
        }

def run_all_tests():
    """Run all intent classification tests"""
    print("=" * 80)
    print("INTENT CLASSIFICATION TEST SUITE")
    print("=" * 80)
    print()
    
    all_results = {}
    total_tests = 0
    correct_tests = 0
    
    for intent_type, queries in TEST_QUERIES.items():
        print(f"\n📊 Testing {intent_type} Intent ({len(queries)} queries)")
        print("-" * 80)
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Testing: {query[:60]}...")
            result = test_query(query, intent_type)
            results.append(result)
            
            total_tests += 1
            if result["correct"]:
                correct_tests += 1
                print(f"    ✅ Correct: {result['actual']} (confidence: {result.get('confidence', 'N/A')})")
            else:
                print(f"    ❌ WRONG: Expected {result['expected']}, got {result['actual']}")
                if 'error' in result:
                    print(f"       Error: {result['error']}")
        
        all_results[intent_type] = results
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    accuracy = (correct_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Correct: {correct_tests}")
    print(f"Wrong: {total_tests - correct_tests}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    print("\n\nBreakdown by Intent Type:")
    print("-" * 80)
    
    for intent_type, results in all_results.items():
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        status = "✅" if accuracy >= 80 else "⚠️" if accuracy >= 60 else "❌"
        print(f"{status} {intent_type}: {correct}/{total} correct ({accuracy:.1f}%)")
    
    # Detailed errors
    errors = []
    for intent_type, results in all_results.items():
        for result in results:
            if not result["correct"]:
                errors.append({
                    "expected": intent_type,
                    "query": result["query"],
                    "actual": result["actual"],
                    "error": result.get("error")
                })
    
    if errors:
        print("\n\n⚠️ CLASSIFICATION ERRORS:")
        print("-" * 80)
        for i, error in enumerate(errors, 1):
            print(f"\n{i}. Query: {error['query']}")
            print(f"   Expected: {error['expected']}")
            print(f"   Got: {error['actual']}")
            if error.get("error"):
                print(f"   Error: {error['error']}")
    
    # Save results
    with open("intent_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n📄 Detailed results saved to: intent_test_results.json")
    print("=" * 80)
    
    return all_results

if __name__ == "__main__":
    results = run_all_tests()
