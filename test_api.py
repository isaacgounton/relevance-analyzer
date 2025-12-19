#!/usr/bin/env python3
"""
Comprehensive test script for the improved Embed Relevance Analyzer API
Tests with real data similar to what Claude processed correctly
"""
import requests
import json

def test_scenarios():
    """Test multiple scenarios that Claude handled correctly"""

    # Scenario 1: Official Bank of Canada tweet (should be KEEP)
    test_case_1 = {
        "name": "Official Bank of Canada Tweet",
        "data": {
            "articleTitle": "Banque du Canada maintient son taux directeur",
            "articleSummary": "La Banque du Canada a décidé de maintenir son taux directeur à 5% lors de sa réunion mensuelle, citant la stabilité économique.",
            "embedType": "tweet",
            "embedContent": "Tweet officiel de la Banque du Canada confirmant exactement le contenu de l'article sur le maintien du taux directeur",
            "articleText": "La Banque du Canada maintient son taux directeur à 5%. Cette décision reflète l'équilibre entre l'inflation et la croissance économique, selon la gouverneure Tiff Macklem."
        },
        "expected": True,
        "reason": "This should be kept - it's the official source confirming the article content"
    }

    # Scenario 2: Trump comments about Canada (should be KEEP)
    test_case_2 = {
        "name": "Trump Comments on Trade Relations",
        "data": {
            "articleTitle": "Tensions commerciales entre le Canada et les États-Unis",
            "articleSummary": "Les récents commentaires de Donald Trump sur les tarifs douaniers créent des incertitudes pour les entreprises canadiennes.",
            "embedType": "video",
            "embedContent": "Trump comments about Canada trade relations and negotiations during press conference",
            "articleText": "Les tensions commerciales s'intensifient après que Donald Trump ait menacé d'imposer de nouveaux tarifs sur les produits canadiens. Les négociateurs canadiens se préparent à des discussions difficiles."
        },
        "expected": True,
        "reason": "This should be kept - directly referenced in the article about trade relations"
    }

    # Scenario 3: Unrelated tweet (should be REJECT)
    test_case_3 = {
        "name": "Unrelated Tweet",
        "data": {
            "articleTitle": "Banque du Canada maintient son taux directeur",
            "articleSummary": "La Banque du Canada a décidé de maintenir son taux directeur à 5% lors de sa réunion mensuelle.",
            "embedType": "tweet",
            "embedContent": "Random tweet about celebrity gossip and entertainment news unrelated to finance or banking",
            "articleText": "La Banque du Canada maintient son taux directeur à 5%. Cette décision reflète l'équilibre entre l'inflation et la croissance économique."
        },
        "expected": False,
        "reason": "This should be rejected - completely unrelated topic"
    }

    # Scenario 4: Economic analysis video (should be KEEP)
    test_case_4 = {
        "name": "Economic Analysis Video",
        "data": {
            "articleTitle": "Décision de la Banque du Canada : impact sur l'hypothèque",
            "articleSummary": "Les experts analysent les conséquences du maintien du taux directeur sur le marché immobilier canadien.",
            "embedType": "video",
            "embedContent": "Financial analyst explaining how Bank of Canada interest rate decisions affect mortgage rates and housing market",
            "articleText": "La décision de la Banque du Canada de maintenir le taux directeur à 5% aura des répercussions directes sur les taux hypothécaires. Les propriétaires et acheteurs potentiels doivent anticiper les changements dans le marché immobilier."
        },
        "expected": True,
        "reason": "This should be kept - directly related to the article's topic"
    }

    return [test_case_1, test_case_2, test_case_3, test_case_4]

def test_single_case(test_case):
    """Test a single test case"""
    try:
        response = requests.post(
            "http://localhost:8000/explain-relevance",
            json=test_case["data"],
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            success = result["keep"] == test_case["expected"]

            print(f"\n{'='*60}")
            print(f"🧪 Testing: {test_case['name']}")
            print(f"{'='*60}")
            print(f"✅ API Success: {result['keep']}")
            print(f"📊 Final Score: {result['final_score']}")
            print(f"🎯 Expected: {test_case['expected']}")
            print(f"🔍 Result: {'✅ CORRECT' if success else '❌ INCORRECT'}")
            print(f"\n📋 Scores: {json.dumps(result['scores'], indent=2)}")
            print(f"\n📝 Explanation:")
            print(result['explanation'])
            print(f"\n💡 Test Case Reason: {test_case['reason']}")

            return success
        else:
            print(f"❌ API Error for {test_case['name']}: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Test failed for {test_case['name']}: {str(e)}")
        return False

def test_api():
    """Run comprehensive tests"""
    print("🚀 Starting Comprehensive Relevance Analyzer Tests")
    print("Testing scenarios similar to what Claude processed correctly...")

    test_cases = test_scenarios()
    results = []

    for test_case in test_cases:
        success = test_single_case(test_case)
        results.append((test_case['name'], success))

    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! The improved analyzer works correctly.")
    else:
        print("⚠️  Some tests failed. The analyzer may need further tuning.")

if __name__ == "__main__":
    test_api()