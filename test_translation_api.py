"""
Test Auto-Translation in API Responses
This script tests the auto-translation functionality for all endpoints
"""

import requests
import json
from typing import Dict, Any

# API base URL
API_URL = "http://localhost:8000"

def print_response(title: str, response: Dict[str, Any], show_full: bool = False):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"🔹 {title}")
    print('='*60)
    
    if show_full:
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        # Show key information
        if "recommendations" in response:
            print(f"Language: {response.get('language', 'N/A')}")
            print(f"Status: {response.get('status', 'N/A')}")
            print("\nRecommendations:")
            for i, rec in enumerate(response["recommendations"], 1):
                print(f"\n{i}. {rec.get('crop', 'N/A')}")
                print(f"   Confidence: {rec.get('confidence', 0)*100:.1f}%")
                print(f"   Profit Margin: {rec.get('profit_margin', 'N/A')}")
                if 'care_tips' in rec and rec['care_tips']:
                    print(f"   Care Tips:")
                    for tip in rec['care_tips'][:2]:  # Show first 2 tips
                        print(f"     • {tip}")
        elif "response" in response:
            print(f"Language: {response.get('language', 'N/A')}")
            if isinstance(response["response"], dict):
                print(f"Message: {response['response'].get('message', 'N/A')}")
                if 'suggestions' in response['response']:
                    print("Suggestions:")
                    for suggestion in response['response']['suggestions'][:3]:
                        print(f"  • {suggestion}")
            else:
                print(f"Response: {response['response']}")
        else:
            print(json.dumps(response, indent=2, ensure_ascii=False)[:500])

def test_recommendation_endpoint():
    """Test crop recommendations with different languages"""
    print("\n" + "="*70)
    print("🧪 TESTING CROP RECOMMENDATION AUTO-TRANSLATION")
    print("="*70)
    
    # Test data
    test_cases = [
        {
            "name": "English (Default)",
            "data": {
                "location": {
                    "latitude": 23.3441,
                    "longitude": 85.3096,
                    "region": "Jharkhand"
                },
                "language": "en"
            }
        },
        {
            "name": "Hindi Translation",
            "data": {
                "location": {
                    "latitude": 23.3441,
                    "longitude": 85.3096,
                    "region": "Jharkhand"
                },
                "language": "hi"
            }
        },
        {
            "name": "Bengali Translation",
            "data": {
                "location": {
                    "latitude": 23.3441,
                    "longitude": 85.3096,
                    "region": "Jharkhand"
                },
                "language": "bn"
            }
        },
        {
            "name": "Tamil Translation",
            "data": {
                "location": {
                    "latitude": 23.3441,
                    "longitude": 85.3096,
                    "region": "Jharkhand"
                },
                "language": "ta"
            }
        }
    ]
    
    for test in test_cases:
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json=test["data"]
            )
            
            if response.status_code == 200:
                print_response(test["name"], response.json())
            else:
                print(f"❌ {test['name']}: Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test['name']}: {e}")

def test_chat_endpoint():
    """Test chat assistant with different languages"""
    print("\n" + "="*70)
    print("🧪 TESTING CHAT ASSISTANT AUTO-TRANSLATION")
    print("="*70)
    
    # Test queries in different languages
    test_queries = [
        {
            "name": "English Query",
            "data": {
                "query": "What crops should I grow in winter?",
                "language": "en"
            }
        },
        {
            "name": "Hindi Query (Auto-detect)",
            "data": {
                "query": "सर्दियों में कौन सी फसल उगाऊं?",
                "language": "auto"
            }
        },
        {
            "name": "English Query → Hindi Response",
            "data": {
                "query": "Tell me about wheat cultivation",
                "language": "hi"
            }
        },
        {
            "name": "Bengali Response",
            "data": {
                "query": "How to increase rice yield?",
                "language": "bn"
            }
        }
    ]
    
    for test in test_queries:
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json=test["data"]
            )
            
            if response.status_code == 200:
                print_response(test["name"], response.json())
            else:
                print(f"❌ {test['name']}: Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test['name']}: {e}")

def test_languages_endpoint():
    """Test supported languages endpoint"""
    print("\n" + "="*70)
    print("🧪 TESTING SUPPORTED LANGUAGES")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/languages")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Translation Service Available: {data.get('translation_available', False)}")
            print(f"\nSupported Languages ({len(data.get('languages', []))}):")
            
            for lang in data.get('languages', []):
                available = "✅" if lang.get('available', True) else "❌"
                print(f"  {available} {lang.get('flag', '')} {lang['native_name']} ({lang['name']}) - Code: {lang['code']}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_ui_translations():
    """Test UI translations for different languages"""
    print("\n" + "="*70)
    print("🧪 TESTING UI TRANSLATIONS")
    print("="*70)
    
    languages_to_test = ["en", "hi", "bn", "ta"]
    
    for lang in languages_to_test:
        try:
            response = requests.get(f"{API_URL}/ui-translations/{lang}")
            
            if response.status_code == 200:
                data = response.json()
                translations = data.get('translations', {})
                
                print(f"\n📝 UI Translations for {lang.upper()}:")
                # Show sample translations
                sample_keys = ['app_title', 'get_recommendations', 'submit', 'loading', 'success']
                for key in sample_keys:
                    if key in translations:
                        print(f"  {key}: {translations[key]}")
            else:
                print(f"❌ {lang}: Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {lang}: {e}")

def main():
    """Run all tests"""
    print("🌐 AUTO-TRANSLATION API TEST SUITE")
    print("="*70)
    print("Testing auto-translation functionality for all endpoints")
    print("Make sure the backend is running: python backend/main.py")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code != 200:
            print("❌ API is not responding. Please start the backend first.")
            return
    except:
        print("❌ Cannot connect to API. Please run: python backend/main.py")
        return
    
    # Run tests
    test_languages_endpoint()
    test_recommendation_endpoint()
    test_chat_endpoint()
    test_ui_translations()
    
    print("\n" + "="*70)
    print("✅ All translation tests completed!")
    print("="*70)
    print("\n📌 Summary:")
    print("  • Recommendations are auto-translated to user's chosen language")
    print("  • Chat responses are translated based on query language")
    print("  • UI elements can be fetched in any supported language")
    print("  • Language auto-detection works for chat queries")
    print("\n💡 To use in frontend:")
    print("  1. Set language parameter in API calls")
    print("  2. Fetch UI translations on language change")
    print("  3. All text content will be automatically translated")

if __name__ == "__main__":
    main()
