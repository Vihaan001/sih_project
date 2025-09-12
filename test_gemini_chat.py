"""
Test Google Gemini Chatbot with Session Context
Demonstrates the enhanced chatbot functionality with context awareness
"""

import requests
import json
from typing import Dict, Any

# API base URL
API_URL = "http://localhost:8000"

def test_gemini_chat():
    """Test the Gemini chat functionality with session context"""
    print("🤖 Google Gemini Agricultural Chatbot Test")
    print("=" * 70)
    
    # Test location and context
    test_location = {
        "latitude": 23.3441,
        "longitude": 85.3096,
        "region": "Ranchi, Jharkhand"
    }
    
    # First, get crop recommendations to establish context
    print("\n📍 Step 1: Getting crop recommendations for context...")
    rec_response = requests.post(
        f"{API_URL}/recommend",
        json={
            "location": test_location,
            "language": "en"
        }
    )
    
    if rec_response.status_code == 200:
        rec_data = rec_response.json()
        print(f"✅ Got recommendations: {[r['crop'] for r in rec_data.get('recommendations', [])]}")
    else:
        print(f"❌ Failed to get recommendations: {rec_response.status_code}")
        rec_data = {}
    
    # Test conversation flow with Gemini
    print("\n💬 Step 2: Testing Gemini Chat with Context")
    print("-" * 70)
    
    # Initial query without session
    queries = [
        {
            "query": "Based on my location and soil conditions, which crop should I grow? How much fertilizer will I need?",
            "description": "Initial query with location context"
        },
        {
            "query": "How should I apply urea for rice cultivation? What's the best timing?",
            "description": "Follow-up about fertilizer application"
        },
        {
            "query": "My rice leaves are turning yellow in patches. What could be the problem and how to solve it?",
            "description": "Disease/pest related query"
        },
        {
            "query": "Given the current weather conditions, how should I manage irrigation for my rice field?",
            "description": "Weather-based irrigation advice"
        },
        {
            "query": "What's the current market price for rice? Should I store or sell immediately after harvest?",
            "description": "Market and economic advice"
        }
    ]
    
    session_id = None
    
    for i, query_data in enumerate(queries, 1):
        print(f"\n🔹 Query {i}: {query_data['description']}")
        print(f"❓ Question: {query_data['query']}")
        
        # Prepare request
        request_data = {
            "query": query_data["query"],
            "location": test_location,
            "language": "en"
        }
        
        # Add session ID if available (maintains conversation context)
        if session_id:
            request_data["session_id"] = session_id
        
        # Add crop recommendations as context for first query
        if i == 1 and rec_data.get('recommendations'):
            request_data["context"] = {
                "crop_recommendations": rec_data['recommendations'][:3]
            }
        
        # Send request to Gemini endpoint
        try:
            response = requests.post(
                f"{API_URL}/chat/gemini",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Store session ID for subsequent queries
                if not session_id and result.get('session_id'):
                    session_id = result['session_id']
                    print(f"📝 Session created: {session_id[:16]}...")
                
                print(f"✅ Response received")
                print(f"Context used: {result.get('context_used', False)}")
                
                # Show response (truncated for readability)
                response_text = result.get('response', '')
                if len(response_text) > 500:
                    print(f"\n📖 Answer: {response_text[:500]}...")
                else:
                    print(f"\n📖 Answer: {response_text}")
                
                # Show structured advice if available
                if result.get('structured_advice'):
                    print(f"\n📋 Structured Advice:")
                    for key, value in result['structured_advice'].items():
                        print(f"   • {key}: {value}")
                
            elif response.status_code == 503:
                print("⚠️ Gemini AI not configured. Please add GEMINI_API_KEY to .env file")
                print("Get your free API key from: https://makersuite.google.com/app/apikey")
                break
            else:
                print(f"❌ Error: {response.status_code}")
                error_detail = response.json().get('detail', 'Unknown error')
                print(f"   {error_detail}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Please run: python backend/main.py")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Get session summary if session was created
    if session_id:
        print("\n" + "=" * 70)
        print("📊 Session Summary")
        print("-" * 70)
        
        try:
            summary_response = requests.get(f"{API_URL}/chat/session/{session_id}")
            if summary_response.status_code == 200:
                summary = summary_response.json()
                print(f"Session ID: {summary.get('session_id', 'N/A')}")
                print(f"Total interactions: {summary.get('total_interactions', 0)}")
                print(f"Location: {summary.get('location', 'N/A')}")
                print(f"Has soil data: {summary.get('has_soil_data', False)}")
                print(f"Has weather data: {summary.get('has_weather_data', False)}")
                print(f"Recommended crops: {summary.get('recommended_crops', [])}")
                print(f"Topics discussed: {', '.join(summary.get('topics_discussed', []))}")
        except:
            pass

def test_multilingual_gemini():
    """Test Gemini with different languages"""
    print("\n" + "=" * 70)
    print("🌐 Multilingual Gemini Chat Test")
    print("=" * 70)
    
    multilingual_queries = [
        {
            "query": "How to increase wheat yield?",
            "language": "en",
            "description": "English"
        },
        {
            "query": "गेहूं की खेती के लिए कितना उर्वरक चाहिए?",
            "language": "auto",
            "description": "Hindi (auto-detect)"
        },
        {
            "query": "Tell me about organic farming methods for rice",
            "language": "hi",
            "description": "English query, Hindi response"
        }
    ]
    
    for query_data in multilingual_queries:
        print(f"\n🔹 {query_data['description']}")
        print(f"❓ Query: {query_data['query']}")
        
        request_data = {
            "query": query_data["query"],
            "language": query_data["language"],
            "location": {
                "latitude": 23.3441,
                "longitude": 85.3096,
                "region": "Ranchi"
            }
        }
        
        try:
            response = requests.post(f"{API_URL}/chat/gemini", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Language: {result.get('language', 'N/A')}")
                response_text = result.get('response', '')[:300]
                print(f"📖 Response: {response_text}...")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run all tests"""
    print("🌾 CROP RECOMMENDATION SYSTEM - GEMINI CHATBOT TEST")
    print("=" * 70)
    print("This test demonstrates the Google Gemini AI integration with:")
    print("• Session-based context management")
    print("• Real-time soil and weather data integration")
    print("• Crop recommendations context")
    print("• Detailed agricultural advice")
    print("• Multilingual support")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code != 200:
            print("\n❌ API is not responding. Please start the backend first.")
            print("Run: python backend/main.py")
            return
    except:
        print("\n❌ Cannot connect to API. Please run: python backend/main.py")
        return
    
    # Run tests
    test_gemini_chat()
    test_multilingual_gemini()
    
    print("\n" + "=" * 70)
    print("✅ Gemini Chatbot Test Complete!")
    print("=" * 70)
    print("\n📌 Key Features Demonstrated:")
    print("• Context-aware responses using location, soil, and weather data")
    print("• Session management maintaining conversation history")
    print("• Detailed fertilizer and cultivation advice")
    print("• Disease diagnosis and treatment recommendations")
    print("• Market and economic guidance")
    print("• Multilingual query and response support")
    
    print("\n💡 To use in your application:")
    print("1. Get a free Gemini API key from: https://makersuite.google.com/app/apikey")
    print("2. Add to .env file: GEMINI_API_KEY=your_key_here")
    print("3. Install dependency: pip install google-generativeai")
    print("4. Use /chat/gemini endpoint with session_id for context")

if __name__ == "__main__":
    main()
