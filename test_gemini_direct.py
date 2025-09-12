#!/usr/bin/env python3
"""
Test the Gemini assistant directly to debug issues
"""

import sys
import os
sys.path.append('backend')

from gemini_assistant import GeminiAssistant
import json

def test_gemini_direct():
    """Test Gemini assistant directly"""
    
    print("🔍 Testing Gemini Assistant Directly...")
    print("=" * 50)
    
    # Initialize Gemini assistant
    assistant = GeminiAssistant()
    
    print(f"Is configured: {assistant.is_configured}")
    print(f"API key present: {bool(assistant.api_key)}")
    print(f"Model: {assistant.model}")
    print()
    
    if not assistant.is_configured:
        print("❌ Gemini is not configured properly")
        return False
    
    # Test a simple query
    test_query = "What crops should I grow in Punjab?"
    
    context_data = {
        'location': 'Punjab (30.3648, 76.3424)',
        'soil_data': {'ph': 7.0, 'nitrogen': 120, 'phosphorus': 40, 'potassium': 50},
        'weather_data': {'temperature': 25, 'humidity': 70, 'rainfall': 60}
    }
    
    print(f"Test query: {test_query}")
    print(f"Context: {context_data}")
    print()
    
    try:
        result = assistant.process_query(test_query, context_data=context_data)
        
        print("✅ Gemini Response:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_direct()
    if success:
        print("\n🎉 Gemini assistant is working!")
    else:
        print("\n❌ Gemini assistant has issues")