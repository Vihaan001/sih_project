#!/usr/bin/env python3
"""
Debug the /recommend endpoint error
"""

import requests
import json

def test_recommend_endpoint():
    """Test the /recommend endpoint to see the exact error"""
    
    # Start the backend first
    print("Testing /recommend endpoint...")
    
    url = "http://localhost:8000/recommend"
    
    # Test data that frontend sends
    test_data = {
        "location": {
            "latitude": 23.3441,
            "longitude": 85.3096,
            "region": "Jharkhand"
        },
        "language": "en"
    }
    
    try:
        print("Sending request to:", url)
        print("Request data:", json.dumps(test_data, indent=2))
        
        response = requests.post(
            url,
            headers={'Content-Type': 'application/json'},
            json=test_data,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print("Response:", json.dumps(result, indent=2))
        else:
            print("ERROR!")
            print("Response text:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to backend server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_recommend_endpoint()