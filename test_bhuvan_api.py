"""
Test your current Bhuvan API key to see what it's trying to do
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_bhuvan_api():
    """
    Test the current Bhuvan API key and see what it's trying to access
    """
    
    print("="*60)
    print("TESTING YOUR CURRENT BHUVAN API KEY")
    print("="*60)
    
    # Your current API key
    api_key = os.getenv("BHUVAN_API_KEY", "")
    print(f"API Key: {api_key}")
    
    # The URL your system is trying to access
    base_url = "https://bhuvan-vec1.nrsc.gov.in"
    soil_endpoint = f"{base_url}/api/soil/properties"
    
    print(f"\\nTrying to access: {soil_endpoint}")
    
    # Test coordinates (Ranchi, Jharkhand)
    test_lat, test_lon = 23.3441, 85.3096
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    params = {
        'lat': test_lat,
        'lon': test_lon,
        'buffer': 1000
    }
    
    print(f"\\nTest Parameters:")
    print(f"  Latitude: {test_lat}")
    print(f"  Longitude: {test_lon}")
    print(f"  Buffer: 1000m")
    
    try:
        print(f"\\nMaking API request...")
        response = requests.get(soil_endpoint, params=params, headers=headers, timeout=10)
        
        print(f"\\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Bhuvan API is working")
            data = response.json()
            print(f"Response Data: {data}")
            
        elif response.status_code == 401:
            print("❌ UNAUTHORIZED (401) - API key is invalid or expired")
            
        elif response.status_code == 403:
            print("❌ FORBIDDEN (403) - API key doesn't have permission for this endpoint")
            
        elif response.status_code == 404:
            print("❌ NOT FOUND (404) - This API endpoint doesn't exist")
            print("   The URL might be incorrect or the API has changed")
            
        elif response.status_code == 429:
            print("❌ TOO MANY REQUESTS (429) - Rate limit exceeded")
            
        else:
            print(f"❌ ERROR ({response.status_code}) - {response.text}")
            
        # Try to get more info about the error
        print(f"\\nFull Response Text:")
        print(response.text[:500])  # First 500 characters
        
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - The API didn't respond within 10 seconds")
        
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Cannot reach the Bhuvan server")
        print("   The server might be down or the URL might be wrong")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

def check_bhuvan_documentation():
    """
    Check what Bhuvan API actually provides
    """
    print("\\n" + "="*60)
    print("WHAT YOUR BHUVAN API KEY IS SUPPOSED TO ACCESS")
    print("="*60)
    
    print("\\n🎯 PURPOSE:")
    print("Your Bhuvan API integration is designed to fetch:")
    print("  📍 Soil properties (pH, nutrients, texture)")
    print("  🌍 India-specific geospatial data") 
    print("  🛰️ Satellite-based soil information")
    print("  📊 ISRO (Indian Space Research Organisation) data")
    
    print("\\n🔗 ENDPOINT BEING ACCESSED:")
    print("  URL: https://bhuvan-vec1.nrsc.gov.in/api/soil/properties")
    print("  Method: GET")
    print("  Authentication: Bearer token")
    
    print("\\n📋 WHAT IT SHOULD RETURN:")
    print("  - Soil pH levels")
    print("  - Nutrient content (N, P, K)")
    print("  - Soil texture (sand, silt, clay)")
    print("  - Organic matter content")
    print("  - Soil moisture data")
    
    print("\\n🚨 CURRENT STATUS:")
    print("  ❌ API key may be invalid/expired")
    print("  ❌ Endpoint might have changed")
    print("  ❌ API might require different authentication")
    print("  ❌ Service might be temporarily unavailable")
    
    print("\\n💡 HOW IT FITS IN YOUR SYSTEM:")
    print("  1. User enters latitude/longitude")
    print("  2. System calls Bhuvan API for Indian soil data")
    print("  3. Combines with weather data from OpenWeather")
    print("  4. Feeds combined data to ML models")
    print("  5. Returns India-specific crop recommendations")

def suggest_alternatives():
    """
    Suggest what to do if Bhuvan API isn't working
    """
    print("\\n" + "="*60)
    print("ALTERNATIVES IF BHUVAN API ISN'T WORKING")
    print("="*60)
    
    print("\\n🔄 IMMEDIATE SOLUTIONS:")
    print("  1. ✅ Continue using SoilGrids (currently working)")
    print("     - Global coverage including India")
    print("     - Free, no authentication required")
    print("     - Already integrated and working")
    
    print("\\n  2. 🔍 Try to fix Bhuvan integration:")
    print("     - Contact ISRO/NRSC for API documentation")
    print("     - Check if endpoint URL has changed")
    print("     - Verify API key is still valid")
    print("     - Register for new API access")
    
    print("\\n  3. 🌐 Use data.gov.in instead:")
    print("     - Government soil health card data")
    print("     - District-wise agricultural statistics")
    print("     - Free with registration")
    
    print("\\n📊 CURRENT DATA FLOW (WITHOUT BHUVAN):")
    print("  User Input → SoilGrids (soil) + OpenWeather (weather) → ML Models")
    print("  ✅ This is already working perfectly in your system!")
    
    print("\\n🎯 RECOMMENDATION:")
    print("  Keep using SoilGrids as primary soil data source")
    print("  Add data.gov.in for historical Indian agricultural data")
    print("  Use Bhuvan as supplementary source when/if it works")

if __name__ == "__main__":
    test_bhuvan_api()
    check_bhuvan_documentation()
    suggest_alternatives()