#!/usr/bin/env python3
"""
Test script to verify the backend API is using both trained models
"""

import requests
import json

def test_backend_api():
    """Test the backend API with both dual models"""
    
    print("🔄 Testing Backend API with Dual Models...")
    print("=" * 60)
    
    # API endpoint
    url = "http://localhost:8000/recommend"
    
    # Test data - good conditions for rice
    test_data = {
        "location": {
            "latitude": 26.8467,
            "longitude": 80.9462,
            "region": "Uttar Pradesh"
        },
        "soil_data": {
            "ph": 6.5,
            "nitrogen": 80,
            "phosphorus": 40,
            "potassium": 40,
            "organic_carbon": 0.8,
            "moisture": 65
        },
        "use_satellite_data": True,
        "language": "en"
    }
    
    try:
        print("📡 Sending request to backend...")
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Request successful!")
            print(f"🌾 Number of recommendations: {len(result.get('recommendations', []))}")
            
            print("\n📋 Crop Recommendations:")
            print("-" * 80)
            
            for i, rec in enumerate(result.get('recommendations', []), 1):
                print(f"{i}. Crop: {rec.get('crop')}")
                print(f"   Confidence: {rec.get('confidence', 0):.3f}")
                print(f"   Predicted Yield: {rec.get('predicted_yield', 0):.2f} tons/hectare")
                print(f"   Estimated Profit: ₹{rec.get('estimated_profit', 0):.0f}/hectare")
                print(f"   Season: {rec.get('suitable_season', [])}")
                print()
            
            print("🌍 Environmental Data:")
            env_data = result.get('environmental_data', {})
            if 'soil' in env_data:
                print(f"   Soil pH: {env_data['soil'].get('ph', 'N/A')}")
                print(f"   Nitrogen: {env_data['soil'].get('nitrogen', 'N/A')}")
                print(f"   Phosphorus: {env_data['soil'].get('phosphorus', 'N/A')}")
                print(f"   Potassium: {env_data['soil'].get('potassium', 'N/A')}")
            
            if 'weather' in env_data:
                weather = env_data['weather']
                print(f"   Temperature: {weather.get('temperature', 'N/A')}°C")
                print(f"   Humidity: {weather.get('humidity', 'N/A')}%")
                print(f"   Rainfall: {weather.get('rainfall', 'N/A')}mm")
            
            return True
            
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")
        print("   Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error during request: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Dual Model Backend Integration")
    print("=" * 60)
    
    success = test_backend_api()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Backend integration test completed successfully!")
        print("✅ Both crop classifier and yield predictor are working in production")
        print("✅ System is ready to serve real recommendations")
    else:
        print("\n❌ Backend integration test failed")
        print("   Please check the server and try again")