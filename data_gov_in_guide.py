"""
Data.gov.in API Integration Guide
Specific APIs and their usage in the crop recommendation system
"""

# The current implementation uses this specific dataset:
# URL: https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
# This is the "District wise, season wise crop production statistics" dataset

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def explore_data_gov_apis():
    """
    Explore available data.gov.in APIs for agriculture
    """
    
    print("="*60)
    print("DATA.GOV.IN AGRICULTURAL APIs - DETAILED GUIDE")
    print("="*60)
    
    # 1. Current Implementation - Crop Production Statistics
    print("\n🌾 1. CURRENTLY IMPLEMENTED:")
    print("-" * 40)
    print("Dataset: District wise, season wise crop production statistics")
    print("Resource ID: 9ef84268-d588-465a-a308-a864a43d0070")
    print("URL: https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070")
    print("Data Type: Historical crop yield, production, area data")
    print("Filters: State, district, crop, season, year")
    print("Usage in System: Training data for yield prediction models")
    
    # 2. Additional Available APIs
    print("\n🌍 2. OTHER AGRICULTURAL APIs AVAILABLE:")
    print("-" * 40)
    
    available_apis = {
        "Rainfall Data": {
            "resource_id": "88a2e56f-a5c0-4521-91ed-f4b16f5f4df3",
            "description": "State and district wise rainfall data",
            "use_case": "Weather patterns for crop recommendations"
        },
        "Soil Health Card": {
            "resource_id": "33e5c2ea-7b61-43df-a976-2c3c9c11e14c", 
            "description": "District wise soil health information",
            "use_case": "Soil nutrient data for ML models"
        },
        "Market Prices": {
            "resource_id": "9ef84268-d588-465a-a308-a864a43d0071",
            "description": "Agricultural market prices",
            "use_case": "Price prediction and profit calculations"
        },
        "Fertilizer Consumption": {
            "resource_id": "6ff0e502-b2f7-4bb3-a8d3-c30c3b7e5d3c",
            "description": "State wise fertilizer consumption",
            "use_case": "Nutrient requirement estimation"
        }
    }
    
    for name, info in available_apis.items():
        print(f"\n📊 {name}:")
        print(f"   Resource ID: {info['resource_id']}")
        print(f"   Description: {info['description']}")
        print(f"   Use Case: {info['use_case']}")
    
    # 3. How to use the APIs
    print("\n💻 3. HOW TO USE DATA.GOV.IN APIs:")
    print("-" * 40)
    
    # Example API call
    api_key = os.getenv('DATA_GOV_IN_API_KEY', 'demo-key')
    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
    
    sample_url = f"https://api.data.gov.in/resource/{resource_id}"
    print(f"\nBase URL: {sample_url}")
    print("Required Parameters:")
    print("  - api-key: Your registered API key")
    print("  - format: json")
    print("  - limit: Number of records (max 1000)")
    print("  - offset: For pagination")
    print("  - filters[field]: To filter data")
    
    # Example filters
    print("\nExample Filters:")
    print("  - filters[state.keyword]: 'Jharkhand'")
    print("  - filters[crop.keyword]: 'Rice'") 
    print("  - filters[season.keyword]: 'Kharif'")
    print("  - filters[crop_year]: '2020'")

def test_crop_production_api():
    """
    Test the current crop production API implementation
    """
    print("\n🧪 TESTING CROP PRODUCTION API:")
    print("-" * 40)
    
    api_key = os.getenv('DATA_GOV_IN_API_KEY', '')
    
    if not api_key:
        print("❌ No API key found. Add DATA_GOV_IN_API_KEY to .env file")
        print("Register at: https://data.gov.in/user/register")
        return
    
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    
    # Test parameters for Jharkhand state
    params = {
        'api-key': api_key,
        'format': 'json',
        'filters[state_name.keyword]': 'Jharkhand',
        'filters[crop_year]': '2020',
        'limit': 10
    }
    
    try:
        print(f"Making API call to: {url}")
        print(f"Parameters: {params}")
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Success! Retrieved {len(data.get('records', []))} records")
            
            if data.get('records'):
                sample_record = data['records'][0]
                print("\nSample Record:")
                for key, value in sample_record.items():
                    print(f"  {key}: {value}")
            
            return data
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def integration_in_ml_system():
    """
    Explain how data.gov.in integrates with the ML system
    """
    print("\n🤖 INTEGRATION WITH ML SYSTEM:")
    print("-" * 40)
    
    print("\n1. DATA COLLECTION FLOW:")
    print("   🌐 User enters latitude/longitude")
    print("   📍 System determines state/district from coordinates")
    print("   🔍 Query data.gov.in for historical data in that region")
    print("   📊 Combine with real-time soil/weather from other APIs")
    print("   🎯 Feed combined data to ML models")
    
    print("\n2. SPECIFIC DATA USAGE:")
    print("   📈 Historical Yield Data: Train yield prediction models")
    print("   🌾 Crop Patterns: Understand what grows well where")
    print("   📅 Seasonal Trends: Factor seasonality into recommendations")
    print("   🏛️ Government Data: Official agricultural statistics")
    
    print("\n3. MODEL ENHANCEMENT:")
    print("   🎯 More Accurate Predictions: Real historical performance")
    print("   🌍 Regional Specificity: India-specific agricultural patterns")
    print("   📊 Validation Data: Cross-check ML predictions with actuals")
    print("   💰 Economic Insights: Production trends for market analysis")

if __name__ == "__main__":
    explore_data_gov_apis()
    test_crop_production_api()
    integration_in_ml_system()