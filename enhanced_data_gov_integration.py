"""
PRACTICAL DATA.GOV.IN INTEGRATION FOR YOUR CROP RECOMMENDATION SYSTEM

This shows how to effectively use data.gov.in APIs to enhance your ML models
with real government agricultural data.
"""

import requests
import pandas as pd
import os
from typing import Dict, List, Optional
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

load_dotenv()

class DataGovInCollector:
    """
    Enhanced data collector specifically for data.gov.in APIs
    """
    
    def __init__(self):
        self.api_key = os.getenv('DATA_GOV_IN_API_KEY', '')
        self.base_url = "https://api.data.gov.in/resource"
        
        # Available datasets with their resource IDs
        self.datasets = {
            'crop_production': '9ef84268-d588-465a-a308-a864a43d0070',
            'rainfall': '88a2e56f-a5c0-4521-91ed-f4b16f5f4df3',
            'soil_health': '33e5c2ea-7b61-43df-a976-2c3c9c11e14c',
            'market_prices': '9ef84268-d588-465a-a308-a864a43d0071',
            'fertilizer_consumption': '6ff0e502-b2f7-4bb3-a8d3-c30c3b7e5d3c'
        }
        
        self.geolocator = Nominatim(user_agent="crop_recommendation_system")
    
    def get_location_details(self, latitude: float, longitude: float) -> Dict:
        """
        Convert latitude/longitude to state/district for data.gov.in queries
        """
        try:
            location = self.geolocator.reverse((latitude, longitude), language='en')
            address = location.raw['address']
            
            return {
                'state': address.get('state', ''),
                'district': address.get('state_district', address.get('county', '')),
                'full_address': location.address
            }
        except Exception as e:
            print(f"Error getting location details: {e}")
            return {'state': 'Jharkhand', 'district': 'Ranchi'}  # Default fallback
    
    def get_historical_crop_data(self, latitude: float, longitude: float, 
                                years: int = 5) -> pd.DataFrame:
        """
        Get historical crop production data for a specific location
        This is the PRIMARY data.gov.in integration for your ML system
        """
        # Get location details
        location = self.get_location_details(latitude, longitude)
        state = location['state']
        district = location['district']
        
        print(f"Fetching crop data for {district}, {state}")
        
        try:
            url = f"{self.base_url}/{self.datasets['crop_production']}"
            
            # Get last 5 years of data
            current_year = 2023
            all_data = []
            
            for year in range(current_year - years, current_year):
                params = {
                    'api-key': self.api_key,
                    'format': 'json',
                    'filters[state_name.keyword]': state,
                    'filters[crop_year]': str(year),
                    'limit': 1000,
                    'offset': 0
                }
                
                # Add district filter if available
                if district:
                    params['filters[district_name.keyword]'] = district
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'records' in data and data['records']:
                        all_data.extend(data['records'])
                        print(f"  ✅ {year}: {len(data['records'])} records")
                    else:
                        print(f"  ❌ {year}: No data")
                else:
                    print(f"  ❌ {year}: API Error {response.status_code}")
            
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Clean and standardize the data
                df = self._process_crop_data(df)
                
                print(f"\\n📊 Total historical records: {len(df)}")
                print(f"📅 Years covered: {df['crop_year'].min()} - {df['crop_year'].max()}")
                print(f"🌾 Unique crops: {df['crop'].nunique()}")
                
                return df
            else:
                print("❌ No historical data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_rainfall_data(self, latitude: float, longitude: float) -> Dict:
        """
        Get rainfall data from data.gov.in
        """
        location = self.get_location_details(latitude, longitude)
        
        try:
            url = f"{self.base_url}/{self.datasets['rainfall']}"
            
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'filters[state.keyword]': location['state'],
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('records'):
                    # Process rainfall data
                    records = data['records']
                    latest_record = records[0]  # Most recent
                    
                    return {
                        'annual_rainfall': float(latest_record.get('annual_rainfall', 1000)),
                        'monsoon_rainfall': float(latest_record.get('monsoon_rainfall', 800)),
                        'year': latest_record.get('year', '2022'),
                        'source': 'data.gov.in'
                    }
            
            # Fallback data
            return {
                'annual_rainfall': 1200.0,
                'monsoon_rainfall': 900.0,
                'year': '2022',
                'source': 'fallback'
            }
            
        except Exception as e:
            print(f"Error fetching rainfall data: {e}")
            return {
                'annual_rainfall': 1200.0,
                'monsoon_rainfall': 900.0,
                'year': '2022',
                'source': 'fallback'
            }
    
    def get_soil_health_data(self, latitude: float, longitude: float) -> Dict:
        """
        Get soil health card data from data.gov.in
        """
        location = self.get_location_details(latitude, longitude)
        
        try:
            url = f"{self.base_url}/{self.datasets['soil_health']}"
            
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'filters[state.keyword]': location['state'],
                'filters[district.keyword]': location['district'],
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('records'):
                    # Process soil health data
                    records = data['records']
                    
                    # Aggregate soil data
                    total_samples = len(records)
                    
                    avg_ph = sum(float(r.get('ph', 6.5)) for r in records) / total_samples
                    avg_oc = sum(float(r.get('organic_carbon', 0.5)) for r in records) / total_samples
                    avg_n = sum(float(r.get('nitrogen', 200)) for r in records) / total_samples
                    avg_p = sum(float(r.get('phosphorus', 20)) for r in records) / total_samples
                    avg_k = sum(float(r.get('potassium', 150)) for r in records) / total_samples
                    
                    return {
                        'ph': avg_ph,
                        'organic_carbon': avg_oc,
                        'nitrogen': avg_n,
                        'phosphorus': avg_p,
                        'potassium': avg_k,
                        'sample_count': total_samples,
                        'source': 'data.gov.in_soil_health'
                    }
                    
        except Exception as e:
            print(f"Error fetching soil health data: {e}")
        
        # Return fallback data
        return {
            'ph': 6.5,
            'organic_carbon': 0.8,
            'nitrogen': 250,
            'phosphorus': 25,
            'potassium': 180,
            'sample_count': 0,
            'source': 'fallback'
        }
    
    def _process_crop_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process crop production data
        """
        # Ensure required columns exist
        required_cols = ['state_name', 'district_name', 'crop_year', 'season', 
                        'crop', 'area', 'production']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
        
        # Clean numeric columns
        numeric_cols = ['crop_year', 'area', 'production']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate yield
        df['yield'] = df['production'] / df['area']
        df['yield'] = df['yield'].replace([float('inf'), -float('inf')], pd.NA)
        
        # Remove invalid records
        df = df.dropna(subset=['crop_year', 'area', 'production', 'yield'])
        df = df[df['area'] > 0]
        df = df[df['production'] >= 0]
        df = df[df['yield'] > 0]
        
        # Standardize crop names
        df['crop'] = df['crop'].str.title().str.strip()
        
        # Standardize season names
        season_mapping = {
            'Kharif': 'Kharif',
            'Rabi': 'Rabi',
            'Summer': 'Summer',
            'Whole Year': 'Annual',
            'Autumn': 'Kharif',
            'Winter': 'Rabi'
        }
        df['season'] = df['season'].map(season_mapping).fillna('Unknown')
        
        return df

def demonstrate_integration_with_ml():
    """
    Show how to integrate data.gov.in data with your ML system
    """
    print("\\n" + "="*60)
    print("INTEGRATING DATA.GOV.IN WITH YOUR ML SYSTEM")
    print("="*60)
    
    collector = DataGovInCollector()
    
    # Example coordinates for Ranchi, Jharkhand
    lat, lon = 23.3441, 85.3096
    
    print(f"\\n📍 Getting data for coordinates: ({lat}, {lon})")
    
    # 1. Get location details
    location = collector.get_location_details(lat, lon)
    print(f"📍 Location: {location['district']}, {location['state']}")
    
    # 2. Get historical crop data (THIS IS THE KEY INTEGRATION)
    print("\\n🌾 Fetching historical crop production data...")
    crop_data = collector.get_historical_crop_data(lat, lon, years=3)
    
    if not crop_data.empty:
        print("\\n📊 SAMPLE DATA FOR ML TRAINING:")
        print(crop_data[['crop', 'season', 'area', 'production', 'yield']].head(10))
        
        # Show how this enhances your ML models
        print("\\n🤖 HOW THIS ENHANCES YOUR ML MODELS:")
        print("1. ✅ Real historical yield data for accurate predictions")
        print("2. ✅ Regional crop performance patterns")
        print("3. ✅ Seasonal trend analysis")
        print("4. ✅ Government-verified agricultural statistics")
        
        # Show specific use cases
        print("\\n💡 SPECIFIC USE CASES IN YOUR SYSTEM:")
        print("📈 Yield Prediction: Train models on actual historical yields")
        print("🌾 Crop Selection: See what crops perform well in specific areas")
        print("📅 Seasonal Planning: Understand optimal planting seasons")
        print("💰 Profit Estimation: Calculate returns based on historical data")
    
    # 3. Get additional data
    print("\\n🌧️ Getting rainfall data...")
    rainfall_data = collector.get_rainfall_data(lat, lon)
    print(f"Annual Rainfall: {rainfall_data['annual_rainfall']}mm")
    
    print("\\n🌱 Getting soil health data...")
    soil_data = collector.get_soil_health_data(lat, lon)
    print(f"Soil pH: {soil_data['ph']:.1f}")
    print(f"Organic Carbon: {soil_data['organic_carbon']:.2f}%")

if __name__ == "__main__":
    demonstrate_integration_with_ml()