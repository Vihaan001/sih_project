"""
Test API connections and data collection
Run this script to verify your API keys and connections are working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.real_data_collector import RealDataCollector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_apis():
    print("🧪 Testing API Connections...")
    print("="*50)
    
    collector = RealDataCollector()
    
    # Test coordinates (Ranchi, Jharkhand)
    lat, lon = 23.3441, 85.3096
    
    # Test 1: SoilGrids (Free, no key needed)
    print("\n1. Testing SoilGrids API...")
    try:
        soil_data = collector.get_soil_data_soilgrids(lat, lon)
        if soil_data and 'ph' in soil_data:
            print(f"   ✅ SUCCESS: pH = {soil_data['ph']}")
            print(f"   📊 Soil texture: {soil_data['texture']}")
            print(f"   🧪 Organic carbon: {soil_data['organic_carbon']}%")
        else:
            print("   ❌ FAILED: No data returned")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 2: OpenWeatherMap
    print("\n2. Testing OpenWeatherMap API...")
    openweather_key = os.getenv('OPENWEATHER_API_KEY', '')
    
    if not openweather_key or openweather_key == 'your_openweather_api_key_here':
        print("   ⚠️  SKIPPED: No API key found")
        print("   📝 Get your free key from: https://openweathermap.org/api")
        print("   🔧 Add to .env: OPENWEATHER_API_KEY=your_key_here")
    else:
        try:
            weather_data = collector.get_weather_data_openweather(lat, lon)
            if weather_data and 'temperature' in weather_data:
                print(f"   ✅ SUCCESS: Temperature = {weather_data['temperature']}°C")
                print(f"   💧 Humidity: {weather_data['humidity']}%")
                print(f"   🌧️  Rainfall: {weather_data['rainfall']}mm")
            else:
                print("   ❌ FAILED: No data returned")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    # Test 3: NASA POWER (Historical data)
    print("\n3. Testing NASA POWER API...")
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_data = collector.get_nasa_power_data(
            lat, lon,
            start_date.strftime('%Y%m%d'),
            end_date.strftime('%Y%m%d')
        )
        
        if not historical_data.empty:
            print(f"   ✅ SUCCESS: Got {len(historical_data)} days of data")
            print(f"   📅 Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
            print(f"   🌡️  Avg temp: {historical_data['temp_avg'].mean():.1f}°C")
        else:
            print("   ❌ FAILED: No historical data returned")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("📋 Summary:")
    print("   - SoilGrids works without API key (global soil data)")
    print("   - OpenWeatherMap needs free API key (current weather)")
    print("   - NASA POWER is free (historical weather for training)")
    print("\n💡 Next steps:")
    print("   1. Get OpenWeather API key if you haven't")
    print("   2. Run training script: python ml_models/train_model.py")
    print("   3. Test the full system with: python backend/main.py")

if __name__ == "__main__":
    test_apis()
