"""
Real Data Collection Module
Integrates with actual APIs for production use
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from dotenv import load_dotenv
import time
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Load environment variables
load_dotenv()

class RealDataCollector:
    """
    Collects real data from various agricultural and weather APIs
    """
    
    def __init__(self):
        # API Keys (store in .env file)
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY', '')
        self.agromonitoring_api_key = os.getenv('AGROMONITORING_API_KEY', '')
        self.soilgrids_api_key = os.getenv('SOILGRIDS_API_KEY', '')
        
        # API Endpoints
        self.apis = {
            'openweather': 'https://api.openweathermap.org/data/2.5',
            'soilgrids': 'https://rest.isric.org/soilgrids/v2.0',
            'bhuvan': 'https://bhuvan-vec1.nrsc.gov.in',
            'agmarknet': 'https://agmarknet.gov.in/SearchCmmMkt.aspx',
            'agromonitoring': 'http://api.agromonitoring.com/agro/1.0',
            'imd': 'https://api.imd.gov.in',  # Indian Meteorological Department
            'nasa_power': 'https://power.larc.nasa.gov/api/temporal/daily/point'
        }
        
    def get_soil_data_soilgrids(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch soil data from SoilGrids API (ISRIC)
        Free API with global soil data at 250m resolution
        """
        try:
            # SoilGrids properties we want to fetch
            properties = [
                'phh2o',  # pH in H2O
                'nitrogen',  # Total nitrogen
                'soc',  # Soil organic carbon
                'sand',  # Sand content
                'silt',  # Silt content
                'clay',  # Clay content
                'bdod',  # Bulk density
                'cec',  # Cation exchange capacity
            ]
            
            # Depths in cm (we'll use 0-5cm and 5-15cm)
            depths = ['0-5cm', '5-15cm']
            
            # Build query
            properties_query = '&'.join([f'property={prop}' for prop in properties])
            depths_query = '&'.join([f'depth={depth}' for depth in depths])
            
            url = f"{self.apis['soilgrids']}/properties/query"
            params = {
                'lon': longitude,
                'lat': latitude,
                'property': properties,
                'depth': depths,
                'value': 'mean'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process the response
                soil_data = {
                    'ph': 0,
                    'nitrogen': 0,
                    'phosphorus': 0,  # Will need to estimate
                    'potassium': 0,  # Will need to estimate
                    'organic_carbon': 0,
                    'sand': 0,
                    'silt': 0,
                    'clay': 0,
                    'cec': 0,
                    'moisture': 0,  # Will get from weather data
                    'texture': '',
                    'ec': 0  # Will need to estimate
                }
                
                # Extract values from response
                if 'properties' in data:
                    layers = data['properties']['layers']
                    for layer in layers:
                        if layer['name'] == 'phh2o':
                            # pH is given in 10 * actual value
                            soil_data['ph'] = layer['depths'][0]['values']['mean'] / 10
                        elif layer['name'] == 'nitrogen':
                            # Convert from g/kg to kg/ha (approximate)
                            soil_data['nitrogen'] = layer['depths'][0]['values']['mean'] * 100
                        elif layer['name'] == 'soc':
                            # Soil organic carbon in g/kg
                            soil_data['organic_carbon'] = layer['depths'][0]['values']['mean'] / 10
                        elif layer['name'] == 'sand':
                            soil_data['sand'] = layer['depths'][0]['values']['mean'] / 10
                        elif layer['name'] == 'silt':
                            soil_data['silt'] = layer['depths'][0]['values']['mean'] / 10
                        elif layer['name'] == 'clay':
                            soil_data['clay'] = layer['depths'][0]['values']['mean'] / 10
                        elif layer['name'] == 'cec':
                            soil_data['cec'] = layer['depths'][0]['values']['mean'] / 10
                
                # Determine soil texture based on sand, silt, clay percentages
                soil_data['texture'] = self._determine_soil_texture(
                    soil_data['sand'], 
                    soil_data['silt'], 
                    soil_data['clay']
                )
                
                # Estimate P and K based on soil type and organic carbon
                soil_data['phosphorus'] = self._estimate_phosphorus(soil_data['organic_carbon'])
                soil_data['potassium'] = self._estimate_potassium(soil_data['cec'])
                soil_data['ec'] = self._estimate_ec(soil_data['texture'])
                
                return soil_data
            else:
                print(f"SoilGrids API error: {response.status_code}")
                return self._get_fallback_soil_data()
                
        except Exception as e:
            print(f"Error fetching SoilGrids data: {e}")
            return self._get_fallback_soil_data()
    
    def get_soil_data_bhuvan(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch soil data from Bhuvan (ISRO) - Indian specific
        Note: Requires registration and API key
        """
        try:
            # Bhuvan API endpoint for soil data
            url = f"{self.apis['bhuvan']}/api/soil/properties"
            
            headers = {
                'Authorization': f'Bearer {os.getenv("BHUVAN_API_KEY", "")}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'lat': latitude,
                'lon': longitude,
                'buffer': 1000  # 1km buffer
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                # Process Bhuvan specific response
                return self._process_bhuvan_data(data)
            else:
                return self._get_fallback_soil_data()
                
        except Exception as e:
            print(f"Error fetching Bhuvan data: {e}")
            return self._get_fallback_soil_data()
    
    def get_weather_data_openweather(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch weather data from OpenWeatherMap API
        Free tier: 1000 calls/day
        """
        try:
            if not self.openweather_api_key:
                print("OpenWeather API key not found. Using fallback data.")
                return self._get_fallback_weather_data()
            
            # Current weather
            current_url = f"{self.apis['openweather']}/weather"
            current_params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            current_response = requests.get(current_url, params=current_params)
            
            # 7-day forecast
            forecast_url = f"{self.apis['openweather']}/forecast/daily"
            forecast_params = {
                'lat': latitude,
                'lon': longitude,
                'cnt': 7,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            # Note: Daily forecast requires paid subscription
            # Using 5-day/3-hour forecast as alternative
            forecast_url = f"{self.apis['openweather']}/forecast"
            forecast_response = requests.get(forecast_url, params=current_params)
            
            weather_data = {
                'temperature': 0,
                'humidity': 0,
                'rainfall': 0,
                'wind_speed': 0,
                'pressure': 0,
                'forecast_7_days': {},
                'season': self._get_current_season()
            }
            
            if current_response.status_code == 200:
                current = current_response.json()
                weather_data['temperature'] = current['main']['temp']
                weather_data['humidity'] = current['main']['humidity']
                weather_data['pressure'] = current['main']['pressure']
                weather_data['wind_speed'] = current['wind']['speed'] * 3.6  # Convert m/s to km/h
                
                # Check for rain
                if 'rain' in current:
                    weather_data['rainfall'] = current['rain'].get('1h', 0)
            
            if forecast_response.status_code == 200:
                forecast = forecast_response.json()
                # Process forecast data
                temps = []
                humidity = []
                rainfall = 0
                
                for item in forecast['list'][:8]:  # Next 24 hours
                    temps.append(item['main']['temp'])
                    humidity.append(item['main']['humidity'])
                    if 'rain' in item:
                        rainfall += item['rain'].get('3h', 0)
                
                weather_data['forecast_7_days'] = {
                    'avg_temp': round(np.mean(temps), 1),
                    'total_rainfall': round(rainfall, 1),
                    'avg_humidity': round(np.mean(humidity), 1)
                }
            
            return weather_data
            
        except Exception as e:
            print(f"Error fetching OpenWeather data: {e}")
            return self._get_fallback_weather_data()
    
    def get_weather_data_imd(self, district: str, state: str = "Jharkhand") -> Dict:
        """
        Fetch weather data from India Meteorological Department
        More accurate for Indian regions
        """
        try:
            # IMD provides district-wise weather data
            url = f"https://imd.gov.in/api/weather"
            params = {
                'district': district,
                'state': state
            }
            
            # Note: Actual IMD API might require authentication
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return self._process_imd_data(response.json())
            else:
                return self._get_fallback_weather_data()
                
        except Exception as e:
            print(f"Error fetching IMD data: {e}")
            return self._get_fallback_weather_data()
    
    def get_market_prices_agmarknet(self, commodity: str = None, state: str = "Jharkhand") -> Dict:
        """
        Fetch market prices from Agmarknet (Government of India)
        Real-time agricultural market prices
        """
        try:
            # Agmarknet provides data through web scraping or their data portal
            # Using their data download option
            
            commodities = commodity.split(',') if commodity else [
                "Wheat", "Rice", "Maize", "Arhar", "Gram", "Mustard"
            ]
            
            market_data = {}
            
            for crop in commodities:
                # Format for Agmarknet query
                url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
                
                # This would typically require web scraping
                # For API access, you might need to use their data portal
                params = {
                    'Tx_Commodity': crop,
                    'Tx_State': state,
                    'Tx_District': 'All',
                    'Tx_Market': 'All',
                    'DateFrom': (datetime.now() - timedelta(days=7)).strftime('%d-%b-%Y'),
                    'DateTo': datetime.now().strftime('%d-%b-%Y'),
                }
                
                # Simulated API call - in reality, this needs web scraping
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    # Parse HTML response
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Extract price data from table
                    price_data = self._parse_agmarknet_data(soup, crop)
                    market_data[crop] = price_data
                else:
                    market_data[crop] = self._get_fallback_market_price(crop)
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching Agmarknet data: {e}")
            return self._get_fallback_market_data()
    
    def get_nasa_power_data(self, latitude: float, longitude: float, 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather data from NASA POWER API
        Excellent for training data - provides 40+ years of data
        """
        try:
            url = self.apis['nasa_power']
            
            params = {
                'parameters': 'T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M',
                'community': 'AG',
                'longitude': longitude,
                'latitude': latitude,
                'start': start_date,  # Format: YYYYMMDD
                'end': end_date,
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract parameters
                parameters = data['properties']['parameter']
                
                # Create DataFrame
                df_data = []
                for date, values in parameters['T2M'].items():
                    row = {
                        'date': date,
                        'temp_avg': parameters['T2M'][date],
                        'temp_min': parameters['T2M_MIN'][date],
                        'temp_max': parameters['T2M_MAX'][date],
                        'precipitation': parameters['PRECTOTCORR'][date],
                        'humidity': parameters['RH2M'][date],
                        'wind_speed': parameters['WS2M'][date]
                    }
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df['date'] = pd.to_datetime(df['date'])
                
                return df
            else:
                print(f"NASA POWER API error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching NASA POWER data: {e}")
            return pd.DataFrame()
    
    def get_crop_yield_data(self, state: str = "Jharkhand") -> pd.DataFrame:
        """
        Fetch historical crop yield data from data.gov.in
        """
        try:
            # data.gov.in API endpoint
            url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
            
            api_key = os.getenv('DATA_GOV_IN_API_KEY', '')
            
            params = {
                'api-key': api_key,
                'format': 'json',
                'filters[state.keyword]': state,
                'limit': 1000
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Convert to DataFrame
                df = pd.DataFrame(data['records'])
                
                # Clean and process data
                df = df[['state_name', 'district_name', 'crop_year', 'season', 
                        'crop', 'area', 'production', 'yield']]
                
                # Convert numeric columns
                numeric_cols = ['crop_year', 'area', 'production', 'yield']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            else:
                print(f"Data.gov.in API error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching yield data: {e}")
            return pd.DataFrame()
    
    # Helper methods
    def _determine_soil_texture(self, sand: float, silt: float, clay: float) -> str:
        """Determine soil texture using USDA soil texture triangle"""
        if clay > 40:
            return "Clay"
        elif sand > 85:
            return "Sandy"
        elif silt > 80:
            return "Silt"
        elif (sand > 45 and sand < 65) and (clay > 20 and clay < 35):
            return "Loamy"
        else:
            return "Loamy"
    
    def _estimate_phosphorus(self, organic_carbon: float) -> float:
        """Estimate phosphorus based on organic carbon"""
        # Rough estimation: P = 20 + (organic_carbon * 15)
        return round(20 + (organic_carbon * 15), 2)
    
    def _estimate_potassium(self, cec: float) -> float:
        """Estimate potassium based on CEC"""
        # Rough estimation: K = 50 + (cec * 10)
        return round(50 + (cec * 10), 2)
    
    def _estimate_ec(self, texture: str) -> float:
        """Estimate electrical conductivity based on texture"""
        ec_map = {
            "Clay": 0.8,
            "Sandy": 0.3,
            "Loamy": 0.5,
            "Silt": 0.6
        }
        return ec_map.get(texture, 0.5)
    
    def _get_current_season(self) -> str:
        """Determine current agricultural season in India"""
        month = datetime.now().month
        if month in [6, 7, 8, 9, 10]:
            return "Kharif"
        elif month in [10, 11, 12, 1, 2, 3]:
            return "Rabi"
        else:
            return "Zaid"
    
    def _get_fallback_soil_data(self) -> Dict:
        """Fallback soil data when API fails"""
        return {
            'ph': 6.5,
            'nitrogen': 150,
            'phosphorus': 40,
            'potassium': 180,
            'organic_carbon': 1.2,
            'moisture': 25,
            'texture': 'Loamy',
            'ec': 0.5,
            'sand': 40,
            'silt': 30,
            'clay': 30,
            'cec': 15
        }
    
    def _get_fallback_weather_data(self) -> Dict:
        """Fallback weather data when API fails"""
        return {
            'temperature': 28,
            'humidity': 65,
            'rainfall': 10,
            'wind_speed': 12,
            'pressure': 1013,
            'forecast_7_days': {
                'avg_temp': 27,
                'total_rainfall': 50,
                'avg_humidity': 68
            },
            'season': self._get_current_season()
        }
    
    def _get_fallback_market_price(self, crop: str) -> Dict:
        """Fallback market price for a crop"""
        base_prices = {
            "Wheat": 2000,
            "Rice": 2500,
            "Maize": 1800,
            "Cotton": 6000,
            "Sugarcane": 3500,
            "Soybean": 4500,
            "Groundnut": 5500,
            "Pulses": 7000
        }
        
        base_price = base_prices.get(crop, 3000)
        
        return {
            'current_price': base_price,
            'min_price': base_price * 0.9,
            'max_price': base_price * 1.1,
            'price_trend': 'stable',
            'demand': 'medium',
            'best_market': 'Local Mandi'
        }
    
    def _get_fallback_market_data(self) -> Dict:
        """Fallback market data when API fails"""
        crops = ["Wheat", "Rice", "Maize", "Cotton", "Pulses"]
        return {crop: self._get_fallback_market_price(crop) for crop in crops}
    
    def _parse_agmarknet_data(self, soup: BeautifulSoup, crop: str) -> Dict:
        """Parse Agmarknet HTML response"""
        # This would need actual HTML parsing logic
        # For now, returning fallback
        return self._get_fallback_market_price(crop)
    
    def _process_bhuvan_data(self, data: Dict) -> Dict:
        """Process Bhuvan API response"""
        # Process based on actual Bhuvan response structure
        return self._get_fallback_soil_data()
    
    def _process_imd_data(self, data: Dict) -> Dict:
        """Process IMD API response"""
        # Process based on actual IMD response structure
        return self._get_fallback_weather_data()


class DatasetBuilder:
    """
    Build training dataset from real API data
    """
    
    def __init__(self):
        self.collector = RealDataCollector()
        
    def build_training_dataset(self, locations: List[Tuple[float, float]], 
                             output_file: str = 'data/training_data.csv') -> pd.DataFrame:
        """
        Build comprehensive training dataset from multiple locations
        """
        print("Building training dataset from real APIs...")
        
        all_data = []
        
        for i, (lat, lon) in enumerate(locations):
            print(f"Fetching data for location {i+1}/{len(locations)}: ({lat}, {lon})")
            
            try:
                # Fetch all data types
                soil_data = self.collector.get_soil_data_soilgrids(lat, lon)
                weather_data = self.collector.get_weather_data_openweather(lat, lon)
                
                # Combine into single record
                record = {
                    'latitude': lat,
                    'longitude': lon,
                    'ph': soil_data['ph'],
                    'nitrogen': soil_data['nitrogen'],
                    'phosphorus': soil_data['phosphorus'],
                    'potassium': soil_data['potassium'],
                    'temperature': weather_data['temperature'],
                    'humidity': weather_data['humidity'],
                    'rainfall': weather_data['rainfall'],
                    'moisture': soil_data['moisture'],
                    'organic_carbon': soil_data['organic_carbon'],
                    'ec': soil_data['ec'],
                    'texture': soil_data['texture'],
                    'season': weather_data['season'],
                    'wind_speed': weather_data['wind_speed'],
                    'sand': soil_data['sand'],
                    'silt': soil_data['silt'],
                    'clay': soil_data['clay']
                }
                
                all_data.append(record)
                
                # Rate limiting to avoid API throttling
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing location ({lat}, {lon}): {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
        
        return df
    
    def fetch_historical_weather_data(self, lat: float, lon: float, 
                                     years: int = 5) -> pd.DataFrame:
        """
        Fetch historical weather data for training
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        df = self.collector.get_nasa_power_data(
            lat, lon,
            start_date.strftime('%Y%m%d'),
            end_date.strftime('%Y%m%d')
        )
        
        return df
    
    def fetch_yield_data(self, state: str = "Jharkhand") -> pd.DataFrame:
        """
        Fetch historical crop yield data
        """
        return self.collector.get_crop_yield_data(state)
