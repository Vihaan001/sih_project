"""
Data Collection Module for Crop Recommendation System
Handles soil data, weather data, and market prices
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import random

class DataCollector:
    def __init__(self):
        self.base_weather_url = "https://api.openweathermap.org/data/2.5"
        # Note: In production, use actual API keys
        self.weather_api_key = "YOUR_API_KEY"
        
    def get_soil_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch soil data for given coordinates
        For MVP: Using mock data, in production use SoilGrids API
        """
        # Mock soil data for MVP
        soil_data = {
            "ph": round(random.uniform(5.5, 8.0), 2),
            "nitrogen": round(random.uniform(100, 400), 2),  # kg/ha
            "phosphorus": round(random.uniform(20, 80), 2),   # kg/ha
            "potassium": round(random.uniform(100, 300), 2),  # kg/ha
            "organic_carbon": round(random.uniform(0.5, 2.5), 2),  # %
            "moisture": round(random.uniform(15, 35), 2),  # %
            "texture": random.choice(["Clay", "Sandy", "Loamy", "Silt"]),
            "ec": round(random.uniform(0.1, 2.0), 2),  # dS/m
        }
        return soil_data
    
    def get_weather_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch weather data for given coordinates
        For MVP: Using mock data, in production use OpenWeather API
        """
        # Mock weather data for MVP
        weather_data = {
            "temperature": round(random.uniform(20, 35), 1),  # Celsius
            "humidity": round(random.uniform(40, 80), 1),  # %
            "rainfall": round(random.uniform(0, 50), 1),  # mm
            "wind_speed": round(random.uniform(5, 20), 1),  # km/h
            "forecast_7_days": {
                "avg_temp": round(random.uniform(22, 32), 1),
                "total_rainfall": round(random.uniform(0, 150), 1),
                "avg_humidity": round(random.uniform(45, 75), 1)
            },
            "season": self._get_current_season()
        }
        return weather_data
    
    def get_market_prices(self, crop_list: List[str] = None) -> Dict:
        """
        Fetch current market prices for crops
        For MVP: Using mock data, in production use Agmarknet API
        """
        if crop_list is None:
            crop_list = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", 
                        "Soybean", "Groundnut", "Pulses", "Vegetables"]
        
        market_data = {}
        for crop in crop_list:
            market_data[crop] = {
                "current_price": round(random.uniform(1500, 5000), 0),  # Rs/quintal
                "price_trend": random.choice(["up", "down", "stable"]),
                "demand": random.choice(["high", "medium", "low"]),
                "best_market": random.choice(["Local Mandi", "District Market", "State Market"])
            }
        return market_data
    
    def get_historical_crop_data(self, region: str) -> List[Dict]:
        """
        Get historical crop rotation data for the region
        """
        crops_history = [
            {"year": 2023, "kharif": "Rice", "rabi": "Wheat", "yield": "Good"},
            {"year": 2022, "kharif": "Maize", "rabi": "Mustard", "yield": "Average"},
            {"year": 2021, "kharif": "Cotton", "rabi": "Gram", "yield": "Good"}
        ]
        return crops_history
    
    def _get_current_season(self) -> str:
        """
        Determine current agricultural season
        """
        month = datetime.now().month
        if month in [6, 7, 8, 9, 10]:
            return "Kharif"
        elif month in [10, 11, 12, 1, 2, 3]:
            return "Rabi"
        else:
            return "Zaid"
    
    def get_comprehensive_data(self, latitude: float, longitude: float, 
                              region: str = "Default") -> Dict:
        """
        Get all data required for crop recommendation
        """
        return {
            "soil": self.get_soil_data(latitude, longitude),
            "weather": self.get_weather_data(latitude, longitude),
            "market": self.get_market_prices(),
            "history": self.get_historical_crop_data(region),
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "region": region
            },
            "timestamp": datetime.now().isoformat()
        }

class DataProcessor:
    """
    Process and prepare data for ML model
    """
    
    @staticmethod
    def prepare_features(data: Dict) -> pd.DataFrame:
        """
        Convert collected data into features for ML model
        Uses column names that match the trained model: N, P, K, temperature, humidity, ph, rainfall
        """
        features = {
            "N": data["soil"]["nitrogen"],        # Nitrogen - matches model expectation
            "P": data["soil"]["phosphorus"],      # Phosphorus - matches model expectation  
            "K": data["soil"]["potassium"],       # Potassium - matches model expectation
            "temperature": data["weather"]["temperature"],
            "humidity": data["weather"]["humidity"],
            "ph": data["soil"]["ph"],
            "rainfall": data["weather"]["rainfall"]
            # Note: Removed extra features that aren't used by the trained model
            # "moisture": data["soil"]["moisture"],
            # "organic_carbon": data["soil"]["organic_carbon"],
            # "ec": data["soil"]["ec"],
            # "texture_encoded": DataProcessor._encode_texture(data["soil"]["texture"]),
            # "season_encoded": DataProcessor._encode_season(data["weather"]["season"])
        }
        
        return pd.DataFrame([features])
    
    @staticmethod
    def _encode_texture(texture: str) -> int:
        """Encode soil texture"""
        texture_map = {"Clay": 0, "Sandy": 1, "Loamy": 2, "Silt": 3}
        return texture_map.get(texture, 2)
    
    @staticmethod
    def _encode_season(season: str) -> int:
        """Encode season"""
        season_map = {"Kharif": 0, "Rabi": 1, "Zaid": 2}
        return season_map.get(season, 0)
