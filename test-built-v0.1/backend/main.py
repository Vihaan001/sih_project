"""
FastAPI Backend for Crop Recommendation System
Main API server with endpoints for recommendations, data retrieval, and chat
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_collector import DataCollector, DataProcessor
from ml_models.crop_recommender import CropRecommendationModel, CropRotationPlanner
from backend.ai_assistant import AIAssistant
from backend.translator import Translator

# Initialize FastAPI app
app = FastAPI(
    title="Crop Recommendation API",
    description="AI-powered crop recommendation system for farmers",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_collector = DataCollector()
crop_model = CropRecommendationModel()
ai_assistant = AIAssistant()
translator = Translator()

# Pydantic models for request/response
class LocationData(BaseModel):
    latitude: float
    longitude: float
    region: Optional[str] = "Default"

class SoilData(BaseModel):
    ph: Optional[float] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    organic_carbon: Optional[float] = None
    moisture: Optional[float] = None
    texture: Optional[str] = None
    ec: Optional[float] = None

class FarmerQuery(BaseModel):
    query: str
    language: str = "en"
    location: Optional[LocationData] = None
    context: Optional[Dict] = None

class CropRecommendationRequest(BaseModel):
    location: LocationData
    soil_data: Optional[SoilData] = None
    use_satellite_data: bool = True
    language: str = "en"

class CropRotationRequest(BaseModel):
    current_crop: str
    location: LocationData
    soil_data: Optional[SoilData] = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crop Recommendation System API",
        "version": "1.0.0",
        "endpoints": [
            "/recommend",
            "/chat",
            "/soil-data",
            "/weather",
            "/market-prices",
            "/crop-rotation"
        ]
    }

@app.post("/recommend")
async def get_crop_recommendations(request: CropRecommendationRequest):
    """
    Get crop recommendations based on location and soil data
    """
    try:
        # Collect comprehensive data
        comprehensive_data = data_collector.get_comprehensive_data(
            request.location.latitude,
            request.location.longitude,
            request.location.region
        )
        
        # Override with provided soil data if available
        if request.soil_data:
            soil_dict = request.soil_data.dict(exclude_none=True)
            comprehensive_data["soil"].update(soil_dict)
        
        # Prepare features for ML model
        features = DataProcessor.prepare_features(comprehensive_data)
        
        # Get recommendations
        recommendations = crop_model.predict_crops(features, top_n=3)
        
        # Add market context
        market_data = comprehensive_data["market"]
        for rec in recommendations:
            crop_name = rec["crop"]
            if crop_name in market_data:
                rec["market_info"] = market_data[crop_name]
        
        # Translate if needed
        if request.language != "en":
            recommendations = translator.translate_recommendations(
                recommendations, request.language
            )
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "environmental_data": {
                "soil": comprehensive_data["soil"],
                "weather": comprehensive_data["weather"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_assistant(query: FarmerQuery):
    """
    Chat endpoint for natural language queries
    """
    try:
        # Get location-specific data if provided
        context_data = {}
        if query.location:
            context_data = data_collector.get_comprehensive_data(
                query.location.latitude,
                query.location.longitude,
                query.location.region or "Default"
            )
        
        # Merge with provided context
        if query.context:
            context_data.update(query.context)
        
        # Get AI response
        response = ai_assistant.process_query(
            query.query,
            context_data,
            query.language
        )
        
        return {
            "status": "success",
            "response": response,
            "language": query.language,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/soil-data/{latitude}/{longitude}")
async def get_soil_data(latitude: float, longitude: float):
    """
    Get soil data for specific coordinates
    """
    try:
        soil_data = data_collector.get_soil_data(latitude, longitude)
        return {
            "status": "success",
            "data": soil_data,
            "location": {"latitude": latitude, "longitude": longitude},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/{latitude}/{longitude}")
async def get_weather_data(latitude: float, longitude: float):
    """
    Get weather data for specific coordinates
    """
    try:
        weather_data = data_collector.get_weather_data(latitude, longitude)
        return {
            "status": "success",
            "data": weather_data,
            "location": {"latitude": latitude, "longitude": longitude},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-prices")
async def get_market_prices(crops: Optional[str] = None):
    """
    Get current market prices for crops
    """
    try:
        crop_list = crops.split(",") if crops else None
        market_data = data_collector.get_market_prices(crop_list)
        return {
            "status": "success",
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crop-rotation")
async def suggest_crop_rotation(request: CropRotationRequest):
    """
    Suggest crop rotation plan
    """
    try:
        # Get soil data if not provided
        soil_data = {}
        if request.soil_data:
            soil_data = request.soil_data.dict(exclude_none=True)
        else:
            soil_data = data_collector.get_soil_data(
                request.location.latitude,
                request.location.longitude
            )
        
        # Get rotation suggestions
        rotation_plan = CropRotationPlanner.suggest_rotation(
            request.current_crop,
            soil_data
        )
        
        return {
            "status": "success",
            "rotation_plan": rotation_plan,
            "soil_data": soil_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image")
async def analyze_crop_image(file: UploadFile = File(...)):
    """
    Analyze uploaded crop/field image for disease detection
    """
    try:
        # For MVP, return mock analysis
        # In production, integrate with computer vision model
        
        contents = await file.read()
        
        mock_analysis = {
            "image_name": file.filename,
            "analysis": {
                "crop_detected": "Wheat",
                "health_status": "Healthy",
                "disease_detected": None,
                "confidence": 0.85,
                "recommendations": [
                    "Continue regular monitoring",
                    "Ensure proper irrigation",
                    "Apply preventive fungicide if weather is humid"
                ]
            }
        }
        
        return {
            "status": "success",
            "data": mock_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/offline-data")
async def get_offline_data_package():
    """
    Get data package for offline use
    """
    try:
        # Package essential data for offline use
        offline_package = {
            "crop_database": crop_model.crop_database,
            "common_queries": ai_assistant.get_common_responses(),
            "seasonal_tips": {
                "Kharif": ["Prepare land before monsoon", "Use quality seeds"],
                "Rabi": ["Check soil moisture", "Apply basal fertilizers"],
                "Zaid": ["Ensure irrigation availability", "Choose short duration crops"]
            },
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "data": offline_package
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
