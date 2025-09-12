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
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collector import DataCollector, DataProcessor
from real_data_collector import RealDataCollector
from ml_models.crop_recommender import CropRecommendationModel, CropRotationPlanner
from ai_assistant import AIAssistant
# Use Google Gemini assistant if available
try:
    from gemini_assistant import GeminiAssistant
    gemini_available = True
except ImportError:
    gemini_available = False
    
# Use enhanced translator with deep-translator
try:
    from translator_enhanced import EnhancedTranslator as Translator
except ImportError:
    from translator import Translator

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
data_collector = RealDataCollector()  # Use real APIs including SoilGrids
crop_model = CropRecommendationModel()

# Load trained ML models
try:
    crop_model.load_model("../ml_models/trained/")
    logger.info("Loaded pre-trained ML models successfully")
except Exception as e:
    logger.warning(f"Failed to load pre-trained models: {e}")
    logger.info("Will train new models when needed")

# Initialize AI assistant (Gemini if available, fallback otherwise)
if gemini_available:
    try:
        gemini_assistant = GeminiAssistant()
        if gemini_assistant.is_configured:
            ai_assistant = gemini_assistant
            logger.info("Using Google Gemini AI assistant")
        else:
            ai_assistant = AIAssistant()
            logger.info("Gemini not configured, using fallback AI assistant")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini: {e}")
        ai_assistant = AIAssistant()
else:
    ai_assistant = AIAssistant()
    logger.info("Using fallback AI assistant")

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
    language: str = "auto"  # Changed default to auto-detect
    location: Optional[LocationData] = None
    context: Optional[Dict] = None
    session_id: Optional[str] = None  # For maintaining conversation context

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
        
        # Auto-translate recommendations based on user's language
        if request.language != "en":
            # Translate each recommendation
            translated_recommendations = []
            for rec in recommendations:
                # Use the enhanced translator's crop recommendation method
                if hasattr(translator, 'translate_crop_recommendation'):
                    translated_rec = translator.translate_crop_recommendation(rec, request.language)
                else:
                    # Fallback to basic translation
                    translated_rec = rec.copy()
                    translated_rec["crop"] = translator.translate(rec["crop"], request.language)
                    if "care_tips" in rec:
                        translated_rec["care_tips"] = [
                            translator.translate(tip, request.language) 
                            for tip in rec["care_tips"]
                        ]
                translated_recommendations.append(translated_rec)
            recommendations = translated_recommendations
            
            # Translate environmental data labels
            environmental_labels = {
                "soil": translator.translate("soil", request.language),
                "weather": translator.translate("weather", request.language),
                "ph": translator.translate("pH", request.language),
                "nitrogen": translator.translate("nitrogen", request.language),
                "phosphorus": translator.translate("phosphorus", request.language),
                "potassium": translator.translate("potassium", request.language),
                "temperature": translator.translate("temperature", request.language),
                "humidity": translator.translate("humidity", request.language),
                "rainfall": translator.translate("rainfall", request.language),
                "season": translator.translate("season", request.language)
            }
        else:
            environmental_labels = None
        
        response_data = {
            "status": "success",
            "recommendations": recommendations,
            "environmental_data": {
                "soil": comprehensive_data["soil"],
                "weather": comprehensive_data["weather"]
            },
            "timestamp": datetime.now().isoformat(),
            "language": request.language
        }
        
        # Add translated labels if applicable
        if environmental_labels:
            response_data["labels"] = environmental_labels
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_assistant(query: FarmerQuery):
    """
    Chat endpoint for natural language queries
    """
    try:
        # Detect language if needed
        detected_language = query.language
        if hasattr(translator, 'detect_language') and query.language == "auto":
            detected_language = translator.detect_language(query.query)
        
        # Translate query to English for processing if needed
        query_text = query.query
        if detected_language != "en":
            query_text = translator.translate(query.query, "en", detected_language)
        
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
        
        # Get AI response (check if using Gemini with sessions)
        if isinstance(ai_assistant, GeminiAssistant) if gemini_available else False:
            # Build comprehensive context for Gemini
            gemini_context = {
                'location': f"{query.location.region if query.location and query.location.region else 'Unknown'}",
                'soil_data': context_data.get('soil', {}) if context_data else {},
                'weather_data': context_data.get('weather', {}) if context_data else {},
                'market_data': context_data.get('market', {}) if context_data else {}
            }
            
            # Add location coordinates if available
            if query.location:
                gemini_context['location'] = f"{query.location.region or ''} ({query.location.latitude}, {query.location.longitude})"
            
            # Process with Gemini (maintains session context)
            response_data = ai_assistant.process_query(
                query_text,
                session_id=query.session_id,
                context_data=gemini_context
            )
            
            # Format response for compatibility
            if response_data['status'] == 'success':
                response = {
                    'type': 'gemini_response',
                    'message': response_data['response'],
                    'session_id': response_data.get('session_id'),
                    'context_used': response_data.get('context_used', False),
                    'structured_advice': response_data.get('structured_advice', {})
                }
            else:
                response = {
                    'type': 'error',
                    'message': 'Failed to process query',
                    'suggestions': ['Please try again or rephrase your question']
                }
        else:
            # Use fallback AI assistant
            response = ai_assistant.process_query(
                query_text,
                context_data,
                "en"  # Process in English
            )
        
        # Auto-translate response to user's language
        if detected_language != "en":
            # Translate message
            if "message" in response:
                response["message"] = translator.translate(response["message"], detected_language)
            
            # Translate suggestions
            if "suggestions" in response and isinstance(response["suggestions"], list):
                response["suggestions"] = [
                    translator.translate(suggestion, detected_language)
                    for suggestion in response["suggestions"]
                ]
            
            # Translate any crop recommendations in the response
            if "recommendations" in response:
                response["recommendations"] = [
                    translator.translate_crop_recommendation(rec, detected_language)
                    if hasattr(translator, 'translate_crop_recommendation')
                    else rec
                    for rec in response["recommendations"]
                ]
            
            # Translate disease/pest information
            if "disease_identified" in response:
                response["disease_identified"] = translator.translate(
                    response["disease_identified"], detected_language
                )
            
            if "symptoms" in response and isinstance(response["symptoms"], list):
                response["symptoms"] = [
                    translator.translate(symptom, detected_language)
                    for symptom in response["symptoms"]
                ]
            
            if "treatment" in response and isinstance(response["treatment"], list):
                response["treatment"] = [
                    translator.translate(treatment, detected_language)
                    for treatment in response["treatment"]
                ]
            
            if "prevention" in response and isinstance(response["prevention"], list):
                response["prevention"] = [
                    translator.translate(prevention, detected_language)
                    for prevention in response["prevention"]
                ]
        
        return {
            "status": "success",
            "response": response,
            "language": detected_language,
            "original_query": query.query,
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

@app.post("/chat/gemini")
async def chat_with_gemini(query: FarmerQuery):
    """
    Enhanced chat endpoint using Google Gemini with session context
    Maintains conversation history and uses all available context
    """
    if not gemini_available or not isinstance(ai_assistant, GeminiAssistant):
        raise HTTPException(
            status_code=503, 
            detail="Gemini AI is not available. Please configure GEMINI_API_KEY in .env file"
        )
    
    try:
        # Collect comprehensive context if location provided
        context_data = {}
        if query.location:
            # Get real-time data
            comprehensive_data = data_collector.get_comprehensive_data(
                query.location.latitude,
                query.location.longitude,
                query.location.region or "Default"
            )
            
            context_data = {
                'location': f"{query.location.region or 'Location'} ({query.location.latitude}, {query.location.longitude})",
                'soil_data': comprehensive_data.get('soil', {}),
                'weather_data': comprehensive_data.get('weather', {}),
                'market_data': comprehensive_data.get('market', {})
            }
            
            # Get recent crop recommendations if available
            try:
                features = DataProcessor.prepare_features(comprehensive_data)
                recommendations = crop_model.predict_crops(features, top_n=3)
                context_data['crop_recommendations'] = recommendations
            except:
                pass
        
        # Merge with provided context
        if query.context:
            context_data.update(query.context)
        
        # Process with Gemini
        response_data = ai_assistant.process_query(
            query.query,
            session_id=query.session_id,
            context_data=context_data
        )
        
        # Translate response if needed
        if query.language != "en" and query.language != "auto":
            if 'response' in response_data:
                response_data['response'] = translator.translate(
                    response_data['response'], 
                    query.language
                )
        
        return {
            "status": "success",
            "response": response_data.get('response', ''),
            "session_id": response_data.get('session_id'),
            "context_used": response_data.get('context_used', False),
            "structured_advice": response_data.get('structured_advice', {}),
            "language": query.language,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/session/{session_id}")
async def get_session_summary(session_id: str):
    """
    Get summary of a chat session
    """
    if not gemini_available or not isinstance(ai_assistant, GeminiAssistant):
        raise HTTPException(
            status_code=503, 
            detail="Gemini AI is not available"
        )
    
    try:
        summary = ai_assistant.get_session_summary(session_id)
        if 'error' in summary:
            raise HTTPException(status_code=404, detail=summary['error'])
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a specific chat session
    """
    if not gemini_available or not isinstance(ai_assistant, GeminiAssistant):
        raise HTTPException(
            status_code=503, 
            detail="Gemini AI is not available"
        )
    
    try:
        ai_assistant.clear_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_supported_languages():
    """
    Get list of supported languages
    """
    try:
        if hasattr(translator, 'get_supported_languages'):
            languages = translator.get_supported_languages()
        else:
            # Fallback language list
            languages = [
                {"code": "en", "name": "English", "native_name": "English"},
                {"code": "hi", "name": "Hindi", "native_name": "हिन्दी"}
            ]
        
        return {
            "status": "success",
            "languages": languages,
            "translation_available": hasattr(translator, 'is_translation_available') and translator.is_translation_available()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui-translations/{language}")
async def get_ui_translations(language: str):
    """
    Get UI element translations for a specific language
    """
    try:
        if hasattr(translator, 'translate_ui_elements'):
            translations = translator.translate_ui_elements(language)
        else:
            # Basic translations
            translations = {
                "app_title": translator.translate("AI Crop Recommendation System", language),
                "get_recommendations": translator.translate("Get Recommendations", language),
                "submit": translator.translate("Submit", language),
                "cancel": translator.translate("Cancel", language),
                "loading": translator.translate("Loading", language),
                "error": translator.translate("Error", language),
                "success": translator.translate("Success", language)
            }
        
        return {
            "status": "success",
            "language": language,
            "translations": translations
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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
