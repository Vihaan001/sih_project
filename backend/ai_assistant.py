"""
AI Assistant Module for Natural Language Processing
Handles farmer queries in multiple languages
"""

from typing import Dict, List, Any
import re
import json

class AIAssistant:
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.common_queries = self._initialize_common_queries()
    
    def _load_knowledge_base(self) -> Dict:
        """Load agricultural knowledge base"""
        return {
            "crop_diseases": {
                "wheat_rust": {
                    "symptoms": ["yellow/orange pustules on leaves", "stunted growth"],
                    "treatment": ["Apply fungicide", "Use resistant varieties"],
                    "prevention": ["Crop rotation", "Timely sowing"]
                },
                "rice_blast": {
                    "symptoms": ["Diamond-shaped spots on leaves", "neck rot"],
                    "treatment": ["Spray tricyclazole", "Remove infected plants"],
                    "prevention": ["Use certified seeds", "Avoid excess nitrogen"]
                }
            },
            "fertilizers": {
                "urea": {"N": 46, "application": "Top dressing", "crops": ["wheat", "rice", "maize"]},
                "DAP": {"N": 18, "P": 46, "application": "Basal", "crops": ["all crops"]},
                "MOP": {"K": 60, "application": "Basal/Top", "crops": ["potato", "sugarcane"]}
            },
            "irrigation": {
                "drip": {"efficiency": "90%", "suitable": ["vegetables", "fruits"]},
                "sprinkler": {"efficiency": "75%", "suitable": ["wheat", "pulses"]},
                "flood": {"efficiency": "50%", "suitable": ["rice", "sugarcane"]}
            }
        }
    
    def _initialize_common_queries(self) -> Dict:
        """Initialize responses for common queries"""
        return {
            "seed_rate": {
                "wheat": "100-125 kg/hectare",
                "rice": "25-30 kg/hectare (transplanted), 75-80 kg/hectare (direct seeded)",
                "maize": "20-25 kg/hectare",
                "cotton": "10-12 kg/hectare"
            },
            "sowing_time": {
                "wheat": "October-November (Rabi)",
                "rice": "June-July (Kharif)",
                "maize": "June-July (Kharif), January-February (Rabi)",
                "cotton": "April-May"
            },
            "water_requirement": {
                "wheat": "400-500 mm",
                "rice": "1200-1500 mm",
                "maize": "500-600 mm",
                "cotton": "700-800 mm"
            }
        }
    
    def process_query(self, query: str, context: Dict = None, language: str = "en") -> Dict:
        """
        Process farmer query and return response
        """
        query_lower = query.lower()
        response = {}
        
        # Detect query intent
        intent = self._detect_intent(query_lower)
        
        if intent == "crop_recommendation":
            response = self._handle_crop_recommendation(query_lower, context)
        elif intent == "disease_identification":
            response = self._handle_disease_query(query_lower)
        elif intent == "fertilizer_advice":
            response = self._handle_fertilizer_query(query_lower)
        elif intent == "irrigation_guidance":
            response = self._handle_irrigation_query(query_lower)
        elif intent == "market_info":
            response = self._handle_market_query(query_lower, context)
        elif intent == "weather_info":
            response = self._handle_weather_query(query_lower, context)
        else:
            response = self._handle_general_query(query_lower)
        
        return response
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query"""
        intents = {
            "crop_recommendation": ["which crop", "what to grow", "suggest crop", "best crop"],
            "disease_identification": ["disease", "pest", "yellow leaves", "spots", "infection"],
            "fertilizer_advice": ["fertilizer", "urea", "dap", "npk", "nutrients"],
            "irrigation_guidance": ["water", "irrigation", "moisture", "drought"],
            "market_info": ["price", "market", "sell", "mandi"],
            "weather_info": ["weather", "rain", "temperature", "forecast"]
        }
        
        for intent, keywords in intents.items():
            if any(keyword in query for keyword in keywords):
                return intent
        
        return "general"
    
    def _handle_crop_recommendation(self, query: str, context: Dict = None) -> Dict:
        """Handle crop recommendation queries"""
        response = {
            "type": "crop_recommendation",
            "message": "Based on your query, here are my recommendations:",
            "suggestions": []
        }
        
        if context and "soil" in context and "weather" in context:
            # Use context to provide specific recommendations
            soil = context["soil"]
            weather = context["weather"]
            
            if soil.get("ph", 7) > 7.5:
                response["suggestions"].append("Consider crops that tolerate alkaline soil like cotton or sugarcane")
            elif soil.get("ph", 7) < 6:
                response["suggestions"].append("Your soil is acidic. Consider liming or grow crops like potato")
            
            if weather.get("rainfall", 0) < 50:
                response["suggestions"].append("Low rainfall expected. Consider drought-resistant crops or arrange irrigation")
            
            response["suggestions"].append("For detailed recommendations, please use the recommendation feature")
        else:
            response["message"] = "Please provide your location for specific crop recommendations"
            response["suggestions"] = [
                "Share your location for personalized recommendations",
                "You can also provide soil test results for better accuracy"
            ]
        
        return response
    
    def _handle_disease_query(self, query: str) -> Dict:
        """Handle disease-related queries"""
        response = {
            "type": "disease_identification",
            "message": "I can help you identify and treat crop diseases.",
            "suggestions": []
        }
        
        # Check for specific disease mentions
        if "rust" in query or "yellow" in query:
            disease_info = self.knowledge_base["crop_diseases"].get("wheat_rust", {})
            response["disease_identified"] = "Possible Wheat Rust"
            response["symptoms"] = disease_info.get("symptoms", [])
            response["treatment"] = disease_info.get("treatment", [])
            response["prevention"] = disease_info.get("prevention", [])
        elif "blast" in query or "spot" in query:
            disease_info = self.knowledge_base["crop_diseases"].get("rice_blast", {})
            response["disease_identified"] = "Possible Rice Blast"
            response["symptoms"] = disease_info.get("symptoms", [])
            response["treatment"] = disease_info.get("treatment", [])
            response["prevention"] = disease_info.get("prevention", [])
        else:
            response["suggestions"] = [
                "Please describe the symptoms in detail",
                "You can upload a photo for better diagnosis",
                "Common symptoms to look for: spots, discoloration, wilting, stunted growth"
            ]
        
        return response
    
    def _handle_fertilizer_query(self, query: str) -> Dict:
        """Handle fertilizer-related queries"""
        response = {
            "type": "fertilizer_advice",
            "message": "Here's information about fertilizers:",
            "suggestions": []
        }
        
        fertilizers = self.knowledge_base.get("fertilizers", {})
        
        if "urea" in query:
            fert_info = fertilizers.get("urea", {})
            response["fertilizer"] = "Urea"
            response["composition"] = f"Nitrogen: {fert_info.get('N', 0)}%"
            response["application"] = fert_info.get("application", "")
            response["suitable_crops"] = fert_info.get("crops", [])
        elif "dap" in query:
            fert_info = fertilizers.get("DAP", {})
            response["fertilizer"] = "DAP (Di-Ammonium Phosphate)"
            response["composition"] = f"N: {fert_info.get('N', 0)}%, P: {fert_info.get('P', 0)}%"
            response["application"] = fert_info.get("application", "")
            response["suitable_crops"] = fert_info.get("crops", [])
        else:
            response["suggestions"] = [
                "Apply fertilizers based on soil test results",
                "Follow recommended dose for your crop",
                "Split application gives better results",
                "Consider organic alternatives for sustainable farming"
            ]
        
        return response
    
    def _handle_irrigation_query(self, query: str) -> Dict:
        """Handle irrigation-related queries"""
        response = {
            "type": "irrigation_guidance",
            "message": "Irrigation recommendations:",
            "suggestions": []
        }
        
        irrigation_methods = self.knowledge_base.get("irrigation", {})
        
        if "drip" in query:
            method_info = irrigation_methods.get("drip", {})
            response["method"] = "Drip Irrigation"
            response["efficiency"] = method_info.get("efficiency", "")
            response["suitable_crops"] = method_info.get("suitable", [])
            response["advantages"] = ["Water saving", "Reduced weed growth", "Fertigation possible"]
        else:
            response["suggestions"] = [
                "Choose irrigation method based on crop and water availability",
                "Monitor soil moisture regularly",
                "Avoid over-irrigation to prevent waterlogging",
                "Consider micro-irrigation for water conservation"
            ]
        
        return response
    
    def _handle_market_query(self, query: str, context: Dict = None) -> Dict:
        """Handle market-related queries"""
        response = {
            "type": "market_information",
            "message": "Market information:",
            "suggestions": []
        }
        
        if context and "market" in context:
            market_data = context["market"]
            response["current_prices"] = {}
            for crop, info in list(market_data.items())[:3]:
                response["current_prices"][crop] = {
                    "price": f"₹{info.get('current_price', 0)}/quintal",
                    "trend": info.get("price_trend", "stable"),
                    "demand": info.get("demand", "medium")
                }
        else:
            response["suggestions"] = [
                "Check local mandi prices before selling",
                "Consider storage if prices are low",
                "Join farmer producer organizations for better prices",
                "Explore direct marketing options"
            ]
        
        return response
    
    def _handle_weather_query(self, query: str, context: Dict = None) -> Dict:
        """Handle weather-related queries"""
        response = {
            "type": "weather_information",
            "message": "Weather information:",
            "suggestions": []
        }
        
        if context and "weather" in context:
            weather = context["weather"]
            response["current_weather"] = {
                "temperature": f"{weather.get('temperature', 0)}°C",
                "humidity": f"{weather.get('humidity', 0)}%",
                "rainfall": f"{weather.get('rainfall', 0)} mm",
                "season": weather.get("season", "")
            }
            
            if weather.get("forecast_7_days"):
                forecast = weather["forecast_7_days"]
                response["forecast"] = {
                    "avg_temperature": f"{forecast.get('avg_temp', 0)}°C",
                    "expected_rainfall": f"{forecast.get('total_rainfall', 0)} mm"
                }
        else:
            response["suggestions"] = [
                "Monitor weather forecasts regularly",
                "Plan agricultural operations based on weather",
                "Prepare for extreme weather events",
                "Use weather-based crop insurance"
            ]
        
        return response
    
    def _handle_general_query(self, query: str) -> Dict:
        """Handle general queries"""
        response = {
            "type": "general",
            "message": "I can help you with various agricultural queries.",
            "suggestions": [
                "Ask about crop recommendations for your area",
                "Get help with pest and disease identification",
                "Learn about fertilizer application",
                "Get market price information",
                "Check weather forecasts",
                "Plan crop rotation"
            ]
        }
        
        # Check common queries
        for topic, info in self.common_queries.items():
            if topic.replace("_", " ") in query:
                response["information"] = info
                break
        
        return response
    
    def get_common_responses(self) -> Dict:
        """Get common responses for offline use"""
        return {
            "greetings": {
                "hello": "Hello! I'm your agricultural assistant. How can I help you today?",
                "namaste": "Namaste! मैं आपका कृषि सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?"
            },
            "common_questions": self.common_queries,
            "tips": {
                "general": [
                    "Test your soil every 2-3 years",
                    "Maintain crop rotation for soil health",
                    "Use organic matter to improve soil structure",
                    "Monitor weather forecasts regularly",
                    "Keep records of all agricultural activities"
                ]
            }
        }
