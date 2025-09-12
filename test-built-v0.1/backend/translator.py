"""
Translator Module for Multilingual Support
Provides translation capabilities for the application
"""

from typing import Dict, List, Any

class Translator:
    def __init__(self):
        # For MVP, using simple translation dictionary
        # In production, integrate with Google Translate API or similar
        self.translations = self._load_translations()
    
    def _load_translations(self) -> Dict:
        """Load translation dictionary"""
        return {
            "hi": {  # Hindi translations
                "crop": "फसल",
                "wheat": "गेहूं",
                "rice": "चावल",
                "maize": "मक्का",
                "cotton": "कपास",
                "sugarcane": "गन्ना",
                "soybean": "सोयाबीन",
                "groundnut": "मूंगफली",
                "pulses": "दालें",
                "soil": "मिट्टी",
                "water": "पानी",
                "fertilizer": "खाद",
                "disease": "रोग",
                "pest": "कीट",
                "market": "बाजार",
                "price": "कीमत",
                "weather": "मौसम",
                "temperature": "तापमान",
                "rainfall": "वर्षा",
                "humidity": "नमी",
                "recommendation": "सिफारिश",
                "yield": "उपज",
                "profit": "लाभ",
                "season": "मौसम",
                "kharif": "खरीफ",
                "rabi": "रबी",
                "high": "उच्च",
                "medium": "मध्यम",
                "low": "कम",
                "messages": {
                    "welcome": "नमस्ते! मैं आपका कृषि सहायक हूं।",
                    "how_can_help": "मैं आपकी कैसे मदद कर सकता हूं?",
                    "recommendation_ready": "आपकी फसल की सिफारिशें तैयार हैं",
                    "based_on_soil": "आपकी मिट्टी के आधार पर",
                    "suitable_crops": "उपयुक्त फसलें",
                    "expected_yield": "अपेक्षित उपज",
                    "market_price": "बाजार मूल्य",
                    "care_tips": "देखभाल सुझाव"
                }
            },
            "mr": {  # Marathi translations (for Jharkhand region)
                "crop": "पीक",
                "wheat": "गहू",
                "rice": "तांदूळ",
                "soil": "माती",
                "water": "पाणी",
                "fertilizer": "खत",
                "recommendation": "शिफारस"
            },
            "bn": {  # Bengali translations
                "crop": "ফসল",
                "wheat": "গম",
                "rice": "ধান",
                "soil": "মাটি",
                "water": "জল",
                "fertilizer": "সার",
                "recommendation": "সুপারিশ"
            }
        }
    
    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text to target language
        """
        if target_language == "en":
            return text
        
        translations = self.translations.get(target_language, {})
        
        # Simple word-by-word translation for MVP
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            if word in translations:
                translated_words.append(translations[word])
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)
    
    def translate_recommendations(self, recommendations: List[Dict], 
                                target_language: str) -> List[Dict]:
        """
        Translate crop recommendations to target language
        """
        if target_language == "en":
            return recommendations
        
        translations = self.translations.get(target_language, {})
        translated_recs = []
        
        for rec in recommendations:
            translated_rec = rec.copy()
            
            # Translate crop name
            crop_name = rec.get("crop", "")
            translated_rec["crop"] = translations.get(crop_name.lower(), crop_name)
            
            # Translate profit margin
            profit_margin = rec.get("profit_margin", "")
            translated_rec["profit_margin"] = translations.get(profit_margin.lower(), profit_margin)
            
            # Translate care tips if simple enough
            if "care_tips" in rec and isinstance(rec["care_tips"], list):
                # Keep original for now (complex translation needed)
                translated_rec["care_tips_original"] = rec["care_tips"]
                translated_rec["care_tips"] = self._translate_tips(rec["care_tips"], target_language)
            
            translated_recs.append(translated_rec)
        
        return translated_recs
    
    def _translate_tips(self, tips: List[str], target_language: str) -> List[str]:
        """
        Translate care tips (simplified for MVP)
        """
        if target_language == "hi":
            # Provide some pre-translated common tips
            tip_translations = {
                "Maintain standing water of 2-5 cm during early growth": "प्रारंभिक विकास के दौरान 2-5 सेमी खड़ा पानी बनाए रखें",
                "Apply nitrogen in 3 splits for better yield": "बेहतर उपज के लिए नाइट्रोजन को 3 भागों में दें",
                "Watch for stem borer and leaf folder pests": "तना छेदक और पत्ता मोड़क कीटों पर नजर रखें",
                "Ensure proper seed rate (100-125 kg/ha)": "उचित बीज दर सुनिश्चित करें (100-125 किग्रा/हेक्टेयर)",
                "Control weeds in first 30-45 days": "पहले 30-45 दिनों में खरपतवार नियंत्रण करें"
            }
            
            translated = []
            for tip in tips:
                translated.append(tip_translations.get(tip, tip))
            return translated
        
        return tips
    
    def get_supported_languages(self) -> List[Dict]:
        """
        Get list of supported languages
        """
        return [
            {"code": "en", "name": "English", "native_name": "English"},
            {"code": "hi", "name": "Hindi", "native_name": "हिन्दी"},
            {"code": "mr", "name": "Marathi", "native_name": "मराठी"},
            {"code": "bn", "name": "Bengali", "native_name": "বাংলা"}
        ]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of the text (simplified for MVP)
        """
        # Check for Hindi characters
        if any('\u0900' <= char <= '\u097F' for char in text):
            return "hi"
        # Check for Bengali characters
        elif any('\u0980' <= char <= '\u09FF' for char in text):
            return "bn"
        # Check for Marathi (uses Devanagari like Hindi)
        elif any('\u0900' <= char <= '\u097F' for char in text):
            return "mr"
        else:
            return "en"
