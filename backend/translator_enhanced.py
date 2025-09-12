"""
Enhanced Translator Module using deep-translator
Provides automatic translation capabilities using Google Translate via deep-translator
"""

from typing import Dict, List, Any, Optional
from deep_translator import GoogleTranslator, single_detection
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTranslator:
    def __init__(self):
        """
        Initialize translator with deep-translator's Google Translate
        Falls back to dictionary-based translation if API fails
        """
        self.use_google = True
        
        # Language code mapping for Indian languages
        self.language_codes = {
            'en': 'en',  # English
            'hi': 'hi',  # Hindi
            'mr': 'mr',  # Marathi
            'bn': 'bn',  # Bengali
            'te': 'te',  # Telugu
            'ta': 'ta',  # Tamil
            'gu': 'gu',  # Gujarati
            'kn': 'kn',  # Kannada
            'ml': 'ml',  # Malayalam
            'pa': 'pa',  # Punjabi
            'or': 'or',  # Odia
            'as': 'as',  # Assamese
            'ur': 'ur',  # Urdu
        }
        
        # Initialize translators for commonly used languages
        self.translators = {}
        self._initialize_translators()
        
        # Fallback dictionary translations
        self.fallback_translations = self._load_fallback_translations()
        
        # Cache for translations to reduce API calls
        self.translation_cache = {}
    
    def _initialize_translators(self):
        """Initialize Google Translators for common language pairs"""
        try:
            # Create translators for English to Indian languages
            for lang_code in ['hi', 'mr', 'bn', 'te', 'ta']:
                self.translators[f'en_{lang_code}'] = GoogleTranslator(
                    source='en', 
                    target=lang_code
                )
            
            # Create auto-detect translator
            self.translators['auto'] = GoogleTranslator(source='auto', target='en')
            
            logger.info("Google Translators initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Google Translators: {e}")
            self.use_google = False
    
    def _load_fallback_translations(self) -> Dict:
        """Load fallback translation dictionary for when Google Translate is unavailable"""
        return {
            "hi": {  # Hindi translations
                # Crops
                "wheat": "गेहूं",
                "rice": "चावल",
                "paddy": "धान",
                "maize": "मक्का",
                "corn": "मक्का",
                "cotton": "कपास",
                "sugarcane": "गन्ना",
                "soybean": "सोयाबीन",
                "groundnut": "मूंगफली",
                "peanut": "मूंगफली",
                "pulses": "दालें",
                "lentil": "मसूर",
                "chickpea": "चना",
                "gram": "चना",
                "pigeon pea": "अरहर",
                "arhar": "अरहर",
                "moong": "मूंग",
                "mung bean": "मूंग",
                "urad": "उड़द",
                "black gram": "उड़द",
                "mustard": "सरसों",
                "jowar": "ज्वार",
                "sorghum": "ज्वार",
                "bajra": "बाजरा",
                "pearl millet": "बाजरा",
                "potato": "आलू",
                "onion": "प्याज",
                "tomato": "टमाटर",
                "vegetables": "सब्जियां",
                
                # Agricultural terms
                "crop": "फसल",
                "soil": "मिट्टी",
                "water": "पानी",
                "fertilizer": "खाद",
                "organic": "जैविक",
                "pesticide": "कीटनाशक",
                "insecticide": "कीटनाशक",
                "herbicide": "खरपतवारनाशी",
                "seed": "बीज",
                "sowing": "बुवाई",
                "harvest": "कटाई",
                "harvesting": "कटाई",
                "irrigation": "सिंचाई",
                "drip": "ड्रिप",
                "sprinkler": "स्प्रिंकलर",
                "yield": "उपज",
                "production": "उत्पादन",
                "productivity": "उत्पादकता",
                "farm": "खेत",
                "field": "खेत",
                "farmer": "किसान",
                "agriculture": "कृषि",
                "cultivation": "खेती",
                
                # Soil terms
                "ph": "पीएच",
                "nitrogen": "नाइट्रोजन",
                "phosphorus": "फास्फोरस",
                "potassium": "पोटेशियम",
                "organic carbon": "जैविक कार्बन",
                "moisture": "नमी",
                "texture": "बनावट",
                "clay": "चिकनी मिट्टी",
                "sandy": "रेतीली",
                "loamy": "दोमट",
                "silt": "गाद",
                
                # Weather terms
                "weather": "मौसम",
                "temperature": "तापमान",
                "rainfall": "वर्षा",
                "rain": "बारिश",
                "humidity": "नमी",
                "drought": "सूखा",
                "flood": "बाढ़",
                "monsoon": "मानसून",
                "winter": "सर्दी",
                "summer": "गर्मी",
                "spring": "वसंत",
                "autumn": "शरद",
                "season": "मौसम",
                "kharif": "खरीफ",
                "rabi": "रबी",
                "zaid": "जायद",
                
                # Market terms
                "market": "बाज़ार",
                "mandi": "मंडी",
                "price": "कीमत",
                "rate": "दर",
                "cost": "लागत",
                "profit": "लाभ",
                "loss": "हानि",
                "demand": "मांग",
                "supply": "आपूर्ति",
                "sell": "बेचना",
                "buy": "खरीदना",
                "trade": "व्यापार",
                
                # UI terms
                "recommendation": "सिफारिश",
                "suggest": "सुझाव",
                "advice": "सलाह",
                "high": "उच्च",
                "medium": "मध्यम",
                "low": "कम",
                "good": "अच्छा",
                "bad": "बुरा",
                "average": "औसत",
                "excellent": "उत्कृष्ट",
                "poor": "खराब",
                
                # Common phrases
                "welcome": "स्वागत है",
                "thank you": "धन्यवाद",
                "please": "कृपया",
                "sorry": "क्षमा करें",
                "yes": "हां",
                "no": "नहीं",
                "okay": "ठीक है",
                "submit": "जमा करें",
                "cancel": "रद्द करें",
                "next": "अगला",
                "previous": "पिछला",
                "back": "वापस",
                "close": "बंद करें",
                "save": "सहेजें",
                "delete": "हटाएं",
                "edit": "संपादित करें",
                "view": "देखें",
                "search": "खोजें",
                "filter": "फ़िल्टर",
                "sort": "क्रमबद्ध करें",
                "download": "डाउनलोड",
                "upload": "अपलोड",
                "loading": "लोड हो रहा है",
                "error": "त्रुटि",
                "success": "सफलता",
                "warning": "चेतावनी",
                "info": "जानकारी",
            }
        }
    
    def translate(self, text: str, target_language: str, source_language: str = 'auto') -> str:
        """
        Translate text using deep-translator's Google Translate
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'hi' for Hindi)
            source_language: Source language code (default: 'auto' for auto-detect)
        
        Returns:
            Translated text
        """
        if not text or text.strip() == '':
            return text
        
        # Return original if target is same as source or English
        if target_language == source_language:
            return text
        
        # Check cache first
        cache_key = f"{text}_{source_language}_{target_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Try Google Translate via deep-translator
        if self.use_google:
            try:
                # Create translator for this language pair if not exists
                translator_key = f"{source_language}_{target_language}"
                if translator_key not in self.translators:
                    self.translators[translator_key] = GoogleTranslator(
                        source=source_language,
                        target=self.language_codes.get(target_language, target_language)
                    )
                
                translator = self.translators[translator_key]
                translated = translator.translate(text)
                
                # Cache the translation
                self.translation_cache[cache_key] = translated
                
                return translated
                
            except Exception as e:
                logger.warning(f"Google Translate failed for '{text}': {e}")
                # Fall back to dictionary
        
        # Use fallback dictionary translation
        return self._fallback_translate(text, target_language)
    
    def _fallback_translate(self, text: str, target_language: str) -> str:
        """
        Fallback translation using dictionary
        """
        if target_language not in self.fallback_translations:
            return text
        
        translations = self.fallback_translations[target_language]
        text_lower = text.lower()
        
        # First try exact match
        if text_lower in translations:
            return translations[text_lower]
        
        # Try word by word translation
        words = text_lower.split()
        translated_words = []
        
        for word in words:
            if word in translations:
                translated_words.append(translations[word])
            else:
                # Keep original word if no translation found
                translated_words.append(word)
        
        result = " ".join(translated_words)
        
        # Capitalize first letter if original was capitalized
        if text and text[0].isupper():
            result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
        
        return result
    
    def translate_batch(self, texts: List[str], target_language: str, source_language: str = 'auto') -> List[str]:
        """
        Translate multiple texts at once
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code
        
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        translated = []
        for text in texts:
            translated.append(self.translate(text, target_language, source_language))
        
        return translated
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text
        
        Args:
            text: Text to analyze
        
        Returns:
            Language code (e.g., 'hi' for Hindi)
        """
        if not text or text.strip() == '':
            return 'en'
        
        try:
            # Use deep-translator's language detection
            detected = single_detection(text, api_key=None)
            return detected
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            # Fallback to character-based detection
            return self._detect_by_characters(text)
    
    def _detect_by_characters(self, text: str) -> str:
        """
        Detect language based on Unicode character ranges
        """
        # Count characters from different scripts
        devanagari = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        bengali = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        tamil = sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')
        telugu = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
        gujarati = sum(1 for char in text if '\u0A80' <= char <= '\u0AFF')
        
        # Find the script with most characters
        scripts = {
            'hi': devanagari,  # Hindi uses Devanagari
            'bn': bengali,
            'ta': tamil,
            'te': telugu,
            'gu': gujarati,
        }
        
        max_script = max(scripts.items(), key=lambda x: x[1])
        
        # Return the detected language if significant characters found
        if max_script[1] > len(text) * 0.3:  # At least 30% of text
            return max_script[0]
        
        # Default to English
        return 'en'
    
    def translate_crop_recommendation(self, recommendation: Dict, target_language: str) -> Dict:
        """
        Translate a complete crop recommendation
        
        Args:
            recommendation: Dictionary containing crop recommendation
            target_language: Target language code
        
        Returns:
            Translated recommendation dictionary
        """
        if target_language == 'en':
            return recommendation
        
        translated = recommendation.copy()
        
        # Translate crop name
        if 'crop' in translated:
            translated['crop'] = self.translate(str(recommendation['crop']), target_language)
        
        # Translate profit margin
        if 'profit_margin' in translated:
            translated['profit_margin'] = self.translate(str(recommendation['profit_margin']), target_language)
        
        # Translate care tips
        if 'care_tips' in translated and isinstance(translated['care_tips'], list):
            translated['care_tips'] = self.translate_batch(recommendation['care_tips'], target_language)
        
        # Translate suitable season
        if 'suitable_season' in translated and isinstance(translated['suitable_season'], list):
            translated['suitable_season'] = self.translate_batch(recommendation['suitable_season'], target_language)
        
        # Add language metadata
        translated['language'] = target_language
        translated['original_language'] = 'en'
        
        return translated
    
    def translate_ui_elements(self, target_language: str) -> Dict[str, str]:
        """
        Get all UI translations for a specific language
        
        Args:
            target_language: Target language code
        
        Returns:
            Dictionary of translated UI elements
        """
        ui_elements = {
            # Headers
            'app_title': 'AI Crop Recommendation System',
            'subtitle': 'Smart farming decisions powered by artificial intelligence',
            
            # Navigation
            'home': 'Home',
            'recommendations': 'Recommendations',
            'soil_analysis': 'Soil Analysis',
            'weather': 'Weather',
            'market': 'Market',
            'chat': 'Chat',
            'settings': 'Settings',
            'help': 'Help',
            'about': 'About',
            
            # Forms
            'enter_location': 'Enter Location',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'select_region': 'Select Region',
            'select_language': 'Select Language',
            'soil_ph': 'Soil pH',
            'nitrogen_content': 'Nitrogen Content',
            'phosphorus_content': 'Phosphorus Content',
            'potassium_content': 'Potassium Content',
            'organic_carbon': 'Organic Carbon',
            'soil_moisture': 'Soil Moisture',
            'soil_texture': 'Soil Texture',
            
            # Buttons
            'submit': 'Submit',
            'cancel': 'Cancel',
            'reset': 'Reset',
            'get_recommendations': 'Get Recommendations',
            'refresh': 'Refresh',
            'save': 'Save',
            'download': 'Download',
            'share': 'Share',
            'print': 'Print',
            
            # Results
            'recommended_crops': 'Recommended Crops',
            'confidence': 'Confidence',
            'expected_yield': 'Expected Yield',
            'profit_margin': 'Profit Margin',
            'market_price': 'Market Price',
            'best_season': 'Best Season',
            'care_instructions': 'Care Instructions',
            'crop_rotation': 'Crop Rotation',
            
            # Messages
            'loading': 'Loading...',
            'please_wait': 'Please wait...',
            'success': 'Success!',
            'error': 'Error',
            'warning': 'Warning',
            'info': 'Information',
            'no_data': 'No data available',
            'offline_mode': 'Offline Mode',
            'online_mode': 'Online Mode',
            'connection_error': 'Connection Error',
            'try_again': 'Try Again',
            
            # Chat
            'type_message': 'Type your message...',
            'send': 'Send',
            'chat_placeholder': 'Ask me anything about farming...',
            'assistant_typing': 'Assistant is typing...',
        }
        
        if target_language == 'en':
            return ui_elements
        
        # Translate all UI elements
        translated_ui = {}
        for key, value in ui_elements.items():
            translated_ui[key] = self.translate(value, target_language)
        
        return translated_ui
    
    def get_supported_languages(self) -> List[Dict]:
        """
        Get list of supported languages
        
        Returns:
            List of dictionaries containing language information
        """
        languages = [
            {"code": "en", "name": "English", "native_name": "English", "flag": "🇬🇧", "available": True},
            {"code": "hi", "name": "Hindi", "native_name": "हिन्दी", "flag": "🇮🇳", "available": True},
            {"code": "mr", "name": "Marathi", "native_name": "मराठी", "flag": "🇮🇳", "available": True},
            {"code": "bn", "name": "Bengali", "native_name": "বাংলা", "flag": "🇮🇳", "available": True},
            {"code": "te", "name": "Telugu", "native_name": "తెలుగు", "flag": "🇮🇳", "available": True},
            {"code": "ta", "name": "Tamil", "native_name": "தமிழ்", "flag": "🇮🇳", "available": True},
            {"code": "gu", "name": "Gujarati", "native_name": "ગુજરાતી", "flag": "🇮🇳", "available": True},
            {"code": "kn", "name": "Kannada", "native_name": "ಕನ್ನಡ", "flag": "🇮🇳", "available": True},
            {"code": "ml", "name": "Malayalam", "native_name": "മലയാളം", "flag": "🇮🇳", "available": True},
            {"code": "pa", "name": "Punjabi", "native_name": "ਪੰਜਾਬੀ", "flag": "🇮🇳", "available": True},
            {"code": "or", "name": "Odia", "native_name": "ଓଡ଼ିଆ", "flag": "🇮🇳", "available": self.use_google},
            {"code": "as", "name": "Assamese", "native_name": "অসমীয়া", "flag": "🇮🇳", "available": self.use_google},
        ]
        
        return languages
    
    def is_translation_available(self) -> bool:
        """
        Check if translation service is available
        
        Returns:
            True if Google Translate is available, False otherwise
        """
        return self.use_google
    
    def clear_cache(self):
        """Clear the translation cache"""
        self.translation_cache.clear()
        logger.info("Translation cache cleared")


# Example usage and testing
if __name__ == "__main__":
    print("🌐 Enhanced Translator Module Test")
    print("=" * 60)
    
    # Initialize translator
    translator = EnhancedTranslator()
    
    print(f"Translation Service: {'✅ Available' if translator.is_translation_available() else '⚠️ Limited (Fallback mode)'}")
    print("\n📚 Supported Languages:")
    for lang in translator.get_supported_languages():
        status = "✅" if lang['available'] else "❌"
        print(f"  {status} {lang['flag']} {lang['native_name']} ({lang['name']}) - Code: {lang['code']}")
    
    # Test translations
    print("\n" + "=" * 60)
    print("🧪 Translation Tests (English → Hindi)")
    print("=" * 60)
    
    test_phrases = [
        "wheat",
        "rice",
        "Apply fertilizer in two splits",
        "The soil pH is 6.5",
        "Recommended crops for your field",
        "High profit margin",
        "Maintain proper irrigation",
        "Market price is increasing",
        "Harvest at right time",
        "Good yield expected"
    ]
    
    for phrase in test_phrases:
        translated = translator.translate(phrase, 'hi')
        print(f"EN: {phrase}")
        print(f"HI: {translated}")
        print()
    
    # Test crop recommendation translation
    print("=" * 60)
    print("🌾 Crop Recommendation Translation")
    print("=" * 60)
    
    sample_recommendation = {
        "crop": "Wheat",
        "confidence": 0.95,
        "profit_margin": "High",
        "predicted_yield": 4.5,
        "care_tips": [
            "Maintain proper irrigation schedule",
            "Apply fertilizer at right time",
            "Monitor for pests and diseases",
            "Harvest when grain is fully mature"
        ],
        "suitable_season": ["Rabi", "Winter"]
    }
    
    print("Original (English):")
    print(json.dumps(sample_recommendation, indent=2))
    
    print("\nTranslated (Hindi):")
    translated_rec = translator.translate_crop_recommendation(sample_recommendation, 'hi')
    print(json.dumps(translated_rec, indent=2, ensure_ascii=False))
    
    # Test language detection
    print("\n" + "=" * 60)
    print("🔍 Language Detection Test")
    print("=" * 60)
    
    test_texts = [
        "This is English text",
        "यह हिंदी पाठ है",
        "এটি বাংলা পাঠ্য",
        "இது தமிழ் உரை",
    ]
    
    for text in test_texts:
        detected = translator.detect_language(text)
        print(f"Text: {text}")
        print(f"Detected: {detected}")
        print()
    
    print("✅ All tests completed!")
