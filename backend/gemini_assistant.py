"""
Google Gemini AI Assistant for Agricultural Queries
Maintains session context and provides detailed farming advice
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import hashlib
import pickle

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiAssistant:
    def __init__(self):
        """Initialize Gemini AI Assistant with agricultural context"""
        # Configure Gemini API
        self.api_key = os.getenv('GEMINI_API_KEY', '')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.is_configured = True
            logger.info("Gemini AI configured successfully")
        else:
            self.model = None
            self.is_configured = False
            logger.warning("Gemini API key not found. Please set GEMINI_API_KEY in .env file")
        
        # Session management
        self.sessions = {}  # Store active sessions
        self.session_timeout = 3600  # 1 hour timeout
        
        # System prompt for agricultural context
        self.system_prompt = """You are an expert agricultural advisor AI assistant for the Crop Recommendation System. 
        You help farmers with detailed, practical advice about:
        - Crop cultivation and recommendations
        - Fertilizer usage and soil management
        - Pest and disease control
        - Irrigation techniques
        - Market information and pricing
        - Weather-related farming decisions
        - Sustainable farming practices
        
        Guidelines:
        1. Always provide practical, actionable advice
        2. Consider the local context (especially for Indian/Jharkhand region when mentioned)
        3. Use simple language that farmers can understand
        4. Include specific measurements, timings, and quantities when relevant
        5. Suggest both modern and traditional farming methods when appropriate
        6. Consider cost-effectiveness in your recommendations
        7. Promote sustainable and organic farming practices when possible
        
        Current Context Information will be provided with each query."""
        
        # Cache for storing responses
        self.response_cache = {}
        
    def create_session(self, user_id: str) -> str:
        """Create a new chat session for a user"""
        session_id = hashlib.md5(f"{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        if self.is_configured:
            # Create a new chat session with Gemini
            chat = self.model.start_chat(history=[])
            
            self.sessions[session_id] = {
                'chat': chat,
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'context': {},
                'conversation_history': [],
                'crop_recommendations': [],
                'location': None
            }
            
            logger.info(f"Created new session: {session_id}")
        else:
            # Fallback session without Gemini
            self.sessions[session_id] = {
                'chat': None,
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'context': {},
                'conversation_history': [],
                'crop_recommendations': [],
                'location': None
            }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get an existing session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Check if session is expired
            if datetime.now() - session['last_activity'] > timedelta(seconds=self.session_timeout):
                logger.info(f"Session {session_id} expired")
                del self.sessions[session_id]
                return None
            # Update last activity
            session['last_activity'] = datetime.now()
            return session
        return None
    
    def update_session_context(self, session_id: str, context_data: Dict):
        """Update session with new context data"""
        session = self.get_session(session_id)
        if session:
            # Update context
            if 'location' in context_data:
                session['location'] = context_data['location']
            
            if 'soil_data' in context_data:
                session['context']['soil'] = context_data['soil_data']
            
            if 'weather_data' in context_data:
                session['context']['weather'] = context_data['weather_data']
            
            if 'crop_recommendations' in context_data:
                session['crop_recommendations'] = context_data['crop_recommendations']
            
            if 'market_data' in context_data:
                session['context']['market'] = context_data['market_data']
            
            logger.info(f"Updated context for session {session_id}")
    
    def build_context_prompt(self, session: Dict, include_history: bool = True) -> str:
        """Build context prompt with all available information"""
        context_parts = []
        
        # Add location context
        if session.get('location'):
            context_parts.append(f"Location: {session['location']}")
        
        # Add soil data
        if session.get('context', {}).get('soil'):
            soil = session['context']['soil']
            soil_info = f"Soil Data: pH={soil.get('ph', 'N/A')}, "
            soil_info += f"N={soil.get('nitrogen', 'N/A')} kg/ha, "
            soil_info += f"P={soil.get('phosphorus', 'N/A')} kg/ha, "
            soil_info += f"K={soil.get('potassium', 'N/A')} kg/ha, "
            soil_info += f"Organic Carbon={soil.get('organic_carbon', 'N/A')}%, "
            soil_info += f"Texture={soil.get('texture', 'N/A')}"
            context_parts.append(soil_info)
        
        # Add weather data
        if session.get('context', {}).get('weather'):
            weather = session['context']['weather']
            weather_info = f"Weather: Temperature={weather.get('temperature', 'N/A')}°C, "
            weather_info += f"Humidity={weather.get('humidity', 'N/A')}%, "
            weather_info += f"Rainfall={weather.get('rainfall', 'N/A')}mm, "
            weather_info += f"Season={weather.get('season', 'N/A')}"
            context_parts.append(weather_info)
        
        # Add recent crop recommendations
        if session.get('crop_recommendations'):
            crops = [rec.get('crop', '') for rec in session['crop_recommendations'][:3]]
            if crops:
                context_parts.append(f"Recent Crop Recommendations: {', '.join(crops)}")
        
        # Add market data if available
        if session.get('context', {}).get('market'):
            market = session['context']['market']
            # Add top 3 crop prices
            market_info = "Market Prices: "
            for i, (crop, info) in enumerate(list(market.items())[:3]):
                if i > 0:
                    market_info += ", "
                market_info += f"{crop}=₹{info.get('current_price', 'N/A')}/quintal"
            context_parts.append(market_info)
        
        # Add conversation history summary if needed
        if include_history and session.get('conversation_history'):
            recent_topics = []
            for entry in session['conversation_history'][-3:]:  # Last 3 interactions
                if 'query' in entry:
                    # Extract key topics from recent queries
                    query = entry['query'].lower()
                    if 'fertilizer' in query:
                        recent_topics.append('fertilizer')
                    elif 'disease' in query or 'pest' in query:
                        recent_topics.append('pest/disease')
                    elif 'irrigation' in query or 'water' in query:
                        recent_topics.append('irrigation')
                    elif 'harvest' in query:
                        recent_topics.append('harvesting')
            
            if recent_topics:
                context_parts.append(f"Recent Discussion Topics: {', '.join(set(recent_topics))}")
        
        # Combine all context
        full_context = "\n".join(context_parts) if context_parts else "No specific context available"
        
        return f"""
        CURRENT CONTEXT:
        {full_context}
        
        Please provide advice considering the above context when relevant.
        """
    
    def process_query(self, query: str, session_id: str = None, context_data: Dict = None) -> Dict:
        """
        Process user query with Gemini AI
        
        Args:
            query: User's question
            session_id: Session ID for maintaining context
            context_data: Additional context (soil, weather, recommendations, etc.)
        
        Returns:
            Response dictionary with detailed answer
        """
        # Create or get session
        if not session_id:
            session_id = self.create_session("anonymous")
        
        session = self.get_session(session_id)
        if not session:
            session_id = self.create_session("anonymous")
            session = self.get_session(session_id)
        
        # Update session context if provided
        if context_data:
            self.update_session_context(session_id, context_data)
        
        # Check cache for similar queries
        cache_key = hashlib.md5(f"{query}_{json.dumps(context_data or {})}".encode()).hexdigest()
        if cache_key in self.response_cache:
            logger.info("Returning cached response")
            cached_response = self.response_cache[cache_key]
            cached_response['cached'] = True
            return cached_response
        
        # Process with Gemini if configured
        if self.is_configured and session['chat']:
            try:
                # Build full prompt with context
                context_prompt = self.build_context_prompt(session)
                full_prompt = f"{self.system_prompt}\n\n{context_prompt}\n\nUser Query: {query}"
                
                # Get response from Gemini
                response = session['chat'].send_message(full_prompt)
                
                # Parse and structure the response
                result = {
                    'status': 'success',
                    'response': response.text,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'context_used': bool(session.get('context')),
                    'cached': False
                }
                
                # Extract specific advice if present
                result['structured_advice'] = self._extract_structured_advice(response.text, query)
                
                # Update conversation history
                session['conversation_history'].append({
                    'query': query,
                    'response': response.text,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only last 10 interactions in history
                if len(session['conversation_history']) > 10:
                    session['conversation_history'] = session['conversation_history'][-10:]
                
                # Cache the response
                self.response_cache[cache_key] = result
                
                # Limit cache size
                if len(self.response_cache) > 100:
                    # Remove oldest entries
                    self.response_cache = dict(list(self.response_cache.items())[-50:])
                
                return result
                
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return self._get_fallback_response(query, session)
        else:
            return self._get_fallback_response(query, session)
    
    def _extract_structured_advice(self, response_text: str, query: str) -> Dict:
        """Extract structured advice from Gemini response"""
        structured = {}
        
        query_lower = query.lower()
        
        # Extract fertilizer advice
        if 'fertilizer' in query_lower:
            structured['type'] = 'fertilizer_advice'
            # Try to extract dosage
            import re
            dosage_pattern = r'(\d+[-\s]?\d*)\s*(kg|gram|g|litre|l|ml)/?(hectare|acre|ha|plant)?'
            dosages = re.findall(dosage_pattern, response_text, re.IGNORECASE)
            if dosages:
                structured['dosage'] = [f"{d[0]} {d[1]}/{d[2] if d[2] else 'unit'}" for d in dosages]
            
            # Extract timing
            if 'before' in response_text.lower() or 'after' in response_text.lower() or 'during' in response_text.lower():
                structured['timing_mentioned'] = True
        
        # Extract disease/pest advice
        elif 'disease' in query_lower or 'pest' in query_lower:
            structured['type'] = 'pest_disease_advice'
            # Check for treatment mentions
            if 'spray' in response_text.lower() or 'apply' in response_text.lower():
                structured['treatment_mentioned'] = True
            if 'prevent' in response_text.lower():
                structured['prevention_mentioned'] = True
        
        # Extract irrigation advice
        elif 'water' in query_lower or 'irrigation' in query_lower:
            structured['type'] = 'irrigation_advice'
            # Look for water quantity
            water_pattern = r'(\d+[-\s]?\d*)\s*(mm|cm|litre|l|inch)'
            water_amounts = re.findall(water_pattern, response_text, re.IGNORECASE)
            if water_amounts:
                structured['water_amount'] = [f"{w[0]} {w[1]}" for w in water_amounts]
        
        # Extract crop-specific advice
        elif 'grow' in query_lower or 'cultivat' in query_lower or 'plant' in query_lower:
            structured['type'] = 'cultivation_advice'
            # Check for season mentions
            seasons = ['kharif', 'rabi', 'zaid', 'summer', 'winter', 'monsoon']
            mentioned_seasons = [s for s in seasons if s in response_text.lower()]
            if mentioned_seasons:
                structured['seasons'] = mentioned_seasons
        
        return structured
    
    def _get_fallback_response(self, query: str, session: Dict) -> Dict:
        """Fallback response when Gemini is not available"""
        # Use the existing knowledge base approach
        from backend.ai_assistant import AIAssistant
        fallback_ai = AIAssistant()
        
        # Get context from session
        context = session.get('context', {})
        
        # Process with fallback AI
        fallback_response = fallback_ai.process_query(query, context)
        
        # Format response
        response_text = fallback_response.get('message', 'I can help you with agricultural queries.')
        
        if fallback_response.get('suggestions'):
            response_text += "\n\nSuggestions:\n"
            for suggestion in fallback_response['suggestions']:
                response_text += f"• {suggestion}\n"
        
        if fallback_response.get('treatment'):
            response_text += "\n\nTreatment:\n"
            for treatment in fallback_response['treatment']:
                response_text += f"• {treatment}\n"
        
        if fallback_response.get('prevention'):
            response_text += "\n\nPrevention:\n"
            for prevention in fallback_response['prevention']:
                response_text += f"• {prevention}\n"
        
        result = {
            'status': 'success',
            'response': response_text,
            'session_id': session.get('session_id', 'fallback'),
            'timestamp': datetime.now().isoformat(),
            'context_used': bool(context),
            'cached': False,
            'fallback_mode': True
        }
        
        # Update conversation history
        if 'conversation_history' in session:
            session['conversation_history'].append({
                'query': query,
                'response': response_text,
                'timestamp': datetime.now().isoformat()
            })
        
        return result
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get a summary of the session conversation"""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found or expired'}
        
        summary = {
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'total_interactions': len(session['conversation_history']),
            'location': session.get('location'),
            'has_soil_data': bool(session.get('context', {}).get('soil')),
            'has_weather_data': bool(session.get('context', {}).get('weather')),
            'recommended_crops': [rec.get('crop') for rec in session.get('crop_recommendations', [])],
            'topics_discussed': []
        }
        
        # Extract topics discussed
        topics = set()
        for entry in session['conversation_history']:
            query = entry.get('query', '').lower()
            if 'fertilizer' in query:
                topics.add('Fertilizer Management')
            if 'disease' in query or 'pest' in query:
                topics.add('Pest & Disease Control')
            if 'irrigation' in query or 'water' in query:
                topics.add('Irrigation')
            if 'harvest' in query:
                topics.add('Harvesting')
            if 'market' in query or 'price' in query:
                topics.add('Market Information')
            if 'weather' in query:
                topics.add('Weather')
        
        summary['topics_discussed'] = list(topics)
        
        return summary
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_activity'] > timedelta(seconds=self.session_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# Example usage and testing
if __name__ == "__main__":
    print("🤖 Google Gemini Agricultural Assistant Test")
    print("=" * 60)
    
    # Initialize assistant
    assistant = GeminiAssistant()
    
    if not assistant.is_configured:
        print("⚠️ Gemini API key not configured!")
        print("Add GEMINI_API_KEY to your .env file")
        print("Get your free API key from: https://makersuite.google.com/app/apikey")
    else:
        print("✅ Gemini AI configured successfully")
    
    # Create a test session
    session_id = assistant.create_session("test_user")
    print(f"\n📝 Created session: {session_id}")
    
    # Add some context
    test_context = {
        'location': 'Ranchi, Jharkhand',
        'soil_data': {
            'ph': 6.5,
            'nitrogen': 280,
            'phosphorus': 45,
            'potassium': 180,
            'organic_carbon': 0.8,
            'texture': 'Loamy'
        },
        'weather_data': {
            'temperature': 28,
            'humidity': 65,
            'rainfall': 120,
            'season': 'Kharif'
        },
        'crop_recommendations': [
            {'crop': 'Rice', 'confidence': 0.92},
            {'crop': 'Maize', 'confidence': 0.85},
            {'crop': 'Soybean', 'confidence': 0.78}
        ]
    }
    
    assistant.update_session_context(session_id, test_context)
    print("📊 Added context data to session")
    
    # Test queries
    test_queries = [
        "How much fertilizer should I use for rice cultivation?",
        "What's the best time to apply urea?",
        "My rice leaves are turning yellow, what should I do?",
        "How to manage water for rice in current weather?"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Agricultural Queries:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        response = assistant.process_query(query, session_id)
        
        if response['status'] == 'success':
            print(f"✅ Response received")
            print(f"📝 Answer: {response['response'][:200]}...")
            if response.get('structured_advice'):
                print(f"📋 Structured Advice: {response['structured_advice']}")
            print(f"🔄 Context Used: {response['context_used']}")
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
    
    # Get session summary
    print("\n" + "=" * 60)
    print("Session Summary:")
    print("=" * 60)
    summary = assistant.get_session_summary(session_id)
    print(json.dumps(summary, indent=2, default=str))
    
    print("\n✅ Test completed!")
