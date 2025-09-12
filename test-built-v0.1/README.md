# AI-Based Crop Recommendation System for Farmers

## 📋 Project Overview

An AI-powered decision support system that provides personalized crop recommendations to farmers based on real-time soil properties, weather data, and market conditions. The system features multilingual support, offline capabilities, and an intuitive chat interface for agricultural queries.

## 🌟 Key Features

### Core Functionality
- **Smart Crop Recommendations**: ML-based recommendations considering soil, weather, and market data
- **Multilingual Support**: Interface available in English and Hindi (expandable to other languages)
- **AI Chat Assistant**: Natural language processing for agricultural queries
- **Offline Mode**: Works in low-connectivity areas with cached data
- **Market Integration**: Real-time market prices and demand analysis
- **Crop Rotation Planning**: Maintains soil health through rotation suggestions
- **Disease Detection**: Image analysis for crop disease identification (MVP ready)

### Technical Features
- RESTful API with FastAPI backend
- Machine Learning models using scikit-learn
- Responsive web interface (mobile-friendly)
- Data caching for offline functionality
- Mock data generation for testing

## 🏗️ Project Structure

```
crop-recommendation-system/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── data_collector.py       # Data collection module
│   ├── ai_assistant.py         # NLP chat assistant
│   └── translator.py           # Multilingual support
├── ml_models/
│   └── crop_recommender.py     # ML models for crop recommendation
├── frontend/
│   └── index.html              # Web interface
├── data/                       # Data storage directory
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd crop-recommendation-system
```

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Application

1. **Start the backend server:**
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`

2. **Open the frontend:**
- Open `frontend/index.html` in your web browser
- Or serve it using Python:
```bash
cd frontend
python -m http.server 8080
```
Then navigate to `http://localhost:8080`

## 📱 Using the System

### Getting Crop Recommendations

1. **Location-Based Recommendations:**
   - Enter your coordinates (latitude/longitude)
   - Specify your region
   - Select preferred language
   - Click "Get Recommendations"

2. **Manual Input:**
   - Switch to "Manual Input" tab
   - Enter soil parameters (pH, NPK values)
   - Provide expected rainfall
   - Submit for recommendations

### Using the Chat Assistant

Ask questions in natural language such as:
- "Which crop should I grow this season?"
- "How to treat wheat rust?"
- "What is the current market price of rice?"
- "When should I sow maize?"
- "How much fertilizer for cotton?"

### Offline Mode

The system automatically switches to offline mode when internet is unavailable:
- Cached crop database remains accessible
- Basic recommendations work offline
- Chat provides limited offline responses
- Data syncs when connection resumes

## 🔧 API Endpoints

### Main Endpoints

- `GET /` - API information
- `POST /recommend` - Get crop recommendations
- `POST /chat` - Chat with AI assistant
- `GET /soil-data/{lat}/{lng}` - Get soil data
- `GET /weather/{lat}/{lng}` - Get weather data
- `GET /market-prices` - Get market prices
- `POST /crop-rotation` - Get rotation suggestions
- `POST /upload-image` - Analyze crop images
- `GET /offline-data` - Get offline data package

### Example API Request

```python
import requests

# Get crop recommendations
response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "location": {
            "latitude": 23.3441,
            "longitude": 85.3096,
            "region": "Jharkhand"
        },
        "language": "en"
    }
)

recommendations = response.json()
```

## 🧪 Testing the MVP

### Test Scenarios

1. **Basic Recommendation Test:**
   - Use default Jharkhand coordinates
   - System returns top 3 crop recommendations
   - Verify confidence scores and yield predictions

2. **Multilingual Test:**
   - Switch to Hindi language
   - Check crop names and basic terms are translated
   - Verify chat responses in selected language

3. **Offline Mode Test:**
   - Stop the backend server
   - Frontend should show "Offline Mode"
   - Basic features should still work

4. **Chat Assistant Test:**
   - Ask about specific crops
   - Query about diseases
   - Request market information

## 🔄 Data Flow

1. **User Input** → Location/Soil Data
2. **Data Collection** → Fetch soil, weather, market data
3. **ML Processing** → Feature extraction and prediction
4. **Recommendation Engine** → Top crops with confidence scores
5. **Response** → JSON with recommendations and metadata

## 🌐 Production Deployment Considerations

### API Integrations Needed
- **Soil Data**: SoilGrids API / Bhuvan (ISRO)
- **Weather**: OpenWeatherMap / IMD API
- **Market Prices**: Agmarknet API
- **Translation**: Google Translate API
- **AI Chat**: OpenAI API / Local LLM

### Mobile App Development
- React Native or Flutter implementation
- Voice input integration
- Push notifications for weather/price alerts
- GPS integration for automatic location

### Scalability Improvements
- Database implementation (PostgreSQL/MongoDB)
- Redis for caching
- Message queue for async processing
- Load balancing for multiple instances
- CDN for static content

### Security Enhancements
- API authentication (JWT tokens)
- Rate limiting
- Data encryption
- Secure file upload handling
- Input validation and sanitization

## 📊 Model Performance (MVP)

- **Accuracy**: ~75% (with synthetic data)
- **Response Time**: <2 seconds
- **Supported Crops**: 8 major crops
- **Languages**: English, Hindi
- **Offline Data Size**: ~5MB

## 🤝 Contributing

This is an MVP implementation. For production deployment:

1. Replace mock data with real API integrations
2. Train models with actual agricultural data
3. Implement comprehensive error handling
4. Add user authentication and profiles
5. Enhance multilingual support
6. Develop native mobile applications

## 📝 License

This project is developed as an MVP for the Government of Jharkhand's agricultural initiative.

## 🙏 Acknowledgments

- Department of Higher and Technical Education, Government of Jharkhand
- Agricultural experts for domain knowledge
- Open-source community for tools and libraries

## 📞 Support

For issues or questions:
- Check the API documentation at `http://localhost:8000/docs`
- Review error logs in the console
- Ensure all dependencies are installed correctly

---

**Note**: This is an MVP (Minimum Viable Product) implementation using mock data for demonstration purposes. Production deployment requires integration with actual data sources and APIs.
