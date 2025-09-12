# 📚 Complete Guide: Real Data Integration & Model Training

## Table of Contents
1. [API Integration Guide](#api-integration-guide)
2. [Dataset Sources](#dataset-sources)
3. [Model Training Guide](#model-training-guide)
4. [Production Deployment](#production-deployment)

---

## 🔌 API Integration Guide

### 1. Free APIs for Agricultural Data

#### A. OpenWeatherMap (Weather Data)
- **URL**: https://openweathermap.org/api
- **Free Tier**: 1,000 calls/day
- **Features**: Current weather, 5-day forecast
- **Setup**:
  ```bash
  1. Sign up at https://openweathermap.org/users/sign_up
  2. Get API key from account dashboard
  3. Add to .env: OPENWEATHER_API_KEY=your_key
  ```

#### B. SoilGrids (Global Soil Data)
- **URL**: https://www.isric.org/explore/soilgrids
- **Free Tier**: Unlimited (no key required)
- **Features**: Soil properties at 250m resolution globally
- **Usage**:
  ```python
  from backend.real_data_collector import RealDataCollector
  collector = RealDataCollector()
  soil_data = collector.get_soil_data_soilgrids(23.3441, 85.3096)
  ```

#### C. NASA POWER (Historical Weather)
- **URL**: https://power.larc.nasa.gov/
- **Free Tier**: Unlimited
- **Features**: 40+ years of weather data
- **Best for**: Training data preparation

#### D. Bhuvan (Indian Specific)
- **URL**: https://bhuvan.nrsc.gov.in/
- **Registration Required**: Yes
- **Features**: Indian soil, land use, crop data
- **Setup**:
  ```bash
  1. Register at https://bhuvan.nrsc.gov.in/
  2. Apply for API access
  3. Add to .env: BHUVAN_API_KEY=your_key
  ```

#### E. Data.gov.in (Indian Government Data)
- **URL**: https://data.gov.in/
- **Free Tier**: 100 calls/day without key, more with registration
- **Features**: Crop yields, production statistics
- **Datasets Available**:
  - District-wise crop production
  - Rainfall data
  - Soil health cards

### 2. Setting Up API Keys

```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Edit .env and add your API keys
# For Windows:
notepad .env

# For Linux/Mac:
nano .env
```

### 3. Testing API Connections

```python
# Test script to verify APIs are working
python -c "
from backend.real_data_collector import RealDataCollector
collector = RealDataCollector()

# Test OpenWeather
weather = collector.get_weather_data_openweather(23.3441, 85.3096)
print('Weather API:', 'OK' if weather else 'Failed')

# Test SoilGrids (no key needed)
soil = collector.get_soil_data_soilgrids(23.3441, 85.3096)
print('SoilGrids API:', 'OK' if soil else 'Failed')
"
```

---

## 📊 Dataset Sources

### 1. Kaggle Datasets

#### A. Crop Recommendation Dataset (Most Popular)
- **URL**: https://www.kaggle.com/atharvaingle/crop-recommendation-dataset
- **Size**: 2200 samples, 22 crops
- **Features**: N, P, K, temperature, humidity, pH, rainfall
- **Download**:
  ```bash
  # Option 1: Manual download
  # Go to the URL and download Crop_recommendation.csv
  # Place in data/ folder
  
  # Option 2: Using Kaggle API
  pip install kaggle
  kaggle datasets download -d atharvaingle/crop-recommendation-dataset
  unzip crop-recommendation-dataset.zip -d data/
  ```

#### B. Indian Agriculture Crop Production
- **URL**: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
- **Size**: 246,000+ records
- **Features**: State, district, year, season, crop, area, production

### 2. Government Datasets

#### A. Agmarknet (Market Prices)
- **URL**: https://agmarknet.gov.in/
- **Access**: Web scraping or bulk download
- **Data**: Daily commodity prices from Indian markets

#### B. India Meteorological Department
- **URL**: https://www.imd.gov.in/
- **Data**: District-wise weather data
- **Access**: Some data freely available, APIs require permission

### 3. Building Your Own Dataset

```python
from backend.real_data_collector import DatasetBuilder

# Create dataset builder
builder = DatasetBuilder()

# Define locations (lat, lon)
locations = [
    (23.3441, 85.3096),  # Ranchi
    (23.7957, 86.4304),  # Dhanbad
    (22.8046, 86.2029),  # Jamshedpur
    # Add more locations
]

# Build dataset
df = builder.build_training_dataset(locations, 'data/training_data.csv')
print(f"Dataset created with {len(df)} samples")
```

---

## 🤖 Model Training Guide

### 1. Quick Start with Kaggle Dataset

```bash
# Step 1: Download the dataset
# Download from: https://www.kaggle.com/atharvaingle/crop-recommendation-dataset
# Save as: data/Crop_recommendation.csv

# Step 2: Train the model
cd ml_models
python train_model.py

# This will:
# - Load the dataset
# - Train Random Forest classifier
# - Train yield predictor
# - Save models to ml_models/trained/
```

### 2. Training with Custom Data

```python
from ml_models.train_model import ModelTrainer
import pandas as pd

# Initialize trainer
trainer = ModelTrainer()

# Load your dataset
df = pd.read_csv('path/to/your/dataset.csv')

# Prepare data (ensure columns: N, P, K, temperature, humidity, ph, rainfall, label)
X, y = trainer.prepare_training_data(df)

# Train classifier
trainer.train_crop_classifier(X, y, optimize_hyperparameters=True)

# Train yield predictor (optional)
trainer.train_yield_predictor()

# Save models
trainer.save_models('ml_models/trained/')

# Test prediction
test_input = {
    'N': 90, 'P': 42, 'K': 43,
    'temperature': 20.87, 'humidity': 82.0,
    'ph': 6.5, 'rainfall': 202.93
}
result = trainer.predict(test_input)
print(result)
```

### 3. Advanced Training with Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Train with optimization
trainer.train_crop_classifier(X, y, optimize_hyperparameters=True)
```

### 4. Model Evaluation Metrics

After training, you'll see:
- **Accuracy**: Overall prediction accuracy (aim for >90%)
- **Cross-validation score**: Model generalization (should be close to accuracy)
- **Classification Report**: Per-crop precision, recall, F1-score
- **Feature Importance**: Which features matter most

Example output:
```
Test Accuracy: 0.9932
Cross-validation scores: [0.991 0.993 0.995 0.991 0.994]
Average CV score: 0.9928 (+/- 0.0031)

Feature Importance:
   feature    importance
0  potassium   0.284
1  nitrogen    0.231
2  phosphorus  0.198
3  humidity    0.142
4  ph          0.089
```

---

## 🚀 Production Deployment

### 1. Using Trained Models in Production

```python
# In your backend/main.py, update to use trained models:

from ml_models.train_model import ModelTrainer

# Load trained models at startup
trainer = ModelTrainer()
trainer.load_models('ml_models/trained/')

# Use in API endpoint
@app.post("/recommend")
async def get_recommendations(request):
    # ... get features ...
    result = trainer.predict(features)
    return result
```

### 2. Scheduled Data Updates

Create a cron job or scheduled task:

```python
# scripts/update_data.py
from backend.real_data_collector import DatasetBuilder
from ml_models.train_model import ModelTrainer
import schedule
import time

def update_and_retrain():
    # Fetch new data
    builder = DatasetBuilder()
    new_data = builder.build_training_dataset(locations)
    
    # Retrain model
    trainer = ModelTrainer()
    df = pd.read_csv('data/combined_dataset.csv')
    X, y = trainer.prepare_training_data(df)
    trainer.train_crop_classifier(X, y)
    trainer.save_models()
    
    print(f"Models updated at {datetime.now()}")

# Schedule weekly updates
schedule.every().monday.at("02:00").do(update_and_retrain)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

### 3. API Rate Limiting & Caching

```python
from functools import lru_cache
import redis

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=128)
def get_cached_soil_data(lat: float, lon: float):
    # Check Redis cache first
    cache_key = f"soil:{lat}:{lon}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # Fetch from API
    data = collector.get_soil_data_soilgrids(lat, lon)
    
    # Cache for 24 hours
    redis_client.setex(cache_key, 86400, json.dumps(data))
    
    return data
```

### 4. Error Handling & Fallbacks

```python
def get_recommendation_with_fallback(location):
    try:
        # Try real APIs
        soil = collector.get_soil_data_soilgrids(location.lat, location.lon)
        weather = collector.get_weather_data_openweather(location.lat, location.lon)
    except Exception as e:
        print(f"API error: {e}")
        # Use fallback data
        soil = collector._get_fallback_soil_data()
        weather = collector._get_fallback_weather_data()
    
    # Continue with recommendation
    return make_recommendation(soil, weather)
```

---

## 📈 Performance Optimization

### 1. Batch Processing

```python
# Process multiple locations at once
def batch_process_locations(locations):
    results = []
    
    # Process in batches of 10
    for i in range(0, len(locations), 10):
        batch = locations[i:i+10]
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_location, loc) for loc in batch]
            results.extend([f.result() for f in futures])
        
        # Rate limiting
        time.sleep(1)
    
    return results
```

### 2. Model Optimization

```python
# Reduce model size for deployment
from sklearn.tree import DecisionTreeClassifier
import pickle

# Train smaller model for edge devices
simple_model = DecisionTreeClassifier(max_depth=10)
simple_model.fit(X_train, y_train)

# Compress model
with open('model_compressed.pkl', 'wb') as f:
    pickle.dump(simple_model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Model size comparison
import os
print(f"Full model: {os.path.getsize('crop_classifier.pkl') / 1024:.2f} KB")
print(f"Compressed: {os.path.getsize('model_compressed.pkl') / 1024:.2f} KB")
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **API Key Errors**
   ```
   Error: "API key not valid"
   Solution: Check .env file, ensure no extra spaces
   ```

2. **Dataset Not Found**
   ```
   Error: "FileNotFoundError: Crop_recommendation.csv"
   Solution: Download from Kaggle, place in data/ folder
   ```

3. **Low Model Accuracy**
   ```
   Issue: Accuracy < 80%
   Solutions:
   - Check data quality
   - Increase training samples
   - Try hyperparameter tuning
   - Check for class imbalance
   ```

4. **API Rate Limits**
   ```
   Error: "429 Too Many Requests"
   Solution: Implement caching and rate limiting
   ```

---

## 📞 Support & Resources

- **Kaggle Datasets**: https://www.kaggle.com/datasets?search=crop
- **SoilGrids Documentation**: https://www.isric.org/explore/soilgrids/api
- **OpenWeather Guide**: https://openweathermap.org/guide
- **NASA POWER**: https://power.larc.nasa.gov/docs/
- **Indian Agricultural Data**: https://data.gov.in/sector/agriculture

---

## 🎯 Next Steps

1. **Get API Keys**: Start with OpenWeatherMap (free & quick)
2. **Download Dataset**: Get the Kaggle crop recommendation dataset
3. **Train Model**: Run `python ml_models/train_model.py`
4. **Test API**: Verify everything works with real data
5. **Deploy**: Move to production with proper monitoring

Remember: Start with free APIs and public datasets, then gradually add paid services as your application scales!
