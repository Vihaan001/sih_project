"""
Quick Start Script - Download Data and Train Models
Run this script to automatically set up the system with real data
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'ml_models/trained', 'logs']
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Created necessary directories")

def download_kaggle_dataset():
    """Download the popular Kaggle crop recommendation dataset"""
    dataset_url = "https://raw.githubusercontent.com/atharvaingle/Crop-Recommendation-Dataset/master/Crop_recommendation.csv"
    dataset_path = "data/Crop_recommendation.csv"
    
    if os.path.exists(dataset_path):
        print("✅ Dataset already exists")
        return True
    
    print("📥 Downloading Kaggle crop recommendation dataset...")
    try:
        response = requests.get(dataset_url, timeout=30)
        if response.status_code == 200:
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            print("✅ Dataset downloaded successfully")
            return True
        else:
            print(f"❌ Failed to download dataset: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("📝 Manual download: https://www.kaggle.com/atharvaingle/crop-recommendation-dataset")
        return False

def create_sample_indian_dataset():
    """Create a sample Indian agricultural dataset"""
    print("🇮🇳 Creating sample Indian agricultural dataset...")
    
    # Sample data for Indian conditions
    indian_crops = [
        'rice', 'wheat', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
        'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
        'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
        'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
    ]
    
    # Generate sample data based on Indian agricultural conditions
    import random
    import numpy as np
    
    np.random.seed(42)
    data = []
    
    for _ in range(1000):
        crop = random.choice(indian_crops)
        
        # Indian soil and climate conditions
        if crop in ['rice', 'jute']:
            # High water requirement crops
            N = np.random.normal(100, 20)
            P = np.random.normal(40, 10)
            K = np.random.normal(40, 10)
            temperature = np.random.normal(25, 3)
            humidity = np.random.normal(80, 10)
            ph = np.random.normal(6.5, 0.5)
            rainfall = np.random.normal(150, 30)
        elif crop in ['wheat', 'chickpea']:
            # Rabi crops
            N = np.random.normal(120, 25)
            P = np.random.normal(60, 15)
            K = np.random.normal(50, 15)
            temperature = np.random.normal(20, 4)
            humidity = np.random.normal(65, 15)
            ph = np.random.normal(7.0, 0.5)
            rainfall = np.random.normal(80, 20)
        elif crop == 'cotton':
            # Cash crop
            N = np.random.normal(140, 30)
            P = np.random.normal(70, 20)
            K = np.random.normal(80, 20)
            temperature = np.random.normal(28, 3)
            humidity = np.random.normal(70, 15)
            ph = np.random.normal(7.5, 0.5)
            rainfall = np.random.normal(100, 25)
        else:
            # Default values
            N = np.random.normal(90, 25)
            P = np.random.normal(50, 15)
            K = np.random.normal(60, 20)
            temperature = np.random.normal(24, 5)
            humidity = np.random.normal(70, 15)
            ph = np.random.normal(6.8, 0.6)
            rainfall = np.random.normal(100, 40)
        
        data.append({
            'N': max(0, N),
            'P': max(0, P),
            'K': max(0, K),
            'temperature': max(10, temperature),
            'humidity': max(20, min(100, humidity)),
            'ph': max(4, min(9, ph)),
            'rainfall': max(0, rainfall),
            'label': crop
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/indian_crops.csv', index=False)
    print(f"✅ Created Indian dataset with {len(df)} samples, {len(df['label'].unique())} crops")
    return True

def install_dependencies():
    """Install required packages if missing"""
    required_packages = [
        'scikit-learn', 'pandas', 'numpy', 'requests', 
        'python-dotenv', 'beautifulsoup4'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"📦 Installing {package}...")
            os.system(f"pip install {package}")

def train_models():
    """Train the ML models"""
    print("🤖 Training models with real data...")
    
    try:
        from ml_models.train_model import ModelTrainer
        trainer = ModelTrainer()
        
        # Try to load Kaggle dataset first
        kaggle_path = 'data/Crop_recommendation.csv'
        indian_path = 'data/indian_crops.csv'
        
        df = None
        if os.path.exists(kaggle_path):
            print("📊 Using Kaggle dataset")
            df = trainer.load_kaggle_dataset(kaggle_path)
        elif os.path.exists(indian_path):
            print("📊 Using Indian dataset")
            df = pd.read_csv(indian_path)
        
        if df is not None:
            # Prepare and train
            X, y = trainer.prepare_training_data(df)
            
            print("🎯 Training crop classifier...")
            trainer.train_crop_classifier(X, y, optimize_hyperparameters=False)
            
            print("📈 Training yield predictor...")
            trainer.train_yield_predictor()
            
            print("💾 Saving models...")
            trainer.save_models()
            
            # Test prediction
            print("\n🧪 Testing trained model...")
            test_input = {
                'N': 90, 'P': 42, 'K': 43,
                'temperature': 20.87, 'humidity': 82.0,
                'ph': 6.5, 'rainfall': 202.93
            }
            
            result = trainer.predict(test_input)
            print(f"Test input: {test_input}")
            print("Predictions:")
            for rec in result['recommendations']:
                print(f"  🌾 {rec['crop']}: {rec['confidence']:.2%} confidence")
            
            return True
        else:
            print("❌ No dataset available for training")
            return False
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_system():
    """Test the complete system"""
    print("\n🔧 Testing complete system...")
    
    try:
        # Test API connections
        print("Testing API connections...")
        exec(open('test_apis.py').read())
        
        return True
    except Exception as e:
        print(f"⚠️  Some tests failed: {e}")
        return False

def main():
    """Main setup process"""
    print("🚀 Quick Start - AI Crop Recommendation System")
    print("="*60)
    
    # Step 1: Setup
    print("\n📁 Step 1: Setting up directories...")
    create_directories()
    
    # Step 2: Dependencies
    print("\n📦 Step 2: Checking dependencies...")
    install_dependencies()
    
    # Step 3: Download data
    print("\n📊 Step 3: Getting training data...")
    kaggle_success = download_kaggle_dataset()
    if not kaggle_success:
        print("📝 Creating sample Indian dataset as fallback...")
        create_sample_indian_dataset()
    
    # Step 4: Train models
    print("\n🤖 Step 4: Training ML models...")
    training_success = train_models()
    
    # Step 5: Test system
    print("\n🔧 Step 5: Testing system...")
    test_success = test_system()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SETUP SUMMARY")
    print("="*60)
    
    if training_success:
        print("✅ Models trained and ready!")
        print("✅ System is ready to use")
        
        print("\n🎯 Next steps:")
        print("   1. Get API keys (optional):")
        print("      • OpenWeather: https://openweathermap.org/api")
        print("   2. Start the backend:")
        print("      cd backend && python main.py")
        print("   3. Open frontend:")
        print("      Open frontend/index.html in your browser")
        print("   4. Test the system with real coordinates")
        
    else:
        print("❌ Setup had some issues")
        print("💡 Try running components individually:")
        print("   • python test_apis.py")
        print("   • python ml_models/train_model.py")
    
    print("\n📚 For detailed guide, see: API_AND_TRAINING_GUIDE.md")

if __name__ == "__main__":
    main()
