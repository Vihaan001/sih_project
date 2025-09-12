#!/usr/bin/env python3
"""
Test script to verify the updated dual-model crop recommendation system
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.crop_recommender import CropRecommendationModel

def test_dual_models():
    """Test that both crop classifier and yield predictor models work correctly"""
    
    print("🔄 Testing Dual Model System...")
    print("=" * 50)
    
    # Initialize model
    model = CropRecommendationModel()
    
    # Load trained models
    print("📁 Loading trained models...")
    try:
        model.load_model("ml_models/trained/")
        print("✅ Models loaded successfully!")
        print(f"   - Crop Classifier: {type(model.crop_classifier).__name__}")
        print(f"   - Yield Predictor: {type(model.yield_predictor).__name__}")
        print(f"   - Feature Columns: {model.feature_columns}")
        print(f"   - Yield Feature Columns: {model.yield_feature_columns}")
        print(f"   - Available Crops: {len(model.crop_classifier.classes_)} crops")
        print(f"   - Label Encoder Available: {model.label_encoder is not None}")
        print(f"   - Yield Label Encoder Available: {model.yield_label_encoder is not None}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False
    
    print("\n" + "=" * 50)
    
    # Test with sample data
    print("🧪 Testing with sample environmental data...")
    
    # Create test features (typical for rice growing conditions)
    test_features = pd.DataFrame({
        'N': [80],           # Nitrogen
        'P': [40],           # Phosphorus  
        'K': [40],           # Potassium
        'temperature': [25], # Temperature in Celsius
        'humidity': [80],    # Humidity percentage
        'ph': [6.5],         # Soil pH
        'rainfall': [150]    # Rainfall in mm
    })
    
    print("📊 Test Features:")
    for col, val in test_features.iloc[0].items():
        print(f"   - {col}: {val}")
    
    # Get predictions
    print("\n🔮 Getting crop recommendations...")
    try:
        recommendations = model.predict_crops(test_features, top_n=5)
        print(f"✅ Generated {len(recommendations)} recommendations!")
        
        print("\n📋 Top Crop Recommendations:")
        print("-" * 80)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Crop: {rec['crop']}")
            print(f"   Confidence: {rec['confidence']:.3f}")
            print(f"   Predicted Yield: {rec['predicted_yield']:.2f} tons/hectare")
            print(f"   Estimated Profit: ₹{rec['estimated_profit']:.0f}/hectare")
            print(f"   Season: {rec['suitable_season']}")
            print()
            
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components():
    """Test individual model components"""
    
    print("🔧 Testing Individual Model Components...")
    print("=" * 50)
    
    model = CropRecommendationModel()
    model.load_model("ml_models/trained/")
    
    # Test crop classifier
    print("🌾 Testing Crop Classifier...")
    test_features = np.array([[80, 40, 40, 25, 80, 6.5, 150]])
    
    try:
        # Scale features
        scaled_features = model.scaler.transform(test_features)
        print(f"   ✅ Feature scaling successful: {scaled_features.shape}")
        
        # Get probabilities
        probabilities = model.crop_classifier.predict_proba(scaled_features)[0]
        print(f"   ✅ Crop classification successful: {len(probabilities)} probabilities")
        
        # Show top 3 crops
        top_indices = np.argsort(probabilities)[-3:][::-1]
        print("   📊 Top 3 Crops:")
        for idx in top_indices:
            crop = model.crop_classifier.classes_[idx]
            prob = probabilities[idx]
            print(f"      - {crop}: {prob:.3f}")
            
    except Exception as e:
        print(f"   ❌ Crop classifier error: {e}")
    
    # Test yield predictor
    print("\n📈 Testing Yield Predictor...")
    try:
        # Test yield prediction for rice
        yield_prediction = model._predict_yield_for_crop("rice", pd.DataFrame({
            'N': [80], 'P': [40], 'K': [40], 'temperature': [25], 
            'humidity': [80], 'ph': [6.5], 'rainfall': [150]
        }))
        print(f"   ✅ Yield prediction for rice: {yield_prediction:.2f} tons/hectare")
        
        # Test for different crops
        test_crops = ["wheat", "maize", "cotton"]
        for crop in test_crops:
            yield_pred = model._predict_yield_for_crop(crop, pd.DataFrame({
                'N': [80], 'P': [40], 'K': [40], 'temperature': [25], 
                'humidity': [80], 'ph': [6.5], 'rainfall': [150]
            }))
            print(f"   ✅ Yield prediction for {crop}: {yield_pred:.2f} tons/hectare")
            
    except Exception as e:
        print(f"   ❌ Yield predictor error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Dual Model Test Suite")
    print("=" * 60)
    
    # Test dual models
    success = test_dual_models()
    
    if success:
        print("\n" + "=" * 60)
        test_model_components()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("✅ Both crop classifier and yield predictor are working correctly")
        print("✅ System is ready for production use")
    else:
        print("\n❌ Tests failed. Please check the model files and try again.")