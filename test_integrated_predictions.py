"""
Test the integrated crop recommendation and yield prediction system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.train_model import ModelTrainer

def test_predictions():
    # Load the trained models
    trainer = ModelTrainer()
    trainer.load_models()
    
    print("="*60)
    print("INTEGRATED CROP RECOMMENDATION & YIELD PREDICTION TEST")
    print("="*60)
    
    # Test different scenarios
    test_scenarios = [
        {
            'name': 'High Rainfall Region',
            'input': {
                'N': 90, 'P': 42, 'K': 43, 
                'temperature': 25.0, 'humidity': 80.0, 
                'ph': 6.5, 'rainfall': 300.0
            }
        },
        {
            'name': 'Dry Region',
            'input': {
                'N': 80, 'P': 35, 'K': 40, 
                'temperature': 30.0, 'humidity': 60.0, 
                'ph': 7.0, 'rainfall': 50.0
            }
        },
        {
            'name': 'Cold Region',
            'input': {
                'N': 70, 'P': 50, 'K': 45, 
                'temperature': 15.0, 'humidity': 70.0, 
                'ph': 6.0, 'rainfall': 150.0
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 40)
        print(f"Input conditions: {scenario['input']}")
        
        result = trainer.predict(scenario['input'])
        print("Recommendations:")
        
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec['crop']}: {rec['confidence']:.1%} confidence")
            print(f"     Expected Yield: {rec['predicted_yield']:.2f} tons/ha")

if __name__ == "__main__":
    test_predictions()