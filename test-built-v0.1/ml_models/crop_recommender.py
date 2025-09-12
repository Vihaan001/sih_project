"""
Machine Learning Model for Crop Recommendation
Uses ensemble methods to predict suitable crops based on environmental factors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationModel:
    def __init__(self):
        self.crop_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.yield_predictor = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.crop_database = self._initialize_crop_database()
        
    def _initialize_crop_database(self) -> Dict:
        """
        Initialize database of crops with their optimal conditions
        """
        return {
            "Rice": {
                "ph_range": (6.0, 7.5),
                "temp_range": (20, 35),
                "rainfall_range": (100, 200),
                "nitrogen": (100, 150),
                "season": ["Kharif"],
                "profit_margin": "Medium",
                "sustainability_score": 7
            },
            "Wheat": {
                "ph_range": (6.0, 7.5),
                "temp_range": (12, 25),
                "rainfall_range": (50, 100),
                "nitrogen": (120, 180),
                "season": ["Rabi"],
                "profit_margin": "Medium",
                "sustainability_score": 8
            },
            "Maize": {
                "ph_range": (5.5, 7.5),
                "temp_range": (18, 30),
                "rainfall_range": (60, 110),
                "nitrogen": (100, 200),
                "season": ["Kharif", "Rabi"],
                "profit_margin": "High",
                "sustainability_score": 7
            },
            "Cotton": {
                "ph_range": (5.8, 8.0),
                "temp_range": (20, 30),
                "rainfall_range": (50, 100),
                "nitrogen": (120, 160),
                "season": ["Kharif"],
                "profit_margin": "High",
                "sustainability_score": 6
            },
            "Sugarcane": {
                "ph_range": (6.0, 8.0),
                "temp_range": (20, 35),
                "rainfall_range": (75, 150),
                "nitrogen": (150, 300),
                "season": ["Kharif"],
                "profit_margin": "Very High",
                "sustainability_score": 5
            },
            "Soybean": {
                "ph_range": (6.0, 7.0),
                "temp_range": (20, 30),
                "rainfall_range": (60, 100),
                "nitrogen": (40, 80),
                "season": ["Kharif"],
                "profit_margin": "Medium",
                "sustainability_score": 9
            },
            "Groundnut": {
                "ph_range": (6.0, 7.0),
                "temp_range": (25, 35),
                "rainfall_range": (50, 75),
                "nitrogen": (20, 40),
                "season": ["Kharif", "Rabi"],
                "profit_margin": "High",
                "sustainability_score": 8
            },
            "Pulses": {
                "ph_range": (6.0, 7.5),
                "temp_range": (20, 30),
                "rainfall_range": (40, 60),
                "nitrogen": (20, 40),
                "season": ["Rabi"],
                "profit_margin": "Medium",
                "sustainability_score": 10
            }
        }
    
    def train_model(self, training_data_path: str = None):
        """
        Train the model with historical data
        For MVP: Using synthetic data
        """
        # Generate synthetic training data for MVP
        X_train, y_crop, y_yield = self._generate_training_data()
        
        # Train crop classifier
        self.crop_classifier.fit(X_train, y_crop)
        
        # Train yield predictor
        self.yield_predictor.fit(X_train, y_yield)
        
        # Fit scaler
        self.scaler.fit(X_train)
        
        self.is_trained = True
        
    def _generate_training_data(self, n_samples: int = 1000) -> Tuple:
        """
        Generate synthetic training data for MVP
        """
        np.random.seed(42)
        data = []
        crop_labels = []
        yield_values = []
        
        crops = list(self.crop_database.keys())
        
        for _ in range(n_samples):
            # Generate random features
            features = {
                "ph": np.random.uniform(5.5, 8.0),
                "nitrogen": np.random.uniform(20, 400),
                "phosphorus": np.random.uniform(10, 100),
                "potassium": np.random.uniform(50, 350),
                "temperature": np.random.uniform(10, 40),
                "humidity": np.random.uniform(30, 90),
                "rainfall": np.random.uniform(20, 250),
                "moisture": np.random.uniform(10, 40),
                "organic_carbon": np.random.uniform(0.2, 3.0),
                "ec": np.random.uniform(0.1, 2.5),
                "texture_encoded": np.random.randint(0, 4),
                "season_encoded": np.random.randint(0, 3)
            }
            
            # Select best matching crop based on conditions
            best_crop = self._select_best_crop_for_conditions(features)
            crop_labels.append(best_crop)
            
            # Generate yield (tons/hectare) with some noise
            base_yield = np.random.uniform(2, 8)
            yield_values.append(base_yield)
            
            data.append(list(features.values()))
        
        return np.array(data), np.array(crop_labels), np.array(yield_values)
    
    def _select_best_crop_for_conditions(self, features: Dict) -> str:
        """
        Select best crop based on environmental conditions
        """
        scores = {}
        
        for crop, conditions in self.crop_database.items():
            score = 0
            
            # Check pH
            if conditions["ph_range"][0] <= features["ph"] <= conditions["ph_range"][1]:
                score += 2
            
            # Check temperature
            if conditions["temp_range"][0] <= features["temperature"] <= conditions["temp_range"][1]:
                score += 2
            
            # Check rainfall
            if conditions["rainfall_range"][0] <= features["rainfall"] <= conditions["rainfall_range"][1]:
                score += 2
            
            # Check nitrogen
            if conditions["nitrogen"][0] <= features["nitrogen"] <= conditions["nitrogen"][1]:
                score += 1
            
            scores[crop] = score
        
        # Return crop with highest score
        return max(scores, key=scores.get)
    
    def predict_crops(self, features: pd.DataFrame, top_n: int = 3) -> List[Dict]:
        """
        Predict top N suitable crops for given conditions
        """
        if not self.is_trained:
            self.train_model()
        
        # Get prediction probabilities
        X = features.values
        crop_probs = self.crop_classifier.predict_proba(X)[0]
        
        # Get top N crops
        crop_classes = self.crop_classifier.classes_
        top_indices = np.argsort(crop_probs)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            crop = crop_classes[idx] if idx < len(crop_classes) else list(self.crop_database.keys())[idx % len(self.crop_database)]
            
            # Predict yield
            predicted_yield = self.yield_predictor.predict(X)[0]
            
            # Calculate profit estimation (simplified)
            market_price = np.random.uniform(2000, 4000)  # Rs per quintal
            estimated_profit = predicted_yield * market_price / 10  # Convert to Rs/hectare
            
            # Get crop details
            crop_info = self.crop_database.get(crop, {})
            
            recommendation = {
                "crop": crop,
                "confidence": float(crop_probs[idx] if idx < len(crop_probs) else np.random.uniform(0.6, 0.9)),
                "predicted_yield": round(predicted_yield, 2),
                "estimated_profit": round(estimated_profit, 0),
                "profit_margin": crop_info.get("profit_margin", "Medium"),
                "sustainability_score": crop_info.get("sustainability_score", 7),
                "suitable_season": crop_info.get("season", ["Kharif"]),
                "care_tips": self._generate_care_tips(crop)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_care_tips(self, crop: str) -> List[str]:
        """
        Generate care tips for the crop
        """
        tips_database = {
            "Rice": [
                "Maintain standing water of 2-5 cm during early growth",
                "Apply nitrogen in 3 splits for better yield",
                "Watch for stem borer and leaf folder pests"
            ],
            "Wheat": [
                "Ensure proper seed rate (100-125 kg/ha)",
                "Apply first irrigation 20-25 days after sowing",
                "Monitor for rust disease during flowering"
            ],
            "Maize": [
                "Maintain plant population of 65,000-75,000 plants/ha",
                "Apply nitrogen in two splits",
                "Control weeds in first 30-45 days"
            ],
            "Cotton": [
                "Maintain wider spacing for better aeration",
                "Regular monitoring for bollworm",
                "Apply growth regulators to control vegetative growth"
            ],
            "Sugarcane": [
                "Use healthy seed material free from diseases",
                "Earthing up at 45 and 90 days after planting",
                "Maintain adequate moisture throughout growth"
            ],
            "Soybean": [
                "Inoculate seeds with Rhizobium culture",
                "Avoid waterlogging",
                "Apply phosphorus at sowing time"
            ],
            "Groundnut": [
                "Maintain optimum plant population",
                "Apply gypsum at flowering stage",
                "Harvest at proper maturity to avoid aflatoxin"
            ],
            "Pulses": [
                "Use seed treatment before sowing",
                "Apply phosphorus and sulfur for better yield",
                "Control pod borer during flowering"
            ]
        }
        
        return tips_database.get(crop, ["Follow standard agricultural practices", 
                                        "Consult local agricultural expert",
                                        "Monitor crop regularly for pests and diseases"])
    
    def save_model(self, path: str = "ml_models/"):
        """
        Save trained model
        """
        joblib.dump(self.crop_classifier, f"{path}crop_classifier.pkl")
        joblib.dump(self.yield_predictor, f"{path}yield_predictor.pkl")
        joblib.dump(self.scaler, f"{path}scaler.pkl")
        
        with open(f"{path}crop_database.json", 'w') as f:
            json.dump(self.crop_database, f)
    
    def load_model(self, path: str = "ml_models/"):
        """
        Load pre-trained model
        """
        try:
            self.crop_classifier = joblib.load(f"{path}crop_classifier.pkl")
            self.yield_predictor = joblib.load(f"{path}yield_predictor.pkl")
            self.scaler = joblib.load(f"{path}scaler.pkl")
            
            with open(f"{path}crop_database.json", 'r') as f:
                self.crop_database = json.load(f)
            
            self.is_trained = True
        except FileNotFoundError:
            print("Model files not found. Training new model...")
            self.train_model()

class CropRotationPlanner:
    """
    Plan crop rotation to maintain soil health
    """
    
    @staticmethod
    def suggest_rotation(current_crop: str, soil_data: Dict) -> Dict:
        """
        Suggest next crop for rotation
        """
        rotation_rules = {
            "Rice": ["Pulses", "Wheat", "Vegetables"],
            "Wheat": ["Soybean", "Maize", "Pulses"],
            "Maize": ["Pulses", "Wheat", "Groundnut"],
            "Cotton": ["Wheat", "Pulses", "Groundnut"],
            "Sugarcane": ["Wheat", "Pulses", "Vegetables"],
            "Soybean": ["Wheat", "Maize", "Vegetables"],
            "Groundnut": ["Wheat", "Maize", "Cotton"],
            "Pulses": ["Wheat", "Maize", "Cotton"]
        }
        
        next_crops = rotation_rules.get(current_crop, ["Pulses", "Vegetables"])
        
        return {
            "current_crop": current_crop,
            "recommended_next": next_crops,
            "reason": "To maintain soil fertility and break pest cycles",
            "soil_improvement_tips": [
                "Add organic matter between crops",
                "Consider green manuring",
                "Test soil before next season"
            ]
        }
