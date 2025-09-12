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
        self.yield_scaler = StandardScaler()
        self.label_encoder = None
        self.yield_label_encoder = None
        self.is_trained = False
        self.crop_database = self._initialize_crop_database()
        self.feature_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        self.yield_feature_columns = ["Area", "Crop_Year", "Crop_encoded", "Season_encoded"]
        
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
        Predict top N suitable crops for given conditions using trained models
        """
        if not self.is_trained:
            self.train_model()
        
        # Prepare features for crop classification
        # Ensure we have the right columns in the right order
        crop_features = features[self.feature_columns].copy()
        
        # Scale features for crop classification
        X_scaled = self.scaler.transform(crop_features.values)
        
        # Get prediction probabilities for crop classification
        crop_probs = self.crop_classifier.predict_proba(X_scaled)[0]
        crop_classes = self.crop_classifier.classes_  # These are integer labels
        
        # Get top N crops
        top_indices = np.argsort(crop_probs)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            crop_encoded = crop_classes[idx]  # This is an integer
            confidence = float(crop_probs[idx])
            
            # Convert encoded crop back to crop name using label encoder
            if self.label_encoder:
                crop = self.label_encoder.inverse_transform([crop_encoded])[0]
            else:
                # Fallback: use crop database keys
                crop_names = list(self.crop_database.keys())
                crop = crop_names[crop_encoded % len(crop_names)]
            
            # Predict yield for each crop
            predicted_yield = self._predict_yield_for_crop(crop, features)
            
            # Calculate profit estimation (simplified)
            market_price = np.random.uniform(2000, 4000)  # Rs per quintal
            estimated_profit = predicted_yield * market_price / 10  # Convert to Rs/hectare
            
            # Get crop details
            crop_info = self.crop_database.get(crop, {})
            
            recommendation = {
                "crop": crop,
                "confidence": round(confidence, 3),
                "predicted_yield": round(predicted_yield, 2),
                "estimated_profit": round(estimated_profit, 0),
                "profit_margin": crop_info.get("profit_margin", "Medium"),
                "sustainability_score": crop_info.get("sustainability_score", 7),
                "suitable_season": crop_info.get("season", ["Kharif"]),
                "care_tips": self._generate_care_tips(crop)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _predict_yield_for_crop(self, crop: str, features: pd.DataFrame) -> float:
        """
        Predict yield for a specific crop using the yield predictor model
        """
        try:
            # The yield predictor predicts total Production (in tons) based on:
            # [Area, Crop_Year, Crop_encoded, Season_encoded]
            # We need to calculate yield per hectare = Production / Area
            
            # Use a reasonable area for yield calculation (1 hectare)
            area_hectares = 1.0
            crop_year = 2023  # Current year
            
            # Encode crop name using yield label encoder
            if self.yield_label_encoder and hasattr(self.yield_label_encoder, 'classes_'):
                try:
                    # Convert crop name to match training data format (capitalized)
                    crop_formatted = crop.capitalize()
                    
                    # Check if crop exists in yield encoder
                    if crop_formatted in self.yield_label_encoder.classes_:
                        crop_encoded = self.yield_label_encoder.transform([crop_formatted])[0]
                    else:
                        # Try common crop name mappings
                        crop_mappings = {
                            'rice': 'Rice',
                            'wheat': 'Wheat', 
                            'maize': 'Maize',
                            'cotton': 'Cotton',
                            'sugarcane': 'Sugarcane',
                            'jute': 'Jute',
                            'banana': 'Banana',
                            'coffee': 'Coffee',
                            'coconut': 'Coconut',
                            'chickpea': 'Gram',
                            'lentil': 'Lentil'
                        }
                        crop_mapped = crop_mappings.get(crop.lower(), crop_formatted)
                        if crop_mapped in self.yield_label_encoder.classes_:
                            crop_encoded = self.yield_label_encoder.transform([crop_mapped])[0]
                        else:
                            # Use first available crop as fallback
                            crop_encoded = 0
                except ValueError:
                    crop_encoded = 0
            else:
                crop_encoded = 0
            
            # Determine season encoding (based on training data: 0=Kharif, 1=Rabi, 2=Summer/Zaid)
            crop_info = self.crop_database.get(crop, {})
            season = crop_info.get("season", ["Kharif"])[0].lower()
            season_encoded = {"kharif": 0, "rabi": 1, "zaid": 2, "summer": 2}.get(season, 0)
            
            # Create yield feature vector [Area, Crop_Year, Crop_encoded, Season_encoded]
            # Use median area from training data (around 580 hectares)
            prediction_area = 580.0  # Use median area for more realistic predictions
            yield_features = np.array([[prediction_area, crop_year, crop_encoded, season_encoded]])
            
            # Scale yield features
            if self.yield_scaler:
                try:
                    yield_features_scaled = self.yield_scaler.transform(yield_features)
                except:
                    # If scaling fails, use original features
                    yield_features_scaled = yield_features
            else:
                yield_features_scaled = yield_features
            
            # Predict total production for the prediction area
            predicted_production = self.yield_predictor.predict(yield_features_scaled)[0]
            
            # Calculate yield per hectare
            predicted_yield_per_hectare = predicted_production / prediction_area
            
            # Ensure reasonable yield values (tons per hectare)
            # Typical yields: Rice: 2-6, Wheat: 2-5, Maize: 3-8, Cotton: 1-3, Sugarcane: 60-100
            if predicted_yield_per_hectare <= 0:
                # Use fallback default yields if prediction is negative or zero
                default_yields = {
                    "rice": 3.5, "wheat": 3.0, "maize": 4.0, "cotton": 2.0,
                    "sugarcane": 75.0, "jute": 2.5, "banana": 15.0, "coffee": 1.0,
                    "coconut": 0.8, "chickpea": 1.5, "lentil": 1.2, "soybean": 2.5
                }
                predicted_yield_per_hectare = default_yields.get(crop.lower(), 3.0)
            else:
                # Cap extremely high values
                predicted_yield_per_hectare = min(predicted_yield_per_hectare, 150.0)
            
            return predicted_yield_per_hectare
            
        except Exception as e:
            print(f"Error predicting yield for {crop}: {e}")
            # Return a reasonable default yield based on crop type
            default_yields = {
                "rice": 3.5, "wheat": 3.0, "maize": 4.0, "cotton": 2.0,
                "sugarcane": 75.0, "jute": 2.5, "banana": 15.0, "coffee": 1.0,
                "coconut": 0.8, "chickpea": 1.5, "lentil": 1.2, "soybean": 2.5
            }
            return default_yields.get(crop.lower(), 3.0)
    
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
        joblib.dump(self.yield_scaler, f"{path}yield_scaler.pkl")
        
        if self.label_encoder:
            joblib.dump(self.label_encoder, f"{path}label_encoder.pkl")
        if self.yield_label_encoder:
            joblib.dump(self.yield_label_encoder, f"{path}yield_label_encoder.pkl")
        
        # Save model metadata
        metadata = {
            "feature_columns": self.feature_columns,
            "yield_feature_columns": self.yield_feature_columns,
            "crops": list(self.crop_classifier.classes_) if hasattr(self.crop_classifier, 'classes_') else list(self.crop_database.keys())
        }
        
        with open(f"{path}model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(f"{path}crop_database.json", 'w') as f:
            json.dump(self.crop_database, f)
    
    def load_model(self, path: str = "ml_models/"):
        """
        Load pre-trained model with all components
        """
        try:
            # Load the main models
            self.crop_classifier = joblib.load(f"{path}crop_classifier.pkl")
            self.yield_predictor = joblib.load(f"{path}yield_predictor.pkl")
            
            # Load scalers
            self.scaler = joblib.load(f"{path}scaler.pkl")
            
            # Try to load yield scaler (may not exist in older models)
            try:
                self.yield_scaler = joblib.load(f"{path}yield_scaler.pkl")
            except FileNotFoundError:
                print("Yield scaler not found, using default scaler")
                self.yield_scaler = StandardScaler()
            
            # Try to load label encoders
            try:
                self.label_encoder = joblib.load(f"{path}label_encoder.pkl")
            except FileNotFoundError:
                print("Label encoder not found")
                self.label_encoder = None
                
            try:
                self.yield_label_encoder = joblib.load(f"{path}yield_label_encoder.pkl")
            except FileNotFoundError:
                print("Yield label encoder not found")
                self.yield_label_encoder = None
            
            # Load metadata if available
            try:
                with open(f"{path}model_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get("feature_columns", self.feature_columns)
                    self.yield_feature_columns = metadata.get("yield_feature_columns", self.yield_feature_columns)
            except FileNotFoundError:
                print("Model metadata not found, using default feature columns")
            
            # Load crop database if available
            try:
                with open(f"{path}crop_database.json", 'r') as f:
                    self.crop_database = json.load(f)
            except FileNotFoundError:
                print("Crop database not found, using default database")
            
            self.is_trained = True
            print("Successfully loaded all model components")
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Training new model...")
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
