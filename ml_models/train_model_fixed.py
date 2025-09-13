"""
Model Training Script with Real Data
Trains crop recommendation models using actual datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import json
import os
import sys
import warnings
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.real_data_collector import RealDataCollector, DatasetBuilder

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Train ML models with real agricultural data
    """
    
    def __init__(self):
        self.crop_classifier = None
        self.yield_predictor = None
        self.scaler = StandardScaler()  # For crop classification features
        self.yield_scaler = StandardScaler()  # For yield prediction features
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.yield_feature_columns = []
        self.model_metrics = {}
        
    def load_crop_production_dataset(self, filepath: str = 'ml_models/data/crop_production.csv') -> pd.DataFrame:
        """
        Load and prepare crop_production.csv for yield prediction
        """
        df = pd.read_csv(filepath)
        # Remove rows with missing Area or Production
        df = df.dropna(subset=['Area', 'Production'])
        # Compute yield (Production per Area)
        df['yield'] = df['Production'] / df['Area']
        # Remove infinite or NaN yields
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['yield'])
        # Select features: Area, Crop_Year, Season, Crop (encoded), State/District (optional)
        # Encode categorical features
        df['Crop'] = df['Crop'].astype(str)
        df['Season'] = df['Season'].astype(str)
        df['Crop_encoded'] = self.label_encoder.fit_transform(df['Crop'])
        df['Season_encoded'] = pd.factorize(df['Season'])[0]
        # Features for yield prediction
        feature_cols = ['Area', 'Crop_Year', 'Crop_encoded', 'Season_encoded']
        self.yield_feature_columns = feature_cols
        print(f"Loaded crop_production.csv: {df.shape}")
        return df
        
    def load_kaggle_dataset(self, filepath: str = None) -> pd.DataFrame:
        """
        Load the popular Crop Recommendation Dataset from Kaggle
        Source: https://www.kaggle.com/atharvaingle/crop-recommendation-dataset
        """
        if filepath is None:
            # Download using kaggle API (requires kaggle.json in ~/.kaggle/)
            try:
                import kaggle
                kaggle.api.dataset_download_files(
                    'atharvaingle/crop-recommendation-dataset',
                    path='data/',
                    unzip=True
                )
                filepath = 'data/Crop_recommendation.csv'
            except:
                print("Kaggle API not configured. Please provide dataset path.")
                return None
        
        # Load the dataset
        df = pd.read_csv(filepath)
        print(f"Loaded Kaggle dataset: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Crops: {df['label'].unique() if 'label' in df.columns else 'N/A'}")
        
        return df
    
    def load_indian_crop_dataset(self, filepath: str = None) -> pd.DataFrame:
        """
        Load Indian Government's Agricultural Dataset
        Source: data.gov.in
        """
        if filepath is None:
            # Use the DatasetBuilder to fetch from API
            builder = DatasetBuilder()
            df = builder.fetch_yield_data("Jharkhand")
            
            if df.empty:
                print("Could not fetch data from API. Using local file if available.")
                filepath = 'data/india_agriculture.csv'
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)
        
        print(f"Loaded Indian dataset: {df.shape}")
            # Add yield column if possible
        if 'Production' in df.columns and 'Area' in df.columns:
            df['yield'] = df['Production'] / df['Area']
            print("Yield column added to dataset.")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'label') -> tuple:
        """
        Prepare data for training
        """
        # Handle different dataset formats
        if target_column in df.columns:
            # Kaggle dataset format
            feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            
            # Check if all required columns exist
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < len(feature_cols):
                print(f"Warning: Some features missing. Using: {available_cols}")
                feature_cols = available_cols
            
            X = df[feature_cols]
            y = df[target_column]
            
            # Store feature columns for later use
            self.feature_columns = feature_cols
            
        else:
            # Custom format - need to adapt
            print("Custom dataset format detected. Adapting...")
            
            # Map column names
            column_mapping = {
                'nitrogen': 'N',
                'phosphorus': 'P',
                'potassium': 'K',
                'temp': 'temperature',
                'hum': 'humidity',
                'ph': 'ph',
                'rain': 'rainfall'
            }
            
            df_renamed = df.rename(columns=column_mapping)
            
            feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            available_cols = [col for col in feature_cols if col in df_renamed.columns]
            
            X = df_renamed[available_cols]
            
            # Find target column
            if 'crop' in df_renamed.columns:
                y = df_renamed['crop']
            elif 'label' in df_renamed.columns:
                y = df_renamed['label']
            else:
                raise ValueError("No target column found (crop/label)")
            
            self.feature_columns = available_cols
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features shape: {X_scaled.shape}")
        print(f"Number of unique crops: {len(np.unique(y_encoded))}")
        print(f"Crops: {self.label_encoder.classes_}")
        
        return X_scaled, y_encoded
    
    def train_crop_classifier(self, X, y, optimize_hyperparameters: bool = True):
        """
        Train the crop classification model
        """
        print("\n=== Training Crop Classifier ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if optimize_hyperparameters:
            print("Optimizing hyperparameters...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Grid search with cross-validation
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, 
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            self.crop_classifier = grid_search.best_estimator_
        else:
            # Use default parameters
            self.crop_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.crop_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.crop_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.crop_classifier, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            digits=3
        ))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.crop_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Store metrics
        self.model_metrics['classifier'] = {
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance.to_dict()
        }
        
        return self.crop_classifier
    
    def train_yield_predictor(self, df: pd.DataFrame = None):
        """
        Train yield prediction model
        """
        print("\n=== Training Yield Predictor ===")
        
        if df is None:
            # Generate synthetic yield data for demonstration
            print("No yield data provided. Using synthetic data...")
            
            # Create synthetic yield data based on features
            n_samples = 1000
            np.random.seed(42)
            
            X_yield = np.random.randn(n_samples, len(self.feature_columns))
            
            # Generate yield based on features (simplified relationship)
            # Higher N, P, K and optimal pH, temperature lead to better yield
            y_yield = (
                X_yield[:, 0] * 0.3 +  # N effect
                X_yield[:, 1] * 0.2 +  # P effect
                X_yield[:, 2] * 0.2 +  # K effect
                np.random.randn(n_samples) * 0.5 + 5  # Base yield + noise
            )
            
            # Ensure positive yields
            y_yield = np.abs(y_yield)
            
        else:
            # Use actual yield data
            if 'yield' in df.columns and all(col in df.columns for col in self.yield_feature_columns):
                # Scale the features for yield prediction
                X_yield = self.yield_scaler.fit_transform(df[self.yield_feature_columns])
                y_yield = df['yield'].values
                print(f"Using actual yield data: {len(y_yield)} samples")
            else:
                print("Yield column not found or feature mismatch. Using synthetic data...")
                return self.train_yield_predictor(None)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_yield, y_yield, test_size=0.2, random_state=42
        )
        
        # Train model
        self.yield_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.yield_predictor.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.yield_predictor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        
        # Store metrics
        self.model_metrics['yield_predictor'] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
        
        return self.yield_predictor
    
    def save_models(self, path: str = 'ml_models/trained/'):
        """
        Save trained models and preprocessors
        """
        os.makedirs(path, exist_ok=True)
        
        # Save models
        joblib.dump(self.crop_classifier, f'{path}crop_classifier.pkl')
        joblib.dump(self.yield_predictor, f'{path}yield_predictor.pkl')
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        joblib.dump(self.yield_scaler, f'{path}yield_scaler.pkl')
        joblib.dump(self.label_encoder, f'{path}label_encoder.pkl')
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'yield_feature_columns': self.yield_feature_columns,
            'crops': self.label_encoder.classes_.tolist(),
            'metrics': self.model_metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f'{path}model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModels saved to {path}")
        
    def load_models(self, path: str = 'ml_models/trained/'):
        """
        Load pre-trained models
        """
        self.crop_classifier = joblib.load(f'{path}crop_classifier.pkl')
        self.yield_predictor = joblib.load(f'{path}yield_predictor.pkl')
        self.scaler = joblib.load(f'{path}scaler.pkl')
        self.yield_scaler = joblib.load(f'{path}yield_scaler.pkl')
        self.label_encoder = joblib.load(f'{path}label_encoder.pkl')
        
        with open(f'{path}model_metadata.json', 'r') as f:
            metadata = json.load(f)
            self.feature_columns = metadata['feature_columns']
            self.yield_feature_columns = metadata['yield_feature_columns']
            self.model_metrics = metadata.get('metrics', {})
        
        print(f"Models loaded from {path}")
        
    def predict(self, input_data: dict) -> dict:
        """
        Make predictions with trained models for crop recommendation
        """
        # Use crop recommendation features (N, P, K, etc.)
        crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Prepare input
        X = pd.DataFrame([input_data])[crop_features]
        X_scaled = self.scaler.transform(X)
        
        # Predict crop
        crop_probs = self.crop_classifier.predict_proba(X_scaled)[0]
        top_3_indices = np.argsort(crop_probs)[-3:][::-1]
        
        # For yield prediction, we'll use a default value since we can't map soil features to yield features
        default_yield = 3.5  # Default yield estimate
        
        # Prepare results
        recommendations = []
        for idx in top_3_indices:
            crop = self.label_encoder.inverse_transform([idx])[0]
            confidence = crop_probs[idx]
            
            recommendations.append({
                'crop': crop,
                'confidence': float(confidence),
                'predicted_yield': float(default_yield)
            })
        
        return {
            'recommendations': recommendations,
            'input_features': input_data
        }
    
    def predict_yield(self, input_data: dict) -> float:
        """
        Predict yield using crop production features
        """
        # Use crop production features (Area, Crop_Year, Crop_encoded, Season_encoded)
        yield_features = ['Area', 'Crop_Year', 'Crop_encoded', 'Season_encoded']
        
        # Prepare input
        X = pd.DataFrame([input_data])[yield_features]
        X_scaled = self.yield_scaler.transform(X)
        
        # Predict yield
        yield_pred = self.yield_predictor.predict(X_scaled)[0]
        
        return float(yield_pred)


def main():
    """
    Main training pipeline
    """
    print("="*50)
    print("Crop Recommendation Model Training")
    print("="*50)
    
    trainer = ModelTrainer()
    
    # Option 1: Train with Kaggle dataset
    print("\n1. Attempting to load Kaggle dataset...")
    kaggle_path = 'data/Crop_recommendation.csv'
    
    if os.path.exists(kaggle_path):
        df = trainer.load_kaggle_dataset(kaggle_path)
    else:
        print(f"Kaggle dataset not found at {kaggle_path}")
        print("Download from: https://www.kaggle.com/atharvaingle/crop-recommendation-dataset")
        print("\n2. Attempting to fetch Indian Government data...")
        df = trainer.load_indian_crop_dataset()
    
    if df is not None and not df.empty:
        # Prepare data for crop classifier (if Kaggle/Indian dataset)
        X, y = trainer.prepare_training_data(df)
        trainer.train_crop_classifier(X, y, optimize_hyperparameters=False)
        
        # Train yield predictor using crop_production.csv
        print("\n=== Training Yield Predictor with crop_production.csv ===")
        df_yield = trainer.load_crop_production_dataset()
        trainer.train_yield_predictor(df_yield)
        trainer.save_models()
        
        # Test prediction
        print("\n=== Testing Prediction ===")
        test_input = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.93
        }
        
        result = trainer.predict(test_input)
        print(f"Input: {test_input}")
        print(f"Predictions:")
        for rec in result['recommendations']:
            print(f"  - {rec['crop']}: {rec['confidence']:.2%} confidence, "
                  f"Yield: {rec['predicted_yield']:.2f} tons/ha")
        
        # Test yield prediction specifically
        print("\n=== Testing Yield Predictor ===")
        yield_test_input = {
            'Area': 100.0,          # Area in hectares
            'Crop_Year': 2023,      # Recent year
            'Crop_encoded': 50,     # Encoded crop value (e.g., Rice)
            'Season_encoded': 1     # Encoded season (e.g., Kharif)
        }
        
        predicted_yield = trainer.predict_yield(yield_test_input)
        
        print(f"Yield Input: {yield_test_input}")
        print(f"Predicted Yield: {predicted_yield:.2f} tons/ha")
    
    else:
        print("\nNo dataset available for training!")
        print("Please provide one of the following:")
        print("1. Download Kaggle dataset to data/Crop_recommendation.csv")
        print("2. Configure API keys in .env file for real data fetching")
        
        # Option 2: Build dataset from APIs
        print("\n3. Building dataset from APIs (requires API keys)...")
        
        builder = DatasetBuilder()
        
        # Define locations in Jharkhand
        jharkhand_locations = [
            (23.3441, 85.3096),  # Ranchi
            (23.7957, 86.4304),  # Dhanbad
            (22.8046, 86.2029),  # Jamshedpur
            (23.6693, 86.1511),  # Bokaro
            (24.2074, 84.3670),  # Hazaribagh
        ]
        
        # Try to build dataset
        try:
            df_api = builder.build_training_dataset(jharkhand_locations)
            if not df_api.empty:
                print("Successfully built dataset from APIs!")
                # Train with API data
                # ... (training code here)
        except Exception as e:
            print(f"Failed to build dataset from APIs: {e}")


if __name__ == "__main__":
    main()