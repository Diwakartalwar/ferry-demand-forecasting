"""
Script to pre-train and save models for Streamlit Cloud deployment.
Run this locally to generate model files in the models/ directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from preprocess import FerryDataPreprocessor
from features import FerryFeatureEngineer
from train import setup_default_models
import joblib

def main():
    sample_path = Path(__file__).resolve().parent / "data/raw/ferry_sample_data.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Primary dataset not found: {sample_path}")
    
    # Load and preprocess
    preprocessor = FerryDataPreprocessor()
    df = preprocessor.load_data(str(sample_path))
    df = preprocessor.preprocess_pipeline(df)
    
    # Feature engineering
    engineer = FerryFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Get feature matrix for 1h horizon
    X, y = engineer.get_feature_matrix(df_features, horizon='1h')
    
    # Time-based split
    trainer = setup_default_models()
    X_train, X_test, y_train, y_test = trainer.time_series_split(X, y, test_size=0.2)
    
    # Train all models
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in trained_models.items():
        filepath = models_dir / f"{model_name}.joblib"
        joblib.dump(model, filepath)
        print(f"Saved model: {model_name} to {filepath}")
    
    print(f"\nSuccessfully trained and saved {len(trained_models)} models")

if __name__ == "__main__":
    main()
