"""
Forecasting Module for Ferry Demand Forecasting System

This module handles:
- Loading trained models
- Making predictions for multiple horizons
- Generating confidence intervals
- Creating forecast dataframes
- Operational alert generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib
from datetime import datetime, timedelta


class FerryForecaster:
    """
    Generate forecasts using trained models.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the forecaster.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.horizons = {
            '15m': 1,
            '30m': 2,
            '1h': 4,
            '2h': 8
        }
        
    def load_model(self, model_name: str, filepath: Optional[str] = None) -> object:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model
            filepath: Path to model file (if None, uses default)
            
        Returns:
            Loaded model
        """
        if filepath is None:
            filepath = f"{self.models_dir}/{model_name.replace(' ', '_').lower()}.joblib"
        
        model = joblib.load(filepath)
        self.models[model_name] = model
        print(f"Loaded model: {model_name}")
        
        return model
    
    def load_all_models(self) -> None:
        """Load all models from the models directory."""
        model_files = Path(self.models_dir).glob("*.joblib")
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_', ' ').title()
            self.load_model(model_name, str(model_file))
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a specific model.
        
        Args:
            model_name: Name of the model to use
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        return model.predict(X)
    
    def predict_all_models(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using all loaded models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping model names to predictions
        """
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                predictions[model_name] = model.predict(X)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
        
        return predictions
    
    def predict_ensemble(self, X: pd.DataFrame, 
                        model_names: Optional[List[str]] = None,
                        method: str = 'mean') -> np.ndarray:
        """
        Make ensemble predictions from multiple models.
        
        Args:
            X: Feature matrix
            model_names: List of models to include (if None, uses all)
            method: Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
            Ensemble predictions
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        predictions = []
        for model_name in model_names:
            if model_name in self.models:
                predictions.append(self.predict(model_name, X))
        
        predictions = np.array(predictions)
        
        if method == 'mean':
            return np.mean(predictions, axis=0)
        elif method == 'median':
            return np.median(predictions, axis=0)
        elif method == 'weighted':
            # Simple equal weights for now
            weights = np.ones(len(predictions)) / len(predictions)
            return np.average(predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def generate_forecast_dataframe(self, X: pd.DataFrame, 
                                   predictions: np.ndarray,
                                   timestamps: pd.DatetimeIndex,
                                   model_name: str) -> pd.DataFrame:
        """
        Create a forecast dataframe with timestamps and predictions.
        
        Args:
            X: Feature matrix
            predictions: Predicted values
            timestamps: Timestamps for predictions
            model_name: Name of the model used
            
        Returns:
            DataFrame with forecast results
        """
        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            'forecast': predictions,
            'model': model_name
        })
        
        forecast_df = forecast_df.set_index('timestamp')
        
        return forecast_df
    
    def calculate_confidence_intervals(self, predictions: np.ndarray,
                                      historical_errors: Optional[np.ndarray] = None,
                                      confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: Predicted values
            historical_errors: Historical prediction errors (for estimating std)
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if historical_errors is not None:
            error_std = np.std(historical_errors)
        else:
            # Default: assume 10% of prediction as std
            error_std = 0.1 * np.abs(predictions)
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin = z_score * error_std
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        # Ensure bounds are non-negative
        lower_bound = np.maximum(lower_bound, 0)
        
        return lower_bound, upper_bound
    
    def generate_operational_alert(self, forecast: float, 
                                  thresholds: Dict[str, float]) -> str:
        """
        Generate operational alert based on forecast.
        
        Args:
            forecast: Forecasted demand
            thresholds: Dictionary of alert thresholds
            
        Returns:
            Alert level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if forecast >= thresholds.get('critical', 100):
            return 'CRITICAL'
        elif forecast >= thresholds.get('high', 75):
            return 'HIGH'
        elif forecast >= thresholds.get('medium', 50):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_forecast_report(self, X: pd.DataFrame,
                                timestamps: pd.DatetimeIndex,
                                model_name: str = 'Random Forest',
                                confidence: float = 0.95,
                                thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate a comprehensive forecast report.
        
        Args:
            X: Feature matrix
            timestamps: Timestamps for predictions
            model_name: Model to use for forecasting
            confidence: Confidence level for intervals
            thresholds: Alert thresholds
            
        Returns:
            DataFrame with forecast report
        """
        if thresholds is None:
            thresholds = {
                'critical': 100,
                'high': 75,
                'medium': 50,
                'low': 0
            }
        
        # Make predictions
        predictions = self.predict(model_name, X)
        
        # Calculate confidence intervals
        lower, upper = self.calculate_confidence_intervals(predictions, confidence=confidence)
        
        # Generate alerts
        alerts = [self.generate_operational_alert(pred, thresholds) for pred in predictions]
        
        # Create report
        report_df = pd.DataFrame({
            'timestamp': timestamps,
            'forecast': predictions,
            'lower_bound': lower,
            'upper_bound': upper,
            'alert_level': alerts,
            'model': model_name
        })
        
        report_df = report_df.set_index('timestamp')
        
        return report_df
    
    def forecast_multiple_horizons(self, X_dict: Dict[str, pd.DataFrame],
                                   timestamps_dict: Dict[str, pd.DatetimeIndex],
                                   model_name: str = 'Random Forest',
                                   confidence: float = 0.95,
                                   thresholds: Optional[Dict[str, float]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for multiple horizons.
        
        Args:
            X_dict: Dictionary mapping horizons to feature matrices
            timestamps_dict: Dictionary mapping horizons to timestamps
            model_name: Model to use
            confidence: Confidence level
            thresholds: Alert thresholds
            
        Returns:
            Dictionary mapping horizons to forecast reports
        """
        reports = {}
        
        for horizon in self.horizons.keys():
            if horizon in X_dict:
                report = self.generate_forecast_report(
                    X_dict[horizon],
                    timestamps_dict[horizon],
                    model_name,
                    confidence,
                    thresholds
                )
                reports[horizon] = report
        
        return reports
    
    def get_next_forecast(self, X: pd.DataFrame, 
                         current_time: pd.Timestamp,
                         model_name: str = 'Random Forest',
                         horizon: str = '1h') -> Dict:
        """
        Get the next forecast for a specific horizon.
        
        Args:
            X: Feature matrix (should be for the next time step)
            current_time: Current timestamp
            model_name: Model to use
            horizon: Forecast horizon
            
        Returns:
            Dictionary with forecast information
        """
        prediction = self.predict(model_name, X)
        forecast_time = current_time + pd.Timedelta(hours=int(horizon.replace('h', '')))
        
        lower, upper = self.calculate_confidence_intervals(prediction)
        
        return {
            'forecast_time': forecast_time,
            'forecast_value': float(prediction[0]),
            'lower_bound': float(lower[0]),
            'upper_bound': float(upper[0]),
            'horizon': horizon,
            'model': model_name
        }


class ForecastPipeline:
    """
    Complete forecasting pipeline from data to predictions.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the forecasting pipeline.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.forecaster = FerryForecaster(models_dir)
        self.is_initialized = False
        
    def initialize(self, model_names: Optional[List[str]] = None) -> None:
        """
        Initialize the pipeline by loading models.
        
        Args:
            model_names: Specific models to load (if None, loads all)
        """
        if model_names is None:
            self.forecaster.load_all_models()
        else:
            for model_name in model_names:
                try:
                    self.forecaster.load_model(model_name)
                except FileNotFoundError:
                    print(f"Model {model_name} not found, skipping")
        
        self.is_initialized = True
        print(f"Pipeline initialized with {len(self.forecaster.models)} models")
    
    def run_forecast(self, X: pd.DataFrame,
                    timestamps: pd.DatetimeIndex,
                    model_name: str = 'Random Forest',
                    confidence: float = 0.95) -> pd.DataFrame:
        """
        Run the forecasting pipeline.
        
        Args:
            X: Feature matrix
            timestamps: Timestamps for predictions
            model_name: Model to use
            confidence: Confidence level
            
        Returns:
            Forecast report
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        return self.forecaster.generate_forecast_report(
            X, timestamps, model_name, confidence
        )
    
    def save_forecast(self, forecast_df: pd.DataFrame, output_path: str) -> None:
        """
        Save forecast to CSV.
        
        Args:
            forecast_df: Forecast dataframe
            output_path: Path to save
        """
        forecast_df.to_csv(output_path)
        print(f"Forecast saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    from preprocess import FerryDataPreprocessor, create_sample_data
    from features import FerryFeatureEngineer
    
    # Create sample data
    create_sample_data("data/raw/ferry_sample_data.csv", days=30)
    
    # Preprocess
    preprocessor = FerryDataPreprocessor()
    df = preprocessor.load_data("data/raw/ferry_sample_data.csv")
    df = preprocessor.preprocess_pipeline(df)
    
    # Feature engineering
    engineer = FerryFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Get feature matrix for 1h horizon
    X, y = engineer.get_feature_matrix(df_features, horizon='1h')
    
    # Get the last few rows for forecasting
    X_forecast = X.tail(10)
    timestamps_forecast = X_forecast.index
    
    # Initialize forecaster (would normally load trained models)
    forecaster = FerryForecaster()
    
    # For demonstration, create dummy predictions
    dummy_predictions = np.random.randint(20, 60, size=len(X_forecast))
    
    # Generate forecast report
    forecast_df = forecaster.generate_forecast_dataframe(
        X_forecast, dummy_predictions, timestamps_forecast, 'Random Forest'
    )
    
    print("Sample Forecast:")
    print(forecast_df.head())
