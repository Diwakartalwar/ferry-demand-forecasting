"""
Training Module for Ferry Demand Forecasting System

This module handles:
- Baseline models (Naive, Moving Average, Linear Regression)
- Machine Learning models (Random Forest, Gradient Boosting, XGBoost)
- Time-series models (ARIMA, SARIMA, Prophet)
- Time-based train/test split
- Model serialization
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

# XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
        
    def save(self, filepath: str) -> None:
        """Save the model."""
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        """Load a saved model."""
        return joblib.load(filepath)


class NaiveForecaster(BaseModel):
    """Naive forecast: last observed value."""
    
    def __init__(self):
        super().__init__("Naive")
        self.last_value = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Store the last observed value."""
        self.last_value = y.iloc[-1]
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the last observed value."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(len(X), self.last_value)


class MovingAverageForecaster(BaseModel):
    """Moving average forecast."""
    
    def __init__(self, window: int = 4):
        super().__init__(f"Moving Average (window={window})")
        self.window = window
        self.mean_value = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate moving average."""
        self.mean_value = y.tail(self.window).mean()
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using moving average."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(len(X), self.mean_value)


class LinearRegressionModel(BaseModel):
    """Linear Regression model."""
    
    def __init__(self):
        super().__init__("Linear Regression")
        self.model = LinearRegression()
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit linear regression."""
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class RandomForestModel(BaseModel):
    """Random Forest Regressor."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit random forest."""
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Regressor."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit gradient boosting."""
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost Regressor."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        super().__init__("XGBoost")
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost."""
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class ARIMAModel(BaseModel):
    """ARIMA time series model."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__(f"ARIMA{order}")
        self.order = order
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit ARIMA model."""
        # ARIMA only needs the time series, not features
        self.model = ARIMA(y, order=self.order)
        self.model = self.model.fit()
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        forecast = self.model.forecast(steps=len(X))
        return forecast.values


class SARIMAModel(BaseModel):
    """SARIMA time series model."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 96)):
        super().__init__(f"SARIMA{order}{seasonal_order}")
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit SARIMA model."""
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
        self.model = self.model.fit(disp=False)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        forecast = self.model.forecast(steps=len(X))
        return forecast.values


class ProphetModel(BaseModel):
    """Facebook Prophet model."""
    
    def __init__(self):
        super().__init__("Prophet")
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed")
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Prophet model."""
        # Prophet requires specific format
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })
        self.model.fit(df)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # Create future dates
        future = pd.DataFrame({
            'ds': X.index
        })
        forecast = self.model.predict(future)
        return forecast['yhat'].values


class ModelTrainer:
    """
    Train and manage multiple forecasting models.
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        
    def register_model(self, name: str, model: BaseModel) -> None:
        """Register a model for training."""
        self.models[name] = model
        
    def time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                         test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                          pd.Series, pd.Series]:
        """
        Perform time-based train/test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, 
                   y_train: pd.Series) -> BaseModel:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        model = self.models[model_name]
        print(f"Training {model.name}...")
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        print(f"Completed training {model.name}")
        
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        skip_models: Optional[list] = None) -> Dict[str, BaseModel]:
        """
        Train all registered models.
        
        Args:
            X_train: Training features
            y_train: Training target
            skip_models: List of model names to skip
            
        Returns:
            Dictionary of trained models
        """
        if skip_models is None:
            skip_models = []
        
        for model_name in self.models:
            if model_name in skip_models:
                print(f"Skipping {model_name}")
                continue
            
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        return self.trained_models
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """Save a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        self.trained_models[model_name].save(filepath)
        print(f"Model saved to {filepath}")
    
    def save_all_models(self, output_dir: str) -> None:
        """Save all trained models."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = f"{output_dir}/{model_name.replace(' ', '_').lower()}.joblib"
            self.save_model(model_name, filepath)
    
    def load_model(self, filepath: str) -> BaseModel:
        """Load a trained model."""
        return joblib.load(filepath)


def setup_default_models() -> ModelTrainer:
    """
    Setup a trainer with default models.
    
    Returns:
        ModelTrainer with default models registered
    """
    trainer = ModelTrainer()
    
    # Baseline models
    trainer.register_model('naive', NaiveForecaster())
    trainer.register_model('ma', MovingAverageForecaster(window=4))
    trainer.register_model('lr', LinearRegressionModel())
    
    # Machine Learning models
    trainer.register_model('rf', RandomForestModel(n_estimators=100, max_depth=10))
    trainer.register_model('gb', GradientBoostingModel(n_estimators=100, learning_rate=0.1))
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        trainer.register_model('xgb', XGBoostModel(n_estimators=100, learning_rate=0.1))
    
    # Time Series models
    trainer.register_model('arima', ARIMAModel(order=(1, 1, 1)))
    # Note: SARIMA and Prophet can be slow, so they're optional
    # trainer.register_model('sarima', SARIMAModel())
    # if PROPHET_AVAILABLE:
    #     trainer.register_model('prophet', ProphetModel())
    
    return trainer


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
    
    # Time-based split
    trainer = setup_default_models()
    X_train, X_test, y_train, y_test = trainer.time_series_split(X, y, test_size=0.2)
    
    # Train all models
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Save models
    trainer.save_all_models("models")
    
    print(f"\nTrained {len(trained_models)} models successfully")
