"""
Feature Engineering Module for Ferry Demand Forecasting System

This module handles:
- Lag features (t-1, t-2, t-4, t-8)
- Rolling features (mean, std, max)
- Temporal features (hour, day of week, month, weekend)
- Forecast horizon targets (15m, 30m, 1h, 2h)
- Optional: Seasonal encoding, Fourier features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class FerryFeatureEngineer:
    """
    Engineer features for ferry demand forecasting.
    
    Creates lag features, rolling statistics, temporal features,
    and forecast targets for multiple horizons.
    """
    
    def __init__(self, target_col: str = 'Sales Count'):
        """
        Initialize the feature engineer.
        
        Args:
            target_col: Name of the target column to forecast
        """
        self.target_col = target_col
        self.feature_names = []
        self.horizons = {
            '15m': 1,    # 1 step ahead (15 min)
            '30m': 2,    # 2 steps ahead (30 min)
            '1h': 4,     # 4 steps ahead (1 hour)
            '2h': 8      # 8 steps ahead (2 hours)
        }
        
    def create_lag_features(self, df: pd.DataFrame, 
                           lags: List[int] = [1, 2, 4, 8]) -> pd.DataFrame:
        """
        Create lag features for the target variable.
        
        Args:
            df: Input DataFrame with datetime index
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{self.target_col}_lag_{lag}'] = df[self.target_col].shift(lag)
            self.feature_names.append(f'{self.target_col}_lag_{lag}')
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               windows: List[int] = [4, 8, 12]) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            windows: List of window sizes (in 15-min intervals)
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'{self.target_col}_rolling_mean_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).mean()
            )
            self.feature_names.append(f'{self.target_col}_rolling_mean_{window}')
            
            # Rolling std
            df[f'{self.target_col}_rolling_std_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).std()
            )
            self.feature_names.append(f'{self.target_col}_rolling_std_{window}')
            
            # Rolling max
            df[f'{self.target_col}_rolling_max_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).max()
            )
            self.feature_names.append(f'{self.target_col}_rolling_max_{window}')
            
            # Rolling min
            df[f'{self.target_col}_rolling_min_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).min()
            )
            self.feature_names.append(f'{self.target_col}_rolling_min_{window}')
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime index.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        # Hour of day (0-23)
        df['hour'] = df.index.hour
        self.feature_names.append('hour')
        
        # Day of week (0-6, Monday=0)
        df['day_of_week'] = df.index.dayofweek
        self.feature_names.append('day_of_week')
        
        # Month (1-12)
        df['month'] = df.index.month
        self.feature_names.append('month')
        
        # Day of month (1-31)
        df['day_of_month'] = df.index.day
        self.feature_names.append('day_of_month')
        
        # Weekend indicator (1 if weekend, 0 otherwise)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        self.feature_names.append('is_weekend')
        
        # Time of day categories
        df['time_of_day'] = pd.cut(
            df.index.hour,
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        # One-hot encode time of day
        time_dummies = pd.get_dummies(df['time_of_day'], prefix='time')
        df = pd.concat([df, time_dummies], axis=1)
        self.feature_names.extend(time_dummies.columns.tolist())
        
        # Drop the categorical column
        df = df.drop('time_of_day', axis=1)
        
        return df
    
    def create_fourier_features(self, df: pd.DataFrame, 
                               period: int = 96, 
                               n_terms: int = 3) -> pd.DataFrame:
        """
        Create Fourier features for seasonal patterns.
        
        Args:
            df: Input DataFrame
            period: Period for seasonality (96 = 24 hours in 15-min intervals)
            n_terms: Number of Fourier terms to create
            
        Returns:
            DataFrame with Fourier features added
        """
        df = df.copy()
        
        # Create a numeric time index
        t = np.arange(len(df))
        
        for i in range(1, n_terms + 1):
            # Sine term
            df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t / period)
            self.feature_names.append(f'fourier_sin_{i}')
            
            # Cosine term
            df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t / period)
            self.feature_names.append(f'fourier_cos_{i}')
        
        return df
    
    def create_forecast_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create shifted target variables for each forecast horizon.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with forecast target columns added
        """
        df = df.copy()
        
        for horizon_name, steps in self.horizons.items():
            # Shift target to create future values
            df[f'target_{horizon_name}'] = df[self.target_col].shift(-steps)
        
        return df
    
    def create_redemption_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on redemption counts.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with redemption features added
        """
        df = df.copy()
        
        # Redemption ratio (redemptions / sales)
        df['redemption_ratio'] = np.where(
            df['Sales Count'] > 0,
            df['Redemption Count'] / df['Sales Count'],
            0
        )
        self.feature_names.append('redemption_ratio')
        
        # Lagged redemption
        df['redemption_lag_1'] = df['Redemption Count'].shift(1)
        self.feature_names.append('redemption_lag_1')
        
        # Rolling redemption mean
        df['redemption_rolling_mean_4'] = (
            df['Redemption Count'].rolling(window=4, min_periods=1).mean()
        )
        self.feature_names.append('redemption_rolling_mean_4')
        
        return df
    
    def create_all_features(self, df: pd.DataFrame,
                           include_fourier: bool = True,
                           include_redemption: bool = True) -> pd.DataFrame:
        """
        Create all features in one pipeline.
        
        Args:
            df: Input DataFrame
            include_fourier: Whether to include Fourier features
            include_redemption: Whether to include redemption features
            
        Returns:
            DataFrame with all features
        """
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        print("Creating rolling features...")
        df = self.create_rolling_features(df)
        
        print("Creating temporal features...")
        df = self.create_temporal_features(df)
        
        if include_fourier:
            print("Creating Fourier features...")
            df = self.create_fourier_features(df)
        
        if include_redemption:
            print("Creating redemption features...")
            df = self.create_redemption_features(df)
        
        print("Creating forecast targets...")
        df = self.create_forecast_targets(df)
        
        print(f"Total features created: {len(self.feature_names)}")
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame, 
                          horizon: str = '1h') -> tuple:
        """
        Get feature matrix and target for a specific horizon.
        
        Args:
            df: DataFrame with all features
            horizon: Forecast horizon ('15m', '30m', '1h', '2h')
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        target_col = f'target_{horizon}'
        
        # Get feature columns (exclude target columns and original columns)
        exclude_cols = [
            'Sales Count', 'Redemption Count', 
            'Sales Count_outlier', '_id'
        ]
        exclude_cols.extend([f'target_{h}' for h in self.horizons.keys()])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Drop rows with NaN in target or features
        df_clean = df.dropna(subset=[target_col] + feature_cols)
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        return X, y
    
    def get_all_feature_matrices(self, df: pd.DataFrame) -> Dict[str, tuple]:
        """
        Get feature matrices for all horizons.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Dictionary mapping horizon names to (X, y) tuples
        """
        matrices = {}
        
        for horizon in self.horizons.keys():
            X, y = self.get_feature_matrix(df, horizon)
            matrices[horizon] = (X, y)
            print(f"Horizon {horizon}: X shape = {X.shape}, y shape = {y.shape}")
        
        return matrices


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from preprocess import FerryDataPreprocessor
    
    sample_path = Path(__file__).resolve().parent.parent / "data/raw/ferry_sample_data.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Primary dataset not found: {sample_path}")

    # Load and preprocess primary data
    preprocessor = FerryDataPreprocessor()
    df = preprocessor.load_data(str(sample_path))
    df = preprocessor.preprocess_pipeline(df)
    
    # Engineer features
    engineer = FerryFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"\nFinal feature matrix shape: {df_features.shape}")
    print(f"Feature names: {engineer.feature_names}")
    
    # Get matrices for all horizons
    matrices = engineer.get_all_feature_matrices(df_features)
