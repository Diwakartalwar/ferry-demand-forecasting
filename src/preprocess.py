"""
Data Preprocessing Module for Ferry Demand Forecasting System

This module handles:
- Data loading and validation
- Datetime conversion and sorting
- Missing interval handling
- Frequency enforcement (15-minute intervals)
- Outlier detection and handling
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class FerryDataPreprocessor:
    """
    Preprocess ferry ticket demand data for time-series forecasting.
    
    Expected schema:
    - _id: Unique identifier
    - Timestamp: Datetime string
    - Sales Count: Number of tickets sold
    - Redemption Count: Number of tickets redeemed
    """
    
    def __init__(self, freq: str = '15min'):
        """
        Initialize the preprocessor.
        
        Args:
            freq: Time frequency for the data (default: '15min')
        """
        self.freq = freq
        self.data = None
        self.original_shape = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        self.data = pd.read_csv(filepath)
        self.original_shape = self.data.shape
        print(f"Loaded data with shape: {self.original_shape}")
        return self.data
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has the required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_columns = ['Timestamp', 'Sales Count', 'Redemption Count']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return True
    
    def convert_datetime(self, df: pd.DataFrame, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
        """
        Convert timestamp column to datetime and set as index.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with datetime index
        """
        df = df.copy()
        
        # Try multiple datetime formats
        for fmt in [None, '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S']:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=fmt)
                break
            except:
                continue
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col)
        
        # Set as index
        df = df.set_index(timestamp_col)
        
        return df
    
    def enforce_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce 15-minute frequency and handle missing intervals.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with complete 15-minute intervals
        """
        df = df.copy()
        
        # Create complete date range
        date_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=self.freq
        )
        
        # Reindex to complete range
        df = df.reindex(date_range)
        
        # Report missing intervals
        missing_count = df['Sales Count'].isna().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing intervals. Will interpolate.")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing values ('interpolate', 'forward_fill', 'backward_fill')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        numeric_cols = ['Sales Count', 'Redemption Count']
        
        if method == 'interpolate':
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='time', limit_direction='both')
        elif method == 'forward_fill':
            for col in numeric_cols:
                df[col] = df[col].fillna(method='ffill')
        elif method == 'backward_fill':
            for col in numeric_cols:
                df[col] = df[col].fillna(method='bfill')
        
        # Fill any remaining NaN with 0
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, column: str = 'Sales Count', 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in the specified column.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df[f'{column}_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df[f'{column}_outlier'] = z_scores > threshold
        
        outlier_count = df[f'{column}_outlier'].sum()
        print(f"Detected {outlier_count} outliers in {column} using {method} method")
        
        return df
    
    def cap_outliers(self, df: pd.DataFrame, column: str = 'Sales Count',
                    method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Cap outliers at the bounds instead of removing them.
        
        Args:
            df: Input DataFrame
            column: Column to cap
            method: Method for outlier detection
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with capped values
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                          handle_outliers: bool = True,
                          outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw input DataFrame
            handle_outliers: Whether to handle outliers
            outlier_method: Method for outlier handling
            
        Returns:
            Fully preprocessed DataFrame
        """
        print("Starting preprocessing pipeline...")
        print(f"Original shape: {df.shape}")
        
        # Validate schema
        self.validate_schema(df)
        
        # Convert datetime
        df = self.convert_datetime(df)
        print(f"After datetime conversion: {df.shape}")
        
        # Enforce frequency
        df = self.enforce_frequency(df)
        print(f"After frequency enforcement: {df.shape}")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        print(f"After handling missing values: {df.shape}")
        
        # Handle outliers
        if handle_outliers:
            df = self.detect_outliers(df, method=outlier_method)
            df = self.cap_outliers(df, method=outlier_method)
            print(f"After outlier handling: {df.shape}")
        
        self.data = df
        print(f"Final shape: {df.shape}")
        print("Preprocessing pipeline completed.")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed data to CSV.
        
        Args:
            df: Processed DataFrame
            output_path: Path to save the file
        """
        df.to_csv(output_path)
        print(f"Processed data saved to {output_path}")


def create_sample_data(output_path: str, days: int = 30) -> None:
    """
    Synthetic data generation is disabled.
    
    Args:
        output_path: Path to save the sample data
        days: Number of days of data to generate
    """
    raise RuntimeError("Synthetic data generation is disabled. Use the primary ferry dataset.")


if __name__ == "__main__":
    # Example usage
    preprocessor = FerryDataPreprocessor()

    sample_path = Path(__file__).resolve().parent.parent / "data/raw/ferry_sample_data.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Primary dataset not found: {sample_path}")

    # Load and preprocess
    df = preprocessor.load_data(str(sample_path))
    processed_df = preprocessor.preprocess_pipeline(df)
    
    # Save processed data
    preprocessor.save_processed_data(processed_df, "data/processed/ferry_processed.csv")
