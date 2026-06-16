"""
Evaluation Module for Ferry Demand Forecasting System

This module handles:
- Model evaluation metrics (MAE, RMSE, MAPE)
- Horizon-wise accuracy analysis
- Model comparison and ranking
- Forecast error visualization
- Confidence interval calculation
- Uncertainty analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ModelEvaluator:
    """
    Evaluate forecasting models with comprehensive metrics.
    """
    
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
        
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, 
                      y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics for a model.
        
        Args:
            model_name: Name of the model
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'MAE': self.calculate_mae(y_true, y_pred),
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'SMAPE': self.calculate_smape(y_true, y_pred),
            'R2': self.calculate_r2(y_true, y_pred)
        }
        
        self.metrics[model_name] = metrics
        self.predictions[model_name] = {'y_true': y_true, 'y_pred': y_pred}
        
        return metrics
    
    def calculate_confidence_intervals(self, y_pred: np.ndarray, 
                                      y_true: np.ndarray,
                                      confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            y_pred: Predicted values
            y_true: Actual values (for estimating error distribution)
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Calculate prediction errors
        errors = y_true - y_pred
        
        # Calculate standard deviation of errors
        error_std = np.std(errors)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate bounds
        margin = z_score * error_std
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        
        return lower_bound, upper_bound
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison table of all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(4)
        
        # Add ranking for each metric
        for metric in comparison_df.columns:
            if metric in ['R2']:  # Higher is better
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
            else:  # Lower is better
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=True)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'MAE') -> Tuple[str, float]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        comparison_df = self.compare_models()
        
        if metric in ['R2']:  # Higher is better
            best_idx = comparison_df[metric].idxmax()
        else:  # Lower is better
            best_idx = comparison_df[metric].idxmin()
        
        return best_idx, comparison_df.loc[best_idx, metric]
    
    def plot_forecast_vs_actual(self, model_name: str, 
                               timestamps: pd.DatetimeIndex,
                               n_points: int = 100) -> go.Figure:
        """
        Create an interactive plot of forecast vs actual.
        
        Args:
            model_name: Name of the model to plot
            timestamps: Timestamps for the data
            n_points: Number of points to plot (most recent)
            
        Returns:
            Plotly figure
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not evaluated")
        
        y_true = self.predictions[model_name]['y_true']
        y_pred = self.predictions[model_name]['y_pred']
        
        # Take the last n_points
        y_true = y_true[-n_points:]
        y_pred = y_pred[-n_points:]
        timestamps = timestamps[-n_points:]
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_pred,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Forecast vs Actual - {model_name}',
            xaxis_title='Timestamp',
            yaxis_title='Ticket Demand',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_forecast_with_confidence(self, model_name: str,
                                     timestamps: pd.DatetimeIndex,
                                     confidence: float = 0.95,
                                     n_points: int = 100) -> go.Figure:
        """
        Create a plot with confidence bands.
        
        Args:
            model_name: Name of the model
            timestamps: Timestamps for the data
            confidence: Confidence level
            n_points: Number of points to plot
            
        Returns:
            Plotly figure
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not evaluated")
        
        y_true = self.predictions[model_name]['y_true']
        y_pred = self.predictions[model_name]['y_pred']
        
        # Take the last n_points
        y_true = y_true[-n_points:]
        y_pred = y_pred[-n_points:]
        timestamps = timestamps[-n_points:]
        
        # Calculate confidence intervals
        lower, upper = self.calculate_confidence_intervals(y_pred, y_true, confidence)
        lower = lower[-n_points:]
        upper = upper[-n_points:]
        
        fig = go.Figure()
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            name=f'{int(confidence*100)}% CI',
            hoverinfo='skip'
        ))
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_pred,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.update_layout(
            title=f'Forecast with {int(confidence*100)}% Confidence Interval - {model_name}',
            xaxis_title='Timestamp',
            yaxis_title='Ticket Demand',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_model_comparison(self, metric: str = 'MAE') -> go.Figure:
        """
        Create a bar chart comparing models on a metric.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Plotly figure
        """
        comparison_df = self.compare_models()
        
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_df.index,
                y=comparison_df[metric],
                marker_color='steelblue'
            )
        ])
        
        fig.update_layout(
            title=f'Model Comparison - {metric}',
            xaxis_title='Model',
            yaxis_title=metric,
            template='plotly_white'
        )
        
        return fig
    
    def plot_error_distribution(self, model_name: str) -> go.Figure:
        """
        Plot the distribution of prediction errors.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not evaluated")
        
        y_true = self.predictions[model_name]['y_true']
        y_pred = self.predictions[model_name]['y_pred']
        errors = y_true - y_pred
        
        fig = go.Figure(data=[
            go.Histogram(
                x=errors,
                nbinsx=30,
                marker_color='steelblue',
                opacity=0.7
            )
        ])
        
        # Add mean line
        mean_error = np.mean(errors)
        fig.add_vline(
            x=mean_error,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Mean: {mean_error:.2f}'
        )
        
        fig.update_layout(
            title=f'Error Distribution - {model_name}',
            xaxis_title='Error (Actual - Predicted)',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        return fig
    
    def generate_evaluation_report(self, output_path: str) -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        comparison_df = self.compare_models()
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("MODEL COMPARISON TABLE\n")
            f.write("-" * 80 + "\n")
            f.write(comparison_df.to_string())
            f.write("\n\n")
            
            f.write("BEST MODELS BY METRIC\n")
            f.write("-" * 80 + "\n")
            for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
                best_model, best_value = self.get_best_model(metric)
                f.write(f"{metric}: {best_model} ({best_value:.4f})\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Evaluation report saved to {output_path}")


def calculate_horizon_wise_metrics(evaluators: Dict[str, ModelEvaluator]) -> pd.DataFrame:
    """
    Calculate metrics across different forecast horizons.
    
    Args:
        evaluators: Dictionary mapping horizon names to ModelEvaluator instances
        
    Returns:
        DataFrame with horizon-wise metrics
    """
    horizon_metrics = []
    
    for horizon, evaluator in evaluators.items():
        comparison_df = evaluator.compare_models()
        comparison_df['horizon'] = horizon
        comparison_df = comparison_df.reset_index()
        comparison_df = comparison_df.rename(columns={'index': 'model'})
        horizon_metrics.append(comparison_df)
    
    result_df = pd.concat(horizon_metrics, ignore_index=True)
    return result_df


if __name__ == "__main__":
    # Example usage
    from train import setup_default_models, ModelTrainer
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
    
    # Train models
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        evaluator.evaluate_model(model_name, y_test.values, y_pred)
    
    # Print comparison
    comparison = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison)
    
    # Get best model
    best_model, best_mae = evaluator.get_best_model('MAE')
    print(f"\nBest model by MAE: {best_model} ({best_mae:.4f})")
    
    # Generate report
    evaluator.generate_evaluation_report("reports/evaluation_report.txt")
