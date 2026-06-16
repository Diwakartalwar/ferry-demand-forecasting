"""
Streamlit Dashboard for Ferry Demand Forecasting System

A professional operations-control-center-style dashboard for:
- Live forecast KPIs
- Forecast vs actual visualization
- Confidence bands
- Peak demand alerts
- Model comparison
- Historical trends
- Prediction tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from preprocess import FerryDataPreprocessor, create_sample_data
from features import FerryFeatureEngineer
from train import setup_default_models, ModelTrainer
from evaluate import ModelEvaluator
from forecast import FerryForecaster, ForecastPipeline


# Page configuration
st.set_page_config(
    page_title="Ferry Demand Forecasting System",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eef5f9 45%, #f8fafc 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2f45 0%, #123b54 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [role="combobox"] {
        color: #111827;
    }
    .main-header {
        display: none;
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f2f45;
        margin-bottom: 0.35rem;
        letter-spacing: 0;
    }
    .main-header + p {
        display: none;
    }
    .kpi-card {
        background: linear-gradient(135deg, #14506b 0%, #1d7a8c 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        box-shadow: 0 12px 24px rgba(20, 80, 107, 0.18);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .kpi-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .alert-critical {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-high {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-medium {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-low {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(15, 47, 69, 0.08);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 8px 18px rgba(15, 47, 69, 0.07);
    }
    h2, h3 {
        color: #0f2f45;
        letter-spacing: 0;
    }
    .hero-panel {
        background:
            linear-gradient(135deg, rgba(15, 47, 69, 0.96) 0%, rgba(29, 122, 140, 0.92) 58%, rgba(232, 248, 244, 0.88) 100%);
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        color: white;
        box-shadow: 0 18px 38px rgba(15, 47, 69, 0.18);
        margin-bottom: 1.15rem;
    }
    .hero-panel h1 {
        color: white;
        margin: 0 0 0.35rem 0;
        font-size: 2.15rem;
        line-height: 1.12;
        letter-spacing: 0;
    }
    .hero-panel p {
        margin: 0;
        color: rgba(255, 255, 255, 0.86);
        font-size: 1rem;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        color: white;
        font-size: 0.86rem;
        backdrop-filter: blur(8px);
    }
    .panel {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 47, 69, 0.08);
        border-radius: 8px;
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: 0 12px 28px rgba(15, 47, 69, 0.08);
        margin-bottom: 1rem;
    }
    .section-label {
        color: #527084;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stTabs"] button {
        border-radius: 999px;
        padding: 0.45rem 0.85rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(15, 47, 69, 0.08);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(15, 47, 69, 0.06);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data():
    """Load and process ferry data."""
    # Check if sample data exists, create if not
    sample_path = "data/raw/ferry_sample_data.csv"
    sample_dir = Path(sample_path).parent
    sample_dir.mkdir(parents=True, exist_ok=True)
    if not Path(sample_path).exists():
        create_sample_data(sample_path, days=30)
    
    # Preprocess
    preprocessor = FerryDataPreprocessor()
    df = preprocessor.load_data(sample_path)
    df = preprocessor.preprocess_pipeline(df)
    
    return df


@st.cache_data
def engineer_features(df):
    """Engineer features from processed data."""
    engineer = FerryFeatureEngineer()
    df_features = engineer.create_all_features(df)
    return df_features, engineer


@st.cache_resource
def load_pretrained_models(X_train, y_train):
    """Load pre-trained models or train new ones if not available."""
    models_dir = Path(__file__).resolve().parent / "models"
    model_files = {
        'rf': models_dir / 'rf.joblib',
        'gb': models_dir / 'gb.joblib',
        'lr': models_dir / 'lr.joblib',
        'naive': models_dir / 'naive.joblib',
        'ma': models_dir / 'ma.joblib'
    }
    
    trained_models = {}
    for key, filepath in model_files.items():
        if filepath.exists():
            trained_models[key] = joblib.load(filepath)
            print(f"Loaded pre-trained model: {key}")
    
    if not trained_models:
        trainer = setup_default_models()
        trained_models = trainer.train_all_models(X_train, y_train)
        print("Trained models (pre-trained models not found)")
    
    return trained_models


def get_alert_color(alert_level):
    """Get color for alert level."""
    colors = {
        'CRITICAL': "#ff0000",
        'HIGH': '#f59e0b',
        'MEDIUM': '#3b82f6',
        'LOW': '#10b981'
    }
    return colors.get(alert_level, '#6b7280')


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">⛴️ Ferry Demand Forecasting System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280; margin-bottom: 2rem;">Toronto Island Park Operations Control Center</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-panel">
        <h1>Ferry Demand Forecasting System</h1>
        <p>Toronto Island Park Operations Control Center</p>
        <div class="hero-meta">
            <span class="hero-pill">Live demand intelligence</span>
            <span class="hero-pill">15m to 2h forecast horizons</span>
            <span class="hero-pill">Operational alerting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Controls")
    
    # Model selector
    model_options = ['Random Forest', 'Gradient Boosting', 'Linear Regression', 'Naive', 'Moving Average']
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)
    
    # Forecast horizon selector
    horizon_options = ['15m', '30m', '1h', '2h']
    selected_horizon = st.sidebar.selectbox("Forecast Horizon", horizon_options, index=2)
    
    # Confidence interval toggle
    show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)
    
    # Alert thresholds
    st.sidebar.subheader("Alert Thresholds")
    critical_threshold = st.sidebar.number_input("Critical Threshold", value=100, min_value=0)
    high_threshold = st.sidebar.number_input("High Threshold", value=75, min_value=0)
    medium_threshold = st.sidebar.number_input("Medium Threshold", value=50, min_value=0)
    
    thresholds = {
        'critical': critical_threshold,
        'high': high_threshold,
        'medium': medium_threshold,
        'low': 0
    }
    
    # Load data
    with st.spinner("Loading and processing data..."):
        df = load_and_process_data()
        df_features, engineer = engineer_features(df)
    
    # Get feature matrix for selected horizon
    X, y = engineer.get_feature_matrix(df_features, horizon=selected_horizon)
    
    # Time-based split
    trainer = setup_default_models()
    X_train, X_test, y_train, y_test = trainer.time_series_split(X, y, test_size=0.2)

    # Date range selector follows the actual test data range.
    st.sidebar.subheader("Date Range")
    data_start = X_test.index.min().date()
    data_end = X_test.index.max().date()
    default_start = max(data_start, data_end - timedelta(days=7))
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(default_start, data_end),
        min_value=data_start,
        max_value=data_end
    )
    
    # Load pre-trained models (cached)
    trained_models = load_pretrained_models(X_train, y_train)
    
    # Make predictions
    model_key_map = {
        'Random Forest': 'rf',
        'Gradient Boosting': 'gb',
        'Linear Regression': 'lr',
        'Naive': 'naive',
        'Moving Average': 'ma'
    }
    model_key = model_key_map[selected_model]
    
    if model_key in trained_models:
        model = trained_models[model_key]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model_key, y_test.values, y_pred)
    else:
        st.error(f"Model {selected_model} not available")
        return
    
    # Alert helper
    forecaster = FerryForecaster()
    
    # KPI Section
    st.header("📊 Live Forecast KPIs")
    
    metric_cols = st.columns(4)
    metric_cols[0].metric("Selected Model", selected_model)
    metric_cols[1].metric("Horizon", selected_horizon)
    metric_cols[2].metric("MAE", f"{metrics['MAE']:.2f}")
    metric_cols[3].metric("R²", f"{metrics['R2']:.3f}")

    # Calculate next forecasts
    next_forecasts = {}
    for horizon in horizon_options:
        h_X, h_y = engineer.get_feature_matrix(df_features, horizon=horizon)
        if model_key in trained_models:
            pred = trained_models[model_key].predict(h_X.tail(1))
            next_forecasts[horizon] = float(pred[0])
    
    # Display KPI cards
    kpi_cols = st.columns(4)
    for i, horizon in enumerate(horizon_options):
        forecast_val = next_forecasts.get(horizon, 0)
        alert_level = forecaster.generate_operational_alert(forecast_val, thresholds)
        alert_color = get_alert_color(alert_level)
        
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, {alert_color} 0%, {alert_color}99 100%);">
                <div class="kpi-label">Next {horizon} Forecast</div>
                <div class="kpi-value">{forecast_val:.1f}</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">Alert: {alert_level}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Forecast vs Actual Visualization
    st.header("📈 Forecast vs Actual")
    
    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (X_test.index >= pd.Timestamp(start_date)) & (X_test.index <= pd.Timestamp(end_date))
        X_test_filtered = X_test[mask]
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        if len(y_test_filtered) == 0:
            st.info("Selected dates have no test records. Showing full test range.")
            X_test_filtered = X_test
            y_test_filtered = y_test
            y_pred_filtered = y_pred
    else:
        X_test_filtered = X_test
        y_test_filtered = y_test
        y_pred_filtered = y_pred
    
    # Create forecast chart
    fig_forecast = go.Figure()
    
    # Add actual values
    fig_forecast.add_trace(go.Scatter(
        x=y_test_filtered.index,
        y=y_test_filtered.values,
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add predicted values
    fig_forecast.add_trace(go.Scatter(
        x=y_test_filtered.index,
        y=y_pred_filtered,
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Add confidence intervals if enabled
    if show_confidence:
        lower, upper = evaluator.calculate_confidence_intervals(
            y_pred_filtered, y_test_filtered.values, confidence=0.95
        )
        
        fig_forecast.add_trace(go.Scatter(
            x=y_test_filtered.index,
            y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=y_test_filtered.index,
            y=lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            name='95% CI',
            hoverinfo='skip'
        ))
    
    fig_forecast.update_layout(
        title=f'{selected_model} - {selected_horizon} Horizon',
        xaxis_title='Timestamp',
        yaxis_title='Ticket Demand',
        hovermode='x unified',
        template='plotly_white',
        height=430,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.82)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=70, b=20)
    )
    fig_forecast.update_xaxes(showgrid=False)
    fig_forecast.update_yaxes(gridcolor='rgba(15, 47, 69, 0.08)')
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Peak Demand Alerts
    st.header("🚨 Peak Demand Alerts")
    
    # Calculate alerts for recent forecasts
    recent_forecasts = y_pred_filtered[-10:]
    recent_alerts = [forecaster.generate_operational_alert(f, thresholds) for f in recent_forecasts]
    recent_timestamps = y_test_filtered.index[-10:]
    
    alert_df = pd.DataFrame({
        'Timestamp': recent_timestamps,
        'Forecast': recent_forecasts,
        'Alert Level': recent_alerts
    })
    
    # Display alerts
    for _, row in alert_df.iterrows():
        alert_class = f"alert-{row['Alert Level'].lower()}"
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>{row['Timestamp'].strftime('%Y-%m-%d %H:%M')}</strong> - 
            Forecast: {row['Forecast']:.1f} - 
            <strong>{row['Alert Level']}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.header("Analytics Workbench")

    comparison_metrics = {}
    for m_key, m_model in trained_models.items():
        try:
            m_pred = m_model.predict(X_test)
            m_evaluator = ModelEvaluator()
            comparison_metrics[m_key] = m_evaluator.evaluate_model(m_key, y_test.values, m_pred)
        except Exception:
            pass

    comparison_df = pd.DataFrame(comparison_metrics).T.round(4)
    if comparison_df.empty:
        comparison_df = pd.DataFrame(columns=['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2'])

    daily_demand = df['Sales Count'].resample('D').sum()
    df['hour'] = df.index.hour
    hourly_demand = df.groupby('hour')['Sales Count'].mean()

    pred_table_df = pd.DataFrame({
        'Timestamp': y_test_filtered.index,
        'Actual': y_test_filtered.values,
        'Forecast': y_pred_filtered,
        'Error': y_test_filtered.values - y_pred_filtered,
        'Absolute Error': np.abs(y_test_filtered.values - y_pred_filtered)
    })

    if show_confidence:
        pred_table_df['Lower Bound'] = lower
        pred_table_df['Upper Bound'] = upper

    pred_table_df = pred_table_df.round(2)

    benchmark_tab, trends_tab, table_tab = st.tabs([
        "Model Benchmarks",
        "Demand Patterns",
        "Prediction Ledger"
    ])

    with benchmark_tab:
        st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)
        st.dataframe(
            comparison_df,
            use_container_width=True,
            column_config={
                "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
                "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
                "SMAPE": st.column_config.NumberColumn("SMAPE", format="%.2f%%"),
                "R2": st.column_config.NumberColumn("R²", format="%.3f"),
            }
        )

        fig_comparison = go.Figure(data=[
            go.Bar(
                x=comparison_df.index,
                y=comparison_df['MAE'],
                marker=dict(color=comparison_df['MAE'], colorscale='Tealgrn', line=dict(width=0)),
                text=comparison_df['MAE'].round(2),
                textposition='outside'
            )
        ])
        fig_comparison.update_layout(
            title='Model Comparison - MAE',
            xaxis_title='Model',
            yaxis_title='MAE',
            template='plotly_white',
            height=330,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.85)',
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )
        fig_comparison.update_xaxes(showgrid=False)
        fig_comparison.update_yaxes(gridcolor='rgba(15, 47, 69, 0.08)')
        st.plotly_chart(fig_comparison, use_container_width=True)

    with trends_tab:
        trend_cols = st.columns(2)

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_demand.index,
            y=daily_demand.values,
            mode='lines',
            name='Daily Demand',
            fill='tozeroy',
            line=dict(color='#14506b', width=2.6)
        ))
        fig_daily.update_layout(
            title='Daily Ticket Demand',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            template='plotly_white',
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.85)',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig_daily.update_xaxes(showgrid=False)
        fig_daily.update_yaxes(gridcolor='rgba(15, 47, 69, 0.08)')

        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly_demand.index,
            y=hourly_demand.values,
            marker=dict(color=hourly_demand.values, colorscale='Blugrn', line=dict(width=0))
        ))
        fig_hourly.update_layout(
            title='Average Hourly Demand Pattern',
            xaxis_title='Hour of Day',
            yaxis_title='Average Sales',
            template='plotly_white',
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.85)',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig_hourly.update_xaxes(showgrid=False)
        fig_hourly.update_yaxes(gridcolor='rgba(15, 47, 69, 0.08)')

        trend_cols[0].plotly_chart(fig_daily, use_container_width=True)
        trend_cols[1].plotly_chart(fig_hourly, use_container_width=True)

    with table_tab:
        st.markdown('<div class="section-label">Latest forecast rows</div>', unsafe_allow_html=True)
        st.dataframe(
            pred_table_df.tail(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "Actual": st.column_config.NumberColumn("Actual", format="%.2f"),
                "Forecast": st.column_config.NumberColumn("Forecast", format="%.2f"),
                "Error": st.column_config.NumberColumn("Error", format="%.2f"),
                "Absolute Error": st.column_config.NumberColumn("Abs Error", format="%.2f"),
                "Lower Bound": st.column_config.NumberColumn("Lower", format="%.2f"),
                "Upper Bound": st.column_config.NumberColumn("Upper", format="%.2f"),
            }
        )

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #527084; padding: 1rem;">
        <p>Ferry Demand Forecasting & Predictive Decision Support System</p>
        <p style="font-size: 0.8rem;">Toronto Island Park Operations</p>
    </div>
    """, unsafe_allow_html=True)
    return

    # Model Comparison Panel
    st.header("🏆 Model Comparison")
    
    # Evaluate all models
    comparison_metrics = {}
    for m_key, m_model in trained_models.items():
        try:
            m_pred = m_model.predict(X_test)
            m_evaluator = ModelEvaluator()
            m_metrics = m_evaluator.evaluate_model(m_key, y_test.values, m_pred)
            comparison_metrics[m_key] = m_metrics
        except:
            pass

    comparison_df = pd.DataFrame(comparison_metrics).T
    comparison_df = comparison_df.round(4)
    if comparison_df.empty:
        comparison_df = pd.DataFrame(columns=['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2'])
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Model comparison chart
    fig_comparison = go.Figure(data=[
        go.Bar(
            x=comparison_df.index,
            y=comparison_df['MAE'],
            marker_color='steelblue'
        )
    ])
    
    fig_comparison.update_layout(
        title='Model Comparison - MAE',
        xaxis_title='Model',
        yaxis_title='MAE',
        template='plotly_white',
        height=300
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Historical Trends
    st.header("📅 Historical Trends")
    
    # Daily demand
    daily_demand = df['Sales Count'].resample('D').sum()
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=daily_demand.index,
        y=daily_demand.values,
        mode='lines+markers',
        name='Daily Demand',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig_daily.update_layout(
        title='Daily Ticket Demand',
        xaxis_title='Date',
        yaxis_title='Total Sales',
        template='plotly_white',
        height=300
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Weekly seasonality
    df['hour'] = df.index.hour
    hourly_demand = df.groupby('hour')['Sales Count'].mean()
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_demand.index,
        y=hourly_demand.values,
        marker_color='steelblue'
    ))
    
    fig_hourly.update_layout(
        title='Average Hourly Demand Pattern',
        xaxis_title='Hour of Day',
        yaxis_title='Average Sales',
        template='plotly_white',
        height=300
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Prediction Table
    st.header("📋 Prediction Table")
    
    pred_table_df = pd.DataFrame({
        'Timestamp': y_test_filtered.index,
        'Actual': y_test_filtered.values,
        'Forecast': y_pred_filtered,
        'Error': y_test_filtered.values - y_pred_filtered,
        'Absolute Error': np.abs(y_test_filtered.values - y_pred_filtered)
    })
    
    if show_confidence:
        pred_table_df['Lower Bound'] = lower
        pred_table_df['Upper Bound'] = upper
    
    pred_table_df = pred_table_df.round(2)
    st.dataframe(pred_table_df.tail(20), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p>Ferry Demand Forecasting & Predictive Decision Support System</p>
        <p style="font-size: 0.8rem;">Toronto Island Park Operations</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
