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

# Custom CSS for a darker, more operational look
st.markdown("""
<style>
    :root {
        --bg-0: #050b14;
        --bg-1: #09111d;
        --bg-2: #101b2b;
        --surface: rgba(11, 18, 29, 0.92);
        --surface-strong: rgba(15, 25, 40, 0.98);
        --line: rgba(148, 163, 184, 0.14);
        --line-strong: rgba(56, 189, 248, 0.22);
        --text: #e5eefc;
        --muted: #94a3b8;
        --accent: #38bdf8;
        --accent-2: #22c55e;
        --warning: #f59e0b;
        --danger: #f87171;
        --shadow: 0 20px 45px rgba(0, 0, 0, 0.28);
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .block-container {
        background: transparent !important;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 27%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.09), transparent 24%),
            linear-gradient(180deg, #050b14 0%, #09111d 52%, #05080f 100%);
        color: var(--text);
    }

    body {
        color: var(--text);
        font-family: "Segoe UI", "Inter", "Aptos", sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050c15 0%, #081523 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #e5eefc !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [role="combobox"] {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] input[type="text"],
    [data-testid="stSidebar"] input[type="date"] {
        background: rgba(15, 23, 42, 0.96) !important;
        color: #f8fafc !important;
        border-color: rgba(148, 163, 184, 0.18) !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #cbd5e1 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
    [data-testid="stSidebar"] [data-baseweb="input"] > div:hover {
        border-color: rgba(56, 189, 248, 0.42) !important;
    }

    .main-header {
        display: none;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.35rem;
        letter-spacing: 0;
    }

    .main-header + p {
        display: none;
    }

    .hero-panel {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(5, 12, 20, 0.98) 0%, rgba(10, 22, 35, 0.96) 55%, rgba(11, 60, 88, 0.94) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        color: white;
        box-shadow: var(--shadow);
        margin-bottom: 1.1rem;
    }

    .hero-panel::after {
        content: "";
        position: absolute;
        inset: auto -8% -42% auto;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.22), transparent 66%);
        pointer-events: none;
    }

    .hero-panel h1 {
        position: relative;
        z-index: 1;
        color: white;
        margin: 0 0 0.35rem 0;
        font-size: 2.2rem;
        line-height: 1.08;
        letter-spacing: -0.02em;
    }

    .hero-panel p {
        position: relative;
        z-index: 1;
        margin: 0;
        color: rgba(226, 232, 240, 0.88);
        font-size: 1rem;
    }

    .hero-meta {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .hero-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.42rem 0.78rem;
        color: white;
        font-size: 0.86rem;
        backdrop-filter: blur(8px);
    }

    .kpi-card,
    .panel {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1rem 0.85rem 1rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .kpi-card {
        min-height: 128px;
    }

    .kpi-card strong,
    .kpi-value,
    .kpi-label,
    .kpi-detail {
        position: relative;
        z-index: 1;
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), transparent 55%);
        pointer-events: none;
    }

    .kpi-label {
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.45rem;
    }

    .kpi-value {
        font-size: 2.05rem;
        line-height: 1.05;
        font-weight: 750;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }

    .kpi-detail {
        font-size: 0.92rem;
        color: #cbd5e1;
    }

    .alert-critical {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.34), rgba(69, 10, 10, 0.42));
        border: 1px solid rgba(248, 113, 113, 0.25);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 12px;
        color: #fee2e2;
        margin-bottom: 0.65rem;
    }

    .alert-high {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.34), rgba(69, 26, 3, 0.42));
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 12px;
        color: #ffedd5;
        margin-bottom: 0.65rem;
    }

    .alert-medium {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.24), rgba(12, 44, 110, 0.34));
        border: 1px solid rgba(59, 130, 246, 0.22);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 12px;
        color: #dbeafe;
        margin-bottom: 0.65rem;
    }

    .alert-low {
        background: linear-gradient(135deg, rgba(6, 95, 70, 0.28), rgba(6, 78, 59, 0.35));
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 12px;
        color: #d1fae5;
        margin-bottom: 0.65rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(14, 23, 37, 0.96), rgba(8, 14, 24, 0.98));
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: var(--shadow);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc;
        letter-spacing: -0.01em;
    }

    .section-label {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px;
        padding: 0.45rem 0.95rem;
        background: rgba(15, 23, 42, 0.7);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.14);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: white;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(34, 197, 94, 0.12));
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    div[data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    div[data-testid="stDataFrame"] div[role="grid"],
    div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] {
        background: transparent !important;
    }

    div[data-testid="stDataFrame"] div[role="grid"] * {
        color: #e5eefc;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.14);
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


MODEL_DISPLAY_NAMES = {
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
    'lr': 'Linear Regression',
    'naive': 'Naive',
    'ma': 'Moving Average',
    'xgb': 'XGBoost',
    'arima': 'ARIMA'
}


def format_model_name(model_key: str) -> str:
    """Convert an internal model key to a friendly label."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace('_', ' ').title())


def render_kpi_card(label: str, value: str, detail: str, accent: str = "#38bdf8") -> None:
    """Render a single dashboard KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {accent};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


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

    # Date range selector follows the full feature timeline.
    st.sidebar.subheader("Date Range")
    data_start = X.index.min()
    data_end = X.index.max()
    start_year = int(data_start.year)
    end_year = int(data_end.year)
    st.sidebar.caption(f"Full range: {data_start.date()} to {data_end.date()}")

    year_range = st.sidebar.slider(
        "Year Range",
        min_value=start_year,
        max_value=end_year,
        value=(start_year, end_year),
        step=1
    )
    month_range = st.sidebar.slider(
        "Month Range",
        min_value=1,
        max_value=12,
        value=(1, 12),
        step=1
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

    if model_key not in trained_models:
        st.error(f"Model {selected_model} not available")
        return

    model = trained_models[model_key]
    y_pred_test = model.predict(X_test)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model_key, y_test.values, y_pred_test)

    comparison_metrics = {}
    for m_key, m_model in trained_models.items():
        try:
            m_pred = m_model.predict(X_test)
            m_evaluator = ModelEvaluator()
            comparison_metrics[m_key] = m_evaluator.evaluate_model(m_key, y_test.values, m_pred)
        except Exception:
            pass

    comparison_df = pd.DataFrame(comparison_metrics).T.round(4)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("MAE")
        comparison_df.index = [format_model_name(idx) for idx in comparison_df.index]
        comparison_df.insert(0, "Rank", np.arange(1, len(comparison_df) + 1))
    else:
        comparison_df = pd.DataFrame(columns=['Rank', 'MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2'])

    best_model_name = comparison_df.index[0] if not comparison_df.empty else "N/A"
    best_model_mae = float(comparison_df.iloc[0]["MAE"]) if not comparison_df.empty else float("nan")

    forecaster = FerryForecaster()

    y_pred_all = model.predict(X)
    y_actual_all = y

    year_start, year_end = year_range
    month_start, month_end = month_range
    mask = (
        (X.index.year >= year_start) &
        (X.index.year <= year_end) &
        (X.index.month >= month_start) &
        (X.index.month <= month_end)
    )
    X_view = X[mask]
    y_view = y_actual_all[mask]
    y_pred_view = y_pred_all[mask]

    if len(y_view) == 0:
        st.warning("No records match that year/month filter. Showing the full timeline instead.")
        X_view = X
        y_view = y_actual_all
        y_pred_view = y_pred_all

    latest_forecast = float(y_pred_all[-1]) if len(y_pred_all) else 0.0
    latest_alert = forecaster.generate_operational_alert(latest_forecast, thresholds)

    st.caption(
        f"Best model on the current test split: {best_model_name} "
        f"({best_model_mae:.2f} MAE). Filter is Year {year_start}-{year_end}, Month {month_start}-{month_end}."
    )

    # KPI Section
    st.header("📊 Live Forecast KPIs")

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_kpi_card("Selected Model", selected_model, "Current dashboard view", "#38bdf8")
    with summary_cols[1]:
        render_kpi_card("Horizon", selected_horizon, "Forecast window", "#22c55e")
    with summary_cols[2]:
        render_kpi_card("Test MAE", f"{metrics['MAE']:.2f}", "Lower is better", "#f59e0b")
    with summary_cols[3]:
        render_kpi_card("Test R²", f"{metrics['R2']:.3f}", "Higher is better", "#a78bfa")

    # Calculate next forecasts
    next_forecasts = {}
    for horizon in horizon_options:
        h_X, _ = engineer.get_feature_matrix(df_features, horizon=horizon)
        pred = trained_models[model_key].predict(h_X.tail(1))
        next_forecasts[horizon] = float(pred[0])

    forecast_cols = st.columns(4)
    for i, horizon in enumerate(horizon_options):
        forecast_val = next_forecasts.get(horizon, 0)
        alert_level = forecaster.generate_operational_alert(forecast_val, thresholds)
        alert_color = get_alert_color(alert_level)
        with forecast_cols[i]:
            render_kpi_card(
                f"Next {horizon}",
                f"{forecast_val:.1f}",
                f"Alert: {alert_level}",
                alert_color
            )

    st.markdown('<div class="section-label">Forecast narrative</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel">
            <strong>Current read:</strong> The selected model predicts <strong>{latest_forecast:.1f}</strong>
            on the latest filtered point, which sits in <strong>{latest_alert}</strong> alert territory.
            Use the benchmark table below to compare model quality and the ledger to inspect individual rows.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Forecast vs Actual Visualization
    st.header("📈 Forecast vs Actual")

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=y_view.index,
        y=y_view.values,
        mode='lines',
        name='Actual',
        line=dict(color='#38bdf8', width=2.6)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=y_view.index,
        y=y_pred_view,
        mode='lines',
        name='Forecast',
        line=dict(color='#f59e0b', width=2.6, dash='dash')
    ))

    if show_confidence:
        lower, upper = evaluator.calculate_confidence_intervals(
            y_pred_view, y_view.values, confidence=0.95
        )
        fig_forecast.add_trace(go.Scatter(
            x=y_view.index,
            y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_forecast.add_trace(go.Scatter(
            x=y_view.index,
            y=lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(245, 158, 11, 0.16)',
            name='95% CI',
            hoverinfo='skip'
        ))

    fig_forecast.update_layout(
        title=f'{selected_model} - {selected_horizon} Horizon',
        xaxis_title='Timestamp',
        yaxis_title='Ticket Demand',
        hovermode='x unified',
        template='plotly_dark',
        height=430,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(7, 12, 20, 0.96)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=70, b=20),
        font=dict(color='#e5eefc')
    )
    fig_forecast.update_xaxes(showgrid=False, linecolor='rgba(148, 163, 184, 0.14)')
    fig_forecast.update_yaxes(gridcolor='rgba(148, 163, 184, 0.12)', linecolor='rgba(148, 163, 184, 0.14)')
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Peak Demand Alerts
    st.header("🚨 Peak Demand Alerts")

    recent_forecasts = y_pred_view[-10:]
    recent_alerts = [forecaster.generate_operational_alert(f, thresholds) for f in recent_forecasts]
    recent_timestamps = y_view.index[-10:]

    alert_df = pd.DataFrame({
        'Timestamp': recent_timestamps,
        'Forecast': recent_forecasts,
        'Alert Level': recent_alerts
    })

    alert_summary = alert_df['Alert Level'].value_counts().reindex(
        ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
        fill_value=0
    )
    alert_summary_cols = st.columns(4)
    for idx, level in enumerate(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']):
        with alert_summary_cols[idx]:
            render_kpi_card(level.title(), str(int(alert_summary[level])), "Recent window", get_alert_color(level))

    for _, row in alert_df.sort_values("Timestamp", ascending=False).iterrows():
        alert_class = f"alert-{row['Alert Level'].lower()}"
        st.markdown(
            f"""
            <div class="{alert_class}">
                <strong>{row['Timestamp'].strftime('%Y-%m-%d %H:%M')}</strong> ·
                Forecast {row['Forecast']:.1f} ·
                <strong>{row['Alert Level']}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    daily_demand = df['Sales Count'].resample('D').sum()
    hourly_demand = df.groupby(df.index.hour)['Sales Count'].mean()

    pred_table_df = pd.DataFrame({
        'Timestamp': y_view.index,
        'Actual': y_view.values,
        'Forecast': y_pred_view,
        'Error': y_view.values - y_pred_view,
        'Absolute Error': np.abs(y_view.values - y_pred_view)
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
        st.markdown(f"<div class=\"panel\">Current best model: <strong>{best_model_name}</strong></div>", unsafe_allow_html=True)

        comparison_df_display = comparison_df.reset_index().rename(columns={"index": "Model"})

        st.dataframe(
            comparison_df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Model": st.column_config.TextColumn("Model"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
                "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
                "SMAPE": st.column_config.NumberColumn("SMAPE", format="%.2f%%"),
                "R2": st.column_config.NumberColumn("R²", format="%.3f"),
            }
        )

        fig_comparison = go.Figure(data=[
            go.Bar(
                x=comparison_df.index if not comparison_df.empty else [],
                y=comparison_df['MAE'] if not comparison_df.empty else [],
                marker=dict(
                    color=comparison_df['MAE'] if not comparison_df.empty else [],
                    colorscale='Tealgrn',
                    line=dict(width=0)
                ),
                text=comparison_df['MAE'].round(2) if not comparison_df.empty else [],
                textposition='outside'
            )
        ])
        fig_comparison.update_layout(
            title='Model Comparison - MAE',
            xaxis_title='Model',
            yaxis_title='MAE',
            template='plotly_dark',
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(7, 12, 20, 0.96)',
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
            font=dict(color='#e5eefc')
        )
        fig_comparison.update_xaxes(showgrid=False)
        fig_comparison.update_yaxes(gridcolor='rgba(148, 163, 184, 0.12)')
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
            line=dict(color='#38bdf8', width=2.6)
        ))
        fig_daily.update_layout(
            title='Daily Ticket Demand',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            template='plotly_dark',
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(7, 12, 20, 0.96)',
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(color='#e5eefc')
        )
        fig_daily.update_xaxes(showgrid=False)
        fig_daily.update_yaxes(gridcolor='rgba(148, 163, 184, 0.12)')

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
            template='plotly_dark',
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(7, 12, 20, 0.96)',
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(color='#e5eefc')
        )
        fig_hourly.update_xaxes(showgrid=False)
        fig_hourly.update_yaxes(gridcolor='rgba(148, 163, 184, 0.12)')

        trend_cols[0].plotly_chart(fig_daily, use_container_width=True)
        trend_cols[1].plotly_chart(fig_hourly, use_container_width=True)

        hourly_table = hourly_demand.reset_index()
        hourly_table.columns = ["Hour", "Average Sales"]
        st.markdown('<div class="section-label">Hourly demand table</div>', unsafe_allow_html=True)
        st.dataframe(
            hourly_table.sort_values("Average Sales", ascending=False).head(6),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hour": st.column_config.NumberColumn("Hour", format="%d"),
                "Average Sales": st.column_config.NumberColumn("Average Sales", format="%.2f"),
            }
        )

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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 1rem;">
        <p>Ferry Demand Forecasting & Predictive Decision Support System</p>
        <p style="font-size: 0.8rem;">Toronto Island Park Operations</p>
    </div>
    """, unsafe_allow_html=True)
    return


if __name__ == "__main__":
    main()
