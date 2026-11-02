"""
AI-Driven Forecasting - Model Training Script
Module 5: AIOps Foundations

This script trains Prophet time-series forecasting models on historical
metrics from Prometheus, with seasonality disabled for the short lab history.
This demonstrates the forecasting pipeline before collecting longer history.

Usage (from the project root, with the virtual environment activated):
    python monitoring/scripts/train_forecasting_model.py
"""

import sys
from datetime import datetime, timedelta, timezone
import logging

# Keep backend progress logs quiet, but retain useful Python warnings.
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

try:
    import pandas as pd
    import numpy as np
    from prometheus_api_client import PrometheusConnect

    # Import Prophet and let it initialize cmdstan
    from prophet import Prophet
    from prophet.serialize import model_to_json

except ImportError as e:
    print(f"❌ Error: Missing required package: {e}")
    print("\n💡 Please install packages:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


from forecasting_config import METRICS_CONFIG, MODEL_DIR, PROMETHEUS_URL, TRAINING_STEP

TRAINING_HOURS = 1  # Use the last hour, or all available data within it.
MIN_DATA_POINTS = 20  # About ten minutes at a 30-second query step.


def print_header():
    """Print script header."""
    print("\n" + "="*70)
    print("🤖 AI-Driven Forecasting - Model Training")
    print("Module 5: AIOps Foundations")
    print("="*70 + "\n")


def connect_to_prometheus():
    """Connect to Prometheus API."""
    try:
        print(f"🔌 Connecting to Prometheus at {PROMETHEUS_URL}...")
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

        # Test connection
        if not prom.check_prometheus_connection():
            raise ConnectionError("Prometheus returned an unsuccessful response")
        print("✅ Connected to Prometheus successfully\n")
        return prom
    except Exception as e:
        print(f"❌ Error connecting to Prometheus: {e}")
        print("💡 Make sure Prometheus is running: docker compose ps")
        sys.exit(1)


def fetch_metric_data(prom, metric_name, query, hours=1):
    """Fetch historical metric data from Prometheus."""
    try:
        print(f"📊 Fetching {metric_name} data...")

        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        # Query at 30-second resolution for this short lab history.
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=TRAINING_STEP
        )

        if not result or len(result) == 0:
            print(f"⚠️  No data returned for {metric_name}")
            return None

        if len(result) != 1:
            raise ValueError("Expected one aggregate series; check the metric query")

        # Extract values
        values = result[0]['values']

        if len(values) < MIN_DATA_POINTS:
            print(f"⚠️  Insufficient data for {metric_name}: {len(values)} points")
            print(f"    (Need at least {MIN_DATA_POINTS} points)")
            print(f"    💡 Tip: Wait longer or check Prometheus is collecting metrics")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(values, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_localize(None)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Discard invalid observations, including infinities, and order in UTC.
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)

        if len(df) < MIN_DATA_POINTS:
            print(f"⚠️  Insufficient valid data after cleaning: {len(df)} points")
            return None

        print(f"✅ Fetched {len(df)} data points for {metric_name}")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Current value: {df['value'].iloc[-1]:.2f}")
        print(f"   Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")

        data_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        print(f"   Available history: {data_hours:.2f} hours (UTC)")
        print("   Lab model: seasonality disabled; this run tests the pipeline only.")
        print("   Forecast accuracy and capacity dates require longer history and backtesting.\n")

        return df

    except Exception as e:
        print(f"❌ Error fetching {metric_name}: {e}")
        return None


def prepare_prophet_data(df, metric_config, metric_key):
    """Convert DataFrame to Prophet format (ds, y) with optional cap/floor."""
    prophet_df = pd.DataFrame({
        'ds': df['timestamp'],
        'y': df['value']
    })

    # For percentage metrics, set cap and floor to prevent impossible values
    if metric_config.get('unit') == '%':
        prophet_df['cap'] = 100.0  # CPU/Disk can't exceed 100%

        prophet_df['floor'] = 0.0  # Idle CPU may legitimately be below 1%.

    return prophet_df


def train_prophet_model(df, metric_name, metric_config, metric_key):
    """Train Prophet model on historical data."""
    try:
        print(f"\n🧠 Training forecasting model for {metric_name}...")

        # Prepare data for Prophet
        prophet_df = prepare_prophet_data(df, metric_config, metric_key)

        # Determine growth type based on metric
        # For percentage metrics (CPU, Disk), use logistic growth with bounds
        # For unbounded metrics (Memory GB), use linear growth
        is_percentage = metric_config.get('unit') == '%'
        growth_type = 'logistic' if is_percentage else 'linear'

        # Initialize Prophet model
        # For lab environment with limited data (1 hour), use conservative settings
        # Allow minimal trend detection for educational purposes (see slight changes)
        model = Prophet(
            growth=growth_type,          # Logistic for %, linear for others
            daily_seasonality=False,     # Disable (need 2+ days of data)
            weekly_seasonality=False,    # Disable (need 2+ weeks of data)
            yearly_seasonality=False,    # Disable
            n_changepoints=3,            # Minimal changepoints (detect slight trends)
            changepoint_prior_scale=0.001,  # Very restrictive (prevent wild extrapolation)
            interval_width=0.95,         # 95% confidence interval
            mcmc_samples=0,              # Disable MCMC (use MAP estimation, much lower memory)
            uncertainty_samples=100      # Reduce uncertainty samples (default 1000)
        )

        # Train the model
        print(f"   Training on {len(prophet_df)} data points...")

        # A backend failure must fail this metric, never continue with an unfitted model.
        model.fit(prophet_df)
        if not model.params or 'k' not in model.params:
            raise RuntimeError("Prophet returned no fitted parameters")

        print(f"✅ Model trained successfully for {metric_name}")

        # Analyze learned components
        data_mean = prophet_df['y'].mean()
        data_std = prophet_df['y'].std()
        data_min = prophet_df['y'].min()
        data_max = prophet_df['y'].max()

        # Calculate simple trend
        data_start = prophet_df['y'].iloc[0]
        data_end = prophet_df['y'].iloc[-1]
        trend = data_end - data_start

        print(f"   Learned baseline: mean={data_mean:.2f}, std={data_std:.2f}")
        print(f"   Value range: [{data_min:.2f}, {data_max:.2f}]")
        print(f"   Observed trend: {trend:+.2f} (from {data_start:.2f} to {data_end:.2f})")
        if is_percentage:
            print(f"   Model type: Logistic growth (bounded 0-100%, prevents impossible values)")
        else:
            print(f"   Model type: Linear growth (minimal, restricted by changepoint_prior=0.001)")

        return model

    except Exception as e:
        print(f"❌ Error training model for {metric_name}: {e}")
        return None


def save_model(model, metric_key):
    """Save trained model to disk."""
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Prophet's JSON format excludes the platform-specific Stan backend.
        model_path = MODEL_DIR / f"{metric_key}_forecast_model.json"
        model_path.write_text(model_to_json(model), encoding='utf-8')

        print(f"💾 Model saved: {model_path}")
        return model_path

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None


def main():
    """Main training workflow."""
    print_header()

    # Connect to Prometheus
    prom = connect_to_prometheus()

    # Track trained models
    trained_models = {}
    failed_metrics = []

    # Train model for each metric
    for metric_key, config in METRICS_CONFIG.items():
        print("\n" + "-"*70)

        # Fetch data
        df = fetch_metric_data(
            prom,
            config['name'],
            config['query'],
            hours=TRAINING_HOURS  # Use last 1 hour (or all available data)
        )

        if df is None:
            failed_metrics.append(metric_key)
            continue

        # Train model
        model = train_prophet_model(df, config['name'], config, metric_key)

        if model is None:
            failed_metrics.append(metric_key)
            continue

        # Save model
        model_path = save_model(model, metric_key)

        if model_path:
            trained_models[metric_key] = {
                'model_path': model_path,
                'name': config['name'],
                'data_points': len(df)
            }
        else:
            failed_metrics.append(metric_key)

    # Print summary
    print("\n" + "="*70)
    print("📊 Training Summary")
    print("="*70)

    if trained_models:
        print(f"\n✅ Successfully trained {len(trained_models)} model(s):\n")
        for metric_key, info in trained_models.items():
            print(f"   • {info['name']}")
            print(f"     Model: {info['model_path']}")
            print(f"     Training data: {info['data_points']} points")

    if failed_metrics:
        print(f"\n⚠️  Failed to train {len(failed_metrics)} model(s):")
        for metric_key in failed_metrics:
            print(f"   • {METRICS_CONFIG[metric_key]['name']}")
        print(f"\n💡 Common issues:")
        print(f"   - Insufficient historical data (need {MIN_DATA_POINTS}+ points)")
        print(f"   - Prometheus not collecting metrics")
        print(f"   - Data quality issues (all NaN values)")
        print(f"\n   Try waiting longer or check: docker compose logs prometheus")

    if trained_models:
        print("\n✅ Models are ready for forecasting!")
        print("\n🚀 Next step: Generate forecasts")
        print("   python monitoring/scripts/forecasting_model.py --days 7")
    else:
        print("\n❌ No models were trained successfully")
        print("   Please resolve the issues above and try again")
        sys.exit(1)

    print("\n" + "="*70 + "\n")
    return 1 if failed_metrics else 0


if __name__ == "__main__":
    sys.exit(main())
