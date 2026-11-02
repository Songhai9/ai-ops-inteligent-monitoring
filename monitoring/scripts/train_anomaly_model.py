"""
AI-Powered Anomaly Detection - Model Training Script
Module 4: AIOps Foundations

This script trains an IsolationForest model on CPU metrics from Prometheus.
It learns what "normal" CPU behavior looks like and saves the model for later use.

Usage: Activate venv first, then run:
    source /root/monitoring/scripts/venv/bin/activate
    python3 /root/monitoring/scripts/train_anomaly_model.py
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from prometheus_api_client import PrometheusConnect

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
PROMETHEUS_URL = "http://localhost:9090"
MODEL_PATH = "/root/monitoring/anomaly_model.pkl"
METRICS_QUERY = '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'

# Training parameters
TRAINING_HOURS = 1  # Use last 1 hour of data for training (or all available data)
CONTAMINATION = 0.1  # Expected proportion of outliers (10%)
MIN_SAMPLES_REQUIRED = 15  # Minimum samples needed for training

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")

def fetch_cpu_metrics(prom: PrometheusConnect, hours: int) -> pd.DataFrame:
    """
    Fetch CPU usage metrics from Prometheus

    Args:
        prom: Prometheus connection
        hours: Number of hours of historical data to fetch

    Returns:
        DataFrame with timestamp and cpu_usage columns
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    print(f"   Querying Prometheus from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Query Prometheus
    result = prom.custom_query_range(
        query=METRICS_QUERY,
        start_time=start_time,
        end_time=end_time,
        step='10s'  # 10-second resolution (matches Prometheus scrape_interval)
    )

    if not result:
        raise ValueError("No data returned from Prometheus. Ensure Node Exporter is running.")

    # Parse results into DataFrame
    timestamps = []
    values = []

    for sample in result[0]['values']:
        timestamps.append(datetime.fromtimestamp(sample[0]))
        values.append(float(sample[1]))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': values
    })

    print(f"   ✓ Fetched {len(df)} data points")

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for machine learning

    Args:
        df: DataFrame with cpu_usage column

    Returns:
        DataFrame with engineered features
    """
    print("   Creating features:")
    print("     - Rolling mean (5 samples)")
    print("     - Rolling std (5 samples)")
    print("     - Rate of change")

    # Rolling statistics (last 5 samples = 50 seconds at 10s resolution)
    df['rolling_mean'] = df['cpu_usage'].rolling(window=5, min_periods=1).mean()
    df['rolling_std'] = df['cpu_usage'].rolling(window=5, min_periods=1).std().fillna(0)

    # Rate of change
    df['rate_of_change'] = df['cpu_usage'].diff().fillna(0)

    # Hour of day (for seasonality)
    df['hour'] = df['timestamp'].dt.hour

    # Drop NaN values
    df = df.dropna()

    print(f"   ✓ Created {len(df.columns) - 2} features")  # -2 for timestamp and cpu_usage

    return df

def train_model(df: pd.DataFrame, contamination: float) -> IsolationForest:
    """
    Train IsolationForest model

    Args:
        df: DataFrame with features
        contamination: Expected proportion of outliers

    Returns:
        Trained IsolationForest model
    """
    # Select features for training
    feature_columns = ['cpu_usage', 'rolling_mean', 'rolling_std', 'rate_of_change', 'hour']
    X = df[feature_columns]

    print(f"   Training on {len(X)} samples with {len(feature_columns)} features")
    print(f"   Contamination: {contamination*100}% (expected outlier proportion)")

    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        verbose=0
    )

    model.fit(X)

    # Calculate training statistics
    predictions = model.predict(X)
    anomalies = (predictions == -1).sum()
    normal = (predictions == 1).sum()

    print(f"   ✓ Model trained successfully")
    print(f"   Training results:")
    print(f"     - Normal samples: {normal} ({normal/len(X)*100:.1f}%)")
    print(f"     - Anomalies detected: {anomalies} ({anomalies/len(X)*100:.1f}%)")

    return model

def save_model(model: IsolationForest, path: str):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"   ✓ Model saved to: {path}")
    print(f"   Model size: {os.path.getsize(path) / 1024:.2f} KB")

def main():
    """Main training workflow"""
    print_header("🤖 AI-Powered Anomaly Detection - Model Training")

    print("\n📊 This script will:")
    print("   1. Connect to Prometheus")
    print(f"   2. Fetch {TRAINING_HOURS} hours of CPU metrics")
    print("   3. Engineer features for machine learning")
    print("   4. Train an IsolationForest model")
    print("   5. Save the model for anomaly detection")

    try:
        # Step 1: Connect to Prometheus
        print_step(1, "Connecting to Prometheus")
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

        # Test connection
        prom.check_prometheus_connection()
        print("   ✓ Successfully connected to Prometheus")

        # Step 2: Fetch metrics
        print_step(2, f"Fetching up to {TRAINING_HOURS} hour(s) of CPU metrics")
        df = fetch_cpu_metrics(prom, TRAINING_HOURS)

        if len(df) < MIN_SAMPLES_REQUIRED:
            print(f"\n❌ Error: Only {len(df)} data points available.")
            print(f"   Minimum required: {MIN_SAMPLES_REQUIRED} samples")
            print("\n🔧 Solution:")
            print("   The monitoring stack just started and needs time to collect data.")
            print(f"   Each sample is collected every 10 seconds.")
            print(f"   Wait time needed: ~{(MIN_SAMPLES_REQUIRED * 10) // 60} minutes")
            print("\n⏱️  Options:")
            print("   1. Wait 15-20 minutes for more data to accumulate")
            print("   2. Generate some CPU load to create varied patterns:")
            print("      stress --cpu 2 --timeout 60s")
            print("      (Run this a few times, waiting 30s between runs)")
            print("\n   Then run this training script again.")
            return 1

        if len(df) < 100:
            print(f"\n⚠️  Note: Only {len(df)} data points available (~{len(df) * 10 // 60} minutes of data).")
            print("   For production, at least 1 week of data is recommended.")
            print("   Continuing with available data for lab demonstration...\n")

        # Step 3: Feature engineering
        print_step(3, "Engineering features for ML")
        df = engineer_features(df)

        # Step 4: Train model
        print_step(4, "Training IsolationForest model")
        model = train_model(df, CONTAMINATION)

        # Step 5: Save model
        print_step(5, "Saving trained model")
        save_model(model, MODEL_PATH)

        # Success summary
        print_header("✅ Training Complete!")
        print("\n📈 Model Summary:")
        print(f"   Algorithm: IsolationForest")
        print(f"   Training samples: {len(df)}")
        print(f"   Features: cpu_usage, rolling_mean, rolling_std, rate_of_change, hour")
        print(f"   Contamination: {CONTAMINATION*100}%")
        print(f"   Model location: {MODEL_PATH}")

        print("\n🎯 What the model learned:")
        print("   ✓ Normal CPU usage patterns and ranges")
        print("   ✓ Expected variability (rolling statistics)")
        print("   ✓ Typical rate of change (how fast CPU changes)")
        print("   ✓ Time-of-day patterns (hourly seasonality)")

        print("\n🚀 Next step:")
        print("   Run detect_anomalies.py to detect anomalies in live data!")
        print("   Example: python3 /root/monitoring/scripts/detect_anomalies.py")

        print("\n" + "="*70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\nTroubleshooting:")
        print("   1. Ensure Prometheus is running: docker-compose ps")
        print("   2. Check Prometheus has data: curl http://localhost:9090/api/v1/query?query=up")
        print("   3. Verify Node Exporter is scraping: curl http://localhost:9100/metrics")
        return 1

if __name__ == "__main__":
    sys.exit(main())