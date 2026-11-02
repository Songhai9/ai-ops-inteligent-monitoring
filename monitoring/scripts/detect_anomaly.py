"""
AI-Powered Anomaly Detection - Real-Time Detection Script
Module 4: AIOps Foundations

This script uses the trained IsolationForest model to detect anomalies
in real-time CPU metrics from Prometheus.

Usage: With your virtual environment activated, run from the project root:
    python monitoring/scripts/detect_anomaly.py
"""

import os
import sys
import pickle
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from prometheus_api_client import PrometheusConnect

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
PROMETHEUS_URL = "http://localhost:9090"
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "anomaly_model.pkl")
METRICS_QUERY = '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'

# Detection parameters
CHECK_INTERVAL = 30  # Check every 30 seconds
LOOKBACK_MINUTES = 10  # Analyze last 10 minutes

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def load_model(path: str):
    """Load trained model from disk"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Please run train_anomaly_model.py first to train the model."
        )

    with open(path, 'rb') as f:
        model = pickle.load(f)

    print(f"   ✓ Model loaded from: {path}")
    return model

def fetch_recent_metrics(prom: PrometheusConnect, minutes: int) -> pd.DataFrame:
    """
    Fetch recent CPU usage metrics from Prometheus

    Args:
        prom: Prometheus connection
        minutes: Number of minutes of recent data to fetch

    Returns:
        DataFrame with timestamp and cpu_usage columns
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=minutes)

    # Query Prometheus
    result = prom.custom_query_range(
        query=METRICS_QUERY,
        start_time=start_time,
        end_time=end_time,
        step='10s'  # 10-second resolution (matches Prometheus scrape_interval)
    )

    if not result:
        raise ValueError("No data returned from Prometheus")

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

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create same features as during training

    Args:
        df: DataFrame with cpu_usage column

    Returns:
        DataFrame with engineered features
    """
    # Rolling statistics (last 5 samples)
    df['rolling_mean'] = df['cpu_usage'].rolling(window=5, min_periods=1).mean()
    df['rolling_std'] = df['cpu_usage'].rolling(window=5, min_periods=1).std().fillna(0)

    # Rate of change
    df['rate_of_change'] = df['cpu_usage'].diff().fillna(0)

    # Hour of day (for seasonality)
    df['hour'] = df['timestamp'].dt.hour

    # Drop NaN values
    df = df.dropna()

    return df

def detect_anomalies(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using trained model

    Args:
        model: Trained IsolationForest model
        df: DataFrame with features

    Returns:
        DataFrame with predictions and anomaly scores
    """
    # Select same features used during training
    feature_columns = ['cpu_usage', 'rolling_mean', 'rolling_std', 'rate_of_change', 'hour']
    X = df[feature_columns]

    # Predict (-1 = anomaly, 1 = normal)
    predictions = model.predict(X)

    # Get anomaly scores (lower = more anomalous)
    scores = model.decision_function(X)

    # Add to DataFrame
    df['prediction'] = predictions
    df['anomaly_score'] = scores
    df['is_anomaly'] = predictions == -1

    return df

def print_detection_summary(df: pd.DataFrame):
    """Print summary of detection results"""
    total = len(df)
    anomalies = df['is_anomaly'].sum()
    normal = total - anomalies

    print(f"\n📊 Detection Summary (last {LOOKBACK_MINUTES} minutes):")
    print(f"   Total samples: {total}")
    print(f"   Normal: {normal} ({normal/total*100:.1f}%)")
    print(f"   Anomalies: {anomalies} ({anomalies/total*100:.1f}%)")

    if anomalies > 0:
        print(f"\n⚠️  {anomalies} ANOMALIES DETECTED!")
        print("\n🔍 Anomaly Details:")

        anomaly_df = df[df['is_anomaly']].copy()

        for idx, row in anomaly_df.iterrows():
            print(f"\n   Timestamp: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   CPU Usage: {row['cpu_usage']:.2f}%")
            print(f"   Anomaly Score: {row['anomaly_score']:.4f} (lower = more anomalous)")
            print(f"   Rolling Mean: {row['rolling_mean']:.2f}%")
            print(f"   Std Deviation: {row['rolling_std']:.2f}%")
            print(f"   Rate of Change: {row['rate_of_change']:.2f}%")

        # Analysis
        print(f"\n💡 Why these are anomalies:")

        high_cpu = anomaly_df[anomaly_df['cpu_usage'] > anomaly_df['cpu_usage'].quantile(0.75)]
        if len(high_cpu) > 0:
            print(f"   • {len(high_cpu)} samples: Unusually HIGH CPU usage")

        low_cpu = anomaly_df[anomaly_df['cpu_usage'] < anomaly_df['cpu_usage'].quantile(0.25)]
        if len(low_cpu) > 0:
            print(f"   • {len(low_cpu)} samples: Unusually LOW CPU usage")

        high_volatility = anomaly_df[anomaly_df['rolling_std'] > anomaly_df['rolling_std'].quantile(0.75)]
        if len(high_volatility) > 0:
            print(f"   • {len(high_volatility)} samples: Unusually HIGH volatility (rapid changes)")

        print("\n   These patterns differ from what the model learned as 'normal'")
        print("   during training. This could indicate:")
        print("     - A legitimate issue (resource spike, process misbehaving)")
        print("     - A new workload pattern (deployment, scheduled job)")
        print("     - Normal variability at boundary of learned behavior")

    else:
        print(f"\n✅ No anomalies detected - all metrics within normal range")
        print("\n   The current CPU behavior matches what the model learned")
        print("   as 'normal' during training:")

        latest = df.iloc[-1]
        print(f"     - Current CPU: {latest['cpu_usage']:.2f}%")
        print(f"     - Rolling Average: {latest['rolling_mean']:.2f}%")
        print(f"     - Std Deviation: {latest['rolling_std']:.2f}%")

def continuous_monitoring(model, prom: PrometheusConnect, duration_minutes: int):
    """
    Run continuous anomaly detection for specified duration

    Args:
        model: Trained model
        prom: Prometheus connection
        duration_minutes: How long to monitor (minutes)
    """
    print_header(f"🔄 Continuous Monitoring ({duration_minutes} minutes)")

    print(f"\n⏱️  Checking for anomalies every {CHECK_INTERVAL} seconds")
    print(f"📊 Analyzing last {LOOKBACK_MINUTES} minutes of data each check")
    print(f"⏰ Will run for {duration_minutes} minutes")
    print("\nPress Ctrl+C to stop early\n")

    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    check_count = 0

    try:
        while datetime.now() < end_time:
            check_count += 1
            current_time = datetime.now()

            print(f"\n{'─'*70}")
            print(f"Check #{check_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─'*70}")

            # Fetch recent data
            df = fetch_recent_metrics(prom, LOOKBACK_MINUTES)

            # Engineer features
            df = engineer_features(df)

            # Detect anomalies
            df = detect_anomalies(model, df)

            # Print results
            print_detection_summary(df)

            # Wait before next check
            time_remaining = (end_time - datetime.now()).total_seconds()
            if time_remaining > CHECK_INTERVAL:
                print(f"\n⏳ Next check in {CHECK_INTERVAL} seconds...")
                time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n✅ Monitoring completed ({elapsed:.1f} minutes, {check_count} checks)")

def main():
    """Main detection workflow"""
    print_header("🤖 AI-Powered Anomaly Detection - Real-Time Detection")

    print("\n🎯 This script will:")
    print("   1. Load the trained anomaly detection model")
    print("   2. Connect to Prometheus")
    print("   3. Run initial anomaly detection")
    print("   4. Automatically monitor for 5 minutes")
    print("   5. Ask if you want to continue monitoring")

    try:
        # Step 1: Load model
        print("\n[Step 1] Loading trained model")
        model = load_model(MODEL_PATH)

        # Step 2: Connect to Prometheus
        print("\n[Step 2] Connecting to Prometheus")
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
        prom.check_prometheus_connection()
        print("   ✓ Successfully connected to Prometheus")

        # Step 3: Initial detection
        print("\n[Step 3] Running initial anomaly detection")
        df = fetch_recent_metrics(prom, LOOKBACK_MINUTES)
        print(f"   ✓ Fetched {len(df)} data points")

        df = engineer_features(df)
        df = detect_anomalies(model, df)
        print_detection_summary(df)

        # Step 4: Automatic 5-minute continuous monitoring
        print("\n" + "="*70)
        print("\n🚀 Starting automatic 5-minute continuous monitoring...")
        print("   This will help you see how the model detects anomalies over time.")
        continuous_monitoring(model, prom, 5)

        # Step 5: Ask if user wants to continue monitoring
        while True:
            print("\n" + "="*70)
            choice = input("\n🔄 Continue monitoring? (y/n): ").strip().lower()

            if choice == 'y':
                try:
                    duration = int(input("⏱️  How many more minutes to monitor? (e.g., 5): ").strip())
                    if duration > 0:
                        continuous_monitoring(model, prom, duration)
                    else:
                        print("\n⚠️  Duration must be greater than 0. Stopping monitoring.")
                        break
                except ValueError:
                    print("\n⚠️  Invalid input. Please enter a number. Stopping monitoring.")
                    break
            else:
                print("\n✅ Monitoring complete!")
                break

        print("\n" + "="*70)
        print("  🎓 Key Takeaways")
        print("="*70)
        print("\n✨ What you've learned:")
        print("   • How ML models detect anomalies in real-time data")
        print("   • The difference between reactive (thresholds) and proactive (ML) monitoring")
        print("   • How to interpret anomaly scores and predictions")
        print("   • The importance of feature engineering for time-series data")

        print("\n🚀 This is how modern AIOps platforms work:")
        print("   1. Learn normal behavior from historical data")
        print("   2. Continuously compare live data to learned patterns")
        print("   3. Flag statistically significant deviations")
        print("   4. Adapt as the system evolves")

        print("\n💡 Advantages over static thresholds:")
        print("   ✓ No manual threshold tuning needed")
        print("   ✓ Adapts to time-of-day patterns")
        print("   ✓ Detects subtle anomalies missed by thresholds")
        print("   ✓ Reduces false positives (alert fatigue)")

        print("\n" + "="*70 + "\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Solution:")
        print("   Run the training script first:")
        print("   python monitoring/scripts/train_anomaly_model.py")
        return 1

    except Exception as e:
        print(f"\n❌ Error during detection: {str(e)}")
        print("\nTroubleshooting:")
        print("   1. Ensure Prometheus is running: docker compose ps")
        print(f"   2. Verify model exists: {MODEL_PATH}")
        print("   3. Check Prometheus has data: curl http://localhost:9090/api/v1/query?query=up")
        return 1

if __name__ == "__main__":
    sys.exit(main())