"""
AI-Driven Forecasting - Prediction Script
Module 5: AIOps Foundations

This script uses trained Prophet models to generate forecasts for infrastructure
metrics and estimate threshold crossings for proactive planning.

Usage (from the project root, with the virtual environment activated):
    python monitoring/scripts/forecasting_model.py --days 7
Omit --days for the interactive horizon menu.
"""

import argparse
import sys
import logging

# Keep backend progress logs quiet, but retain useful Python warnings.
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

try:
    import pandas as pd
    import numpy as np
    from prometheus_api_client import PrometheusConnect

    from prophet.serialize import model_from_json

except ImportError as e:
    print(f"❌ Error: Missing required package: {e}")
    print("\n💡 Please install packages:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


from forecasting_config import METRICS_CONFIG, MODEL_DIR, PROMETHEUS_URL


def print_header():
    """Print script header."""
    print("\n" + "="*70)
    print("📈 AI-Driven Forecasting - Generate Predictions")
    print("Module 5: AIOps Foundations")
    print("="*70 + "\n")


def connect_to_prometheus():
    """Connect to Prometheus API."""
    try:
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
        if not prom.check_prometheus_connection():
            raise ConnectionError("Prometheus returned an unsuccessful response")
        return prom
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to Prometheus: {e}")
        return None


def get_current_value(prom, metric_key):
    """Fetch current metric value from Prometheus."""
    if prom is None:
        return None

    try:
        query = METRICS_CONFIG[metric_key]['query']

        # Query current value
        result = prom.custom_query(query=query)

        if result and len(result) == 1:
            value = float(result[0]['value'][1])
            return value if np.isfinite(value) else None
        else:
            return None
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch current value for {metric_key}: {e}")
        return None


def load_model(metric_key):
    """Load trained model from disk."""
    model_path = MODEL_DIR / f"{metric_key}_forecast_model.json"

    if not model_path.exists():
        return None

    try:
        return model_from_json(model_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ Error loading model for {metric_key}: {e}")
        return None


def generate_forecast(model, metric_key, periods=7, start_time=None):
    """Forecast from now on an hourly grid, without historical fitted rows."""
    try:
        if not isinstance(periods, int) or not 1 <= periods <= 90:
            raise ValueError("Forecast horizon must be an integer from 1 to 90 days")
        start = pd.Timestamp.now(tz='UTC') if start_time is None else pd.Timestamp(start_time)
        if start.tzinfo is not None:
            start = start.tz_convert('UTC').tz_localize(None)
        start = max(start, model.history['ds'].max())
        # Day-scale horizons do not require 30-second predictions (259,200 rows
        # for 90 days). Include the starting estimate and exact horizon endpoint.
        future = pd.DataFrame({'ds': pd.date_range(start, periods=periods * 24 + 1, freq='h')})

        if model.growth == 'logistic':
            # Reuse the fitted bounds, including models trained with an older floor.
            future['cap'] = float(model.history['cap'].iloc[-1])
            future['floor'] = float(model.history['floor'].iloc[-1])

        # Logistic saturation can overflow exp() while still yielding a valid
        # limit of 0 or cap. Silence only that expected numerical warning here;
        # reject any non-finite final prediction below.
        with np.errstate(over='ignore'):
            forecast = model.predict(future)
        # Reporting constraints apply to the estimates AND uncertainty bounds.
        upper = 100.0 if METRICS_CONFIG[metric_key]['unit'] == '%' else None
        columns = ['yhat', 'yhat_lower', 'yhat_upper']
        if not np.isfinite(forecast[columns].to_numpy()).all():
            raise ValueError("Prophet produced non-finite predictions")
        forecast[columns] = forecast[columns].clip(lower=0.0, upper=upper)
        return forecast
    except Exception as e:
        print(f"❌ Error generating forecast for {metric_key}: {e}")
        return None


def analyze_forecast(forecast, config, current_value, periods=7):
    """Analyze forecast results and identify capacity issues."""
    # generate_forecast returns only the requested horizon, starting at now.
    forecast_period = forecast.sort_values('ds')
    if forecast_period.empty:
        raise ValueError("Cannot analyze an empty forecast")

    # Extract key metrics from forecast period
    forecast_mean = forecast_period['yhat'].mean()
    forecast_max = forecast_period['yhat'].max()
    forecast_min = forecast_period['yhat'].min()

    # Calculate trend
    trend_start = forecast_period['yhat'].iloc[0]
    trend_end = forecast_period['yhat'].iloc[-1]
    trend_change = trend_end - trend_start
    trend_per_day = trend_change / periods

    # Calculate confidence interval width
    ci_width = (forecast_period['yhat_upper'] - forecast_period['yhat_lower']).mean()

    # Check for threshold breach
    threshold = config['threshold']
    threshold_type = config['threshold_type']

    days_to_threshold = None
    threshold_breached = False

    if threshold_type == 'upper':
        breach_df = forecast_period[forecast_period['yhat'] > threshold]
    elif threshold_type == 'lower':
        breach_df = forecast_period[forecast_period['yhat'] < threshold]
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    if not breach_df.empty:
        threshold_breached = True
        # Use timestamps, not positions in the filtered breach dataframe (whose
        # first row always has position zero). Works with either threshold type.
        elapsed = breach_df['ds'].iloc[0] - forecast_period['ds'].iloc[0]
        days_to_threshold = elapsed.total_seconds() / 86400

    return {
        'current_value': current_value,
        'forecast_mean': forecast_mean,
        'forecast_max': forecast_max,
        'forecast_min': forecast_min,
        'trend_per_day': trend_per_day,
        'ci_width': ci_width,
        'threshold_breached': threshold_breached,
        'days_to_threshold': days_to_threshold
    }


def print_forecast_report(metric_key, config, analysis, periods, value_source='Current'):
    """Print clean, compact forecast report."""
    fmt = config['format']
    unit = config['unit']

    print(f"\n{config['name']} Forecast:")
    print(f"  {value_source}: {analysis['current_value']:{fmt}}{unit}")

    # {periods}-day prediction with range
    forecast_range = f"{analysis['forecast_min']:{fmt}}{unit} - {analysis['forecast_max']:{fmt}}{unit}"
    print(f"  {periods}-day prediction: {analysis['forecast_mean']:{fmt}}{unit} (range: {forecast_range})")

    # Trend analysis with arrow
    trend = analysis['trend_per_day']
    if abs(trend) > 0.01:
        arrow = "↗" if trend > 0 else "↘"
        direction = "Increasing" if trend > 0 else "Decreasing"
        total_change = trend * periods
        print(f"  Trend: {arrow} {direction} ({total_change:+{fmt}}{unit} over {periods} days, {trend:+{fmt}}{unit}/day)")
    else:
        print(f"  Trend: → Stable (minimal change)")

    # Threshold status
    threshold = config['threshold']
    threshold_type = config['threshold_type']

    if analysis['threshold_breached']:
        days = analysis['days_to_threshold']

        if threshold_type == 'upper':
            threshold_msg = f"may exceed {threshold}{unit}"
        else:
            threshold_msg = f"may drop below {threshold}{unit}"

        # Urgency indicator
        if days < 7:
            urgency = "🚨"
        elif days < 14:
            urgency = "⚠️"
        else:
            urgency = "💡"

        print(f"  Alert: {urgency} {config['name']} {threshold_msg} in ~{days:.1f} days")

        # Single-line recommendation
        if metric_key == 'disk_usage':
            recommendation = "Plan storage expansion or cleanup"
        elif metric_key == 'cpu_usage':
            recommendation = "Consider scaling up or optimizing workloads"
        elif metric_key == 'memory_available':
            recommendation = "Investigate memory leaks or add more RAM"
        else:
            recommendation = "Review capacity planning"

        print(f"  Recommendation (after validation): {recommendation}")
    else:
        if threshold_type == 'upper':
            status_msg = f"is not predicted to exceed {threshold}{unit}"
        else:
            status_msg = f"is not predicted to drop below {threshold}{unit}"

        print(f"  Status: ✅ {config['name']} {status_msg} within {periods} days")
        print("  Recommendation: Keep collecting data and compare predictions with actual values")


def prompt_forecast_horizon():
    """Ask user for forecast horizon."""
    print("📅 Select forecast horizon:")
    print("   1) 7 days")
    print("   2) 14 days")
    print("   3) 30 days")
    print("   4) Custom")

    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                return 7
            elif choice == '2':
                return 14
            elif choice == '3':
                return 30
            elif choice == '4':
                days = input("Enter number of days (1-90): ").strip()
                days = int(days)
                if 1 <= days <= 90:
                    return days
                else:
                    print("❌ Please enter a value between 1 and 90")
            else:
                print("❌ Invalid choice, please enter 1-4")

        except (ValueError, KeyboardInterrupt, EOFError):
            print("\n❌ Invalid input, using default: 7 days")
            return 7


def main():
    """Main forecasting workflow."""
    parser = argparse.ArgumentParser(description="Generate lab forecasts (hourly resolution).")
    parser.add_argument('--days', type=int, choices=range(1, 91), metavar='1-90',
                        help="Forecast horizon; omit for the interactive menu")
    args = parser.parse_args()
    print_header()

    # Check if model directory exists
    if not MODEL_DIR.exists():
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("\n💡 Please train models first:")
        print("   python monitoring/scripts/train_forecasting_model.py")
        sys.exit(1)

    # Load all available models
    print("🔍 Loading trained models...")
    models = {}
    for metric_key, config in METRICS_CONFIG.items():
        model = load_model(metric_key)
        if model:
            models[metric_key] = model
            print(f"   ✅ Loaded model for {config['name']}")
        else:
            print(f"   ⚠️  Model not found for {config['name']}")

    if not models:
        print("\n❌ No trained models found!")
        print("💡 Please train models first:")
        print("   python monitoring/scripts/train_forecasting_model.py")
        sys.exit(1)

    print(f"\n✅ Loaded {len(models)} model(s)\n")

    # Connect to Prometheus to fetch current values
    print("🔌 Connecting to Prometheus...")
    prom = connect_to_prometheus()
    if prom:
        print("✅ Connected to Prometheus\n")
    else:
        print("⚠️  Could not connect to Prometheus - will use training end values\n")

    # Get forecast horizon from user
    periods = args.days if args.days is not None else prompt_forecast_horizon()
    forecast_start = pd.Timestamp.now(tz='UTC').tz_localize(None)
    print(f"Forecast start: {forecast_start} UTC (hourly resolution)")
    print("Lab estimates: short history cannot validate forecast accuracy or capacity dates.")

    print(f"\n🔮 Generating {periods}-day forecasts...\n")

    # Generate forecasts for each metric
    forecast_results = {}

    for metric_key, model in models.items():
        config = METRICS_CONFIG[metric_key]

        print(f"📊 Processing {config['name']}...")

        history_start = model.history['ds'].min()
        history_end = model.history['ds'].max()
        print(f"   Training history: {history_start} to {history_end} UTC")
        if forecast_start - history_end > pd.Timedelta(hours=1):
            print("   ⚠️  Model is over one hour old; retrain before using these estimates.")
        current_value = get_current_value(prom, metric_key)
        value_source = 'Current'
        if current_value is None:
            # Prophet retains the original observations in history['y'].
            # Avoid an extra forecast and never label an old observation as current.
            current_value = float(model.history['y'].iloc[-1])
            value_source = f'Last training observation ({history_end} UTC)'

        # Generate forecast
        forecast = generate_forecast(model, metric_key, periods=periods, start_time=forecast_start)

        if forecast is None:
            print(f"   ❌ Failed to generate forecast for {config['name']}")
            continue

        # Analyze forecast
        analysis = analyze_forecast(forecast, config, current_value, periods=periods)

        # Store results
        forecast_results[metric_key] = {
            'forecast': forecast,
            'analysis': analysis,
            'value_source': value_source
        }

        print(f"   ✅ Forecast generated for {config['name']}")

    if not forecast_results:
        print("❌ No forecasts were generated successfully")
        return 1

    # Print detailed reports
    print("\n" + "="*70)
    print(f"📊 {periods}-Day Forecast Results")
    print("="*70)

    for metric_key, results in forecast_results.items():
        config = METRICS_CONFIG[metric_key]
        print_forecast_report(
            metric_key,
            config,
            results['analysis'],
            periods,
            value_source=results['value_source']
        )

    # Print summary
    print("\n" + "="*70)
    print("📋 Summary")
    print("="*70)

    alerts = []
    ok_metrics = []

    for metric_key, results in forecast_results.items():
        config = METRICS_CONFIG[metric_key]
        analysis = results['analysis']

        if analysis['threshold_breached']:
            alerts.append({
                'metric': config['name'],
                'days': analysis['days_to_threshold']
            })
        else:
            ok_metrics.append(config['name'])

    if alerts:
        print(f"\n⚠️  Capacity Alerts ({len(alerts)}):")
        # Sort by urgency (days to threshold)
        alerts.sort(key=lambda x: x['days'])
        for alert in alerts:
            urgency = "🚨" if alert['days'] < 7 else "⚠️" if alert['days'] < 14 else "💡"
            print(f"  {urgency} {alert['metric']}: Unvalidated threshold crossing in ~{alert['days']:.1f} days")

    if ok_metrics:
        print(f"\nMetrics without a predicted threshold breach ({len(ok_metrics)}):")
        for metric in ok_metrics:
            print(f"  • {metric}")

    print("\n" + "="*70)
    print("✨ Forecast generation complete!")
    print("\n💡 Next Steps:")
    print("  • Review forecasts weekly and retrain models monthly")
    print("  • Compare predictions to actual values to validate accuracy")
    print("  • View detailed metrics in Grafana dashboard")
    print("="*70 + "\n")
    return 0 if len(forecast_results) == len(METRICS_CONFIG) else 1


if __name__ == "__main__":
    sys.exit(main())
