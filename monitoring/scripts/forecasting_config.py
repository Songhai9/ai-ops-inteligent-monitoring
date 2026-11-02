"""Shared paths, queries and thresholds for the forecasting lab."""

from pathlib import Path

# Resolve from the scripts, so both commands work from any working directory.
MODEL_DIR = Path(__file__).resolve().parents[1] / "forecasting_models"
PROMETHEUS_URL = "http://localhost:9090"
TRAINING_STEP = "30s"

# Each query deliberately returns a single monitoring-wide series.
# CPU is the average across CPUs/nodes; memory is the least available memory;
# disk is the most-used filesystem. These are aggregate lab forecasts.
# Docker Desktop exposes Linux VM/container metrics, not native macOS metrics.
METRICS_CONFIG = {
    'cpu_usage': {
        'query': '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
        'name': 'CPU Usage',
        'unit': '%',
        'threshold': 80,
        'threshold_type': 'upper',
        'format': '.1f',
    },
    'memory_available': {
        'query': 'min(node_memory_MemAvailable_bytes) / 1024 / 1024 / 1024',
        'name': 'Memory Available',
        'unit': 'GB',
        'threshold': 3.0,
        'threshold_type': 'lower',
        'format': '.2f',
    },
    'disk_usage': {
        'query': 'max((1 - node_filesystem_avail_bytes{fstype=~"ext4|xfs|btrfs"} / node_filesystem_size_bytes{fstype=~"ext4|xfs|btrfs"}) * 100)',
        'name': 'Disk Usage',
        'unit': '%',
        'threshold': 90,
        'threshold_type': 'upper',
        'format': '.1f',
    },
}
