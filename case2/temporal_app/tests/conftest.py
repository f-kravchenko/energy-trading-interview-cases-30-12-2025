"""
Pytest fixtures and configuration for testing.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_device_id():
    """Device ID for testing."""
    return "21183900202529220938026983N5"


@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing."""
    csv_path = Path(__file__).parent.parent.parent / "data/sample_devices_actual_power_consumption.csv"
    df = pd.read_csv(csv_path)

    # Get first device
    device_id = df['serial_number'].iloc[0]
    device_data = df[df['serial_number'] == device_id].head(100)

    return device_data


@pytest.fixture
def forecast_start_time():
    """Forecast start time for testing (aligned with data)."""
    # Use Feb 28, 2025 to align with historical data ending Feb 27
    return datetime(2025, 2, 28, 1, 0, tzinfo=timezone.utc)


@pytest.fixture
def model_path():
    """Path to trained model."""
    return Path(__file__).parent.parent.parent / "trained_model"


@pytest.fixture
def sample_forecast_request():
    """Sample forecast request for workflow testing."""
    return {
        "device_id": "21183900202529220938026983N5",
        "forecast_start": "2025-02-28T01:00:00+00:00",
        "horizon_hours": 6,
        "requestor": "test_suite"
    }
