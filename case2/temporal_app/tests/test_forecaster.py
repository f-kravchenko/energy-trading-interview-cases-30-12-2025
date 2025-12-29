"""
Unit tests for HeatPumpForecaster.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecaster import HeatPumpForecaster


class TestHeatPumpForecaster:
    """Test suite for HeatPumpForecaster."""

    def test_forecaster_loads_model(self, model_path):
        """Test that forecaster can load trained model."""
        forecaster = HeatPumpForecaster(model_path=model_path)

        assert forecaster.model is not None
        assert forecaster.feature_cols is not None
        assert len(forecaster.feature_cols) > 0

    def test_forecaster_predict_basic(self, model_path, sample_historical_data, sample_device_id):
        """Test basic prediction functionality."""
        forecaster = HeatPumpForecaster(model_path=model_path)

        # Parse timestamps
        df = sample_historical_data.copy()
        df['start'] = pd.to_datetime(df['start'])
        last_date = df['start'].max()

        # Forecast from last available data
        forecast_start = pd.to_datetime(last_date) + pd.Timedelta(hours=1)

        result = forecaster.predict(
            device_id=sample_device_id,
            start_time=forecast_start.to_pydatetime(),
            horizon_hours=6,
            historical_data=sample_historical_data
        )

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'predicted_power' in result.columns

        # Verify predictions
        assert len(result) == 24  # 6 hours * 4 intervals per hour
        assert result['predicted_power'].notna().all()

    def test_forecaster_non_negative_predictions(self, model_path, sample_historical_data, sample_device_id):
        """Test that predictions are non-negative."""
        forecaster = HeatPumpForecaster(model_path=model_path)

        df = sample_historical_data.copy()
        df['start'] = pd.to_datetime(df['start'])
        last_date = df['start'].max()
        forecast_start = pd.to_datetime(last_date) + pd.Timedelta(hours=1)

        result = forecaster.predict(
            device_id=sample_device_id,
            start_time=forecast_start.to_pydatetime(),
            horizon_hours=6,
            historical_data=sample_historical_data
        )

        # All predictions should be non-negative (clipped)
        assert (result['predicted_power'] >= 0).all(), \
            f"Found negative predictions: {result[result['predicted_power'] < 0]}"

    def test_forecaster_horizon_hours(self, model_path, sample_historical_data, sample_device_id):
        """Test different horizon hours."""
        forecaster = HeatPumpForecaster(model_path=model_path)

        df = sample_historical_data.copy()
        df['start'] = pd.to_datetime(df['start'])
        last_date = df['start'].max()
        forecast_start = pd.to_datetime(last_date) + pd.Timedelta(hours=1)

        # Test different horizons
        for horizon in [1, 6, 12, 24]:
            result = forecaster.predict(
                device_id=sample_device_id,
                start_time=forecast_start.to_pydatetime(),
                horizon_hours=horizon,
                historical_data=sample_historical_data
            )

            expected_intervals = horizon * 4
            assert len(result) == expected_intervals, \
                f"Expected {expected_intervals} predictions for {horizon}h, got {len(result)}"

    def test_forecaster_timestamp_sequence(self, model_path, sample_historical_data, sample_device_id):
        """Test that forecast timestamps are properly sequenced."""
        forecaster = HeatPumpForecaster(model_path=model_path)

        df = sample_historical_data.copy()
        df['start'] = pd.to_datetime(df['start'])
        last_date = df['start'].max()
        forecast_start = pd.to_datetime(last_date) + pd.Timedelta(hours=1)

        result = forecaster.predict(
            device_id=sample_device_id,
            start_time=forecast_start.to_pydatetime(),
            horizon_hours=6,
            historical_data=sample_historical_data
        )

        # Check timestamps are sequential with 15-minute intervals
        timestamps = pd.to_datetime(result['timestamp'])
        diffs = timestamps.diff().dropna()

        assert (diffs == pd.Timedelta(minutes=15)).all(), \
            "Timestamps should be 15 minutes apart"

        # First timestamp should match forecast_start
        assert timestamps.iloc[0] == forecast_start

    def test_forecaster_requires_historical_data(self, model_path, sample_device_id):
        """Test that forecaster requires historical data."""
        forecaster = HeatPumpForecaster(model_path=model_path)

        with pytest.raises(ValueError, match="historical_data is required"):
            forecaster.predict(
                device_id=sample_device_id,
                start_time=datetime.now(timezone.utc),
                horizon_hours=6,
                historical_data=None
            )

    def test_forecaster_requires_trained_model(self):
        """Test that predict requires trained model."""
        forecaster = HeatPumpForecaster()

        with pytest.raises(ValueError, match="Model not trained"):
            forecaster.predict(
                device_id="test",
                start_time=datetime.now(timezone.utc),
                horizon_hours=6,
                historical_data=pd.DataFrame()
            )

    def test_forecaster_reasonable_predictions(self, model_path, sample_historical_data, sample_device_id):
        """Test that predictions are in reasonable range."""
        forecaster = HeatPumpForecaster(model_path=model_path)

        df = sample_historical_data.copy()
        df['start'] = pd.to_datetime(df['start'])
        last_date = df['start'].max()
        forecast_start = pd.to_datetime(last_date) + pd.Timedelta(hours=1)

        result = forecaster.predict(
            device_id=sample_device_id,
            start_time=forecast_start.to_pydatetime(),
            horizon_hours=6,
            historical_data=sample_historical_data
        )

        # Predictions should be within reasonable range
        # Heat pumps typically consume 0-5000 Wh per 15-min interval
        assert (result['predicted_power'] <= 5000).all(), \
            f"Found unreasonably high predictions: {result[result['predicted_power'] > 5000]}"

        # Average should be reasonable
        avg_power = result['predicted_power'].mean()
        assert 0 <= avg_power <= 2000, \
            f"Average power {avg_power} Wh is outside expected range"
