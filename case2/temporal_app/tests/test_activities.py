"""
Unit tests for Temporal activities.
"""

import pytest
import json

from activities import forecast_activities


class TestValidateRequest:
    """Tests for validate_request activity."""

    @pytest.mark.asyncio
    async def test_validate_request_valid(self):
        """Test validation passes for valid request."""
        request = {
            "device_id": "test_device_123",
            "horizon_hours": 24,
        }

        result = await forecast_activities.validate_request(request)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_request_horizon_too_small(self):
        """Test validation fails for horizon < 1."""
        request = {
            "device_id": "test_device_123",
            "horizon_hours": 0,
        }

        with pytest.raises(ValueError, match="Invalid horizon"):
            await forecast_activities.validate_request(request)

    @pytest.mark.asyncio
    async def test_validate_request_horizon_too_large(self):
        """Test validation fails for horizon > 72."""
        request = {
            "device_id": "test_device_123",
            "horizon_hours": 100,
        }

        with pytest.raises(ValueError, match="Invalid horizon"):
            await forecast_activities.validate_request(request)

    @pytest.mark.asyncio
    async def test_validate_request_edge_cases(self):
        """Test validation at boundary values."""
        # Minimum valid
        result = await forecast_activities.validate_request({
            "device_id": "test",
            "horizon_hours": 1
        })
        assert result is True

        # Maximum valid
        result = await forecast_activities.validate_request({
            "device_id": "test",
            "horizon_hours": 72
        })
        assert result is True


class TestFetchHistoricalData:
    """Tests for fetch_historical_data activity."""

    @pytest.mark.asyncio
    async def test_fetch_historical_data_returns_json(self, sample_device_id):
        """Test that fetch returns valid JSON string."""
        result = await forecast_activities.fetch_historical_data(sample_device_id)

        assert isinstance(result, str)

        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_fetch_historical_data_has_required_columns(self, sample_device_id):
        """Test that fetched data includes required columns."""
        result = await forecast_activities.fetch_historical_data(sample_device_id)
        data = json.loads(result)

        # Check first row has required columns
        required_cols = [
            'serial_number',
            'start',
            'end',
            'centralHeating.electricity',
            'domesticHotWater.electricity',
            'cooling.electricity',
            'manufacturer',
            'price_zone'
        ]

        first_row = data[0]
        for col in required_cols:
            assert col in first_row, f"Missing required column: {col}"

    @pytest.mark.asyncio
    async def test_fetch_historical_data_limits_rows(self, sample_device_id):
        """Test that data is limited to reasonable size."""
        result = await forecast_activities.fetch_historical_data(sample_device_id)
        data = json.loads(result)

        # Should limit to 7 days (672 intervals max)
        assert len(data) <= 672, f"Data has {len(data)} rows, should be max 672"

    @pytest.mark.asyncio
    async def test_fetch_historical_data_caching(self, sample_device_id):
        """Test that subsequent fetches use cache."""
        forecast_activities.redis_client.store.clear()
        result1 = await forecast_activities.fetch_historical_data(sample_device_id)
        result2 = await forecast_activities.fetch_historical_data(sample_device_id)
        assert result1 == result2


class TestLoadModelFromCache:
    """Tests for load_model_from_cache activity."""

    @pytest.mark.asyncio
    async def test_load_model_returns_metadata(self, sample_device_id):
        """Test that load_model returns model metadata."""
        result = await forecast_activities.load_model_from_cache(sample_device_id)

        assert isinstance(result, dict)
        assert 'version' in result
        assert 'source' in result
        assert 'device_id' in result

    @pytest.mark.asyncio
    async def test_load_model_caches_in_memory(self, sample_device_id):
        """Test that model is cached in worker memory."""
        forecast_activities.model_cache.clear()

        result1 = await forecast_activities.load_model_from_cache(sample_device_id)
        assert result1['source'] == 'storage'

        result2 = await forecast_activities.load_model_from_cache(sample_device_id)
        assert result2['source'] == 'memory'


class TestGenerateForecast:
    """Tests for generate_forecast activity."""

    @pytest.mark.asyncio
    async def test_generate_forecast_basic(self, sample_device_id, sample_historical_data):
        """Test basic forecast generation."""
        historical_json = sample_historical_data.to_json(orient='records', date_format='iso')

        input_data = {
            "device_id": sample_device_id,
            "forecast_start": "2025-02-28T01:00:00+00:00",
            "horizon_hours": 6,
            "historical_data": historical_json,
        }

        result = await forecast_activities.generate_forecast(input_data)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'device_id' in result
        assert 'predictions_count' in result
        assert 'total_power_wh' in result
        assert 'avg_power_wh' in result

    @pytest.mark.asyncio
    async def test_generate_forecast_count(self, sample_device_id, sample_historical_data):
        """Test forecast generates correct number of predictions."""
        historical_json = sample_historical_data.to_json(orient='records', date_format='iso')

        for horizon in [1, 6, 12, 24]:
            input_data = {
                "device_id": sample_device_id,
                "forecast_start": "2025-02-28T01:00:00+00:00",
                "horizon_hours": horizon,
                "historical_data": historical_json,
            }

            result = await forecast_activities.generate_forecast(input_data)

            expected_count = horizon * 4  # 4 intervals per hour
            assert result['predictions_count'] == expected_count, \
                f"Expected {expected_count} predictions for {horizon}h, got {result['predictions_count']}"

    @pytest.mark.asyncio
    async def test_generate_forecast_non_negative(self, sample_device_id, sample_historical_data):
        """Test that forecasts are non-negative."""
        historical_json = sample_historical_data.to_json(orient='records', date_format='iso')

        input_data = {
            "device_id": sample_device_id,
            "forecast_start": "2025-02-28T01:00:00+00:00",
            "horizon_hours": 6,
            "historical_data": historical_json,
        }

        result = await forecast_activities.generate_forecast(input_data)

        assert result['total_power_wh'] >= 0
        assert result['avg_power_wh'] >= 0

    @pytest.mark.asyncio
    async def test_generate_forecast_includes_samples(self, sample_device_id, sample_historical_data):
        """Test that forecast includes prediction samples."""
        historical_json = sample_historical_data.to_json(orient='records', date_format='iso')

        input_data = {
            "device_id": sample_device_id,
            "forecast_start": "2025-02-28T01:00:00+00:00",
            "horizon_hours": 6,
            "historical_data": historical_json,
        }

        result = await forecast_activities.generate_forecast(input_data)

        assert 'predictions_sample' in result
        assert isinstance(result['predictions_sample'], list)
        assert len(result['predictions_sample']) > 0


class TestPublishForecast:
    """Tests for publish_forecast activity."""

    @pytest.mark.asyncio
    async def test_publish_forecast_success(self):
        """Test that publish returns success."""
        data = {
            "device_id": "test_device",
            "predictions_count": 96,
            "total_energy_kwh": 6.44,
            "avg_power_wh": 67.09,
            "workflow_id": "test_workflow_123"
        }

        result = await forecast_activities.publish_forecast(data)
        assert result is True


class TestStoreAuditLog:
    """Tests for store_audit_log activity."""

    @pytest.mark.asyncio
    async def test_store_audit_log_success(self):
        """Test that audit log stores successfully."""
        data = {
            "workflow_id": "test_workflow_123",
            "request": {
                "device_id": "test_device",
                "forecast_start": "2025-02-28T01:00:00+00:00",
                "horizon_hours": 6,
                "requestor": "test_suite"
            },
            "result": {
                "predictions_count": 24,
                "model_version": "v1.0"
            }
        }

        result = await forecast_activities.store_audit_log(data)
        assert result is True
