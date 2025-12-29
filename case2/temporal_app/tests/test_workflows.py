"""
Integration tests for Temporal workflows.

These tests use Temporal's testing framework to test workflows
without needing a running Temporal server.
"""

import pytest

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from workflows.intraday_forecast import (
    IntradayForecastWorkflow,
    ForecastRequest,
    ForecastResult,
)
from activities import forecast_activities


class TestIntradayForecastWorkflow:
    """Integration tests for IntradayForecastWorkflow."""

    @pytest.mark.asyncio
    async def test_workflow_completes_successfully(self):
        """Test that workflow completes end-to-end."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            # Create worker with workflow and activities
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IntradayForecastWorkflow],
                activities=[
                    forecast_activities.validate_request,
                    forecast_activities.fetch_historical_data,
                    forecast_activities.load_model_from_cache,
                    forecast_activities.generate_forecast,
                    forecast_activities.publish_forecast,
                    forecast_activities.store_audit_log,
                ],
            ):
                # Execute workflow
                request = ForecastRequest(
                    device_id="21183900202529220938026983N5",
                    forecast_start="2025-02-28T01:00:00+00:00",
                    horizon_hours=6,
                    requestor="test_suite"
                )

                result = await env.client.execute_workflow(
                    IntradayForecastWorkflow.run,
                    request,
                    id="test-workflow-1",
                    task_queue="test-queue",
                )

                # Verify result
                assert isinstance(result, ForecastResult)
                assert result.device_id == request.device_id
                assert result.predictions_count == 24  # 6 hours * 4 intervals
                assert result.total_energy_kwh >= 0
                assert result.avg_power_wh >= 0
                assert result.model_version == "v1.0"

    @pytest.mark.asyncio
    async def test_workflow_returns_correct_prediction_count(self):
        """Test that workflow generates correct number of predictions."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IntradayForecastWorkflow],
                activities=[
                    forecast_activities.validate_request,
                    forecast_activities.fetch_historical_data,
                    forecast_activities.load_model_from_cache,
                    forecast_activities.generate_forecast,
                    forecast_activities.publish_forecast,
                    forecast_activities.store_audit_log,
                ],
            ):
                # Test different horizons
                for horizon in [1, 6, 12, 24]:
                    request = ForecastRequest(
                        device_id="21183900202529220938026983N5",
                        forecast_start="2025-02-28T01:00:00+00:00",
                        horizon_hours=horizon,
                        requestor="test_suite"
                    )

                    result = await env.client.execute_workflow(
                        IntradayForecastWorkflow.run,
                        request,
                        id=f"test-workflow-horizon-{horizon}",
                        task_queue="test-queue",
                    )

                    expected_count = horizon * 4
                    assert result.predictions_count == expected_count, \
                        f"Expected {expected_count} predictions for {horizon}h, got {result.predictions_count}"

    @pytest.mark.asyncio
    async def test_workflow_validates_horizon_range(self):
        """Test that workflow rejects invalid horizon."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IntradayForecastWorkflow],
                activities=[
                    forecast_activities.validate_request,
                    forecast_activities.fetch_historical_data,
                    forecast_activities.load_model_from_cache,
                    forecast_activities.generate_forecast,
                    forecast_activities.publish_forecast,
                    forecast_activities.store_audit_log,
                ],
            ):
                # Invalid horizon (too large)
                request = ForecastRequest(
                    device_id="21183900202529220938026983N5",
                    forecast_start="2025-02-28T01:00:00+00:00",
                    horizon_hours=100,  # Invalid
                    requestor="test_suite"
                )

                with pytest.raises(Exception):  # Should fail validation
                    await env.client.execute_workflow(
                        IntradayForecastWorkflow.run,
                        request,
                        id="test-workflow-invalid-horizon",
                        task_queue="test-queue",
                    )

    @pytest.mark.asyncio
    async def test_workflow_records_latency(self):
        """Test that workflow records execution latency."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IntradayForecastWorkflow],
                activities=[
                    forecast_activities.validate_request,
                    forecast_activities.fetch_historical_data,
                    forecast_activities.load_model_from_cache,
                    forecast_activities.generate_forecast,
                    forecast_activities.publish_forecast,
                    forecast_activities.store_audit_log,
                ],
            ):
                request = ForecastRequest(
                    device_id="21183900202529220938026983N5",
                    forecast_start="2025-02-28T01:00:00+00:00",
                    horizon_hours=6,
                    requestor="test_suite"
                )

                result = await env.client.execute_workflow(
                    IntradayForecastWorkflow.run,
                    request,
                    id="test-workflow-latency",
                    task_queue="test-queue",
                )

                # Latency should be recorded and reasonable
                assert result.latency_ms >= 0
                # In test environment with time skipping, latency may be very low

    @pytest.mark.asyncio
    async def test_workflow_handles_multiple_devices(self):
        """Test workflow with different device IDs."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[IntradayForecastWorkflow],
                activities=[
                    forecast_activities.validate_request,
                    forecast_activities.fetch_historical_data,
                    forecast_activities.load_model_from_cache,
                    forecast_activities.generate_forecast,
                    forecast_activities.publish_forecast,
                    forecast_activities.store_audit_log,
                ],
            ):
                devices = [
                    "21183900202529220938026983N5",
                    "21232200202609620933097655N9",
                    "21231500202609620933056238N3",
                ]

                for device_id in devices:
                    request = ForecastRequest(
                        device_id=device_id,
                        forecast_start="2025-02-28T01:00:00+00:00",
                        horizon_hours=6,
                        requestor="test_suite"
                    )

                    result = await env.client.execute_workflow(
                        IntradayForecastWorkflow.run,
                        request,
                        id=f"test-workflow-device-{device_id}",
                        task_queue="test-queue",
                    )

                    assert result.device_id == device_id
                    assert result.predictions_count == 24


class TestForecastRequestValidation:
    """Tests for ForecastRequest dataclass."""

    def test_forecast_request_creation(self):
        """Test creating ForecastRequest."""
        request = ForecastRequest(
            device_id="test_device",
            forecast_start="2025-02-28T01:00:00+00:00",
            horizon_hours=24,
            requestor="test_suite"
        )

        assert request.device_id == "test_device"
        assert request.forecast_start == "2025-02-28T01:00:00+00:00"
        assert request.horizon_hours == 24
        assert request.requestor == "test_suite"

    def test_forecast_request_defaults(self):
        """Test ForecastRequest default values."""
        request = ForecastRequest(
            device_id="test_device",
            forecast_start="2025-02-28T01:00:00+00:00",
        )

        assert request.horizon_hours == 24  # Default
        assert request.requestor == "trading_system"  # Default


class TestForecastResultValidation:
    """Tests for ForecastResult dataclass."""

    def test_forecast_result_creation(self):
        """Test creating ForecastResult."""
        result = ForecastResult(
            device_id="test_device",
            forecast_start="2025-02-28T01:00:00+00:00",
            predictions_count=96,
            total_energy_kwh=6.44,
            avg_power_wh=67.09,
            generated_at="2025-02-28T02:00:00+00:00",
            model_version="v1.0",
            latency_ms=500
        )

        assert result.device_id == "test_device"
        assert result.predictions_count == 96
        assert result.total_energy_kwh == 6.44
        assert result.avg_power_wh == 67.09
        assert result.model_version == "v1.0"
        assert result.latency_ms == 500
