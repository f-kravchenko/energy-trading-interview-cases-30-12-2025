"""
Intraday Forecast Workflow for Temporal.io

This workflow orchestrates on-demand power consumption forecasts
for intraday energy trading.
"""

from datetime import timedelta, datetime, timezone
from dataclasses import dataclass
from typing import List, Dict

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.converter import DataConverter

# Import activity functions (defined in activities module)
with workflow.unsafe.imports_passed_through():
    from activities.forecast_activities import (
        validate_request,
        fetch_historical_data,
        load_model_from_cache,
        generate_forecast,
        publish_forecast,
        store_audit_log,
    )


@dataclass
class ForecastRequest:
    """Request to generate a forecast."""
    device_id: str
    forecast_start: str  # ISO format datetime string
    horizon_hours: int = 24
    requestor: str = "trading_system"


@dataclass
class ForecastResult:
    """Forecast generation result."""
    device_id: str
    forecast_start: str  # ISO format datetime string
    predictions_count: int  # Number of predictions generated
    total_energy_kwh: float  # Total predicted energy for the period
    avg_power_wh: float  # Average predicted power
    generated_at: str  # ISO format datetime string
    model_version: str
    latency_ms: int
    # Note: Full predictions are published to Pub/Sub, not returned in result


@workflow.defn
class IntradayForecastWorkflow:
    """
    Orchestrates on-demand forecast generation for intraday trading.

    Workflow guarantees:
    - Exactly-once execution (idempotent)
    - Automatic retries with exponential backoff
    - Timeout: 60 seconds (fail fast for trading)
    - Full audit trail for compliance

    Steps:
    1. Validate request (device exists, time range valid)
    2. Fetch historical data (last 30 days from ClickHouse)
    3. Load model from cache (Redis) or storage (GCS)
    4. Generate forecast (LightGBM inference)
    5. Publish to trading system (Pub/Sub)
    6. Store audit log (PostgreSQL)
    """

    @workflow.run
    async def run(self, request: ForecastRequest) -> ForecastResult:
        """Execute the forecast workflow."""

        workflow_start = workflow.now()

        # Step 1: Validate request
        workflow.logger.info(
            f"Starting forecast workflow for device {request.device_id}"
        )

        await workflow.execute_activity(
            validate_request,
            request,
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(
                maximum_attempts=2,
                initial_interval=timedelta(seconds=1),
            ),
        )

        # Step 2: Fetch historical data (with retry for transient DB errors)
        workflow.logger.info("Fetching historical data from ClickHouse")

        historical_data_json = await workflow.execute_activity(
            fetch_historical_data,
            request.device_id,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
                backoff_coefficient=2.0,
            ),
        )

        # Step 3: Load model from cache
        workflow.logger.info("Loading forecasting model")

        model_info = await workflow.execute_activity(
            load_model_from_cache,
            request.device_id,
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(
                maximum_attempts=2,
                initial_interval=timedelta(milliseconds=500),
            ),
        )

        # Step 4: Generate forecast
        workflow.logger.info("Generating forecast predictions")

        forecast_data = await workflow.execute_activity(
            generate_forecast,
            {
                "device_id": request.device_id,
                "forecast_start": request.forecast_start,
                "horizon_hours": request.horizon_hours,
                "historical_data": historical_data_json,
                "model_version": model_info["version"],
            },
            start_to_close_timeout=timedelta(seconds=20),
            retry_policy=RetryPolicy(
                maximum_attempts=2,
                initial_interval=timedelta(seconds=1),
            ),
        )

        # Step 5: Publish forecast summary (fire-and-forget pattern)
        workflow.logger.info("Publishing forecast to trading system")

        await workflow.execute_activity(
            publish_forecast,
            {
                "device_id": request.device_id,
                "predictions_count": forecast_data["predictions_count"],
                "total_energy_kwh": forecast_data["total_power_wh"] / 1000,
                "avg_power_wh": forecast_data["avg_power_wh"],
                "workflow_id": workflow.info().workflow_id,
            },
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
            ),
        )

        # Step 6: Store audit log (async, don't fail workflow if this fails)
        try:
            await workflow.execute_activity(
                store_audit_log,
                {
                    "workflow_id": workflow.info().workflow_id,
                    "request": {
                        "device_id": request.device_id,
                        "forecast_start": request.forecast_start,
                        "horizon_hours": request.horizon_hours,
                        "requestor": request.requestor,
                    },
                    "result": {
                        "predictions_count": forecast_data["predictions_count"],
                        "model_version": model_info["version"],
                    },
                },
                start_to_close_timeout=timedelta(seconds=3),
                retry_policy=RetryPolicy(
                    maximum_attempts=1,  # Don't retry audit logs
                ),
            )
        except Exception as e:
            workflow.logger.warning(f"Failed to store audit log: {e}")

        # Calculate latency
        workflow_end = workflow.now()
        latency_ms = int((workflow_end - workflow_start).total_seconds() * 1000)

        # Extract summary statistics (activity already calculated them)
        predictions_count = forecast_data["predictions_count"]
        total_power_wh = forecast_data["total_power_wh"]
        total_energy_kwh = total_power_wh / 1000  # Convert Wh to kWh
        avg_power_wh = forecast_data["avg_power_wh"]

        workflow.logger.info(
            f"Forecast completed in {latency_ms}ms for device {request.device_id}: "
            f"{predictions_count} predictions, {total_energy_kwh:.2f} kWh total"
        )

        return ForecastResult(
            device_id=request.device_id,
            forecast_start=request.forecast_start,
            predictions_count=predictions_count,
            total_energy_kwh=total_energy_kwh,
            avg_power_wh=avg_power_wh,
            generated_at=workflow_end.isoformat(),
            model_version=model_info["version"],
            latency_ms=latency_ms,
        )
