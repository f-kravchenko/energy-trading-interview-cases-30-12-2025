"""
Example client to trigger intraday forecasts.

This demonstrates how to trigger forecast workflows from
the trading system or API.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from temporalio.client import Client
from workflows.intraday_forecast import (
    IntradayForecastWorkflow,
    ForecastRequest,
)


async def trigger_forecast(device_id: str, horizon_hours: int = 24):
    """
    Trigger an intraday forecast for a device.

    Args:
        device_id: Heat pump device identifier
        horizon_hours: How many hours ahead to forecast

    Returns:
        ForecastResult with predictions
    """
    # Connect to Temporal
    client = await Client.connect("localhost:7233")

    # Create unique workflow ID
    workflow_id = f"intraday-{device_id}-{int(datetime.now().timestamp())}"

    # Define forecast request
    # NOTE: For demo, using Feb 28, 2025 as "now" since historical data ends Feb 27, 2025
    # In production, this would use actual datetime.now()
    demo_now = datetime(2025, 2, 28, 0, 0, tzinfo=timezone.utc)
    request = ForecastRequest(
        device_id=device_id,
        forecast_start=(demo_now + timedelta(hours=1)).isoformat(),
        horizon_hours=horizon_hours,
        requestor="trading_system"
    )

    print(f"\n{'='*60}")
    print(f"Triggering intraday forecast workflow")
    print(f"{'='*60}")
    print(f"Workflow ID: {workflow_id}")
    print(f"Device ID: {device_id}")
    print(f"Forecast Start: {request.forecast_start}")
    print(f"Horizon: {horizon_hours} hours")
    print(f"{'='*60}\n")

    # Start workflow
    handle = await client.start_workflow(
        IntradayForecastWorkflow.run,
        request,
        id=workflow_id,
        task_queue="forecast-queue",
        execution_timeout=timedelta(seconds=60),
    )

    print(f"Workflow started! Waiting for result...\n")

    # Wait for result (or use handle.result() for async)
    result = await handle.result()

    print(f"\n{'='*60}")
    print(f"Forecast completed successfully!")
    print(f"{'='*60}")
    print(f"Device: {result.device_id}")
    print(f"Generated at: {result.generated_at}")
    print(f"Latency: {result.latency_ms}ms")
    print(f"Model version: {result.model_version}")
    print(f"\nForecast Summary:")
    print(f"  Predictions: {result.predictions_count} intervals")
    print(f"  Total energy: {result.total_energy_kwh:.2f} kWh")
    print(f"  Average power: {result.avg_power_wh:.2f} Wh")
    print(f"\nNote: Full predictions published to Pub/Sub topic")
    print(f"      Check Temporal Web UI (http://localhost:8080) for sample predictions")
    print(f"{'='*60}\n")

    return result


async def trigger_multiple_forecasts():
    """
    Example: Trigger forecasts for multiple devices in parallel.

    This simulates intraday trading where many forecasts are needed
    when market prices change.
    """
    devices = [
        "21183900202529220938026983N5",
        "21232200202609620933097655N9",
        "21231500202609620933056238N3",
    ]

    print(f"\n{'='*60}")
    print(f"Triggering forecasts for {len(devices)} devices in parallel")
    print(f"This simulates intraday trading scenario")
    print(f"{'='*60}\n")

    # Trigger all forecasts in parallel
    tasks = [trigger_forecast(device_id, horizon_hours=6) for device_id in devices]

    results = await asyncio.gather(*tasks)

    print(f"\n{'='*60}")
    print(f"All forecasts completed!")
    print(f"{'='*60}")
    print(f"Total devices: {len(results)}")
    print(f"Average latency: {sum(r.latency_ms for r in results) / len(results):.0f}ms")
    print(f"{'='*60}\n")


async def check_workflow_status(workflow_id: str):
    """
    Check the status of a running workflow.

    Useful for monitoring long-running forecasts.
    """
    client = await Client.connect("localhost:7233")

    try:
        handle = client.get_workflow_handle(workflow_id)
        result = await handle.result()

        print(f"Workflow {workflow_id} completed successfully")
        print(f"Latency: {result.latency_ms}ms")
        return result

    except Exception as e:
        print(f"Workflow {workflow_id} failed or not found: {e}")
        return None


if __name__ == "__main__":
    # Example 1: Single forecast
    print("Example 1: Single forecast")
    asyncio.run(trigger_forecast(
        device_id="21183900202529220938026983N5",
        horizon_hours=24
    ))

    # Example 2: Multiple parallel forecasts
    print("\n\nExample 2: Multiple parallel forecasts")
    asyncio.run(trigger_multiple_forecasts())
