"""
Forecast Activities for Temporal Workflows

These activities are executed by Temporal workers and perform
the actual work of data fetching, model inference, and publishing.
"""

import pickle
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import sys

import pandas as pd
import numpy as np
from temporalio import activity

# Add parent directory to path for forecaster import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from forecaster import HeatPumpForecaster


# Simulated external dependencies
# In production, these would be real connections
class MockRedisClient:
    """Mock Redis client for demo."""

    def __init__(self):
        self.store = {}

    def get(self, key: str) -> Optional[bytes]:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: bytes):
        self.store[key] = value

    def exists(self, key: str) -> bool:
        return key in self.store


class MockClickHouseClient:
    """Mock ClickHouse client for demo."""

    def query_dataframe(self, query: str, params: Dict) -> pd.DataFrame:
        # In production, this would query actual ClickHouse
        # For demo, load from CSV
        csv_path = Path(__file__).parent.parent.parent / "data/sample_devices_actual_power_consumption.csv"
        df = pd.read_csv(csv_path)

        # Filter by device_id
        device_id = params.get('device_id')
        if device_id:
            df = df[df['serial_number'] == device_id]

        return df


class MockPubSubClient:
    """Mock Pub/Sub client for demo."""

    def publish(self, topic: str, data: bytes):
        activity.logger.info(f"Publishing to {topic}: {len(data)} bytes")
        return True


# Shared resources (initialized once per worker)
redis_client = MockRedisClient()
clickhouse_client = MockClickHouseClient()
pubsub_client = MockPubSubClient()
model_cache = {}  # In-memory model cache


@activity.defn
async def validate_request(request: Dict[str, Any]) -> bool:
    """
    Validate forecast request.

    Checks:
    - Device ID exists in system
    - Forecast time range is valid
    - Horizon is reasonable (1-72 hours)
    """
    device_id = request["device_id"]
    horizon_hours = request.get("horizon_hours", 24)

    activity.logger.info(f"Validating request for device {device_id}")

    # Check horizon
    if not (1 <= horizon_hours <= 72):
        raise ValueError(f"Invalid horizon: {horizon_hours} hours (must be 1-72)")

    # In production, check if device exists in database
    # For demo, always pass
    activity.logger.info(f"Validation passed for device {device_id}")

    return True


@activity.defn
async def fetch_historical_data(device_id: str) -> str:
    """
    Fetch historical data from ClickHouse with Redis caching.

    Returns JSON string of DataFrame (for Temporal serialization).
    """
    activity.logger.info(f"Fetching historical data for {device_id}")

    # Check Redis cache first
    cache_key = f"history:{device_id}:30d"
    cached = redis_client.get(cache_key)

    if cached:
        activity.logger.info(f"Cache HIT for {device_id}")
        return cached.decode('utf-8')

    activity.logger.info(f"Cache MISS for {device_id}, querying ClickHouse")

    # Query ClickHouse for last 30 days
    query = """
        SELECT * FROM power_consumption
        WHERE device_id = %(device_id)s
        AND timestamp >= now() - INTERVAL 30 DAY
        ORDER BY timestamp
    """

    df = clickhouse_client.query_dataframe(query, {'device_id': device_id})

    activity.logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns from ClickHouse")

    # Limit to last 7 days to reduce data size (for demo)
    # In production, you'd keep 30 days but store in object storage
    if len(df) > 672:  # 7 days * 24 hours * 4 intervals
        df = df.tail(672)

    # Convert to JSON for caching - only include necessary columns
    essential_cols = [col for col in df.columns if any(x in col for x in
                     ['serial_number', 'start', 'end', 'electricity', 'manufacturer', 'price_zone'])]
    df_subset = df[essential_cols] if essential_cols else df

    df_json = df_subset.to_json(orient='records', date_format='iso')

    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, df_json.encode('utf-8'))

    activity.logger.info(f"Fetched {len(df)} rows for {device_id} (returning {len(df_subset)} cols)")

    return df_json


@activity.defn
async def load_model_from_cache(device_id: str) -> Dict[str, Any]:
    """
    Load forecasting model from cache or storage.

    Cache hierarchy:
    1. Worker memory (fastest)
    2. Redis (fast)
    3. Cloud Storage (slowest)
    """
    activity.logger.info(f"Loading model for {device_id}")

    # Check worker memory cache
    cache_key = f"model:global:v1"  # In production, use device-specific models

    if cache_key in model_cache:
        activity.logger.info("Model found in worker memory")
        return {
            "version": "v1.0",
            "source": "memory",
            "device_id": device_id,
        }

    # Check Redis cache
    redis_key = f"model:{device_id}:v1"
    if redis_client.exists(redis_key):
        activity.logger.info("Model found in Redis")
        # In production, deserialize model from Redis
        model_cache[cache_key] = "model_object"
        return {
            "version": "v1.0",
            "source": "redis",
            "device_id": device_id,
        }

    # Load from Cloud Storage (slowest)
    activity.logger.info("Loading model from Cloud Storage")

    # In production, load from GCS
    # For demo, use local model
    model_path = Path(__file__).parent.parent.parent / "trained_model"

    if model_path.exists():
        forecaster = HeatPumpForecaster(model_path=model_path)
        model_cache[cache_key] = forecaster

        activity.logger.info("Model loaded and cached")

        return {
            "version": "v1.0",
            "source": "storage",
            "device_id": device_id,
        }

    raise FileNotFoundError(f"Model not found for {device_id}")


@activity.defn
async def generate_forecast(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate power consumption forecast using LightGBM model.

    This is the core forecasting activity.
    """
    device_id = input_data["device_id"]
    forecast_start = datetime.fromisoformat(input_data["forecast_start"])
    horizon_hours = input_data["horizon_hours"]
    historical_data_json = input_data["historical_data"]

    activity.logger.info(
        f"Generating {horizon_hours}h forecast for {device_id} starting {forecast_start}"
    )

    # Load historical data from JSON
    from io import StringIO
    historical_df = pd.read_json(StringIO(historical_data_json), orient='records')

    activity.logger.info(
        f"Historical data loaded: {len(historical_df)} rows, "
        f"{len(historical_df.columns)} columns: {list(historical_df.columns)}"
    )

    # Get model from cache
    cache_key = f"model:global:v1"
    forecaster = model_cache.get(cache_key)

    if forecaster is None:
        # If not in cache, load it
        model_path = Path(__file__).parent.parent.parent / "trained_model"
        forecaster = HeatPumpForecaster(model_path=model_path)
        model_cache[cache_key] = forecaster

    # Generate forecast
    forecast_df = forecaster.predict(
        device_id=device_id,
        start_time=forecast_start,
        horizon_hours=horizon_hours,
        historical_data=historical_df,
    )

    activity.logger.info(
        f"Forecast generated: {len(forecast_df)} predictions, "
        f"min={forecast_df['predicted_power'].min():.2f}, "
        f"max={forecast_df['predicted_power'].max():.2f}, "
        f"avg={forecast_df['predicted_power'].mean():.2f} Wh"
    )

    # Calculate summary statistics (don't return all predictions to avoid size limit)
    total_power = forecast_df['predicted_power'].sum()
    predictions_count = len(forecast_df)

    # For demo, only include first and last few predictions
    predictions_sample = []
    if predictions_count > 0:
        # First 3 and last 3 predictions as examples
        sample_df = pd.concat([forecast_df.head(3), forecast_df.tail(3)])
        predictions_sample = sample_df.to_dict('records')
        for pred in predictions_sample:
            if isinstance(pred['timestamp'], pd.Timestamp):
                pred['timestamp'] = pred['timestamp'].isoformat()

    activity.logger.info(f"Generated {predictions_count} predictions, total {total_power:.2f} Wh")

    return {
        "device_id": device_id,
        "predictions_count": predictions_count,
        "total_power_wh": total_power,
        "avg_power_wh": total_power / predictions_count if predictions_count > 0 else 0,
        "predictions_sample": predictions_sample,  # Just for demo/verification
        "forecast_start": forecast_start.isoformat(),
        "horizon_hours": horizon_hours,
    }


@activity.defn
async def publish_forecast(data: Dict[str, Any]) -> bool:
    """
    Publish forecast summary to trading system via Pub/Sub.

    Topic: intraday-forecasts
    Message format: JSON with device_id, summary stats, metadata

    Note: In production, full predictions would be stored in Cloud Storage
    and a reference/URL would be published here.
    """
    device_id = data["device_id"]
    predictions_count = data["predictions_count"]
    total_energy_kwh = data["total_energy_kwh"]
    avg_power_wh = data["avg_power_wh"]
    workflow_id = data["workflow_id"]

    activity.logger.info(f"Publishing forecast summary for {device_id} to Pub/Sub")

    message = {
        "device_id": device_id,
        "workflow_id": workflow_id,
        "predictions_count": predictions_count,
        "total_energy_kwh": total_energy_kwh,
        "avg_power_wh": avg_power_wh,
        "published_at": datetime.now(timezone.utc).isoformat(),
        # In production, add: "forecast_url": "gs://bucket/forecasts/{workflow_id}.parquet"
    }

    message_json = json.dumps(message)

    # Publish to Pub/Sub
    pubsub_client.publish(
        topic="projects/YOUR_PROJECT/topics/intraday-forecasts",
        data=message_json.encode('utf-8')
    )

    activity.logger.info(
        f"Published forecast summary: {predictions_count} predictions, "
        f"{total_energy_kwh:.2f} kWh total"
    )

    return True


@activity.defn
async def store_audit_log(data: Dict[str, Any]) -> bool:
    """
    Store audit log to PostgreSQL for compliance.

    Records:
    - Who requested the forecast
    - When it was generated
    - What data was used
    - Model version
    """
    workflow_id = data["workflow_id"]
    request = data["request"]
    result = data["result"]

    activity.logger.info(f"Storing audit log for workflow {workflow_id}")

    # In production, insert to PostgreSQL
    # For demo, just log
    audit_entry = {
        "workflow_id": workflow_id,
        "device_id": request["device_id"],
        "forecast_start": request["forecast_start"],
        "horizon_hours": request["horizon_hours"],
        "requestor": request["requestor"],
        "predictions_count": result["predictions_count"],
        "model_version": result["model_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    activity.logger.info(f"Audit log: {json.dumps(audit_entry, indent=2)}")

    return True
