"""
Temporal Worker - Executes forecast workflows and activities.

Run this worker to process intraday forecast requests.

Usage:
    python worker.py
"""

import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker

from workflows.intraday_forecast import IntradayForecastWorkflow
from activities import forecast_activities

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Start the Temporal worker."""

    # Connect to Temporal server
    # In production, use environment variables for connection
    client = await Client.connect(
        "localhost:7233",
        namespace="default",
    )

    logger.info("Connected to Temporal server")

    # Create worker that processes forecast requests
    worker = Worker(
        client,
        task_queue="forecast-queue",
        workflows=[IntradayForecastWorkflow],
        activities=[
            forecast_activities.validate_request,
            forecast_activities.fetch_historical_data,
            forecast_activities.load_model_from_cache,
            forecast_activities.generate_forecast,
            forecast_activities.publish_forecast,
            forecast_activities.store_audit_log,
        ],
        max_concurrent_activities=10,
        max_concurrent_workflow_tasks=10,
    )

    logger.info("Starting forecast worker on task queue: forecast-queue")
    logger.info("Worker will process IntradayForecastWorkflow requests")
    logger.info("Press Ctrl+C to stop")

    # Run worker until interrupted
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
