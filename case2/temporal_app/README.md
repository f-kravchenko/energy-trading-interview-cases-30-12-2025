# Intraday Forecast - Temporal.io Implementation

This is a working implementation of on-demand power consumption forecasting for intraday energy trading using Temporal.io.

## Architecture

```
Trading System → Temporal Workflow → Activities → Forecast Result
                      ↓
            [Validate, Fetch, Predict, Publish]
```

## Components

### 1. Workflows (`workflows/`)
- **IntradayForecastWorkflow**: Orchestrates the entire forecast process
  - Handles retries, timeouts, and error recovery
  - Guarantees exactly-once execution
  - Provides full audit trail

### 2. Activities (`activities/`)
- **validate_request**: Check device ID and parameters
- **fetch_historical_data**: Load 30 days of data from ClickHouse (with Redis cache)
- **load_model_from_cache**: Load LightGBM model (memory → Redis → Cloud Storage)
- **generate_forecast**: Run model inference
- **publish_forecast**: Send to trading system via Pub/Sub
- **store_audit_log**: Save execution details for compliance

### 3. Worker (`worker.py`)
- Connects to Temporal server
- Processes workflows and activities from task queue
- Can be horizontally scaled (run multiple workers)

### 4. Client (`client_example.py`)
- Example code to trigger forecasts
- Shows single and parallel forecast execution

## Quick Start

### 1. Install Dependencies

```bash
cd temporal_app
uv sync
```

### 2. Start Temporal Server (Local Development)

```bash
# Option A: Using Temporal CLI (recommended for local dev)
brew install temporal
temporal server start-dev

# Option B: Using Docker
docker-compose up -d
```

The Temporal Web UI will be available at: http://localhost:8233

### 3. Start Worker

In one terminal:

```bash
cd temporal_app
uv run python worker.py
```

You should see:
```
INFO:__main__:Connected to Temporal server
INFO:__main__:Starting forecast worker on task queue: forecast-queue
INFO:__main__:Worker will process IntradayForecastWorkflow requests
INFO:__main__:Press Ctrl+C to stop
```

### 4. Trigger Forecast

In another terminal:

```bash
cd temporal_app
uv run python client_example.py
```

This will:
1. Start a forecast workflow
2. Process it through activities
3. Return the forecast result

## Example Output

```
============================================================
Triggering intraday forecast workflow
============================================================
Workflow ID: intraday-21183900202529220938026983N5-1735506789
Device ID: 21183900202529220938026983N5
Forecast Start: 2025-12-30 22:13:09+00:00
Horizon: 24 hours
============================================================

Workflow started! Waiting for result...

============================================================
Forecast completed successfully!
============================================================
Device: 21183900202529220938026983N5
Generated at: 2025-12-30 21:13:15.234567+00:00
Latency: 5234ms
Model version: v1.0
Predictions count: 96

First 5 predictions:
  2025-12-30T22:13:09: 245.32 Wh
  2025-12-30T22:28:09: 243.87 Wh
  2025-12-30T22:43:09: 241.54 Wh
  2025-12-30T22:58:09: 239.12 Wh
  2025-12-30T23:13:09: 236.89 Wh
============================================================
```

## Monitoring

### Temporal Web UI

Visit http://localhost:8233 to see:

- **Workflows**: All executed workflows with status
- **Task Queues**: Current queue depth and worker count
- **Activity History**: Step-by-step execution details
- **Error Stack Traces**: Debug failed workflows

### Metrics

Key metrics to monitor:

```python
# Workflow metrics
temporal_workflow_started_total
temporal_workflow_completed_total
temporal_workflow_failed_total
temporal_workflow_timeout_total

# Activity metrics
temporal_activity_execution_latency_seconds
temporal_activity_task_queue_depth

# Worker metrics
temporal_worker_task_slots_available
```

## Scaling

### Horizontal Scaling

Run multiple workers to handle more load:

```bash
# Terminal 1
uv run python worker.py

# Terminal 2
uv run python worker.py

# Terminal 3
uv run python worker.py
```

Temporal automatically distributes work across all workers.

### Auto-scaling

In production (GKE), use HPA:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: forecast-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: forecast-workers
  minReplicas: 2
  maxReplicas: 100
  metrics:
  - type: Pods
    pods:
      metric:
        name: temporal_task_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
```

## Testing

Comprehensive test suite with 33 tests covering forecaster, activities, and workflows.

### Install Test Dependencies

```bash
cd temporal_app
uv sync --extra dev
```

### Run All Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_forecaster.py
uv run pytest tests/test_activities.py
uv run pytest tests/test_workflows.py

# Run with test summary
uv run pytest tests/ -ra
```

**Expected Output**:
```
============================== test session starts ===============================
collected 33 items

tests/test_activities.py ................                                 [ 48%]
tests/test_forecaster.py ........                                         [ 72%]
tests/test_workflows.py .........                                         [100%]

============================== 33 passed in 4.32s ================================
```

**Test Coverage**:
- **8 tests** - Forecaster unit tests ([test_forecaster.py](tests/test_forecaster.py))
- **16 tests** - Activity unit tests ([test_activities.py](tests/test_activities.py))
- **9 tests** - Workflow integration tests ([test_workflows.py](tests/test_workflows.py))

**Runtime**: ~4-10 seconds total

**Note**: If you get `zsh: command not found: pytest`, you must use `uv run pytest` instead of `pytest` directly, as pytest is installed in the virtual environment managed by uv.

See [tests/README.md](tests/README.md) for detailed test documentation.

### Load Testing

Trigger 1000 forecasts in parallel:

```python
async def load_test():
    device_id = "21183900202529220938026983N5"
    tasks = [trigger_forecast(device_id, horizon_hours=6) for _ in range(1000)]
    results = await asyncio.gather(*tasks)
    print(f"Avg latency: {sum(r.latency_ms for r in results) / len(results):.0f}ms")

asyncio.run(load_test())
```

## Production Deployment

### 1. Deploy Temporal Cluster

```bash
# Create GKE cluster
gcloud container clusters create-auto temporal-prod \
    --region=europe-west1

# Install Temporal via Helm
helm install temporal temporal/temporal \
    --namespace temporal \
    --create-namespace \
    --values temporal-values.yaml
```

### 2. Deploy Workers

```bash
# Build Docker image
docker build -t gcr.io/YOUR_PROJECT/forecast-worker:v1 .
docker push gcr.io/YOUR_PROJECT/forecast-worker:v1

# Deploy to GKE
kubectl apply -f k8s/worker-deployment.yaml
```

### 3. Configure Cloud Resources

```bash
# ClickHouse for timeseries data
# Redis Memorystore for caching
# Cloud Storage for models
# Pub/Sub for trading system integration
```

## Architecture Benefits

### vs Traditional Task Queues (Celery, RQ)

| Feature | Temporal.io | Celery |
|---------|-------------|--------|
| **Durability** | ✅ Workflows survive crashes | ❌ Tasks lost if worker dies |
| **Retries** | ✅ Built-in exponential backoff | ⚠️ Manual implementation |
| **Observability** | ✅ Full execution history in UI | ❌ No built-in visibility |
| **State Management** | ✅ Workflow state persisted | ❌ Must use external DB |
| **Debugging** | ✅ Replay workflows locally | ❌ Hard to debug production |
| **Versioning** | ✅ Workflow version management | ⚠️ Manual migration |

### vs Cloud Tasks / Cloud Run

| Feature | Temporal.io | Cloud Tasks |
|---------|-------------|-------------|
| **Stateful Workflows** | ✅ Multi-step with state | ❌ Stateless tasks only |
| **Complex Orchestration** | ✅ Sagas, child workflows | ❌ Simple retry logic |
| **Local Development** | ✅ Full local testing | ⚠️ Cloud emulator limited |
| **Vendor Lock-in** | ✅ Open source, portable | ❌ GCP only |

## Key Features Demonstrated

### 1. Exactly-Once Execution
Workflows are idempotent - safe to retry without duplicates

### 2. Automatic Retries
Activities retry with exponential backoff on transient failures

### 3. Timeout Handling
60-second workflow timeout ensures fast failure for trading

### 4. Saga Pattern
Audit log failure doesn't fail entire workflow

### 5. Parallel Execution
Multiple forecasts processed concurrently by workers

### 6. Observability
Full execution history in Temporal UI

### 7. Testability
Workflows can be unit tested with mocked activities

## Cost Estimation

For 50,000 forecasts/day:

| Component | Cost/month |
|-----------|-----------|
| Temporal cluster (GKE) | $200 |
| Workers (2-100 pods) | $400 |
| PostgreSQL (Temporal state) | $100 |
| **Total** | **$700** |

## Troubleshooting

### Issue: Negative Power Predictions

**Symptom**: Some devices show negative predicted power values

**Root Cause**: LightGBM model can produce negative predictions for devices with sparse/low consumption patterns

**Solution**: Applied post-processing clipping in [forecaster.py:350](../forecaster.py#L350):
```python
# Ensure non-negative (power consumption cannot be negative)
pred = max(0.0, pred)
```

**Production fix**: Retrain model with tree constraints or use monotonic constraints in LightGBM

### Issue: Result Exceeds Size Limit

**Symptom**: `Complete result exceeds size limit` error

**Root Cause**: Temporal has a 2MB limit on workflow results

**Solution**: Return summary statistics instead of all predictions:
```python
# Instead of returning all 96 predictions:
return {
    "predictions_count": 96,
    "total_energy_kwh": 6.44,
    "avg_power_wh": 67.09
}
```

Full predictions stored in Cloud Storage in production.

### Issue: Forecast Date in Future

**Symptom**: Model produces poor predictions

**Root Cause**: Demo uses Feb 28, 2025 as "now" because historical data ends Feb 27, 2025

**Solution**: In [client_example.py:37](client_example.py#L37):
```python
# For demo
demo_now = datetime(2025, 2, 28, 0, 0, tzinfo=timezone.utc)

# In production, use:
# demo_now = datetime.now(timezone.utc)
```

## Next Steps

1. **Add Weather Data**: Integrate forecast.solar or Open-Meteo API
2. **Implement Caching**: Redis for models and features
3. **Add Monitoring**: Prometheus + Grafana dashboards
4. **Load Testing**: Benchmark 1000 concurrent forecasts
5. **Production Deploy**: GKE + Temporal Cloud

## Documentation

- [Temporal Docs](https://docs.temporal.io/)
- [Python SDK](https://docs.temporal.io/dev-guide/python)
- [Best Practices](https://docs.temporal.io/dev-guide/best-practices)

## Support

For questions about this implementation, see:
- [intraday_trading_architecture.md](../intraday_trading_architecture.md) - Full architecture details
- Temporal community: https://community.temporal.io/
