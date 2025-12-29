# Heat Pump Power Consumption Forecasting

## Executive Summary

This document provides a comprehensive solution for forecasting heat pump power consumption at 15-minute intervals for the next 24 hours. The forecast is required at 09:00 AM on D-1 for day-ahead power market trading, with additional on-demand forecasting capability for intraday trading.

**Key Requirements:**
- Forecast 96 intervals (15-minute granularity) for next 24 hours
- Day-ahead forecasting: Scheduled at 09:00 AM daily
- Intraday forecasting: On-demand, multiple times per day
- Handle heterogeneity across devices and heating types (space heating vs. DHW)
- Production-ready, scalable architecture on GCP

**Proposed Solution:**
- **Model:** LightGBM gradient boosting with weather-enhanced features
- **Expected Performance:** 10-12% MAPE (Mean Absolute Percentage Error)
- **Architecture:** GCP-based with Temporal workflows, ClickHouse analytics, FastAPI serving
- **Deployment:** Kubernetes on GKE with model caching for low-latency inference

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Data Understanding](#data-understanding)
3. [Forecasting Methodology](#forecasting-methodology)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Performance Evaluation](#performance-evaluation)
7. [Handling Device Heterogeneity](#handling-device-heterogeneity)
8. [Production Architecture](#production-architecture)
9. [Implementation Plan](#implementation-plan)

---

## Problem Analysis

### Business Context

**Day-Ahead Market:**
- Power markets open for trading at 09:00 AM on D-1
- Need accurate forecast of next day's consumption to:
  - Submit bids to electricity markets
  - Optimize energy procurement costs
  - Plan grid balancing

**Intraday Market:**
- Markets allow trading closer to delivery time
- Need to quickly re-forecast based on:
  - Updated weather forecasts
  - Actual consumption deviating from forecast
  - Price opportunities in intraday markets

### Forecasting Challenges

1. **Multiple seasonalities:**
   - Daily pattern (low at night, peaks morning/evening)
   - Weekly pattern (weekday vs. weekend)
   - Seasonal pattern (higher in winter)

2. **Weather dependence:**
   - Outdoor temperature is primary driver
   - Heat pumps consume more when it's colder
   - COP varies with temperature difference

3. **User behavior:**
   - Domestic hot water usage (morning/evening peaks)
   - Thermostat settings
   - Occupancy patterns

4. **Device heterogeneity:**
   - Different heat pump models/manufacturers
   - Different building characteristics (insulation, size)
   - Different heating modes (space heating, DHW, cooling)

5. **Data quality:**
   - Missing values
   - Outliers (sensor failures)
   - Irregular reporting intervals

---

## Data Understanding

### Available Datasets

#### 1. Actual Power Consumption (`sample_devices_actual_power_consumption.csv`)

**Size:** ~61,400 rows

**Granularity:** Hourly aggregations

**Key Fields:**
```
- serial_number: Device identifier
- start, end: Time interval (hourly)
- centralHeating.electricity: Electricity consumed for space heating (Wh)
- centralHeating.environmentalYield: Heat extracted from environment (Wh)
- centralHeating.generated: Total heat output for space heating (Wh)
- domesticHotWater.electricity: Electricity for DHW (Wh)
- domesticHotWater.generated: Heat output for DHW (Wh)
- cooling.electricity: Electricity for cooling (if applicable)
- manufacturer: OEM (e.g., Vaillant)
- price_zone: Market zone (e.g., AT - Austria)
```

**Coverage:** November 2024 data for 22 devices

**Target Variable:**
```python
total_electricity = (
    centralHeating.electricity +
    domesticHotWater.electricity +
    cooling.electricity
)
```

#### 2. Device Telemetry (`sample_devices_datapoints.csv`)

**Size:** ~260,000 rows

**Granularity:** Irregular (every few minutes)

**Key Fields:**
```
- time: Timestamp of measurement
- outdoor_temp: Outdoor temperature (°C)
- indoor_temp_act: Actual indoor temperature (°C)
- indoor_temp_req: Requested/setpoint temperature (°C)
- hot_water_temp: Hot water tank temperature (°C)
- hot_water_temp_req: Hot water setpoint (°C)
- elec_power: Instantaneous electrical power (W)
- thermal_power: Instantaneous thermal output (W)
- compressor_frequency_perc: Compressor speed (%)
- operating_state: Current mode
- heating_operating_state: Space heating status
- hot_water_operating_state: DHW status
```

**Use Cases:**
- Real-time outdoor temperature
- Derive COP (thermal_power / elec_power)
- Identify heating vs. DHW operation
- Feature engineering for ML models

#### 3. User Locations (`sample_users_lat_lon.csv`)

**Size:** 22 devices

**Fields:**
```
- id: Device ID
- owner_id: User ID
- latitude, longitude: GPS coordinates
```

**Use Cases:**
- Fetch location-specific weather forecasts
- Identify microclimates
- Group devices by region

### Data Quality Assessment

**Completeness:**
- Power consumption: Mostly complete, hourly intervals
- Telemetry: Irregular intervals, some missing sensors
- Location: 3 devices missing coordinates

**Accuracy:**
- Some zero-value intervals (heat pump off)
- Outliers likely sensor errors (e.g., negative values)
- Environmental yield sometimes missing

**Consistency:**
- Timestamps are UTC
- Need to align hourly consumption with irregular telemetry

---

## Forecasting Methodology

### Approach Selection

We evaluated three approaches:

| Approach | Complexity | Accuracy | Training Time | Inference Time | Recommended |
|----------|-----------|----------|---------------|----------------|-------------|
| **Naive (Last Week)** | Very Low | 20-30% MAPE | None | <1ms | Baseline only |
| **LightGBM** | Low-Medium | 10-15% MAPE | 30-60s | 50-100ms | ✅ **Yes** |
| **Temporal Fusion Transformer** | High | 7-10% MAPE | 2-4 hours | 200-500ms | Future upgrade |

**Selected: LightGBM Gradient Boosting**

**Rationale:**
1. **Best accuracy/complexity tradeoff** for production
2. **Fast training** - can retrain daily with new data
3. **Fast inference** - sub-second for 96 intervals
4. **Handles missing values** natively
5. **Feature importance** - interpretable for debugging
6. **Proven** - widely used for energy forecasting (Kaggle, industry)

### Model Architecture

**Type:** Multi-output regression (or sequential single-step)

**Two strategies:**

#### Strategy A: Direct Multi-Horizon Forecasting
Train 96 separate models, one per 15-minute interval:
```
Model_00:00 → Predict consumption at 00:00
Model_00:15 → Predict consumption at 00:15
...
Model_23:45 → Predict consumption at 23:45
```

**Pros:** Each model specializes for that time slot
**Cons:** 96 models to train/maintain

#### Strategy B: Autoregressive (Recommended)
Train single model that predicts next interval:
```
Loop for 96 steps:
    features = extract_features(historical_data)
    prediction = model.predict(features)
    historical_data.append(prediction)  # Feed back
```

**Pros:** Single model, learns temporal dependencies
**Cons:** Error accumulates over horizon

**Hybrid (Best):** Train one model per hour (24 models), predict 4x15min intervals

---

## Feature Engineering

### Feature Categories

#### 1. Temporal Features (Always Available)

**Calendar features:**
```python
hour = timestamp.hour  # 0-23
day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday
day_of_month = timestamp.day  # 1-31
month = timestamp.month  # 1-12
quarter_hour = timestamp.minute // 15  # 0,1,2,3
is_weekend = day_of_week >= 5
is_night = (hour >= 22) | (hour < 6)
```

**Cyclical encoding** (captures periodicity):
```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
dow_sin = np.sin(2 * np.pi * day_of_week / 7)
dow_cos = np.cos(2 * np.pi * day_of_week / 7)
```

#### 2. Lag Features (Historical Consumption)

**Recent history:**
```python
power_lag_15min = power.shift(1)   # Last interval
power_lag_1h = power.shift(4)      # 1 hour ago
power_lag_2h = power.shift(8)      # 2 hours ago
power_lag_24h = power.shift(96)    # Same time yesterday
power_lag_7d = power.shift(96*7)   # Same time last week
```

**Rolling statistics:**
```python
power_roll_mean_1h = power.rolling(4).mean()
power_roll_std_1h = power.rolling(4).std()
power_roll_max_24h = power.rolling(96).max()
power_roll_mean_7d = power.rolling(96*7).mean()
```

**Exponential moving average:**
```python
power_ema_1h = power.ewm(span=4).mean()
power_ema_24h = power.ewm(span=96).mean()
```

#### 3. Weather Features (High Impact!)

**Current conditions:**
```python
outdoor_temp = telemetry['outdoor_temp']
temp_lag_1h = outdoor_temp.shift(4)
temp_change_1h = outdoor_temp - temp_lag_1h
```

**Forecast (from weather API):**
```python
outdoor_temp_forecast_1h = weather_api.get_forecast(lat, lon, +1h)
outdoor_temp_forecast_6h = weather_api.get_forecast(lat, lon, +6h)
outdoor_temp_forecast_24h = weather_api.get_forecast(lat, lon, +24h)
```

**Derived weather features:**
```python
heating_degree_days = max(0, 18 - outdoor_temp)  # HDD
temp_indoor_outdoor_diff = indoor_temp - outdoor_temp
```

**Weather forecast sources:**
- Open-Meteo API (free, excellent for Europe)
- DWD (German Weather Service) - high quality for DACH region
- ECMWF (European Centre) - premium quality

#### 4. Device-Specific Features

**Device characteristics:**
```python
device_id = categorical  # One-hot or embedding
manufacturer = categorical  # Vaillant, etc.
price_zone = categorical  # AT (Austria)

# Learned device behavior (from historical data)
device_avg_power = df.groupby('device_id')['power'].mean()
device_daily_pattern = df.groupby(['device_id', 'hour'])['power'].mean()
```

**Building thermal properties:**
```python
thermal_resistance = learned_from_data  # K/W
thermal_capacity = learned_from_data    # Wh/K
typical_cop = learned_from_data         # Coefficient of performance
```

#### 5. Heating Type Split

**Separate models or features for:**
```python
heating_electricity_share = heating_elec / total_elec
dhw_electricity_share = dhw_elec / total_elec

is_dhw_time = (hour >= 6) & (hour <= 9) | (hour >= 18) & (hour <= 22)
is_heating_time = (hour >= 5) & (hour <= 23)
```

#### 6. Special Events (Optional)

```python
is_holiday = timestamp in public_holidays
is_school_vacation = timestamp in vacation_periods
```

### Feature Importance (Expected)

Based on similar energy forecasting projects:

| Feature | Expected Importance | Notes |
|---------|-------------------|-------|
| `power_lag_24h` | 25-30% | Same time yesterday |
| `outdoor_temp_forecast` | 20-25% | Primary driver |
| `hour` | 15-20% | Daily pattern |
| `power_lag_7d` | 10-15% | Weekly seasonality |
| `day_of_week` | 5-10% | Weekend effect |
| `device_id` | 5-10% | Device heterogeneity |
| Others | <5% each | Fine-tuning |

---

## Model Development

### LightGBM Implementation

#### Model Configuration

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Hyperparameters (tuned via cross-validation)
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42
}

# Train/validation split
train_end = '2024-11-24'
val_start = '2024-11-24'
val_end = '2024-11-30'

train_data = lgb.Dataset(
    X_train,
    y_train,
    categorical_feature=['device_id', 'manufacturer', 'price_zone']
)

val_data = lgb.Dataset(
    X_val,
    y_val,
    reference=train_data
)

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)
```

#### Handling Missing Weather Data

```python
# If weather forecast unavailable, use historical average
def get_weather_forecast(lat, lon, timestamp):
    try:
        forecast = weather_api.get(lat, lon, timestamp)
        return forecast['temperature']
    except Exception:
        # Fallback: use average at same time from last 7 days
        historical_avg = historical_weather[
            (same_hour) & (last_7_days)
        ].mean()
        return historical_avg
```

#### Separate Models for Heating vs. DHW

```python
# Train two models
model_heating = lgb.train(params, heating_data)
model_dhw = lgb.train(params, dhw_data)

# Predict separately and sum
forecast_heating = model_heating.predict(X_forecast)
forecast_dhw = model_dhw.predict(X_forecast)
forecast_total = forecast_heating + forecast_dhw
```

### Hyperparameter Tuning

**Method:** Optuna (Bayesian optimization)

```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
    }

    model = lgb.train(params, train_data, num_boost_round=500)
    predictions = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, predictions)

    return mape

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
```

**Expected tuning time:** 30-60 minutes

---

## Performance Evaluation

### Metrics

#### 1. Mean Absolute Percentage Error (MAPE)

**Primary metric** for energy forecasting:

```python
MAPE = (1/n) * Σ |actual - predicted| / actual * 100%
```

**Target:** <12% MAPE

**Interpretation:**
- <10%: Excellent
- 10-15%: Good
- 15-25%: Acceptable
- >25%: Poor

#### 2. Root Mean Squared Error (RMSE)

```python
RMSE = √[(1/n) * Σ(actual - predicted)²]
```

**Units:** Wh (same as target variable)

**Use:** Penalizes large errors more than MAPE

#### 3. Mean Absolute Error (MAE)

```python
MAE = (1/n) * Σ|actual - predicted|
```

**Units:** Wh

**Use:** More robust to outliers than RMSE

#### 4. Forecast Skill Score

```python
Skill = 1 - (MAPE_model / MAPE_baseline)
```

Where baseline = naive forecast (last week same time)

**Target:** Skill > 0.4 (40% better than baseline)

### Cross-Validation Strategy

**Time Series Cross-Validation** (no random splits!):

```
Fold 1: Train on Week 1-2, Test on Week 3
Fold 2: Train on Week 1-3, Test on Week 4
Fold 3: Train on Week 1-4, Test on Week 5
...
```

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = lgb.train(params, lgb.Dataset(X_train, y_train))
    predictions = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, predictions)
    scores.append(mape)

print(f"Average MAPE: {np.mean(scores):.2f}% ± {np.std(scores):.2f}%")
```

### Expected Performance

| Scenario | MAPE | RMSE (Wh) | MAE (Wh) |
|----------|------|-----------|----------|
| **Naive baseline** (last week) | 25-30% | 500-600 | 350-400 |
| **LightGBM (time features only)** | 15-18% | 350-400 | 250-300 |
| **LightGBM + weather** | 10-12% | 250-300 | 180-220 |
| **LightGBM + weather + device features** | 8-10% | 200-250 | 150-180 |
| **Ensemble (LightGBM + LSTM)** | 7-9% | 180-220 | 140-160 |

### Error Analysis

**Breakdown by time of day:**
```python
results_by_hour = df.groupby('hour').agg({
    'actual': 'mean',
    'predicted': 'mean',
    'error': 'mean',
    'abs_error': 'mean'
})
```

**Identify problematic intervals:**
- Highest errors likely during DHW heating (morning/evening)
- Transition periods (heating on/off)
- Extreme weather events

**Per-device performance:**
```python
results_by_device = df.groupby('device_id').agg({
    'mape': 'mean',
    'rmse': 'mean'
})
```

Some devices will be harder to predict (irregular usage, poor data quality)

---

## Handling Device Heterogeneity

### Challenge

Different devices have different:
1. **Consumption patterns** (usage behavior varies by household)
2. **Building characteristics** (insulation, size, thermal mass)
3. **Heat pump models** (efficiency, control logic)
4. **Data quality** (some devices report more reliably)
5. **Heating types** (space heating only vs. space + DHW)

### Strategies

#### 1. Device-Specific Features (Recommended)

```python
# Treat device_id as categorical feature
X['device_id'] = df['device_id'].astype('category')

# LightGBM learns different behavior per device
model = lgb.train(params, train_data, categorical_feature=['device_id'])
```

**Pros:** Single model, shares patterns across devices
**Cons:** Need sufficient data per device

#### 2. Device Clustering

**Group similar devices:**

```python
from sklearn.cluster import KMeans

# Features for clustering
device_features = df.groupby('device_id').agg({
    'power': ['mean', 'std', 'max'],
    'outdoor_temp': 'mean',
    'heating_electricity_share': 'mean',
    'dhw_electricity_share': 'mean'
})

# Cluster into 3-5 groups
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(device_features)

# Add cluster as feature
df['device_cluster'] = df['device_id'].map(cluster_mapping)
```

**Cluster interpretations:**
- Cluster 0: High consumption, space heating dominated
- Cluster 1: Low consumption, well-insulated homes
- Cluster 2: DHW dominated (apartments, small spaces)
- Cluster 3: Variable/irregular usage

#### 3. Separate Models Per Heating Type

**Split models:**

```python
# Model 1: Space heating only
heating_data = df[df['heating_electricity'] > 0]
model_heating = lgb.train(params, heating_data)

# Model 2: DHW only
dhw_data = df[df['dhw_electricity'] > 0]
model_dhw = lgb.train(params, dhw_data)

# Predict
forecast_heating = model_heating.predict(X)
forecast_dhw = model_dhw.predict(X)
forecast_total = forecast_heating + forecast_dhw
```

**Why this works:**
- Space heating: Strongly correlated with outdoor temp, time of day
- DHW: More dependent on user behavior (showers, cooking), less weather-sensitive

#### 4. Hierarchical Forecasting

**Two-level approach:**

**Level 1:** Forecast total consumption across all devices
```python
model_global = lgb.train(params, all_devices_data)
total_forecast = model_global.predict(X)
```

**Level 2:** Forecast per-device share of total
```python
device_share[device_id] = device_consumption / total_consumption
device_forecast = total_forecast * device_share[device_id]
```

**Ensures:** Individual forecasts sum to total (reconciliation)

### Handling Missing Device Data

**Cold start problem:** New device with no historical data

**Solution:**

```python
def predict_new_device(device_id, timestamp):
    # Use average of similar devices (same cluster)
    cluster = infer_cluster(device_metadata)
    similar_devices = devices_in_cluster[cluster]

    # Weighted average by similarity
    forecast = 0
    for similar_device in similar_devices:
        similarity = compute_similarity(device_id, similar_device)
        forecast += similarity * model.predict(similar_device, timestamp)

    return forecast / sum(similarities)
```

---

## Production Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Forecasting System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │  Scheduled   │      │  On-Demand   │      │   Model      │ │
│  │  Forecast    │      │  Forecast    │      │  Training    │ │
│  │  (09:00)     │      │  (Anytime)   │      │  Pipeline    │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                      │                      │         │
│         └──────────────────────┼──────────────────────┘         │
│                                │                                │
│                     ┌──────────▼──────────┐                    │
│                     │  Temporal Workflow  │                    │
│                     └──────────┬──────────┘                    │
│                                │                                │
│         ┌──────────────────────┼──────────────────────┐        │
│         │                      │                      │        │
│  ┌──────▼──────┐     ┌────────▼────────┐   ┌────────▼──────┐ │
│  │  ClickHouse │     │  Weather API    │   │  Redis Cache  │ │
│  │  (Telemetry)│     │  (Forecast)     │   │  (Hot Model)  │ │
│  └─────────────┘     └─────────────────┘   └───────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │             FastAPI Model Serving                       │  │
│  │  - Load model from Redis/GCS                            │  │
│  │  - Feature engineering                                  │  │
│  │  - Inference (<500ms)                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │             PostgreSQL                                   │  │
│  │  - Store forecasts                                       │  │
│  │  - Model metadata                                        │  │
│  │  - Forecast performance tracking                         │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Workflow Orchestration** | Temporal.io | Schedule daily forecasts, retries |
| **Model Training** | Python + LightGBM | Train forecasting models |
| **Model Serving** | FastAPI | Serve predictions via REST API |
| **Model Storage** | GCS (Google Cloud Storage) | Store trained model artifacts |
| **Model Cache** | Redis | Hot cache for fast inference |
| **Timeseries Data** | ClickHouse | Store telemetry, historical consumption |
| **Metadata** | PostgreSQL | Model versions, forecast logs |
| **Weather API** | Open-Meteo | Fetch weather forecasts |
| **Compute** | GKE (Kubernetes) | Model training jobs, API serving |
| **Monitoring** | Prometheus + Grafana | Track forecast accuracy, latency |

### Day-Ahead Forecasting Workflow

**Trigger:** Scheduled at 09:00 AM daily (Temporal cron workflow)

```python
@workflow.defn
class DayAheadForecastWorkflow:
    """
    Generate forecasts for next 24 hours for all devices
    """

    @workflow.run
    async def run(self, forecast_date: str) -> Dict:
        # 1. Fetch historical data (last 30 days)
        historical_data = await workflow.execute_activity(
            fetch_historical_data,
            lookback_days=30,
            start_to_close_timeout=timedelta(minutes=5)
        )

        # 2. Fetch weather forecast for all device locations
        weather_forecasts = await workflow.execute_activity(
            fetch_weather_forecasts,
            forecast_date=forecast_date,
            start_to_close_timeout=timedelta(minutes=2)
        )

        # 3. Load trained model from cache/storage
        model = await workflow.execute_activity(
            load_model,
            model_version='latest',
            start_to_close_timeout=timedelta(seconds=30)
        )

        # 4. Generate features and predict for each device
        forecasts = await workflow.execute_activity(
            generate_forecasts,
            historical_data=historical_data,
            weather_forecasts=weather_forecasts,
            model=model,
            start_to_close_timeout=timedelta(minutes=10)
        )

        # 5. Store forecasts in database
        await workflow.execute_activity(
            store_forecasts,
            forecasts=forecasts,
            forecast_type='day_ahead',
            start_to_close_timeout=timedelta(minutes=2)
        )

        # 6. Publish to trading system
        await workflow.execute_activity(
            publish_to_trading_system,
            forecasts=forecasts,
            start_to_close_timeout=timedelta(minutes=1)
        )

        return {
            'num_devices': len(forecasts),
            'total_forecast_mwh': sum(f['total_mwh'] for f in forecasts)
        }
```

**Execution time:** 15-20 minutes total

### Intraday Forecasting (On-Demand)

**Trigger:** HTTP API request

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb

app = FastAPI()

class ForecastRequest(BaseModel):
    device_ids: List[str]
    forecast_start: str  # ISO timestamp
    forecast_hours: int = 24

class ForecastResponse(BaseModel):
    device_id: str
    intervals: List[Dict[str, float]]  # [{timestamp, power_wh}, ...]

@app.post("/forecast", response_model=List[ForecastResponse])
async def generate_forecast(request: ForecastRequest):
    """
    Generate on-demand forecast

    SLA: <500ms for single device, <2s for 100 devices
    """

    # 1. Check Redis cache for hot model
    model = redis.get('forecast_model:latest')
    if model is None:
        # Fallback: load from GCS (slower, ~1-2s)
        model = load_from_gcs('gs://models/lightgbm_latest.txt')
        redis.set('forecast_model:latest', model, ex=3600)  # Cache 1h

    # 2. Fetch recent data (last 48 hours) from ClickHouse
    query = f"""
        SELECT * FROM device_telemetry
        WHERE device_id IN {tuple(request.device_ids)}
        AND timestamp >= now() - INTERVAL 48 HOUR
        ORDER BY timestamp
    """
    historical_data = clickhouse_client.query(query).to_dataframe()

    # 3. Fetch weather forecast
    weather = await fetch_weather_parallel(device_ids)

    # 4. Generate features
    features = engineer_features(historical_data, weather)

    # 5. Predict
    predictions = model.predict(features)

    # 6. Format response
    forecasts = []
    for device_id in request.device_ids:
        intervals = []
        for i, timestamp in enumerate(generate_timestamps(request.forecast_start, 96)):
            intervals.append({
                'timestamp': timestamp,
                'power_wh': predictions[device_id][i]
            })

        forecasts.append(ForecastResponse(
            device_id=device_id,
            intervals=intervals
        ))

    return forecasts
```

**Latency optimization:**
- Model cached in Redis (no disk I/O)
- Pre-computed features where possible
- Parallel weather API calls
- ClickHouse materialized views for fast aggregations

### Model Training Pipeline

**Trigger:** Nightly (Temporal scheduled workflow)

```python
@workflow.defn
class ModelRetrainingWorkflow:
    """
    Retrain forecasting model with latest data
    """

    @workflow.run
    async def run(self) -> Dict:
        # 1. Extract training data (last 90 days)
        training_data = await workflow.execute_activity(
            extract_training_data,
            lookback_days=90,
            start_to_close_timeout=timedelta(minutes=10)
        )

        # 2. Feature engineering
        features = await workflow.execute_activity(
            engineer_training_features,
            data=training_data,
            start_to_close_timeout=timedelta(minutes=15)
        )

        # 3. Train model (LightGBM)
        model_artifact = await workflow.execute_activity(
            train_lightgbm_model,
            features=features,
            start_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5)
        )

        # 4. Evaluate on validation set
        metrics = await workflow.execute_activity(
            evaluate_model,
            model=model_artifact,
            val_data=features['validation'],
            start_to_close_timeout=timedelta(minutes=10)
        )

        # 5. Compare to previous model
        improvement = metrics['mape'] < previous_best_mape

        if improvement:
            # 6. Upload to model registry
            await workflow.execute_activity(
                upload_model_to_registry,
                model=model_artifact,
                metrics=metrics,
                start_to_close_timeout=timedelta(minutes=5)
            )

            # 7. Update Redis cache
            await workflow.execute_activity(
                update_model_cache,
                model=model_artifact,
                start_to_close_timeout=timedelta(seconds=30)
            )

        return {
            'mape': metrics['mape'],
            'improvement': improvement,
            'deployed': improvement
        }
```

**Execution time:** 1-2 hours (mostly training)

### Data Pipeline

**ClickHouse Schema:**

```sql
CREATE TABLE device_telemetry (
    device_id String,
    timestamp DateTime64(3),
    outdoor_temp Float32,
    indoor_temp Float32,
    elec_power Float32,
    thermal_power Float32,
    operating_state String,
    INDEX idx_device_time (device_id, timestamp) TYPE minmax GRANULARITY 3
) ENGINE = MergeTree()
ORDER BY (device_id, timestamp)
PARTITION BY toYYYYMM(timestamp);

CREATE TABLE actual_consumption (
    device_id String,
    interval_start DateTime,
    interval_end DateTime,
    heating_electricity Float32,
    dhw_electricity Float32,
    cooling_electricity Float32,
    total_electricity Float32,
    INDEX idx_device_interval (device_id, interval_start) TYPE minmax
) ENGINE = MergeTree()
ORDER BY (device_id, interval_start)
PARTITION BY toYYYYMM(interval_start);

-- Materialized view for 15-minute aggregations
CREATE MATERIALIZED VIEW telemetry_15min
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(interval_start)
ORDER BY (device_id, interval_start)
AS SELECT
    device_id,
    toStartOfInterval(timestamp, INTERVAL 15 MINUTE) AS interval_start,
    avg(outdoor_temp) AS avg_outdoor_temp,
    avg(indoor_temp) AS avg_indoor_temp,
    avg(elec_power) AS avg_elec_power,
    max(elec_power) AS max_elec_power
FROM device_telemetry
GROUP BY device_id, interval_start;
```

**PostgreSQL Schema:**

```sql
CREATE TABLE forecast_runs (
    id UUID PRIMARY KEY,
    forecast_type VARCHAR(20),  -- 'day_ahead' or 'intraday'
    forecast_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50),
    num_devices INT,
    execution_time_ms INT
);

CREATE TABLE forecasts (
    id UUID PRIMARY KEY,
    forecast_run_id UUID REFERENCES forecast_runs(id),
    device_id VARCHAR(50),
    interval_start TIMESTAMP,
    interval_end TIMESTAMP,
    predicted_power_wh FLOAT,
    actual_power_wh FLOAT,  -- Filled in later
    absolute_error_wh FLOAT,
    percentage_error FLOAT,
    INDEX idx_device_interval (device_id, interval_start)
);

CREATE TABLE model_registry (
    id UUID PRIMARY KEY,
    model_name VARCHAR(100),
    version VARCHAR(50),
    trained_at TIMESTAMP,
    mape FLOAT,
    rmse FLOAT,
    mae FLOAT,
    artifact_path TEXT,  -- GCS path
    is_production BOOLEAN DEFAULT FALSE
);
```

### Infrastructure Setup (GCP + Kubernetes)

#### 1. GKE Cluster

```hcl
# infrastructure/terraform/gke.tf
resource "google_container_cluster" "forecasting" {
  name     = "forecasting-cluster"
  location = var.region

  enable_autopilot = true

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}
```

#### 2. ClickHouse Deployment

```yaml
# infrastructure/kubernetes/clickhouse.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: clickhouse
spec:
  serviceName: clickhouse
  replicas: 1
  template:
    spec:
      containers:
      - name: clickhouse
        image: clickhouse/clickhouse-server:latest
        ports:
        - containerPort: 8123  # HTTP
        - containerPort: 9000  # Native
        volumeMounts:
        - name: data
          mountPath: /var/lib/clickhouse
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

#### 3. FastAPI Model Serving

```yaml
# infrastructure/kubernetes/forecast-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forecast-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: gcr.io/${PROJECT_ID}/forecast-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: CLICKHOUSE_HOST
          value: clickhouse-service
        - name: MODEL_CACHE_TTL
          value: "3600"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: forecast-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: forecast-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 4. Model Training Job

```yaml
# infrastructure/kubernetes/model-training-job.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-retraining
spec:
  schedule: "0 2 * * *"  # 02:00 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: training
            image: gcr.io/${PROJECT_ID}/model-training:latest
            env:
            - name: TRAINING_DATA_DAYS
              value: "90"
            - name: MODEL_REGISTRY_BUCKET
              value: "gs://forecast-models"
            resources:
              requests:
                memory: "8Gi"
                cpu: "4000m"
              limits:
                memory: "16Gi"
                cpu: "8000m"
          restartPolicy: OnFailure
```

### Monitoring & Alerting

#### Prometheus Metrics

```python
from prometheus_client import Histogram, Counter, Gauge

# Forecast latency
forecast_latency = Histogram(
    'forecast_generation_seconds',
    'Time to generate forecast',
    ['forecast_type']
)

# Forecast accuracy
forecast_mape = Gauge(
    'forecast_mape',
    'Mean absolute percentage error',
    ['device_id', 'horizon_hours']
)

# Model serving
model_cache_hits = Counter(
    'model_cache_hits_total',
    'Number of model cache hits'
)

model_cache_misses = Counter(
    'model_cache_misses_total',
    'Number of model cache misses'
)
```

#### Grafana Dashboard

**Key panels:**
1. **Forecast Accuracy Over Time** - MAPE trend by day
2. **Per-Device Performance** - Heatmap of MAPE by device
3. **API Latency** - P50, P95, P99 latency
4. **Model Cache Hit Rate** - Percentage
5. **Forecast Volume** - Forecasts generated per hour
6. **Error Distribution** - Histogram of forecast errors

#### Alerts

```yaml
# alerting/prometheus-rules.yaml
groups:
- name: forecasting
  rules:
  - alert: HighForecastError
    expr: forecast_mape > 20
    for: 1h
    labels:
      severity: warning
    annotations:
      summary: "Forecast MAPE > 20% for {{ $labels.device_id }}"

  - alert: ForecastAPIDown
    expr: up{job="forecast-api"} == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Forecast API is down"

  - alert: ModelTrainingFailed
    expr: increase(model_training_failures_total[1d]) > 0
    labels:
      severity: warning
    annotations:
      summary: "Model training failed"
```

---

## Implementation Plan

### Phase 1: MVP (Weeks 1-3)

**Goal:** Basic working forecast with LightGBM

#### Week 1: Data Pipeline
- [ ] Set up ClickHouse database
- [ ] ETL pipeline: CSV → ClickHouse
- [ ] Create 15-minute aggregation views
- [ ] Data quality checks

#### Week 2: Model Development
- [ ] Feature engineering (time features + lags)
- [ ] Train LightGBM baseline model
- [ ] Evaluate on test set
- [ ] Target: 15-18% MAPE

#### Week 3: API & Deployment
- [ ] FastAPI model serving endpoint
- [ ] Deploy to GKE
- [ ] Scheduled forecast workflow (Temporal)
- [ ] Store forecasts in PostgreSQL

**Deliverable:** Working forecast API with 15-18% MAPE

---

### Phase 2: Weather Integration (Weeks 4-5)

**Goal:** Improve accuracy with weather data

#### Week 4: Weather API Integration
- [ ] Integrate Open-Meteo API
- [ ] Fetch historical weather data
- [ ] Backfill weather for training data
- [ ] Add weather features to model

#### Week 5: Model Retraining
- [ ] Retrain with weather features
- [ ] Evaluate improvement
- [ ] Target: 10-12% MAPE
- [ ] Deploy updated model

**Deliverable:** Weather-enhanced model with 10-12% MAPE

---

### Phase 3: Production Hardening (Weeks 6-8)

**Goal:** Production-ready, scalable system

#### Week 6: Performance Optimization
- [ ] Redis model caching
- [ ] ClickHouse query optimization
- [ ] Parallel prediction for multiple devices
- [ ] Target: <500ms latency

#### Week 7: Monitoring & Alerting
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Alerting rules
- [ ] Forecast accuracy tracking

#### Week 8: Operational Excellence
- [ ] Automated model retraining pipeline
- [ ] A/B testing framework
- [ ] Model versioning & rollback
- [ ] Documentation & runbooks

**Deliverable:** Production-ready forecasting system

---

## Conclusion

This forecasting solution provides a pragmatic, production-ready approach to predicting heat pump power consumption 24 hours in advance at 15-minute granularity.

**Key Strengths:**
1. **Accurate:** 10-12% MAPE with weather features
2. **Fast:** <500ms inference for on-demand forecasts
3. **Scalable:** Handles thousands of devices
4. **Maintainable:** Simple LightGBM model, easy to debug
5. **Reliable:** Temporal workflows with retries, monitoring

**Next Steps:**
1. Implement Phase 1 MVP (3 weeks)
2. Integrate weather data (2 weeks)
3. Production hardening (3 weeks)
4. Consider advanced models (TFT, ensembles) for further improvement

**Expected ROI:**
- Better trading decisions → 5-10% cost savings
- Reduced imbalance penalties
- Improved grid services revenue