"""
Heat Pump Power Consumption Forecaster
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pickle


class HeatPumpForecaster:
    """
    Example:
        >>> forecaster = HeatPumpForecaster()
        >>> forecaster.train(consumption_df, telemetry_df)
        >>> predictions = forecaster.predict(device_id='ABC123', horizon_hours=24)
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize forecaster.

        Args:
            model_path: Path to saved model. If None, creates new model.
        """
        self.model = None
        self.feature_cols = None
        self.device_metadata = {}

        if model_path and model_path.exists():
            self.load(model_path)

    def preprocess_data(
        self,
        consumption_df: pd.DataFrame,
        telemetry_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Preprocess raw data into 15-minute intervals with features.

        Args:
            consumption_df: Hourly power consumption data
            telemetry_df: Device telemetry (optional, for future use)

        Returns:
            DataFrame with 15-minute intervals and engineered features
        """
        # Parse timestamps
        consumption_df = consumption_df.copy()
        consumption_df['start'] = pd.to_datetime(consumption_df['start'])
        consumption_df['end'] = pd.to_datetime(consumption_df['end'])

        # Calculate total electricity
        consumption_df['total_electricity'] = (
            consumption_df['centralHeating.electricity'].fillna(0) +
            consumption_df['domesticHotWater.electricity'].fillna(0) +
            consumption_df['cooling.electricity'].fillna(0)
        )

        # Calculate heating vs DHW shares
        consumption_df['heating_share'] = (
            consumption_df['centralHeating.electricity'].fillna(0) /
            (consumption_df['total_electricity'] + 1)
        )
        consumption_df['dhw_share'] = (
            consumption_df['domesticHotWater.electricity'].fillna(0) /
            (consumption_df['total_electricity'] + 1)
        )

        # Create 15-minute intervals
        rows = []
        for _, row in consumption_df.iterrows():
            start_time = row['start']
            for i in range(4):  # 4x15min = 1 hour
                interval_start = start_time + timedelta(minutes=15*i)
                rows.append({
                    'timestamp': interval_start,
                    'device_id': row['serial_number'],
                    'power': row['total_electricity'] / 4,
                    'heating_electricity': row['centralHeating.electricity'] / 4 if pd.notna(row['centralHeating.electricity']) else 0,
                    'dhw_electricity': row['domesticHotWater.electricity'] / 4 if pd.notna(row['domesticHotWater.electricity']) else 0,
                    'heating_share': row['heating_share'],
                    'dhw_share': row['dhw_share'],
                    'manufacturer': row['manufacturer'],
                    'price_zone': row['price_zone']
                })

        df = pd.DataFrame(rows)
        df = df.sort_values(['device_id', 'timestamp']).reset_index(drop=True)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for forecasting.

        Args:
            df: DataFrame with 15-minute intervals

        Returns:
            DataFrame with additional features
        """
        df = df.copy()

        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter_hour'] = df['timestamp'].dt.minute // 15

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Time indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 23)).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] < 6)).astype(int)

        # Lag features
        lag_periods = {
            'lag_1h': 4,
            'lag_2h': 8,
            'lag_6h': 24,
            'lag_24h': 96,
            'lag_48h': 192,
            'lag_7d': 672
        }

        for name, periods in lag_periods.items():
            df[f'power_{name}'] = df.groupby('device_id')['power'].shift(periods)

        # Rolling statistics
        for window, name in [(4, '1h'), (24, '6h'), (96, '24h')]:
            df[f'power_roll_mean_{name}'] = df.groupby('device_id')['power'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        df['power_roll_std_24h'] = df.groupby('device_id')['power'].transform(
            lambda x: x.rolling(96, min_periods=1).std()
        )
        df['power_roll_max_24h'] = df.groupby('device_id')['power'].transform(
            lambda x: x.rolling(96, min_periods=1).max()
        )
        df['power_roll_min_24h'] = df.groupby('device_id')['power'].transform(
            lambda x: x.rolling(96, min_periods=1).min()
        )

        # Exponential moving averages
        df['power_ema_1h'] = df.groupby('device_id')['power'].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        )
        df['power_ema_24h'] = df.groupby('device_id')['power'].transform(
            lambda x: x.ewm(span=96, adjust=False).mean()
        )

        # Device-specific features
        device_avg = df.groupby('device_id')['power'].mean().to_dict()
        df['device_avg_power'] = df['device_id'].map(device_avg)

        # Encode categorical
        df['device_id_encoded'] = df['device_id'].astype('category').cat.codes
        df['manufacturer_encoded'] = df['manufacturer'].astype('category').cat.codes
        df['price_zone_encoded'] = df['price_zone'].astype('category').cat.codes

        return df

    def train(
        self,
        consumption_df: pd.DataFrame,
        telemetry_df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Train forecasting model.

        Args:
            consumption_df: Historical power consumption data
            telemetry_df: Device telemetry (optional)
            test_size: Fraction of data for testing
            params: LightGBM parameters (optional)

        Returns:
            Dictionary with training metrics
        """
        # Preprocess
        df = self.preprocess_data(consumption_df, telemetry_df)
        df = self.engineer_features(df)

        # Define features
        self.feature_cols = [
            'hour', 'day_of_week', 'day_of_month', 'month', 'quarter_hour',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_morning', 'is_evening', 'is_night',
            'power_lag_1h', 'power_lag_2h', 'power_lag_6h',
            'power_lag_24h', 'power_lag_48h', 'power_lag_7d',
            'power_roll_mean_1h', 'power_roll_mean_6h', 'power_roll_mean_24h',
            'power_roll_std_24h', 'power_roll_max_24h', 'power_roll_min_24h',
            'power_ema_1h', 'power_ema_24h',
            'device_avg_power', 'heating_share', 'dhw_share',
            'device_id_encoded', 'manufacturer_encoded', 'price_zone_encoded'
        ]

        # Split data
        df = df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx].dropna(subset=self.feature_cols)
        test_df = df.iloc[split_idx:].dropna(subset=self.feature_cols)

        X_train = train_df[self.feature_cols]
        y_train = train_df['power']
        X_test = test_df[self.feature_cols]
        y_test = test_df['power']

        # Default parameters
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'seed': 42
            }

        # Train model
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=['device_id_encoded', 'manufacturer_encoded', 'price_zone_encoded']
        )

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500
        )

        # Store metadata
        self.device_metadata = {
            'device_avg': df.groupby('device_id')['power'].mean().to_dict(),
            'device_map': df[['device_id', 'device_id_encoded']].drop_duplicates().set_index('device_id')['device_id_encoded'].to_dict(),
            'manufacturer_map': df[['manufacturer', 'manufacturer_encoded']].drop_duplicates().set_index('manufacturer')['manufacturer_encoded'].to_dict(),
            'price_zone_map': df[['price_zone', 'price_zone_encoded']].drop_duplicates().set_index('price_zone')['price_zone_encoded'].to_dict()
        }

        # Evaluate
        y_pred = self.model.predict(X_test)

        from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

        mask = y_test > 0
        mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return {
            'mape': mape,
            'mae': mae,
            'rmse': rmse,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def predict(
        self,
        device_id: str,
        start_time: datetime,
        horizon_hours: int = 24,
        historical_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate forecast for a device.

        Args:
            device_id: Device identifier
            start_time: Start time for forecast
            horizon_hours: Number of hours to forecast
            historical_data: Recent historical data (last 7 days recommended)

        Returns:
            DataFrame with timestamp and predicted_power columns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if historical_data is None:
            raise ValueError("historical_data is required for prediction")

        # Engineer features from historical data
        df = self.preprocess_data(historical_data)
        df = self.engineer_features(df)

        # Filter to specific device
        device_data = df[df['device_id'] == device_id].copy()

        # Generate timestamps
        num_intervals = horizon_hours * 4  # 4 intervals per hour
        forecast_timestamps = [start_time + timedelta(minutes=15*i) for i in range(num_intervals)]

        predictions = []

        for ts in forecast_timestamps:
            # Get latest features
            latest = device_data[device_data['timestamp'] < ts].iloc[-1:].copy()

            if len(latest) == 0:
                # No historical data, use defaults
                predictions.append(0)
                continue

            # Update temporal features for forecast time
            latest['timestamp'] = ts
            latest['hour'] = ts.hour
            latest['day_of_week'] = ts.weekday()  # Use weekday() for datetime objects
            latest['day_of_month'] = ts.day
            latest['month'] = ts.month
            latest['quarter_hour'] = ts.minute // 15

            latest['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
            latest['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
            latest['dow_sin'] = np.sin(2 * np.pi * ts.weekday() / 7)
            latest['dow_cos'] = np.cos(2 * np.pi * ts.weekday() / 7)

            latest['is_weekend'] = int(ts.weekday() >= 5)
            latest['is_morning'] = int((ts.hour >= 6) & (ts.hour < 12))
            latest['is_evening'] = int((ts.hour >= 18) & (ts.hour < 23))
            latest['is_night'] = int((ts.hour >= 23) | (ts.hour < 6))

            # Predict
            X = latest[self.feature_cols]
            pred = self.model.predict(X)[0]

            # Ensure non-negative (power consumption cannot be negative)
            pred = max(0.0, pred)

            predictions.append(pred)

            # Update historical data with prediction for next iteration
            new_row = latest.copy()
            new_row['power'] = pred
            device_data = pd.concat([device_data, new_row], ignore_index=True)

        return pd.DataFrame({
            'timestamp': forecast_timestamps,
            'predicted_power': predictions
        })

    def save(self, path: Path):
        """Save model and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        self.model.save_model(str(path / 'model.txt'))

        # Save metadata
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'feature_cols': self.feature_cols,
                'device_metadata': self.device_metadata
            }, f)

    def load(self, path: Path):
        """Load model and metadata."""
        path = Path(path)

        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(path / 'model.txt'))

        # Load metadata
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.feature_cols = metadata['feature_cols']
            self.device_metadata = metadata['device_metadata']

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")

        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
