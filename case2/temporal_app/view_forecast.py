"""
View detailed forecast data for a device.

This script generates a forecast and displays/saves the full predictions
without going through Temporal (for development/debugging).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from forecaster import HeatPumpForecaster


def view_forecast(device_id: str, horizon_hours: int = 24):
    """
    Generate and display forecast for a device.

    Args:
        device_id: Device serial number
        horizon_hours: Number of hours to forecast
    """
    print(f"\n{'='*70}")
    print(f"GENERATING FORECAST FOR DEVICE: {device_id}")
    print(f"{'='*70}\n")

    # Load the trained model
    model_path = Path(__file__).parent.parent / "trained_model"

    if not model_path.exists():
        print("ERROR: Trained model not found!")
        print(f"Expected at: {model_path}")
        print("\nRun the forecast_implementation.py script first to train the model.")
        return

    print(f"Loading model from: {model_path}")
    forecaster = HeatPumpForecaster(model_path=model_path)
    print("✓ Model loaded\n")

    # Load historical data
    csv_path = Path(__file__).parent.parent / "sample_devices_actual_power_consumption.csv"
    print(f"Loading historical data from: {csv_path.name}")
    consumption_df = pd.read_csv(csv_path)

    # Filter to specific device
    device_data = consumption_df[consumption_df['serial_number'] == device_id]

    if len(device_data) == 0:
        print(f"\nERROR: No data found for device {device_id}")
        print("\nAvailable devices:")
        for dev in consumption_df['serial_number'].unique()[:10]:
            print(f"  - {dev}")
        return

    print(f"✓ Loaded {len(device_data)} historical records\n")

    # Generate forecast starting 1 hour from now
    forecast_start = datetime.now(timezone.utc) + timedelta(hours=1)

    print(f"Generating {horizon_hours}-hour forecast...")
    print(f"Forecast start: {forecast_start}")
    print(f"Forecast end:   {forecast_start + timedelta(hours=horizon_hours)}\n")

    # Generate forecast
    forecast_df = forecaster.predict(
        device_id=device_id,
        start_time=forecast_start,
        horizon_hours=horizon_hours,
        historical_data=device_data,
    )

    print(f"{'='*70}")
    print(f"FORECAST RESULTS")
    print(f"{'='*70}\n")

    # Summary statistics
    total_energy_kwh = forecast_df['predicted_power'].sum() / 1000
    avg_power_wh = forecast_df['predicted_power'].mean()
    min_power_wh = forecast_df['predicted_power'].min()
    max_power_wh = forecast_df['predicted_power'].max()

    print(f"Summary Statistics:")
    print(f"  Total intervals: {len(forecast_df)}")
    print(f"  Total energy:    {total_energy_kwh:.2f} kWh")
    print(f"  Average power:   {avg_power_wh:.2f} Wh")
    print(f"  Min power:       {min_power_wh:.2f} Wh")
    print(f"  Max power:       {max_power_wh:.2f} Wh")

    # Show first 10 predictions
    print(f"\nFirst 10 Predictions:")
    print("-" * 70)
    for _, row in forecast_df.head(10).iterrows():
        ts = row['timestamp']
        power = row['predicted_power']
        print(f"  {ts}  →  {power:>7.2f} Wh")

    print("\n...")

    # Show last 10 predictions
    print(f"\nLast 10 Predictions:")
    print("-" * 70)
    for _, row in forecast_df.tail(10).iterrows():
        ts = row['timestamp']
        power = row['predicted_power']
        print(f"  {ts}  →  {power:>7.2f} Wh")

    # Save to CSV
    output_file = Path(__file__).parent / f"forecast_{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    forecast_df.to_csv(output_file, index=False)

    print(f"\n{'='*70}")
    print(f"✓ Full forecast saved to: {output_file.name}")
    print(f"{'='*70}\n")

    return forecast_df


if __name__ == "__main__":
    # Get device ID from command line or use default
    if len(sys.argv) > 1:
        device_id = sys.argv[1]
    else:
        # Use first device from dataset
        csv_path = Path(__file__).parent.parent / "sample_devices_actual_power_consumption.csv"
        df = pd.read_csv(csv_path)
        device_id = df['serial_number'].iloc[0]
        print(f"No device specified, using: {device_id}\n")

    # Get horizon hours from command line or use default
    horizon_hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    # Generate and view forecast
    forecast_df = view_forecast(device_id, horizon_hours)

    if forecast_df is not None:
        print("To view forecast for a different device:")
        print(f"  python view_forecast.py <device_id> [horizon_hours]")
        print("\nExample:")
        print(f"  python view_forecast.py 21183900202529220938026983N5 6")
