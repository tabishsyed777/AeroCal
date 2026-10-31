"""Synthetic collocation data for the AeroCal demo."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(seed: int = 42, rows_per_station: int = 320) -> pd.DataFrame:
    """Create reproducible synthetic sensor/reference collocations.

    The generated reference values are simulated ground truth, not observations.
    """
    rng = np.random.default_rng(seed)
    station_profiles = [
        ("North", 2.8, 0.045, 0.010, 0.0),
        ("Harbor", -1.5, 0.090, -0.018, 1.1),
        ("Campus", 1.2, 0.025, 0.025, 2.2),
        ("Market", -2.2, 0.070, 0.012, 0.7),
        ("Garden", 0.5, 0.110, -0.010, 1.7),
        ("Airport", 3.0, 0.035, 0.020, 2.8),
    ]
    chunks = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for station, bias, humidity_effect, temp_effect, phase in station_profiles:
        t = np.arange(rows_per_station)
        humidity = np.clip(55 + 22 * np.sin(t / 31 + phase) + rng.normal(0, 7, rows_per_station), 15, 98)
        temperature = 18 + 10 * np.sin(t / 53 + phase / 2) + rng.normal(0, 2.5, rows_per_station)
        reference = np.clip(
            8 + 9 * np.maximum(0, np.sin(t / 19 + phase))
            + 4 * np.maximum(0, np.sin(t / 71 + phase / 3))
            + rng.gamma(1.8, 2.2, rows_per_station), 1, 100
        )
        sensor = (
            reference + bias + humidity_effect * (humidity - 50)
            + temp_effect * (temperature - 20) + rng.normal(0, 2.2, rows_per_station)
        )
        sensor = np.clip(sensor, 0, None)
        chunks.append(pd.DataFrame({
            "timestamp": start + pd.to_timedelta(t * 3, unit="h"),
            "station_id": station,
            "sensor_pm25": sensor,
            "reference_pm25": reference,
            "humidity": humidity,
            "temperature": temperature,
        }))
    return pd.concat(chunks, ignore_index=True)


def validate_collocations(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize and validate a user-supplied collocation table."""
    required = ["station_id", "sensor_pm25", "reference_pm25", "humidity", "temperature"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        return frame, [f"Missing required columns: {', '.join(missing)}"]

    clean = frame.copy()
    clean["station_id"] = clean["station_id"].astype("string").str.strip()
    for name in required[1:]:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=required)
    clean = clean[clean["station_id"].notna() & (clean["station_id"] != "")]
    if "timestamp" in clean.columns:
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True)

    messages = []
    if clean.empty:
        messages.append("No complete numeric collocation rows remain after cleaning.")
    elif clean["station_id"].nunique() < 3:
        messages.append("At least three distinct station_id values are required.")
    elif clean["station_id"].value_counts().min() < 5:
        messages.append("Each station needs at least five complete rows.")
    return clean.reset_index(drop=True), messages
