"""Site-separated model fitting and split-conformal intervals."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["sensor_pm25", "humidity", "temperature"]
TARGET = "reference_pm25"


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted))),
        "Bias": float(np.mean(error)),
        "R²": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


def run_site_holdout(
    frame: pd.DataFrame,
    test_station: str,
    coverage: float = 0.90,
    seed: int = 17,
) -> dict:
    """Fit on training sites, calibrate on a separate site, evaluate on test site."""
    stations = sorted(frame["station_id"].astype(str).unique())
    if len(stations) < 3:
        raise ValueError("At least three stations are needed for fitting, calibration, and testing.")
    if test_station not in stations:
        raise ValueError("Selected test station is not present in the dataset.")
    if not 0.5 <= coverage < 1:
        raise ValueError("Target coverage must be between 50% and 99%.")

    remaining = [name for name in stations if name != test_station]
    calibration_station = remaining[-1]
    fit_stations = remaining[:-1]
    train = frame[frame["station_id"].astype(str).isin(fit_stations)]
    calibration = frame[frame["station_id"].astype(str) == calibration_station]
    test = frame[frame["station_id"].astype(str) == test_station]

    model = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=160, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed
    )
    model.fit(train[FEATURES], train[TARGET])
    calibration_prediction = model.predict(calibration[FEATURES])
    residuals = np.abs(calibration[TARGET].to_numpy() - calibration_prediction)
    n = len(residuals)
    rank = min(n, max(1, math.ceil((n + 1) * coverage)))
    radius = float(np.sort(residuals)[rank - 1])

    raw = test["sensor_pm25"].to_numpy()
    actual = test[TARGET].to_numpy()
    corrected = model.predict(test[FEATURES])
    lower = np.maximum(0.0, corrected - radius)
    upper = corrected + radius

    scored = test.copy()
    scored["raw_prediction"] = raw
    scored["corrected_prediction"] = corrected
    scored["interval_low"] = lower
    scored["interval_high"] = upper
    scored["covered"] = (actual >= lower) & (actual <= upper)

    return {
        "model": model,
        "test_station": test_station,
        "calibration_station": calibration_station,
        "fit_stations": fit_stations,
        "radius": radius,
        "target_coverage": coverage,
        "observed_coverage": float(scored["covered"].mean()),
        "interval_width": float(np.mean(upper - lower)),
        "scores": {
            "Raw sensor": _metrics(actual, raw),
            "ML corrected": _metrics(actual, corrected),
        },
        "predictions": scored,
    }
