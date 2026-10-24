"""Offline ground-truth evaluation for the project's custom quantile model.

Examples
--------
python evaluate_accuracy_standalone.py --target T2_data.xlsx --references H1_data.xlsx
python evaluate_accuracy_standalone.py --target data/T2.csv --references data/H1.csv data/H2.csv

The target wind-speed column is never supplied as a model feature.  Test-set
labels are retained only for metric calculation after prediction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    # This is the project's actual custom model, not a substitute estimator.
    from src.quantile_model import QuantileLossPredictor
except ImportError as exc:
    raise ImportError(
        "Could not import src.quantile_model.QuantileLossPredictor. "
        "Run this script from the project root."
    ) from exc

try:
    from wind_imputation import apply_physics_features, load_power_curve, power_curve
except ImportError as exc:
    raise ImportError("Could not import the project's physics feature utilities.") from exc


def read_table(path: str | Path, require_test_set: bool = False) -> pd.DataFrame:
    """Read CSV or Excel SCADA data and validate the fields used by evaluation."""
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    required = {"timestamp", "wind_speed"} | ({"test_set"} if require_test_set else set())
    if missing := required - set(frame.columns):
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["wind_speed"] = pd.to_numeric(frame["wind_speed"], errors="coerce")
    return frame.sort_values("timestamp").drop_duplicates("timestamp")


def is_test_flag(values: pd.Series) -> pd.Series:
    """Normalise integer, boolean, and common string test-set flags."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "test"})


def physics_ready_reference(reference: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Apply the project's physics features, adapting supplied SCADA column names."""
    prepared = reference.copy().rename(columns={
        "ambient_temperature": "temperature",
        "air_pressure": "pressure",
    })
    required = {"wind_speed", "wind_direction", "temperature", "pressure"}
    if missing := required - set(prepared.columns):
        raise ValueError(f"{source_name} cannot create physics features; missing {sorted(missing)}")
    return apply_physics_features(prepared)


def build_features(target: pd.DataFrame, references: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Merge every reference by timestamp and create spatial and calendar features."""
    combined = target[["timestamp", "wind_speed", "test_set"]].rename(columns={"wind_speed": "target_wind_speed"})
    feature_columns: list[str] = []
    for number, reference in enumerate(references, start=1):
        prefix = f"reference_{number}"
        reference = physics_ready_reference(reference, prefix)
        physics_columns = [
            "wind_speed", "normalized_wind_speed", "sin_wind_dir", "cos_wind_dir",
            "is_wake_sector", "wind_speed_rolling_std_1h",
        ]
        selected = reference[["timestamp", *physics_columns]].rename(
            columns={column: f"{prefix}_{column}" for column in physics_columns}
        )
        feature_columns.extend(f"{prefix}_{column}" for column in physics_columns)
        combined = combined.merge(selected, on="timestamp", how="left")
    combined["hour"] = combined["timestamp"].dt.hour
    combined["month"] = combined["timestamp"].dt.month
    feature_columns.extend(["hour", "month"])
    return combined, feature_columns


def evaluate(target_path: str | Path, reference_paths: list[str | Path], power_curve_path: str | Path | None = None) -> pd.DataFrame:
    """Fit the actual model on non-test rows and print hold-out test accuracy."""
    target = read_table(target_path, require_test_set=True)
    references = [read_table(path) for path in reference_paths]
    frame, features = build_features(target, references)
    test_mask = is_test_flag(frame["test_set"])
    train_mask = ~test_mask & frame["target_wind_speed"].notna()
    metric_mask = test_mask & frame["target_wind_speed"].notna()
    if train_mask.sum() < 50 or metric_mask.sum() < 2:
        raise ValueError("Need at least 50 observed training rows and 2 observed test-set rows.")

    model = QuantileLossPredictor()
    model.fit(frame.loc[train_mask, features], frame.loc[train_mask, "target_wind_speed"])
    # Target wind speed is intentionally absent from this prediction input.
    forecast = model.predict_quantiles(frame.loc[metric_mask, features])
    actual = frame.loc[metric_mask, "target_wind_speed"].to_numpy(dtype=float)
    predicted = forecast["P50_yield"].to_numpy(dtype=float)
    errors = predicted - actual
    nonzero = np.abs(actual) > np.finfo(float).eps
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mse = float(mean_squared_error(actual, predicted))
    mean_bias = float(np.mean(errors))
    nrmse = float(rmse / np.mean(actual) * 100) if np.mean(actual) else float("nan")
    curve = load_power_curve(power_curve_path) if power_curve_path else None
    actual_power = power_curve(actual, curve)
    predicted_power = power_curve(predicted, curve)
    power_errors = predicted_power - actual_power
    interval_minutes = float(frame.loc[metric_mask, "timestamp"].diff().dt.total_seconds().median() / 60.0)
    interval_minutes = interval_minutes if np.isfinite(interval_minutes) and interval_minutes > 0 else 10.0
    actual_energy = float(actual_power.sum() * interval_minutes / 60.0)
    predicted_energy = float(predicted_power.sum() * interval_minutes / 60.0)
    positive_power_gap = np.maximum(power_errors, 0.0)
    positive_energy_gap = float(positive_power_gap.sum() * interval_minutes / 60.0)

    print("\n" + "=" * 64)
    print("GROUND TRUTH TEST-SET ACCURACY  CUSTOM QUANTILE MODEL WITH PHYSICS FEATURES")
    print("=" * 64)
    print(f"MAE   : {mean_absolute_error(actual, predicted):.3f} m/s")
    print(f"RMSE  : {rmse:.3f} m/s")
    print(f"R²    : {r2_score(actual, predicted):.3f}")
    print(f"MAPE  : {mape:.3f}%")
    print(f"MSE   : {mse:.3f} (m/s)²")
    print(f"Bias  : {mean_bias:.3f} m/s")
    print(f"NRMSE : {nrmse:.3f}% of mean wind speed")
    print(f"Rows  : {len(actual):,} observed test-set labels")
    print("\n" + "-" * 64)
    print("POWER AND ENERGY DIAGNOSTICS")
    print("-" * 64)
    print(f"Power MAE                 : {mean_absolute_error(actual_power, predicted_power):.3f} kW")
    print(f"Power RMSE                : {np.sqrt(mean_squared_error(actual_power, predicted_power)):.3f} kW")
    print(f"Power MSE                 : {mean_squared_error(actual_power, predicted_power):.3f} kW²")
    print(f"Peak positive power gap   : {positive_power_gap.max():.3f} kW")
    print(f"Actual test energy        : {actual_energy:.3f} kWh")
    print(f"Predicted test energy     : {predicted_energy:.3f} kWh")
    print(f"Energy yield deviation    : {predicted_energy - actual_energy:.3f} kWh")
    print(f"Energy yield deviation %  : {(predicted_energy - actual_energy) / actual_energy * 100:.3f}%")
    print(f"Positive energy gap/loss  : {positive_energy_gap:.3f} kWh")

    comparison = pd.DataFrame({
        "Timestamp": frame.loc[metric_mask, "timestamp"].to_numpy(),
        "Actual Wind Speed (m/s)": actual,
        "Predicted Wind Speed (m/s)": predicted,
        "Error (m/s)": errors,
    })
    print("\nSample comparison (first 10 rows)")
    print(comparison.head(10).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the custom wind-speed imputer against test_set ground truth.")
    parser.add_argument("--target", default="T2_data.xlsx", help="Target CSV/XLSX containing timestamp, wind_speed, and test_set.")
    parser.add_argument("--references", nargs="+", default=["H1_data.xlsx"], help="One or more reference CSV/XLSX files.")
    parser.add_argument("--power-curve", default=None, help="Optional target-turbine power-curve CSV/XLSX.")
    args = parser.parse_args()
    evaluate(args.target, args.references, args.power_curve)


if __name__ == "__main__":
    main()
