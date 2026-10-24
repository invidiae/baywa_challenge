"""SCADA validation and leakage-resistant cross-farm feature engineering."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"turbine_name", "timestamp", "wind_speed", "wind_direction", "test_set"}


def load_scada(path: str | Path) -> pd.DataFrame:
    """Load SCADA data without converting telemetry gaps into zero output."""
    frame = pd.read_excel(path)
    if missing := REQUIRED - set(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}.")
    return frame.sort_values("timestamp").reset_index(drop=True)


def external_features(references: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate external farms; directions are averaged as vectors, not degrees."""
    parts: list[pd.DataFrame] = []
    for name, frame in references.items():
        values = frame.set_index("timestamp")
        part = pd.DataFrame(index=values.index)
        for col in ("wind_speed", "ambient_temperature", "air_pressure", "wind_speed_stddev"):
            if col in values:
                part[f"{name}_{col}"] = pd.to_numeric(values[col], errors="coerce")
        radians = np.deg2rad(pd.to_numeric(values["wind_direction"], errors="coerce"))
        part[f"{name}_direction_sin"] = np.sin(radians)
        part[f"{name}_direction_cos"] = np.cos(radians)
        parts.append(part)
    joined = pd.concat(parts, axis=1).sort_index()
    result = pd.DataFrame(index=joined.index)
    for suffix in ("wind_speed", "ambient_temperature", "air_pressure", "wind_speed_stddev"):
        cols = [col for col in joined if col.endswith(suffix)]
        if cols:
            result[f"reference_{suffix}_median"] = joined[cols].median(axis=1)
    result["reference_direction_sin"] = joined.filter(like="_direction_sin").mean(axis=1)
    result["reference_direction_cos"] = joined.filter(like="_direction_cos").mean(axis=1)
    result["hour"] = result.index.hour
    result["day_of_year"] = result.index.dayofyear
    return result
