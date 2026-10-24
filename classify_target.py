"""
Wind Park Classification and Spatial Correlation Analysis Module.

Analyzes the statistical relationships and spatial proximity between
the Target wind park (T1, T2, T3) and Reference wind parks:
- Park H: H1, H2, H3, H4, H5
- Park O: O1, O2
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TURBINES_TARGET = ["T1", "T2", "T3"]
TURBINES_H = ["H1", "H2", "H3", "H4", "H5"]
TURBINES_O = ["O1", "O2"]
ALL_TURBINES = TURBINES_TARGET + TURBINES_H + TURBINES_O


def load_all_turbine_data(data_dir: str | Path = "data") -> Dict[str, pd.DataFrame]:
    """Load all turbine Excel files into a dictionary indexed by timestamp."""
    data_dir = Path(data_dir)
    data = {}
    for name in ALL_TURBINES:
        file_path = data_dir / f"{name}_data.xlsx"
        if file_path.exists():
            df = pd.read_excel(file_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            data[name] = df.set_index("timestamp").sort_index()
    return data


def extract_feature_matrix(
    data: Dict[str, pd.DataFrame],
    feature: str,
    turbines: List[str] = ALL_TURBINES,
) -> pd.DataFrame:
    """Extract a synchronized DataFrame for a given feature across turbines."""
    cols = {t: data[t][feature] for t in turbines if t in data and feature in data[t].columns}
    return pd.DataFrame(cols)


def compute_correlation_matrices(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Compute Pearson correlation matrices for wind speed, temperature, and wind direction."""
    speed_df = extract_feature_matrix(data, "wind_speed")
    temp_df = extract_feature_matrix(data, "ambient_temperature")
    pres_df = extract_feature_matrix(data, "air_pressure")

    dir_df = extract_feature_matrix(data, "wind_direction")
    sin_dir_df = np.sin(np.radians(dir_df))
    cos_dir_df = np.cos(np.radians(dir_df))

    return {
        "wind_speed": speed_df.corr(),
        "temperature": temp_df.corr(),
        "air_pressure": pres_df.corr(),
        "wind_dir_sin": sin_dir_df.corr(),
        "wind_dir_cos": cos_dir_df.corr(),
    }


def analyze_park_affinity(speed_corr: pd.DataFrame) -> pd.DataFrame:
    """Quantify affinity of each target turbine to Park H and Park O."""
    rows = []
    for t in TURBINES_TARGET:
        corr_h = speed_corr.loc[t, TURBINES_H]
        corr_o = speed_corr.loc[t, TURBINES_O]
        rows.append({
            "target_turbine": t,
            "mean_corr_H": float(corr_h.mean()),
            "max_corr_H": float(corr_h.max()),
            "best_H_turbine": str(corr_h.idxmax()),
            "mean_corr_O": float(corr_o.mean()),
            "max_corr_O": float(corr_o.max()),
            "best_O_turbine": str(corr_o.idxmax()),
            "affinity_delta (O - H)": float(corr_o.mean() - corr_h.mean()),
            "closer_park": "Park O" if corr_o.mean() > corr_h.mean() else "Park H",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = load_all_turbine_data("data")
    corrs = compute_correlation_matrices(data)
    print("Wind speed correlation:")
    print(corrs["wind_speed"].round(3))
    print("\nAffinity:")
    print(analyze_park_affinity(corrs["wind_speed"]))
