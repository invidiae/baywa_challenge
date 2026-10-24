"""
Bayesian Ridge Wind Speed Prediction and Explainable Transfer Module.

Stage 1: Predicts target turbine wind speeds (and probabilistic quantiles)
from reference wind farms (Park O and Park H) using Bayesian Linear Regression.

Stage 2 (Integrated): The predicted wind speeds are subsequently mapped through
the manufacturer power curve (speed_to_power.py) to calculate electrical power
generation (kW) and energy yield loss (kWh / MWh), with optional IEC 61400-12-1
air density correction.

Why Bayesian Ridge?
1. 100% Explainable: Yields a single, transparent mathematical formula per turbine:
   v_T = w_0 + w_1 * v_O + w_2 * v_H + w_3 * u_O + w_4 * v_O + w_5 * u_H + w_6 * v_H
2. Exact Analytical Uncertainty: Estimates posterior distribution N(mu, sigma^2)
   giving closed-form, monotonic quantiles (P10, P50, P90) with zero quantile crossing.
3. Regularized & Robust: Hyperparameters (alpha, lambda) are automatically learned
   via Maximum Marginal Likelihood (Type-II MLE), preventing overfitting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from speed_to_power import PowerCurveModel, load_target_power_curves


REQUIRED_COLUMNS = {"turbine_name", "timestamp", "wind_speed", "wind_direction"}


def load_turbine_scada(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load turbine SCADA Excel file, parse timestamps, and sort chronologically."""
    file_path = Path(file_path)
    df = pd.read_excel(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def build_reference_features(
    data_dir: Union[str, Path] = "data",
    reference_names: Sequence[str] = ("O1", "O2", "H1", "H2", "H3", "H4", "H5"),
) -> pd.DataFrame:
    """
    Extract physics-grounded, robust features from external reference wind parks.

    Features constructed:
    - speed_O: Median wind speed across Park O turbines (O1, O2)
    - speed_H: Median wind speed across Park H turbines (H1-H5)
    - u_O, v_O: Orthogonal East-West (u) and North-South (v) wind vector components for Park O
    - u_H, v_H: Orthogonal East-West (u) and North-South (v) wind vector components for Park H
    """
    data_dir = Path(data_dir)
    refs = {name: load_turbine_scada(data_dir / f"{name}_data.xlsx").set_index("timestamp") for name in reference_names}

    # Park O wind speed & direction
    o_turbines = [name for name in ("O1", "O2") if name in refs]
    speed_o = pd.DataFrame({name: refs[name]["wind_speed"] for name in o_turbines}).median(axis=1)

    # Use O1 for reference direction; fall back to O2 if missing
    dir_o = refs["O1"]["wind_direction"].combine_first(refs["O2"]["wind_direction"])
    rad_o = np.deg2rad(dir_o)

    # Park H wind speed & direction
    h_turbines = [name for name in ("H1", "H2", "H3", "H4", "H5") if name in refs]
    speed_h = pd.DataFrame({name: refs[name]["wind_speed"] for name in h_turbines}).median(axis=1)

    # Vector-averaged wind direction across Park H
    sin_h = pd.DataFrame({name: np.sin(np.deg2rad(refs[name]["wind_direction"])) for name in h_turbines}).mean(axis=1)
    cos_h = pd.DataFrame({name: np.cos(np.deg2rad(refs[name]["wind_direction"])) for name in h_turbines}).mean(axis=1)
    rad_h = np.arctan2(sin_h, cos_h)

    # Physics features: Speed magnitudes + orthogonal vector components (u = v*sin(theta), v_comp = v*cos(theta))
    features = pd.DataFrame(
        {
            "speed_O": speed_o,
            "speed_H": speed_h,
            "u_O": speed_o * np.sin(rad_o),
            "v_O": speed_o * np.cos(rad_o),
            "u_H": speed_h * np.sin(rad_h),
            "v_H": speed_h * np.cos(rad_h),
        },
        index=speed_o.index,
    )

    return features


class BayesianWindSpeedPredictor:
    """
    Explainable Bayesian Ridge Regressor for wind speed transfer with analytical quantiles.
    """

    def __init__(self, target_turbine: str = "T1"):
        self.target_turbine = target_turbine
        self.model = BayesianRidge(compute_score=True)
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BayesianWindSpeedPredictor":
        """Fit Bayesian Ridge regression on valid historical wind speed records."""
        valid_mask = y.notna() & np.isfinite(y) & X.notna().all(axis=1)
        if valid_mask.sum() < 30:
            raise ValueError(f"Insufficient training observations: {valid_mask.sum()}")

        self.feature_names = list(X.columns)
        self.model.fit(X.loc[valid_mask, self.feature_names], y.loc[valid_mask])
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict expected wind speed (m/s) and optionally standard deviation."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        mu, std = self.model.predict(X[self.feature_names], return_std=True)
        # Physical clamp: wind speed cannot be negative
        mu = np.clip(mu, 0.0, None)

        if return_std:
            return mu, std
        return mu

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        """
        Derive exact, monotonic analytical quantiles from the Bayesian posterior:
        v_q = max(0, mu + Phi^{-1}(q) * sigma)
        """
        mu, std = self.predict(X, return_std=True)
        q_dict = {}

        for q in quantiles:
            z = norm.ppf(q)
            q_val = np.clip(mu + z * std, 0.0, None)
            col_name = f"q{int(round(q * 100)):02d}" if 0 < q < 1 else f"q_{q}"
            q_dict[col_name] = q_val

        return pd.DataFrame(q_dict, index=X.index)

    def get_formula(self, decimals: int = 3) -> str:
        """Return the closed-form mathematical formula representing the model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_formula()")

        intercept = round(float(self.model.intercept_), decimals)
        parts = [f"{intercept:+.{decimals}f}"]
        for name, coef in zip(self.feature_names, self.model.coef_):
            parts.append(f"{float(coef):+.{decimals}f} * {name}")

        return f"v_{self.target_turbine} = " + " ".join(parts)

    def get_feature_table(self) -> pd.DataFrame:
        """Return table of feature coefficients and normalized percentage weights."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_feature_table()")

        coefs = self.model.coef_
        abs_weights = np.abs(coefs)
        rel_importance = (abs_weights / abs_weights.sum()) * 100.0

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": coefs,
                "abs_importance_pct": rel_importance,
            }
        ).sort_values("abs_importance_pct", ascending=False).reset_index(drop=True)

        return df


def evaluate_bayesian_pipeline(
    target_name: str,
    data_dir: Union[str, Path] = "data",
    include_direction: bool = True,
    apply_density_correction: bool = False,
) -> Dict[str, Any]:
    """
    Run chronological training and evaluation on the official shown validation window
    (2026-09-04 to 2026-09-09), computing wind speed metrics and power/energy metrics.
    """
    data_dir = Path(data_dir)
    target_df = load_turbine_scada(data_dir / f"{target_name}_data.xlsx").set_index("timestamp")
    all_features = build_reference_features(data_dir=data_dir)

    # Feature selection based on directional toggle
    feature_cols = (
        ["speed_O", "speed_H", "u_O", "v_O", "u_H", "v_H"]
        if include_direction
        else ["speed_O", "speed_H"]
    )
    features = all_features[feature_cols]

    common_idx = target_df.index.intersection(features.index)
    target_df = target_df.loc[common_idx]
    features = features.loc[common_idx]

    train_mask = (target_df.index < "2026-09-01") & target_df["wind_speed"].notna() & features.notna().all(axis=1)
    val_mask = (target_df["test_set"] == True) & target_df["wind_speed"].notna() & features.notna().all(axis=1)

    # 1. Train Bayesian Ridge Model
    predictor = BayesianWindSpeedPredictor(target_turbine=target_name)
    predictor.fit(features.loc[train_mask], target_df.loc[train_mask, "wind_speed"])

    # 2. Stage 1: Predict Wind Speeds & Quantiles
    mu_speed, std_speed = predictor.predict(features.loc[val_mask], return_std=True)
    speed_quantiles = predictor.predict_quantiles(features.loc[val_mask], quantiles=(0.1, 0.5, 0.9))

    y_val_speed = target_df.loc[val_mask, "wind_speed"].to_numpy(dtype=float)
    mae_speed = float(np.mean(np.abs(mu_speed - y_val_speed)))
    rmse_speed = float(np.sqrt(np.mean((mu_speed - y_val_speed) ** 2)))
    r2_speed = float(1.0 - np.sum((y_val_speed - mu_speed) ** 2) / np.sum((y_val_speed - y_val_speed.mean()) ** 2))
    p10_p90_cov = float(np.mean((y_val_speed >= speed_quantiles["q10"]) & (y_val_speed <= speed_quantiles["q90"])) * 100.0)

    # 3. Stage 2: Convert Predicted Speeds to Power via Manufacturer Power Curve
    power_curve = PowerCurveModel.from_excel(data_dir / f"{target_name}_power_curve.xlsx", sheet_name=target_name)

    actual_power_kw = power_curve.predict_power(y_val_speed)
    p50_speed_val = speed_quantiles["q50"].to_numpy(dtype=float)

    if apply_density_correction:
        # IEC 61400-12-1: v_norm = v * (rho / rho_0)^(1/3)
        val_pressure_pa = target_df.loc[val_mask, "air_pressure"].to_numpy(dtype=float) * 100.0
        val_temp_k = target_df.loc[val_mask, "ambient_temperature"].to_numpy(dtype=float) + 273.15
        rho = val_pressure_pa / (287.05 * val_temp_k)
        density_factor = (rho / 1.225) ** (1.0 / 3.0)
        p50_speed_effective = p50_speed_val * density_factor
    else:
        p50_speed_effective = p50_speed_val

    pred_p50_kw = power_curve.predict_power(p50_speed_effective)
    mae_power = float(np.mean(np.abs(pred_p50_kw - actual_power_kw)))
    rmse_power = float(np.sqrt(np.mean((pred_p50_kw - actual_power_kw) ** 2)))

    actual_energy_mwh = float(actual_power_kw.sum() / 6000.0)
    pred_energy_mwh = float(pred_p50_kw.sum() / 6000.0)
    energy_dev_pct = float((pred_energy_mwh - actual_energy_mwh) / actual_energy_mwh * 100.0)

    return {
        "target": target_name,
        "include_direction": include_direction,
        "apply_density_correction": apply_density_correction,
        "predictor": predictor,
        "formula": predictor.get_formula(),
        "feature_table": predictor.get_feature_table(),
        "speed_quantiles": speed_quantiles,
        "metrics": {
            "val_samples": int(val_mask.sum()),
            "speed_mae_mps": round(mae_speed, 3),
            "speed_rmse_mps": round(rmse_speed, 3),
            "speed_r2": round(r2_speed, 3),
            "p10_p90_coverage_pct": round(p10_p90_cov, 1),
            "power_mae_kw": round(mae_power, 1),
            "power_rmse_kw": round(rmse_power, 1),
            "actual_energy_mwh": round(actual_energy_mwh, 2),
            "pred_energy_mwh": round(pred_energy_mwh, 2),
            "energy_deviation_pct": round(energy_dev_pct, 2),
        },
    }


def run_direction_ablation_study(data_dir: Union[str, Path] = "data") -> pd.DataFrame:
    """Run full comparative study comparing model performance without vs with wind direction."""
    records = []
    for target in ["T1", "T2", "T3"]:
        res_no = evaluate_bayesian_pipeline(target, data_dir=data_dir, include_direction=False)
        res_dir = evaluate_bayesian_pipeline(target, data_dir=data_dir, include_direction=True)

        m_no = res_no["metrics"]
        m_dir = res_dir["metrics"]

        records.append({
            "Turbine": target,
            "Speed MAE (No Dir)": m_no["speed_mae_mps"],
            "Speed MAE (With Dir)": m_dir["speed_mae_mps"],
            "Delta Speed MAE": round(m_dir["speed_mae_mps"] - m_no["speed_mae_mps"], 3),
            "Power MAE (No Dir)": m_no["power_mae_kw"],
            "Power MAE (With Dir)": m_dir["power_mae_kw"],
            "Delta Power MAE (kW)": round(m_dir["power_mae_kw"] - m_no["power_mae_kw"], 1),
            "Energy Dev (No Dir) %": m_no["energy_deviation_pct"],
            "Energy Dev (With Dir) %": m_dir["energy_deviation_pct"],
            "Abs Dev Improvement %": round(abs(m_no["energy_deviation_pct"]) - abs(m_dir["energy_deviation_pct"]), 2),
        })

    return pd.DataFrame(records)


def main():
    print("=" * 80)
    print(" BAYESIAN RIDGE WIND SPEED PREDICTION & EXPLAINABLE POWER EVALUATION")
    print("=" * 80)

    for target in ["T1", "T2", "T3"]:
        res = evaluate_bayesian_pipeline(target, include_direction=True)
        print(f"\n[{target}] Explainable Closed-Form Formula (with Direction):")
        print("  " + res["formula"])
        print("\nFeature Weights:")
        print(res["feature_table"].to_string(index=False))
        print("-" * 80)

    print("\nWIND DIRECTION ABLATION STUDY (Validation Window: Sept 4 - Sept 9, 2026):")
    print("=" * 80)
    ablation_df = run_direction_ablation_study()
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
