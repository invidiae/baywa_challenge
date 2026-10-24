"""
Bayesian Ridge Wind Speed Prediction and Explainable Transfer Module.

Stage 1: Predicts target turbine wind speeds (and probabilistic quantiles)
from reference wind farms (Park O and Park H) using Bayesian Linear Regression
with directional vector components (u = v*sin(theta), v = v*cos(theta)).

Stage 2: Maps the predicted wind speed quantiles through the manufacturer power curve
(speed_to_power.py) to calculate electrical power (kW) and energy yield loss (kWh / MWh),
with full IEC 61400-12-1 air density normalization (Addon 2).

Why Bayesian Ridge + Physics Air Density?
1. 100% Explainable: Yields a single, transparent mathematical formula per turbine:
   v_T = w_0 + w_1 * v_O + w_2 * v_H + w_3 * u_O + w_4 * v_O + w_5 * u_H + w_6 * v_H
2. Exact Analytical Uncertainty: Estimates posterior distribution N(mu, sigma^2)
   giving closed-form, monotonic quantiles (P10, P50, P90) with zero quantile crossing.
3. Aerodynamic Fidelity (IEC 61400-12-1): Normalizes wind speed by air density rho:
   v_norm = v * (rho / 1.225)^(1/3) using live ambient pressure and temperature.
   During an outage, rho is reconstructed from Reference Park H (r = 0.9785 with Park T).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from speed_to_power import PowerCurveModel, compute_air_density, load_target_power_curves


REQUIRED_COLUMNS = {"turbine_name", "timestamp", "wind_speed", "wind_direction"}


def load_turbine_scada(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load turbine SCADA file (Excel or CSV), validating required fields."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SCADA file not found: {filepath}")

    if filepath.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{filepath.name} missing required columns: {missing}")

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

    # Use O1 for reference direction; fall back to O2 if missing, then to Park H
    dir_o = refs["O1"]["wind_direction"].combine_first(refs["O2"]["wind_direction"])

    # Park H wind speed & direction
    h_turbines = [name for name in ("H1", "H2", "H3", "H4", "H5") if name in refs]
    speed_h = pd.DataFrame({name: refs[name]["wind_speed"] for name in h_turbines}).median(axis=1)

    # Vector-averaged wind direction across Park H
    sin_h = pd.DataFrame({name: np.sin(np.deg2rad(refs[name]["wind_direction"])) for name in h_turbines}).mean(axis=1)
    cos_h = pd.DataFrame({name: np.cos(np.deg2rad(refs[name]["wind_direction"])) for name in h_turbines}).mean(axis=1)
    rad_h = np.arctan2(sin_h, cos_h)

    # If Park O direction has isolated gap, fall back to Park H direction
    deg_h = np.rad2deg(rad_h) % 360.0
    dir_o = dir_o.combine_first(pd.Series(deg_h, index=speed_o.index)).interpolate().bfill().ffill()
    rad_o = np.deg2rad(dir_o)

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

    # Guarantee zero NaNs in reference telemetry via linear interpolation
    features = features.interpolate(method="linear").bfill().ffill()
    return features


def compute_reference_air_density(data_dir: Union[str, Path] = "data") -> pd.Series:
    """
    Compute reference air density rho (kg/m^3) across Park H turbines (H1-H5),
    which provide reliable, time-varying barometric pressure and temperature.
    (Correlation with Target Park T air density: r = 0.9785).
    """
    data_dir = Path(data_dir)
    h_pressures, h_temps = [], []
    for h in ("H1", "H2", "H3", "H4", "H5"):
        hdf = load_turbine_scada(data_dir / f"{h}_data.xlsx").set_index("timestamp")
        h_pressures.append(hdf["air_pressure"])
        h_temps.append(hdf["ambient_temperature"])

    med_press = pd.concat(h_pressures, axis=1).median(axis=1)
    med_temp = pd.concat(h_temps, axis=1).median(axis=1)
    rho_series = pd.Series(compute_air_density(med_press, med_temp), index=med_press.index)
    return rho_series.interpolate().bfill().ffill()


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
        self.feature_names = list(X.columns)
        valid_mask = y.notna() & np.isfinite(y) & X[self.feature_names].notna().all(axis=1)
        if valid_mask.sum() < 30:
            raise ValueError(f"Insufficient training observations: {valid_mask.sum()}")

        self.model.fit(X.loc[valid_mask, self.feature_names], y.loc[valid_mask])
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict expected wind speed (m/s) and optionally standard deviation."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        X_in = X[self.feature_names].copy()
        if X_in.isna().any().any():
            X_in = X_in.interpolate().bfill().ffill()

        mu, std = self.model.predict(X_in, return_std=True)
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

    def get_formula(self) -> str:
        """Return the closed-form algebraic formula representing the trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_formula()")

        terms = [f"{self.model.intercept_:+.3f}"]
        for name, coef in zip(self.feature_names, self.model.coef_):
            terms.append(f"{coef:+.3f} * {name}")
        return f"v_{self.target_turbine} = " + " ".join(terms)

    def get_feature_table(self) -> pd.DataFrame:
        """Return table of coefficients and absolute relative feature importance."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_feature_table()")

        abs_coefs = np.abs(self.model.coef_)
        total_importance = abs_coefs.sum() if abs_coefs.sum() > 0 else 1.0
        importance_pct = (abs_coefs / total_importance) * 100.0

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": self.model.coef_,
                "abs_importance_pct": importance_pct,
            }
        ).sort_values("abs_importance_pct", ascending=False)
        return df


def evaluate_bayesian_pipeline(
    target_name: str,
    data_dir: Union[str, Path] = "data",
    include_direction: bool = True,
    density_mode: str = "none",  # "none", "local", "reference_H"
) -> Dict[str, Any]:
    """
    Run chronological training and evaluation on the official shown validation window
    (2026-09-04 to 2026-09-09), computing wind speed metrics and power/energy metrics.
    """
    data_dir = Path(data_dir)
    target_df = load_turbine_scada(data_dir / f"{target_name}_data.xlsx").set_index("timestamp")
    all_features = build_reference_features(data_dir=data_dir)

    feature_cols = (
        ["speed_O", "speed_H", "u_O", "v_O", "u_H", "v_H"]
        if include_direction
        else ["speed_O", "speed_H"]
    )
    features = all_features[feature_cols]

    common_idx = target_df.index.intersection(features.index)
    target_df = target_df.loc[common_idx]
    features = features.loc[common_idx]

    train_mask = (target_df["test_set"] == False) & target_df["wind_speed"].notna() & features.notna().all(axis=1)
    val_mask = (target_df.index >= "2026-09-04") & (target_df.index < "2026-09-10") & target_df["wind_speed"].notna() & features.notna().all(axis=1)

    predictor = BayesianWindSpeedPredictor(target_turbine=target_name)
    predictor.fit(features.loc[train_mask], target_df.loc[train_mask, "wind_speed"])

    speed_quantiles = predictor.predict_quantiles(features.loc[val_mask], quantiles=(0.1, 0.5, 0.9))
    mu_speed = speed_quantiles["q50"].to_numpy(dtype=float)
    y_val_speed = target_df.loc[val_mask, "wind_speed"].to_numpy(dtype=float)

    mae_speed = float(np.mean(np.abs(mu_speed - y_val_speed)))
    rmse_speed = float(np.sqrt(np.mean((mu_speed - y_val_speed) ** 2)))
    r2_speed = float(1.0 - np.sum((y_val_speed - mu_speed) ** 2) / np.sum((y_val_speed - y_val_speed.mean()) ** 2))
    p10_p90_cov = float(np.mean((y_val_speed >= speed_quantiles["q10"]) & (y_val_speed <= speed_quantiles["q90"])) * 100.0)

    # 3. Stage 2: Convert Predicted Speeds to Power via Manufacturer Power Curve
    power_curve = PowerCurveModel.from_excel(data_dir / f"{target_name}_power_curve.xlsx", sheet_name=target_name)
    actual_power_kw = power_curve.predict_power(y_val_speed)

    # Determine air density vector based on density_mode
    val_density = None
    if density_mode == "local":
        val_press = target_df.loc[val_mask, "air_pressure"]
        val_temp = target_df.loc[val_mask, "ambient_temperature"]
        val_density = compute_air_density(val_press, val_temp)
    elif density_mode == "reference_H":
        rho_ref_series = compute_reference_air_density(data_dir=data_dir)
        val_density = rho_ref_series.loc[val_mask].to_numpy(dtype=float)

    # Predict power with density correction
    p50_speed_val = speed_quantiles["q50"].to_numpy(dtype=float)
    pred_p50_kw = power_curve.predict_power(p50_speed_val, air_density=val_density)

    mae_power = float(np.mean(np.abs(pred_p50_kw - actual_power_kw)))
    rmse_power = float(np.sqrt(np.mean((pred_p50_kw - actual_power_kw) ** 2)))

    actual_energy_mwh = float(actual_power_kw.sum() / 6000.0)
    pred_energy_mwh = float(pred_p50_kw.sum() / 6000.0)
    energy_dev_pct = float((pred_energy_mwh - actual_energy_mwh) / actual_energy_mwh * 100.0)

    loss_quantiles = power_curve.estimate_outage_loss_quantiles(
        speed_quantiles,
        timestamps=target_df.loc[val_mask].index,
        air_density=val_density,
    )

    return {
        "target": target_name,
        "include_direction": include_direction,
        "density_mode": density_mode,
        "predictor": predictor,
        "formula": predictor.get_formula(),
        "feature_table": predictor.get_feature_table(),
        "speed_quantiles": speed_quantiles,
        "loss_quantiles": loss_quantiles,
        "metrics": {
            "train_samples": int(train_mask.sum()),
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


def run_full_comparative_benchmark(data_dir: Union[str, Path] = "data") -> pd.DataFrame:
    """Run comparative benchmark between standard power curves and IEC 61400-12-1 density correction."""
    rows = []
    for target in ["T1", "T2", "T3"]:
        res_std = evaluate_bayesian_pipeline(target, data_dir=data_dir, include_direction=True, density_mode="none")
        res_dens = evaluate_bayesian_pipeline(target, data_dir=data_dir, include_direction=True, density_mode="reference_H")

        m_s = res_std["metrics"]
        m_d = res_dens["metrics"]

        rows.append({
            "Turbine": target,
            "Power MAE (Standard)": m_s["power_mae_kw"],
            "Power MAE (IEC Density)": m_d["power_mae_kw"],
            "Energy Dev (Standard) %": m_s["energy_deviation_pct"],
            "Energy Dev (IEC Density) %": m_d["energy_deviation_pct"],
            "Actual MWh": m_s["actual_energy_mwh"],
            "Pred MWh (Standard)": m_s["pred_energy_mwh"],
            "Pred MWh (IEC Density)": m_d["pred_energy_mwh"],
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print(" BAYESIAN RIDGE WIND TRANSFER & IEC 61400-12-1 AIR DENSITY NORMALIZATION")
    print("=" * 80)

    for target in ["T1", "T2", "T3"]:
        res = evaluate_bayesian_pipeline(target, include_direction=True, density_mode="reference_H")
        print(f"\n[{target}] Closed-Form Model Equation:")
        print("  " + res["formula"])
        print("\nFeature Weights:")
        print(res["feature_table"].to_string(index=False))
        print("-" * 80)

    print("\nSTAGE 2 BENCHMARK: STANDARD vs. IEC 61400-12-1 AIR DENSITY CORRECTION:")
    print("=" * 80)
    bench_df = run_full_comparative_benchmark()
    print(bench_df.to_string(index=False))


if __name__ == "__main__":
    main()
