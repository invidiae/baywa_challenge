"""
Bayesian Ridge Wind Speed Prediction and Explainable Transfer Module.

Stage 1: Predicts target turbine wind speeds (and probabilistic quantiles +
coefficient confidence intervals) from reference wind farms using Bayesian
Linear Regression with directional vector components (u = v*sin(theta), v = v*cos(theta)).

Stage 2: Maps the predicted wind speed quantiles through the manufacturer power curve
(speed_to_power.py) to calculate electrical power (kW) and energy yield loss (MWh),
with full IEC 61400-12-1 air density normalization.
"""

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from speed_to_power import PowerCurveModel, compute_air_density, load_target_power_curves


REQUIRED_COLUMNS = {"timestamp", "wind_speed"}


def load_turbine_scada(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load turbine SCADA file (Excel or CSV), validating required fields."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SCADA file not found: {filepath}")

    if filepath.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath, sep=None, engine="python")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{filepath.name} missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _infer_group_key(name: str) -> str:
    """Group turbine identifiers into logical wind farms (e.g. O1/O2 -> O, H1..H5 -> H)."""
    clean = str(name).strip()
    m = re.match(r"^([A-Za-z]+)\d+$", clean)
    if m:
        return m.group(1).upper()
    return re.sub(r"[^A-Za-z0-9]+", "_", clean).strip("_") or "REF"


def build_features_from_frames(
    refs: Mapping[str, pd.DataFrame],
    include_direction: bool = True,
) -> pd.DataFrame:
    """
    Construct spatial wind speed and orthogonal directional vector features (u, v)
    from any uploaded or disk-loaded dictionary of reference turbine DataFrames
    (each indexed by timestamp).
    """
    if not refs:
        raise ValueError("At least one reference turbine dataset is required.")

    # Normalize index to timestamp
    indexed_refs: Dict[str, pd.DataFrame] = {}
    for name, df in refs.items():
        d = df.copy()
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            d = d.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
        d.index = pd.to_datetime(d.index)
        indexed_refs[name] = d.sort_index()

    # Group turbines by wind farm prefix, keeping O before H when both exist
    groups: Dict[str, List[str]] = {}
    for name in indexed_refs:
        g = _infer_group_key(name)
        groups.setdefault(g, []).append(name)

    ordered_groups = sorted(
        groups.keys(),
        key=lambda g: (0 if g == "O" else (1 if g == "H" else 2), g),
    )

    # Build union timestamp index
    union_idx = indexed_refs[next(iter(indexed_refs))].index
    for d in indexed_refs.values():
        union_idx = union_idx.union(d.index)
    union_idx = union_idx.sort_values()

    speed_cols: Dict[str, pd.Series] = {}
    rad_cols: Dict[str, pd.Series] = {}

    # First pass: compute farm median speeds and circular mean directions
    fallback_deg_series: Optional[pd.Series] = None
    for g in ordered_groups:
        names = groups[g]
        sp_df = pd.DataFrame(
            {n: pd.to_numeric(indexed_refs[n]["wind_speed"], errors="coerce").reindex(union_idx) for n in names},
            index=union_idx,
        )
        speed_cols[g] = sp_df.median(axis=1)

        if include_direction:
            dir_names = [n for n in names if "wind_direction" in indexed_refs[n].columns]
            if dir_names:
                if g == "O" and "O1" in dir_names:
                    # Preserve exact O1-primary / O2-fallback convention for Park O
                    d_o = pd.to_numeric(indexed_refs["O1"]["wind_direction"], errors="coerce").reindex(union_idx)
                    for other_o in [n for n in dir_names if n != "O1"]:
                        d_o = d_o.combine_first(
                            pd.to_numeric(indexed_refs[other_o]["wind_direction"], errors="coerce").reindex(union_idx)
                        )
                    rad_cols[g] = np.deg2rad(d_o)
                else:
                    sin_df = pd.DataFrame(
                        {
                            n: np.sin(np.deg2rad(pd.to_numeric(indexed_refs[n]["wind_direction"], errors="coerce").reindex(union_idx)))
                            for n in dir_names
                        },
                        index=union_idx,
                    )
                    cos_df = pd.DataFrame(
                        {
                            n: np.cos(np.deg2rad(pd.to_numeric(indexed_refs[n]["wind_direction"], errors="coerce").reindex(union_idx)))
                            for n in dir_names
                        },
                        index=union_idx,
                    )
                    rad_g = pd.Series(np.arctan2(sin_df.mean(axis=1), cos_df.mean(axis=1)), index=union_idx)
                    rad_cols[g] = rad_g
                    if fallback_deg_series is None:
                        fallback_deg_series = (np.rad2deg(rad_g) % 360.0)

    # Fill isolated directional gaps using cross-farm fallback + linear interpolation
    for g in list(rad_cols.keys()):
        deg_g = pd.Series(np.rad2deg(rad_cols[g]) % 360.0, index=union_idx)
        if fallback_deg_series is not None:
            deg_g = deg_g.combine_first(fallback_deg_series)
        deg_g = deg_g.interpolate(method="linear").bfill().ffill()
        rad_cols[g] = np.deg2rad(deg_g)

    # Assemble feature columns: [speed_g1, speed_g2, ..., u_g1, v_g1, u_g2, v_g2, ...]
    feat_dict: Dict[str, pd.Series] = {}
    for g in ordered_groups:
        feat_dict[f"speed_{g}"] = speed_cols[g]
    if include_direction:
        for g in ordered_groups:
            if g in rad_cols:
                feat_dict[f"u_{g}"] = speed_cols[g] * np.sin(rad_cols[g])
                feat_dict[f"v_{g}"] = speed_cols[g] * np.cos(rad_cols[g])

    features = pd.DataFrame(feat_dict, index=union_idx)
    features = features.interpolate(method="linear").bfill().ffill()
    return features


def compute_air_density_from_frames(refs: Mapping[str, pd.DataFrame]) -> Optional[pd.Series]:
    """
    Compute regional air density rho (kg/m^3) from reference turbines that carry
    valid, time-varying barometric pressure and ambient temperature telemetry.
    Returns None if no uploaded reference file contains valid pressure & temperature.
    """
    press_list: List[pd.Series] = []
    temp_list: List[pd.Series] = []

    for df in refs.values():
        d = df.copy()
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            d = d.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
        if {"air_pressure", "ambient_temperature"}.issubset(d.columns):
            p = pd.to_numeric(d["air_pressure"], errors="coerce")
            t = pd.to_numeric(d["ambient_temperature"], errors="coerce")
            # Exclude constant placeholder pressure columns (such as Park O)
            if p.notna().sum() > 30 and float(p.std()) > 0.05 and t.notna().sum() > 30:
                press_list.append(p)
                temp_list.append(t)

    if not press_list:
        return None

    med_press = pd.concat(press_list, axis=1).median(axis=1).sort_index()
    med_temp = pd.concat(temp_list, axis=1).median(axis=1).sort_index()
    rho = pd.Series(compute_air_density(med_press, med_temp), index=med_press.index)
    return rho.interpolate(method="linear").bfill().ffill()


def build_reference_features(
    data_dir: Union[str, Path] = "data",
    reference_names: Sequence[str] = ("O1", "O2", "H1", "H2", "H3", "H4", "H5"),
    include_direction: bool = True,
) -> pd.DataFrame:
    """Load standard reference turbines from disk and build physics features."""
    data_dir = Path(data_dir)
    refs = {
        name: load_turbine_scada(data_dir / f"{name}_data.xlsx").set_index("timestamp")
        for name in reference_names
    }
    return build_features_from_frames(refs, include_direction=include_direction)


def compute_reference_air_density(data_dir: Union[str, Path] = "data") -> pd.Series:
    """Compute reference air density rho (kg/m^3) across Park H turbines on disk."""
    data_dir = Path(data_dir)
    refs = {
        h: load_turbine_scada(data_dir / f"{h}_data.xlsx").set_index("timestamp")
        for h in ("H1", "H2", "H3", "H4", "H5")
    }
    rho = compute_air_density_from_frames(refs)
    if rho is None:
        raise RuntimeError("Could not compute reference air density from Park H files.")
    return rho


class BayesianWindSpeedPredictor:
    """
    Explainable Bayesian Ridge Regressor for wind speed transfer with:
    - Posterior coefficient confidence intervals (from posterior covariance Sigma_w)
    - Decomposed prediction uncertainty (pure coefficient CI vs full predictive interval)
    """

    def __init__(self, target_turbine: str = "T1"):
        self.target_turbine = target_turbine
        self.model = BayesianRidge(compute_score=True)
        self.feature_names: List[str] = []
        self.n_train_: int = 0
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BayesianWindSpeedPredictor":
        """Fit Bayesian Ridge regression on valid historical wind speed records."""
        self.feature_names = list(X.columns)
        valid_mask = y.notna() & np.isfinite(y) & X[self.feature_names].notna().all(axis=1)
        if valid_mask.sum() < 30:
            raise ValueError(f"Insufficient training observations: {valid_mask.sum()}")

        X_fit = X.loc[valid_mask, self.feature_names]
        y_fit = y.loc[valid_mask]
        self.model.fit(X_fit, y_fit)
        self.n_train_ = int(len(X_fit))
        self.is_fitted = True
        return self

    def _clean_input(self, X: pd.DataFrame) -> pd.DataFrame:
        X_in = X[self.feature_names].copy()
        if X_in.isna().any().any():
            X_in = X_in.interpolate(method="linear").bfill().ffill()
        return X_in

    def predict_decomposed(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Return posterior mean wind speed along with BOTH uncertainty components:
        - std_coef: Standard error of the regression surface from coefficient covariance
                    Var(f(x)) = 1/(alpha * N) + (x - x_bar)^T Sigma_w (x - x_bar)
        - std_pred: Full Bayesian predictive standard deviation including wind turbulence
                    Var(y|x)  = 1/alpha       + (x - x_bar)^T Sigma_w (x - x_bar)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_decomposed()")

        X_in = self._clean_input(X)
        mu_raw, std_pred = self.model.predict(X_in, return_std=True)
        mu = np.clip(mu_raw, 0.0, None)

        dx = X_in.to_numpy(dtype=float) - np.asarray(self.model.X_offset_, dtype=float)
        quad_form = np.einsum("ij,jk,ik->i", dx, self.model.sigma_, dx)
        var_intercept_mean = 1.0 / (float(self.model.alpha_) * max(self.n_train_, 1))
        std_coef = np.sqrt(np.maximum(var_intercept_mean + quad_form, 0.0))

        return {
            "mu": mu,
            "std_coef": std_coef,
            "std_pred": std_pred,
        }

    def predict(self, X: pd.DataFrame, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict expected wind speed (m/s) and optionally predictive standard deviation."""
        dec = self.predict_decomposed(X)
        if return_std:
            return dec["mu"], dec["std_pred"]
        return dec["mu"]

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        """
        Derive exact, monotonic analytical quantiles from the Bayesian posterior:
        v_q = max(0, mu + Phi^{-1}(q) * sigma_pred)
        """
        dec = self.predict_decomposed(X)
        mu, std = dec["mu"], dec["std_pred"]
        q_dict = {}

        for q in quantiles:
            z = norm.ppf(q)
            q_val = np.clip(mu + z * std, 0.0, None)
            pct = q * 100.0
            if abs(pct - round(pct)) < 1e-6 and 0 < q < 1:
                col_name = f"q{int(round(pct)):02d}"
            else:
                col_name = f"q_{pct:.1f}".replace(".", "p")
            q_dict[col_name] = q_val

        return pd.DataFrame(q_dict, index=X.index)

    def predict_certainty_band(
        self,
        X: pd.DataFrame,
        certainty_pct: float = 80.0,
        coef_ci_pct: float = 95.0,
    ) -> pd.DataFrame:
        """
        Compute symmetric Bayesian predictive band for any chosen percentile certainty
        (e.g. 80% -> P10..P90; 90% -> P5..P95; 95% -> P2.5..P97.5) alongside the
        mean-regression confidence interval propagated from coefficient covariance.
        """
        if not (0.0 < certainty_pct < 100.0):
            raise ValueError("certainty_pct must be strictly between 0 and 100.")

        dec = self.predict_decomposed(X)
        mu = dec["mu"]
        std_pred = dec["std_pred"]
        std_coef = dec["std_coef"]

        alpha_tail = (1.0 - certainty_pct / 100.0) / 2.0
        z_pred = float(norm.ppf(1.0 - alpha_tail))
        z_coef = float(norm.ppf(1.0 - (1.0 - coef_ci_pct / 100.0) / 2.0))

        return pd.DataFrame(
            {
                "lower": np.clip(mu - z_pred * std_pred, 0.0, None),
                "median": mu,
                "upper": np.clip(mu + z_pred * std_pred, 0.0, None),
                "coef_ci_lower": np.clip(mu - z_coef * std_coef, 0.0, None),
                "coef_ci_upper": np.clip(mu + z_coef * std_coef, 0.0, None),
                "coef_ci_halfwidth": z_coef * std_coef,
                "pred_std": std_pred,
            },
            index=X.index,
        )

    def get_formula(self) -> str:
        """Return the closed-form algebraic formula representing the trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_formula()")

        terms = [f"{self.model.intercept_:+.3f}"]
        for name, coef in zip(self.feature_names, self.model.coef_):
            terms.append(f"{coef:+.3f} * {name}")
        return f"v_{self.target_turbine} = " + " ".join(terms)

    def get_coefficient_summary(self, ci_level: float = 0.95) -> pd.DataFrame:
        """
        Return full parameter table (Intercept + all slopes) with posterior
        standard errors and confidence intervals derived from Sigma_w.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_coefficient_summary()")

        z = float(norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
        x_bar = np.asarray(self.model.X_offset_, dtype=float)
        sigma_w = np.asarray(self.model.sigma_, dtype=float)

        # Intercept standard error: Var(w_0) = 1/(alpha * N) + x_bar^T Sigma_w x_bar
        var_w0 = 1.0 / (float(self.model.alpha_) * max(self.n_train_, 1)) + float(x_bar @ sigma_w @ x_bar)
        se_w0 = float(np.sqrt(max(var_w0, 0.0)))
        w0 = float(self.model.intercept_)

        se_slopes = np.sqrt(np.maximum(np.diag(sigma_w), 0.0))
        coefs = np.asarray(self.model.coef_, dtype=float)
        abs_coefs = np.abs(coefs)
        total_imp = abs_coefs.sum() if abs_coefs.sum() > 0 else 1.0

        rows = [
            {
                "parameter": "Intercept (w_0)",
                "coefficient": w0,
                "std_error": se_w0,
                "ci_lower": w0 - z * se_w0,
                "ci_upper": w0 + z * se_w0,
                "abs_importance_pct": np.nan,
            }
        ]
        for name, c, se in zip(self.feature_names, coefs, se_slopes):
            rows.append(
                {
                    "parameter": name,
                    "coefficient": float(c),
                    "std_error": float(se),
                    "ci_lower": float(c - z * se),
                    "ci_upper": float(c + z * se),
                    "abs_importance_pct": float(abs(c) / total_imp * 100.0),
                }
            )

        return pd.DataFrame(rows)

    def get_feature_table(self, ci_level: float = 0.95) -> pd.DataFrame:
        """Return feature weights table with posterior CIs and relative importance."""
        summary = self.get_coefficient_summary(ci_level=ci_level)
        feat_df = summary.iloc[1:].copy().rename(columns={"parameter": "feature"})
        return feat_df.sort_values("abs_importance_pct", ascending=False).reset_index(drop=True)


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
    features = build_reference_features(data_dir=data_dir, include_direction=include_direction)

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

    power_curve = PowerCurveModel.from_excel(data_dir / f"{target_name}_power_curve.xlsx", sheet_name=target_name)
    actual_power_kw = power_curve.predict_power(y_val_speed)

    val_density = None
    if density_mode == "local":
        val_press = target_df.loc[val_mask, "air_pressure"]
        val_temp = target_df.loc[val_mask, "ambient_temperature"]
        val_density = compute_air_density(val_press, val_temp)
    elif density_mode == "reference_H":
        rho_ref_series = compute_reference_air_density(data_dir=data_dir)
        val_density = rho_ref_series.loc[val_mask].to_numpy(dtype=float)

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
        "coefficient_summary": predictor.get_coefficient_summary(),
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
        print("\nPosterior Coefficient Summary (95% Confidence Intervals):")
        print(res["coefficient_summary"].to_string(index=False))
        print("-" * 80)

    print("\nSTAGE 2 BENCHMARK: STANDARD vs. IEC 61400-12-1 AIR DENSITY CORRECTION:")
    print("=" * 80)
    bench_df = run_full_comparative_benchmark()
    print(bench_df.to_string(index=False))


if __name__ == "__main__":
    main()
