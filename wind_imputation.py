"""Multi-reference SCADA imputation, power conversion, and chronological metrics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

@dataclass
class ImputationResult:
    test: pd.DataFrame; outage: pd.DataFrame; metrics: dict[str, float]
    selected_model: str; cv_scores: dict[str, float]; interval_minutes: float

def apply_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append IEC-aligned atmospheric and wake-aware SCADA features.

    Parameters
    ----------
    df:
        A DataFrame with ``wind_speed`` (m/s), ``wind_direction`` (degrees),
        ``temperature`` (degrees C), and ``pressure`` (hPa or mbar). Rows are
        assumed to be ordered chronologically at the usual 10-minute cadence.

    Returns
    -------
    pandas.DataFrame
        A copy of the input with air density, density-normalised wind speed,
        cyclical direction terms, the 250-290 degree wake-sector indicator,
        and a six-period (one-hour) wind-speed rolling standard deviation.

    Notes
    -----
    The air-density calculation follows the requested dry-air ideal-gas form
    used in IEC 61400-12-1-style normalisation. The initial rolling-standard-
    deviation value is set to zero because no prior 10-minute observation
    exists from which to estimate turbulence.
    """
    required = {"wind_speed", "wind_direction", "temperature", "pressure"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing physics input columns: {sorted(missing)}")

    result = df.copy()
    wind_speed = pd.to_numeric(result["wind_speed"], errors="coerce")
    direction = pd.to_numeric(result["wind_direction"], errors="coerce") % 360.0
    temperature_kelvin = pd.to_numeric(result["temperature"], errors="coerce") + 273.15
    pressure_pa = pd.to_numeric(result["pressure"], errors="coerce") * 100.0

    # rho = p / (R_specific * T), with R_specific for dry air in J/(kg K).
    result["air_density_kg_m3"] = pressure_pa / (287.05 * temperature_kelvin)
    result["normalized_wind_speed"] = wind_speed * (result["air_density_kg_m3"] / 1.225) ** (1.0 / 3.0)
    radians = np.deg2rad(direction)
    result["sin_wind_dir"] = np.sin(radians)
    result["cos_wind_dir"] = np.cos(radians)
    result["is_wake_sector"] = direction.between(250.0, 290.0, inclusive="both").astype(float)
    result["wind_speed_rolling_std_1h"] = wind_speed.rolling(window=6, min_periods=1).std(ddof=0).fillna(0.0)
    return result

def load_scada(upload: Any) -> pd.DataFrame:
    """Read CSV/XLSX SCADA and preserve missing wind speeds as outage candidates."""
    name = str(getattr(upload, "name", upload)).lower()
    df = pd.read_excel(upload) if name.endswith((".xlsx", ".xls")) else pd.read_csv(upload)
    if missing := {"timestamp", "wind_speed"} - set(df.columns): raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[["timestamp", "wind_speed"]].copy(); df.timestamp = pd.to_datetime(df.timestamp, errors="coerce"); df.wind_speed = pd.to_numeric(df.wind_speed, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.timestamp.duplicated().any(): raise ValueError("Timestamps must be unique.")
    return df

def load_power_curve(upload: Any | None) -> pd.DataFrame | None:
    """Read an optional CSV/XLSX curve with ``wind_speed`` and ``power`` columns."""
    if upload is None: return None
    name = str(getattr(upload, "name", upload)).lower(); curve = pd.read_excel(upload) if name.endswith((".xlsx", ".xls")) else pd.read_csv(upload)
    if missing := {"wind_speed", "power"} - set(curve.columns): raise ValueError(f"Power curve missing: {sorted(missing)}")
    return curve[["wind_speed", "power"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("wind_speed").drop_duplicates("wind_speed")

def power_curve(speed: pd.Series | np.ndarray, curve: pd.DataFrame | None = None) -> np.ndarray:
    """Use an uploaded turbine curve or the documented 3/12/25 m/s, 3000-kW curve."""
    v = np.asarray(speed, dtype=float)
    if curve is not None:
        return np.where(np.isfinite(v), np.interp(v, curve.wind_speed, curve.power, left=0., right=0.), np.nan)
    p = np.zeros_like(v); ramp, rated = (v >= 3) & (v < 12), (v >= 12) & (v <= 25)
    p[ramp] = 3000 * ((v[ramp] - 3) / 9) ** 3; p[rated] = 3000
    return np.where(np.isfinite(v), p, np.nan)

def _frame(target: pd.DataFrame, references: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Time-align every reference and derive robust ensemble wind features."""
    base = target.rename(columns={"wind_speed":"actual_wind_speed"}).copy()
    columns = []
    for i, ref in enumerate(references):
        col = f"reference_{i+1}_wind_speed"; columns.append(col)
        base = base.merge(ref.rename(columns={"wind_speed":col}), on="timestamp", how="left")
    if len(base) < 100: raise ValueError("Need at least 100 target observations.")
    base["reference_wind_mean"] = base[columns].mean(axis=1); base["reference_wind_median"] = base[columns].median(axis=1)
    base["reference_wind_std"] = base[columns].std(axis=1); base["reference_available_count"] = base[columns].notna().sum(axis=1)
    base["hour"], base["day_of_year"] = base.timestamp.dt.hour, base.timestamp.dt.dayofyear
    return base.sort_values("timestamp").reset_index(drop=True)

def _models() -> dict[str, Pipeline]:
    pre = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    params = dict(n_estimators=300, min_samples_leaf=2, max_features=.9, n_jobs=-1, random_state=42)
    return {"Extra Trees":Pipeline(pre+[("model",ExtraTreesRegressor(**params))]), "Random Forest":Pipeline(pre+[("model",RandomForestRegressor(**params))])}

def _predict(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    trees = np.vstack([tree.predict(model.named_steps["imputer"].transform(x)) for tree in model.named_steps["model"].estimators_])
    return tuple(np.quantile(trees, q, axis=0) for q in (.1,.5,.9))

def train_and_evaluate(target: pd.DataFrame, references: Sequence[pd.DataFrame], curve: pd.DataFrame | None = None, test_fraction: float = .25) -> ImputationResult:
    """Select a model on development-only chronological folds; evaluate final block."""
    if not .2 <= test_fraction <= .3: raise ValueError("Test fraction must be 20-30%.")
    df = _frame(target, references)
    # Keep every uploaded reference as an explicit feature, alongside robust
    # fleet summaries that help when one reference has a communication gap.
    individual_references = [column for column in df.columns if column.startswith("reference_") and column.endswith("_wind_speed")]
    features = individual_references + ["reference_wind_mean","reference_wind_median","reference_wind_std","reference_available_count","hour","day_of_year"]
    split = int(len(df)*(1-test_fraction)); observed = df.actual_wind_speed.notna()
    dev = df.iloc[:split].loc[observed.iloc[:split]].reset_index(drop=True); test = df.iloc[split:].loc[observed.iloc[split:]].copy()
    if len(dev)<80 or len(test)<20: raise ValueError("Insufficient observed values for chronological assessment.")
    scores={}
    for name, candidate in _models().items():
        fold=[]
        for tr, va in TimeSeriesSplit(n_splits=3).split(dev):
            candidate.fit(dev.loc[tr,features],dev.loc[tr,"actual_wind_speed"]); fold.append(np.sqrt(mean_squared_error(dev.loc[va,"actual_wind_speed"],candidate.predict(dev.loc[va,features]))))
        scores[name]=float(np.mean(fold))
    selected=min(scores,key=scores.get); assessed=_models()[selected].fit(dev[features],dev.actual_wind_speed)
    p10,p50,p90=_predict(assessed,test[features]); test[["p10_wind_speed","predicted_wind_speed","p90_wind_speed"]]=np.column_stack((p10,p50,p90))
    test["actual_power_kw"],test["predicted_power_kw"]=power_curve(test.actual_wind_speed,curve),power_curve(p50,curve)
    mins=float(test.timestamp.diff().dt.total_seconds().median()/60); mins=mins if np.isfinite(mins) and mins>0 else 10.
    actual,predicted=(test.actual_power_kw*mins/60).sum(),(test.predicted_power_kw*mins/60).sum()
    m={"wind_rmse_mps":float(np.sqrt(mean_squared_error(test.actual_wind_speed,p50))),"wind_mae_mps":float(mean_absolute_error(test.actual_wind_speed,p50)),"power_rmse_kw":float(np.sqrt(mean_squared_error(test.actual_power_kw,test.predicted_power_kw))),"power_mae_kw":float(mean_absolute_error(test.actual_power_kw,test.predicted_power_kw)),"actual_total_kwh":float(actual),"predicted_total_kwh":float(predicted),"energy_yield_deviation_kwh":float(predicted-actual)}
    # Retrain on every observed target row for operational outage imputation.
    production=_models()[selected].fit(df.loc[observed,features],df.loc[observed,"actual_wind_speed"]); outage=df.loc[~observed].copy()
    if len(outage):
        lo,mid,hi=_predict(production,outage[features]); outage[["p10_wind_speed","predicted_wind_speed","p90_wind_speed"]]=np.column_stack((lo,mid,hi)); outage["predicted_power_kw"]=power_curve(mid,curve); outage["predicted_energy_kwh"]=outage.predicted_power_kw*mins/60
    return ImputationResult(test,outage,m,selected,scores,mins)
