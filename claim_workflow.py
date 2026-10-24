"""Auditable end-to-end outage forecast and business-interruption scenario."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import external_features, load_scada
from financial_engine import BusinessInterruptionEngine
from quantile_model import QuantileLossPredictor
from speed_to_power import PowerCurveModel


def run_claim_scenario(target: str, data_dir: str | Path = "data", *, ppa_rate: float = 0.08,
                       ptc_rate: float = 0.025, gross_profit_rate: float = 0.85,
                       deductible_hours: int = 24) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Forecast missing target power from external farms and price each outage event."""
    data_dir = Path(data_dir)
    target_frame = load_scada(data_dir / f"{target}_data.xlsx")
    reference_names = ("H1", "H2", "H3", "H4", "H5", "O1", "O2")
    features = external_features({name: load_scada(data_dir / f"{name}_data.xlsx") for name in reference_names})
    frame = target_frame.set_index("timestamp").join(features, how="left")
    curve = PowerCurveModel.from_excel(data_dir / f"{target}_power_curve.xlsx")
    power = curve.predict_power(frame["wind_speed"])
    training = np.isfinite(power)
    model = QuantileLossPredictor().fit(frame.loc[training, features.columns], pd.Series(power[training], index=frame.index[training]))
    missing = frame["wind_speed"].isna()
    forecast = model.predict_quantiles(frame.loc[missing, features.columns])
    forecast.insert(0, "timestamp", forecast.index)
    forecast["outage_event_id"] = pd.Series(forecast.index, index=forecast.index).diff().gt(pd.Timedelta(minutes=10)).cumsum().to_numpy()
    forecast["interval_energy_kwh"] = forecast["P50_yield"] / 6.0
    engine = BusinessInterruptionEngine(ppa_rate, ptc_rate, gross_profit_rate, deductible_hours)
    scenarios: dict[str, dict[str, float]] = {}
    for label in ("P90", "P50", "P10"):
        event_claims = []
        for _, event in forecast.groupby("outage_event_id"):
            gross = float(event[f"{label}_yield"].sum() / 6.0 * (ppa_rate + ptc_rate))
            event_claims.append(engine.calculate_net_claim(gross, len(event) / 6.0))
        scenarios[label] = {key: round(sum(item[key] for item in event_claims), 2) for key in event_claims[0]}
    return forecast, scenarios
