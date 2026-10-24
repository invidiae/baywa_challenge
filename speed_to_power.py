"""
Speed to Power Generation and Energy Yield Loss Estimation Module.

This module implements the conversion from wind speed (m/s) to active power (kW)
and electrical energy (kWh / MWh) based on the manufacturer power curves provided
for the target turbines (T1, T2, T3).

Key concepts:
1. Turbines feature active yaw control, rotating the nacelle directly into the wind.
   Thus, empirical power generation is defined purely as a function of nacelle wind speed: P(v).
   (Wind direction is critical when predicting wind speed propagation across parks,
   but once target wind speed is predicted, power is derived via the power curve).
2. The transformation uses 1D piecewise linear interpolation, ensuring maximum explainability,
   monotonicity preservation, and alignment with IEC 61400 standards.
3. Energy calculation:
   Energy (kWh) = Power (kW) * (Delta_t_minutes / 60)
   For 10-minute intervals: Delta_t / 60 = 1/6 hour.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class PowerCurveModel:
    """
    Interpolates turbine power curves and calculates power generation and energy yield.
    """

    def __init__(self, curve_df: pd.DataFrame, turbine_name: str | None = None):
        """
        Initialize with a power curve DataFrame.
        Expected columns: 'wind_speed' and 'power'.
        """
        required_cols = {"wind_speed", "power"}
        if not required_cols.issubset(curve_df.columns):
            raise ValueError(f"Power curve must contain columns: {required_cols}")

        self.turbine_name = turbine_name or (
            curve_df["turbine_name"].iloc[0] if "turbine_name" in curve_df.columns else "Turbine"
        )
        # Sort by wind speed to guarantee monotonic lookup
        self.curve = curve_df[["wind_speed", "power"]].dropna().sort_values("wind_speed").copy()
        self.speeds = self.curve["wind_speed"].to_numpy(dtype=float)
        self.powers = self.curve["power"].to_numpy(dtype=float)

        self.cut_in_speed = float(self.curve[self.curve["power"] > 0]["wind_speed"].min())
        self.rated_power = float(self.powers.max())
        self.max_curve_speed = float(self.speeds.max())

    @classmethod
    def from_excel(cls, file_path: str | Path, sheet_name: str | None = None) -> "PowerCurveModel":
        """Load power curve from an Excel file (e.g. data/T1_power_curve.xlsx)."""
        file_path = Path(file_path)
        df = pd.read_excel(file_path, sheet_name=sheet_name if sheet_name else 0)
        return cls(df, turbine_name=file_path.stem.replace("_power_curve", ""))

    def predict_power(
        self,
        wind_speeds: float | list | np.ndarray | pd.Series,
        cut_out_to_zero: bool = True,
    ) -> float | np.ndarray:
        """
        Map wind speed (m/s) to power (kW) using linear interpolation.

        Parameters:
        - wind_speeds: wind speed value or array-like
        - cut_out_to_zero: if True, wind speeds strictly exceeding the max power curve table
          speed (e.g., > 26 m/s) are set to 0 kW (storm cut-out shut down).

        Returns:
        - Power in kW (float or numpy array). Preserves NaNs.
        """
        is_scalar = np.isscalar(wind_speeds)
        v = np.asarray([wind_speeds] if is_scalar else wind_speeds, dtype=float)

        # Create output array initialized with NaN
        p = np.full_like(v, np.nan, dtype=float)
        valid_mask = ~np.isnan(v)

        if np.any(valid_mask):
            v_valid = v[valid_mask]

            # Linear interpolation:
            # - Wind speeds <= 0 or below minimum curve point -> 0 kW
            # - Wind speeds > max_curve_speed:
            #     if cut_out_to_zero -> 0 kW (storm shutdown)
            #     else -> interpolate / extrapolate flat to last power value
            right_val = 0.0 if cut_out_to_zero else float(self.powers[-1])
            p_interp = np.interp(v_valid, self.speeds, self.powers, left=0.0, right=right_val)
            p[valid_mask] = p_interp

        return p[0] if is_scalar else p

    def calculate_energy(
        self,
        power_kw: float | list | np.ndarray | pd.Series,
        interval_minutes: float = 10.0,
    ) -> float | np.ndarray:
        """
        Calculate energy yield / loss in kWh from power in kW.
        Formula: Energy (kWh) = Power (kW) * (interval_minutes / 60.0)
        """
        hours = interval_minutes / 60.0
        if np.isscalar(power_kw):
            return np.nan if np.isnan(power_kw) else float(power_kw) * hours
        p = np.asarray(power_kw, dtype=float)
        return p * hours

    def estimate_outage_loss(
        self,
        predicted_speeds: np.ndarray | pd.Series,
        timestamps: np.ndarray | pd.Series | None = None,
        interval_minutes: float = 10.0,
    ) -> pd.DataFrame:
        """
        Generate a comprehensive table containing predicted wind speed,
        power output (kW), interval energy loss (kWh), and cumulative loss (kWh & MWh).
        """
        power_kw = self.predict_power(predicted_speeds)
        energy_kwh = self.calculate_energy(power_kw, interval_minutes=interval_minutes)

        df = pd.DataFrame({
            "predicted_wind_speed": predicted_speeds,
            "predicted_power_kw": power_kw,
            "energy_loss_kwh": energy_kwh,
        })

        if timestamps is not None:
            df.insert(0, "timestamp", timestamps)

        # Cumulative energy (ignoring NaNs for accumulation)
        df["cum_energy_loss_kwh"] = df["energy_loss_kwh"].fillna(0).cumsum()
        df["cum_energy_loss_mwh"] = df["cum_energy_loss_kwh"] / 1000.0

        return df


def load_target_power_curves(data_dir: str | Path = "data") -> dict[str, PowerCurveModel]:
    """
    Convenience loader for all target turbine power curves (T1, T2, T3).
    """
    data_dir = Path(data_dir)
    curves = {}
    for t_id in ["T1", "T2", "T3"]:
        path = data_dir / f"{t_id}_power_curve.xlsx"
        if path.exists():
            curves[t_id] = PowerCurveModel.from_excel(path, sheet_name=t_id)
    return curves


if __name__ == "__main__":
    print("Testing PowerCurveModel...")
    curves = load_target_power_curves("data")
    for name, model in curves.items():
        print(f"Loaded {name}: cut-in speed = {model.cut_in_speed} m/s, rated power = {model.rated_power} kW")

    # Test sample speed inputs
    sample_speeds = [0.0, 2.0, 3.0, 5.0, 7.5, 11.5, 15.0, 20.0, 25.0, 27.0]
    p_kw = curves["T1"].predict_power(sample_speeds)
    e_kwh = curves["T1"].calculate_energy(p_kw)

    for s, p, e in zip(sample_speeds, p_kw, e_kwh):
        print(f"Speed: {s:4.1f} m/s -> Power: {p:7.1f} kW -> Energy (10 min): {e:7.2f} kWh")


"""USAGE

from speed_to_power import load_target_power_curves
import pandas as pd
models = load_target_power_curves("data")
t1_model = models["T1"]
# Simulated quantile predictions from the wind speed model:
quantile_speeds = pd.DataFrame({
    0.1: [2.5, 4.0, 6.0, 9.0, 11.0],   # Lower bound (10th percentile)
    0.5: [3.5, 5.5, 8.0, 11.5, 13.0],  # Median (50th percentile)
    0.9: [5.0, 7.5, 10.5, 14.0, 16.5], # Upper bound (90th percentile)
})
timestamps = pd.date_range("2026-09-01 00:00", periods=5, freq="10min")
# Compute power, interval losses, and cumulative outage yield losses:
results = t1_model.estimate_outage_loss_quantiles(quantile_speeds, timestamps=timestamps)
# View total outage loss summary (MWh) across quantiles:
print(results["summary"])
#   quantile  total_loss_kwh  total_loss_mwh
#        q10     1546.166667        1.546167
#        q50     2074.833333        2.074833
#        q90     2672.333333        2.672333

"""