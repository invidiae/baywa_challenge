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
3. Quantile / Probabilistic Support:
   Wind speed predictions with quantiles (e.g. q=0.1, q=0.5, q=0.9) are mapped directly through
   the power curve: P_q = P(v_q), preserving uncertainty intervals into power (kW) and energy loss (kWh/MWh).
4. IEC 61400-12-1 Air Density Normalization (Addon 2):
   If air pressure (hPa) and ambient temperature (°C) are provided, wind speeds are normalized
   prior to power curve lookup:
   rho = (P * 100) / (287.05 * (T + 273.15))
   v_norm = v * (rho / 1.225)^(1/3)
5. Energy calculation:
   Energy (kWh) = Power (kW) * (Delta_t_minutes / 60)
   For 10-minute intervals: Delta_t / 60 = 1/6 hour.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

R_SPECIFIC_AIR = 287.05  # J/(kg*K)
RHO_STANDARD = 1.225      # kg/m^3 (IEC standard air density at 15°C, 1013.25 hPa)


def compute_air_density(
    air_pressure_hpa: Union[float, Sequence, np.ndarray, pd.Series],
    temperature_c: Union[float, Sequence, np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """
    Compute atmospheric air density (kg/m^3) using the Ideal Gas Law:
    rho = (P_hpa * 100) / (287.05 * (T_c + 273.15))
    """
    is_scalar = np.isscalar(air_pressure_hpa) and np.isscalar(temperature_c)
    p_pa = np.asarray([air_pressure_hpa] if is_scalar else air_pressure_hpa, dtype=float) * 100.0
    t_k = np.asarray([temperature_c] if is_scalar else temperature_c, dtype=float) + 273.15

    rho = p_pa / (R_SPECIFIC_AIR * t_k)
    return float(rho[0]) if is_scalar else rho


class PowerCurveModel:
    """
    Interpolates turbine power curves and calculates power generation and energy yield,
    supporting both deterministic point predictions and probabilistic quantile forecasts,
    with optional IEC 61400-12-1 air density normalization.
    """

    def __init__(self, curve_df: pd.DataFrame, turbine_name: Optional[str] = None):
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
        self.curve = curve_df[["wind_speed", "power"]].dropna().sort_values("wind_speed").copy()
        self.speeds = self.curve["wind_speed"].to_numpy(dtype=float)
        self.powers = self.curve["power"].to_numpy(dtype=float)

        self.cut_in_speed = float(self.curve[self.curve["power"] > 0]["wind_speed"].min())
        self.rated_power = float(self.powers.max())
        self.max_curve_speed = float(self.speeds.max())

    @classmethod
    def from_excel(cls, file_path: Union[str, Path], sheet_name: Optional[str] = None) -> "PowerCurveModel":
        """Load power curve from an Excel file (e.g. data/T1_power_curve.xlsx)."""
        file_path = Path(file_path)
        df = pd.read_excel(file_path, sheet_name=sheet_name if sheet_name else 0)
        return cls(df, turbine_name=file_path.stem.replace("_power_curve", ""))

    def _normalize_speed_for_density(
        self,
        speeds: np.ndarray,
        air_density: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        air_pressure_hpa: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        temperature_c: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """
        Normalize wind speed according to IEC 61400-12-1:
        v_norm = v * (rho / rho_0)^(1/3)
        """
        rho = None
        if air_density is not None:
            rho = np.asarray(air_density, dtype=float)
        elif air_pressure_hpa is not None and temperature_c is not None:
            rho = compute_air_density(air_pressure_hpa, temperature_c)

        if rho is None:
            return speeds

        density_ratio = np.clip(rho / RHO_STANDARD, 0.5, 2.0)
        correction_factor = density_ratio ** (1.0 / 3.0)

        # Broadcast if correction_factor is 1D and speeds is 2D
        if correction_factor.ndim == 1 and speeds.ndim == 2:
            correction_factor = correction_factor[:, np.newaxis]

        return speeds * correction_factor

    def _interpolate_speeds(self, speeds: np.ndarray, cut_out_to_zero: bool = True) -> np.ndarray:
        """Internal helper to interpolate a numpy array of arbitrary shape."""
        orig_shape = speeds.shape
        flat_v = speeds.ravel()
        flat_p = np.full_like(flat_v, np.nan, dtype=float)

        valid_mask = ~np.isnan(flat_v)
        if np.any(valid_mask):
            v_valid = flat_v[valid_mask]
            right_val = 0.0 if cut_out_to_zero else float(self.powers[-1])
            flat_p[valid_mask] = np.interp(v_valid, self.speeds, self.powers, left=0.0, right=right_val)

        return flat_p.reshape(orig_shape)

    def predict_power(
        self,
        wind_speeds: Union[float, Sequence, np.ndarray, pd.Series, pd.DataFrame, Dict[Any, Any]],
        cut_out_to_zero: bool = True,
        air_density: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        air_pressure_hpa: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        temperature_c: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
    ) -> Union[float, np.ndarray, pd.Series, pd.DataFrame, Dict[Any, np.ndarray]]:
        """
        Map wind speed (m/s) to power (kW) using piecewise linear interpolation,
        with optional IEC 61400-12-1 air density normalization.
        """
        if isinstance(wind_speeds, pd.DataFrame):
            p_df = pd.DataFrame(index=wind_speeds.index)
            for col in wind_speeds.columns:
                arr = wind_speeds[col].to_numpy(dtype=float)
                arr_norm = self._normalize_speed_for_density(
                    arr, air_density=air_density, air_pressure_hpa=air_pressure_hpa, temperature_c=temperature_c
                )
                p_df[col] = self._interpolate_speeds(arr_norm, cut_out_to_zero=cut_out_to_zero)
            return p_df

        if isinstance(wind_speeds, dict):
            out = {}
            for q, val in wind_speeds.items():
                arr = np.asarray(val, dtype=float)
                arr_norm = self._normalize_speed_for_density(
                    arr, air_density=air_density, air_pressure_hpa=air_pressure_hpa, temperature_c=temperature_c
                )
                out[q] = self._interpolate_speeds(arr_norm, cut_out_to_zero=cut_out_to_zero)
            return out

        if isinstance(wind_speeds, pd.Series):
            arr = wind_speeds.to_numpy(dtype=float)
            arr_norm = self._normalize_speed_for_density(
                arr, air_density=air_density, air_pressure_hpa=air_pressure_hpa, temperature_c=temperature_c
            )
            interp = self._interpolate_speeds(arr_norm, cut_out_to_zero=cut_out_to_zero)
            return pd.Series(interp, index=wind_speeds.index, name="predicted_power_kw")

        is_scalar = np.isscalar(wind_speeds)
        v = np.asarray([wind_speeds] if is_scalar else wind_speeds, dtype=float)
        v_norm = self._normalize_speed_for_density(
            v, air_density=air_density, air_pressure_hpa=air_pressure_hpa, temperature_c=temperature_c
        )
        p = self._interpolate_speeds(v_norm, cut_out_to_zero=cut_out_to_zero)
        return float(p[0]) if is_scalar else p

    def calculate_energy(
        self,
        power_kw: Union[float, Sequence, np.ndarray, pd.Series, pd.DataFrame, Dict[Any, Any]],
        interval_minutes: float = 10.0,
    ) -> Union[float, np.ndarray, pd.Series, pd.DataFrame, Dict[Any, Any]]:
        """
        Calculate energy yield / loss in kWh from power in kW.
        Formula: Energy (kWh) = Power (kW) * (interval_minutes / 60.0)
        """
        hours = interval_minutes / 60.0

        if isinstance(power_kw, (pd.DataFrame, pd.Series)):
            return power_kw * hours

        if isinstance(power_kw, dict):
            return {k: (v * hours if isinstance(v, np.ndarray) else np.asarray(v) * hours) for k, v in power_kw.items()}

        if np.isscalar(power_kw):
            return np.nan if np.isnan(power_kw) else float(power_kw) * hours

        return np.asarray(power_kw, dtype=float) * hours

    def estimate_outage_loss(
        self,
        predicted_speeds: Union[np.ndarray, pd.Series],
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
        interval_minutes: float = 10.0,
        air_density: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        air_pressure_hpa: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        temperature_c: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Generate a table containing predicted wind speed, power output (kW),
        interval energy loss (kWh), and cumulative loss (kWh & MWh).
        """
        power_kw = self.predict_power(
            predicted_speeds,
            air_density=air_density,
            air_pressure_hpa=air_pressure_hpa,
            temperature_c=temperature_c,
        )
        energy_kwh = self.calculate_energy(power_kw, interval_minutes=interval_minutes)

        df = pd.DataFrame({
            "predicted_wind_speed": np.asarray(predicted_speeds),
            "predicted_power_kw": np.asarray(power_kw),
            "energy_loss_kwh": np.asarray(energy_kwh),
        })

        if timestamps is not None:
            df.insert(0, "timestamp", np.asarray(timestamps))

        df["cum_energy_loss_kwh"] = df["energy_loss_kwh"].fillna(0).cumsum()
        df["cum_energy_loss_mwh"] = df["cum_energy_loss_kwh"] / 1000.0
        return df

    def estimate_outage_loss_quantiles(
        self,
        quantile_speeds: Union[pd.DataFrame, Dict[Union[float, str], Sequence], np.ndarray],
        quantiles: Optional[Sequence[Union[float, str]]] = None,
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
        interval_minutes: float = 10.0,
        air_density: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        air_pressure_hpa: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
        temperature_c: Optional[Union[float, Sequence, np.ndarray, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate power generation and energy yield losses across multiple quantiles,
        supporting optional IEC 61400-12-1 air density normalization.
        """
        if isinstance(quantile_speeds, pd.DataFrame):
            df_speeds = quantile_speeds.copy()
        elif isinstance(quantile_speeds, dict):
            df_speeds = pd.DataFrame(quantile_speeds)
        elif isinstance(quantile_speeds, np.ndarray):
            if quantile_speeds.ndim == 1:
                col_name = quantiles[0] if quantiles else "q50"
                df_speeds = pd.DataFrame({col_name: quantile_speeds})
            elif quantile_speeds.ndim == 2:
                cols = quantiles if quantiles is not None else [f"q{i+1}" for i in range(quantile_speeds.shape[1])]
                df_speeds = pd.DataFrame(quantile_speeds, columns=cols)
            else:
                raise ValueError("quantile_speeds ndarray must be 1D or 2D")
        else:
            raise TypeError(f"Unsupported type for quantile_speeds: {type(quantile_speeds)}")

        col_map = {}
        for c in df_speeds.columns:
            if isinstance(c, (float, int)):
                col_map[c] = f"q{int(round(c * 100)):02d}" if 0 <= c <= 1 else f"q_{c}"
            else:
                col_map[c] = str(c)
        df_speeds = df_speeds.rename(columns=col_map)
        quantile_names = list(df_speeds.columns)

        # 1. Power (kW) per quantile
        df_power = pd.DataFrame(index=df_speeds.index)
        for q in quantile_names:
            arr = df_speeds[q].to_numpy(dtype=float)
            arr_norm = self._normalize_speed_for_density(
                arr, air_density=air_density, air_pressure_hpa=air_pressure_hpa, temperature_c=temperature_c
            )
            df_power[f"power_kw_{q}"] = self._interpolate_speeds(arr_norm)

        # 2. Interval energy loss (kWh)
        hours = interval_minutes / 60.0
        df_loss_kwh = pd.DataFrame(index=df_speeds.index)
        for q in quantile_names:
            df_loss_kwh[f"energy_kwh_{q}"] = df_power[f"power_kw_{q}"] * hours

        # 3. Cumulative loss (kWh and MWh)
        df_cum_kwh = pd.DataFrame(index=df_speeds.index)
        df_cum_mwh = pd.DataFrame(index=df_speeds.index)
        for q in quantile_names:
            df_cum_kwh[f"cum_loss_kwh_{q}"] = df_loss_kwh[f"energy_kwh_{q}"].fillna(0).cumsum()
            df_cum_mwh[f"cum_loss_mwh_{q}"] = df_cum_kwh[f"cum_loss_kwh_{q}"] / 1000.0

        # 4. Summary table
        summary_rows = []
        for q in quantile_names:
            total_kwh = float(df_cum_kwh[f"cum_loss_kwh_{q}"].iloc[-1]) if len(df_cum_kwh) > 0 else 0.0
            total_mwh = float(df_cum_mwh[f"cum_loss_mwh_{q}"].iloc[-1]) if len(df_cum_mwh) > 0 else 0.0
            summary_rows.append({
                "quantile": q,
                "total_loss_kwh": total_kwh,
                "total_loss_mwh": total_mwh,
            })
        df_summary = pd.DataFrame(summary_rows)

        # 5. Combined tidy DataFrame
        combined = pd.DataFrame(index=df_speeds.index)
        if timestamps is not None:
            combined["timestamp"] = np.asarray(timestamps)

        for q in quantile_names:
            combined[f"wind_speed_{q}"] = df_speeds[q].values
            combined[f"power_kw_{q}"] = df_power[f"power_kw_{q}"].values
            combined[f"energy_kwh_{q}"] = df_loss_kwh[f"energy_kwh_{q}"].values
            combined[f"cum_mwh_{q}"] = df_cum_mwh[f"cum_loss_mwh_{q}"].values

        return {
            "power_kw": df_power,
            "interval_loss_kwh": df_loss_kwh,
            "cum_loss_kwh": df_cum_kwh,
            "cum_loss_mwh": df_cum_mwh,
            "summary": df_summary,
            "combined_df": combined,
        }


def load_target_power_curves(data_dir: Union[str, Path] = "data") -> Dict[str, PowerCurveModel]:
    """Convenience loader for all target turbine power curves (T1, T2, T3)."""
    data_dir = Path(data_dir)
    curves = {}
    for t_id in ["T1", "T2", "T3"]:
        path = data_dir / f"{t_id}_power_curve.xlsx"
        if path.exists():
            curves[t_id] = PowerCurveModel.from_excel(path, sheet_name=t_id)
    return curves


if __name__ == "__main__":
    print("Testing PowerCurveModel with IEC 61400-12-1 Air Density Correction...")
    curves = load_target_power_curves("data")
    t1 = curves["T1"]

    sample_speeds = [3.0, 5.0, 7.5, 11.5, 15.0]
    p_std = t1.predict_power(sample_speeds)
    
    # Warm summer air: 25°C, 995 hPa -> rho ≈ 1.162 kg/m^3 (thin air)
    p_warm = t1.predict_power(sample_speeds, air_pressure_hpa=995.0, temperature_c=25.0)
    
    # Cold winter air: 0°C, 1020 hPa -> rho ≈ 1.300 kg/m^3 (dense air)
    p_cold = t1.predict_power(sample_speeds, air_pressure_hpa=1020.0, temperature_c=0.0)

    for s, ps, pw, pc in zip(sample_speeds, p_std, p_warm, p_cold):
        print(f"Speed: {s:4.1f} m/s | Standard: {ps:6.1f} kW | Warm (Thin): {pw:6.1f} kW | Cold (Dense): {pc:6.1f} kW")
