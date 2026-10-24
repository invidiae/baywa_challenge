"""
BayWa r.e. Wind Turbine Outage Energy Loss Estimator
====================================================
Interactive Streamlit application powered by:
1. Manual Excel / CSV upload workflow for Target, Reference(s), and Power Curve.
2. Explainable Bayesian Ridge Regression with always-on directional vectors (u, v),
   posterior 95% coefficient confidence intervals, and full Bayesian predictive bands.
3. IEC 61400-12-1 Air Density Normalization from reference pressure & temperature.
4. Synchronized Manual Calendar/Time Entry + Range Slider for the outage period.
5. Dedicated 4th Tab for 10-minute interval predictions with interactive symmetric
   percentile certainty slider (50%–99%) and MWh-standardized energy accounting.
"""

from datetime import datetime, timedelta
import hashlib
import io
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bayesian_wind_model import (
    BayesianWindSpeedPredictor,
    build_features_from_frames,
    compute_air_density_from_frames,
)
from speed_to_power import PowerCurveModel


# ---------------- Cached Upload Parsers & Helpers ----------------
def _normalize_col_name(col: str) -> str:
    c = str(col).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    return c


@st.cache_data(show_spinner=False)
def parse_scada_bytes(filename: str, raw_bytes: bytes) -> Tuple[str, pd.DataFrame]:
    """Parse an uploaded Excel or CSV SCADA file into a standardized timestamp-indexed DataFrame."""
    suffix = Path(filename).suffix.lower()
    buf = io.BytesIO(raw_bytes)
    if suffix in {".xlsx", ".xls"}:
        df_raw = pd.read_excel(buf)
    else:
        df_raw = pd.read_csv(buf, sep=None, engine="python")

    if df_raw.empty:
        raise ValueError(f"Uploaded file '{filename}' is empty.")

    col_map = {_normalize_col_name(c): c for c in df_raw.columns}

    ts_col = None
    for candidate in ("timestamp", "time", "datetime", "date_time", "date"):
        if candidate in col_map:
            ts_col = col_map[candidate]
            break
    if ts_col is None:
        ts_col = df_raw.columns[0]

    ws_col = None
    for candidate in ("wind_speed", "wind_speed_m_s", "windspeed", "ws", "speed"):
        if candidate in col_map:
            ws_col = col_map[candidate]
            break
    if ws_col is None:
        for norm_k, orig_k in col_map.items():
            if "wind_speed" in norm_k and "std" not in norm_k:
                ws_col = orig_k
                break
    if ws_col is None and len(df_raw.columns) >= 2:
        ws_col = df_raw.columns[1]
    if ws_col is None:
        raise ValueError(f"Could not find a wind_speed column in '{filename}'.")

    def _to_float(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df_raw[ts_col], errors="coerce")
    if out["timestamp"].dt.tz is not None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    out["wind_speed"] = _to_float(df_raw[ws_col])

    for std_name, aliases in [
        ("wind_direction", ("wind_direction", "direction", "wind_dir", "wd")),
        ("ambient_temperature", ("ambient_temperature", "temperature", "temp", "air_temperature")),
        ("air_pressure", ("air_pressure", "pressure", "barometric_pressure", "pres")),
        ("wind_speed_stddev", ("wind_speed_stddev", "wind_speed_std", "ws_std")),
    ]:
        for alias in aliases:
            if alias in col_map:
                out[std_name] = _to_float(df_raw[col_map[alias]])
                break

    if "test_set" in col_map:
        raw_ts = df_raw[col_map["test_set"]]
        if pd.api.types.is_bool_dtype(raw_ts):
            out["test_set"] = raw_ts.fillna(False).astype(bool)
        else:
            out["test_set"] = raw_ts.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "test"})
    else:
        out["test_set"] = False

    turbine_id = None
    if "turbine_name" in col_map:
        vals = df_raw[col_map["turbine_name"]].dropna().astype(str).str.strip()
        if len(vals) > 0:
            turbine_id = vals.iloc[0]
    if not turbine_id:
        stem = Path(filename).stem
        turbine_id = re.sub(r"_?data$", "", stem, flags=re.IGNORECASE).strip() or stem

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    return turbine_id, out


@st.cache_data(show_spinner=False)
def parse_power_curve_bytes(filename: str, raw_bytes: bytes, preferred_sheet: Optional[str] = None) -> PowerCurveModel:
    """Parse an uploaded Excel or CSV power curve into a PowerCurveModel."""
    suffix = Path(filename).suffix.lower()
    buf = io.BytesIO(raw_bytes)
    if suffix in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(buf)
        sheet = preferred_sheet if (preferred_sheet and preferred_sheet in xl.sheet_names) else xl.sheet_names[0]
        df_raw = xl.parse(sheet_name=sheet)
    else:
        df_raw = pd.read_csv(buf, sep=None, engine="python")

    col_map = {_normalize_col_name(c): c for c in df_raw.columns}
    ws_col = col_map.get("wind_speed") or col_map.get("wind_speed_m_s") or df_raw.columns[0]
    pw_col = col_map.get("power") or col_map.get("power_kw") or (df_raw.columns[1] if len(df_raw.columns) > 1 else None)
    if pw_col is None:
        raise ValueError(f"Power curve file '{filename}' must contain wind_speed and power columns.")

    def _col_num(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    curve_df = pd.DataFrame(
        {
            "wind_speed": _col_num(df_raw[ws_col]),
            "power": _col_num(df_raw[pw_col]),
        }
    ).dropna()

    t_name = preferred_sheet or Path(filename).stem.replace("_power_curve", "")
    return PowerCurveModel(curve_df, turbine_name=t_name)


def find_longest_nan_run(series: pd.Series) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    missing = series.isna()
    if not missing.any():
        return None
    grp = (missing != missing.shift()).cumsum()
    runs = [(g.index[0], g.index[-1]) for _, g in missing[missing].groupby(grp[missing])]
    return max(runs, key=lambda r: r[1] - r[0])


def clamp_timestamp(ts: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Timestamp:
    return max(lo, min(hi, ts))


# ---------------- Main Streamlit App ----------------
def render_app() -> None:
    st.set_page_config(
        page_title="Wind Outage Loss Estimator | BayWa r.e.",
        page_icon="🌬️",
        layout="wide",
    )

    st.markdown(
        """
<style>
[data-testid="stAppDeployButton"],
.stAppDeployButton,
.stDeployButton,
[data-testid="stToolbarActions"] {
    display: none !important;
    visibility: hidden !important;
}
.hero {
    background: linear-gradient(90deg, #0b3d91, #1a73e8);
    padding: 22px 28px;
    border-radius: 12px;
    color: white;
    margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.0rem; color: white; }
.hero p { margin: 6px 0 0 0; opacity: 0.92; font-size: 1.02rem; }
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 12px 16px;
    border-radius: 10px;
}
div[data-testid="stMetric"] * { color: #0f172a !important; }
</style>
<div class="hero">
  <h1>🌬️ Wind Farm Outage: Energy Loss Estimator</h1>
  <p>Bayesian Ridge Wind Transfer with Coefficient Confidence Intervals & IEC 61400-12-1 Normalization</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---------------- Sidebar: 1. File Uploads ----------------
    st.sidebar.header("1. Upload SCADA & Curve Files")
    st.sidebar.caption("Upload Excel (.xlsx/.xls) or CSV files for Target (e.g. T1–T3), References (e.g. O1–O2, H1–H5), and Power Curve.")

    target_upload = st.sidebar.file_uploader(
        "Target Turbine SCADA (e.g. T1–T3)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
        key="uploader_target",
    )
    reference_uploads = st.sidebar.file_uploader(
        "Reference Turbine SCADA (e.g. O1–O2, H1–H5)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="uploader_refs",
    )
    curve_upload = st.sidebar.file_uploader(
        "Target Turbine Power Curve",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
        key="uploader_curve",
    )

    if not target_upload or not reference_uploads or not curve_upload:
        st.info(
            "### 👋 Upload Required Files in the Sidebar to Begin\n"
            "Please upload all three inputs on the left:\n"
            f"- {'✅' if target_upload else '⏳'} **Target Turbine SCADA** — e.g. `T1_data.xlsx`, `T2_data.xlsx`, or `T3_data.xlsx`\n"
            f"- {'✅' if reference_uploads else '⏳'} **Reference Turbine SCADA (one or more files)** — e.g. `O1_data.xlsx`, `O2_data.xlsx`, `H1_data.xlsx` … `H5_data.xlsx`\n"
            f"- {'✅' if curve_upload else '⏳'} **Target Turbine Power Curve** — e.g. `T1_power_curve.xlsx`"
        )
        st.stop()
        return

    try:
        target_name, target_raw = parse_scada_bytes(target_upload.name, target_upload.getvalue())
        full_idx = pd.date_range(target_raw.index.min(), target_raw.index.max(), freq="10min")
        target_df = target_raw.reindex(full_idx)
        target_df["test_set"] = target_df["test_set"].fillna(False).astype(bool)
        target_df.index.name = "timestamp"

        raw_refs: Dict[str, pd.DataFrame] = {}
        skipped_same_as_target: List[str] = []
        for up in reference_uploads:
            r_name, r_df = parse_scada_bytes(up.name, up.getvalue())
            if r_name.upper() == target_name.upper():
                skipped_same_as_target.append(r_name)
                continue
            key_name = r_name if r_name not in raw_refs else f"{r_name}_{len(raw_refs)+1}"
            raw_refs[key_name] = r_df

        if skipped_same_as_target:
            st.sidebar.warning(f"Excluded `{', '.join(skipped_same_as_target)}` from references because it matches Target `{target_name}`.")
        if not raw_refs:
            st.error("At least one Reference Turbine file distinct from the Target Turbine is required.")
            st.stop()
            return

        power_curve = parse_power_curve_bytes(curve_upload.name, curve_upload.getvalue(), preferred_sheet=target_name)
    except Exception as exc:
        st.error(f"Error reading uploaded files: {exc}")
        st.stop()
        return

    # ---------------- Sidebar: 2. Outage Period (Manual Calendar + Slider) ----------------
    st.sidebar.header("2. Outage Period")

    nan_in_test = target_df.loc[target_df["test_set"], "wind_speed"]
    auto_outage = find_longest_nan_run(nan_in_test) or find_longest_nan_run(target_df["wind_speed"])

    min_ts = target_df.index.min()
    max_ts = target_df.index.max()

    if auto_outage:
        default_start, default_end = auto_outage[0], auto_outage[1]
        st.sidebar.success(
            f"Auto-detected missing data:\n**{default_start:%Y-%m-%d %H:%M}** → **{default_end:%Y-%m-%d %H:%M}**"
        )
    else:
        default_end = max_ts
        default_start = max(min_ts, max_ts - timedelta(days=3))
        st.sidebar.warning("No NaN gap detected in target file; defaulting to final 3 days.")

    sig = hashlib.md5(f"{target_upload.name}_{len(target_df)}_{min_ts}_{max_ts}".encode()).hexdigest()[:8]
    k_win = f"win_{sig}"
    k_sd, k_st = f"sd_{sig}", f"st_{sig}"
    k_ed, k_et = f"ed_{sig}", f"et_{sig}"
    k_sl = f"sl_{sig}"

    if k_win not in st.session_state:
        st.session_state[k_win] = (default_start.to_pydatetime(), default_end.to_pydatetime())
        st.session_state[k_sd] = default_start.date()
        st.session_state[k_st] = default_start.time().replace(second=0, microsecond=0)
        st.session_state[k_ed] = default_end.date()
        st.session_state[k_et] = default_end.time().replace(second=0, microsecond=0)
        st.session_state[k_sl] = (default_start.to_pydatetime(), default_end.to_pydatetime())

    def _sync_from_calendar() -> None:
        s_dt = clamp_timestamp(
            pd.Timestamp(datetime.combine(st.session_state[k_sd], st.session_state[k_st])),
            min_ts,
            max_ts,
        ).floor("10min")
        e_dt = clamp_timestamp(
            pd.Timestamp(datetime.combine(st.session_state[k_ed], st.session_state[k_et])),
            min_ts,
            max_ts,
        ).floor("10min")
        if e_dt < s_dt:
            e_dt = s_dt
        pair = (s_dt.to_pydatetime(), e_dt.to_pydatetime())
        st.session_state[k_win] = pair
        st.session_state[k_sl] = pair
        st.session_state[k_sd], st.session_state[k_st] = s_dt.date(), s_dt.time()
        st.session_state[k_ed], st.session_state[k_et] = e_dt.date(), e_dt.time()

    def _sync_from_slider() -> None:
        s_py, e_py = st.session_state[k_sl]
        s_dt = pd.Timestamp(s_py).floor("10min")
        e_dt = pd.Timestamp(e_py).floor("10min")
        st.session_state[k_win] = (s_dt.to_pydatetime(), e_dt.to_pydatetime())
        st.session_state[k_sd], st.session_state[k_st] = s_dt.date(), s_dt.time()
        st.session_state[k_ed], st.session_state[k_et] = e_dt.date(), e_dt.time()

    def _reset_to_auto() -> None:
        pair = (default_start.to_pydatetime(), default_end.to_pydatetime())
        st.session_state[k_win] = pair
        st.session_state[k_sl] = pair
        st.session_state[k_sd] = default_start.date()
        st.session_state[k_st] = default_start.time().replace(second=0, microsecond=0)
        st.session_state[k_ed] = default_end.date()
        st.session_state[k_et] = default_end.time().replace(second=0, microsecond=0)

    st.sidebar.markdown("**Manual Calendar & Time Entry**")
    c_sd, c_st = st.sidebar.columns([1.25, 1.0])
    c_sd.date_input("Start Date", min_value=min_ts.date(), max_value=max_ts.date(), key=k_sd, on_change=_sync_from_calendar)
    c_st.time_input("Start Time", step=timedelta(minutes=10), key=k_st, on_change=_sync_from_calendar)

    c_ed, c_et = st.sidebar.columns([1.25, 1.0])
    c_ed.date_input("End Date", min_value=min_ts.date(), max_value=max_ts.date(), key=k_ed, on_change=_sync_from_calendar)
    c_et.time_input("End Time", step=timedelta(minutes=10), key=k_et, on_change=_sync_from_calendar)

    st.sidebar.slider(
        "Or Drag Outage Window",
        min_value=min_ts.to_pydatetime(),
        max_value=max_ts.to_pydatetime(),
        step=timedelta(minutes=10),
        format="DD MMM HH:mm",
        key=k_sl,
        on_change=_sync_from_slider,
    )

    if auto_outage:
        st.sidebar.button("↺ Reset to Auto-Detected Outage", on_click=_reset_to_auto, width="stretch")

    outage_start = pd.Timestamp(st.session_state[k_win][0])
    outage_end = pd.Timestamp(st.session_state[k_win][1])
    outage_mask = (target_df.index >= outage_start) & (target_df.index <= outage_end)
    outage_idx = target_df.index[outage_mask]

    # ---------------- Filter Offline References & Build Physics Features ----------------
    active_refs: Dict[str, pd.DataFrame] = {}
    offline_refs: List[str] = []
    for r_name, r_df in raw_refs.items():
        r_out = r_df["wind_speed"].reindex(outage_idx)
        if len(outage_idx) > 0 and r_out.notna().mean() < 0.50:
            offline_refs.append(r_name)
        else:
            active_refs[r_name] = r_df

    if offline_refs:
        st.sidebar.warning(
            f"Excluded `{', '.join(offline_refs)}` because >50% of its wind speeds are missing during the selected outage window."
        )
    if not active_refs:
        st.error("None of the uploaded reference turbines have valid telemetry during the selected outage period.")
        st.stop()
        return

    features_all = build_features_from_frames(active_refs, include_direction=True)
    features_aligned = features_all.reindex(target_df.index).interpolate(method="linear").bfill().ffill()
    feature_cols = list(features_aligned.columns)

    rho_ref_series = compute_air_density_from_frames(active_refs)

    # ---------------- Sidebar: 3. Physics Options ----------------
    st.sidebar.header("3. Physics Options")
    st.sidebar.caption("Orthogonal directional vectors `(u, v)` are permanently enabled.")
    if rho_ref_series is not None:
        use_density = st.sidebar.toggle(
            "IEC 61400-12-1 Air Density Normalization",
            value=True,
            help="Normalizes predicted wind speeds via v_norm = v * (rho / 1.225)^(1/3) using barometric pressure and ambient temperature from uploaded references.",
        )
    else:
        use_density = False
        st.sidebar.caption("ℹ️ IEC Air Density Normalization unavailable (uploaded reference files do not contain variable pressure & temperature columns).")

    # ---------------- Model Training & Outage Inference ----------------
    non_outage_observed = (~outage_mask) & target_df["wind_speed"].notna() & features_aligned[feature_cols].notna().all(axis=1)
    flagged_test_observed = (~outage_mask) & target_df["test_set"] & target_df["wind_speed"].notna()

    if flagged_test_observed.sum() >= 12 and (non_outage_observed & (~target_df["test_set"])).sum() >= 50:
        train_mask = non_outage_observed & (~target_df["test_set"])
        val_mask = flagged_test_observed
        val_description = (
            f"Unseen `test_set = True` validation rows outside the outage "
            f"({target_df.index[val_mask].min():%Y-%m-%d %H:%M} → {target_df.index[val_mask].max():%Y-%m-%d %H:%M}, "
            f"{int(val_mask.sum())} intervals)"
        )
    else:
        obs_indices = target_df.index[non_outage_observed]
        if len(obs_indices) < 60:
            st.error("Not enough observed historical rows outside the outage window to train and validate the model (need ≥ 60 intervals).")
            st.stop()
            return
        split_pos = int(len(obs_indices) * 0.80)
        train_idx_set = obs_indices[:split_pos]
        val_idx_set = obs_indices[split_pos:]
        train_mask = target_df.index.isin(train_idx_set)
        val_mask = target_df.index.isin(val_idx_set)
        val_description = (
            f"20% chronological hold-out validation window "
            f"({val_idx_set.min():%Y-%m-%d %H:%M} → {val_idx_set.max():%Y-%m-%d %H:%M}, {len(val_idx_set)} intervals)"
        )

    predictor = BayesianWindSpeedPredictor(target_turbine=target_name)
    predictor.fit(features_aligned.loc[train_mask, feature_cols], target_df.loc[train_mask, "wind_speed"])

    outage_feats = features_aligned.loc[outage_idx, feature_cols]
    outage_band_80 = predictor.predict_certainty_band(outage_feats, certainty_pct=80.0, coef_ci_pct=95.0)

    p50_speed_outage = outage_band_80["median"].to_numpy(dtype=float)
    p10_speed_outage = outage_band_80["lower"].to_numpy(dtype=float)
    p90_speed_outage = outage_band_80["upper"].to_numpy(dtype=float)
    coef_lo_speed_outage = outage_band_80["coef_ci_lower"].to_numpy(dtype=float)
    coef_hi_speed_outage = outage_band_80["coef_ci_upper"].to_numpy(dtype=float)

    outage_density = None
    if use_density and rho_ref_series is not None:
        outage_density = rho_ref_series.reindex(outage_idx).interpolate(method="linear").bfill().ffill().to_numpy(dtype=float)

    pw_p50_kw = power_curve.predict_power(p50_speed_outage, air_density=outage_density)
    pw_p10_kw = power_curve.predict_power(p10_speed_outage, air_density=outage_density)
    pw_p90_kw = power_curve.predict_power(p90_speed_outage, air_density=outage_density)
    pw_coef_lo_kw = power_curve.predict_power(coef_lo_speed_outage, air_density=outage_density)
    pw_coef_hi_kw = power_curve.predict_power(coef_hi_speed_outage, air_density=outage_density)

    loss_p50_mwh = float(pw_p50_kw.sum() / 6000.0)
    loss_p10_mwh = float(pw_p10_kw.sum() / 6000.0)
    loss_p90_mwh = float(pw_p90_kw.sum() / 6000.0)
    loss_coef_lo_mwh = float(pw_coef_lo_kw.sum() / 6000.0)
    loss_coef_hi_mwh = float(pw_coef_hi_kw.sum() / 6000.0)

    # ---------------- Validation Evaluation ----------------
    val_feats = features_aligned.loc[val_mask, feature_cols]
    val_band_80 = predictor.predict_certainty_band(val_feats, certainty_pct=80.0, coef_ci_pct=95.0)
    val_act_ws = target_df.loc[val_mask, "wind_speed"].to_numpy(dtype=float)
    val_pred_ws = val_band_80["median"].to_numpy(dtype=float)

    val_density_arr = None
    if use_density and rho_ref_series is not None:
        val_density_arr = rho_ref_series.reindex(target_df.index[val_mask]).interpolate(method="linear").bfill().ffill().to_numpy(dtype=float)

    val_act_pw = power_curve.predict_power(val_act_ws)
    val_pred_pw = power_curve.predict_power(val_pred_ws, air_density=val_density_arr)

    bt_ws_mae = float(mean_absolute_error(val_act_ws, val_pred_ws))
    bt_ws_rmse = float(np.sqrt(mean_squared_error(val_act_ws, val_pred_ws)))
    bt_ws_r2 = float(r2_score(val_act_ws, val_pred_ws))
    bt_pw_mae = float(mean_absolute_error(val_act_pw, val_pred_pw))
    bt_pw_rmse = float(np.sqrt(mean_squared_error(val_act_pw, val_pred_pw)))
    bt_act_mwh = float(val_act_pw.sum() / 6000.0)
    bt_pred_mwh = float(val_pred_pw.sum() / 6000.0)
    bt_dev_mwh = bt_pred_mwh - bt_act_mwh
    bt_dev_pct = float(100.0 * bt_dev_mwh / bt_act_mwh) if bt_act_mwh > 0 else float("nan")
    bt_cov_pct = float(100.0 * np.mean((val_act_ws >= val_band_80["lower"]) & (val_act_ws <= val_band_80["upper"])))

    corr_rows = []
    tgt_train_ws = target_df.loc[train_mask, "wind_speed"]
    for r_name, r_df in active_refs.items():
        r_ws = r_df["wind_speed"].reindex(tgt_train_ws.index)
        ok = tgt_train_ws.notna() & r_ws.notna()
        corr_val = float(tgt_train_ws[ok].corr(r_ws[ok])) if ok.sum() > 10 else float("nan")
        corr_rows.append({"Reference Turbine": r_name, "Training Overlap (rows)": int(ok.sum()), "Pearson Correlation (r)": round(corr_val, 3)})
    ranking_df = pd.DataFrame(corr_rows).sort_values("Pearson Correlation (r)", ascending=False).reset_index(drop=True)

    # ---------------- 4-Tab Interface ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 1. Wind Speed & Uncertainty",
        "🔬 2. Prediction Method & Coefficient CIs",
        "⚡ 3. Energy Loss (MWh) & Claim",
        "📋 4. 10-Min Interval Predictions",
    ])

    # ==================== TAB 1 ====================
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Target / Outage Duration", f"{target_name} · {len(outage_idx) / 6.0:.1f} h", help=f"{len(outage_idx)} 10-minute intervals.")
        col2.metric("Mean Predicted Wind (P50)", f"{p50_speed_outage.mean():.2f} m/s")
        col3.metric("Mean Coeff 95% CI Half-Width", f"±{outage_band_80['coef_ci_halfwidth'].mean():.3f} m/s", help="Uncertainty of the mean regression line propagated from posterior coefficient covariance Σ_w.")
        col4.metric("Validation Wind RMSE", f"{bt_ws_rmse:.3f} m/s", help="Evaluated on unseen validation rows.")

        pad = timedelta(days=2)
        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(x=outage_idx, y=p90_speed_outage, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig1.add_trace(
            go.Scatter(
                x=outage_idx,
                y=p10_speed_outage,
                fill="tonexty",
                fillcolor="rgba(37, 99, 235, 0.14)",
                line=dict(width=0),
                name="80% Predictive Band (P10–P90: Coeff + Gusts)",
            )
        )

        fig1.add_trace(go.Scatter(x=outage_idx, y=coef_hi_speed_outage, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig1.add_trace(
            go.Scatter(
                x=outage_idx,
                y=coef_lo_speed_outage,
                fill="tonexty",
                fillcolor="rgba(29, 78, 216, 0.42)",
                line=dict(width=0),
                name="95% Mean Curve CI (Coefficient Uncertainty xᵀΣ_wx)",
            )
        )

        speed_feat_cols = [c for c in feature_cols if c.startswith("speed_")]
        palette = ["#94a3b8", "#cbd5e1", "#64748b", "#a1a1aa"]
        for idx_c, s_col in enumerate(speed_feat_cols):
            fig1.add_trace(
                go.Scatter(
                    x=target_df.index,
                    y=features_aligned[s_col],
                    name=f"Ref {s_col.replace('speed_', 'Park ')}",
                    line=dict(color=palette[idx_c % len(palette)], width=1.1, dash="dot"),
                )
            )

        fig1.add_trace(
            go.Scatter(
                x=target_df.index,
                y=target_df["wind_speed"],
                name=f"{target_name} Measured",
                line=dict(color="#0f172a", width=1.4),
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=outage_idx,
                y=p50_speed_outage,
                name=f"{target_name} Predicted (P50)",
                line=dict(color="#2563eb", width=2.4),
            )
        )

        fig1.add_vrect(
            x0=outage_start,
            x1=outage_end,
            fillcolor="red",
            opacity=0.08,
            line_width=0,
            annotation_text="Outage Window",
            annotation_position="top left",
        )

        fig1.update_xaxes(range=[outage_start - pad, outage_end + pad], title="Timestamp")
        fig1.update_yaxes(title="Wind Speed (m/s)")
        fig1.update_layout(height=490, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig1, width="stretch")

    # ==================== TAB 2 ====================
    with tab2:
        left_c, right_c = st.columns([1.18, 1.0])

        coef_summary = predictor.get_coefficient_summary(ci_level=0.95)

        with left_c:
            st.subheader("Closed-Form Bayesian Ridge Equation")
            st.markdown(
                "Directional vectors (`u = v·sin(θ)`, `v = v·cos(θ)`) are permanently active. "
                "Posterior weights `w` and their full covariance matrix `Σ_w` are learned analytically via Evidence Maximization:"
            )
            st.code(predictor.get_formula(), language="text")

            st.subheader("Posterior Coefficients & 95% Confidence Intervals")
            st.caption(
                f"Trained on {predictor.n_train_:,} historical 10-minute intervals "
                f"(learned noise precision α = {predictor.model.alpha_:.3f}, weight regularization λ = {predictor.model.lambda_:.2f})."
            )
            disp_coef = coef_summary.copy()
            disp_coef["95% Confidence Interval"] = disp_coef.apply(
                lambda r: f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]", axis=1
            )
            disp_coef = disp_coef.rename(
                columns={
                    "parameter": "Parameter",
                    "coefficient": "Weight (w_i)",
                    "std_error": "Std. Error (SE)",
                    "abs_importance_pct": "Slope Share (%)",
                }
            )[["Parameter", "Weight (w_i)", "Std. Error (SE)", "95% Confidence Interval", "Slope Share (%)"]]
            st.dataframe(
                disp_coef.style.format(
                    {
                        "Weight (w_i)": "{:+.4f}",
                        "Std. Error (SE)": "{:.4f}",
                        "Slope Share (%)": lambda v: "—" if pd.isna(v) else f"{v:.1f}%",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

            slopes_only = coef_summary.iloc[1:].copy()
            fig_coef = go.Figure()
            fig_coef.add_vline(x=0.0, line_dash="dash", line_color="#94a3b8")
            fig_coef.add_trace(
                go.Scatter(
                    x=slopes_only["coefficient"],
                    y=slopes_only["parameter"],
                    mode="markers",
                    marker=dict(size=9, color="#1d4ed8"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(slopes_only["ci_upper"] - slopes_only["coefficient"]).to_numpy(),
                        arrayminus=(slopes_only["coefficient"] - slopes_only["ci_lower"]).to_numpy(),
                        color="#1e40af",
                        thickness=2,
                        width=6,
                    ),
                    name="Weight ± 95% CI",
                )
            )
            fig_coef.update_layout(
                height=280,
                title="Slope Coefficients with Posterior 95% Confidence Intervals",
                xaxis_title="Coefficient Value",
                yaxis_title="Feature",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_coef, width="stretch")

        with right_c:
            st.subheader("Validation Scatter Check")
            lim = float(max(val_act_ws.max(), val_pred_ws.max())) + 1.0
            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scatter(
                    x=val_act_ws,
                    y=val_pred_ws,
                    mode="markers",
                    marker=dict(size=5, color="#2563eb", opacity=0.45),
                    name="Validation Intervals",
                )
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[0, lim],
                    y=[0, lim],
                    mode="lines",
                    line=dict(color="#ef4444", dash="dash"),
                    name="Perfect 1:1 Line",
                )
            )
            fig_scatter.update_layout(
                height=340,
                xaxis_title=f"Measured {target_name} Wind Speed (m/s)",
                yaxis_title=f"Predicted {target_name} Wind Speed (m/s)",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_scatter, width="stretch")
            st.caption(f"Scored on: {val_description}")

            st.subheader("Uploaded Reference Turbine Correlations")
            st.dataframe(ranking_df, hide_index=True, width="stretch")

    # ==================== TAB 3 ====================
    with tab3:
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric(
            "Estimated Outage Loss (P50)",
            f"{loss_p50_mwh:,.2f} MWh",
            help="Central counterfactual energy lost during the selected outage window.",
        )
        col_e2.metric(
            "95% Coefficient CI Loss",
            f"{loss_coef_lo_mwh:,.2f} – {loss_coef_hi_mwh:,.2f} MWh",
            help="Energy loss confidence interval propagated purely from the 95% confidence intervals of the regression coefficients.",
        )
        col_e3.metric(
            "80% Predictive Band (P10–P90)",
            f"{loss_p10_mwh:,.2f} – {loss_p90_mwh:,.2f} MWh",
            help="Full Bayesian predictive interval combining coefficient uncertainty and 10-minute wind turbulence.",
        )
        col_e4.metric(
            "Mean Lost Power (P50)",
            f"{pw_p50_kw.mean():,.1f} kW",
            help="Average counterfactual active power across the outage window.",
        )

        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader("Counterfactual Lost Power (kW)")
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=outage_idx, y=pw_p90_kw, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_p.add_trace(
                go.Scatter(
                    x=outage_idx,
                    y=pw_p10_kw,
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.14)",
                    line=dict(width=0),
                    name="80% Predictive Band (P10–P90)",
                )
            )
            fig_p.add_trace(go.Scatter(x=outage_idx, y=pw_coef_hi_kw, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_p.add_trace(
                go.Scatter(
                    x=outage_idx,
                    y=pw_coef_lo_kw,
                    fill="tonexty",
                    fillcolor="rgba(29, 78, 216, 0.42)",
                    line=dict(width=0),
                    name="95% Coefficient CI Band",
                )
            )
            fig_p.add_trace(
                go.Scatter(
                    x=outage_idx,
                    y=pw_p50_kw,
                    name="P50 Lost Power (kW)",
                    line=dict(color="#2563eb", width=2.0),
                )
            )
            fig_p.update_layout(height=340, xaxis_title="Timestamp", yaxis_title="Active Power (kW)", hovermode="x unified")
            st.plotly_chart(fig_p, width="stretch")

        with c_right:
            st.subheader("Cumulative Energy Yield Loss (MWh)")
            cum_p50 = pd.Series(pw_p50_kw / 6000.0, index=outage_idx).cumsum()
            cum_p10 = pd.Series(pw_p10_kw / 6000.0, index=outage_idx).cumsum()
            cum_p90 = pd.Series(pw_p90_kw / 6000.0, index=outage_idx).cumsum()
            cum_clo = pd.Series(pw_coef_lo_kw / 6000.0, index=outage_idx).cumsum()
            cum_chi = pd.Series(pw_coef_hi_kw / 6000.0, index=outage_idx).cumsum()

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=outage_idx, y=cum_p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_cum.add_trace(
                go.Scatter(
                    x=outage_idx,
                    y=cum_p10,
                    fill="tonexty",
                    fillcolor="rgba(14, 165, 233, 0.15)",
                    line=dict(width=0),
                    name="Cumulative P10–P90 Band",
                )
            )
            fig_cum.add_trace(go.Scatter(x=outage_idx, y=cum_chi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_cum.add_trace(
                go.Scatter(
                    x=outage_idx,
                    y=cum_clo,
                    fill="tonexty",
                    fillcolor="rgba(2, 132, 199, 0.40)",
                    line=dict(width=0),
                    name="Cumulative 95% Coeff CI",
                )
            )
            fig_cum.add_trace(
                go.Scatter(
                    x=outage_idx,
                    y=cum_p50,
                    line=dict(color="#0284c7", width=2.3),
                    name="Cumulative P50 Loss (MWh)",
                )
            )
            fig_cum.update_layout(height=340, xaxis_title="Timestamp", yaxis_title="Cumulative Loss (MWh)", hovermode="x unified")
            st.plotly_chart(fig_cum, width="stretch")

        st.subheader("Model Accuracy Check on Unseen Validation Period")
        st.caption(
            f"**Evaluated on:** {val_description}. "
            "**Summed Energy Yield Error (%)** measures `(Predicted MWh − Actual MWh) / Actual MWh` over the known validation period "
            "(where real turbine measurements exist) to audit whether cumulative MWh claims have any systematic bias."
        )
        scorecard_df = pd.DataFrame(
            [
                {"Metric": "Wind Speed MAE", "Value": f"{bt_ws_mae:.3f} m/s", "Description": "Mean absolute 10-minute wind error"},
                {"Metric": "Wind Speed RMSE", "Value": f"{bt_ws_rmse:.3f} m/s", "Description": "Root mean squared 10-minute wind error"},
                {"Metric": "Wind Speed R²", "Value": f"{bt_ws_r2:.3f}", "Description": "Share of wind speed variance explained"},
                {"Metric": "Power MAE", "Value": f"{bt_pw_mae:.1f} kW", "Description": "Mean absolute 10-minute active power error"},
                {"Metric": "Power RMSE", "Value": f"{bt_pw_rmse:.1f} kW", "Description": "Root mean squared 10-minute active power error"},
                {"Metric": "Validation Actual vs. Predicted Energy", "Value": f"{bt_act_mwh:.2f} MWh vs. {bt_pred_mwh:.2f} MWh", "Description": f"Net difference: {bt_dev_mwh:+.2f} MWh"},
                {"Metric": "Summed Energy Yield Error (Validation)", "Value": f"{bt_dev_pct:+.2f}%", "Description": "Official hackathon metric: (Pred MWh − Actual MWh) / Actual MWh"},
                {"Metric": "80% Band Empirical Coverage (P10–P90)", "Value": f"{bt_cov_pct:.1f}%", "Description": "Nominal calibration target: 80.0%"},
            ]
        )
        st.dataframe(scorecard_df, hide_index=True, width="stretch")

        st.markdown("---")
        st.subheader("📨 Insurance Claim Summary & Settlement Notice")
        with st.expander("Preview Formal Business Interruption Insurance Claim Notice", expanded=False):
            hours = len(outage_idx) / 6.0
            claim_text = f"""Subject: Business Interruption Insurance Claim Notice - Turbine {target_name} Outage

To: Claims Adjustment Team (Renewable Energy Business Interruption Division)
From: Asset Operations & Yield Management (BayWa r.e.)
Date: {pd.Timestamp.now():%d %B %Y}

CLAIM REFERENCE: BI-CLM-{target_name}-{outage_start:%Y%m%d}
POLICY ID: WIND-ALL-RISK-2026-EU

1. INCIDENT OVERVIEW
* Affected Asset: Turbine {target_name}
* Outage Start: {outage_start:%Y-%m-%d %H:%M} (Berlin / Germany Time)
* Outage End:   {outage_end:%Y-%m-%d %H:%M} (Berlin / Germany Time)
* Total Duration: {hours:.1f} Operating Hours ({len(outage_idx)} consecutive 10-minute intervals)

2. COUNTERFACTUAL LOSS ADJUSTMENT (IEC 61400 COMPLIANT)
* Reference Turbines Utilized: {', '.join(active_refs.keys())}
* Methodology: Bayesian Ridge Regression with permanent orthogonal directional vectors (u, v).
* Closed-Form Transfer Equation: {predictor.get_formula()}
* Atmospheric Correction: IEC 61400-12-1 Air Density Normalization ({'Enabled' if use_density else 'Disabled'}).

ESTIMATED PRODUCTION LOSS SUMMARY (MWh):
* Central Loss Estimate (P50):             {loss_p50_mwh:,.2f} MWh
* 95% Coefficient Confidence Interval:     {loss_coef_lo_mwh:,.2f} MWh to {loss_coef_hi_mwh:,.2f} MWh
* 80% Bayesian Predictive Band (P10–P90):  {loss_p10_mwh:,.2f} MWh to {loss_p90_mwh:,.2f} MWh
* Outage Mean Lost Power:                  {pw_p50_kw.mean():,.1f} kW

3. AUDIT TRAIL & VALIDATION
* Unseen Validation Summed Energy Error: {bt_dev_pct:+.2f}% ({bt_pred_mwh:.2f} MWh predicted vs. {bt_act_mwh:.2f} MWh actual).
* Unseen Validation Wind RMSE / MAE: {bt_ws_rmse:.3f} m/s / {bt_ws_mae:.3f} m/s.

Authorized Signature:
Asset Performance & Risk Engineering Team, BayWa r.e.
"""
            st.text_area("Generated Claim Notice", claim_text, height=320)
            st.download_button(
                "Download Claim Notice (.txt)",
                data=claim_text,
                file_name=f"insurance_claim_{target_name}_{outage_start:%Y%m%d}.txt",
                mime="text/plain",
            )

    # ==================== TAB 4 ====================
    with tab4:
        st.subheader(f"10-Minute Interval Predictions — Turbine {target_name}")
        certainty_pct = st.slider(
            "Symmetric Percentile Certainty Level (%)",
            min_value=50,
            max_value=99,
            value=80,
            step=1,
            help="Controls the symmetric Bayesian credible band around the P50 median for every 10-minute interval (e.g. 80% → P10.0 to P90.0; 90% → P5.0 to P95.0; 95% → P2.5 to P97.5).",
        )

        p_low_pct = (100.0 - certainty_pct) / 2.0
        p_high_pct = 100.0 - p_low_pct
        low_label = f"P{p_low_pct:g}"
        high_label = f"P{p_high_pct:g}"

        dyn_band = predictor.predict_certainty_band(outage_feats, certainty_pct=float(certainty_pct), coef_ci_pct=95.0)
        ws_lo = dyn_band["lower"].to_numpy(dtype=float)
        ws_md = dyn_band["median"].to_numpy(dtype=float)
        ws_hi = dyn_band["upper"].to_numpy(dtype=float)
        ws_coef_hw = dyn_band["coef_ci_halfwidth"].to_numpy(dtype=float)

        pw_lo = power_curve.predict_power(ws_lo, air_density=outage_density)
        pw_md = power_curve.predict_power(ws_md, air_density=outage_density)
        pw_hi = power_curve.predict_power(ws_hi, air_density=outage_density)

        en_lo_mwh = pw_lo / 6000.0
        en_md_mwh = pw_md / 6000.0
        en_hi_mwh = pw_hi / 6000.0
        cum_md_mwh = np.cumsum(en_md_mwh)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Lower Bound Total ({low_label})", f"{en_lo_mwh.sum():,.2f} MWh")
        k2.metric("Median Total (P50)", f"{en_md_mwh.sum():,.2f} MWh")
        k3.metric(f"Upper Bound Total ({high_label})", f"{en_hi_mwh.sum():,.2f} MWh")
        k4.metric(
            f"{certainty_pct}% Band Width",
            f"{(en_hi_mwh.sum() - en_lo_mwh.sum()):,.2f} MWh",
            help=f"Symmetric {low_label}–{high_label} range across {len(outage_idx)} intervals.",
        )

        interval_df = pd.DataFrame(
            {
                "Timestamp": outage_idx,
                f"Wind Lower {low_label} (m/s)": np.round(ws_lo, 3),
                "Wind Median P50 (m/s)": np.round(ws_md, 3),
                f"Wind Upper {high_label} (m/s)": np.round(ws_hi, 3),
                "Coeff 95% CI ± (m/s)": np.round(ws_coef_hw, 4),
                f"Power Lower {low_label} (kW)": np.round(pw_lo, 1),
                "Power Median P50 (kW)": np.round(pw_md, 1),
                f"Power Upper {high_label} (kW)": np.round(pw_hi, 1),
                f"Energy Lower {low_label} (MWh)": np.round(en_lo_mwh, 5),
                "Energy Median P50 (MWh)": np.round(en_md_mwh, 5),
                f"Energy Upper {high_label} (MWh)": np.round(en_hi_mwh, 5),
                "Cumulative Loss P50 (MWh)": np.round(cum_md_mwh, 4),
            }
        )

        st.dataframe(interval_df, hide_index=True, width="stretch", height=440)

        csv_bytes = interval_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download 10-Minute Predictions ({certainty_pct}% Band, MWh) as CSV",
            data=csv_bytes,
            file_name=f"predictions_10min_{target_name}_{certainty_pct}pct_{outage_start:%Y%m%d}.csv",
            mime="text/csv",
        )


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
except ImportError:
    _get_ctx = lambda: None

if __name__ == "__main__" or _get_ctx() is not None:
    if _get_ctx() is not None:
        render_app()
