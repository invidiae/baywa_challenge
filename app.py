import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta

from outage_model import predict_wind, expected_power, backtest, to_power

DATA = Path(__file__).parent / "data"
TARGETS = ["T1", "T2", "T3"]
REFS = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]

st.set_page_config(page_title="Wind Outage Loss Estimator", page_icon="🌬️", layout="wide")
st.markdown("""
<style>
.hero {background: linear-gradient(90deg, #0b3d91, #1a73e8); padding: 28px 32px;
       border-radius: 14px; color: white; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 6px 0 0 0; opacity: 0.9;}
div[data-testid="stMetric"] {background: #f1f5f9; padding: 14px 18px; border-radius: 12px;}
div[data-testid="stMetric"] * {color: #0f172a !important;}
</style>
<div class="hero">
  <h1>🌬️ Wind Farm Outage: Energy Loss Estimator</h1>
  <p>Estimate the energy lost during a communication outage using neighbouring turbines.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- data loading ----------------
@st.cache_data
def load_ws(name):
    d = pd.read_excel(DATA / f"{name}_data.xlsx")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()

@st.cache_data
def load_curve(name):
    c = pd.read_excel(DATA / f"{name}_power_curve.xlsx").sort_values("wind_speed")
    return c["wind_speed"].to_numpy(float), c["power"].to_numpy(float)

def read_any(file):
    """Read a CSV or Excel file with timestamp + wind speed columns."""
    name = getattr(file, "name", str(file)).lower()
    d = pd.read_csv(file, sep=None, engine="python") if name.endswith(".csv") else pd.read_excel(file)
    cols = {str(c).strip().lower().replace(" ", "_"): c for c in d.columns}
    tcol = cols.get("timestamp", d.columns[0])
    wcol = cols.get("wind_speed", d.columns[1])
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(d[tcol]),
        "wind_speed": pd.to_numeric(d[wcol].astype(str).str.replace(",", "."), errors="coerce"),
        "test_set": d[cols["test_set"]].astype(bool) if "test_set" in cols else False,
    })
    if out["timestamp"].dt.tz is not None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    return out.drop_duplicates("timestamp").set_index("timestamp").sort_index()

@st.cache_data
def rank_refs(target):
    t = load_ws(target)
    train = t.loc[~t["test_set"].astype(bool), "wind_speed"]
    rows = []
    for r in REFS:
        both = pd.concat([train, load_ws(r)["wind_speed"]], axis=1, keys=["t", "r"]).dropna()
        rows.append({"Turbine": r, "Correlation": round(both["t"].corr(both["r"]), 3)})
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False).reset_index(drop=True)

# ---------------- helpers ----------------
def longest_nan_run(s):
    miss = s.isna()
    if not miss.any():
        return None
    grp = (miss != miss.shift()).cumsum()
    runs = [(g.index[0], g.index[-1]) for _, g in miss[miss].groupby(grp[miss])]
    return max(runs, key=lambda r: r[1] - r[0])

# ---------------- sidebar ----------------
st.sidebar.header("1. Turbines")
target = st.sidebar.selectbox("Affected turbine (lost connection)", TARGETS)
st.sidebar.caption(f"Power curve used: {target}")
with st.sidebar.expander("Upload test data instead (CSV or Excel)"):
    up_t = st.file_uploader("Affected turbine file", type=["csv", "xlsx"], key="up_t")
    up_r = st.file_uploader("Reference turbine file(s)", type=["csv", "xlsx"],
                            accept_multiple_files=True, key="up_r")
ranking = rank_refs(target)
if up_r:
    ref_data = {Path(f.name).stem: read_any(f)["wind_speed"] for f in up_r}
    refs = list(ref_data)
    st.sidebar.caption("Using uploaded references: " + ", ".join(refs))
else:
    refs = st.sidebar.multiselect("Reference turbines", REFS,
                                  default=ranking["Turbine"].tolist(), key=f"refs_{target}")
    ref_data = {r: load_ws(r)["wind_speed"] for r in refs}
if not refs:
    st.warning("Pick at least one reference turbine in the sidebar.")
    st.stop()

st.sidebar.header("2. Settings")
unit = st.sidebar.selectbox("Power curve unit", ["kW", "MW"])
factor = 1000 if unit == "MW" else 1

# ---------------- build the table ----------------
tgt_raw = read_any(up_t) if up_t else load_ws(target)
idx = pd.date_range(tgt_raw.index.min(), tgt_raw.index.max(), freq="10min")
df = pd.DataFrame(ref_data).reindex(idx)
if df.notna().sum().sum() == 0:
    st.error("The reference data does not overlap with the affected turbine's timestamps.")
    st.stop()
df["tgt"] = tgt_raw["wind_speed"].reindex(idx)
df["test"] = tgt_raw["test_set"].reindex(idx).fillna(False).astype(bool)
curve = load_curve(target)

# ---------------- outage ----------------
st.sidebar.header("3. Outage period")
auto = longest_nan_run(df.loc[df["test"], "tgt"]) or longest_nan_run(df["tgt"])
if auto:
    st.sidebar.success(f"Auto-detected: {auto[0]:%d %b %H:%M} to {auto[1]:%d %b %H:%M}")
    default = (auto[0].to_pydatetime(), auto[1].to_pydatetime())
else:
    st.sidebar.warning("No missing data found. Showing the last 3 days as a test outage.")
    default = ((idx[-1] - timedelta(days=3)).to_pydatetime(), idx[-1].to_pydatetime())

start, end = st.sidebar.slider("Adjust outage window", min_value=idx.min().to_pydatetime(),
                               max_value=idx.max().to_pydatetime(), value=default,
                               step=timedelta(minutes=10), format="DD MMM HH:mm")
start, end = pd.Timestamp(start), pd.Timestamp(end)
out_idx = df.index[(df.index >= start) & (df.index <= end)]

# ---------------- predict + backtest ----------------
fit_mask = ~df["test"] & ~((df.index >= start) & (df.index <= end))
pred, info = predict_wind(df, refs, fit_mask, out_idx)
pred = pred.interpolate(limit_direction="both")
if pred.isna().all():
    st.error("No reference data is available during the outage, so nothing can be predicted.")
    st.stop()

# backtest: practise on the history BEFORE the outage (learn from the past, hide a window, compare)
hist = df[fit_mask & (df.index < start)]
if len(hist) < 300:
    st.error("Not enough normal data before the outage to learn from and validate. "
             "Move the outage window later.")
    st.stop()
m, act_bt, pred_bt = backtest(hist, refs, curve, factor)
sigma = (pred_bt - act_bt).std()                    # typical wind error on unseen data

# energy for the real outage (expected power over the typical error)
pw = expected_power(pred, curve, info["sigma"]) * factor
pw_lo = expected_power((pred - 1.96 * sigma).clip(lower=0), curve, info["sigma"]) * factor
pw_hi = expected_power(pred + 1.96 * sigma, curve, info["sigma"]) * factor
E = pw.sum() / 6                                    # kW x 10 min -> kWh
dev_typ = m["Energy deviation, mean abs (%)"] / 100
E_lo, E_hi = E * (1 - dev_typ), E * (1 + dev_typ)

# ---------------- tabs ----------------
tab1, tab2, tab3 = st.tabs(["Wind speed", "Prediction method", "Energy loss"])

with tab1:
    pad = timedelta(days=2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[refs].mean(axis=1), name="Reference (average)",
                             line=dict(color="#9aa0a6", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["tgt"], name=f"{target} (measured)",
                             line=dict(color="#1a73e8", width=1.5)))
    fig.add_trace(go.Scatter(x=pred.index, y=pred, name=f"{target} (predicted)",
                             line=dict(color="#e8710a", width=2)))
    fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Outage")
    fig.update_xaxes(range=[start - pad, end + pad])
    fig.update_layout(height=450, xaxis_title="Time", yaxis_title="Wind speed (m/s)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag on the chart to zoom, double-click to reset.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outage length", f"{len(out_idx) / 6:.1f} h")
    c2.metric("Mean predicted wind", f"{pred.mean():.2f} m/s")
    c3.metric("Backtest RMSE", f"{m['RMSE wind (m/s)']:.3f} m/s")
    c4.metric("Backtest MAE", f"{m['MAE wind (m/s)']:.3f} m/s")

with tab2:
    left, right = st.columns([3, 2])
    with left:
        lim = float(max(act_bt.max(), pred_bt.max())) + 1
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=act_bt, y=pred_bt, mode="markers", name="Backtest points",
                                marker=dict(size=5, opacity=0.5, color="#1a73e8")))
        f2.add_trace(go.Scatter(x=[0, lim], y=[0, lim], name="Perfect prediction",
                                line=dict(color="#e8710a", dash="dash")))
        f2.update_layout(height=420, xaxis_title=f"Measured {target} (m/s)",
                         yaxis_title=f"Predicted {target} (m/s)")
        st.plotly_chart(f2, width="stretch")
        st.caption("Backtest: windows of normal data were hidden, predicted from the data before them, "
                   "then compared with the truth.")
    with right:
        st.subheader("How the prediction works")
        st.markdown("1. **Align** each neighbour in time (wind reaches the farms at different moments).\n"
                    "2. **Learn** a weighted formula from the period before the outage.\n"
                    "3. **Predict** the missing wind speed from the neighbours' wind speed.\n"
                    "4. **Convert** wind to power with the power curve, averaged over the typical "
                    "prediction error (the power curve is not a straight line).\n"
                    "5. **Add up** the power over the outage to get kWh.")
        st.metric("Typical wind error (±1σ)", f"{sigma:.2f} m/s")
        st.metric("Typical energy error (backtest)", f"{m['Energy deviation, mean abs (%)']:.1f} %")
        st.write("Time shift per turbine (10-min steps):", info["lags"])
        st.write("Weights:", {k: round(v, 2) for k, v in info["weights"].items()})
    if not (up_t or up_r):
        st.subheader(f"How well does each turbine match {target}?")
        st.dataframe(ranking, width="stretch", hide_index=True)

with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated energy loss", f"{E:,.0f} kWh")
    c2.metric("Typical range (± backtest error)", f"{E_lo/1000:,.0f}k to {E_hi/1000:,.0f}k kWh")
    c3.metric("Average lost power", f"{pw.mean():,.0f} kW")
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=pw_hi.index, y=pw_hi, line=dict(width=0), showlegend=False))
    f3.add_trace(go.Scatter(x=pw_lo.index, y=pw_lo, fill="tonexty", fillcolor="rgba(232,113,10,0.15)",
                            line=dict(width=0), name="Wind uncertainty (95%)"))
    f3.add_trace(go.Scatter(x=pw.index, y=pw, name="Lost power (kW)", line=dict(color="#e8710a", width=2)))
    f3.update_layout(height=320, xaxis_title="Time", yaxis_title="Power (kW)", hovermode="x unified")
    st.plotly_chart(f3, width="stretch")
    f4 = go.Figure(go.Scatter(x=pw.index, y=(pw / 6).cumsum(), fill="tozeroy",
                              line=dict(color="#1a73e8"), name="Cumulative loss"))
    f4.update_layout(height=280, xaxis_title="Time", yaxis_title="Cumulative lost energy (kWh)")
    st.plotly_chart(f4, width="stretch")
    st.subheader("Accuracy (backtest)")
    st.dataframe(pd.DataFrame({"Metric": list(m.keys()), "Value": [round(v, 3) for v in m.values()]}),
                 width="stretch", hide_index=True)