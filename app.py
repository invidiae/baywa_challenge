"""Business-friendly multi-reference wind imputation dashboard."""
import urllib.parse

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wind_imputation import load_power_curve, load_scada, power_curve, train_and_evaluate


def rmse(actual, predicted) -> float:
    """Use sklearn's requested RMSE API with compatibility for newer releases."""
    try:
        return float(mean_squared_error(actual, predicted, squared=False))
    except TypeError:  # scikit-learn >= 1.6 removed ``squared``
        return float(np.sqrt(mean_squared_error(actual, predicted)))


st.set_page_config(page_title="Wind Imputation Lab", page_icon="🌬️", layout="wide")
st.title("Wind Farm Missing Data Imputation")
business_view = st.toggle("Business View Mode", value=True)
st.caption("Upload the three files below. Missing target readings are detected and estimated automatically.")

accepted_formats = ["csv", "xlsx", "xls"]
target_upload = st.file_uploader("1. Target Turbine SCADA", type=accepted_formats, accept_multiple_files=False)
reference_uploads = st.file_uploader("2. Reference Farm SCADA files", type=accepted_formats, accept_multiple_files=True)
curve_upload = st.file_uploader("3. Power Curve Data", type=accepted_formats, accept_multiple_files=False)
test_fraction = st.sidebar.slider("Final chronological test fraction", .20, .30, .25, .05)

if not target_upload or not reference_uploads or not curve_upload:
    st.info("Please provide one target SCADA file, one or more reference SCADA files, and one power-curve file. CSV and Excel are supported.")
    st.stop()

try:
    with st.spinner("Checking the data and training the prediction model..."):
        target = load_scada(target_upload)
        references = [load_scada(upload) for upload in reference_uploads]
        curve = load_power_curve(curve_upload)
        result = train_and_evaluate(target, references, curve, test_fraction)
except Exception as error:
    st.error(f"Unable to train the model: {error}")
    st.stop()

test = result.test
# These values are intentionally calculated here from the actual chronological
# test rows, rather than displayed from precomputed/static text.
wind_rmse = rmse(test["actual_wind_speed"], test["predicted_wind_speed"])
wind_mae = float(mean_absolute_error(test["actual_wind_speed"], test["predicted_wind_speed"]))
power_rmse = rmse(test["actual_power_kw"], test["predicted_power_kw"])
power_mae = float(mean_absolute_error(test["actual_power_kw"], test["predicted_power_kw"]))
actual_kwh = float((test["actual_power_kw"] * result.interval_minutes / 60).sum())
predicted_kwh = float((test["predicted_power_kw"] * result.interval_minutes / 60).sum())
energy_deviation = predicted_kwh - actual_kwh
wind_r2 = float(r2_score(test["actual_wind_speed"], test["predicted_wind_speed"]))

if business_view:
    st.header("How accurate is the prediction")
    cards = [
        ("Average wind speed error", wind_mae, "m/s", "This means the prediction is usually this far from the real wind speed."),
        ("Typical larger wind-speed error", wind_rmse, "m/s", "This gives extra weight to the bigger misses."),
        ("Average power error", power_mae, "kW", "This is how far the predicted turbine power is usually from reality."),
        ("Total energy difference", energy_deviation, "kWh", "Positive means we predicted more energy than was produced; negative means less."),
    ]
else:
    st.header("Accuracy Metrics")
    cards = [("Wind RMSE", wind_rmse, "m/s", ""), ("Wind MAE", wind_mae, "m/s", ""),
             ("Power RMSE", power_rmse, "kW", ""), ("Power MAE", power_mae, "kW", ""),
             ("Energy Yield Deviation", energy_deviation, "kWh", "")]
for column, (label, value, unit, help_text) in zip(st.columns(len(cards)), cards):
    column.metric(label, f"{value:,.2f} {unit}", help=help_text or None)

st.header("Wind Speed Visualization" if not business_view else "Wind speed: real versus estimated")
wind_chart = px.line(test, x="timestamp", y=["actual_wind_speed", "predicted_wind_speed"],
                     labels={"value": "Wind speed (m/s)", "variable": "Series"})
st.plotly_chart(wind_chart, use_container_width=True)

st.header("Prediction Method & Uncertainty" if not business_view else "How we make the estimate")
if business_view:
    st.write(f"We look at the wind readings from {len(references)} nearby reference files to estimate the missing target reading. The shaded band shows a cautious low-to-high range around each estimate.")
else:
    st.markdown(f"**{result.selected_model} regression** was selected by three expanding chronological development folds (mean validation RMSE: **{result.cv_scores[result.selected_model]:.3f} m/s**). It uses every uploaded reference turbine as a feature, temporal features, and P10/P50/P90 empirical tree-ensemble quantiles. Final wind-speed R²: **{wind_r2:.3f}**.")
uncertainty = go.Figure()
uncertainty.add_trace(go.Scatter(x=test.timestamp, y=test.p90_wind_speed, line=dict(width=0), name="P90"))
uncertainty.add_trace(go.Scatter(x=test.timestamp, y=test.p10_wind_speed, fill="tonexty", line=dict(width=0), name="P10-P90 range"))
uncertainty.add_trace(go.Scatter(x=test.timestamp, y=test.actual_wind_speed, name="Actual"))
uncertainty.add_trace(go.Scatter(x=test.timestamp, y=test.predicted_wind_speed, name="Prediction"))
uncertainty.update_yaxes(title="Wind speed (m/s)")
st.plotly_chart(uncertainty, use_container_width=True)

st.header("Energy Loss Calculation" if not business_view else "Missing-period energy estimate")
if result.outage.empty:
    st.info("No missing target wind-speed intervals were detected.")
else:
    auto_start,auto_end=result.outage.timestamp.min().date(),result.outage.timestamp.max().date()
    window=st.date_input("Detected outage period editable",value=(auto_start,auto_end),min_value=auto_start,max_value=auto_end)
    start,end=(window if isinstance(window,tuple) and len(window)==2 else (auto_start,auto_end)); outage=result.outage[(result.outage.timestamp.dt.date>=start)&(result.outage.timestamp.dt.date<=end)].copy()
    edited=st.data_editor(outage[["timestamp","p10_wind_speed","predicted_wind_speed","p90_wind_speed"]],disabled=["timestamp","p10_wind_speed","p90_wind_speed"],use_container_width=True,key="outage_editor")
    edited["predicted_power_kw"]=power_curve(edited.predicted_wind_speed,curve); edited["predicted_energy_kwh"]=edited.predicted_power_kw*result.interval_minutes/60
    st.metric("Predicted outage energy" if business_view else "Estimated outage energy yield",f"{edited.predicted_energy_kwh.sum():,.1f} kWh")
    st.caption("You may edit the central predicted wind-speed values above; power and energy update using your uploaded curve.")
    st.subheader("Draft Email Report")
    default_turbine_id = target_upload.name.rsplit(".", 1)[0]
    turbine_id = st.text_input("Turbine ID", value=default_turbine_id, key="email_turbine_id")
    outage_start = start.strftime("%d %B %Y")
    outage_end = end.strftime("%d %B %Y")
    total_kwh_lost = float(edited["predicted_energy_kwh"].sum())
    subject = f"Wind Turbine Outage Report - {turbine_id}"
    body = f"""Wind Turbine Outage Report

Turbine ID: {turbine_id}
Outage period: {outage_start} to {outage_end}
Estimated energy lost: {total_kwh_lost:,.1f} kWh

This estimate is based on the dashboard's current model predictions and selected outage period.

Kind regards,
Wind Energy Analytics Team
"""
    email_link = f"mailto:?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
    st.markdown(
        f'<a href="{email_link}" style="display:inline-block;padding:0.6rem 1rem;'
        'background:#ff4b4b;color:white;border-radius:0.5rem;text-decoration:none;font-weight:600;">'
        'Draft Email in Local Mail Client</a>',
        unsafe_allow_html=True,
    )

st.header("Energy Loss Visualization" if not business_view else "Power difference over time")
power_chart=go.Figure()
power_chart.add_trace(go.Scatter(x=test.timestamp,y=test.actual_power_kw,name="Actual power"))
power_chart.add_trace(go.Scatter(x=test.timestamp,y=test.predicted_power_kw,name="Predicted power",fill="tonexty",fillcolor="rgba(228,87,86,.20)"))
power_chart.update_yaxes(title="Power (kW)")
st.plotly_chart(power_chart,use_container_width=True)
