"""Run with: streamlit run app.py."""

import pandas as pd
import streamlit as st

from claim_workflow import run_claim_scenario


st.set_page_config(page_title="InSureWind", page_icon="🌬️", layout="wide")
st.title("InSureWind counterfactual claim estimator")
st.write("External-farm SCADA → P90/P50/P10 counterfactual power → auditable BI claim.")
with st.sidebar:
    target = st.selectbox("Target turbine", ("T1", "T2", "T3"))
    ppa = st.number_input("PPA EUR/kWh", min_value=0.0, value=0.08, step=0.005)
    ptc = st.number_input("PTC EUR/kWh", min_value=0.0, value=0.025, step=0.005)
    gross_profit = st.slider("Gross-profit rate", 0.0, 1.0, 0.85, 0.01)
    deductible = st.number_input("Time deductible hours", min_value=0, value=24)

try:
    forecast, scenarios = run_claim_scenario(target, ppa_rate=ppa, ptc_rate=ptc,
                                               gross_profit_rate=gross_profit, deductible_hours=deductible)
except Exception as error:
    st.error(f"Unable to calculate scenario: {error}")
    st.stop()

for column, label in zip(st.columns(3), ("P90", "P50", "P10")):
    column.metric(f"{label} net claim", f"EUR {scenarios[label]['net_claim_payable_euro']:,.2f}")
st.subheader("Claim scenarios")
st.dataframe(pd.DataFrame(scenarios).T.style.format("EUR {:,.2f}"), use_container_width=True)
st.subheader("Counterfactual power range")
st.line_chart(forecast.set_index("timestamp")[["P90_yield", "P50_yield", "P10_yield"]])
st.download_button("Download interval-level audit CSV", forecast.to_csv(index=False).encode(),
                   file_name=f"{target}_counterfactual_claim_forecast.csv", mime="text/csv")
