# BayWa challenge - InSureWind

An auditable business-interruption claim demonstrator for wind-turbine
telemetry outages. The workflow uses only reference turbines outside the
target park, forecasts P90/P50/P10 counterfactual power with LightGBM quantile
regression, maps the result to 10-minute energy loss, and applies explicit
insurance policy terms.

## Run

```bash
uv sync
uv run streamlit run app.py
```

The dashboard lets an analyst switch targets, adjust PPA/PTC, gross-profit and
deductible assumptions, inspect uncertainty, and download interval-level audit
data.

## Claim safeguards

- Missing SCADA remains missing; it is never changed to zero wind speed.
- Target-turbine weather is a training label only, not an outage-period input.
- Wind direction is aggregated with sine/cosine vectors, avoiding the 359/1
  degree averaging error.
- Independent quantiles are ordered before display.
- Each contiguous outage is priced separately, so a deductible is not diluted
  across unrelated communication gaps.

The operational-analysis framing is informed by [NREL OpenOA](https://github.com/NatLabRockies/OpenOA).
