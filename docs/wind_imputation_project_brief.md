# Wind Farm Missing Data Imputation Project Brief

## 1. Project goal

Build a transparent tool that estimates missing wind-speed readings for a target wind turbine using SCADA data from nearby reference farms. The tool converts estimated wind speed into power and energy with a turbine power curve, quantifies uncertainty, and provides a simple interface for technical and business users.

The project focuses on physics, machine learning, power, and energy validation. It does not use revenue or insurance calculations in the dashboard workflow.

## 2. Business problem

Wind-farm SCADA data can contain communication gaps. When a target turbine has missing wind-speed data, operators need a credible estimate of:

- What the wind speed was likely to be.
- How much power the turbine would likely have produced.
- How much energy is associated with the missing period.
- How uncertain the estimate is.

## 3. Data used

### Target turbines

- T1
- T2
- T3

### Reference turbines

- H1, H2, H3, H4, H5
- O1, O2

### Available SCADA variables

- Timestamp
- Wind speed
- Wind direction
- Ambient temperature
- Air pressure
- Wind-speed standard deviation
- Test-set flag

The data is sampled at 10-minute resolution. Target-specific power curves are available for T1, T2, and T3.

## 4. End-to-end workflow

```text
Target SCADA + multiple reference-farm SCADA files + target power curve
                    |
                    v
Time alignment and data-quality validation
                    |
                    v
Physics-informed reference features
                    |
                    v
Chronological model selection and custom quantile prediction
                    |
                    v
P10 / P50 / P90 target wind-speed estimates
                    |
                    v
Power-curve conversion to kW and kWh
                    |
                    v
Validation metrics, outage review, editable predictions, email report
```

## 5. Chronological validation methodology

The model is evaluated without random shuffling.

1. The SCADA timeline is sorted by timestamp.
2. Rows where `test_set == False` are used for model training.
3. Rows where `test_set == True` are reserved for ground-truth validation.
4. Missing target wind-speed values are never used as training labels.
5. The target wind speed is never provided as a prediction feature.

This approach reflects the real operational problem: estimate the target from information that would be available externally during a data outage.

## 6. Machine-learning approach

### Custom model

The evaluation tool imports the project’s actual custom class:

```python
from src.quantile_model import QuantileLossPredictor
```

The custom predictor fits three LightGBM quantile regression models:

- P90: conservative lower-yield / lower-wind estimate
- P50: central estimate used for the main prediction
- P10: optimistic upper-yield / upper-wind estimate

### Model-selection approach in the dashboard

The Streamlit dashboard also benchmarks Extra Trees and Random Forest models using expanding chronological development folds. The best validation-RMSE model is selected, then tested on the held-out final time block.

### Reference features

Every uploaded reference turbine is included individually. The model also receives ensemble summary features:

- Mean reference wind speed
- Median reference wind speed
- Standard deviation across reference wind speeds
- Number of available reference readings
- Hour of day
- Day of year

## 7. Physics-informed features

The feature-engineering function `apply_physics_features(df)` adds domain features before model training.

### Air density

The function applies the dry-air ideal gas law:

```text
rho = (pressure × 100) / (287.05 × (temperature + 273.15))
```

This creates `air_density_kg_m3`.

### Density-normalized wind speed

Wind speed is normalized to the IEC reference density of 1.225 kg/m³:

```text
normalized_wind_speed = wind_speed × (rho / 1.225)^(1/3)
```

This accounts for the fact that wind-energy flux changes with air density.

### Direction and wake proxy

Wind direction is transformed into cyclical components:

- `sin_wind_dir`
- `cos_wind_dir`

A directional wake-sector proxy flags 250° to 290°:

```text
is_wake_sector = 1 when 250° <= wind direction <= 290°, otherwise 0
```

### Turbulence proxy

The function creates a six-period rolling standard deviation:

```text
wind_speed_rolling_std_1h
```

For 10-minute SCADA data, six periods represent one hour. This captures short-term wind variability that can affect production nonlinearly.

## 8. Power and energy conversion

The dashboard can use an uploaded target-turbine power curve. The standard fallback physical curve uses:

- Cut-in speed: 3 m/s
- Rated speed: 12 m/s
- Cut-out speed: 25 m/s
- Rated power: 3,000 kW

At 10-minute resolution:

```text
Energy kWh = Power kW × 10 / 60
```

## 9. Validation metrics

### Wind-speed accuracy

- MAE: average absolute wind-speed error
- RMSE: error metric that gives more weight to large errors
- MSE: squared-error measure
- R²: explained variance versus the observed target wind speed
- MAPE: average percentage error
- Bias: average signed error
- NRMSE: RMSE as a percentage of average wind speed

### Power and energy diagnostics

- Power MAE, RMSE, and MSE
- Peak positive power gap
- Actual and predicted energy yield
- Energy yield deviation in kWh and percent
- Positive energy gap/loss

## 10. Ground-truth energy validation results

These results use the full `test_set == True` period, all seven reference turbines, target-specific power curves, and the physics-enhanced custom quantile model.

| Target | Actual energy | Predicted energy | Deviation | Deviation percent |
|---|---:|---:|---:|---:|
| T1 | 213,159.820 kWh | 213,577.929 kWh | +418.109 kWh | +0.196% |
| T2 | 312,230.998 kWh | 292,361.796 kWh | -19,869.202 kWh | -6.364% |
| T3 | 271,666.147 kWh | 239,114.983 kWh | -32,551.164 kWh | -11.982% |

Interpretation:

- Positive deviation means predicted energy was higher than actual energy.
- Negative deviation means predicted energy was lower than actual energy.
- T1 is very closely calibrated on total energy over the validation period.
- T2 and T3 show under-prediction, making them priority candidates for further feature, reference-selection, or hyperparameter improvement.

## 11. T2 wind-speed accuracy with all reference turbines

Using H1-H5 and O1-O2:

| Metric | Result |
|---|---:|
| MAE | 1.014 m/s |
| RMSE | 1.360 m/s |
| R² | 0.745 |
| MAPE | 15.695% |
| MSE | 1.848 (m/s)² |
| Bias | -0.314 m/s |

The multi-reference approach materially outperformed a single-reference setup. Earlier T2 evaluation with H1 alone had an RMSE of 1.773 m/s; using all seven reference turbines reduced it to approximately 1.36 m/s.

## 12. Streamlit dashboard functionality

### Three-input architecture

The dashboard supports:

1. Target turbine SCADA file
2. Multiple reference-farm SCADA files
3. Target-turbine power-curve file

CSV and Excel (`.xlsx`, `.xls`) are accepted for dashboard inputs.

### Business View Mode

The dashboard includes a `Business View Mode` toggle.

- Business mode uses plain-language descriptions, such as average wind-speed error and total energy difference.
- Technical mode displays engineering terminology, including RMSE, MAE, R², model-validation score, and quantile uncertainty.

### Outage review and manual adjustment

- Missing target wind-speed intervals are detected automatically.
- The outage date selection is editable.
- P10 and P90 uncertainty bounds are shown.
- Users can edit P50 wind-speed estimates during the selected outage period.
- Power and kWh update from the edited estimate and uploaded turbine curve.

### Email report

The dashboard can generate a local-mail-client draft containing:

- Turbine ID
- Selected outage start and end dates
- Predicted outage energy in kWh

The user only needs to add a recipient and send the email.

## 13. Key code files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard, uploads, business/technical views, outage editing, email draft |
| `wind_imputation.py` | Multi-reference preprocessing, physics features, tree models, power conversion |
| `src/quantile_model.py` | Custom LightGBM P10/P50/P90 quantile predictor |
| `evaluate_accuracy_standalone.py` | Offline terminal ground-truth evaluator |

## 14. Terminal command for standalone evaluation

```bash
cd /Users/abhayrajak/Documents/Hackathon_EGsM
source .venv/bin/activate

OMP_NUM_THREADS=1 python evaluate_accuracy_standalone.py \
  --target T2_data.xlsx \
  --references H1_data.xlsx H2_data.xlsx H3_data.xlsx H4_data.xlsx H5_data.xlsx O1_data.xlsx O2_data.xlsx \
  --power-curve T2_power_curve.xlsx
```

## 15. Collaboration repository

Shared development branch:

```text
https://github.com/invidiae/baywa_challenge/tree/codex/insurewind-mvp
```

Latest pushed commit at the time of this brief:

```text
280e92c Add physics-based offline accuracy evaluation
```

## 16. Suggested slide structure

1. Problem: SCADA data gaps and uncertainty in target-turbine output
2. Solution architecture: reference farms to prediction to power and energy
3. Physics-informed feature engineering
4. Chronological validation methodology
5. Model and uncertainty approach
6. Dashboard workflow for business and technical users
7. T1/T2/T3 energy-validation results
8. T2 detailed model-quality result
9. Outage review, manual adjustment, and email workflow
10. Next steps: improve T2/T3 calibration and compare team models
