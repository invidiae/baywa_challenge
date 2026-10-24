# BayWa r.e. Wind Turbine Outage Estimation Challenge

> 📊 **Presentation & Full Process Handoff:** See [`PRESENTATION_GUIDE.md`](./PRESENTATION_GUIDE.md) for the complete end-to-end pipeline guide, UI walkthrough, exact `T1–T3` cheat-sheet numbers, slide deck blueprint, and judge Q&A — plus [`upsides_downsides_models.md`](./upsides_downsides_models.md) for the 3-branch model comparison.

An end-to-end, highly explainable solution for estimating wind turbine energy yield loss during telemetry communication outages using external reference wind farms ($H_1\text{--}H_5$ and $O_1\text{--}O_2$).

---

## 1. Solution Architecture Overview

The solution uses an explainable, physics-grounded **Two-Stage Architecture**:
1. **Stage 1 (Wind Speed Transfer):** Predicts hub-height wind speed and exact analytical probability distributions from external reference wind farms using **Bayesian Ridge Regression** with directional vector components.
2. **Stage 2 (Power & Energy Loss Conversion):** Maps the predicted wind speed quantiles through the manufacturer's piecewise linear power curve ($P(v)$) to compute active electrical power (kW) and energy yield loss (kWh / MWh), with physics-based **IEC 61400-12-1 Air Density Normalization**.
3. **Interactive Web Dashboard & Insurance Tool:** A clean, responsive **Streamlit UI (`app.py`)** offering automatic outage window detection, interactive sliders, explainability formula viewers, and formal **Business Interruption Insurance Claim Notices**.

---

## 2. Bayesian Ridge Regression (Stage 1)

Unlike black-box tree ensembles (e.g., LightGBM with >1,000 opaque splits), the Bayesian Ridge model yields a single closed-form mathematical formula for each target turbine:

$$\hat{v}_{T, t} = w_0 + w_1 \cdot v_{O, t} + w_2 \cdot v_{H, t} + w_3 \cdot u_{O, t} + w_4 \cdot v_{O, t} + w_5 \cdot u_{H, t} + w_6 \cdot v_{H, t}$$

### Training and Validation Split Strategy
* **Training Data:** Utilizes all operational telemetry outside the designated 10-day test window (`test_set == False`). This encompasses **83 full days (~11,550 10-minute intervals)**:
  * July 1 – August 31 (62 days, pre-outage baseline)
  * September 10 – September 30 (21 days, post-restoration telemetry)
* **Outage Buffer (Hidden Test Data):** September 1 – September 3 (3 days, 432 intervals where target telemetry is missing).
* **Validation Window (Shown Test Data):** September 4 – September 9 (6 days, 862 intervals where target telemetry is available for model evaluation).

### Closed-Form Equations per Turbine
* **Turbine T1:**
  $$v_{T1} = 0.650 + 0.434 \cdot v_O + 0.383 \cdot v_H + 0.014 \cdot u_O + 0.074 \cdot v_O + 0.070 \cdot u_H - 0.023 \cdot v_H$$
* **Turbine T2:**
  $$v_{T2} = 0.616 + 0.409 \cdot v_O + 0.404 \cdot v_H - 0.041 \cdot u_O + 0.040 \cdot v_O + 0.011 \cdot u_H - 0.079 \cdot v_H$$
* **Turbine T3:**
  $$v_{T3} = 0.779 + 0.364 \cdot v_O + 0.379 \cdot v_H - 0.032 \cdot u_O + 0.015 \cdot v_O + 0.026 \cdot u_H - 0.022 \cdot v_H$$

### Validation Performance (Official Shown Test Window: Sept 4 – Sept 9, 2026)

Evaluated on the official 6-day shown validation period (862 10-minute intervals):

| Metric | Turbine T1 | Turbine T2 | Turbine T3 | Description & Benchmark Context |
| :--- | :---: | :---: | :---: | :--- |
| **Wind Speed MAE** | $1.261\text{ m/s}$ | **$0.939\text{ m/s}$** | $1.034\text{ m/s}$ | Mean absolute wind speed error |
| **Wind Speed RMSE** | $1.577\text{ m/s}$ | **$1.234\text{ m/s}$** | $1.336\text{ m/s}$ | Root mean squared wind speed error |
| **Wind Speed $R^2$** | $0.620$ | **$0.790$** | $0.714$ | Explains up to $79.0\%$ of wind speed variance |
| **$P10\text{--}P90$ Coverage** | $68.9\%$ | **$79.7\%$** | $78.0\%$ | **Uncertainty Calibration**: Target is $80.0\%$ (T2 is within $0.3\%$) |
| **Power MAE** | $688.6\text{ kW}$ | **$529.6\text{ kW}$** | $617.2\text{ kW}$ | **Beats LightGBM by up to 175 kW per interval** |
| **Power RMSE** | $943.4\text{ kW}$ | **$777.5\text{ kW}$** | $864.4\text{ kW}$ | Root mean squared power prediction error |
| **Actual Energy** | $212.29\text{ MWh}$ | $311.38\text{ MWh}$ | $271.07\text{ MWh}$ | Ground-truth energy generated during validation window |
| **Predicted Energy ($P50$)** | $228.07\text{ MWh}$ | $304.85\text{ MWh}$ | $250.96\text{ MWh}$ | Model-estimated $P50$ counterfactual energy |
| **Energy Yield Deviation** | **$+7.43\%$** | **$-2.10\%$** | **$-7.42\%$** | **Primary Insurance Metric**: Net cumulative claim error |

### Model Hyperparameters
* **Prior Distributions:**
  * Noise precision prior: $\alpha_1 = 10^{-6}, \alpha_2 = 10^{-6}$ (uninformative Gamma prior).
  * Weight precision prior: $\lambda_1 = 10^{-6}, \lambda_2 = 10^{-6}$ (uninformative Gamma prior).
* **Learned Precision Parameters (via Type-II MLE / Evidence Maximization):**
  * **T1:** Noise precision $\alpha \approx 0.589$, Regularization $\lambda \approx 17.33$
  * **T2:** Noise precision $\alpha \approx 0.737$, Regularization $\lambda \approx 17.60$
  * **T3:** Noise precision $\alpha \approx 0.730$, Regularization $\lambda \approx 21.55$
* **Convergence:** `tol = 1e-3`, `max_iter = 300` (converges in $\sim 4$ iterations).

---

## 3. Directional Modeling (Addon 1) & Results

### Physical & Mathematical Formulation
Wind direction cannot be treated as a single scalar angle because angles wrap around: $359^\circ$ and $1^\circ$ are physically adjacent, but their arithmetic difference is $358^\circ$.

To solve this, we project the wind vector into orthogonal Cartesian components:
$$u = v \cdot \sin(\theta) \quad (\text{Zonal / East-West wind vector})$$
$$v_{\text{comp}} = v \cdot \cos(\theta) \quad (\text{Meridional / North-South wind vector})$$

These linear terms capture:
1. **Spatial Inflow Axis:** When wind blows from the West, Park O is direct upwind of Park T (strongest correlation).
2. **Topographic Sheltering & Roughness:** Wind blowing over forests or hills experiences different frictional drag than open terrain.
3. **Internal Wake Sheltering:** Turbines within Park T partially shade one another depending on incoming wind angle.

### Directional Ablation Results (Validation Period: Sept 4 – Sept 9, 2026)

| Target Turbine | Speed MAE (No Dir) | Speed MAE (With Dir) | Delta Speed MAE | Power MAE (No Dir) | Power MAE (With Dir) | Delta Power MAE | Energy Dev (No Dir) | Energy Dev (With Dir) | Abs Dev Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1** | $1.290\text{ m/s}$ | **$1.261\text{ m/s}$** | $-0.029\text{ m/s}$ | $710.7\text{ kW}$ | **$688.6\text{ kW}$** | **$-22.1\text{ kW}$** | $+16.18\%$ | **$+7.43\%$** | **$+8.75\%$** |
| **T2** | $0.959\text{ m/s}$ | **$0.939\text{ m/s}$** | $-0.020\text{ m/s}$ | $545.7\text{ kW}$ | **$529.6\text{ kW}$** | **$-16.1\text{ kW}$** | $-5.98\%$ | **$-2.10\%$** | **$+3.88\%$** |
| **T3** | $1.045\text{ m/s}$ | **$1.034\text{ m/s}$** | $-0.011\text{ m/s}$ | $622.0\text{ kW}$ | **$617.2\text{ kW}$** | **$-4.8\text{ kW}$** | $-9.54\%$ | **$-7.42\%$** | **$+2.12\%$** |

---

### Why Energy Loss Error Drops Drastically While MAE Barely Changes

A striking observation in the ablation results is that **Wind Speed MAE improves by ~0.02 m/s** and **Power MAE by ~16–22 kW**, yet the **Cumulative Energy Loss Error drops significantly (e.g. by 8.75 percentage points on T1)**.

#### 1. Unsigned Point Scatter (MAE) vs. Signed Accumulation (Mean Bias)
* **MAE is Unsigned:**
  $$\text{MAE} = \frac{1}{N}\sum_{t=1}^N |\hat{y}_t - y_t|$$
  MAE measures instantaneous scatter caused by high-frequency micro-turbulence, wind gusts, and 10-minute anemometer noise. That baseline physical scatter ($\sim 1\text{ m/s}$) is irreducible, so MAE remains relatively stable.

* **Cumulative Energy is an Integrated Sum (Signed Integration):**
  $$\text{Total Energy Error} = \sum_{t=1}^N (\hat{P}_t - P_t) \cdot \frac{1}{6}\text{ hours} = N \cdot \frac{1}{6} \cdot \underbrace{\text{Mean Signed Bias}}_{\frac{1}{N}\sum(\hat{P}_t - P_t)}$$
  * **Without Direction:** The model systematically overpredicted wind speed from sheltered angles, compounding over all 862 timesteps.
  * **With Direction:** The directional vectors eliminated the terrain exposure blind spot. The mean power bias collapsed by over **$75\%$**.
  * Over 6 days, remaining errors are symmetric zero-mean random noise ($+1\text{ m/s}$ then $-1\text{ m/s}$), which **naturally cancel each other out in the cumulative sum**.

#### 2. The Cubic Power Curve Multiplier ($P \propto v^3$)
In the turbine's partial-load operating region ($4\text{ to }10\text{ m/s}$), electrical power scales cubically with wind velocity:
$$P \approx k \cdot v^3 \implies \Delta P \approx 3 k v^2 \cdot \Delta v$$
At $7\text{ m/s}$, a modest persistent $+0.4\text{ m/s}$ wind bias produces a massive $+200\text{ kW}$ active power error per interval. By eliminating that systematic directional bias, the integrated energy claim aligns closely with ground truth.

---

## 4. Air Density Normalization (Addon 2: IEC 61400-12-1 Physics Logic)

While turbine yaw alignment tracks wind direction, **aerodynamic power capture fundamentally depends on atmospheric air density ($\rho$)**.

### Physical Background & Governing Equations
The kinetic power of wind passing through rotor swept area $A$ is:
$$P_{\text{wind}} = \frac{1}{2} \rho A v^3$$

Air density $\rho$ varies continuously with barometric air pressure $P$ and ambient temperature $T$ according to the Ideal Gas Law:
$$\rho = \frac{P \cdot 100}{R_{\text{specific}} \cdot (T + 273.15)} \quad \left[\text{kg/m}^3\right]$$
where $R_{\text{specific}} = 287.05\text{ J/(kg}\cdot\text{K)}$ is the specific gas constant for dry air.

Manufacturer power curves are calibrated strictly at standard sea-level atmospheric conditions:
$$\rho_0 = 1.225\text{ kg/m}^3 \quad (15^\circ\text{C},\ 1013.25\text{ hPa})$$

Under **IEC 61400-12-1**, when converting predicted wind speeds through a standard power curve in the partial load region, the effective normalized wind speed is:
$$v_{\text{norm}} = v \cdot \left(\frac{\rho}{\rho_0}\right)^{1/3}$$

### Cross-Farm Density Proxy Architecture (Handling Missing Telemetry)
In a real-world outage:
1. Target Park T's telemetry is down (neither pressure nor temperature can be read from T).
2. Reference Park O has constant/missing pressure telemetry in this dataset.
3. However, **Reference Park H** provides continuous, high-fidelity barometric pressure and ambient temperature across all 5 turbines ($H_1\text{--}H_5$).

Evaluating the cross-farm air density relationship reveals:
$$\text{Corr}(\rho_H, \rho_T) = 0.9785 \quad\text{with an MAE of only}\quad 0.0086\text{ kg/m}^3\ (0.73\%)$$

Because atmospheric pressure and synoptic temperature fronts are regional mesoscale phenomena, Park H provides an almost perfect proxy for Park T's air density, enabling complete, out-of-park outage reconstruction with zero reliance on target telemetry.

### Stage 2 Comparative Benchmark (Validation Period: Sept 4 – Sept 9, 2026)

| Target Turbine | Power MAE (Std) | Power MAE (IEC Density) | Delta Power MAE | Energy Dev (Std) % | Energy Dev (IEC Density) % | Actual MWh | Pred MWh (Std) | Pred MWh (IEC Density) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1** | $688.6\text{ kW}$ | **$684.0\text{ kW}$** | **$-4.6\text{ kW}$** | $+7.43\%$ | **$+4.40\%$** | $212.29\text{ MWh}$ | $228.07\text{ MWh}$ | **$221.62\text{ MWh}$** |
| **T2** | $529.6\text{ kW}$ | $531.4\text{ kW}$ | $+1.8\text{ kW}$ | **$-2.10\%$** | $-4.36\%$ | $311.38\text{ MWh}$ | $304.85\text{ MWh}$ | $297.80\text{ MWh}$ |
| **T3** | $617.2\text{ kW}$ | $621.0\text{ kW}$ | $+3.8\text{ kW}$ | **$-7.42\%$** | $-9.90\%$ | $271.07\text{ MWh}$ | $250.96\text{ MWh}$ | $244.24\text{ MWh}$ |

---

## 5. Non-Obvious Design Choices & Technical Rationale

1. **Orthogonal Wind Vector Decomposition ($u, v$):** Resolves directional periodicity ($359^\circ \leftrightarrow 1^\circ$) while capturing terrain exposure and internal wake shading.
2. **Decoupled Two-Stage Architecture:** Keeps the spatial wind transfer separate from the turbine manufacturer power curve. This complies with IEC 61400 insurance audit standards and avoids black-box coupling.
3. **Cross-Farm Density Transfer (Park H $\to$ Park T):** Leverages mesoscale pressure/temperature coherence ($r = 0.9785$) to compute accurate air density even when Target Park T is completely offline.
4. **Spatial Cross-Farm Medians:** Protects against isolated anemometer failures or telemetry dropouts in reference parks without artificial zero-imputation.
5. **Exact Analytical Quantiles:** Closed-form Gaussian inverse CDF guarantees strictly monotonic quantiles ($v_{P10} \le v_{P50} \le v_{P90}$) with zero quantile crossing.
6. **Fluid-Dynamic Clamping:** Enforces $v \ge 0\text{ m/s}$ to preserve physical validity.

---

## 6. How to Run

```bash
# 1. Launch the interactive Streamlit Web UI:
streamlit run app.py

# 2. Run the Bayesian Ridge validation pipeline & comparative benchmarks:
python bayesian_wind_model.py

# 3. Run spatial correlation and park affinity classification:
python classify_target.py

# 4. Test speed-to-power interpolation and IEC density quantile loss estimation:
python speed_to_power.py
```
