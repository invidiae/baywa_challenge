# BayWa r.e. Wind Turbine Outage Estimation Challenge

An end-to-end, highly explainable solution for estimating wind turbine energy yield loss during telemetry communication outages using external reference wind farms ($H_1\text{--}H_5$ and $O_1\text{--}O_2$).

---

## 1. Solution Architecture Overview

The solution uses an explainable, physics-grounded **Two-Stage Architecture**:
1. **Stage 1 (Wind Speed Transfer):** Predicts hub-height wind speed and exact analytical probability distributions from external reference wind farms using **Bayesian Ridge Regression** with directional vector components.
2. **Stage 2 (Power & Energy Loss Conversion):** Maps the predicted wind speed quantiles through the manufacturer's piecewise linear power curve ($P(v)$) to compute active electrical power (kW) and energy yield loss (kWh / MWh).

---

## 2. Bayesian Ridge Regression (Stage 1)

Unlike black-box tree ensembles (e.g., LightGBM with >1,000 opaque splits), the Bayesian Ridge model yields a single closed-form mathematical formula for each target turbine:

$$\hat{v}_{T, t} = w_0 + w_1 \cdot v_{O, t} + w_2 \cdot v_{H, t} + w_3 \cdot u_{O, t} + w_4 \cdot v_{O, t} + w_5 \cdot u_{H, t} + w_6 \cdot v_{H, t}$$

### Closed-Form Equations per Turbine
* **Turbine T1:**
  $$v_{T1} = 0.831 + 0.445 \cdot v_O + 0.334 \cdot v_H + 0.039 \cdot u_O + 0.078 \cdot v_O + 0.040 \cdot u_H - 0.024 \cdot v_H$$
* **Turbine T2:**
  $$v_{T2} = 0.814 + 0.419 \cdot v_O + 0.363 \cdot v_H - 0.024 \cdot u_O + 0.040 \cdot v_O - 0.016 \cdot u_H - 0.084 \cdot v_H$$
* **Turbine T3:**
  $$v_{T3} = 0.935 + 0.379 \cdot v_O + 0.340 \cdot v_H - 0.014 \cdot u_O + 0.023 \cdot v_O + 0.012 \cdot u_H - 0.032 \cdot v_H$$

### Validation Performance (Official Shown Test Window: Sept 4 – Sept 9, 2026)

Trained chronologically on pre-outage historical data (`timestamp < 2026-09-01`) and evaluated on the official 6-day shown validation period (864 10-minute intervals):

| Metric | Turbine T1 | Turbine T2 | Turbine T3 | Description & Benchmark Context |
| :--- | :---: | :---: | :---: | :--- |
| **Wind Speed MAE** | $1.255\text{ m/s}$ | **$0.945\text{ m/s}$** | $1.041\text{ m/s}$ | Mean absolute wind speed error |
| **Wind Speed RMSE** | $1.566\text{ m/s}$ | **$1.241\text{ m/s}$** | $1.344\text{ m/s}$ | Root mean squared wind speed error |
| **Wind Speed $R^2$** | $0.625$ | **$0.788$** | $0.710$ | Explains up to $78.8\%$ of wind speed variance |
| **$P10\text{--}P90$ Coverage** | $69.8\%$ | **$80.6\%$** | $77.5\%$ | **Uncertainty Calibration**: Target is $80.0\%$ (T2 is within $0.6\%$) |
| **Power MAE** | $677.3\text{ kW}$ | **$530.7\text{ kW}$** | $620.4\text{ kW}$ | **Beats LightGBM by up to 174 kW per interval** |
| **Power RMSE** | $928.4\text{ kW}$ | **$777.1\text{ kW}$** | $868.2\text{ kW}$ | Root mean squared power prediction error |
| **Actual Energy** | $212.29\text{ MWh}$ | $311.38\text{ MWh}$ | $271.07\text{ MWh}$ | Ground-truth energy generated during validation window |
| **Predicted Energy ($P50$)** | $218.87\text{ MWh}$ | $305.85\text{ MWh}$ | $246.03\text{ MWh}$ | Model-estimated $P50$ counterfactual energy |
| **Energy Yield Deviation** | **$+3.10\%$** | **$-1.78\%$** | **$-9.24\%$** | **Primary Insurance Metric**: Net cumulative claim error |

### Model Hyperparameters
* **Prior Distributions:**
  * Noise precision prior: $\alpha_1 = 10^{-6}, \alpha_2 = 10^{-6}$ (uninformative Gamma prior).
  * Weight precision prior: $\lambda_1 = 10^{-6}, \lambda_2 = 10^{-6}$ (uninformative Gamma prior).
* **Learned Precision Parameters (via Type-II MLE / Evidence Maximization):**
  * **T1:** Noise precision $\alpha \approx 0.576$ ($\sigma_{\text{noise}} \approx 1.318\text{ m/s}$), Regularization $\lambda \approx 18.75$
  * **T2:** Noise precision $\alpha \approx 0.691$ ($\sigma_{\text{noise}} \approx 1.203\text{ m/s}$), Regularization $\lambda \approx 18.95$
  * **T3:** Noise precision $\alpha \approx 0.706$ ($\sigma_{\text{noise}} \approx 1.190\text{ m/s}$), Regularization $\lambda \approx 22.97$
* **Convergence:** `tol = 1e-3`, `max_iter = 300` (converges in $\sim 4$ iterations).

---

## 3. Directional Modeling (Step 1 Addon) & Results

### Physical & Mathematical Formulation
Wind direction cannot be treated as a single scalar angle because angles wrap around: $359^\circ$ and $1^\circ$ are physically adjacent, but arithmetic difference is $358^\circ$.

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
| **T1** | $1.290\text{ m/s}$ | **$1.255\text{ m/s}$** | $-0.035\text{ m/s}$ | $710.7\text{ kW}$ | **$677.3\text{ kW}$** | **$-33.4\text{ kW}$** | $+16.18\%$ | **$+3.10\%$** | **$+13.08\%$** |
| **T2** | $0.959\text{ m/s}$ | **$0.945\text{ m/s}$** | $-0.014\text{ m/s}$ | $545.7\text{ kW}$ | **$530.7\text{ kW}$** | **$-15.0\text{ kW}$** | $-5.98\%$ | **$-1.78\%$** | **$+4.20\%$** |
| **T3** | $1.045\text{ m/s}$ | **$1.041\text{ m/s}$** | $-0.004\text{ m/s}$ | $622.0\text{ kW}$ | **$620.4\text{ kW}$** | **$-1.6\text{ kW}$** | $-9.54\%$ | **$-9.24\%$** | **$+0.30\%$** |

> [!IMPORTANT]
> **Key Finding:** Adding directional vectors consistently improves all metrics across all target turbines. The gain is especially dramatic on **T1**, where energy loss estimation error drops by **$13.08$ percentage points** (from $+16.18\%$ down to $+3.10\%$).

---

## 4. Possible Lead: Step 2 Air Density Normalization (IEC 61400-12-1)

While the wind turbine yaw drive ensures that the power curve $P(v)$ does not depend on wind direction, **aerodynamic power generation does depend on atmospheric air density ($\rho$)**.

### Physical Background
The kinetic power of wind passing through rotor area $A$ is:
$$P_{\text{wind}} = \frac{1}{2} \rho A v^3$$

Air density $\rho$ varies continuously with barometric air pressure $P$ and ambient temperature $T$ according to the Ideal Gas Law:
$$\rho = \frac{P \cdot 100}{R_{\text{specific}} \cdot (T + 273.15)} \quad \left[\text{kg/m}^3\right]$$
where $R_{\text{specific}} = 287.05\text{ J/(kg}\cdot\text{K)}$.

Manufacturer power curves are standardized to sea-level air density:
$$\rho_0 = 1.225\text{ kg/m}^3 \quad (15^\circ\text{C},\ 1013.25\text{ hPa})$$

Under **IEC 61400-12-1**, the effective normalized wind speed entering the power curve in the partial load region is:
$$v_{\text{norm}} = v \cdot \left(\frac{\rho}{\rho_0}\right)^{1/3}$$

### Empirical Impact on Validation Data
During the September 4–9 validation window:
* Average air density was $\bar{\rho} \approx 1.189\text{ kg/m}^3$ (thinner air due to warm late-summer temperatures of $\sim 20^\circ\text{C}$).
* Ratio $\left(\rho / \rho_0\right)^{1/3} \approx 0.990$ (a $\sim 1\%$ effective wind speed derate, leading to $\sim 3\%$ lower power output in the cubic region).
* **Impact on T1:** Applying density normalization reduces the energy yield deviation from **$+3.10\%$ down to just $+0.69\%$**!

### Challenge Caveat & Future Lead
* **Data Constraint:** In this hackathon dataset, barometric pressure is unavailable for Park O (replaced with a constant value).
* **Future Solution:** Regional air pressure can be reconstructed via spatial kriging or transferred from Park H (which has full barometric telemetry, $r > 0.98$ with Park T). Implementing full IEC 61400-12-1 density normalization represents a viable presentation lead and domain extension.

---

## 5. Non-Obvious Design Choices & Technical Rationale

1. **Orthogonal Wind Vector Decomposition ($u, v$):** Resolves directional periodicity ($359^\circ \leftrightarrow 1^\circ$) while capturing terrain exposure and internal wake shading.
2. **Decoupled Two-Stage Architecture:** Keeps the spatial wind transfer separate from the turbine manufacturer power curve. This complies with IEC 61400 insurance audit standards and avoids black-box coupling.
3. **Spatial Cross-Farm Medians:** Protects against isolated anemometer failures or telemetry dropouts in reference parks without artificial zero-imputation.
4. **Exact Analytical Quantiles:** Closed-form Gaussian inverse CDF guarantees strictly monotonic quantiles ($v_{P10} \le v_{P50} \le v_{P90}$) with zero quantile crossing.
5. **Fluid-Dynamic Clamping:** Enforces $v \ge 0\text{ m/s}$ to preserve physical validity.

---

## 6. How to Run

```bash
# Run the Bayesian Ridge validation pipeline & directional ablation study:
python bayesian_wind_model.py

# Run spatial correlation and park affinity classification:
python classify_target.py

# Test speed-to-power interpolation and quantile loss estimation:
python speed_to_power.py
```
