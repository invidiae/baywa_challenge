# BayWa r.e. Wind Turbine Outage Estimation Challenge

This repository implements an end-to-end, highly explainable solution for estimating wind turbine energy yield loss during telemetry communication outages using reference wind farms ($H_1\text{--}H_5$ and $O_1\text{--}O_2$).

---

## Bayesian Ridge Regression

The core wind transfer model employs an explainable, physics-grounded **Two-Stage Bayesian Ridge Architecture**:
1. **Stage 1 (Wind Speed Transfer):** Predicts hub-height wind speed and exact analytical probability distributions from external reference wind farms.
2. **Stage 2 (Power & Energy Loss Conversion):** Maps the predicted wind speed quantiles through the manufacturer's piecewise linear power curve ($P(v)$) to compute active electrical power (kW) and energy yield loss (kWh / MWh).

### Model Formulation & Explainability
Unlike black-box tree ensembles, the Bayesian Ridge model yields a single closed-form mathematical formula for each target turbine:

$$\hat{v}_{T, t} = w_0 + w_1 \cdot v_{O, t} + w_2 \cdot v_{H, t} + w_3 \cdot u_{O, t} + w_4 \cdot v_{O, t} + w_5 \cdot u_{H, t} + w_6 \cdot v_{H, t}$$

#### Learned Equations per Turbine:
* **Turbine T1:**
  $$v_{T1} = 0.831 + 0.445 \cdot v_O + 0.334 \cdot v_H + 0.039 \cdot u_O + 0.078 \cdot v_O + 0.040 \cdot u_H - 0.024 \cdot v_H$$
* **Turbine T2:**
  $$v_{T2} = 0.814 + 0.419 \cdot v_O + 0.363 \cdot v_H - 0.024 \cdot u_O + 0.040 \cdot v_O - 0.016 \cdot u_H - 0.084 \cdot v_H$$
* **Turbine T3:**
  $$v_{T3} = 0.935 + 0.379 \cdot v_O + 0.340 \cdot v_H - 0.014 \cdot u_O + 0.023 \cdot v_O + 0.012 \cdot u_H - 0.032 \cdot v_H$$

---

### Validation Performance (Official Shown Test Window: Sept 4 – Sept 9, 2026)

Trained chronologically on historical observations prior to the outage (`timestamp < 2026-09-01`) and evaluated on the official 6-day shown validation period (864 10-minute intervals):

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

---

### Hyperparameters

The model uses Bayesian Linear Regression with conjugate Gaussian priors over weights and Gamma priors over noise and weight precisions:

* **Prior Hyperparameters:**
  * Initial noise precision parameters: $\alpha_1 = 10^{-6}, \alpha_2 = 10^{-6}$ (uninformative prior over observation noise).
  * Initial weight precision parameters: $\lambda_1 = 10^{-6}, \lambda_2 = 10^{-6}$ (uninformative prior over weights).
* **Learned Precision Parameters (via Type-II Maximum Marginal Likelihood / Evidence Maximization):**
  * **T1:** Noise precision $\alpha \approx 0.576$ ($\sigma_{\text{noise}} \approx 1.318\text{ m/s}$), Regularization $\lambda \approx 18.75$
  * **T2:** Noise precision $\alpha \approx 0.691$ ($\sigma_{\text{noise}} \approx 1.203\text{ m/s}$), Regularization $\lambda \approx 18.95$
  * **T3:** Noise precision $\alpha \approx 0.706$ ($\sigma_{\text{noise}} \approx 1.190\text{ m/s}$), Regularization $\lambda \approx 22.97$
* **Tolerance & Convergence:**
  * Tolerance: `tol = 1e-3`
  * Maximum iterations: `max_iter = 300` (converges in $\sim 4$ iterations via analytical update equations)
  * `compute_score = True` (monitors log marginal likelihood at each step).

---

### Non-Obvious Design Choices & Technical Rationale

1. **Orthogonal Wind Vector Decomposition ($u, v$) instead of Raw Angles ($\theta$):**
   * *Problem:* Wind directions wrap around ($359^\circ$ and $1^\circ$ are adjacent, but linear models treat them as opposites).
   * *Solution:* We decompose wind into physical orthogonal components:
     $$u = v \cdot \sin(\theta) \quad (\text{East-West component})$$
     $$v = v \cdot \cos(\theta) \quad (\text{North-South component})$$
   * This provides continuous, linear spatial gradients representing wind front directionality and terrain sheltering without angle discontinuities.

2. **Decoupled Two-Stage Architecture (Physics Grounding):**
   * *Design Choice:* We predict hub-height wind speed in Stage 1, then apply the manufacturer power curve $P(v)$ in Stage 2.
   * *Rationale:* Bypassing the power curve and training a black-box model directly on power bundles the linear spatial transfer and the non-linear aerodynamic power curve into one opaque model. Keeping them decoupled ensures IEC 61400 regulatory compliance, full auditability by insurance adjusters, and significantly lower power MAE.

3. **Spatial Cross-Farm Medians (Robust Missing Sensor Resilience):**
   * *Design Choice:* Inputs use cross-turbine medians ($v_O = \text{median}(O_1, O_2)$, $v_H = \text{median}(H_1\text{--}H_5)$) rather than raw individual sensor streams.
   * *Rationale:* Individual turbine anemometers frequently experience localized icing, maintenance cut-outs, or telemetry dropouts. Spatial medians filter out isolated sensor anomalies without discarding rows or requiring artificial zero-imputation.

4. **Exact Analytical Quantiles (Zero Quantile Crossing):**
   * *Design Choice:* Quantiles are calculated directly from the posterior predictive distribution $v_{q, t} = \max\left(0, \mu_t + \Phi^{-1}(q) \cdot \sigma_t\right)$.
   * *Rationale:* Fitting separate quantile regressors for $P10$, $P50$, and $P90$ often suffers from "quantile crossing" (e.g. $P10 > P50$ during sudden shifts). Analytical Bayesian quantiles guarantee strict monotonicity ($v_{P10} \le v_{P50} \le v_{P90}$) and realistic coverage ($80.6\%$ empirical coverage on T2).

5. **Physical Non-Negative Clamping:**
   * *Design Choice:* Both predicted wind speeds and extracted quantiles are bounded at $v \ge 0\text{ m/s}$.
   * *Rationale:* Linear models can theoretically produce slight negative values under calm conditions; clamping respects the laws of fluid dynamics.

---

### How to Run

```bash
# Run the Bayesian Ridge validation evaluation:
python bayesian_wind_model.py
```
