# BayWa r.e. Wind Outage Challenge — 5-Slide Presentation Deck

> **Files available in repo:**
> * **`presentation_slides.html`** — Open in any web browser for a full-screen 16:9 interactive keynote deck (`←` / `→` / `Space` to navigate, `F` for fullscreen, `Ctrl+P` to save as PDF).
> * **`BayWa_Presentation.pptx`** — Native 16:9 widescreen PowerPoint file (editable in PowerPoint, Google Slides, Keynote, or LibreOffice).

---

## Slide 1: Problem & Motivation
### **When a Wind Turbine Goes Dark, How Much Energy Was Lost?**
*Turning missing blackout data into an auditable Business Interruption insurance claim.*

* **1. The Grid Outage:** Cable damage or grid faults disconnect the wind farm. Turbines stop generating electricity and revenue drops to zero.
* **2. The Sensor Blind Spot:** Normally, we calculate lost energy from the turbine's own wind sensor. During a blackout, **that sensor is offline too (`NaN`)**.
* **3. Our Automated Solution:** We learn how neighbouring wind parks (**Park O** & **Park H**) relate to our turbine before the outage, reconstruct the missing wind, and calculate lost **MWh** through the power curve.

> **Bottom Callout:** *72-Hour Fleet Outage Loss (`T1 + T2 + T3`): **283.9 MWh** — Accurate, transparent, and easy for any insurance adjuster to verify.*

---

## Slide 2: Simple, Physical & 100% Explainable Model
### **Why Our Model Is Simple, Physical & 100% Explainable**
*Prioritizing explainability — one clean equation per turbine instead of a black box.*

$$\widehat{\text{Wind Speed}} = w_0 + w_O \cdot \text{Speed}_{\text{Park O}} + w_H \cdot \text{Speed}_{\text{Park H}} + \text{Direction Adjustments }(u, v)$$

* **One Readable Equation (Bayesian Ridge):** Every turbine gets a single linear formula you can check on a calculator — zero hidden layers or opaque rules.
* **Why Aggregate Turbines into Parks (`O1–O2 → Park O`, `H1–H5 → Park H`):**
  * Turbines in the same park feel **>96% identical wind**.
  * Giving each turbine its own weight makes them fight and creates **fake negative weights**.
  * Park medians give clean **~40% / ~40%** positive weights and survive single-sensor dropouts.
* **Why This Beats Other Models:**
  * **vs. LightGBM (450 Trees):** Black box that fails seasonal shifts (`-20%` error) and cannot be audited.
  * **vs. Raw Multilinear Regression:** Suffers from negative collinear weights.
  * **Explainability is key** for insurance sign-off.

---

## Slide 3: Native Bayesian Uncertainties
### **Native Bayesian Uncertainty: Two Bands, Zero Crossing**
*Uncertainty comes directly from probability math — not ad-hoc quantile trees.*

* **Why Native Bayesian Beats Quantile Estimates:**
  * **Quantile Models (e.g. LightGBM):** Train 3 separate models for `P10`, `P50`, and `P90`. Because they don't share math, **curves can cross (`P10 > P50`)** and require manual patching.
  * **Native Bayesian Ridge:** Predicts a complete bell curve (mean `μ` & spread `σ`) at every 10-minute step — percentiles (`50%–99%` slider in our app) **mathematically never cross**.
* **Two Uncertainties Separated Automatically:**
  1. **95% Coefficient CI (Dark Inner Band — `±2.0 MWh`):** *"How sure are we about our formula?"* With `12,400+` training intervals, our formula weights are pinned down extremely tightly.
  2. **80% Predictive Band `P10–P90` (Light Outer Band):** *"How gusty is a single 10-minute step?"* Over a 3-day outage, random 10-minute gusts cancel out — so true energy stays close to **P50**.

---

## Slide 4: Cross-Validated Accuracy Metrics
### **Robust Accuracy via 5-Fold Cross-Validation**
*Tested across all 86 operational days (12,421 intervals) without permanently withholding training data.*

| Target Turbine | 5-Fold Wind MAE | 5-Fold Wind RMSE | 5-Fold Power MAE | 80% Band Coverage (`P10–P90`) | 5-Fold Energy Error | 72h Outage Loss (`P50`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Turbine T1** | `0.990 m/s` | `1.338 m/s` | `492.6 kW` | **`82.4%`** | `-12.78%` | **`75.32 MWh`** |
| **Turbine T2** | **`0.880 m/s`** | **`1.182 m/s`** | `453.4 kW` | **`82.8%`** | **`-8.77%`** | **`117.16 MWh`** |
| **Turbine T3** | `0.893 m/s` | `1.194 m/s` | **`440.2 kW`** | **`83.0%`** | `-11.35%` | **`91.46 MWh`** |

> **Why Background 5-Fold CV?** Every chronological block is evaluated strictly out-of-sample, while the final outage model still trains on **100% of available history** (zero data wasted).

---

## Slide 5: Physics Addons & Bonus Challenges
### **Physics Addons & All 3 Bonus Challenges**
*Going beyond wind speed with atmospheric physics, smart filtering, and instant claim emails.*

* **1. Physics — Wind Direction Vectors (`u, v`) & Air Density (`IEC 61400-12-1`):**
  * **Direction Vectors (`u, v`):** Fixes the `359° ↔ 1°` compass jump and captures hill & wake sheltering.
  * **Air Density Correction:** Uses live **Temperature & Pressure** from Park H (`97.9%` regional match) to adjust blade push for cold dense vs. warm thin air.
* **2. Automatic Reference Selection:**
  * Automatically scans uploaded reference turbines (`O1–O2`, `H1–H5`), removes any turbine that was offline during the outage, and ranks references by correlation.
* **3. One-Click Insurance Claim (`mailto:`):**
  * Generates a clean claim notice with only relevant insurance data (outage times, reference turbines, **Estimated MWh**, and 5-Fold CV accuracy) and **opens directly in the computer's default Mail App**.
