BayWa r.e. detailed Challenge
1. Background & Problem
The communication connection to a wind farm is monitored. If the connection fails, the wind farm may have lost power and stopped operating.
Possible causes include cable damage between the wind farm and the grid connection station or technical faults preventing electricity from being fed into the grid.
Accurately estimating the resulting energy yield losses is important, particularly for business interruption insurance claims.
Normally, energy losses can be calculated using wind speeds measured at the turbine's nacelle and the turbine's power curve. However, during a complete outage, these wind measurements are missing.
Current approach: Wind data from a neighboring wind farm is used to estimate the missing wind speeds and calculate energy yield losses.
Challenge objective: Develop a tool that automates this process.
2. How the Tool Should Work
User inputs:
Wind data: Time series from the reference and affected wind farms (CSV upload).
Power curves: Power curves for target turbines.
Outage period: Automatically detected from missing data, with manual adjustment possible.
Process:
Analyze: Learn the relationship between wind speeds at both wind farms using historical data before the outage.
Predict: Apply this relationship to reference wind data during the outage to estimate missing wind speeds.
Calculate: Use the predicted wind speeds and power curve to calculate the energy yield loss (kWh).
3. Data Provided
All wind data consists of 10-minute average wind speeds measured at hub height.
Training Data
Historical wind speeds from both wind farms covering the same period of up to 30 days.
Teams should split the data chronologically, for example reserving the final days for validation.
Test Data (Hidden)
An outage is simulated by removing actual measurements from the affected wind farm.
Missing values are marked as NaN.
Teams predict the missing values using reference wind farm data.
Predictions are compared against actual measurements.
The test period is not disclosed in advance.
4. Required Deliverables
GUI: A user-friendly interface running locally in a standard web browser, containing:
Accuracy Metrics of your choice
Wind Speed Visualization: Display the predicted wind speed time series.
Prediction Method: Explain the method and its accuracy/uncertainty (e.g., correlation analysis).
Energy Loss Calculation: Calculate the energy yield loss in kWh.
Energy Loss Visualization: Visualize the calculated losses.
Accuracy on test dataset (see Assessment criteria)
Wind speed (RMSE, MAE)
Power (RMSE, MAE)
Deviation of summed energy yield
Assessment criteria
40% model accuracy (using defined metrics)
10% presentation
30% User experience
20% Solution concept
Important: The workflow and methodology must be easy to explain. If there is a trade-off between prediction accuracy and explainability, prioritize explainability.
Bonus Challenges
If time permits:
Insurance Report: Generate an email summarizing the outage for the insurance company.
Additional Parameters: Include temperature, air pressure, or other parameters to improve prediction accuracy.
Automatic Turbine Selection: Tool chooses input automatically from pool of available turbines.
Technical Specifications
File format: CSV
Columns: Timestamp (Berlin/Germany time zone) | Wind speed (m/s)
Measurement interval: 10-minute averages at hub height
 What You Can Win
€1,000 in prize money for the winning team.
