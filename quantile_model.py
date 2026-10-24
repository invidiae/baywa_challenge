"""LightGBM P90/P50/P10 counterfactual power forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


class QuantileLossPredictor:
    """Train separate quantile regressors; NaN features remain native missing values."""

    def __init__(self, quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)) -> None:
        if tuple(sorted(quantiles)) != (0.1, 0.5, 0.9):
            raise ValueError("quantiles must be 0.1, 0.5, and 0.9.")
        self.quantiles = quantiles
        self.models: dict[float, object] = {}
        self.feature_names: list[str] = []
        self.rated_power_kw: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileLossPredictor":
        """Fit 150-estimator LightGBM quantile models for SCADA features."""
        from lightgbm import LGBMRegressor
        X = X.apply(pd.to_numeric, errors="raise").astype(float)
        y = pd.to_numeric(y, errors="raise").astype(float)
        valid = y.notna() & np.isfinite(y)
        if len(X) != len(y) or valid.sum() < 50:
            raise ValueError("Need equal-sized inputs and at least 50 valid targets.")
        self.feature_names, self.rated_power_kw = list(X.columns), float(y[valid].max())
        self.models = {}
        for q in self.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, n_estimators=150, learning_rate=0.03,
                                  max_depth=6, num_leaves=31, min_child_samples=30, colsample_bytree=0.9,
                                  reg_lambda=1.0, random_state=42, n_jobs=1, verbosity=-1)
            model.fit(X.loc[valid, self.feature_names], y.loc[valid])
            self.models[q] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ordered conservative P90, median P50, and optimistic P10 yields."""
        if not self.models or self.rated_power_kw is None:
            raise RuntimeError("Call fit first.")
        if set(X.columns) != set(self.feature_names):
            raise ValueError("Prediction features must match training features.")
        values = np.sort(np.column_stack([self.models[q].predict(X[self.feature_names]) for q in (0.1, 0.5, 0.9)]), axis=1)
        result = pd.DataFrame(values, columns=["P90_yield", "P50_yield", "P10_yield"], index=X.index)
        return result.clip(lower=0.0, upper=self.rated_power_kw)
