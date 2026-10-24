"""Quantile LightGBM forecasts for renewable-energy loss estimation.

The model deliberately learns separate lower, central, and upper conditional
quantiles.  This gives a claim analyst a range of plausible counterfactual
production instead of a single opaque point estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class QuantileLossPredictor:
    """Fit P90, P50 and P10 conditional production forecasts with LightGBM.

    ``P90_yield`` is the conservative lower bound (quantile 0.10), while
    ``P10_yield`` is the optimistic upper bound (quantile 0.90).  The response
    must be power in kW; that convention allows the class to report the
    available headroom to rated power as a diagnostic, not as claim energy.

    LightGBM handles ``numpy.nan`` feature values natively. Do not replace
    missing SCADA data with zero: zero is a physically meaningful wind speed.
    """

    _REQUIRED_QUANTILES = (0.1, 0.5, 0.9)
    _LABELS = {0.1: "P90", 0.5: "P50", 0.9: "P10"}

    def __init__(self, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> None:
        """Create an unfitted predictor with the requested quantiles."""
        quantiles = tuple(float(q) for q in quantiles)
        if tuple(sorted(quantiles)) != self._REQUIRED_QUANTILES:
            raise ValueError("quantiles must contain exactly 0.1, 0.5, and 0.9.")
        self.quantiles = quantiles
        self.models: dict[float, object] = {}
        self.feature_names: list[str] = []
        self.rated_power_kw: float | None = None

    @staticmethod
    def _validate_features(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X must be a non-empty pandas DataFrame.")
        if X.columns.duplicated().any():
            raise ValueError("X contains duplicate feature names.")
        numeric = X.apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("X may not contain infinite values.")
        return numeric

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileLossPredictor":
        """Fit independent LightGBM quantile regressors.

        The defaults are deliberately conservative for 10-minute SCADA data:
        shallow-enough trees, a low learning rate, and minimum leaf support to
        reduce sensitivity to short-lived turbine operating regimes.
        """
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise ImportError("Install lightgbm to train QuantileLossPredictor.") from exc

        features = self._validate_features(X)
        target = pd.to_numeric(y, errors="raise").astype(float)
        if len(features) != len(target):
            raise ValueError("X and y must have the same number of rows.")
        valid = target.notna() & np.isfinite(target)
        if valid.sum() < 50:
            raise ValueError("At least 50 finite target observations are required.")

        self.feature_names = list(features.columns)
        self.rated_power_kw = float(target.loc[valid].max())
        self.models = {}
        for quantile in self.quantiles:
            model = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=150,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                min_child_samples=30,
                colsample_bytree=0.9,
                subsample=0.9,
                reg_lambda=1.0,
                random_state=42,
                # A dashboard request trains three models; one worker avoids
                # oversubscribing constrained laptop/demo environments.
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(features.loc[valid], target.loc[valid])
            self.models[quantile] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict ordered P90/P50/P10 yield and power-loss headroom in kW."""
        if not self.models or self.rated_power_kw is None:
            raise RuntimeError("Call fit before predict_quantiles.")
        features = self._validate_features(X)
        missing = set(self.feature_names) - set(features.columns)
        unexpected = set(features.columns) - set(self.feature_names)
        if missing or unexpected:
            raise ValueError(f"Feature mismatch. Missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        features = features.loc[:, self.feature_names]
        raw = np.column_stack([self.models[q].predict(features) for q in self._REQUIRED_QUANTILES])
        # Separate models can cross. Sorting maintains a coherent uncertainty interval.
        ordered = np.sort(raw, axis=1)
        result = pd.DataFrame(
            ordered,
            columns=["P90_yield", "P50_yield", "P10_yield"],
            index=X.index,
        ).clip(lower=0.0, upper=self.rated_power_kw)
        for label in ("P90", "P50", "P10"):
            result[f"{label}_power_loss_kw"] = (self.rated_power_kw - result[f"{label}_yield"]).clip(lower=0.0)
        return result

    def get_financial_risk_spread(
        self, quantiles_df: pd.DataFrame, ppa_rate: float = 0.08
    ) -> dict[str, float]:
        """Convert 10-minute predicted yield intervals into revenue scenarios.

        The method assumes each row represents one 10-minute SCADA interval.
        Revenue reflects gross production value, before PTC, deductibles, or
        saved costs; apply those policy terms in ``BusinessInterruptionEngine``.
        """
        if ppa_rate < 0:
            raise ValueError("ppa_rate must be non-negative.")
        required = ["P90_yield", "P50_yield", "P10_yield"]
        if missing := set(required) - set(quantiles_df.columns):
            raise ValueError(f"quantiles_df is missing {sorted(missing)}")
        energy_factor = 1.0 / 6.0  # kW measured over ten minutes -> kWh
        revenue = {label: float(quantiles_df[f"{label}_yield"].sum() * energy_factor * ppa_rate)
                   for label in ("P90", "P50", "P10")}
        return {
            "p90_revenue_euro": round(revenue["P90"], 2),
            "p50_revenue_euro": round(revenue["P50"], 2),
            "p10_revenue_euro": round(revenue["P10"], 2),
            "uncertainty_spread_euro": round(revenue["P10"] - revenue["P90"], 2),
        }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 720
    wind_speed = rng.uniform(0, 25, n)
    direction = rng.uniform(0, 360, n)
    power = np.clip(((wind_speed / 12) ** 3) * 3_000 + rng.normal(0, 100, n), 0, 3_000)
    X_demo = pd.DataFrame({
        "reference_wind_speed_mean": wind_speed + rng.normal(0, 0.5, n),
        "wind_direction_sin": np.sin(np.deg2rad(direction)),
        "wind_direction_cos": np.cos(np.deg2rad(direction)),
        "hour": np.arange(n) % 24,
    })
    predictor = QuantileLossPredictor().fit(X_demo.iloc[:600], pd.Series(power[:600]))
    forecast = predictor.predict_quantiles(X_demo.iloc[600:])
    print(forecast.head())
    print(predictor.get_financial_risk_spread(forecast))
