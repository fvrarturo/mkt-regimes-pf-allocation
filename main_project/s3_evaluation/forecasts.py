"""
Conditional regression models and macro forecast simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

MIN_OBSERVATIONS = 60


def _prepare_target(erp: pd.Series) -> pd.Series:
    return erp.shift(-1).rename("target")


def _fit_lasso_model(X: pd.DataFrame, y: pd.Series) -> "LassoConditionalModel":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LassoCV(cv=min(5, len(X) - 1), random_state=42).fit(X_scaled, y)
    return LassoConditionalModel(model=model, scaler=scaler, feature_names=list(X.columns))


@dataclass
class LassoConditionalModel:
    model: LassoCV
    scaler: StandardScaler
    feature_names: List[str]

    def predict_row(self, row: pd.Series) -> float:
        vector = row[self.feature_names].values.reshape(1, -1)
        scaled = self.scaler.transform(vector)
        return float(self.model.predict(scaled)[0])


class TwoByTwoRegimeClassifier:
    def __init__(self, method: str = "median") -> None:
        self.method = method
        self.growth_threshold: float | None = None
        self.inflation_threshold: float | None = None

    def fit(self, growth: pd.Series, inflation: pd.Series) -> None:
        if self.method == "median":
            self.growth_threshold = growth.median()
            self.inflation_threshold = inflation.median()
        elif self.method == "mean":
            self.growth_threshold = growth.mean()
            self.inflation_threshold = inflation.mean()
        else:
            self.growth_threshold = 0.0
            self.inflation_threshold = 0.0

    def predict_value(self, growth: float, inflation: float) -> int:
        if self.growth_threshold is None or self.inflation_threshold is None:
            raise RuntimeError("Regime classifier not fitted.")
        if growth >= self.growth_threshold:
            if inflation >= self.inflation_threshold:
                return 1
            return 0
        else:
            if inflation >= self.inflation_threshold:
                return 2
            return 3

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        series = df.apply(
            lambda row: self.predict_value(row["growth_factor"], row["inflation_factor"]),
            axis=1,
        )
        return series.rename("regime")


class IsolationExtremenessDetector:
    def __init__(self, contamination: float = 0.1) -> None:
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, df: pd.DataFrame) -> pd.Series:
        X = self.scaler.fit_transform(df)
        pred = self.model.fit_predict(X)
        states = (pred == -1).astype(int)
        return pd.Series(states, index=df.index, name="iso_state")

    def predict_row(self, row: pd.Series) -> int:
        X = self.scaler.transform(row.values.reshape(1, -1))
        return int(self.model.predict(X)[0] == -1)


class PCAExtremenessDetector:
    def __init__(self, percentile: float = 0.9) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.percentile = percentile
        self.threshold: float | None = None

    def fit(self, df: pd.DataFrame) -> pd.Series:
        X = self.scaler.fit_transform(df)
        scores = self.pca.fit_transform(X)
        distances = np.linalg.norm(scores, axis=1)
        self.threshold = float(np.quantile(distances, self.percentile))
        states = (distances >= self.threshold).astype(int)
        return pd.Series(states, index=df.index, name="pca_state")

    def predict_row(self, row: pd.Series) -> int:
        if self.threshold is None:
            raise RuntimeError("PCA detector not fitted.")
        X = self.scaler.transform(row.values.reshape(1, -1))
        distance = np.linalg.norm(self.pca.transform(X))
        return int(distance >= self.threshold)


@dataclass
class ConditionalForecaster:
    feature_names: List[str]
    full_model: LassoConditionalModel
    regime_classifier: TwoByTwoRegimeClassifier
    regime_models: Dict[int, LassoConditionalModel]
    iso_detector: IsolationExtremenessDetector
    iso_models: Dict[int, LassoConditionalModel]
    pca_detector: PCAExtremenessDetector
    pca_models: Dict[int, LassoConditionalModel]

    def forecast_all(self, macro_df: pd.DataFrame) -> Dict[str, pd.Series]:
        macro_df = macro_df[self.feature_names].dropna()
        forecasts = {
            "full_regression": self._forecast_full(macro_df),
            "regime_2x2": self._forecast_regime(macro_df),
            "extreme_isolation": self._forecast_extreme(macro_df, self.iso_detector, self.iso_models),
            "extreme_pca": self._forecast_extreme(macro_df, self.pca_detector, self.pca_models),
        }
        return forecasts

    def _forecast_full(self, macro_df: pd.DataFrame) -> pd.Series:
        return macro_df.apply(self.full_model.predict_row, axis=1).rename("full_regression")

    def _forecast_regime(self, macro_df: pd.DataFrame) -> pd.Series:
        def predict(row: pd.Series) -> float:
            regime_id = self.regime_classifier.predict_value(row["growth_factor"], row["inflation_factor"])
            model = self.regime_models.get(regime_id, self.full_model)
            return model.predict_row(row)

        return macro_df.apply(predict, axis=1).rename("regime_2x2")

    def _forecast_extreme(
        self,
        macro_df: pd.DataFrame,
        detector,
        model_dict: Dict[int, LassoConditionalModel],
    ) -> pd.Series:
        def predict(row: pd.Series) -> float:
            state = detector.predict_row(row)
            model = model_dict.get(state, self.full_model)
            return model.predict_row(row)

        name = "extreme_isolation" if isinstance(detector, IsolationExtremenessDetector) else "extreme_pca"
        return macro_df.apply(predict, axis=1).rename(name)


def fit_conditional_forecaster(macro_df: pd.DataFrame, erp: pd.Series) -> ConditionalForecaster:
    feature_names = list(macro_df.columns)
    target = _prepare_target(erp)
    data = macro_df.join(target).dropna()

    full_model = _fit_lasso_model(data[feature_names], data["target"])

    # 2x2 regimes
    if "growth_factor" not in macro_df.columns or "inflation_factor" not in macro_df.columns:
        raise ValueError("Macro features must include growth_factor and inflation_factor for 2x2 regimes.")
    regime_classifier = TwoByTwoRegimeClassifier(method="median")
    regime_classifier.fit(macro_df["growth_factor"], macro_df["inflation_factor"])
    regimes = regime_classifier.predict_series(macro_df).reindex(data.index)

    regime_models: Dict[int, LassoConditionalModel] = {}
    for regime_id in range(4):
        subset = data[regimes == regime_id]
        if len(subset) < MIN_OBSERVATIONS:
            continue
        regime_models[regime_id] = _fit_lasso_model(subset[feature_names], subset["target"])

    # Extremeness detectors & models
    iso_detector = IsolationExtremenessDetector(contamination=0.1)
    iso_states = iso_detector.fit(macro_df).reindex(data.index)
    iso_models = _fit_state_models(data, iso_states, feature_names)

    pca_detector = PCAExtremenessDetector(percentile=0.9)
    pca_states = pca_detector.fit(macro_df).reindex(data.index)
    pca_models = _fit_state_models(data, pca_states, feature_names)

    return ConditionalForecaster(
        feature_names=feature_names,
        full_model=full_model,
        regime_classifier=regime_classifier,
        regime_models=regime_models,
        iso_detector=iso_detector,
        iso_models=iso_models,
        pca_detector=pca_detector,
        pca_models=pca_models,
    )


def _fit_state_models(
    data: pd.DataFrame,
    states: pd.Series,
    feature_names: List[str],
) -> Dict[int, LassoConditionalModel]:
    models: Dict[int, LassoConditionalModel] = {}
    for state in [0, 1]:
        subset = data[states == state]
        if len(subset) < MIN_OBSERVATIONS:
            continue
        models[state] = _fit_lasso_model(subset[feature_names], subset["target"])
    return models


def generate_macro_forecasts(
    macro_df: pd.DataFrame,
    accuracy: float,
    seed: int
) -> pd.DataFrame:
    """
    Simulate macro forecasts with a target correlation ("accuracy") to actual values.
    """
    accuracy = float(np.clip(accuracy, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    means = macro_df.mean()
    stds = macro_df.std(ddof=0).replace(0, 1.0)
    z_actual = (macro_df - means) / stds

    noise = pd.DataFrame(
        rng.standard_normal(size=z_actual.shape),
        index=macro_df.index,
        columns=macro_df.columns,
    )
    scale = np.sqrt(np.maximum(0.0, 1.0 - accuracy ** 2))
    z_forecast = accuracy * z_actual + scale * noise
    forecasts = z_forecast * stds + means
    return forecasts
