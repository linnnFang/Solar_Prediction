"""
Model wrappers for solar power forecasting.

Every model implements the same `BaseForecaster` interface (fit / predict), so
any of them plugs into the walk-forward backtest without changing the loop.
`build_model` creates one by name. To add a model, register a factory in
MODEL_REGISTRY; estimator libraries are imported lazily so only the model you
use needs to be installed.
"""

from abc import ABC, abstractmethod
import pickle


class BaseForecaster(ABC):
    """Common fit/predict interface shared by every forecasting model."""

    @abstractmethod
    def fit(self, X, y):
        """Train on features X and target y; return self."""

    @abstractmethod
    def predict(self, X):
        """Return predictions for features X."""

    def save(self, path):
        """Pickle this forecaster to path."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load a forecaster previously saved with `save`."""
        with open(path, "rb") as f:
            return pickle.load(f)


class SklearnForecaster(BaseForecaster):
    """Wraps any sklearn-style estimator (XGBoost, LightGBM, RandomForest, linear)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


def _linear(**params):
    from sklearn.linear_model import Ridge
    return Ridge(**params)


def _random_forest(**params):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**params)


def _xgboost(**params):
    from xgboost import XGBRegressor
    return XGBRegressor(**params)


def _lightgbm(**params):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**params)


MODEL_REGISTRY = {
    "linear": _linear,
    "rf": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}


def build_model(name, **params):
    """
    Create a forecaster by name.
    Input : name from MODEL_REGISTRY; params forwarded to the underlying estimator.
    Output: a BaseForecaster ready to fit.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}', choose from {list(MODEL_REGISTRY)}")
    return SklearnForecaster(MODEL_REGISTRY[name](**params))
