"""
PV power scalers for SKIPP'D.

A scaler maps power in kW to the model's training space and back. It is **fit on
the training split only**; validation/test only ``transform``. Predictions are
``inverse_transform``-ed back to kW before any metric is computed. Scalers are
tiny JSON artifacts (``save`` / ``load``) so a run is fully reproducible.

Operations are backend-agnostic (numpy arrays or torch tensors) — they are plain
arithmetic, so ``transform``/``inverse_transform`` work on either.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class PVScaler(ABC):
    name = "base"

    @abstractmethod
    def fit(self, pv_kw):
        """Fit on 1-D training power values (kW); return self."""

    @abstractmethod
    def transform(self, x):
        """kW -> scaled."""

    @abstractmethod
    def inverse_transform(self, x):
        """scaled -> kW."""

    @abstractmethod
    def to_dict(self):
        """Serializable parameters (includes ``name``)."""

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return Path(path)

    @staticmethod
    def load(path):
        return scaler_from_dict(json.loads(Path(path).read_text()))


class IdentityPVScaler(PVScaler):
    """No-op: model trains directly in kW."""
    name = "identity"

    def fit(self, pv_kw):
        return self

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def to_dict(self):
        return {"name": self.name}


class CapacityPVScaler(PVScaler):
    """Normalise by nameplate capacity -> roughly [0, 1]. No fitting needed."""
    name = "capacity"

    def __init__(self, capacity_kw=30.1):
        self.capacity_kw = float(capacity_kw)

    def fit(self, pv_kw):
        return self

    def transform(self, x):
        return x / self.capacity_kw

    def inverse_transform(self, x):
        return x * self.capacity_kw

    def to_dict(self):
        return {"name": self.name, "capacity_kw": self.capacity_kw}


class StandardPVScaler(PVScaler):
    """Zero-mean / unit-variance using **train** statistics."""
    name = "standard"

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, pv_kw):
        pv = np.asarray(pv_kw, dtype=np.float64)
        self.mean = float(pv.mean())
        self.std = float(pv.std()) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

    def to_dict(self):
        return {"name": self.name, "mean": self.mean, "std": self.std}


_REGISTRY = {c.name: c for c in (IdentityPVScaler, CapacityPVScaler, StandardPVScaler)}


def build_scaler(name, **params):
    """Create a scaler by name (``identity`` / ``capacity`` / ``standard``)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown scaler '{name}', choose from {list(_REGISTRY)}")
    return _REGISTRY[name](**params)


def scaler_from_dict(d):
    """Reconstruct a (possibly fitted) scaler from its ``to_dict`` output."""
    params = {k: v for k, v in d.items() if k != "name"}
    return build_scaler(d["name"], **params)
