"""
Clear-sky-adjusted persistence (empirical clear-sky, decision B of Phase 0).

We do NOT use pvlib: the clear-sky *power* curve is fit directly from the data as
a high quantile of observed power per local minute-of-day (smoothed), so it is in
kW and self-calibrated to this array — no coordinates, no panel parameters.

Forecast (no future target ever used):
    CSI_t          = P_t / P_clear(minute_of_day(t))
    y_hat(t+15)    = CSI_t * P_clear(minute_of_day(t+15)),  clipped to [0, capacity]
"""

import json
from pathlib import Path

import numpy as np


class ClearSkyModel:
    """P_clear(minute_of_day) fit from training power (a 1440-length curve)."""

    def __init__(self, curve, quantile, smooth):
        self.curve = np.asarray(curve, dtype=np.float64)   # [1440]
        self.quantile = quantile
        self.smooth = smooth

    @classmethod
    def fit(cls, store, split_manifest, window_index, quantile=0.95, smooth=15):
        """Fit on TRAIN windows only (their power + local minute-of-day)."""
        from src.baseline_skippd.baselines import local_minute_of_day
        wi = window_index.subset("train")
        pv = store.pv("trainval")
        # use each train window's issue power at its issue minute (train days only)
        minute = local_minute_of_day(wi.issue_time)
        power = pv[wi.issue_row_index].astype(np.float64)
        curve = np.zeros(1440)
        for m in range(1440):
            vals = power[minute == m]
            curve[m] = np.quantile(vals, quantile) if len(vals) else 0.0
        curve = _smooth_wrap(curve, smooth)
        return cls(curve, quantile, smooth)

    def clear_power(self, minute):
        return self.curve[np.asarray(minute) % 1440]

    def to_dict(self):
        return {"curve": self.curve.tolist(), "quantile": self.quantile, "smooth": self.smooth}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()))
        return Path(path)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["curve"], d["quantile"], d["smooth"])


class ClearSkyAdjustedPersistence:
    name = "clear_sky_persistence"

    def __init__(self, clear_sky_model, capacity_kw=30.1, floor_kw=0.5):
        self.csm = clear_sky_model
        self.capacity_kw = float(capacity_kw)
        self.floor_kw = float(floor_kw)

    def clear_sky_index(self, minute_now, p_now):
        return p_now / np.maximum(self.csm.clear_power(minute_now), self.floor_kw)

    def predict_kw(self, frame):
        csi = self.clear_sky_index(frame["issue_minute"], np.asarray(frame["p_now_kw"], np.float64))
        y = csi * self.csm.clear_power(frame["target_minute"])
        return np.clip(y, 0.0, self.capacity_kw).astype(np.float32)


def _smooth_wrap(curve, k):
    """Circular moving-average smoothing of a day curve."""
    if k <= 1:
        return curve
    pad = np.concatenate([curve[-k:], curve, curve[:k]])
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(pad, kernel, mode="same")[k:-k]
