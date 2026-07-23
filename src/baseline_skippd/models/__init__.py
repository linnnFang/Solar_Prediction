"""Trainable models for SKIPP'D forecasting."""

from src.baseline_skippd.models.pv_only import PVOnlyMLP
from src.baseline_skippd.models.sunset import SunsetForecastCNN

__all__ = ["PVOnlyMLP", "SunsetForecastCNN"]
