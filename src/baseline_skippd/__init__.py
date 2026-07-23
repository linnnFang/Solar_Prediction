"""
SKIPP'D short-term (15-min) PV nowcasting pipeline.

Dataset-specific code for the SKIPP'D benchmark, kept out of the reusable
``vanilla_transformer`` package. Built in staged phases.
"""

from src.baseline_skippd.io import SKIPPDProcessedStore
from src.baseline_skippd.schema import SchemaError
from src.baseline_skippd.split import SplitManifest, make_split
from src.baseline_skippd.windows import WindowIndex, build_window_index
from src.baseline_skippd.scalers import (
    build_scaler, IdentityPVScaler, CapacityPVScaler, StandardPVScaler)
from src.baseline_skippd.datasets import CanonicalSKIPPDDataset
from src.baseline_skippd.datamodule import SKIPPDDataModule

__all__ = [
    "SKIPPDProcessedStore", "SchemaError",
    "SplitManifest", "make_split",
    "WindowIndex", "build_window_index",
    "build_scaler", "IdentityPVScaler", "CapacityPVScaler", "StandardPVScaler",
    "CanonicalSKIPPDDataset", "SKIPPDDataModule",
]
