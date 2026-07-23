"""
SKIPPDDataModule — the single data interface every model shares.

Holds one store + one split manifest + one window index + one scaler, and hands
out DataLoaders over the canonical dataset. Guarantees that CNN, MLP, and the
vanilla Transformer all see the *same* windows, split, and scaler:

- the PV scaler is fit on the **train** windows only (in ``setup``);
- train is shuffled, val/test/predict are not (aligned, reproducible eval);
- ``drop_last=False`` everywhere (every window is used / evaluated);
- everything is worker-safe (store handles are per-process lazy).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baseline_skippd.datasets import CanonicalSKIPPDDataset


class SKIPPDDataModule:
    def __init__(self, store, split_manifest, window_index, scaler,
                 batch_size=256, num_workers=0, seed=42, device=None, predict_split="test"):
        self.store = store
        self.split_manifest = split_manifest
        self.window_index = window_index
        self.scaler = scaler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.predict_split = predict_split
        self._pin = device is not None and str(device).startswith("cuda")
        self._datasets = {}

    def setup(self):
        """Fit the scaler on TRAIN windows only, then build the per-split datasets."""
        train_wi = self.window_index.subset("train")
        rows = np.unique(np.concatenate(
            [train_wi.history_row_indices.reshape(-1), train_wi.target_row_index]))
        self.scaler.fit(self.store.pv("trainval")[rows])       # train power only -> no leakage
        for split in ("train", "validation", "test"):
            self._datasets[split] = CanonicalSKIPPDDataset(
                self.store, self.window_index.subset(split), self.scaler)
        return self

    def _loader(self, split, shuffle):
        if split not in self._datasets:
            raise RuntimeError("call setup() before requesting a dataloader")
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        return DataLoader(
            self._datasets[split], batch_size=self.batch_size,
            shuffle=shuffle, drop_last=False, num_workers=self.num_workers,
            pin_memory=self._pin, generator=generator)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("validation", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)

    def predict_dataloader(self):
        return self._loader(self.predict_split, shuffle=False)

    def dataset(self, split):
        return self._datasets[split]

    def sample_schema(self):
        """Static description of the canonical sample (shapes/dtypes)."""
        h, w, c = self.store.image_shape                       # (H, W, C)
        t = self.window_index.T
        return {
            "images": {"dtype": "float32", "shape": (t, c, h, w)},
            "pv_history": {"dtype": "float32", "shape": (t,)},
            "target": {"dtype": "float32", "shape": (1,)},
            "issue_time": {"dtype": "int64", "shape": ()},
            "target_time": {"dtype": "int64", "shape": ()},
            "split": {"dtype": "str", "shape": ()},
            "sample_index": {"dtype": "int64", "shape": ()},
        }
