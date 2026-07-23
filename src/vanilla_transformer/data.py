"""
Sliding-window dataset for neural time-series forecasting.

Dataset-agnostic building blocks that `src/helper` does not cover:
  - `Standardizer`   : fit input scaling on training frames only (no leakage).
  - `WindowDataset`  : turn a list of contiguous frames into (input, target)
                       windows of shape ([L, n_features], [H]).
  - `make_dataloaders`: wrap {split: WindowDataset} in DataLoaders.

Loading, feature engineering and the leakage-safe time split live in
`src/helper` (`GEFComTask15`, `FeatureBuilder`, `time_split`). Split the data by
time *first*, then pass the resulting per-frame DataFrames here.

First-version split policy (deliberately conservative): every split builds
windows only inside its own frames. No history is stitched across split
boundaries, so the first `L` timesteps of each frame are never predicted. This
guarantees windows never cross a split and keeps the leakage story trivial;
optional carried-over context can be added later.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Standardizer:
    """
    Feature-wise standardizer, fit on training frames only.

    Fit once on the concatenated train frames, then reuse the same instance on
    val/test so no future statistics leak into the inputs. The feature column
    order is stored and re-checked at transform time.
    """

    def __init__(self, mean, std, columns, min_std=1e-8):
        self.columns = list(columns)
        self.mean = np.asarray(mean, np.float32)
        std = np.asarray(std, np.float32)
        # Protect against constant / near-constant columns: a ~0 std would blow
        # up the scaled values, so treat those columns as unit-scale.
        self.std = np.where(std < min_std, 1.0, std).astype(np.float32)

    @classmethod
    def fit_frames(cls, frames, feature_cols, min_std=1e-8):
        """
        Fit mean/std over the concatenation of `frames`.
        Input : frames = list of training DataFrames; feature_cols = input columns.
        Output: a fitted Standardizer that remembers `feature_cols`' order.
        """
        stack = np.concatenate([f[feature_cols].to_numpy(np.float32) for f in frames])
        return cls(stack.mean(0), stack.std(0), feature_cols, min_std)

    def transform(self, arr, columns=None):
        """
        Standardize an array shaped [..., n_features].
        If `columns` is given it must match the fitted column order exactly.
        """
        if columns is not None and list(columns) != self.columns:
            raise ValueError(
                f"column mismatch: standardizer was fit on {self.columns}, "
                f"got {list(columns)}")
        return ((np.asarray(arr, np.float32) - self.mean) / self.std).astype(np.float32)


class WindowDataset(Dataset):
    """
    Map contiguous frames to standardized-input / raw-target windows.

    Data contract for `frames`: each frame is a DataFrame that is already sorted
    by time, contiguous in time, and independent of the others (e.g. one zone's
    train slice). Windows never cross a frame boundary.

    Each item is (x, y):
      x : standardized inputs, shape [lookback, n_features]  (float32 tensor)
      y : raw target,          shape [horizon]               (float32 tensor)
    """

    def __init__(self, feats, targets, lookback, horizon):
        """Low-level constructor; prefer `from_frames`. `feats`/`targets` are
        parallel lists of per-frame arrays ([T_i, n_features] and [T_i])."""
        if lookback <= 0 or horizon <= 0:
            raise ValueError(f"lookback and horizon must be > 0, got L={lookback}, H={horizon}")
        self.lookback, self.horizon = lookback, horizon
        self.feats = [np.asarray(f, np.float32) for f in feats]
        self.targets = [np.asarray(t, np.float32) for t in targets]
        self.index = [
            (sid, t)
            for sid, f in enumerate(self.feats)
            for t in range(lookback, len(f) - horizon + 1)
        ]

    @classmethod
    def from_frames(cls, frames, feature_cols, target_col, lookback, horizon,
                    standardizer=None):
        """
        Build a WindowDataset from a list of DataFrame frames.
        Input : frames        = list of contiguous, time-sorted DataFrames
                                 (e.g. the slices returned by `time_split`);
                feature_cols   = input columns (standardized);
                target_col     = column to forecast (kept in raw units);
                lookback, horizon;
                standardizer   = a fitted Standardizer (fit on train, reuse on
                                 val/test). None means no scaling.
        Frames shorter than lookback + horizon are skipped (too short to yield a
        window). Raises if a required column is missing or contains NaN.
        """
        feats, targets = [], []
        for i, f in enumerate(frames):
            missing = [c for c in list(feature_cols) + [target_col] if c not in f.columns]
            if missing:
                raise ValueError(f"frame {i} is missing columns {missing}")
            if len(f) < lookback + horizon:
                continue  # too short to form even one window
            x = f[feature_cols].to_numpy(np.float32)
            y = f[target_col].to_numpy(np.float32)
            if np.isnan(x).any() or np.isnan(y).any():
                raise ValueError(f"frame {i} contains NaN in feature/target columns")
            feats.append(standardizer.transform(x, feature_cols) if standardizer is not None else x)
            targets.append(y)
        if not feats:
            raise ValueError("no frame was long enough to build a single window")
        return cls(feats, targets, lookback, horizon)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sid, t = self.index[i]
        x = self.feats[sid][t - self.lookback:t]   # [L, n_features]
        y = self.targets[sid][t:t + self.horizon]  # [H]
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def n_features(self):
        return self.feats[0].shape[1]


def make_dataloaders(datasets, batch_size=64, num_workers=0, seed=None, device=None):
    """
    Wrap a {split: WindowDataset} mapping in DataLoaders.
    Behaviour is fixed: train is shuffled, val/test are not, and drop_last is
    always False so every window is evaluated.
    Input : datasets = dict like {"train": ds, "val": ds, "test": ds};
            batch_size, num_workers; seed (seeds the train shuffle for
            reproducibility); device ("cuda"/... — enables pin_memory on CUDA).
    Output: dict of DataLoaders keyed by the same split names.
    """
    pin = device is not None and str(device).startswith("cuda")
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    loaders = {}
    for name, ds in datasets.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=is_train, drop_last=False,
            num_workers=num_workers, pin_memory=pin,
            generator=generator if is_train else None,
        )
    return loaders
