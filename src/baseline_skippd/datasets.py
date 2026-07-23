"""
Canonical SKIPP'D dataset — one fixed sample schema shared by every model.

``__getitem__`` ALWAYS returns the same dict, regardless of which model will
consume it. Model-specific reshaping (CNN channel stacking, Transformer feature
projection) is the adapters' job, never the dataset's::

    {
        "images":       FloatTensor [T, C, H, W]   (uint8 RGB -> /255, no augmentation)
        "pv_history":   FloatTensor [T]            (scaled)
        "target":       FloatTensor [1]            (scaled)
        "issue_time":   int64  (ns, UTC)
        "target_time":  int64  (ns, UTC)
        "split":        str
        "sample_index": int    (position within this split's window subset)
    }

T = 16 (t-15 .. t, oldest->newest). The dataset reads through the worker-safe
store, so it is safe under multi-worker DataLoaders.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.baseline_skippd.windows import SPLIT_NAMES


class CanonicalSKIPPDDataset(Dataset):
    """Reads one split's windows into the canonical sample dict."""

    def __init__(self, store, window_index, scaler):
        """``window_index`` should already be subset to a single split."""
        self.store = store
        self.wi = window_index
        self.scaler = scaler

    def __len__(self):
        return len(self.wi)

    def __getitem__(self, i):
        wi = self.wi
        group = wi.group_name(i)
        hist_rows = wi.history_row_indices[i]              # [T] int
        target_row = int(wi.target_row_index[i])

        imgs = self.store.read_images(group, hist_rows)    # [T, H, W, C] uint8
        imgs = (torch.from_numpy(imgs).to(torch.float32).div_(255.0)
                .permute(0, 3, 1, 2).contiguous())         # [T, C, H, W]

        pv = self.store.pv(group)
        pv_hist = np.asarray(self.scaler.transform(pv[hist_rows].astype(np.float32)), dtype=np.float32)
        target = np.float32(self.scaler.transform(np.float32(pv[target_row])))

        return {
            "images": imgs,
            "pv_history": torch.from_numpy(pv_hist),                      # [T]
            "target": torch.tensor([target], dtype=torch.float32),       # [1]
            "issue_time": torch.tensor(int(wi.issue_time[i]), dtype=torch.int64),
            "target_time": torch.tensor(int(wi.target_time[i]), dtype=torch.int64),
            "split": SPLIT_NAMES[int(wi.split[i])],
            "sample_index": i,
        }
