"""
Rule-based (training-free) baselines + a lightweight PV frame extractor.

``pv_window_frame`` pulls everything the rule baselines need for one split
*without decoding images* (just power + timestamps), so persistence and
clear-sky evaluate over the full split in a fraction of a second.
"""

import numpy as np
import pandas as pd

from src.baseline_skippd.baselines.persistence import NaivePersistence
from src.baseline_skippd.baselines.clear_sky import ClearSkyModel, ClearSkyAdjustedPersistence

__all__ = ["NaivePersistence", "ClearSkyModel", "ClearSkyAdjustedPersistence", "pv_window_frame",
           "local_minute_of_day"]


def local_minute_of_day(times_ns, tz="US/Pacific"):
    """int64 ns (UTC) -> minute of local day (0..1439)."""
    local = pd.to_datetime(np.asarray(times_ns), utc=True).tz_convert(tz)
    return (local.hour * 60 + local.minute).to_numpy()


def pv_window_frame(store, window_index, split):
    """Image-free per-window arrays for one split (used by the rule baselines)."""
    wi = window_index.subset(split)
    group = wi.group_name(0) if len(wi) else "trainval"
    pv = store.pv(group)
    return {
        "sample_index": np.arange(len(wi)),
        "issue_time": wi.issue_time,
        "target_time": wi.target_time,
        "split": split,
        "p_now_kw": pv[wi.issue_row_index].astype(np.float32),      # power at t
        "y_true_kw": pv[wi.target_row_index].astype(np.float32),    # power at t+15
        "issue_minute": local_minute_of_day(wi.issue_time),
        "target_minute": local_minute_of_day(wi.target_time),
    }
