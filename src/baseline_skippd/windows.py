"""
Forecast window index for SKIPP'D — a lightweight index, never a forecast HDF5.

For each issue time ``t`` the forecast task is::

    inputs  : frames + power at  t-15, t-14, ..., t      (T = 16 steps, oldest->newest)
    target  : power at           t + 15

We only store integer row indices + timestamps per window, so the same 16 frames
are never duplicated on disk. A window is kept only if **every** history minute
and the target minute exist *exactly* (no interpolation, no forward fill) and all
its rows fall in the same split. Issue times are sub-sampled by a per-split
stride (train 2 min, val/test 1 min by default).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

NS_PER_MIN = 60 * 1_000_000_000
SPLIT_NAMES = ("train", "validation", "test")
GROUP_NAMES = ("trainval", "test")
DEFAULT_STRIDES = {"train": 2, "validation": 1, "test": 1}


@dataclass
class WindowIndex:
    """Integer index of valid forecast windows (arrays are parallel, length M)."""

    history_row_indices: np.ndarray   # [M, T] int32 — oldest..newest, within `group`
    issue_row_index: np.ndarray       # [M] int32  (== history_row_indices[:, -1])
    target_row_index: np.ndarray      # [M] int32
    issue_time: np.ndarray            # [M] int64 ns (UTC)
    target_time: np.ndarray           # [M] int64 ns (UTC)
    split: np.ndarray                 # [M] int8 codes into SPLIT_NAMES
    group: np.ndarray                 # [M] int8 codes into GROUP_NAMES
    history_minutes: int
    history_step_minutes: int
    horizon_minutes: int

    def __len__(self):
        return int(self.issue_row_index.shape[0])

    @property
    def T(self):
        return int(self.history_row_indices.shape[1])

    def split_mask(self, split):
        return self.split == SPLIT_NAMES.index(split)

    def subset(self, split):
        """A new WindowIndex with only one split's windows."""
        m = self.split_mask(split)
        return WindowIndex(
            self.history_row_indices[m], self.issue_row_index[m], self.target_row_index[m],
            self.issue_time[m], self.target_time[m], self.split[m], self.group[m],
            self.history_minutes, self.history_step_minutes, self.horizon_minutes)

    def group_name(self, i):
        return GROUP_NAMES[int(self.group[i])]

    def to_dataframe(self):
        return pd.DataFrame({
            "issue_row_index": self.issue_row_index,
            "target_row_index": self.target_row_index,
            "issue_time": pd.to_datetime(self.issue_time, utc=True),
            "target_time": pd.to_datetime(self.target_time, utc=True),
            "split": [SPLIT_NAMES[c] for c in self.split],
            "group": [GROUP_NAMES[c] for c in self.group],
        })

    def save(self, path):
        np.savez_compressed(
            path,
            history_row_indices=self.history_row_indices,
            issue_row_index=self.issue_row_index,
            target_row_index=self.target_row_index,
            issue_time=self.issue_time, target_time=self.target_time,
            split=self.split, group=self.group,
            meta=np.array([self.history_minutes, self.history_step_minutes, self.horizon_minutes]))
        return Path(str(path) if str(path).endswith(".npz") else str(path) + ".npz")

    @classmethod
    def load(cls, path):
        z = np.load(path)
        hm, hs, ho = (int(x) for x in z["meta"])
        return cls(z["history_row_indices"], z["issue_row_index"], z["target_row_index"],
                   z["issue_time"], z["target_time"], z["split"], z["group"], hm, hs, ho)


def build_window_index(store, split_manifest, history_minutes=15, history_step_minutes=1,
                       horizon_minutes=15, strides=None):
    """
    Build the forecast WindowIndex + a rejection report.
    Output: (WindowIndex, report dict). Rejection counts partition the candidates.
    """
    strides = {**DEFAULT_STRIDES, **(strides or {})}
    offsets = list(range(-history_minutes, 1, history_step_minutes))   # [-15..0] oldest->newest
    off_ns = np.array(offsets, dtype=np.int64) * NS_PER_MIN
    horizon_ns = horizon_minutes * NS_PER_MIN

    rows = {"history_row_indices": [], "issue_row_index": [], "target_row_index": [],
            "issue_time": [], "target_time": [], "split": [], "group": []}
    counters = {"candidate_issue_times": 0, "valid_windows": 0, "rejected_missing_history": 0,
                "rejected_missing_target": 0, "rejected_split_boundary": 0, "rejected_stride": 0}
    per_split = {s: dict(counters) for s in SPLIT_NAMES}

    for group in GROUP_NAMES:
        ts = store.timestamps(group)                                   # int64 ns, sorted
        ts_to_row = {int(t): i for i, t in enumerate(ts)}
        dates = store.timestamps_index(group).strftime("%Y-%m-%d")
        gcode = GROUP_NAMES.index(group)
        last_ts = {s: None for s in SPLIT_NAMES}

        for i in range(len(ts)):
            split = split_manifest.split_of_date(dates[i])
            if split is None:                                          # date outside the manifest
                continue
            counters["candidate_issue_times"] += 1
            per_split[split]["candidate_issue_times"] += 1
            t = int(ts[i])
            stride_ns = strides[split] * NS_PER_MIN

            if last_ts[split] is not None and (t - last_ts[split]) < stride_ns:
                counters["rejected_stride"] += 1; per_split[split]["rejected_stride"] += 1
                continue
            last_ts[split] = t

            hist = [ts_to_row.get(t + o) for o in off_ns]
            if any(r is None for r in hist):
                counters["rejected_missing_history"] += 1
                per_split[split]["rejected_missing_history"] += 1
                continue
            tgt = ts_to_row.get(t + horizon_ns)
            if tgt is None:
                counters["rejected_missing_target"] += 1
                per_split[split]["rejected_missing_target"] += 1
                continue
            win_splits = {split_manifest.split_of_date(dates[r]) for r in hist}
            win_splits.add(split_manifest.split_of_date(dates[tgt]))
            if win_splits != {split}:
                counters["rejected_split_boundary"] += 1
                per_split[split]["rejected_split_boundary"] += 1
                continue

            rows["history_row_indices"].append(hist)
            rows["issue_row_index"].append(i)
            rows["target_row_index"].append(tgt)
            rows["issue_time"].append(t)
            rows["target_time"].append(t + horizon_ns)
            rows["split"].append(SPLIT_NAMES.index(split))
            rows["group"].append(gcode)
            counters["valid_windows"] += 1
            per_split[split]["valid_windows"] += 1

    M = len(rows["issue_row_index"])
    T = len(offsets)
    index = WindowIndex(
        history_row_indices=np.asarray(rows["history_row_indices"], np.int32).reshape(M, T),
        issue_row_index=np.asarray(rows["issue_row_index"], np.int32),
        target_row_index=np.asarray(rows["target_row_index"], np.int32),
        issue_time=np.asarray(rows["issue_time"], np.int64),
        target_time=np.asarray(rows["target_time"], np.int64),
        split=np.asarray(rows["split"], np.int8),
        group=np.asarray(rows["group"], np.int8),
        history_minutes=history_minutes, history_step_minutes=history_step_minutes,
        horizon_minutes=horizon_minutes)

    report = {**counters, "per_split": per_split,
              "config": {"history_minutes": history_minutes, "history_step_minutes": history_step_minutes,
                         "horizon_minutes": horizon_minutes, "T": T, "strides": strides,
                         "split_config_hash": split_manifest.config_hash}}
    return index, report


def save_report(report, path):
    Path(path).write_text(json.dumps(report, indent=2))
    return Path(path)
