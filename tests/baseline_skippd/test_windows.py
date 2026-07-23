"""Phase 3 tests: forecast window index."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conftest import write_shard
from src.baseline_skippd.io import SKIPPDProcessedStore
from src.baseline_skippd.split import make_split
from src.baseline_skippd.windows import NS_PER_MIN, WindowIndex, build_window_index

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "SKIPPD"


@pytest.fixture
def win_store(tmp_path):
    """2 contiguous trainval days (40 min each) + 1 contiguous test day (40 min)."""
    d = tmp_path / "data"
    d.mkdir()
    write_shard(d / "train-00000.parquet", 40, start="2017-06-01 07:00")
    write_shard(d / "train-00001.parquet", 40, start="2017-06-02 07:00")
    write_shard(d / "test-00000.parquet", 40, start="2017-07-01 07:00")
    return SKIPPDProcessedStore(tmp_path)


@pytest.fixture
def built(win_store):
    m = make_split(win_store, strategy="chronological_day_holdout", val_frac=0.5)  # 1 train, 1 val day
    idx, report = build_window_index(win_store, m)
    return win_store, m, idx, report


def test_forecast_input_contains_t(built):
    store, _, idx, _ = built
    assert len(idx) > 0
    # the newest history row IS the issue row, and t itself is part of the input window
    assert np.array_equal(idx.history_row_indices[:, -1], idx.issue_row_index)
    for k in range(len(idx)):
        g = idx.group_name(k)
        ts = store.timestamps(g)[idx.history_row_indices[k]]
        assert idx.issue_time[k] in ts


def test_target_is_t_plus_horizon(built):
    _, _, idx, _ = built
    assert np.all(idx.target_time - idx.issue_time == 15 * NS_PER_MIN)


def test_history_order_oldest_to_newest(built):
    store, _, idx, _ = built
    # history timestamps increase by exactly 1 min, ending at the issue time
    for k in range(len(idx)):
        g = idx.group_name(k)
        ts = store.timestamps(g)[idx.history_row_indices[k]]
        assert np.all(np.diff(ts) == NS_PER_MIN)
        assert ts[-1] == idx.issue_time[k]
        assert ts[0] == idx.issue_time[k] - 15 * NS_PER_MIN


def test_missing_history_rejected(tmp_path):
    # a day with an interior 3-min gap: windows needing the gap are dropped
    d = tmp_path / "data"; d.mkdir()
    t = pd.date_range("2017-06-01 07:00", periods=20, freq="min", tz="US/Pacific")
    t2 = pd.date_range("2017-06-01 07:23", periods=20, freq="min", tz="US/Pacific")  # gap 07:20..07:22
    write_shard(d / "train-00000.parquet", times=t.append(t2))
    write_shard(d / "train-00001.parquet", 40, start="2017-06-02 07:00")
    write_shard(d / "test-00000.parquet", 40, start="2017-07-01 07:00")
    store = SKIPPDProcessedStore(tmp_path)
    m = make_split(store, val_frac=0.5)
    _, report = build_window_index(store, m)
    assert report["rejected_missing_history"] > 0


def test_missing_target_rejected(built):
    # end-of-day issue times whose t+15 doesn't exist must be rejected
    _, _, _, report = built
    assert report["rejected_missing_target"] > 0


def test_no_window_crosses_split(built):
    store, m, idx, _ = built
    names = ["train", "validation", "test"]
    for k in range(len(idx)):
        g = idx.group_name(k)
        rows = list(idx.history_row_indices[k]) + [int(idx.target_row_index[k])]
        dates = store.timestamps_index(g)[rows].strftime("%Y-%m-%d")
        splits = {m.split_of_date(x) for x in dates}
        assert splits == {names[idx.split[k]]}


def test_stride_behavior(built):
    _, _, idx, report = built
    # train stride 2: consecutive train issue times are >= 2 min apart
    for split, min_gap in [("train", 2), ("validation", 1)]:
        t = np.sort(idx.issue_time[idx.split_mask(split)])
        if len(t) > 1:
            assert np.min(np.diff(t)) >= min_gap * NS_PER_MIN
    # counters partition the candidates
    c = report
    assert (c["valid_windows"] + c["rejected_missing_history"] + c["rejected_missing_target"]
            + c["rejected_split_boundary"] + c["rejected_stride"]) == c["candidate_issue_times"]


def test_index_roundtrip(built, tmp_path):
    _, _, idx, _ = built
    p = idx.save(tmp_path / "win.npz")
    loaded = WindowIndex.load(p)
    assert len(loaded) == len(idx)
    assert np.array_equal(loaded.history_row_indices, idx.history_row_indices)
    assert loaded.T == idx.T == 16


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="real SKIPP'D data not present")
def test_real_window_index_smoke():
    store = SKIPPDProcessedStore(REAL_ROOT)
    m = make_split(store)
    idx, report = build_window_index(store, m)
    assert report["valid_windows"] > 0
    assert idx.T == 16
    # every window's history+target actually exist and are 1-min spaced (spot check first 50)
    for k in range(0, min(50, len(idx))):
        g = idx.group_name(k)
        ts = store.timestamps(g)[idx.history_row_indices[k]]
        assert np.all(np.diff(ts) == NS_PER_MIN)
