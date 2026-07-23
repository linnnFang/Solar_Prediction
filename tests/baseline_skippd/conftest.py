"""Shared fixtures for baseline_skippd tests (synthetic parquet store)."""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

from src.baseline_skippd.io import SKIPPDProcessedStore

_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _png(hw=(64, 64)):
    arr = (np.random.rand(*hw, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def write_shard(path, n=None, hw=(64, 64), cols=("image", "time", "pv"),
                start="2017-06-01 07:00", times=None):
    """Write a small synthetic SKIPP'D-shaped parquet shard.

    Pass ``n`` for a contiguous 1-min run from ``start``, or ``times`` (a list of
    pandas Timestamps) to control the exact timestamps (e.g. to punch a gap)."""
    if times is not None:
        times = pd.DatetimeIndex(times)
        n = len(times)
    else:
        times = pd.date_range(start, periods=n, freq="min", tz="US/Pacific")
    data = {}
    if "image" in cols:
        data["image"] = pa.array([{"bytes": _png(hw), "path": None} for _ in range(n)], type=_STRUCT)
    if "time" in cols:
        data["time"] = pa.array(times)
    if "pv" in cols:
        data["pv"] = pa.array(np.random.rand(n).astype("float32"))
    pq.write_table(pa.table(data), path)


@pytest.fixture
def synth_store(tmp_path):
    """3 trainval days (rows 5/5/4) + 1 test day (6 rows) of 64x64 frames."""
    d = tmp_path / "data"
    d.mkdir()
    for i, n in enumerate([5, 5, 4]):
        write_shard(d / f"train-0000{i}.parquet", n, start=f"2017-06-0{i + 1} 07:00")
    write_shard(d / "test-00000.parquet", 6, start="2017-07-01 07:00")
    return SKIPPDProcessedStore(tmp_path)


@pytest.fixture
def pipeline_dm(tmp_path):
    """Full stack on synthetic data: store -> split -> windows -> set-up DataModule."""
    from src.baseline_skippd.split import make_split
    from src.baseline_skippd.windows import build_window_index
    from src.baseline_skippd.scalers import CapacityPVScaler
    from src.baseline_skippd.datamodule import SKIPPDDataModule
    d = tmp_path / "data"; d.mkdir()
    write_shard(d / "train-00000.parquet", 60, start="2017-06-01 07:00")
    write_shard(d / "train-00001.parquet", 60, start="2017-06-02 07:00")
    write_shard(d / "test-00000.parquet", 60, start="2017-07-01 07:00")
    store = SKIPPDProcessedStore(tmp_path)
    manifest = make_split(store, val_frac=0.5)
    index, _ = build_window_index(store, manifest)
    return SKIPPDDataModule(store, manifest, index, CapacityPVScaler(30.1),
                            batch_size=8, seed=0).setup()
