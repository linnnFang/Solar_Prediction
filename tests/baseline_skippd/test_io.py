"""Phase 1 tests: SKIPPDProcessedStore (parquet backend)."""

import io
import os
import pickle
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
from src.baseline_skippd.schema import SchemaError, IMAGE_SHAPE

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "SKIPPD"
_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _png(hw=(64, 64)):
    arr = (np.random.rand(*hw, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _write_shard(path, n, hw=(64, 64), cols=("image", "time", "pv"), start="2017-06-01 07:00"):
    data = {}
    if "image" in cols:
        data["image"] = pa.array([{"bytes": _png(hw), "path": None} for _ in range(n)], type=_STRUCT)
    if "time" in cols:
        data["time"] = pa.array(pd.date_range(start, periods=n, freq="min", tz="US/Pacific"))
    if "pv" in cols:
        data["pv"] = pa.array(np.random.rand(n).astype("float32"))
    pq.write_table(pa.table(data), path)


@pytest.fixture
def synth_store(tmp_path):
    """A 3-shard trainval + 1-shard test store of small 64x64 frames."""
    d = tmp_path / "data"
    d.mkdir()
    for i, n in enumerate([5, 5, 4]):
        _write_shard(d / f"train-0000{i}.parquet", n, start=f"2017-06-0{i+1} 07:00")
    _write_shard(d / "test-00000.parquet", 6, start="2017-07-01 07:00")
    return SKIPPDProcessedStore(tmp_path)


# ---------- real data ---------------------------------------------------------
@pytest.mark.skipif(not REAL_ROOT.exists(), reason="real SKIPP'D data not present")
def test_processed_store_shapes():
    store = SKIPPDProcessedStore(REAL_ROOT)
    assert store.groups() == ("trainval", "test")
    assert store.num_rows("trainval") == 349372
    assert store.num_rows("test") == 14003
    assert store.image_shape == IMAGE_SHAPE
    img = store.read_image("trainval", 123)
    assert img.shape == IMAGE_SHAPE and img.dtype == np.uint8


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="real SKIPP'D data not present")
def test_timestamp_alignment():
    store = SKIPPDProcessedStore(REAL_ROOT)
    for g in store.groups():
        n = store.num_rows(g)
        assert len(store.pv(g)) == n
        assert len(store.timestamps(g)) == n
        assert store.timestamps(g).dtype == np.int64
        # first row of pv/timestamps lines up with a decodable image
        assert store.read_image(g, 0).shape == IMAGE_SHAPE


# ---------- synthetic ---------------------------------------------------------
def test_processed_store_shapes_synth(synth_store):
    assert synth_store.num_rows("trainval") == 14
    assert synth_store.num_rows("test") == 6
    assert synth_store.image_shape == IMAGE_SHAPE
    # row 12 lives in the 3rd trainval shard (offsets 5,10,14) -> local row 2
    assert synth_store.read_image("trainval", 12).shape == IMAGE_SHAPE


def test_worker_safe_lazy_open(synth_store, monkeypatch):
    # init opens no handles
    assert synth_store._handles == {}
    synth_store.read_image("trainval", 0)
    assert len(synth_store._handles) >= 1

    # pickled state carries NO open handles (spawn-worker safety)
    restored = pickle.loads(pickle.dumps(synth_store))
    assert restored._handles == {}
    assert restored.read_image("trainval", 3).shape == IMAGE_SHAPE  # reopens locally

    # simulated fork: a different pid drops the parent's handles
    synth_store.read_image("trainval", 1)
    fake_pid = os.getpid() + 999
    monkeypatch.setattr("src.baseline_skippd.io.os.getpid", lambda: fake_pid)
    synth_store.read_image("trainval", 2)
    assert synth_store._pid == fake_pid


def test_schema_validation_errors(tmp_path):
    d = tmp_path / "data"
    d.mkdir()

    # (a) no shards for a group
    with pytest.raises(SchemaError, match="no parquet shards"):
        SKIPPDProcessedStore(tmp_path)

    # (b) missing required column (pv)
    _write_shard(d / "train-00000.parquet", 4, cols=("image", "time"))
    _write_shard(d / "test-00000.parquet", 4)
    with pytest.raises(SchemaError, match="missing columns"):
        SKIPPDProcessedStore(tmp_path)

    # (c) wrong image shape
    d2 = tmp_path / "data2"
    d2.mkdir()
    _write_shard(d2 / "train-00000.parquet", 4, hw=(32, 32))
    _write_shard(d2 / "test-00000.parquet", 4, hw=(32, 32))
    store = SKIPPDProcessedStore(tmp_path, group_patterns={
        "trainval": "data2/train-*.parquet", "test": "data2/test-*.parquet"})
    with pytest.raises(SchemaError, match="image shape"):
        _ = store.image_shape


def test_report_smoke(synth_store):
    rep = synth_store.check("trainval")
    assert rep["rows"] == 14
    assert rep["image_shape"] == IMAGE_SHAPE
    assert set(rep) >= {"negative_pv", "duplicate_timestamps", "non_monotonic_steps",
                        "gaps_over_1min", "largest_gap_minutes"}
