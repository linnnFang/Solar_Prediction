"""Phase 4 tests: canonical dataset, scalers, datamodule."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conftest import write_shard
from src.baseline_skippd.io import SKIPPDProcessedStore
from src.baseline_skippd.split import make_split
from src.baseline_skippd.windows import build_window_index
from src.baseline_skippd.scalers import (
    IdentityPVScaler, CapacityPVScaler, StandardPVScaler, build_scaler)
from src.baseline_skippd.datamodule import SKIPPDDataModule


@pytest.fixture
def dm(tmp_path):
    """Store (2 trainval days + 1 test day, 60 min each) -> split -> windows -> datamodule."""
    d = tmp_path / "data"; d.mkdir()
    write_shard(d / "train-00000.parquet", 60, start="2017-06-01 07:00")
    write_shard(d / "train-00001.parquet", 60, start="2017-06-02 07:00")
    write_shard(d / "test-00000.parquet", 60, start="2017-07-01 07:00")
    store = SKIPPDProcessedStore(tmp_path)
    manifest = make_split(store, val_frac=0.5)
    index, _ = build_window_index(store, manifest)
    module = SKIPPDDataModule(store, manifest, index, StandardPVScaler(), batch_size=8, seed=0)
    return module.setup()


# ---------- scalers -----------------------------------------------------------
@pytest.mark.parametrize("scaler", [IdentityPVScaler(), CapacityPVScaler(30.1), StandardPVScaler()])
def test_scaler_roundtrip(scaler, tmp_path):
    x = np.array([0.0, 5.0, 15.0, 29.5], dtype=np.float32)
    scaler.fit(x)
    back = scaler.inverse_transform(scaler.transform(x))
    assert np.allclose(back, x, atol=1e-4)
    # artifact save/restore preserves behaviour
    p = scaler.save(tmp_path / "scaler.json")
    from src.baseline_skippd.scalers import PVScaler
    restored = PVScaler.load(p)
    assert np.allclose(restored.transform(x), scaler.transform(x), atol=1e-6)


def test_capacity_scaler_maps_to_unit():
    s = build_scaler("capacity", capacity_kw=30.1)
    assert np.isclose(s.transform(np.float32(30.1)), 1.0)


def test_scaler_fit_train_only(dm):
    store, manifest, index = dm.store, dm.split_manifest, dm.window_index
    train_rows = np.unique(np.concatenate(
        [index.subset("train").history_row_indices.reshape(-1),
         index.subset("train").target_row_index]))
    train_pv = store.pv("trainval")[train_rows]
    assert np.isclose(dm.scaler.mean, train_pv.mean(), atol=1e-5)
    # differs from a fit over ALL trainval power (val leakage would change it)
    assert not np.isclose(dm.scaler.mean, store.pv("trainval").mean(), atol=1e-6)


# ---------- dataset / dataloader ---------------------------------------------
def test_image_normalization(dm):
    sample = dm.dataset("train")[0]
    img = sample["images"]
    assert img.shape == (16, 3, 64, 64) and img.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


def test_dataset_shapes(dm):
    s = dm.dataset("test")[0]
    assert s["pv_history"].shape == (16,) and s["pv_history"].dtype == torch.float32
    assert s["target"].shape == (1,)
    assert s["issue_time"].dtype == torch.int64
    assert s["split"] == "test" and isinstance(s["sample_index"], int)


def test_dataloader_batch_schema(dm):
    batch = next(iter(dm.train_dataloader()))
    b = batch["images"].shape[0]
    assert batch["images"].shape == (b, 16, 3, 64, 64)
    assert batch["pv_history"].shape == (b, 16)
    assert batch["target"].shape == (b, 1)
    assert batch["issue_time"].shape == (b,) and batch["issue_time"].dtype == torch.int64
    assert isinstance(batch["split"], list) and len(batch["split"]) == b
    assert dm.sample_schema()["images"]["shape"] == (16, 3, 64, 64)


def test_test_loader_not_shuffled(dm):
    idx1 = [int(i) for batch in dm.test_dataloader() for i in batch["sample_index"]]
    idx2 = [int(i) for batch in dm.test_dataloader() for i in batch["sample_index"]]
    assert idx1 == sorted(idx1)      # in-order
    assert idx1 == idx2              # deterministic across passes


def test_worker_safe_multiprocess(dm):
    dm.num_workers = 2
    batch = next(iter(dm.val_dataloader()))
    assert batch["images"].shape[1:] == (16, 3, 64, 64)
