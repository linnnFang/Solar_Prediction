"""Phase 2 tests: day-level split manifest."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baseline_skippd.split import SplitManifest, make_split, unique_dates

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "SKIPPD"


def test_train_val_dates_disjoint(synth_store):
    m = make_split(synth_store)
    assert set(m.train_dates) & set(m.validation_dates) == set()
    assert m.train_dates and m.validation_dates            # both non-empty


def test_test_group_preserved(synth_store):
    m = make_split(synth_store)
    assert m.test_dates == unique_dates(synth_store, "test")
    # test days are untouched by the trainval split
    assert set(m.test_dates) & (set(m.train_dates) | set(m.validation_dates)) == set()


def test_no_date_split_across_sets(synth_store):
    m = make_split(synth_store)
    tv = set(unique_dates(synth_store, "trainval"))
    # train ∪ val partitions the trainval days exactly (no day lost/duplicated/split)
    assert set(m.train_dates) | set(m.validation_dates) == tv
    assert len(m.train_dates) + len(m.validation_dates) == len(tv)
    m.validate()                                           # pairwise-disjoint or raises


def test_chronological_holdout_val_is_latest(synth_store):
    m = make_split(synth_store, strategy="chronological_day_holdout")
    assert max(m.train_dates) < min(m.validation_dates)    # validation strictly later


def test_split_manifest_roundtrip(synth_store, tmp_path):
    m = make_split(synth_store, seed=7)
    p = m.save(tmp_path / "split.json")
    loaded = SplitManifest.load(p)
    assert loaded == m
    assert loaded.config_hash == m.config_hash == m._hash()   # stable, reproducible


def test_kfold_by_unique_date(synth_store):
    m = make_split(synth_store, strategy="repo_day_10fold", n_folds=3, fold=0, seed=1)
    tv = set(unique_dates(synth_store, "trainval"))
    assert set(m.train_dates) | set(m.validation_dates) == tv
    assert set(m.train_dates) & set(m.validation_dates) == set()


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="real SKIPP'D data not present")
def test_real_split_no_leakage():
    from src.baseline_skippd import SKIPPDProcessedStore
    m = make_split(SKIPPDProcessedStore(REAL_ROOT))
    m.validate()
    assert m.trainval_test_overlap == []                   # trainval and test days disjoint
    assert max(m.train_dates) < min(m.validation_dates)
