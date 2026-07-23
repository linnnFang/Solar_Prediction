"""
Nested run-config for the SKIPP'D pipeline (one YAML fully describes one run).

Reuses the ``yaml.safe_load`` pattern from ``src/helper/params.py`` but the shape
is nested (data / window / split / scaler / model / trainer / eval / artifacts) so
a single file drives an entire run. Missing keys fall back to DEFAULTS.
"""

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "run_name": "run",
    "seed": 42,
    "device": None,
    "verbose": True,
    "data": {"root": "data/raw/SKIPPD"},
    "window": {"history_minutes": 15, "history_step_minutes": 1, "horizon_minutes": 15,
               "strides": {"train": 2, "validation": 1, "test": 1}},
    "split": {"strategy": "chronological_day_holdout", "seed": 42, "val_frac": 0.15},
    "scaler": {"name": "capacity", "capacity_kw": 30.1},
    "model": {"name": "pv_only"},
    "trainer": {"lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0,
                "max_epochs": 30, "patience": 6, "batch_size": 256, "num_workers": 0},
    "eval": {"capacity_kw": 30.1},
    "clear_sky": {"quantile": 0.95, "smooth": 15},
    "artifacts": {"dir": "artifacts/skippd"},
}


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_skippd_config(path):
    """Load a run-config YAML, merged over DEFAULTS, with a content hash attached."""
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    cfg = _deep_merge(DEFAULTS, raw)
    cfg["config_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in cfg.items() if k != "config_hash"}, sort_keys=True, default=str)
        .encode()).hexdigest()[:16]
    return cfg
