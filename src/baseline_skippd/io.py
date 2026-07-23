"""
SKIPPDProcessedStore — read-only access to the SKIPP'D frames + power.

Backend is the HuggingFace **parquet** redistribution (decision A of the Phase-0
audit): the Stanford processed HDF5 does not exist here and h5py is not
installed. The store exposes the same *logical* schema either way (see
`schema.py`): two groups (``trainval`` / ``test``), each a flat row space of
``images_log`` / ``pv_log`` / ``timestamps`` aligned one-to-one.

Design:
- **Cheap init**: only reads parquet footers (row counts / row-group sizes) and
  builds offset tables. No image data is loaded into RAM.
- **Lazy small columns**: ``pv`` and ``timestamps`` are small; loaded and cached
  on first access (a few MB), never the images.
- **Worker-safe**: parquet handles and row-group caches are opened lazily and
  keyed to the current process id, and are dropped from the pickled state, so a
  forked/spawned DataLoader worker always uses its own handles.
"""

import glob
import io as _io
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.baseline_skippd import schema
from src.baseline_skippd.schema import SchemaError

_RG_CACHE_MAX = 8   # row-group image tables kept per process (window reads stay local)


class SKIPPDProcessedStore:
    """Lazy, worker-safe reader over the SKIPP'D parquet shards."""

    def __init__(self, root, group_patterns=None):
        """
        Input : root = SKIPP'D dataset dir (contains ``data/``);
                group_patterns = {group: glob} override (defaults to parquet shards).
        Validates that every group has shards and every shard has the raw columns.
        """
        self.root = Path(root)
        self.group_patterns = dict(group_patterns or schema.DEFAULT_GROUP_PATTERNS)

        self._shards = {}          # group -> [shard paths]
        self._nrows = {}           # group -> total rows
        self._shard_start = {}     # group -> np.array cumulative shard start offsets
        self._rg_cumrows = {}      # group -> [per-shard np.array of row-group cum offsets]
        for group, pattern in self.group_patterns.items():
            paths = sorted(glob.glob(str(self.root / pattern)))
            if not paths:
                raise SchemaError(f"group '{group}': no parquet shards match {self.root / pattern}")
            self._index_group(group, paths)

        # lazy caches (excluded from pickled state)
        self._pv_cache = {}
        self._ts_cache = {}
        self._img_shape = None
        self._pid = os.getpid()
        self._handles = {}
        self._rg_cache = OrderedDict()

    # ---- init-time indexing (footers only) ----------------------------------
    def _index_group(self, group, paths):
        shard_nrows, rg_cumrows = [], []
        for p in paths:
            pf = pq.ParquetFile(p)
            names = pf.schema_arrow.names
            missing = [c for c in schema.RAW_COLUMNS if c not in names]
            if missing:
                raise SchemaError(f"group '{group}' shard {Path(p).name}: missing columns {missing}")
            shard_nrows.append(pf.metadata.num_rows)
            counts = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
            rg_cumrows.append(np.concatenate([[0], np.cumsum(counts)]))
        self._shards[group] = paths
        self._nrows[group] = int(sum(shard_nrows))
        self._shard_start[group] = np.concatenate([[0], np.cumsum(shard_nrows)])
        self._rg_cumrows[group] = rg_cumrows

    # ---- pickling: never carry open handles across processes ----------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_rg_cache"] = OrderedDict()
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pid = os.getpid()

    # ---- basic accessors -----------------------------------------------------
    def groups(self):
        return tuple(self._shards.keys())

    def num_rows(self, group):
        self._require_group(group)
        return self._nrows[group]

    def _require_group(self, group):
        if group not in self._shards:
            raise SchemaError(f"unknown group '{group}'; have {list(self._shards)}")

    def pv(self, group):
        """[N] float32 power (kW), cached."""
        self._require_group(group)
        if group not in self._pv_cache:
            parts = [pq.read_table(p, columns=[schema.PV_COL]).column(0).to_numpy(zero_copy_only=False)
                     for p in self._shards[group]]
            pv = np.concatenate(parts).astype(np.float32)
            if len(pv) != self._nrows[group]:
                raise SchemaError(f"group '{group}': pv length {len(pv)} != rows {self._nrows[group]}")
            self._pv_cache[group] = pv
        return self._pv_cache[group]

    def timestamps_index(self, group):
        """[N] tz-aware pandas DatetimeIndex (US/Pacific), cached."""
        self._require_group(group)
        if group not in self._ts_cache:
            parts = [pq.read_table(p, columns=[schema.TIME_COL]).column(0).to_pandas()
                     for p in self._shards[group]]
            ts = pd.DatetimeIndex(pd.concat(parts, ignore_index=True))
            if len(ts) != self._nrows[group]:
                raise SchemaError(f"group '{group}': timestamps length {len(ts)} != rows {self._nrows[group]}")
            self._ts_cache[group] = ts
        return self._ts_cache[group]

    def timestamps(self, group):
        """[N] int64 nanoseconds since epoch (UTC), aligned to rows.

        Forced to ns regardless of the parquet's datetime resolution (the real
        data is datetime64[ms]); ``.asi8`` returns the index's own unit, so we
        normalise here to keep a single ns contract for every downstream user."""
        return self.timestamps_index(group).as_unit("ns").asi8

    @property
    def image_shape(self):
        """(H, W, C) of a decoded frame; validated against the schema once."""
        if self._img_shape is None:
            arr = self.read_image(self.groups()[0], 0)
            if tuple(arr.shape) != schema.IMAGE_SHAPE:
                raise SchemaError(f"image shape {tuple(arr.shape)} != expected {schema.IMAGE_SHAPE}")
            self._img_shape = tuple(arr.shape)
        return self._img_shape

    # ---- worker-local lazy handles ------------------------------------------
    def _handle(self, path):
        pid = os.getpid()
        if self._pid != pid:                       # forked/spawned: drop parent's handles
            self._handles, self._rg_cache, self._pid = {}, OrderedDict(), pid
        h = self._handles.get(path)
        if h is None:
            h = self._handles[path] = pq.ParquetFile(path)
        return h

    def _locate(self, group, row):
        """global row -> (shard_idx, local_row)."""
        n = self._nrows[group]
        if not (0 <= row < n):
            raise IndexError(f"group '{group}' row {row} out of range [0, {n})")
        starts = self._shard_start[group]
        shard_idx = int(np.searchsorted(starts, row, side="right") - 1)
        return shard_idx, int(row - starts[shard_idx])

    def _read_rowgroup(self, path, rg_idx):
        key = (path, rg_idx)
        if self._pid != os.getpid():
            self._handles, self._rg_cache, self._pid = {}, OrderedDict(), os.getpid()
        tbl = self._rg_cache.get(key)
        if tbl is None:
            tbl = self._handle(path).read_row_group(rg_idx, columns=[schema.IMAGE_COL])
            self._rg_cache[key] = tbl
            if len(self._rg_cache) > _RG_CACHE_MAX:
                self._rg_cache.popitem(last=False)
        else:
            self._rg_cache.move_to_end(key)
        return tbl

    def read_image(self, group, row):
        """Decode one frame -> [H, W, 3] uint8 (worker-safe, lazy)."""
        self._require_group(group)
        shard_idx, local = self._locate(group, row)
        path = self._shards[group][shard_idx]
        cum = self._rg_cumrows[group][shard_idx]
        rg_idx = int(np.searchsorted(cum, local, side="right") - 1)
        rg_local = int(local - cum[rg_idx])
        cell = self._read_rowgroup(path, rg_idx).column(0)[rg_local].as_py()
        return np.asarray(_decode_png(cell["bytes"]))

    def read_images(self, group, rows):
        """Decode a list of frames -> [k, H, W, 3] uint8 (consecutive rows hit the cache)."""
        return np.stack([self.read_image(group, r) for r in rows])

    # ---- data-check report ---------------------------------------------------
    def check(self, group):
        """Return a data-quality report dict for one group."""
        self._require_group(group)
        n = self.num_rows(group)
        pv = self.pv(group)
        ts = self.timestamps_index(group)
        img0 = self.read_image(group, 0)
        diffs = ts.to_series().reset_index(drop=True).diff().dt.total_seconds().dropna()
        return {
            "group": group,
            "rows": int(n),
            "image_shape": tuple(img0.shape),
            "image_dtype": str(img0.dtype),
            "pv_dtype": str(pv.dtype),
            "pv_min": float(pv.min()), "pv_max": float(pv.max()),
            "negative_pv": int((pv < 0).sum()),
            "timestamp_start": str(ts.min()), "timestamp_end": str(ts.max()),
            "duplicate_timestamps": int(ts.duplicated().sum()),
            "non_monotonic_steps": int((diffs < 0).sum()),
            "gaps_over_1min": int((diffs > 60).sum()),
            "gaps_over_1day": int((diffs > 86400).sum()),
            "largest_gap_minutes": float(diffs.max() / 60) if len(diffs) else 0.0,
        }

    def report(self):
        """Data-check report for every group."""
        return {g: self.check(g) for g in self.groups()}

    # ---- context manager -----------------------------------------------------
    def close(self):
        self._handles.clear()
        self._rg_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _decode_png(raw_bytes):
    from PIL import Image
    return Image.open(_io.BytesIO(raw_bytes))
