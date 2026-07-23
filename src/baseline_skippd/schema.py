"""
Canonical schema constants and errors for the SKIPP'D data store.

The processed SKIPP'D benchmark is described as an HDF5 layout (``trainval`` /
``test`` groups, each with ``images_log`` / ``pv_log`` + aligned timestamps). In
this repo the physical backend is the HuggingFace **parquet** redistribution, so
the *logical* schema below is what the store guarantees, independent of backend:

    group "trainval"  <-  data/train-*.parquet   (image + time + pv)
    group "test"      <-  data/test-*.parquet
    images_log        <-  column "image"  (struct<bytes, path>, 64x64x3 uint8 PNG)
    pv_log            <-  column "pv"      (float, kW)
    timestamps        <-  column "time"    (tz-aware, one per row)

Only these constants and the ``SchemaError`` live here; the store enforces them.
"""

GROUPS = ("trainval", "test")

# raw parquet column names (the physical backend)
TIME_COL = "time"
PV_COL = "pv"
IMAGE_COL = "image"
RAW_COLUMNS = (IMAGE_COL, TIME_COL, PV_COL)

# image geometry (SKIPP'D fisheye frames)
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
IMAGE_DTYPE = "uint8"

# default backend location relative to a store root (SKIPP'D dataset dir)
DEFAULT_GROUP_PATTERNS = {
    "trainval": "data/train-*.parquet",
    "test": "data/test-*.parquet",
}


class SchemaError(ValueError):
    """Raised when the store's files violate the expected SKIPP'D schema."""
