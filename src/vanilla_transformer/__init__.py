"""
Reusable vanilla-Transformer forecasting toolkit.

Dataset-agnostic pieces for direct multi-step time-series forecasting:
    Standardizer, WindowDataset, make_dataloaders   (data.py)
    PositionalEncoding, TransformerForecaster        (model.py)
    Trainer, pick_device                             (trainer.py)
    forecast_report, seasonal_naive                  (evaluate.py)

Dataset-specific choices (which columns, how to split, seasonality period) stay
in the caller / notebook; feed this package pre-split DataFrame frames.
"""

import os

import torch

# This environment ships several OpenMP runtimes (torch, sklearn, skimage and
# cvxopt each bundle their own libomp/libgomp). Once one of the others has been
# loaded -- e.g. a notebook trains an XGBoost baseline before it gets here --
# torch's threaded CPU kernels deadlock at 0% CPU or segfault on the first
# parallel op, in practice the torch.sin inside PositionalEncoding. Pinning
# torch to one CPU thread sidesteps the shared thread pool entirely.
#
# Only done when a GPU backend will carry the training, where CPU intra-op
# threads are irrelevant anyway. On a CPU-only box we leave the default alone
# rather than silently crippling CPU training; set SOLAR_TORCH_THREADS to
# override in either direction (SOLAR_TORCH_THREADS=1 restores the guard).
_threads = os.environ.get("SOLAR_TORCH_THREADS")
if _threads is not None:
    torch.set_num_threads(int(_threads))
elif torch.backends.mps.is_available() or torch.cuda.is_available():
    torch.set_num_threads(1)

from src.vanilla_transformer.data import Standardizer, WindowDataset, make_dataloaders
from src.vanilla_transformer.model import PositionalEncoding, TransformerForecaster
from src.vanilla_transformer.trainer import Trainer, pick_device
from src.vanilla_transformer.evaluate import forecast_report, seasonal_naive

__all__ = [
    "Standardizer", "WindowDataset", "make_dataloaders",
    "PositionalEncoding", "TransformerForecaster",
    "Trainer", "pick_device",
    "forecast_report", "seasonal_naive",
]
