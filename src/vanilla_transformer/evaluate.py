"""
Evaluation helpers for the vanilla-transformer forecaster.

Metric formulas are not reimplemented here: RMSE / MAE / accuracy come from
`src/helper/metrics.py`. This module only composes them into a report and
provides a seasonal-naive baseline whose period is passed in explicitly (so it
works for any dataset's seasonality, not just 24h).
"""

import numpy as np

from src.helper.metrics import _rmse, _mae, accuracy_rmse, accuracy_mae


def forecast_report(y_true, y_pred, cap=None):
    """
    Combine helper metrics into one report over the flattened arrays.
    Input : y_true, y_pred = arrays of matching shape; cap = plant capacity to
            also report the GEFCom-style accuracy (optional, grid-specific).
    Output: dict with rmse, mae, and (when cap is given) acc_rmse, acc_mae.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    report = {"rmse": _rmse(yt, yp), "mae": _mae(yt, yp)}
    if cap is not None:
        report["acc_rmse"] = accuracy_rmse(yt, yp, cap)
        report["acc_mae"] = accuracy_mae(yt, yp, cap)
    return report


def seasonal_naive(dataset, period):
    """
    Seasonal-naive baseline aligned to a WindowDataset's window order.

    For each window predicting target[t : t+H], use target[t-period : t-period+H]
    (e.g. period=24 -> same hours one day earlier; period could be 7, 96, 168...).
    Requires period <= dataset.lookback so the earlier block stays inside the
    frame. The output rows line up with `dataset` iterated in order, i.e. with
    predictions from a shuffle=False DataLoader over the same dataset.

    Output: array of shape [n_windows, horizon].
    """
    H = dataset.horizon
    if period > dataset.lookback:
        raise ValueError(
            f"period ({period}) must be <= lookback ({dataset.lookback}) so the "
            "previous-season block exists inside the window")
    return np.array([dataset.targets[sid][t - period:t - period + H]
                     for sid, t in dataset.index])
