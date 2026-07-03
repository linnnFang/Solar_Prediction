"""
Forecasting accuracy metrics for PV power (capacity-normalised, Chinese grid style).

Accuracy is defined as 1 - error / capacity. `monthly_accuracy` reports the
day-ahead monthly-average accuracy used for short-term PV forecasting.
"""

import numpy as np
import pandas as pd


def _rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def accuracy_rmse(y_true, y_pred, cap=1.0):
    """
    RMSE-based accuracy.
    Input : y_true, y_pred arrays; cap = plant capacity used to normalise error.
    Output: float, 1 - RMSE / cap.
    """
    return 1 - _rmse(y_true, y_pred) / cap


def accuracy_mae(y_true, y_pred, cap=1.0):
    """
    MAE-based accuracy.
    Input : y_true, y_pred arrays; cap = plant capacity used to normalise error.
    Output: float, 1 - MAE / cap.
    """
    return 1 - _mae(y_true, y_pred) / cap


def monthly_accuracy(df, cap=1.0, time_col="ts_local", zone_col="ZONE"):
    """
    Day-ahead monthly-average accuracy computed on daytime points.

    Input : df with columns time_col (local timestamps), zone_col, y_true,
            y_pred and is_daytime; cap = plant capacity.
    Output: DataFrame indexed by (zone, month) with columns acc_rmse and acc_mae.
            Daily 1-RMSE/cap and 1-MAE/cap are computed on that day's daytime
            points, then averaged over the days in each month.
    """
    day = df[df["is_daytime"]].copy()
    day["_date"] = day[time_col].dt.normalize()
    day["_month"] = day[time_col].dt.to_period("M")

    def _daily(g):
        return pd.Series({
            "acc_rmse": accuracy_rmse(g["y_true"], g["y_pred"], cap),
            "acc_mae": accuracy_mae(g["y_true"], g["y_pred"], cap),
        })

    daily = day.groupby([zone_col, "_month", "_date"]).apply(_daily)
    monthly = daily.groupby([zone_col, "_month"]).mean()
    monthly.index = monthly.index.set_names([zone_col, "month"])
    return monthly
