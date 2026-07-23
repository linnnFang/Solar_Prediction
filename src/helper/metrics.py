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


def monthly_accuracy_15min(
    df,
    time_col="target_ts",
    site_col="site",
    capacity_col="capacity",
):
    """Score individual 15-minute predictions within each site-month.

    The monthly MAE and RMSE are computed over all 15-minute target points in
    that month, then normalised by that station's capacity.  This intentionally
    differs from ``monthly_accuracy`` above, which first computes daytime daily
    scores for the GEFCom day-ahead task.

    Required columns: ``y_true``, ``y_pred``, time/site/capacity columns.
    Returns one row per site-month plus raw errors and sample counts.
    """
    required = {time_col, site_col, capacity_col, "y_true", "y_pred"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"monthly_accuracy_15min missing columns: {missing}")

    work = df[list(required)].dropna().copy()
    work[time_col] = pd.to_datetime(work[time_col])
    work["month"] = work[time_col].dt.to_period("M")

    def _score(group):
        capacities = group[capacity_col].unique()
        if len(capacities) != 1:
            raise ValueError("capacity must be constant within each site-month")
        cap = float(capacities[0])
        mae = _mae(group["y_true"], group["y_pred"])
        rmse = _rmse(group["y_true"], group["y_pred"])
        return pd.Series({
            "n_15min": len(group),
            "mae_mw": mae,
            "rmse_mw": rmse,
            "acc_mae": 1 - mae / cap,
            "acc_rmse": 1 - rmse / cap,
        })

    return (
        work.groupby([site_col, "month"], sort=True, observed=True)
        # Select value columns before apply for compatibility with both older
        # pandas (no include_groups keyword) and newer pandas (where grouping
        # columns inside apply are deprecated).
        [[capacity_col, "y_true", "y_pred"]]
        .apply(_score)
        .reset_index()
    )


def mean_monthly_accuracy(monthly, site_col="site"):
    """Arithmetic mean of monthly accuracy rows, reported per site and overall."""
    per_site = (
        monthly.groupby(site_col, sort=True)[["acc_mae", "acc_rmse"]]
        .mean()
        .rename(columns={"acc_mae": "mean_monthly_acc_mae",
                         "acc_rmse": "mean_monthly_acc_rmse"})
        .reset_index()
    )
    overall = {
        "mean_monthly_acc_mae": float(monthly["acc_mae"].mean()),
        "mean_monthly_acc_rmse": float(monthly["acc_rmse"].mean()),
    }
    return per_site, overall
