import numpy as np
import pandas as pd

from src.helper.metrics import monthly_accuracy_15min, mean_monthly_accuracy
from src.national_renewable import make_xgb_historical_table


def _prepared_fixture():
    ts = pd.date_range("2019-09-29", periods=200, freq="15min")
    frame = pd.DataFrame({
        "ts": ts,
        "site": 1,
        "capacity": 10.0,
        "cf": np.arange(len(ts), dtype=float) / 1000,
        "cf_input": np.arange(len(ts), dtype=float) / 1000,
        "irr_total": 100.0,
        "dni": 80.0,
        "ghi": 60.0,
        "temp": 20.0,
        "pressure": 900.0,
        "humidity": 30.0,
        "split": np.where(ts < pd.Timestamp("2019-10-01"), "train", "val"),
        "site_1": 1,
    })
    for col in ("irr_total", "dni", "ghi", "temp", "pressure", "humidity"):
        frame[f"{col}_missing"] = 0
    return frame


def test_xgb_target_is_next_15min_and_split_follows_target():
    table, _ = make_xgb_historical_table(_prepared_fixture(), horizon=1)
    assert (table["target_ts"] - table["ts"] == pd.Timedelta("15min")).all()
    assert np.allclose(table["target_cf"], table["cf_input"] + 0.001)

    boundary = table.loc[table["target_ts"] == pd.Timestamp("2019-10-01")].iloc[0]
    assert boundary["ts"] == pd.Timestamp("2019-09-30 23:45")
    assert boundary["split"] == "val"


def test_monthly_accuracy_uses_all_15min_points_before_rmse():
    predictions = pd.DataFrame({
        "site": [1, 1],
        "capacity": [10.0, 10.0],
        "target_ts": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:15"]),
        "y_true": [0.0, 0.0],
        "y_pred": [0.0, 2.0],
    })
    monthly = monthly_accuracy_15min(predictions)
    assert monthly.loc[0, "mae_mw"] == 1.0
    assert np.isclose(monthly.loc[0, "rmse_mw"], np.sqrt(2))
    assert np.isclose(monthly.loc[0, "acc_mae"], 0.9)
    assert np.isclose(monthly.loc[0, "acc_rmse"], 1 - np.sqrt(2) / 10)

    per_site, overall = mean_monthly_accuracy(monthly)
    assert np.isclose(per_site.loc[0, "mean_monthly_acc_mae"], 0.9)
    assert np.isclose(overall["mean_monthly_acc_rmse"], 1 - np.sqrt(2) / 10)

