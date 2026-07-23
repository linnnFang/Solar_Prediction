"""
Unified evaluator — every model is scored the same way, in kW.

Metric focus (per project decision): the Chinese-grid **monthly-average accuracy**
    acc = 1 - error / capacity        (capacity = 30.1 kW)
reported for both RMSE and MAE, plus the raw kW errors they come from. Metric
formulas are reused from ``src/helper/metrics.py`` (not re-implemented).

Inputs are the uniform prediction result dict (already inverse-transformed to kW)
produced by ``training.predict_kw`` or a rule baseline, so all models are scored
on identical samples. Outputs: metrics.json, predictions.parquet, plots.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.helper.metrics import _rmse, _mae, accuracy_rmse, accuracy_mae, monthly_accuracy

CAPACITY_KW = 30.1
TZ = "US/Pacific"


def compute_metrics(result, capacity_kw=CAPACITY_KW):
    """Overall + monthly-average accuracy for one prediction result (kW)."""
    yt = np.asarray(result["y_true_kw"], float)
    yp = np.asarray(result["y_pred_kw"], float)
    overall = {
        "n": int(yt.size),
        "rmse_kw": float(_rmse(yt, yp)), "mae_kw": float(_mae(yt, yp)),
        "mbe_kw": float(np.mean(yp - yt)),
        "acc_rmse": float(accuracy_rmse(yt, yp, capacity_kw)),   # 1 - RMSE/Cap
        "acc_mae": float(accuracy_mae(yt, yp, capacity_kw)),     # 1 - MAE/Cap
    }
    local = pd.to_datetime(result["issue_time"], utc=True).tz_convert(TZ)
    df = pd.DataFrame({"ts_local": local, "ZONE": "SKIPPD",
                       "y_true": yt, "y_pred": yp, "is_daytime": True})
    monthly = monthly_accuracy(df, cap=capacity_kw)              # per (zone, month)
    monthly = monthly.reset_index()
    return {
        "capacity_kw": capacity_kw,
        "overall": overall,
        "monthly_mean_acc_rmse": float(monthly["acc_rmse"].mean()),
        "monthly_mean_acc_mae": float(monthly["acc_mae"].mean()),
        "per_month": [{"month": str(r["month"]), "acc_rmse": float(r["acc_rmse"]),
                       "acc_mae": float(r["acc_mae"])} for _, r in monthly.iterrows()],
    }


def predictions_frame(result, model_name):
    yt = np.asarray(result["y_true_kw"], float)
    yp = np.asarray(result["y_pred_kw"], float)
    return pd.DataFrame({
        "sample_index": np.asarray(result["sample_index"]),
        "model_name": model_name,
        "issue_time": pd.to_datetime(result["issue_time"], utc=True),
        "target_time": pd.to_datetime(result["target_time"], utc=True),
        "split": result["split"],
        "y_true_kw": yt, "y_pred_kw": yp,
        "absolute_error_kw": np.abs(yp - yt),
        "squared_error_kw": (yp - yt) ** 2,
    })


def evaluate(result, model_name, capacity_kw=CAPACITY_KW, out_dir=None, make_plots=True):
    """Compute metrics, and (if out_dir) write predictions.parquet / metrics.json / plots."""
    metrics = compute_metrics(result, capacity_kw)
    metrics["model_name"] = model_name
    preds = predictions_frame(result, model_name)
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out / "predictions.parquet", index=False)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if make_plots:
            _plots(preds, model_name, out / "plots")
    return metrics


def _plots(preds, model_name, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    yt, yp = preds["y_true_kw"].to_numpy(), preds["y_pred_kw"].to_numpy()
    hour = preds["issue_time"].dt.tz_convert(TZ).dt.hour

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(yt, yp, s=4, alpha=.2); ax[0].plot([0, 30], [0, 30], "r--", lw=1)
    ax[0].set_xlabel("actual kW"); ax[0].set_ylabel("pred kW"); ax[0].set_title(f"{model_name}: y_true vs y_pred")
    ax[1].hist(yp - yt, bins=60); ax[1].set_title("residuals (pred-actual, kW)"); ax[1].set_xlabel("kW")
    err_by_h = pd.Series(np.abs(yp - yt)).groupby(hour.to_numpy()).mean()
    ax[2].bar(err_by_h.index, err_by_h.values); ax[2].set_title("MAE by local hour"); ax[2].set_xlabel("hour")
    plt.tight_layout(); plt.savefig(plot_dir / "diagnostics.png", dpi=90); plt.close(fig)
