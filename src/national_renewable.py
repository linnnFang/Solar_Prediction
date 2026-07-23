"""Historical-only baselines for the Chinese State Grid solar-station data.

This module contains dataset-specific preparation only.  Models remain in the
reusable ``src.helper.models`` and ``src.vanilla_transformer`` packages.

Forecast contract used here
---------------------------
At forecast origin ``t`` only observations at or before ``t`` are inputs.  The
target is power at ``t + horizon * 15 minutes``.  No target-time measured
weather is used.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


RAW_SOLAR_DIR = (
    Path(__file__).resolve().parents[1]
    / "data/raw/Renewable"
    / "Renewable-energy-generation-input-feature-variables-analysis-main"
    / "data_original/solar_stations"
)

WEATHER_COLS = ["irr_total", "dni", "ghi", "temp", "pressure", "humidity"]
IRRADIANCE_COLS = ["irr_total", "dni", "ghi"]


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        name = str(col).strip().lower()
        if name.startswith("time"):
            mapping[col] = "ts"
        elif "total solar" in name:
            mapping[col] = "irr_total"
        elif "direct normal" in name:
            mapping[col] = "dni"
        elif "global hori" in name:  # site 2 spells horizontal as horicontal
            mapping[col] = "ghi"
        elif "temperature" in name:
            mapping[col] = "temp"
        elif "atmosphere" in name:
            mapping[col] = "pressure"
        elif "humidity" in name:
            mapping[col] = "humidity"
        elif "power" in name:
            mapping[col] = "power"
    return df.rename(columns=mapping)


def load_raw_solar(raw_dir=RAW_SOLAR_DIR, sites=None) -> pd.DataFrame:
    """Load and harmonise raw station files without imputing or filtering rows."""
    selected = None if sites is None else set(map(int, sites))
    frames = []
    for path in sorted(glob.glob(str(Path(raw_dir) / "*.xlsx"))):
        site = int(re.search(r"site (\d+)", path).group(1))
        if selected is not None and site not in selected:
            continue
        capacity = float(re.search(r"capacity-(\d+)MW", path).group(1))
        frame = _rename_columns(pd.read_excel(path, sheet_name=0))
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        for col in WEATHER_COLS + ["power"]:
            if col not in frame:
                frame[col] = np.nan
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["site"] = site
        frame["capacity"] = capacity
        frames.append(frame[["ts", *WEATHER_COLS, "power", "site", "capacity"]])
    if not frames:
        raise FileNotFoundError(f"no selected solar-station files found in {raw_dir}")
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["ts"])
        .sort_values(["site", "ts"])
        .drop_duplicates(["site", "ts"], keep="first")
        .reset_index(drop=True)
    )


def regularise_15min_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Insert missing timestamps inside each station's observed time span."""
    pieces = []
    for site, station in df.groupby("site", sort=True):
        station = station.sort_values("ts")
        grid = pd.date_range(station["ts"].min(), station["ts"].max(), freq="15min")
        capacity = float(station["capacity"].iloc[0])
        station = station.set_index("ts").reindex(grid).rename_axis("ts").reset_index()
        station["site"] = int(site)
        station["capacity"] = capacity
        pieces.append(station)
    return pd.concat(pieces, ignore_index=True).sort_values(["site", "ts"]).reset_index(drop=True)


def clean_physical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply broad physical checks while retaining missingness indicators.

    Negative irradiance and negative generation are clipped to zero.  Grossly
    impossible high/low weather readings are made missing and later imputed
    from training data only.  The un-imputed ``cf`` column remains the target.
    """
    out = df.copy()
    original = out[WEATHER_COLS + ["power"]].copy()

    for col in IRRADIANCE_COLS:
        out[col] = out[col].clip(lower=0)
        out.loc[out[col] > 2000, col] = np.nan
    out.loc[~out["temp"].between(-50, 60), "temp"] = np.nan
    out.loc[~out["pressure"].between(300, 1200), "pressure"] = np.nan
    out.loc[~out["humidity"].between(0, 100), "humidity"] = np.nan

    # We forecast generated active power, so small negative inverter self-draw
    # is represented as zero generation.  Over-capacity values are invalid.
    out["power"] = out["power"].clip(lower=0)
    out.loc[out["power"] > out["capacity"], "power"] = np.nan
    out["cf"] = out["power"] / out["capacity"]

    for col in WEATHER_COLS:
        out[f"{col}_missing"] = (original[col].isna() | out[col].isna()).astype("int8")
    out["power_missing"] = (original["power"].isna() | out["power"].isna()).astype("int8")
    return out


def _add_time_and_site_features(df: pd.DataFrame, site_levels) -> pd.DataFrame:
    out = df.copy()
    minute = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    doy = out["ts"].dt.dayofyear
    out["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for site in site_levels:
        out[f"site_{site}"] = (out["site"] == site).astype("int8")
    return out


def prepare_historical_data(
    raw_dir=RAW_SOLAR_DIR,
    sites=(1,),
    train_end="2019-10-01",
    val_end="2020-01-01",
):
    """Load, clean and leakage-safely impute inputs for historical forecasting.

    Weather medians and the fallback CF median are fitted only on the training
    period.  ``cf`` itself is never imputed because it is the prediction target.
    ``cf_input`` is a past-only input: up to four missing steps are forward-filled,
    then a training-period station median is used as a fallback.

    Returns ``(prepared_dataframe, imputation_summary)``.
    """
    raw = regularise_15min_grid(load_raw_solar(raw_dir, sites=sites))
    clean = clean_physical_values(raw)
    site_levels = sorted(clean["site"].unique().tolist())
    clean = _add_time_and_site_features(clean, site_levels)

    train_mask = clean["ts"] < pd.Timestamp(train_end)
    if not train_mask.any():
        raise ValueError("training period is empty; check train_end")

    summary = {}
    for col in WEATHER_COLS:
        by_site = clean.loc[train_mask].groupby("site")[col].median()
        global_median = float(clean.loc[train_mask, col].median())
        if not np.isfinite(global_median):
            raise ValueError(f"training data has no finite values for {col}")
        clean[col] = clean[col].fillna(clean["site"].map(by_site)).fillna(global_median)
        summary[col] = {"global_train_median": global_median, "site_train_medians": by_site.to_dict()}

    # Forward fill is causal.  The median fallback is learned from train only.
    clean["cf_input"] = clean.groupby("site", sort=False)["cf"].ffill(limit=4)
    cf_by_site = clean.loc[train_mask].groupby("site")["cf"].median()
    cf_global = float(clean.loc[train_mask, "cf"].median())
    clean["cf_input"] = clean["cf_input"].fillna(clean["site"].map(cf_by_site)).fillna(cf_global)
    summary["cf_input"] = {
        "global_train_median": cf_global,
        "site_train_medians": cf_by_site.to_dict(),
    }

    clean["split"] = np.select(
        [clean["ts"] < pd.Timestamp(train_end), clean["ts"] < pd.Timestamp(val_end)],
        ["train", "val"],
        default="test",
    )
    return clean, summary


def make_xgb_historical_table(df: pd.DataFrame, horizon=1):
    """Create tabular origin-time features for a direct historical-only forecast."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive number of 15-minute steps")
    out = df.sort_values(["site", "ts"]).copy()
    groups = out.groupby("site", sort=False)

    out["target_cf"] = groups["cf"].shift(-horizon)
    out["target_ts"] = groups["ts"].shift(-horizon)
    # Assign samples by target time, not origin time, so a boundary target can
    # never leak into the preceding training split.
    out["split"] = groups["split"].shift(-horizon)
    expected_delta = pd.Timedelta(minutes=15 * horizon)
    out.loc[(out["target_ts"] - out["ts"]) != expected_delta, "target_cf"] = np.nan

    lag_columns = ["cf_input", "irr_total", "dni", "ghi"]
    for col in lag_columns:
        for lag in (1, 4, 8, 96):
            out[f"{col}_lag{lag}"] = groups[col].shift(lag)

    for window in (4, 16, 96):
        out[f"cf_roll{window}_mean"] = groups["cf_input"].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        out[f"cf_roll{window}_std"] = groups["cf_input"].transform(
            lambda x: x.rolling(window, min_periods=2).std()).fillna(0)
        out[f"irr_roll{window}_mean"] = groups["irr_total"].transform(
            lambda x: x.rolling(window, min_periods=1).mean())

    target_minute = out["target_ts"].dt.hour * 60 + out["target_ts"].dt.minute
    target_doy = out["target_ts"].dt.dayofyear
    out["target_tod_sin"] = np.sin(2 * np.pi * target_minute / 1440)
    out["target_tod_cos"] = np.cos(2 * np.pi * target_minute / 1440)
    out["target_doy_sin"] = np.sin(2 * np.pi * target_doy / 365.25)
    out["target_doy_cos"] = np.cos(2 * np.pi * target_doy / 365.25)
    out["lead_steps"] = int(horizon)

    site_cols = sorted(c for c in out if c.startswith("site_"))
    missing_cols = [f"{c}_missing" for c in WEATHER_COLS]
    feature_cols = [
        "cf_input", *WEATHER_COLS, *missing_cols,
        "target_tod_sin", "target_tod_cos", "target_doy_sin", "target_doy_cos",
        "lead_steps", *site_cols,
    ]
    feature_cols += [f"{c}_lag{lag}" for c in lag_columns for lag in (1, 4, 8, 96)]
    feature_cols += [
        f"cf_roll{w}_{stat}" for w in (4, 16, 96) for stat in ("mean", "std")
    ]
    feature_cols += [f"irr_roll{w}_mean" for w in (4, 16, 96)]

    valid = out["target_cf"].notna() & out[feature_cols].notna().all(axis=1)
    return out.loc[valid].reset_index(drop=True), feature_cols


def transformer_feature_columns(df: pd.DataFrame):
    """Return the common feature order used by the vanilla Transformer."""
    site_cols = sorted(c for c in df if c.startswith("site_"))
    missing_cols = [f"{c}_missing" for c in WEATHER_COLS]
    return [
        "cf_input", *WEATHER_COLS, *missing_cols,
        "tod_sin", "tod_cos", "doy_sin", "doy_cos", *site_cols,
    ]


def make_contiguous_frames(df, split, feature_cols, min_length=2):
    """Build per-site contiguous frames without missing targets or inputs."""
    subset = df[df["split"] == split].sort_values(["site", "ts"]).copy()
    valid = subset["cf"].notna() & subset[feature_cols].notna().all(axis=1)
    subset = subset.loc[valid]
    frames = []
    for _, station in subset.groupby("site", sort=True):
        breaks = station["ts"].diff().ne(pd.Timedelta("15min"))
        for _, block in station.groupby(breaks.cumsum()):
            if len(block) >= min_length:
                frames.append(block.reset_index(drop=True))
    return frames


def window_prediction_frame(frames, dataset, y_true, y_pred):
    """Attach timestamps/site/capacity to flattened WindowDataset predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.shape[0] != len(dataset):
        raise ValueError("prediction arrays are not aligned with dataset")
    records = []
    for row, (frame_id, target_start) in enumerate(dataset.index):
        frame = frames[frame_id]
        for step in range(dataset.horizon):
            meta = frame.iloc[target_start + step]
            records.append({
                "site": int(meta["site"]),
                "capacity": float(meta["capacity"]),
                "target_ts": meta["ts"],
                "horizon_step": step + 1,
                "y_true": float(y_true[row, step]),
                "y_pred": float(y_pred[row, step]),
            })
    return pd.DataFrame(records)
