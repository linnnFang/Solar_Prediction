"""
Assemble the GEFCom2014 Task-15 solar data into one clean, analysis-ready table.

Task 15 is the final rolling task and covers the whole competition span. Three
source files are combined per zone:
  - train15.csv      : POWER truth for the historical period
  - Solution ...csv  : POWER truth for the forecast month (2014-06)
  - predictors15.csv : the 12 NWP weather variables

The four accumulated NWP fields (radiation/precipitation, reset daily at 01:00)
are de-accumulated into true hourly values in separate `*_dea` columns.
"""

import pandas as pd

from src.config import SOLAR_DIR, PROCESSED_DIR

NWP_VARS = ["VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165",
            "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228"]

# NWP fields accumulated since the daily forecast start; need de-accumulating.
ACCUM_VARS = ["VAR169", "VAR175", "VAR178", "VAR228"]

ZONES = (1, 2, 3)


def _read(path):
    """Read one Task-15 csv and parse its TIMESTAMP column into a `ts` datetime."""
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d %H:%M")
    return df.rename(columns={"ZONEID": "ZONE"})


class GEFComTask15:
    """
    Load, merge, clean and de-accumulate the GEFCom2014 Task-15 solar data.

    Typical use:
        ds = GEFComTask15().load().deaccumulate()
        df = ds.combined          # long table, all zones
        ds.to_processed()         # save to data/process
    """

    def __init__(self, solar_dir=SOLAR_DIR):
        self.solar_dir = solar_dir
        self.data = None  # combined long table, filled by load()

    def load(self):
        """
        Build the combined POWER + weather table for all zones.

        POWER truth = train15 (history) + Solution (forecast month), merged onto
        the predictors15 weather grid. The result is continuous hourly, one row
        per (zone, hour), with no missing or duplicate timestamps.
        """
        task = self.solar_dir / "Task 15"
        weather = _read(task / "predictors15.csv")
        train = _read(task / "train15.csv")
        solution = _read(self.solar_dir / "Solution to Task 15" / "Solution to Task 15.csv")

        power = pd.concat([train, solution], ignore_index=True)[["ZONE", "ts", "POWER"]]
        power = power.drop_duplicates(subset=["ZONE", "ts"])

        df = weather[["ZONE", "ts"] + NWP_VARS].merge(power, on=["ZONE", "ts"], how="left")

        self._check_power(df, weather)
        self.data = df[["ts", "ZONE", "POWER"] + NWP_VARS].sort_values(
            ["ZONE", "ts"]).reset_index(drop=True)
        return self

    @staticmethod
    def _check_power(df, weather):
        """Fail loudly if the merged POWER disagrees with predictors15's own POWER."""
        assert df["POWER"].notna().all(), "POWER has gaps after the merge"
        ref = weather.set_index(["ZONE", "ts"])["POWER"]
        merged = df.set_index(["ZONE", "ts"])["POWER"]
        assert (merged - ref).abs().max() < 1e-6, "POWER disagrees with predictors15"

    def deaccumulate(self):
        """
        Convert the accumulated NWP fields into true hourly values.

        Each field accumulates within a forecast day that runs 01:00 -> next 00:00
        and resets at 01:00. Grouping by (zone, forecast_day) and differencing
        gives the hourly increment; the first step keeps its raw value and any
        negative noise is clipped to 0. Results go into new `*_dea` columns.
        """
        df = self.data
        forecast_day = (df["ts"] - pd.Timedelta("1h")).dt.normalize()
        groups = df.groupby(["ZONE", forecast_day])
        for var in ACCUM_VARS:
            hourly = groups[var].diff()
            hourly = hourly.fillna(df[var]).clip(lower=0)
            df[f"{var}_dea"] = hourly
        return self

    @property
    def combined(self):
        """The full long table (all zones stacked)."""
        return self.data

    def per_zone(self):
        """Return {zone: DataFrame} split by zone."""
        return {z: g.reset_index(drop=True) for z, g in self.data.groupby("ZONE")}

    def to_processed(self, filename="gefcom_task15_clean.parquet"):
        """Save the combined table to the processed data directory and return its path."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / filename
        self.data.to_parquet(path, index=False)
        return path


def load_all_zones(nwp_vars=NWP_VARS, zones=ZONES):
    """
    Backwards-compatible loader.

    Returns (dict {zone: DataFrame}, combined long DataFrame) with columns
    ts, ZONE, POWER and the requested NWP vars, sourced from the clean Task-15
    assembly.
    """
    ds = GEFComTask15().load()
    cols = ["ts", "ZONE", "POWER"] + list(nwp_vars)
    combined = ds.combined[ds.combined["ZONE"].isin(zones)][cols].reset_index(drop=True)
    zone_dfs = {z: g.reset_index(drop=True) for z, g in combined.groupby("ZONE")}
    return zone_dfs, combined
