"""
Feature engineering for the GEFCom2014 solar table.

`FeatureBuilder` adds features with chainable `add_*` methods. To add a new
feature, write one more `add_*` method and (optionally) call it in `add_all`;
nothing else needs to change.

Expects a DataFrame with at least: ts (UTC), VAR165, VAR166, VAR169_dea.
"""

import numpy as np

from src.config import TZ_OFFSET


class FeatureBuilder:
    """
    Build solar-forecasting features on a copy of the input DataFrame.

    Example:
        df = FeatureBuilder(df).add_all().build()
    """

    def __init__(self, df, tz_offset=TZ_OFFSET):
        self.df = df.copy()
        self.tz_offset = tz_offset

    def add_local_time(self):
        """Add local calendar fields: ts_local, hour, dayofyear, month, is_daytime."""
        local = self.df["ts"] + np.timedelta64(self.tz_offset, "h")
        self.df["ts_local"] = local
        self.df["hour"] = local.dt.hour
        self.df["dayofyear"] = local.dt.dayofyear
        self.df["month"] = local.dt.month
        self.df["is_daytime"] = self.df["VAR169_dea"] > 0
        return self

    def add_cyclic_time(self):
        """Add sine/cosine encodings of hour-of-day and day-of-year."""
        hour = self.df["ts_local"].dt.hour
        doy = self.df["ts_local"].dt.dayofyear
        self.df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        self.df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
        self.df["doy_cos"] = np.cos(2 * np.pi * doy / 366)
        return self

    def add_wind(self):
        """Add 10 m wind speed from the u/v components (VAR165, VAR166)."""
        self.df["wind_speed"] = np.hypot(self.df["VAR165"], self.df["VAR166"])
        return self

    # Placeholder for later (needs site latitude + pvlib):
    # def add_solar_geometry(self): ...   # solar zenith / elevation / azimuth
    # def add_clear_sky(self): ...        # clear-sky GHI and clear-sky index

    def add_all(self):
        """Apply every feature step in dependency order."""
        return self.add_local_time().add_cyclic_time().add_wind()

    def build(self):
        """Return the DataFrame with all added features."""
        return self.df
