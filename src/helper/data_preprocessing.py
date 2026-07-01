"""
Data loading for the GEFCom2014 solar dataset.
"""

import pandas as pd
from pathlib import Path

from src.config import GEFCOM_DIR


def load_zone(z, nwp_vars, data_dir=GEFCOM_DIR):
    """
    Load one assembled zone table
    output:
    - power
    - the given NWP vars
    - time-aligned).
    """
    d = pd.read_csv(Path(data_dir) / f"GEFCom2014_Zone{z}_Assembled.csv", parse_dates=["date"])
    d = d.rename(columns={"date": "ts", "Power": "POWER"})
    d = d.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    d["ZONE"] = z
    return d[["ts", "ZONE", "POWER"] + list(nwp_vars)]


def load_all_zones(nwp_vars, zones=(1, 2, 3), data_dir=GEFCOM_DIR):
    """
    Load all zones. 
    Returns (dict {zone: DataFrame}, combined long DataFrame).
    """
    zone_dfs = {z: load_zone(z, nwp_vars, data_dir) for z in zones}
    combined = pd.concat(zone_dfs.values(), ignore_index=True)
    return zone_dfs, combined
