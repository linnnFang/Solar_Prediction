"""Paths from config.yaml."""

from pathlib import Path
import yaml

_cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())

GEFCOM_DIR = Path(_cfg["gefcom_dir"])
SOLAR_DIR = GEFCOM_DIR / "GEFCom2014-S_V2" / "Solar"
PROCESSED_DIR = Path(_cfg["processed_dir"])
