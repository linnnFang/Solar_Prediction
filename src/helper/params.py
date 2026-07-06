"""
Load model hyperparameters from configs/model_params.yaml.

Keeps tuning in one editable file instead of scattered across notebooks:
    model = build_model("xgboost", **get_params("xgboost"))
"""

from pathlib import Path
import yaml

PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_params.yaml"


def get_params(preset, path=PARAMS_PATH):
    """
    Return the hyperparameter dict for a named preset.
    Input : preset name (top-level key in the YAML); optional path override.
    Output: a fresh dict of parameters to pass to build_model.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    if preset not in cfg:
        raise KeyError(f"no preset '{preset}' in {path}; available: {list(cfg)}")
    return dict(cfg[preset])
