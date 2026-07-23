"""
Notebook-friendly orchestration — thin wrappers only, no business logic.

Every function just wires together the DataModule, model factory, trainer, and
evaluator that live elsewhere. Crucially, a single ``build_skippd_datamodule``
result is shared by every model, so CNN, MLP, and the vanilla Transformer use the
**same** split manifest, window index, scaler, and test samples.
"""

from pathlib import Path

import pandas as pd
import torch

from src.baseline_skippd.config import load_skippd_config
from src.baseline_skippd.io import SKIPPDProcessedStore
from src.baseline_skippd.split import make_split
from src.baseline_skippd.windows import build_window_index
from src.baseline_skippd.scalers import build_scaler
from src.baseline_skippd.datamodule import SKIPPDDataModule
from src.baseline_skippd.adapters import CNNForecastBatchAdapter, PVOnlyBatchAdapter
from src.baseline_skippd.models import PVOnlyMLP, SunsetForecastCNN
from src.baseline_skippd.integrations import (
    VanillaTransformerBatchAdapter, build_transformer, train_transformer)
from src.baseline_skippd.training import train_torch_model, predict_kw
from src.baseline_skippd.evaluator import evaluate
from src.baseline_skippd.baselines import (
    pv_window_frame, NaivePersistence, ClearSkyModel, ClearSkyAdjustedPersistence)

__all__ = [
    "load_skippd_config", "build_skippd_datamodule", "build_model", "build_model_adapter",
    "train_model", "evaluate_model", "run_rule_baseline", "compare_runs",
    "load_existing_checkpoint", "get_sample_batch",
]


def build_skippd_datamodule(cfg):
    """Assemble the shared DataModule (store -> split -> windows -> scaler)."""
    store = SKIPPDProcessedStore(cfg["data"]["root"])
    s, w = cfg["split"], cfg["window"]
    manifest = make_split(store, strategy=s["strategy"], seed=s.get("seed", 42),
                          val_frac=s.get("val_frac", 0.15),
                          n_folds=s.get("n_folds", 10), fold=s.get("fold", 0))
    index, _ = build_window_index(store, manifest, history_minutes=w["history_minutes"],
                                  history_step_minutes=w["history_step_minutes"],
                                  horizon_minutes=w["horizon_minutes"], strides=w.get("strides"))
    scaler = build_scaler(cfg["scaler"]["name"],
                          **{k: v for k, v in cfg["scaler"].items() if k != "name"})
    t = cfg["trainer"]
    return SKIPPDDataModule(store, manifest, index, scaler, batch_size=t.get("batch_size", 256),
                            num_workers=t.get("num_workers", 0), seed=cfg.get("seed", 42),
                            device=cfg.get("device"))


def build_model_adapter(model_name):
    if model_name in ("sunset_forecast", "sunset"):
        return CNNForecastBatchAdapter()
    if model_name in ("pv_only", "pv_only_mlp"):
        return PVOnlyBatchAdapter()
    if model_name == "vanilla_transformer":
        return VanillaTransformerBatchAdapter()
    raise ValueError(f"unknown model '{model_name}'")


def build_model(model_name, cfg, dm):
    """Factory. Attaches ``.skippd`` metadata (name/adapter/kind) used by train/eval."""
    t = dm.window_index.T
    m = cfg.get("model", {})
    if model_name in ("sunset_forecast", "sunset"):
        model = SunsetForecastCNN(in_channels=t * dm.store.image_shape[2], pv_len=t,
                                  hidden=m.get("hidden", 1024), dropout=m.get("dropout", 0.4))
        kind = "torch"
    elif model_name in ("pv_only", "pv_only_mlp"):
        model = PVOnlyMLP(pv_len=t, hidden=m.get("hidden", 64), dropout=m.get("dropout", 0.0))
        kind = "torch"
    elif model_name == "vanilla_transformer":
        model = build_transformer(cfg, dm)
        kind = "transformer"
    else:
        raise ValueError(f"unknown model '{model_name}'")
    model.skippd = {"name": model_name, "adapter": build_model_adapter(model_name), "kind": kind}
    return model


def _run_dir(cfg, model_name):
    return Path(cfg["artifacts"]["dir"]) / f"{cfg.get('run_name', 'run')}_{model_name}"


def train_model(model, dm, cfg):
    """Dispatch: PV-only Transformer -> existing vanilla_transformer.Trainer; else thin trainer."""
    meta = model.skippd
    out_dir = _run_dir(cfg, meta["name"])
    if meta["kind"] == "transformer":
        _, history = train_transformer(model, dm, cfg)
        ckpt = out_dir / "checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt / "best.pt")
    else:
        _, history = train_torch_model(model, dm, cfg, meta["adapter"], out_dir=out_dir)
    return {"history": history, "run_dir": str(out_dir)}


def evaluate_model(model, dm, cfg, split="test"):
    meta = model.skippd
    result = predict_kw(model, dm, cfg, meta["adapter"], split)
    return evaluate(result, meta["name"], capacity_kw=cfg["eval"]["capacity_kw"],
                    out_dir=_run_dir(cfg, meta["name"]))


def run_rule_baseline(name, dm, cfg, split="test"):
    """Naive / clear-sky persistence — no training, image-free, full split."""
    frame = pv_window_frame(dm.store, dm.window_index, split)
    if name in ("naive", "naive_persistence"):
        y_pred, model_name = NaivePersistence().predict_kw(frame), "naive_persistence"
    elif name in ("clear_sky", "clear_sky_persistence"):
        cs = cfg.get("clear_sky", {})
        csm = ClearSkyModel.fit(dm.store, dm.split_manifest, dm.window_index,
                                quantile=cs.get("quantile", 0.95), smooth=cs.get("smooth", 15))
        y_pred = ClearSkyAdjustedPersistence(csm, cfg["eval"]["capacity_kw"]).predict_kw(frame)
        model_name = "clear_sky_persistence"
    else:
        raise ValueError(f"unknown baseline '{name}'")
    result = {"y_pred_kw": y_pred, "y_true_kw": frame["y_true_kw"], "issue_time": frame["issue_time"],
              "target_time": frame["target_time"], "sample_index": frame["sample_index"], "split": split}
    return evaluate(result, model_name, capacity_kw=cfg["eval"]["capacity_kw"],
                    out_dir=_run_dir(cfg, model_name))


def compare_runs(metrics_list):
    """Tidy comparison table from a list of metrics dicts."""
    rows = [{"model": m["model_name"], "acc_rmse": m["overall"]["acc_rmse"],
             "acc_mae": m["overall"]["acc_mae"], "rmse_kw": m["overall"]["rmse_kw"],
             "mae_kw": m["overall"]["mae_kw"], "monthly_acc_rmse": m["monthly_mean_acc_rmse"],
             "n": m["overall"]["n"]} for m in metrics_list]
    return pd.DataFrame(rows).sort_values("acc_rmse", ascending=False).reset_index(drop=True)


def load_existing_checkpoint(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def get_sample_batch(dm, split="train"):
    if not dm._datasets:
        dm.setup()
    loader = {"train": dm.train_dataloader, "validation": dm.val_dataloader,
              "test": dm.test_dataloader}[split]()
    return next(iter(loader))
