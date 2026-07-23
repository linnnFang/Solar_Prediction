"""
Thin training / prediction loop for models that need the canonical dict batch.

The existing ``vanilla_transformer.Trainer`` only handles single-tensor ``(x, y)``
inputs, so the SUNSET CNN (which needs *both* images and pv_history) cannot use
it. This thin loop trains any model through its adapter over the canonical
batches, reusing the same recipe (AdamW, MSE, grad clip, early stopping,
best-weights restore) and ``pick_device`` from the existing trainer. The PV-only
Transformer still trains through the UNMODIFIED existing Trainer (see
``integrations/vanilla_transformer.py``).

``predict_kw`` is uniform for every model: it runs an unshuffled loader, applies
the model via its adapter, inverse-transforms to kW, and keeps issue/target times
and sample ids aligned — so every model is evaluated on identical samples.
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.vanilla_transformer.trainer import pick_device


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_torch_model(model, datamodule, cfg, adapter, out_dir=None):
    """Train a model over canonical batches; return (model, history). Saves best/last if out_dir."""
    t = cfg.get("trainer", {})
    device = pick_device(cfg.get("device"))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t.get("lr", 1e-3),
                           weight_decay=t.get("weight_decay", 0.0))
    loss_fn = nn.MSELoss()
    grad_clip = t.get("grad_clip", None)
    epochs, patience = t.get("max_epochs", 30), t.get("patience", 6)
    verbose = cfg.get("verbose", True)

    train_dl, val_dl = datamodule.train_dataloader(), datamodule.val_dataloader()
    history = {"train_loss": [], "val_loss": []}
    best, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for ep in range(1, epochs + 1):
        tr = _run_epoch(model, train_dl, adapter, loss_fn, device, opt, grad_clip)
        va = _run_epoch(model, val_dl, adapter, loss_fn, device, None, None)
        history["train_loss"].append(tr); history["val_loss"].append(va)
        improved = va < best - 1e-9
        if improved:
            best, best_epoch, wait = va, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
        if verbose:
            print(f"epoch {ep:3d} | train {tr:.5f} | val {va:.5f}{'  <- best' if improved else ''}")
        if wait >= patience:
            if verbose:
                print(f"early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"], history["best_val_loss"] = best_epoch, best
    if out_dir is not None:
        ckpt = Path(out_dir) / "checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt / "best.pt")
        torch.save(model.state_dict(), ckpt / "last.pt")
    return model, history


def _run_epoch(model, dl, adapter, loss_fn, device, opt, grad_clip):
    train = opt is not None
    model.train() if train else model.eval()
    total, count = 0.0, 0
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for batch in dl:
            batch = _move(batch, device)
            pred = adapter.forward(model, batch)
            loss = loss_fn(pred, batch["target"])
            if train:
                opt.zero_grad(); loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            bs = batch["target"].shape[0]
            total += loss.item() * bs; count += bs
    return total / max(count, 1)


@torch.inference_mode()
def predict_kw(model, datamodule, cfg, adapter, split="test"):
    """Run one split unshuffled; return kW predictions + aligned metadata."""
    device = pick_device(cfg.get("device"))
    model = model.to(device).eval()
    scaler = datamodule.scaler
    loader = {"train": datamodule.train_dataloader, "validation": datamodule.val_dataloader,
              "test": datamodule.test_dataloader}[split]()
    preds, trues, issue, target, sidx = [], [], [], [], []
    for batch in loader:
        pred = adapter.forward(model, _move(batch, device)).cpu().numpy().reshape(-1)
        preds.append(scaler.inverse_transform(pred))
        trues.append(scaler.inverse_transform(batch["target"].numpy().reshape(-1)))
        issue.append(batch["issue_time"].numpy()); target.append(batch["target_time"].numpy())
        sidx.append(np.asarray(batch["sample_index"]))
    return {
        "y_pred_kw": np.concatenate(preds).astype(np.float32),
        "y_true_kw": np.concatenate(trues).astype(np.float32),
        "issue_time": np.concatenate(issue), "target_time": np.concatenate(target),
        "sample_index": np.concatenate(sidx), "split": split,
    }
