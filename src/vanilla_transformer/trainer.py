"""
Generic training / prediction loop for a forecasting `nn.Module`.

`Trainer` is model-agnostic: hand it any module whose forward maps
[B, L, n_features] -> [B, H]. It provides sample-weighted epoch losses, gradient
clipping, early stopping with best-weights restore, and a `predict` that can
return aligned targets (so evaluation never depends on DataLoader order).
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def pick_device(prefer=None):
    """Return the best available torch device string (mps > cuda > cpu), or `prefer`."""
    if prefer is not None:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Trainer:
    """
    Train a forecasting module with early stopping.

    Args:
        model       : any nn.Module mapping [B, L, F] -> [B, H].
        loss_fn     : loss module (default nn.MSELoss()).
        lr, weight_decay : AdamW hyperparameters.
        grad_clip   : max grad norm (None disables clipping).
        device      : device string; auto-selected when None.
    """

    def __init__(self, model, loss_fn=None, lr=1e-3, weight_decay=1e-4,
                 grad_clip=1.0, device=None):
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _run_epoch(self, dl, train, show_progress=False, description=None):
        """One pass over `dl`; returns the sample-weighted mean loss."""
        self.model.train() if train else self.model.eval()
        total, count = 0.0, 0
        batches = tqdm(
            dl,
            desc=description,
            leave=False,
            mininterval=0.5,
            disable=not show_progress,
        )
        context = torch.enable_grad() if train else torch.inference_mode()
        with context:
            for x, y in batches:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.model(x), y)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.opt.step()
                bs = x.size(0)
                total += loss.item() * bs   # weight by batch size (last batch may be smaller)
                count += bs
                if show_progress:
                    batches.set_postfix(loss=f"{total / count:.5f}")
        return total / count

    def fit(self, train_dl, val_dl, epochs=30, patience=6, verbose=True,
            show_progress=False):
        """
        Train with early stopping on the validation loss; restore best weights.
        Output: history dict with keys
            train_loss, val_loss (per-epoch lists), best_epoch, best_val_loss.
        """
        history = {"train_loss": [], "val_loss": []}
        best, best_state, best_epoch, wait = float("inf"), None, 0, 0

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            tr = self._run_epoch(
                train_dl, True, show_progress,
                description=f"epoch {ep}/{epochs} train",
            )
            va = self._run_epoch(
                val_dl, False, show_progress,
                description=f"epoch {ep}/{epochs} val",
            )
            elapsed = time.perf_counter() - t0
            history["train_loss"].append(tr)
            history["val_loss"].append(va)

            improved = va < best - 1e-6
            if improved:
                best, best_epoch, wait = va, ep, 0
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            else:
                wait += 1
            if verbose:
                print(f"epoch {ep:2d} | {elapsed:5.1f}s | train {tr:.5f} | val {va:.5f}"
                      f"{'  <- best' if improved else ''}", flush=True)
            if wait >= patience:
                if verbose:
                    print(f"early stop at epoch {ep}", flush=True)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best
        if verbose:
            print("best val loss:", round(best, 5), flush=True)
        return history

    @torch.inference_mode()
    def predict(self, dl, return_targets=False, clip=None):
        """
        Predict over `dl` (which must be shuffle=False for aligned output).
        Input : clip = (lo, hi) to clamp predictions into a range, or None.
        Output: preds [n, H]; also targets [n, H] when return_targets=True.
        """
        self.model.eval()
        preds, targets = [], []
        for x, y in dl:
            preds.append(self.model(x.to(self.device)).cpu().numpy())
            if return_targets:
                targets.append(y.numpy())
        pred = np.concatenate(preds)
        if clip is not None:
            pred = np.clip(pred, clip[0], clip[1])
        if return_targets:
            return pred, np.concatenate(targets)
        return pred
