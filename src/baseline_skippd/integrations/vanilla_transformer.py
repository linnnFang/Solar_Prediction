"""
Integration with the repo's existing ``vanilla_transformer`` — no rewrite.

The existing ``TransformerForecaster`` only accepts a numeric sequence
[B, L, n_features] and the existing ``Trainer`` iterates ``for x, y in dl`` (see
the Phase-0 audit). This is **Case 1**: the first-version Transformer is
**PV-only** — the canonical batch's ``pv_history`` [B, T] is projected to
[B, T, 1] and predicts P_{t+15}. It does NOT use the sky images.

The adapter converts the canonical dict into the ``(x, y)`` tuples the existing
Trainer wants; ``_TupleLoader`` wraps a canonical DataLoader so the **unmodified**
Trainer can consume it. Image-feature Transformer (Case 3) would add a separate
ImageEncoder here and widen n_features — not done in this version.
"""

import torch

from src.vanilla_transformer.model import TransformerForecaster
from src.vanilla_transformer.trainer import Trainer as VanillaTrainer


class VanillaTransformerBatchAdapter:
    """canonical batch <-> the existing Transformer's (x, y) / forward."""
    name = "vanilla_transformer"

    def to_xy(self, batch):
        x = batch["pv_history"].unsqueeze(-1)   # [B, T] -> [B, T, 1]  (view; source untouched)
        return x, batch["target"]               # y: [B, 1]

    def forward(self, model, batch):
        return model(batch["pv_history"].unsqueeze(-1))     # [B, 1]


class _TupleLoader:
    """Adapt a canonical DataLoader to yield (x, y) tuples for the existing Trainer."""

    def __init__(self, loader, adapter):
        self.loader = loader
        self.adapter = adapter

    def __iter__(self):
        for batch in self.loader:
            yield self.adapter.to_xy(batch)

    def __len__(self):
        return len(self.loader)


def build_transformer(cfg, datamodule):
    """Construct the EXISTING TransformerForecaster sized to the canonical window."""
    m = cfg.get("model", {})
    return TransformerForecaster(
        n_features=1,                                   # PV-only (Case 1)
        context_length=datamodule.window_index.T,       # 16
        horizon=1,                                       # P_{t+15}
        d_model=m.get("d_model", 64), nhead=m.get("nhead", 8),
        num_layers=m.get("num_layers", 2), dim_ff=m.get("dim_ff", 128),
        dropout=m.get("dropout", 0.1),
    )


def train_transformer(model, datamodule, cfg):
    """Train via the UNMODIFIED vanilla_transformer.Trainer."""
    adapter = VanillaTransformerBatchAdapter()
    t = cfg.get("trainer", {})
    trainer = VanillaTrainer(model, lr=t.get("lr", 1e-3), weight_decay=t.get("weight_decay", 1e-4),
                             grad_clip=t.get("grad_clip", 1.0), device=cfg.get("device"))
    history = trainer.fit(
        _TupleLoader(datamodule.train_dataloader(), adapter),
        _TupleLoader(datamodule.val_dataloader(), adapter),
        epochs=t.get("max_epochs", 30), patience=t.get("patience", 6),
        verbose=cfg.get("verbose", True))
    return trainer, history
