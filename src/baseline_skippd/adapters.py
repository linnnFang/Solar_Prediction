"""
Model adapters — translate the canonical batch into each model's input.

The canonical dataset always returns the same dict (images [B,T,C,H,W],
pv_history [B,T], target [B,1], ...). Adapters do the model-specific reshaping so
the dataset never has to. Each adapter exposes ``forward(model, batch) -> pred``
and never mutates the source batch (reshapes/views only).

The Transformer's adapter lives in ``integrations/vanilla_transformer.py`` so no
SKIPP'D field names leak into that model.
"""


class CNNForecastBatchAdapter:
    """SUNSET-style CNN: stack the T frames along channels -> [B, T*C, H, W]."""
    name = "sunset_forecast"

    def to_inputs(self, batch):
        imgs = batch["images"]                       # [B, T, C, H, W]
        b, t, c, h, w = imgs.shape
        return {"images_stacked": imgs.reshape(b, t * c, h, w),   # view, source untouched
                "pv_history": batch["pv_history"]}

    def forward(self, model, batch):
        ins = self.to_inputs(batch)
        return model(ins["images_stacked"], ins["pv_history"])    # -> [B, 1]


class PVOnlyBatchAdapter:
    """PV-history-only models (MLP): pass the [B, T] power history straight through."""
    name = "pv_only"

    def to_inputs(self, batch):
        return {"pv_history": batch["pv_history"]}   # [B, T]

    def forward(self, model, batch):
        return model(batch["pv_history"])            # -> [B, 1]
