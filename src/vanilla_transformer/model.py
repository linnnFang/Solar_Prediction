"""
Encoder-only Transformer for multivariate time-series forecasting.

`TransformerForecaster` maps a lookback window of features to the full horizon
in one shot (direct multi-step):

    input  x    : [batch_size, context_length, n_features]
    output pred : [batch_size, horizon]

Pipeline: linear input projection -> sinusoidal positional encoding ->
`nn.TransformerEncoder` (batch_first) -> mean-pool over time -> linear head.
The model is dataset-agnostic; n_features / context_length / horizon come from
the dataset.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the projected inputs."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))       # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]                 # add positions (broadcast over batch)


class TransformerForecaster(nn.Module):
    """
    Standard encoder-only Transformer forecaster.

    Args (keyword-only recommended):
        n_features      : number of input channels per timestep.
        context_length  : lookback length L (used to size positional encoding).
        horizon         : forecast length H (output width).
        d_model, nhead, num_layers, dim_ff, dropout : usual Transformer knobs;
                          d_model must be divisible by nhead.
    """

    def __init__(self, n_features, context_length, horizon,
                 d_model=64, nhead=8, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.context_length = context_length
        self.horizon = horizon

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max(context_length, 1))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        """x: [B, context_length, n_features] -> pred: [B, horizon]."""
        x = self.pos(self.input_proj(x))      # [B, L, d_model]
        z = self.encoder(x).mean(dim=1)       # pool over time -> [B, d_model]
        return self.head(z)                   # [B, horizon]
