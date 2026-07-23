"""PV-history-only MLP — a weather/image-free baseline over the [B, T] history."""

import torch.nn as nn


class PVOnlyMLP(nn.Module):
    def __init__(self, pv_len=16, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pv_len, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pv_history):        # [B, T] -> [B, 1]
        return self.net(pv_history)
