"""
SUNSET Forecast CNN (Nie et al.) — 15-min-ahead PV from stacked sky frames + PV.

Input via the CNN adapter:
    images_stacked : [B, T*C, H, W] = [B, 48, 64, 64]   (T=16 frames, C=3)
    pv_history     : [B, T]         = [B, 16]

Two conv blocks -> flatten -> concat the PV history -> two dense layers -> 1.
"""

import torch
import torch.nn as nn


class SunsetForecastCNN(nn.Module):
    def __init__(self, in_channels=48, pv_len=16, hidden=1024, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(24), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(48), nn.MaxPool2d(2),
        )
        self.flat = nn.Flatten()                         # 64->32->16 spatial => 48*16*16 = 12288
        self.head = nn.Sequential(
            nn.Linear(48 * 16 * 16 + pv_len, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, images_stacked, pv_history):
        z = self.flat(self.features(images_stacked))     # [B, 12288]
        z = torch.cat([z, pv_history], dim=1)            # [B, 12288 + 16]
        return self.head(z)                              # [B, 1]
