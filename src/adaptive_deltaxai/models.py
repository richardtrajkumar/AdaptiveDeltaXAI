"""Detector architectures used by the canonical Master v1.7 protocol."""

from __future__ import annotations

import torch
import torch.nn as nn


class DataPlaneGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 gru_dropout: float = 0.20, head_dropout: float = 0.10):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=n_layers, batch_first=True,
                          dropout=gru_dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(head_dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class DataPlaneTransformer(nn.Module):
    """Small causal-free window classifier used only with the GRU protocol."""
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=d_model * 2,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.input(x))[:, -1])

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class ReconGRU(nn.Module):
    """Reconstruction detector used only for the public SMD/PSM datasets."""
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True,
                          dropout=0.2)
        self.decoder = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                     nn.Linear(32, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.decoder(hidden[-1])

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Stabilized reconstruction anomaly score used for public XAI.

        The raw reconstruction loss can span several orders of magnitude
        across SMD machines.  ``log1p`` is a fixed, monotone and
        differentiable score transformation; it is part of the detector
        definition, rather than a post-hoc clipping or a metric repair.
        """
        loss = (self.forward(x) - x[:, -1, :]).square().mean(dim=1)
        return torch.log1p(loss)
