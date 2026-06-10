"""LSTM-based imitation learning agent — default model for all games."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from .base_model import BaseImitatonModel


class LSTMAgent(BaseImitatonModel):
    """2-layer LSTM that maps a context window of game frames to the next action.

    Architecture: LSTM → LayerNorm → Linear → action
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._feature_dim = feature_dim
        self._action_dim = action_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, action_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, feature_dim)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # take last timestep
        return cast(torch.Tensor, self.head(self.norm(last)))
