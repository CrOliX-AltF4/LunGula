"""LSTM-based imitation learning agent — default model for all games."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from .base_model import BaseImitatonModel


class _LSTMCell(nn.Module):
    """Single LSTM cell implemented via Linear + elementwise ops.

    Uses only matmul, sigmoid, tanh, and elementwise mul/add — ops that are
    natively supported on DirectML (AMD/Intel on Windows) without CPU fallback.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gates(torch.cat([x, h], dim=-1))
        # split with explicit sizes → ONNX Split opset 13 (chunk → num_outputs needs opset 18)
        i, f, g, o = gates.split(self.hidden_size, dim=-1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class LSTMAgent(BaseImitatonModel):
    """2-layer LSTM that maps a context window of game frames to the next action.

    Architecture: LSTM (manual cells) → LayerNorm → Linear → action

    Uses explicit LSTM cells instead of nn.LSTM to avoid aten::_thnn_fused_lstm_cell
    which is unsupported on DirectML and falls back to a missing CPU kernel.
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
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        cells: list[nn.Module] = []
        for layer in range(num_layers):
            in_size = feature_dim if layer == 0 else hidden_size
            cells.append(_LSTMCell(in_size, hidden_size))
        self.lstm_cells = nn.ModuleList(cells)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 and num_layers > 1 else nn.Identity()
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
        batch_size, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        h = [torch.zeros(batch_size, self._hidden_size, device=device, dtype=dtype)
             for _ in range(self._num_layers)]
        c = [torch.zeros(batch_size, self._hidden_size, device=device, dtype=dtype)
             for _ in range(self._num_layers)]

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer, cell in enumerate(self.lstm_cells):
                assert isinstance(cell, _LSTMCell)
                h[layer], c[layer] = cell(inp, h[layer], c[layer])
                inp = self.drop(h[layer]) if layer < self._num_layers - 1 else h[layer]

        return cast(torch.Tensor, self.head(self.norm(h[-1])))
