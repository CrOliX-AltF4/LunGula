"""Abstract base for all lun'imago models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseImitatonModel(ABC, nn.Module):
    """A model that predicts the next action given a context window of game frames."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, feature_dim) → (batch, action_dim)"""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...
