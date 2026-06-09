"""Abstract interfaces every game plugin must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class GameFrame:
    """One timestep of game state + the human action taken at that timestep."""
    features: np.ndarray   # encoded game state vector
    action:   np.ndarray   # encoded action vector (cursor + clicks, etc.)
    timestamp_ms: float


class BaseReplayParser(ABC):
    """Parses raw replay files into a list of GameFrames."""

    @abstractmethod
    def parse(self, replay_path: str, beatmap_path: str) -> list[GameFrame]:
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector output by this parser."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action vector output by this parser."""
        ...


class BaseGameEncoder(ABC):
    """Encodes raw game data into normalized feature vectors."""

    @abstractmethod
    def encode_state(self, raw: dict) -> np.ndarray:
        ...

    @abstractmethod
    def encode_action(self, raw: dict) -> np.ndarray:
        ...
