"""Generic sequence dataset for imitation learning."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_game import BaseReplayParser


class ReplayDataset(Dataset):
    """Sliding-window dataset built from a list of replay files.

    Each sample is (context_window, next_action) — the model sees the last
    `window` frames and predicts the action for the next frame.
    """

    def __init__(
        self,
        replay_paths: list[tuple[str, str]],  # [(replay, beatmap), ...]
        parser: BaseReplayParser,
        window: int = 32,
    ) -> None:
        self.window = window
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []

        for replay_path, beatmap_path in replay_paths:
            frames = parser.parse(replay_path, beatmap_path)
            if len(frames) <= window:
                continue
            features = np.stack([f.features for f in frames])
            actions  = np.stack([f.action   for f in frames])
            for i in range(window, len(frames)):
                self._samples.append((features[i - window:i], actions[i]))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ctx, act = self._samples[idx]
        return torch.from_numpy(ctx).float(), torch.from_numpy(act).float()
