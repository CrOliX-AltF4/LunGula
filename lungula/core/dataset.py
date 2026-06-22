"""Generic sequence dataset for imitation learning."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from .base_game import BaseReplayParser


class ReplayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset built from a list of replay files.

    Each sample is (context_window, next_action) — the model sees the last
    `window` frames and predicts the action for the next frame.

    Optional online augmentation: pass ``flip_feat_indices`` and
    ``flip_act_indices`` to enable a 50 % random sign-flip on selected
    feature/action dimensions.  For osu!, this is a vertical Y-axis mirror
    that balances the natural downward-movement bias in replays.
    """

    def __init__(
        self,
        replay_paths: list[tuple[str, str]],  # [(replay, beatmap), ...]
        parser: BaseReplayParser,
        window: int = 32,
        flip_feat_indices: tuple[int, ...] | None = None,
        flip_act_indices: tuple[int, ...] | None = None,
    ) -> None:
        self.window = window
        self._flip_feat = list(flip_feat_indices) if flip_feat_indices else []
        self._flip_act = list(flip_act_indices) if flip_act_indices else []
        self._augment = bool(self._flip_feat or self._flip_act)
        self._samples: list[tuple[npt.NDArray[Any], npt.NDArray[Any]]] = []

        for replay_path, beatmap_path in replay_paths:
            frames = parser.parse(replay_path, beatmap_path)
            if len(frames) <= window:
                continue
            features = np.stack([f.features for f in frames])
            actions = np.stack([f.action for f in frames])
            for i in range(window, len(frames)):
                self._samples.append((features[i - window : i], actions[i]))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ctx, act = self._samples[idx]
        ctx = ctx.copy()
        act = act.copy()
        if self._augment and random.random() < 0.5:
            if self._flip_feat:
                ctx[:, self._flip_feat] *= -1.0
            if self._flip_act:
                act[self._flip_act] *= -1.0
        return torch.from_numpy(ctx).float(), torch.from_numpy(act).float()
