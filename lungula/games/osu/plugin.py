"""osu! plugin entry point — wires parser + model for the CLI."""

from __future__ import annotations

import os

from ...core.models.lstm_agent import LSTMAgent
from .parser import _ACTION_DIM, _FEATURE_DIM, OsuReplayParser

# Vertical Y-axis flip augmentation indices for the osu! feature/action layout.
# Feature vector (dim=10): [0]dx [1]dy [2]timeToNext [3]n0x [4]n0y [5]n1x [6]n1y [7]n2x [8]n2y [9]combo
# Action vector (dim=4):   [0]cursorDx [1]cursorDy [2]lClick [3]rClick
FLIP_FEAT_Y: tuple[int, ...] = (1, 4, 6, 8)
FLIP_ACT_Y: tuple[int, ...] = (1,)


def make_parser() -> OsuReplayParser:
    return OsuReplayParser()


def make_model() -> LSTMAgent:
    return LSTMAgent(feature_dim=_FEATURE_DIM, action_dim=_ACTION_DIM)


def collect_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Scan data_dir for (.osr, .osu) pairs.

    Expected layout:
      data_dir/
        map_001/
          replay.osr
          beatmap.osu
        map_002/
          ...
    """
    pairs: list[tuple[str, str]] = []
    for root, _, files in os.walk(data_dir):
        osrs = [f for f in files if f.endswith(".osr")]
        osus = [f for f in files if f.endswith(".osu")]
        if osrs and osus:
            pairs.append(
                (
                    os.path.join(root, osrs[0]),
                    os.path.join(root, osus[0]),
                )
            )
    return pairs
