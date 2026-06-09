"""osu! plugin entry point — wires parser + model for the CLI."""
from __future__ import annotations
import os
from .parser import OsuReplayParser, _FEATURE_DIM, _ACTION_DIM
from ...core.models.lstm_agent import LSTMAgent


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
            pairs.append((
                os.path.join(root, osrs[0]),
                os.path.join(root, osus[0]),
            ))
    return pairs
