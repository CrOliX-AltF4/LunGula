"""Tests for core.dataset — ReplayDataset sliding-window logic."""
import numpy as np
import torch
import pytest
from lunaimago.core.base_game import BaseReplayParser, GameFrame
from lunaimago.core.dataset import ReplayDataset

_FEATURE_DIM = 6
_ACTION_DIM  = 3
_WINDOW      = 8


class _FakeParser(BaseReplayParser):
    """Returns N synthetic frames; replay_path is used as the frame count string."""

    @property
    def feature_dim(self) -> int:
        return _FEATURE_DIM

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    def parse(self, replay_path: str, beatmap_path: str) -> list[GameFrame]:
        n = int(replay_path)
        return [
            GameFrame(
                features=np.full(_FEATURE_DIM, float(i), dtype=np.float32),
                action=np.full(_ACTION_DIM, float(i), dtype=np.float32),
                timestamp_ms=float(i * 16),
            )
            for i in range(n)
        ]


@pytest.fixture()
def parser() -> _FakeParser:
    return _FakeParser()


class TestReplayDataset:
    def test_length_single_replay(self, parser: _FakeParser) -> None:
        ds = ReplayDataset([("50", "map.osu")], parser, window=_WINDOW)
        assert len(ds) == 50 - _WINDOW

    def test_length_multiple_replays(self, parser: _FakeParser) -> None:
        ds = ReplayDataset([("50", ""), ("30", "")], parser, window=_WINDOW)
        assert len(ds) == (50 - _WINDOW) + (30 - _WINDOW)

    def test_replay_too_short_is_skipped(self, parser: _FakeParser) -> None:
        # Exactly window length — must be skipped (condition: len > window)
        ds = ReplayDataset([("8", "")], parser, window=_WINDOW)
        assert len(ds) == 0

    def test_one_frame_longer_than_window_gives_one_sample(self, parser: _FakeParser) -> None:
        ds = ReplayDataset([("9", "")], parser, window=_WINDOW)
        assert len(ds) == 1

    def test_getitem_shapes(self, parser: _FakeParser) -> None:
        ds = ReplayDataset([("50", "")], parser, window=_WINDOW)
        ctx, act = ds[0]
        assert isinstance(ctx, torch.Tensor)
        assert isinstance(act, torch.Tensor)
        assert ctx.shape == (_WINDOW, _FEATURE_DIM)
        assert act.shape == (_ACTION_DIM,)

    def test_getitem_dtype_float32(self, parser: _FakeParser) -> None:
        ds = ReplayDataset([("20", "")], parser, window=_WINDOW)
        ctx, act = ds[0]
        assert ctx.dtype == torch.float32
        assert act.dtype == torch.float32

    def test_context_window_is_sequential(self, parser: _FakeParser) -> None:
        ds = ReplayDataset([("20", "")], parser, window=_WINDOW)
        ctx, act = ds[0]
        # Frame 0..7 are context, frame 8 is the target action
        # Feature value at frame i = i (from FakeParser)
        for i in range(_WINDOW):
            assert ctx[i, 0].item() == pytest.approx(float(i))
        assert act[0].item() == pytest.approx(float(_WINDOW))

    def test_empty_pairs(self, parser: _FakeParser) -> None:
        ds = ReplayDataset([], parser, window=_WINDOW)
        assert len(ds) == 0
