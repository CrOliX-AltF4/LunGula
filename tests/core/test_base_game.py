"""Tests for core.base_game — GameFrame, BaseReplayParser contract."""
import numpy as np
import pytest
from lunimago.core.base_game import GameFrame, BaseReplayParser, BaseGameEncoder


class _ConcreteParser(BaseReplayParser):
    @property
    def feature_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2

    def parse(self, replay_path: str, beatmap_path: str) -> list[GameFrame]:
        return [
            GameFrame(
                features=np.zeros(4, dtype=np.float32),
                action=np.ones(2, dtype=np.float32),
                timestamp_ms=0.0,
            )
        ]


class _ConcreteEncoder(BaseGameEncoder):
    def encode_state(self, raw: dict) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    def encode_action(self, raw: dict) -> np.ndarray:
        return np.ones(2, dtype=np.float32)


class TestGameFrame:
    def test_construction(self) -> None:
        features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action   = np.array([0.5, -0.5], dtype=np.float32)
        frame    = GameFrame(features=features, action=action, timestamp_ms=16.0)
        assert frame.timestamp_ms == 16.0
        assert frame.features.shape == (3,)
        assert frame.action.shape == (2,)

    def test_fields_are_preserved(self) -> None:
        feat = np.array([0.1, 0.2], dtype=np.float32)
        act  = np.array([1.0], dtype=np.float32)
        frame = GameFrame(features=feat, action=act, timestamp_ms=100.0)
        np.testing.assert_array_equal(frame.features, feat)
        np.testing.assert_array_equal(frame.action, act)


class TestBaseReplayParser:
    def test_concrete_subclass_instantiates(self) -> None:
        parser = _ConcreteParser()
        assert parser.feature_dim == 4
        assert parser.action_dim == 2

    def test_parse_returns_frames(self) -> None:
        parser = _ConcreteParser()
        frames = parser.parse("replay.osr", "beatmap.osu")
        assert len(frames) == 1
        assert isinstance(frames[0], GameFrame)

    def test_abstract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseReplayParser()  # type: ignore[abstract]


class TestBaseGameEncoder:
    def test_concrete_subclass_works(self) -> None:
        enc = _ConcreteEncoder()
        state  = enc.encode_state({})
        action = enc.encode_action({})
        assert state.shape  == (4,)
        assert action.shape == (2,)

    def test_abstract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseGameEncoder()  # type: ignore[abstract]
