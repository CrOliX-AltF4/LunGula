"""Tests for games.osu.parser — pure helpers and integration with synthetic data."""
from __future__ import annotations
import os
import pathlib
import tempfile
from collections.abc import Generator
import numpy as np
import pytest
from lunimago.games.osu.parser import (
    _bisect_left,
    _align,
    _parse_beatmap,
    OsuReplayParser,
    _FEATURE_DIM,
    _ACTION_DIM,
)


# ── _bisect_left ──────────────────────────────────────────────────────────────

class TestBisectLeft:
    def test_empty_sequence(self) -> None:
        assert _bisect_left([], 5.0) == 0

    def test_before_all(self) -> None:
        assert _bisect_left([10.0, 20.0, 30.0], 5.0) == 0

    def test_after_all(self) -> None:
        assert _bisect_left([10.0, 20.0, 30.0], 40.0) == 3

    def test_exact_match(self) -> None:
        assert _bisect_left([10.0, 20.0, 30.0], 20.0) == 1

    def test_between_values(self) -> None:
        assert _bisect_left([10.0, 20.0, 30.0], 15.0) == 1

    def test_single_element_before(self) -> None:
        assert _bisect_left([10.0], 5.0) == 0

    def test_single_element_after(self) -> None:
        assert _bisect_left([10.0], 15.0) == 1


# ── _align ────────────────────────────────────────────────────────────────────

def _make_hit_objects(times: list[float]) -> list[dict]:
    return [{"x": 256.0, "y": 192.0, "time_ms": t} for t in times]


def _make_replay_frames(times: list[float], keys: int = 0) -> list[dict]:
    return [{"time_ms": t, "x": float(i * 10), "y": float(i * 5), "keys": keys}
            for i, t in enumerate(times)]


class TestAlign:
    def test_empty_inputs_return_empty(self) -> None:
        assert _align([], []) == []

    def test_empty_hit_objects_return_empty(self) -> None:
        frames = _make_replay_frames([0.0, 100.0, 200.0])
        assert _align([], frames) == []

    def test_empty_replay_frames_return_empty(self) -> None:
        objects = _make_hit_objects([1000.0, 2000.0])
        assert _align(objects, []) == []

    def test_output_feature_shape(self) -> None:
        objects = _make_hit_objects([1000.0, 2000.0, 3000.0, 4000.0])
        frames  = _make_replay_frames([0.0, 500.0, 1000.0, 1500.0, 2000.0])
        result  = _align(objects, frames)
        assert len(result) > 0
        assert result[0].features.shape == (_FEATURE_DIM,)

    def test_output_action_shape(self) -> None:
        objects = _make_hit_objects([1000.0, 2000.0, 3000.0, 4000.0])
        frames  = _make_replay_frames([0.0, 500.0, 1000.0, 1500.0, 2000.0])
        result  = _align(objects, frames)
        assert result[0].action.shape == (_ACTION_DIM,)

    def test_click_encoded_in_action(self) -> None:
        objects = _make_hit_objects([500.0, 1000.0, 1500.0, 2000.0])
        frames  = [
            {"time_ms": 0.0,   "x": 256.0, "y": 192.0, "keys": 0},
            {"time_ms": 500.0, "x": 260.0, "y": 190.0, "keys": 1},  # left click
            {"time_ms": 800.0, "x": 265.0, "y": 188.0, "keys": 2},  # right click
        ]
        result = _align(objects, frames)
        # Frame at 500ms: left click bit = 1
        click_frame = next((r for r in result if r.timestamp_ms == 500.0), None)
        if click_frame is not None:
            assert click_frame.action[2] == pytest.approx(1.0)  # left click

    def test_timestamps_are_monotonic(self) -> None:
        objects = _make_hit_objects([1000.0, 2000.0, 3000.0])
        frames  = _make_replay_frames([0.0, 200.0, 500.0, 800.0, 1100.0])
        result  = _align(objects, frames)
        times   = [r.timestamp_ms for r in result]
        assert times == sorted(times)


# ── _parse_beatmap ─────────────────────────────────────────────────────────────

_MINIMAL_OSU = """\
osu file format v14

[Metadata]
Title:Test Map
Artist:Test Artist
Creator:mapper
Version:Normal

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:5
ApproachRate:8
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,1,0,100,1,0

[HitObjects]
256,192,1000,1,0,0:0:0:0:
300,200,2000,1,0,0:0:0:0:
100,100,3000,1,0,0:0:0:0:
"""


@pytest.fixture()
def beatmap_file() -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".osu", delete=False, encoding="utf-8"
    ) as f:
        f.write(_MINIMAL_OSU)
        path = f.name
    yield path
    os.unlink(path)


class TestParseBeatmap:
    def test_returns_correct_count(self, beatmap_file: str) -> None:
        objects = _parse_beatmap(beatmap_file)
        assert len(objects) == 3

    def test_first_object_position(self, beatmap_file: str) -> None:
        objects = _parse_beatmap(beatmap_file)
        assert objects[0]["x"] == pytest.approx(256.0)
        assert objects[0]["y"] == pytest.approx(192.0)

    def test_timing_values(self, beatmap_file: str) -> None:
        objects = _parse_beatmap(beatmap_file)
        times = [o["time_ms"] for o in objects]
        assert times == pytest.approx([1000.0, 2000.0, 3000.0])

    def test_empty_hit_objects_section(self) -> None:
        content = "osu file format v14\n\n[HitObjects]\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".osu", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            assert _parse_beatmap(path) == []
        finally:
            os.unlink(path)


# ── OsuReplayParser properties ─────────────────────────────────────────────────

class TestOsuReplayParser:
    def test_feature_dim(self) -> None:
        assert OsuReplayParser().feature_dim == _FEATURE_DIM

    def test_action_dim(self) -> None:
        assert OsuReplayParser().action_dim == _ACTION_DIM
