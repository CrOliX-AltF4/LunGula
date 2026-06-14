"""osu! replay + beatmap parser.

Replay format (.osr): https://osu.ppy.sh/wiki/en/Client/File_formats/osr_(file_format)
Beatmap format (.osu): https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format)

Each GameFrame encodes:
  features: [dx, dy, time_to_next_note, note_x_norm, note_y_norm,
             next2_x, next2_y, next3_x, next3_y, combo_ratio]  (10 dims)
  action:   [cursor_dx, cursor_dy, left_click, right_click]     (4 dims)
"""

from __future__ import annotations

import lzma
import struct
from typing import Any

import numpy as np

from ...core.base_game import BaseReplayParser, GameFrame

_FEATURE_DIM = 10
_ACTION_DIM = 4

# osu! standard playfield: 512 × 384 px
_PF_W, _PF_H = 512.0, 384.0


class OsuReplayParser(BaseReplayParser):
    @property
    def feature_dim(self) -> int:
        return _FEATURE_DIM

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    def parse(self, replay_path: str, beatmap_path: str) -> list[GameFrame]:
        hit_objects = _parse_beatmap(beatmap_path)
        replay_frames = _parse_replay(replay_path)
        return _align(hit_objects, replay_frames)


# ── .osu beatmap parser (hit objects only) ────────────────────────────────────


def _parse_beatmap(path: str) -> list[dict[str, Any]]:
    """Return list of {time_ms, x, y} for HitCircles and Slider heads."""
    objects: list[dict[str, Any]] = []
    in_section = False
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "[HitObjects]":
                in_section = True
                continue
            if in_section:
                if line.startswith("["):
                    break
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                objects.append(
                    {
                        "x": float(parts[0]),
                        "y": float(parts[1]),
                        "time_ms": float(parts[2]),
                    }
                )
    return objects


# ── .osr replay parser ────────────────────────────────────────────────────────


def _read_string(data: bytes, pos: int) -> tuple[str, int]:
    if data[pos] == 0x00:
        return "", pos + 1
    pos += 1  # skip 0x0b
    length, shift = 0, 0
    while True:
        byte = data[pos]
        pos += 1
        length |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return data[pos : pos + length].decode("utf-8", errors="ignore"), pos + length


def _parse_replay(path: str) -> list[dict[str, Any]]:
    """Return list of {time_ms, x, y, keys} from the replay data frames."""
    with open(path, "rb") as f:
        data = f.read()

    pos = 0
    pos += 1  # game mode
    pos += 4  # version
    _, pos = _read_string(data, pos)  # beatmap hash
    _, pos = _read_string(data, pos)  # player name
    _, pos = _read_string(data, pos)  # replay hash
    pos += 2 + 2 + 2 + 2 + 2 + 2  # counts: n300, n100, n50, ngeki, nkatu, nmiss
    pos += 4  # total score
    pos += 2  # max combo
    pos += 1  # perfect
    pos += 4  # mods

    # life bar graph (string)
    _, pos = _read_string(data, pos)

    pos += 8  # timestamp

    compressed_len = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    compressed = data[pos : pos + compressed_len]
    raw = lzma.decompress(compressed).decode("utf-8", errors="ignore")

    frames: list[dict[str, Any]] = []
    t = 0.0
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("|")
        if len(parts) < 4:
            continue
        dt = float(parts[0])
        if dt == -12345:
            continue
        t += dt
        frames.append(
            {
                "time_ms": t,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "keys": int(float(parts[3])),
            }
        )
    return frames


# ── Alignment: map replay cursor positions to note context ───────────────────


def _align(
    hit_objects: list[dict[str, Any]],
    replay_frames: list[dict[str, Any]],
) -> list[GameFrame]:
    if not hit_objects or not replay_frames:
        return []

    samples: list[GameFrame] = []
    ho_times = [h["time_ms"] for h in hit_objects]

    prev_x, prev_y = replay_frames[0]["x"], replay_frames[0]["y"]

    for i in range(1, len(replay_frames)):
        rf = replay_frames[i]
        t = rf["time_ms"]

        # find the next upcoming note
        next_idx = _bisect_left(ho_times, t)
        if next_idx >= len(hit_objects):
            break

        def _note_feat(idx: int) -> tuple[float, float]:
            if idx < len(hit_objects):
                return (
                    hit_objects[idx]["x"] / _PF_W * 2 - 1,
                    hit_objects[idx]["y"] / _PF_H * 2 - 1,
                )
            return 0.0, 0.0

        n0x, n0y = _note_feat(next_idx)
        n1x, n1y = _note_feat(next_idx + 1)
        n2x, n2y = _note_feat(next_idx + 2)
        time_to = max(0.0, hit_objects[next_idx]["time_ms"] - t) / 1000.0
        combo_r = min(1.0, next_idx / max(1, len(hit_objects)))

        features = np.array(
            [
                (rf["x"] - prev_x) / _PF_W,
                (rf["y"] - prev_y) / _PF_H,
                time_to,
                n0x,
                n0y,
                n1x,
                n1y,
                n2x,
                n2y,
                combo_r,
            ],
            dtype=np.float32,
        )

        keys = rf["keys"]
        action = np.array(
            [
                (rf["x"] - prev_x) / _PF_W,
                (rf["y"] - prev_y) / _PF_H,
                float(bool(keys & 1)),
                float(bool(keys & 2)),
            ],
            dtype=np.float32,
        )

        samples.append(GameFrame(features=features, action=action, timestamp_ms=t))
        prev_x, prev_y = rf["x"], rf["y"]

    return samples


def _bisect_left(seq: list[float], val: float) -> int:
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo
