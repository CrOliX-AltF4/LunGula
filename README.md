<div align="center">

# ◆ Lun'Imago

[![License](https://img.shields.io/badge/license-MIT-333333?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/CrOliX-AltF4/LunImago/ci.yml?style=flat-square&label=CI)](https://github.com/CrOliX-AltF4/LunImago/actions)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-555555?style=flat-square)](.)

**replay → model**

_A game imitation learning framework. Feed it human replays — a trained ONNX model comes out._

</div>

---

## What it does

Lun'Imago trains neural networks to imitate human game behavior from recorded replays, then exports them as ONNX models for use in any runtime (Python, Node.js, C++, etc.).

```
  human replays (.osr, ...)
           │
           ▼
  ┌──────────────────┐
  │   Game Parser    │  decodes raw replay + beatmap → GameFrame[]
  └────────┬─────────┘
           │  normalized feature / action sequences
           ▼
  ┌──────────────────┐
  │  ReplayDataset   │  sliding-window samples (context → next action)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   LSTM Agent     │  sequence model: context window → predicted action
  └────────┬─────────┘
           │  training loop (AdamW, MSE, val split)
           ▼
  ┌──────────────────┐
  │  ONNX Exporter   │  standard cross-platform model format
  └────────┬─────────┘
           │
           ▼
      ✦ model sealed
```

Why a framework instead of a one-off script? Every game needs only two things: a parser that decodes its replay format, and an encoder that normalizes game state into feature vectors. Everything else — the dataset, training loop, model architecture, device detection, and ONNX export — is shared.

> [!NOTE]
> "Imago" is the Latin word for _image_ or _likeness_. In biology, it is the final adult form reached after metamorphosis. A lun'imago model is the final learned likeness of a human player. Part of the [Lun' ecosystem](https://github.com/CrOliX-AltF4).

---

## Quick start

```bash
git clone https://github.com/CrOliX-AltF4/LunImago.git
cd LunImago
pip install -e ".[dev]"

# Train on osu! replays (one map per subdirectory: replay.osr + beatmap.osu)
lunaimago train --game osu --data ./data/replays --out ./checkpoints --export model.onnx
```

---

## Hardware support

Lun'Imago auto-detects the best available backend:

| Backend   | Hardware              | OS              | Install                     |
|-----------|-----------------------|-----------------|-----------------------------|
| CUDA      | NVIDIA GPU            | Win / Linux     | `pip install torch` (CUDA)  |
| DirectML  | AMD / Intel GPU       | Windows         | `pip install torch-directml`|
| ROCm      | AMD GPU               | Linux           | `pip install torch` (ROCm)  |
| MPS       | Apple Silicon         | macOS           | built-in                    |
| CPU       | any                   | any             | always available             |

Override with `--device cuda`, `--device directml`, `--device cpu`, etc.

---

## CLI

```bash
lunaimago train --game osu --data ./replays          # train with defaults
lunaimago train --game osu --data ./replays \
    --epochs 30 --batch 256 --window 48 \
    --device directml \
    --export model.onnx                              # full options
```

| Option      | Default          | Description                              |
|-------------|------------------|------------------------------------------|
| `--game`    | required         | Game plugin: `osu` (more coming)         |
| `--data`    | required         | Directory of replay pairs                |
| `--out`     | `checkpoints/`   | Checkpoint output directory              |
| `--epochs`  | `20`             | Training epochs                          |
| `--batch`   | `128`            | Batch size                               |
| `--window`  | `32`             | Context window size (frames)             |
| `--device`  | `auto`           | Compute device                           |
| `--export`  | none             | Export final model to `.onnx`            |

---

## Data layout

Each map goes in its own subdirectory with one replay and one beatmap file:

```
data/replays/
├── map_001/
│   ├── replay.osr
│   └── beatmap.osu
├── map_002/
│   ├── replay.osr
│   └── beatmap.osu
└── ...
```

osu! replays are publicly available via the [osu! API](https://osu.ppy.sh/docs/index#get-apiv2beatmapsbeatidreplays).

---

## Adding a game

Implement two abstract classes and register a plugin module:

```python
# lunaimago/games/my_game/parser.py
from lunaimago.core.base_game import BaseReplayParser, GameFrame

class MyGameParser(BaseReplayParser):
    @property
    def feature_dim(self) -> int: return 12

    @property
    def action_dim(self) -> int: return 3

    def parse(self, replay_path: str, beatmap_path: str) -> list[GameFrame]:
        ...

# lunaimago/games/my_game/plugin.py
def make_parser(): return MyGameParser()
def make_model(): return LSTMAgent(feature_dim=12, action_dim=3)
def collect_pairs(data_dir: str): ...
```

Then register it in `lunaimago/cli.py`:

```python
GAMES: dict[str, str] = {
    "osu":     "lunaimago.games.osu.plugin",
    "my_game": "lunaimago.games.my_game.plugin",
}
```

---

## Project structure

```
lunaimago/
├── core/
│   ├── base_game.py        # BaseReplayParser, BaseGameEncoder, GameFrame
│   ├── dataset.py          # ReplayDataset — sliding-window sequences
│   ├── device.py           # auto device detection
│   ├── trainer.py          # generic training loop
│   ├── models/
│   │   ├── base_model.py   # BaseImitationModel
│   │   └── lstm_agent.py   # 2-layer LSTM agent
│   └── export/
│       └── onnx_exporter.py
└── games/
    └── osu/
        ├── parser.py       # .osr + .osu decoder → GameFrame[]
        └── plugin.py       # CLI entry point
```

---

> [!WARNING]
> Lun'Imago trains models from human replays. Model quality depends entirely on the quality and quantity of training data. A model trained on 50 replays will not generalize well — aim for 1 000+ for usable results.

---

<div align="center">

Built by **[CrOliX-AltF4](https://github.com/CrOliX-AltF4)** · MIT License · © 2026

_Where observed play finds its learned form._

</div>
