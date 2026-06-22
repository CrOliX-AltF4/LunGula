<div align="center">

# ◆ Lun'Gula

[![License](https://img.shields.io/badge/license-MIT-333333?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/CrOliX-AltF4/LunGula/ci.yml?style=flat-square&label=CI)](https://github.com/CrOliX-AltF4/LunGula/actions)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-555555?style=flat-square)](.)

**replay → model**

_A game imitation learning framework. Feed it human replays — a trained ONNX model comes out._

</div>

> [!NOTE]
> Fully standalone — any runtime that can load ONNX models can consume the output (Python, Node.js, C++). Part of the [Lun' ecosystem](https://github.com/CrOliX-AltF4).

---

## What it does

Lun'Gula trains neural networks to imitate human game behavior from recorded replays, then exports them as ONNX models for use in any runtime (Python, Node.js, C++, etc.).

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
> "Gula" is the sin of gluttony — consuming endlessly to grow stronger. A lun'gula model is trained by feeding on human replays until it has absorbed the patterns of play. Part of the [Lun' ecosystem](https://github.com/CrOliX-AltF4).

---

## Quick start

```bash
git clone https://github.com/CrOliX-AltF4/LunGula.git
cd LunGula
pip install -e ".[dev]"

# Train on osu! replays (one map per subdirectory: replay.osr + beatmap.osu)
lungula train --game osu --data ./data/replays --out ./checkpoints --export model.onnx
```

---

## Hardware support

Lun'Gula auto-detects the best available backend:

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
lungula train --game osu --data ./replays          # train with defaults
lungula train --game osu --data ./replays \
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
| `--resume`  | off              | Resume from latest checkpoint in `--out` |

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

## Training guide

### Dataset requirements

Each map directory must contain exactly one replay + one beatmap:

```
data/replays/
├── map_001/
│   ├── replay.osr
│   └── beatmap.osu
├── map_002/
│   └── ...
```

| Target difficulty | Min maps | Recommended |
|---|---|---|
| Normal (1.5–2.5★) | 50 | 200+ |
| Hard (2.5–3.5★) | 100 | 500+ |

**Diversity matters more than volume** — 200 maps across different BPMs, patterns, and mappers generalise far better than 500 replays of the same map.

osu! replays are publicly available via the [osu! API](https://osu.ppy.sh/docs/index#get-apiv2beatmapsbeatidreplays).

### Recommended command — osu! Normal maps

```bash
lungula train \
  --game osu \
  --data ./data/replays \
  --out ./checkpoints \
  --epochs 200 \
  --batch 256 \
  --window 32 \
  --lr 1e-3 \
  --device directml \
  --export model.onnx
```

### Reading the output

```
[baseline] null model (predict-zero) MSE = 0.00318
[001/200] train=0.03120  val=0.02891  lr=1.00e-03
[006/200] train=0.00820  val=0.00711  lr=1.00e-03
[012/200] train=0.00410  val=0.00390  lr=5.00e-04  ↓ lr=5.00e-04
...
[080/200] train=0.00028  val=0.00031  lr=1.25e-04
```

**Baseline MSE** — what a model predicting all zeros would score. Use it to calibrate your training loss.

| val loss vs baseline | Verdict |
|---|---|
| `val > baseline` | Underfitting or data issue — check data quality first |
| `val ≈ 0.5 × baseline` | Partial convergence — more epochs or data |
| `val < 0.1 × baseline` | Good convergence — deploy and test |

**LR drops** (marked `↓ lr=…`) happen after 5 epochs without val improvement. Two consecutive drops with no improvement → you can stop early.

### Diagnosing cursor drift in the bot

If the deployed model drifts toward a screen edge:

| Symptom | Cause | Fix |
|---|---|---|
| `val > baseline` | Active underfitting | More data, check pipeline alignment |
| `val ≈ 0.3–0.9 × baseline`, drift constant | Bias in training set | Add maps with varied note positions |
| `val < 0.1 × baseline`, drift still present | Distribution shift | More map diversity; train on harder maps too |
| Drift only at map start | Window not yet full (32 frames) | Normal — ignore first ~0.5s |

---

## Adding a game

Implement two abstract classes and register a plugin module:

```python
# lungula/games/my_game/parser.py
from lungula.core.base_game import BaseReplayParser, GameFrame

class MyGameParser(BaseReplayParser):
    @property
    def feature_dim(self) -> int: return 12

    @property
    def action_dim(self) -> int: return 3

    def parse(self, replay_path: str, beatmap_path: str) -> list[GameFrame]:
        ...

# lungula/games/my_game/plugin.py
def make_parser(): return MyGameParser()
def make_model(): return LSTMAgent(feature_dim=12, action_dim=3)
def collect_pairs(data_dir: str): ...
```

Then register it in `lungula/cli.py`:

```python
GAMES: dict[str, str] = {
    "osu":     "lungula.games.osu.plugin",
    "my_game": "lungula.games.my_game.plugin",
}
```

---

## Project structure

```
lungula/
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

## Lun ecosystem

| Project | Role |
|---|---|
| [LunIra](https://github.com/CrOliX-AltF4/LunIra) | AI dev pipeline — intent → code |
| [LunAcedia](https://github.com/CrOliX-AltF4/LunAcedia) | Information infrastructure — events · actions · AI butler |
| [LunAvaritia](https://github.com/CrOliX-AltF4/LunAvaritia) | Mobile companion — Android |
| **LunGula** | Imitation learning — gameplay → ONNX policy |
| LunAnima | AI companion core — private |

---

> [!WARNING]
> Lun'Gula trains models from human replays. Model quality depends entirely on the quality and quantity of training data. A model trained on 50 replays will not generalize well — aim for 1 000+ for usable results.

---

<div align="center">

Built by **[CrOliX-AltF4](https://github.com/CrOliX-AltF4)** · MIT License · © 2026

_Where observed play finds its learned form._

</div>
