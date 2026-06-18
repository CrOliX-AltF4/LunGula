# Contributing to Lun'Gula

Thank you for contributing! This guide will help you get started and ensure your contribution fits smoothly into the project.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Contribution workflow](#contribution-workflow)
- [Commit conventions](#commit-conventions)
- [Code standards](#code-standards)
- [Tests](#tests)
- [Adding a game plugin](#adding-a-game-plugin)
- [Submitting a PR](#submitting-a-pr)

---

## Code of conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/). By contributing, you agree to abide by its terms. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Quick start

### Prerequisites

- **Python** >= 3.10
- **pip** >= 23
- Git

### Setup

```bash
git clone https://github.com/CrOliX-AltF4/LunGula.git
cd LunGula
pip install -e ".[dev]"
```

### Available commands

| Command                        | Description                              |
|--------------------------------|------------------------------------------|
| `lungula train --game osu …` | Train a model on replay data             |
| `pytest`                       | Run all tests                            |
| `pytest --cov=lungula`       | Run tests with coverage report           |
| `ruff check .`                 | Lint the codebase                        |
| `ruff format .`                | Format the codebase                      |
| `mypy lungula`               | Type-check                               |

---

## Project structure

```
lungula/
├── core/
│   ├── base_game.py        # BaseReplayParser, BaseGameEncoder, GameFrame
│   ├── dataset.py          # ReplayDataset
│   ├── device.py           # auto device detection
│   ├── trainer.py          # generic training loop
│   ├── models/             # model architectures
│   └── export/             # ONNX export
└── games/
    └── osu/                # osu! plugin (parser + plugin entry point)
tests/
├── core/                   # tests mirroring lungula/core/
└── games/osu/              # tests for the osu! plugin
```

---

## Contribution workflow

1. **Fork** the repo and create a branch from `master`:

   ```bash
   git checkout -b feat/my-feature master
   ```

2. **Develop** your feature with atomic commits.

3. **Make sure** all checks pass locally:

   ```bash
   ruff check . && mypy lungula && pytest
   ```

4. **Open a Pull Request** targeting `master`.

---

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <short description in lowercase>

[optional body]

[optional footer: Closes #123]
```

### Accepted types

| Type       | Usage                           |
|------------|---------------------------------|
| `feat`     | New feature                     |
| `fix`      | Bug fix                         |
| `docs`     | Documentation only              |
| `style`    | Formatting, no logic change     |
| `refactor` | Refactoring without fix or feat |
| `perf`     | Performance improvement         |
| `test`     | Adding or updating tests        |
| `build`    | Build tools, dependencies       |
| `ci`       | CI/CD                           |
| `chore`    | Maintenance, background tasks   |

### Examples

```bash
feat(games/osu): add slider parsing to osu parser
fix(trainer): handle empty dataset gracefully
test(core): add ReplayDataset edge case for short replays
docs(contributing): add game plugin guide
```

---

## Code standards

- **Python** >= 3.10, type hints everywhere
- **Ruff** for linting and formatting
- **Mypy** for type checking (`strict` mode recommended)
- **Imports**: standard library → third-party → local, separated by blank lines
- Abstract base classes in `core/` must remain game-agnostic

---

## Tests

- Framework: **pytest**
- Tests live in `tests/` (mirroring the `lungula/` structure)
- Unit tests must not touch real replay files — use fixtures with synthetic data
- Name test files `test_*.py`

```bash
pytest                      # run all tests
pytest --cov=lungula      # with coverage
pytest tests/core/          # specific directory
```

---

## Adding a game plugin

1. Create `lungula/games/<your_game>/` with `parser.py` and `plugin.py`
2. Implement `BaseReplayParser` in `parser.py`
3. Implement `make_parser()`, `make_model()`, `collect_pairs()` in `plugin.py`
4. Register the plugin in `lungula/cli.py` under `GAMES`
5. Add tests in `tests/games/<your_game>/`

See the [osu! plugin](lungula/games/osu/) as a reference implementation.

---

## Submitting a PR

1. Make sure the target branch is `master`
2. Fill in the PR template
3. Verify all CI checks pass
4. Request a review from at least one maintainer

---

Questions? Open a [GitHub Discussion](../../discussions) or an issue with the `question` label.
