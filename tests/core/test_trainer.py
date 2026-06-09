"""Tests for core.trainer — Trainer fit loop."""
import os
import pathlib
import torch
from torch.utils.data import TensorDataset
import pytest
from lunimago.core.trainer import Trainer
from lunimago.core.models.lstm_agent import LSTMAgent

_FEATURE_DIM = 4
_ACTION_DIM  = 2
_WINDOW      = 8
_N_SAMPLES   = 60


@pytest.fixture()
def tiny_dataset() -> TensorDataset:
    x = torch.randn(_N_SAMPLES, _WINDOW, _FEATURE_DIM)
    y = torch.randn(_N_SAMPLES, _ACTION_DIM)
    return TensorDataset(x, y)


@pytest.fixture()
def trainer() -> Trainer:
    model  = LSTMAgent(feature_dim=_FEATURE_DIM, action_dim=_ACTION_DIM, hidden_size=32)
    device = torch.device("cpu")
    return Trainer(model, device, lr=1e-3)


class TestTrainer:
    def test_fit_returns_history_with_correct_length(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        history = trainer.fit(tiny_dataset, epochs=3, batch_size=16)
        assert len(history) == 3

    def test_history_has_required_keys(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        history = trainer.fit(tiny_dataset, epochs=1, batch_size=16)
        assert "epoch" in history[0]
        assert "train" in history[0]
        assert "val"   in history[0]

    def test_epoch_numbers_are_sequential(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        history = trainer.fit(tiny_dataset, epochs=4, batch_size=16)
        for i, entry in enumerate(history, start=1):
            assert entry["epoch"] == i

    def test_loss_values_are_finite(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        history = trainer.fit(tiny_dataset, epochs=2, batch_size=16)
        for entry in history:
            assert isinstance(entry["train"], float)
            assert isinstance(entry["val"],   float)
            assert entry["train"] >= 0
            assert entry["val"]   >= 0

    def test_loss_does_not_increase_wildly(
        self, trainer: Trainer, tiny_dataset: TensorDataset
    ) -> None:
        # Training loss after 5 epochs must stay below 100 (sanity guard only)
        history = trainer.fit(tiny_dataset, epochs=5, batch_size=16)
        assert history[-1]["train"] < 100.0

    def test_checkpoint_dir_creates_files(
        self, trainer: Trainer, tiny_dataset: TensorDataset, tmp_path: pathlib.Path
    ) -> None:
        ckpt_dir = str(tmp_path / "ckpts")
        trainer.fit(tiny_dataset, epochs=2, batch_size=16, checkpoint_dir=ckpt_dir)
        files = os.listdir(ckpt_dir)
        assert len(files) == 2
        assert all(f.endswith(".pt") for f in files)
