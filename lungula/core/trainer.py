"""Generic training loop — game-agnostic."""

from __future__ import annotations

import glob
import os
from collections.abc import Sized
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .models.base_model import BaseImitatonModel

_Batch = tuple[torch.Tensor, torch.Tensor]


class Trainer:
    def __init__(
        self,
        model: BaseImitatonModel,
        device: torch.device,
        lr: float = 1e-3,
        val_split: float = 0.1,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.opt = AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.val_split = val_split

    def fit(
        self,
        dataset: Dataset[_Batch],
        epochs: int = 20,
        batch_size: int = 128,
        checkpoint_dir: str | None = None,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        start_epoch = 1
        if resume and checkpoint_dir and os.path.isdir(checkpoint_dir):
            saved = sorted(glob.glob(f"{checkpoint_dir}/epoch_*.pt"))
            if saved:
                latest = saved[-1]
                self.model.load_state_dict(torch.load(latest, map_location=self.device))
                # epoch_007.pt → start from epoch 8
                start_epoch = int(os.path.basename(latest).split("_")[1].split(".")[0]) + 1
                print(f"Resumed from {latest} — starting at epoch {start_epoch}")

        n_total = len(cast(Sized, dataset))
        val_len = max(1, int(n_total * self.val_split))
        train_len = n_total - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        train_dl: DataLoader[_Batch] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl: DataLoader[_Batch] = DataLoader(val_ds, batch_size=batch_size)

        history: list[dict[str, Any]] = []
        for epoch in range(start_epoch, epochs + 1):
            train_loss = self._epoch(train_dl, train=True)
            val_loss = self._epoch(val_dl, train=False)
            history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
            print(f"[{epoch:03d}/{epochs}] train={train_loss:.5f}  val={val_loss:.5f}")

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    f"{checkpoint_dir}/epoch_{epoch:03d}.pt",
                )

        return history

    def _epoch(self, loader: DataLoader[_Batch], *, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, leave=False):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                total += float(loss.item()) * len(x)
        assert loader.dataset is not None
        return total / len(cast(Sized, loader.dataset))
