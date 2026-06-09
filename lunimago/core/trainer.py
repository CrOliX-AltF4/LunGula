"""Generic training loop — game-agnostic."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .models.base_model import BaseImitatonModel


class Trainer:
    def __init__(
        self,
        model: BaseImitatonModel,
        device: torch.device,
        lr: float = 1e-3,
        val_split: float = 0.1,
    ) -> None:
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.val_split = val_split

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        epochs: int = 20,
        batch_size: int = 128,
        checkpoint_dir: str | None = None,
    ) -> list[dict]:
        val_len   = max(1, int(len(dataset) * self.val_split))
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        history = []
        for epoch in range(1, epochs + 1):
            train_loss = self._epoch(train_dl, train=True)
            val_loss   = self._epoch(val_dl,   train=False)
            history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
            print(f"[{epoch:03d}/{epochs}] train={train_loss:.5f}  val={val_loss:.5f}")

            if checkpoint_dir:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    f"{checkpoint_dir}/epoch_{epoch:03d}.pt",
                )

        return history

    def _epoch(self, loader: DataLoader, *, train: bool) -> float:
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
                total += loss.item() * len(x)
        return total / len(loader.dataset)
