"""Auto-detect the best available compute device."""

from __future__ import annotations

from typing import cast

import torch


def resolve_device(preference: str | None = None) -> torch.device:
    """Return the best available device, or respect an explicit preference.

    Priority (auto): CUDA → DirectML → MPS → CPU
    """
    if preference and preference != "auto":
        if preference == "directml":
            import torch_directml

            return cast(torch.device, torch_directml.device())
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml

        if torch_directml.is_available():
            return cast(torch.device, torch_directml.device())
    except ImportError:
        pass

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
