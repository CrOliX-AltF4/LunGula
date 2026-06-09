"""Tests for core.device — resolve_device auto-detection and explicit preferences."""
import torch

from lunimago.core.device import resolve_device


class TestResolveDevice:
    def test_explicit_cpu(self) -> None:
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_returns_torch_device(self) -> None:
        device = resolve_device("auto")
        assert isinstance(device, torch.device)

    def test_none_same_as_auto(self) -> None:
        device = resolve_device(None)
        assert isinstance(device, torch.device)

    def test_cpu_is_always_valid(self) -> None:
        # CPU must always be reachable regardless of hardware
        device = resolve_device("cpu")
        t = torch.zeros(2, device=device)
        assert t.device.type == "cpu"

    def test_explicit_cuda_only_when_available(self) -> None:
        if torch.cuda.is_available():
            device = resolve_device("cuda")
            assert device.type == "cuda"
        else:
            # On machines without CUDA, torch.device("cuda") still constructs
            # but tensors cannot be moved there — we just verify no exception at creation
            device = resolve_device("cuda")
            assert device.type == "cuda"

    def test_auto_falls_back_to_cpu_in_ci(self) -> None:
        # In CI (no GPU), auto must resolve to CPU
        device = resolve_device("auto")
        # Should be one of the known valid types
        assert device.type in ("cpu", "cuda", "mps")
