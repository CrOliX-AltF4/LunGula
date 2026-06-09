"""Tests for core.models — LSTMAgent forward pass and properties."""
import torch
import pytest
from lunimago.core.models.lstm_agent import LSTMAgent


@pytest.fixture()
def model() -> LSTMAgent:
    return LSTMAgent(feature_dim=10, action_dim=4)


class TestLSTMAgent:
    def test_feature_dim_property(self, model: LSTMAgent) -> None:
        assert model.feature_dim == 10

    def test_action_dim_property(self, model: LSTMAgent) -> None:
        assert model.action_dim == 4

    def test_forward_output_shape(self, model: LSTMAgent) -> None:
        x   = torch.zeros(8, 32, 10)   # batch=8, window=32, features=10
        out = model(x)
        assert out.shape == (8, 4)

    def test_forward_batch_size_1(self, model: LSTMAgent) -> None:
        x   = torch.zeros(1, 16, 10)
        out = model(x)
        assert out.shape == (1, 4)

    def test_forward_no_nan(self, model: LSTMAgent) -> None:
        x   = torch.randn(4, 32, 10)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_custom_hidden_size(self) -> None:
        m = LSTMAgent(feature_dim=6, action_dim=3, hidden_size=64)
        x = torch.zeros(2, 8, 6)
        assert m(x).shape == (2, 3)

    def test_single_layer_no_dropout(self) -> None:
        # num_layers=1 must not crash (dropout disabled on single layer)
        m   = LSTMAgent(feature_dim=4, action_dim=2, num_layers=1)
        out = m(torch.zeros(1, 4, 4))
        assert out.shape == (1, 2)

    def test_gradients_flow(self, model: LSTMAgent) -> None:
        x      = torch.randn(2, 8, 10, requires_grad=False)
        target = torch.randn(2, 4)
        out    = model(x)
        loss   = (out - target).pow(2).mean()
        loss.backward()
        # Verify at least one parameter has a gradient
        assert any(p.grad is not None for p in model.parameters())
