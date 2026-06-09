"""Export a trained model to ONNX for cross-platform inference."""
from __future__ import annotations
import torch
from ..models.base_model import BaseImitatonModel


def export_onnx(
    model: BaseImitatonModel,
    output_path: str,
    window: int = 32,
    opset: int = 17,
) -> None:
    """Export model to ONNX.

    The exported model takes a single input:
      context: float32[1, window, feature_dim]
    and produces:
      action: float32[1, action_dim]
    """
    model.eval()
    dummy = torch.zeros(1, window, model.feature_dim)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["context"],
        output_names=["action"],
        dynamic_axes={"context": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported to {output_path}")
