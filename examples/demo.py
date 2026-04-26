"""eml-cost-torch demo — per-layer Pfaffian profile of three model shapes.

Run:
    pip install eml-cost-torch[torch]
    python examples/demo.py

Walks three increasingly-realistic models and prints per-layer
Pfaffian profile + cost-class summary for each.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from eml_cost_torch import profile, profile_dict, summary


def example_1_tiny_mlp() -> None:
    """A 4-layer MLP — the smallest interesting model."""
    print("\n" + "=" * 90)
    print("EXAMPLE 1 — Tiny MLP (Linear / GELU / Linear / Sigmoid)")
    print("=" * 90)
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, 16),
        nn.Sigmoid(),
    )
    print(summary(model))


def example_2_resnet_block_shape() -> None:
    """A ResNet-style block: Conv -> BN -> ReLU -> Conv -> BN.

    BatchNorm + ReLU are the interesting per-layer shapes here;
    Conv layers reduce to polynomial-form (r=0).
    """
    print("\n" + "=" * 90)
    print("EXAMPLE 2 — ResNet-style block shape")
    print("=" * 90)
    model = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
    )
    print(summary(model))


def example_3_transformer_block_shape() -> None:
    """A Transformer-encoder-style block.

    Notice how the high-r operations (LayerNorm + GELU) dominate
    the per-layer profile, and the linear projections sit at r=0.
    """
    print("\n" + "=" * 90)
    print("EXAMPLE 3 — Transformer-encoder-style block shape")
    print("=" * 90)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int = 64, n_heads: int = 4,
                     d_ff: int = 256):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads,
                                              batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff_in = nn.Linear(d_model, d_ff)
            self.ff_act = nn.GELU()
            self.ff_out = nn.Linear(d_ff, d_model)

    print(summary(TransformerBlock()))


def example_4_programmatic_dict() -> None:
    """Programmatic JSON-friendly access for tooling pipelines."""
    print("\n" + "=" * 90)
    print("EXAMPLE 4 — programmatic dict output (for pipelines)")
    print("=" * 90)
    model = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 2))
    rows = profile_dict(model)
    for r in rows:
        print(f"  {r['name']:<6}  {r['class_name']:<10}  "
              f"axes={r['axes']:<18}  r={r['pfaffian_r']}")
    print()
    print("  Each row is a dict: {name, class_name, pfaffian_r, "
          "max_path_r, eml_depth, predicted_depth, "
          "is_pfaffian_not_eml, axes, is_unknown}")


if __name__ == "__main__":
    print(f"torch={torch.__version__}")
    example_1_tiny_mlp()
    example_2_resnet_block_shape()
    example_3_transformer_block_shape()
    example_4_programmatic_dict()
    print("\n" + "=" * 90)
    print("DONE — see https://pypi.org/project/eml-cost-torch/")
    print("=" * 90)
