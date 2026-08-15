"""Project Leviathan tiny policy model.

The C++ engine consumes a quantized 12 -> 16 -> 1 MLP. Training may use
floating point; export.py performs deterministic symmetric quantization.
"""

from __future__ import annotations

import torch
from torch import nn

FEATURE_COUNT = 12
HIDDEN_SIZE = 16


class LeviathanPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(FEATURE_COUNT, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(x))).squeeze(-1)
