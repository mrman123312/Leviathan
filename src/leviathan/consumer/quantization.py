"""Portable packed INT4 correctness reference, NOT an AWQ/NF4 speed kernel.

Quantization error and cellization error are separate experiments. Only this format
and ordinary torch Linear expose slices. Foreign packed formats fail closed.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Int4Linear(nn.Module):
    def __init__(self, packed: Tensor, scales: Tensor, *, in_features: int,
                 out_features: int, group_size: int, bias: Tensor | None = None):
        super().__init__()
        if min(in_features, out_features, group_size) <= 0 or group_size % 2 or in_features % group_size:
            raise ValueError("Invalid INT4 geometry")
        if packed.dtype != torch.uint8 or packed.shape != (out_features, in_features // 2):
            raise ValueError("Malformed packed INT4 buffer")
        if scales.shape != (out_features, in_features // group_size) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Malformed quantization scales")
        if bias is not None and bias.shape != (out_features,):
            raise ValueError("Malformed bias")
        self.in_features, self.out_features = in_features, out_features
        self.group_size = group_size
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, layer: nn.Linear, group_size: int = 64) -> "Int4Linear":
        if type(layer) is not nn.Linear:
            raise TypeError("Convert ordinary Linear, not an already packed foreign format")
        if group_size <= 0 or group_size % 2 or layer.in_features % group_size:
            raise ValueError("Even quantization group_size must divide input width")
        with torch.no_grad():
            weight = layer.weight.detach().float()
            if not torch.isfinite(weight).all():
                raise ValueError("Cannot quantize non-finite weights")
            groups = weight.reshape(layer.out_features, -1, group_size)
            scales = (groups.abs().amax(-1) / 7).clamp_min(torch.finfo(torch.float32).tiny)
            q = (groups / scales[..., None]).round().clamp(-7, 7).to(torch.int16)
            q = (q + 8).reshape(layer.out_features, layer.in_features).to(torch.uint8)
            packed = q[:, 0::2] | (q[:, 1::2] << 4)
            bias = None if layer.bias is None else layer.bias.detach().clone()
        return cls(packed, scales, in_features=layer.in_features,
                   out_features=layer.out_features, group_size=group_size, bias=bias)

    def weight_slice(self, rows: slice = slice(None), cols: slice = slice(None)) -> Tensor:
        start, stop, step = cols.indices(self.in_features)
        if step != 1 or start >= stop:
            raise ValueError("Nonempty contiguous column slice required")
        positions = torch.arange(start, stop, device=self.packed.device)
        bytes_ = self.packed[rows][:, positions // 2]
        nibble = (bytes_ >> ((positions % 2) * 4).to(torch.uint8)) & 15
        scale = self.scales[rows][:, positions // self.group_size]
        return (nibble.to(torch.float32) - 8) * scale

    @property
    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers() if t is not None)

    def forward(self, x: Tensor) -> Tensor:
        chunks = []
        for start in range(0, self.out_features, 256):
            stop = min(start + 256, self.out_features)
            bias = None if self.bias is None else self.bias[start:stop].to(x.dtype)
            chunks.append(F.linear(x, self.weight_slice(slice(start, stop)).to(x.dtype), bias))
        return torch.cat(chunks, dim=-1)


def slice_weight(layer: nn.Module, rows: slice = slice(None),
                 cols: slice = slice(None)) -> Tensor:
    if isinstance(layer, Int4Linear):
        return layer.weight_slice(rows, cols)
    if type(layer) is nn.Linear and type(layer.weight) is nn.Parameter:
        return layer.weight[rows, cols]
    raise TypeError("Ancestral slicing supports plain Linear or Leviathan INT4 only; "
                    "NF4/AWQ/GPTQ need format-aware kernels. Use NRDF-only for opaque weights.")


def supports_slicing(layer: nn.Module) -> bool:
    return isinstance(layer, Int4Linear) or (
        type(layer) is nn.Linear and type(layer.weight) is nn.Parameter)
