"""Export a trained Leviathan policy checkpoint to the LVTP1 text format.

The exporter uses simple per-layer symmetric scaling, folds those scales into
bias/output terms approximately, and emits integer weights accepted by the C++
engine. Engine Elo, not offline loss, decides whether an exported net survives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import FEATURE_COUNT, HIDDEN_SIZE, LeviathanPolicy


def q127(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    peak = float(t.abs().max())
    scale = peak / 127.0 if peak > 0 else 1.0
    q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int32)
    return q, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = LeviathanPolicy()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    hw, hs = q127(model.hidden.weight.detach())
    # C++ hidden activations are clamped integer values. Hidden biases are put
    # into the same integer domain as hidden weights.
    hb = torch.round(model.hidden.bias.detach() / hs).clamp(-32768, 32767).to(torch.int32)

    ow, os = q127(model.output.weight.detach().squeeze(0))
    # The C++ scorer divides its final accumulator by 16. Map floating output
    # bias into that accumulator domain. The two learned scales are recorded in
    # comments for auditability; Elo tests tune practical PolicyWeight.
    ob = int(round(float(model.output.bias.detach()) / max(hs * os, 1e-12)))
    ob = max(-(2**31), min(2**31 - 1, ob))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(f"LVTP1 {FEATURE_COUNT} {HIDDEN_SIZE}\n")
        for row in hw.tolist():
            out.write(" ".join(str(v) for v in row) + "\n")
        out.write(" ".join(str(v) for v in hb.tolist()) + "\n")
        out.write(" ".join(str(v) for v in ow.tolist()) + "\n")
        out.write(str(ob) + "\n")

    print(f"wrote={args.output} hidden_scale={hs:.9g} output_scale={os:.9g}")


if __name__ == "__main__":
    main()
