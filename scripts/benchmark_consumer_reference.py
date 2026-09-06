#!/usr/bin/env python3
"""Reproducible CPU experiments, NOT pretrained Qwen/ARC/GSM8K benchmark scores."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from torch import nn
from torch.nn import functional as F
from leviathan.consumer.cells import SwiGLUCells
from leviathan.consumer.quantization import Int4Linear
from leviathan.consumer.recurrence import NRDFConfig, RecurrentFabric
from leviathan.consumer.efficiency import ExactDeltaCache, CacheScope
from leviathan.consumer.training import sample_depth

class ReferenceFFN(nn.Module):
    def __init__(self, hidden=128, intermediate=512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = F.silu
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def timed(fn, repeats=20):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return sorted(samples)[len(samples) // 2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/consumer/reference.json")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(902)
    random.seed(902)
    report = {"scope": "synthetic CPU reference, not pretrained language capability",
              "seed": 902, "torch": torch.__version__, "cpu": platform.processor(),
              "cuda_available": torch.cuda.is_available(), "language_benchmarks": {
                  name: None for name in ("ARC-Easy", "ARC-Challenge", "WikiText", "MMLU", "GSM8K")}}
    donor = ReferenceFFN().eval()
    x = torch.randn(32, 128)
    with torch.inference_mode():
        expected = donor(x)
        baseline = timed(lambda: donor(x))
        cell_results = []
        for width in (32, 64, 128, 256):
            bank = SwiGLUCells(donor, width)
            actual = bank.reconstruct(x)
            seconds = timed(lambda: bank.reconstruct(x))
            cell_results.append({"width": width, "max_abs_error": float((actual - expected).abs().max()),
                                 "median_seconds": seconds, "slowdown_vs_dense": seconds / baseline,
                                 "promotion": "reference only; no speed claim"})
        report["cell_geometry"] = cell_results
        report["dense_seconds"] = baseline
    quantized = ReferenceFFN().eval()
    quantized.load_state_dict(donor.state_dict())
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(quantized, name, Int4Linear.from_linear(getattr(quantized, name)))
    with torch.inference_mode():
        qout = quantized(x)
        tiled = SwiGLUCells(quantized, 128).reconstruct(x)
        report["int4"] = {"quantization_max_abs_error": float((qout - expected).abs().max()),
                          "cellization_max_abs_error": float((qout - tiled).abs().max()),
                          "format": "Leviathan symmetric group INT4 reference, not NF4/AWQ"}
    report["delta_cache"] = []
    for hidden, intermediate in ((32, 64), (512, 2048)):
        ff = ReferenceFFN(hidden, intermediate).eval()
        original = torch.randn(128, hidden)
        changed = original.clone()
        changed[:4] += .1
        scope = CacheScope("synthetic", 0, "one-test", "fp32", f"FFN-{hidden}")
        cache = ExactDeltaCache()
        with torch.inference_mode():
            cache.run(original, ff, scope)
            actual = cache.run(changed, ff, scope)
            error = float((actual - ff(changed)).abs().max())
            toggle = [False]
            def cached():
                toggle[0] = not toggle[0]
                return cache.run(original if toggle[0] else changed, ff, scope)
            full = timed(lambda: ff(changed))
            reuse = timed(cached)
            report["delta_cache"].append({"hidden": hidden, "intermediate": intermediate,
                "unchanged_rows": 124, "rows": 128, "max_abs_error": error,
                "dense_seconds": full, "cache_seconds": reuse, "speedup": full / reuse,
                "decision": "keep for this workload" if reuse < full else "reject for this workload"})
    cfg = NRDFConfig(latent_dim=32, heads=4, slots=4, max_loops=6, cell_width=128)
    fabric = RecurrentFabric(128, cfg)
    matrix = torch.randn(128, 128) / 128**.5
    train_gen = torch.Generator().manual_seed(731)
    test_gen = torch.Generator().manual_seed(991)
    test = torch.randn(96, 128, generator=test_gen)
    target = lambda z: .15 * torch.tanh(z @ matrix)
    def evaluate():
        fabric.eval()
        with torch.inference_mode():
            values = {str(depth): float(F.mse_loss(fabric(test, loops=depth)[0], target(test)))
                      for depth in (1, 2, 4, 6)}
        fabric.train()
        return values
    before = evaluate()
    optimizer = torch.optim.AdamW(fabric.parameters(), lr=2e-3)
    start = time.perf_counter()
    checkpoints = []
    for step in range(args.steps):
        z = torch.randn(16, 128, generator=train_gen)
        depth = sample_depth(1, 4, train_gen)
        predicted, trace = fabric(z, loops=depth)
        loss = F.mse_loss(predicted, target(z))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(fabric.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 40 == 0:
            checkpoints.append({"step": step + 1, "train_mse": float(loss.detach())})
            print(f"Synthetic recurrence step {step + 1}: MSE={float(loss.detach()):.6f}", flush=True)
    after = evaluate()
    report["synthetic_training"] = {"steps": args.steps, "train_depths": [1, 2, 3, 4],
        "untrained_test_depth": 6, "before_mse": before, "after_mse": after,
        "training_seconds": time.perf_counter() - start, "checkpoints": checkpoints,
        "interpretation": "Optimization smoke test only; no claim of language reasoning or AGI",
        "depth6_better_than4": after["6"] < after["4"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
