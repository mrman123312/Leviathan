#!/usr/bin/env python3
"""Train a candidate NRDF overlay only; never automatically promote it."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="JSONL containing text fields")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--max-loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=902)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--quantization", choices=["nf4", "none"], default="nf4")
    args = parser.parse_args()
    if min(args.steps, args.tokens, args.max_loops) <= 0:
        parser.error("Budgets must be positive")
    import torch
    from leviathan.consumer.runtime import load_model, save_graft
    from leviathan.consumer.profiles import get_profile
    from leviathan.consumer.recurrence import NRDFConfig, QwenNRDFWrapper, install_nrdf, graft_parameters
    from leviathan.consumer.training import EvaluationSplit, content_hash, sample_depth
    def read(path):
        texts = [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError(f"Empty/invalid dataset: {path}")
        return texts
    training, replay, heldout = read(args.train), read(args.replay), read(args.heldout)
    split = EvaluationSplit(frozenset(map(content_hash, training + replay)), frozenset(map(content_hash, heldout)))
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    profile = get_profile("rtx3060")
    model, tokenizer, metadata = load_model(profile, device=args.device, quantization=args.quantization)
    device = model.get_input_embeddings().weight.device
    def batch(text):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.tokens).input_ids.to(device)
        if ids.shape[-1] < 2:
            raise ValueError("At least two tokens required")
        return {"input_ids": ids, "labels": ids, "use_cache": False}
    def evaluate():
        model.eval()
        nll, count = 0., 0
        with torch.inference_mode():
            for text in heldout:
                b = batch(text)
                n = b["input_ids"].shape[-1] - 1
                nll += float(model(**b).loss) * n
                count += n
        return nll / count
    baseline_loss = evaluate()
    for p in model.parameters():
        p.requires_grad_(False)
    paths = install_nrdf(model, NRDFConfig(max_loops=args.max_loops))
    params = list(graft_parameters(model))
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    history = []
    model.train()
    for step in range(args.steps):
        source = training if step % 2 == 0 else replay
        text = source[int(torch.randint(len(source), (), generator=generator))]
        depth = sample_depth(1, args.max_loops, generator)
        for module in model.modules():
            if isinstance(module, QwenNRDFWrapper):
                module.loops = depth
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch(text)).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; candidate not saved")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        if step % 10 == 0:
            item = {"step": step, "depth": depth, "training_loss": float(loss.detach())}
            history.append(item)
            print(json.dumps(item), flush=True)
    for module in model.modules():
        if isinstance(module, QwenNRDFWrapper):
            module.loops = args.max_loops
    candidate_loss = evaluate()
    save_graft(model, args.output, profile)
    metadata.update({"graft_paths": paths, "baseline_heldout_loss": baseline_loss,
        "candidate_heldout_loss": candidate_loss, "relative_loss_change": candidate_loss / baseline_loss - 1,
        "retention_gate_pass": candidate_loss <= baseline_loss * 1.02,
        "promoted": False, "reason": "Broad capability/calibration/safety/efficiency gates not evaluated",
        "training_hashes": sorted(split.training_hashes), "heldout_hashes": sorted(split.heldout_hashes),
        "heldout_scope": "Excluded from this run's optimizer and replay; pretraining contamination unknown",
        "seed": args.seed, "history": history})
    (args.output / "training-report.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
