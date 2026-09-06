#!/usr/bin/env python3
"""Prompt one pinned consumer model. Default: genuine Qwen3-1.7B-Base, NF4."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["rtx3060", "qwen27b"], default="rtx3060")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--quantization", choices=["nf4", "none"], default="nf4")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-large-model", action="store_true")
    parser.add_argument("--nrdf", action="store_true", help="Install zero-gated recurrent adapter")
    parser.add_argument("--cells", action="store_true", help="Requires plain float donor; rejects opaque NF4 slices")
    parser.add_argument("--pulse-interval", type=int, default=0)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--observe-at-zero", action="store_true")
    parser.add_argument("--graft")
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    from leviathan.consumer.profiles import get_profile
    from leviathan.consumer.runtime import load_model, load_graft, prompt
    from leviathan.consumer.recurrence import NRDFConfig, QwenNRDFWrapper, install_nrdf
    profile = get_profile(args.profile)
    if args.context + args.max_new_tokens > profile.context_limit:
        parser.error("Requested context plus generation exceeds native model limit")
    if args.cells and args.quantization == "nf4":
        parser.error("NF4 has no ancestral tile kernel yet. Use --nrdf alone or --quantization none --cells")
    if args.cells and not args.nrdf:
        parser.error("--cells requires --nrdf")
    if args.graft and args.nrdf:
        parser.error("Use a saved --graft or a new --nrdf, not both")
    try:
        model, tokenizer, report = load_model(profile, quantization=args.quantization,
            device=args.device, local_files_only=args.local_files_only,
            allow_large_model=args.allow_large_model)
        if args.graft:
            load_graft(model, args.graft, profile, allow_experimental=args.allow_experimental)
        if args.nrdf:
            report["graft_layers"] = install_nrdf(model, NRDFConfig(
                max_loops=args.loops, ancestral_cells=args.cells, pulse_interval=args.pulse_interval))
        for module in model.modules():
            if isinstance(module, QwenNRDFWrapper):
                module.observe_at_zero = args.observe_at_zero
        model.eval()
        print(f"{profile.repo_id} [{profile.stage}] at {profile.revision}", file=sys.stderr)
        print("Base profiles produce raw completions, not an instruction-tuned assistant.", file=sys.stderr)
        report["prompts"] = []
        while True:
            text = args.prompt if args.prompt is not None else input("prompt> ")
            if not text.strip() or text.strip() in {"/quit", "/exit"}:
                break
            answer, metrics = prompt(model, tokenizer, text, max_new_tokens=args.max_new_tokens,
                                     max_input_tokens=args.context)
            print(answer, flush=True)
            print(json.dumps(metrics), file=sys.stderr)
            report["prompts"].append(metrics)
            if args.prompt is not None:
                break
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2))
    except (ValueError, RuntimeError, MemoryError, OSError) as exc:
        print(f"Consumer runtime: {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
