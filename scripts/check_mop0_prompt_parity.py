#!/usr/bin/env python3
"""Compare one real prompt through original DeepSeek V4 and Leviathan MoP-0.

This loads the local full checkpoint once, runs a normal forward pass, temporarily
wraps routed experts with the 128-channel reference decomposition, runs the same
prompt again, restores the donor experts, and reports logit drift.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leviathan.mop0_reference import compare_prompt_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "LEVIATHAN_DEEPSEEK_V4_DIR",
            "models/checkpoints/deepseek-v4-pro-base",
        ),
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Raw base-model prompt used for parity.",
    )
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--max-abs-tolerance",
        type=float,
        default=None,
        help="Optional hard failure threshold for max absolute logit drift.",
    )
    parser.add_argument(
        "--require-argmax-match",
        action="store_true",
        help="Exit non-zero if the next-token argmax differs.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON result path.")
    return parser.parse_args()


def input_device(model: Any) -> Any:
    try:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        if str(device) != "meta":
            return device
    except Exception:
        pass
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError("could not determine model input device")


def main() -> int:
    args = parse_args()
    if args.tile_width <= 0:
        raise SystemExit("--tile-width must be positive")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install local inference dependencies with: "
            "python -m pip install -e '.[inference]'"
        ) from exc

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    print(f"Loading checkpoint: {model_dir}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=args.device_map,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    device = input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    print("Running original V4 forward, then MoP-0 tiled forward...", file=sys.stderr)
    result = compare_prompt_logits(model, encoded, tile_width=args.tile_width)
    payload = {
        "prompt": args.prompt,
        "tile_width": args.tile_width,
        **result.as_dict(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    failed = False
    if args.max_abs_tolerance is not None:
        failed |= result.max_abs_logit_diff > args.max_abs_tolerance
    if args.require_argmax_match:
        failed |= not result.last_token_argmax_match
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
