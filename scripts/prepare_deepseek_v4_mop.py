#!/usr/bin/env python3
"""Validate the full DeepSeek-V4-Pro-Base checkpoint and emit a Leviathan MoP manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from leviathan.deepseek_v4 import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "LEVIATHAN_DEEPSEEK_V4_DIR",
            "models/checkpoints/deepseek-v4-pro-base",
        ),
        help="Local DeepSeek-V4-Pro-Base checkpoint directory.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=128,
        help="Contiguous SwiGLU intermediate channels per MoP tile.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate architecture/config without requiring all 64 weight shards.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON manifest path. Defaults to stdout only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    manifest = build_manifest(
        model_dir,
        tile_width=args.tile_width,
        require_weights=not args.config_only,
    )
    payload = manifest.as_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
