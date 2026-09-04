#!/usr/bin/env python3
"""Validate the full DeepSeek-V4-Pro-Base checkpoint and emit the R4 MoP manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leviathan.deepseek_v4 import build_manifest as build_checkpoint_manifest  # noqa: E402
from leviathan.deepseek_v4_mop import build_transplant_manifest  # noqa: E402


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
        "--revision",
        required=True,
        help="Immutable upstream Hugging Face commit SHA used for this experiment.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=128,
        help="Contiguous SwiGLU intermediate channels per MoP tile. R4 canonical default is 128.",
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
    if not args.revision or args.revision in {"main", "master", "latest"}:
        raise SystemExit("--revision must be an immutable upstream commit SHA, not a moving ref")

    model_dir = Path(args.model_dir).expanduser().resolve()
    checkpoint_manifest = build_checkpoint_manifest(
        model_dir,
        tile_width=args.tile_width,
        require_weights=not args.config_only,
    )

    # The strict R4 manifest validates the complete architecture/FP8 contract, records
    # the immutable revision and carries the benchmark/retention/performance gates.
    strict_manifest = build_transplant_manifest(
        model_dir / "config.json",
        revision=args.revision,
    )

    if args.tile_width != strict_manifest["mop"]["tile_width"]:
        raise SystemExit(
            "Non-canonical tile widths belong to the later MoP-2 sweep. "
            f"R4 preflight currently requires {strict_manifest['mop']['tile_width']} channels."
        )

    payload = {
        "checkpoint": checkpoint_manifest.as_dict(),
        "r4_transplant": strict_manifest,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    mop = strict_manifest["mop"]
    print(
        "MoP-0 plan: "
        f"{mop['tiles_per_expert']} tiles/expert, "
        f"{mop['routed_tiles_per_layer']} routed tiles/layer, "
        f"{mop['initial_active_routed_tiles']} routed tiles/token; "
        f"full_checkpoint_verified={not args.config_only}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
