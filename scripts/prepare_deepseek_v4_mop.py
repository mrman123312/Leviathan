#!/usr/bin/env python3
"""Validate a DeepSeek-V4-Pro-Base config and emit a Leviathan MoP transplant manifest.

This does not download or rewrite the 1.6T checkpoint. It is the preflight step that
locks the exact upstream architecture and the function-preserving MoP-0 tile plan
before any distributed weight job is allowed to start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leviathan.deepseek_v4_mop import build_transplant_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.json from the pinned DeepSeek-V4-Pro-Base checkpoint.",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable upstream Hugging Face commit SHA used for this experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, the manifest is printed to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_transplant_manifest(args.config, revision=args.revision)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"DeepSeek V4 MoP preflight FAILED: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")

    mop = manifest["mop"]
    print(
        "MoP-0 plan: "
        f"{mop['tiles_per_expert']} tiles/expert, "
        f"{mop['routed_tiles_per_layer']} routed tiles/layer, "
        f"{mop['initial_active_routed_tiles']} routed tiles/token.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
