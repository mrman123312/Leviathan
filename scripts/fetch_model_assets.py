#!/usr/bin/env python3
"""Fetch model metadata or checkpoints declared in spec/model-registry.toml.

The default operation is intentionally metadata-only. Multi-terabyte checkpoints require
both an explicit model ID and --weights. Models marked enabled_for_download=false additionally
require --allow-disabled.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "spec" / "model-registry.toml"

METADATA_PATTERNS = [
    "*.json",
    "*.txt",
    "*.md",
    "*.model",
    "*.tiktoken",
    "tokenizer*",
    "LICENSE*",
    "NOTICE*",
    "generation_config.json",
]

WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.index.json",
]


def load_registry() -> dict:
    with REGISTRY.open("rb") as f:
        return tomllib.load(f)


def model_map(registry: dict) -> dict[str, dict]:
    return {m["id"]: m for m in registry.get("models", [])}


def print_models(models: dict[str, dict]) -> None:
    header = f"{'id':32} {'role':34} {'stage':14} {'enabled':7} repo_id"
    print(header)
    print("-" * len(header))
    for model_id, item in sorted(models.items(), key=lambda kv: kv[1].get("priority", 999)):
        print(
            f"{model_id:32} {item.get('role', ''):34} {item.get('stage', ''):14} "
            f"{str(item.get('enabled_for_download', False)):7} {item.get('repo_id', '')}"
        )


def require_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for downloads. Install with: "
            "python -m pip install 'huggingface_hub>=0.27'"
        ) from exc
    return snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ids", nargs="*", help="Registry IDs to fetch")
    parser.add_argument("--list", action="store_true", help="List registered models")
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Also fetch weight shards. WARNING: some registered models are multi-terabyte.",
    )
    parser.add_argument(
        "--allow-disabled",
        action="store_true",
        help="Allow models marked enabled_for_download=false. Required for frontier-size entries.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable Hugging Face revision/commit. Strongly recommended for experiments.",
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("LEVIATHAN_MODEL_DIR", str(ROOT / "models" / "checkpoints")),
        help="Destination root. Defaults to LEVIATHAN_MODEL_DIR or models/checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry()
    models = model_map(registry)

    if args.list or not args.model_ids:
        print_models(models)
        if not args.model_ids:
            return 0

    unknown = [mid for mid in args.model_ids if mid not in models]
    if unknown:
        print(f"Unknown model IDs: {', '.join(unknown)}", file=sys.stderr)
        return 2

    destination_root = Path(args.model_root).expanduser().resolve()
    allow_patterns = list(METADATA_PATTERNS)
    if args.weights:
        allow_patterns.extend(WEIGHT_PATTERNS)

    if args.weights and args.revision is None:
        print(
            "WARNING: --weights without --revision is not reproducible. "
            "Pin an immutable Hugging Face commit for serious experiments.",
            file=sys.stderr,
        )

    downloader = None if args.dry_run else require_huggingface_hub()

    for model_id in args.model_ids:
        item = models[model_id]
        enabled = bool(item.get("enabled_for_download", False))
        if not enabled and not args.allow_disabled:
            print(
                f"Refusing {model_id}: registry marks it disabled for automatic download. "
                "Use --allow-disabled after checking storage, license and compute requirements.",
                file=sys.stderr,
            )
            return 3

        repo_id = item["repo_id"]
        local_dir = destination_root / model_id
        mode = "weights+metadata" if args.weights else "metadata-only"
        revision = args.revision or "<upstream-default>"
        print(f"{model_id}: {repo_id} -> {local_dir} [{mode}] revision={revision}")

        if args.dry_run:
            continue

        local_dir.mkdir(parents=True, exist_ok=True)
        assert downloader is not None
        downloader(
            repo_id=repo_id,
            revision=args.revision,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=os.environ.get("HF_TOKEN"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
