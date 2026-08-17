#!/usr/bin/env python3
"""Fetch a pinned pretrained model and verify its SHA-256 filename prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "MODEL_REGISTRY.json"


def load_models():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {m["id"]: m for m in data["models"]}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "cache"))
    args = parser.parse_args()

    models = load_models()
    if args.model_id not in models:
        raise SystemExit(f"unknown model: {args.model_id}")
    model = models[args.model_id]
    outdir = pathlib.Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    destination = outdir / model["filename"]

    if destination.exists():
        digest = sha256(destination)
        if digest.startswith(model["sha256_prefix"]):
            print(f"MODEL OK: {destination} sha256={digest}")
            return 0
        raise SystemExit(f"existing model hash mismatch: {destination} sha256={digest}")

    url = model["fetch_template"].format(filename=model["filename"])
    with tempfile.NamedTemporaryFile(delete=False, dir=outdir, prefix=".download-") as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        print(f"fetching {args.model_id} -> {destination}")
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        digest = sha256(tmp_path)
        if not digest.startswith(model["sha256_prefix"]):
            raise SystemExit(
                f"download hash mismatch: expected prefix {model['sha256_prefix']}, got {digest}"
            )
        tmp_path.replace(destination)
        print(f"MODEL OK: {destination} sha256={digest} license={model['license']}")
        return 0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
