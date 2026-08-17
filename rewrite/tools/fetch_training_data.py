#!/usr/bin/env python3
"""Materialize approved training-data shards without turning the repository into a data dump."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "training" / "DATA_SOURCE_LOCKS.json"


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise SystemExit("unsupported DATA_SOURCE_LOCKS schema")
    return data


def source_by_id(manifest: dict, source_id: str) -> dict:
    matches = [x for x in manifest.get("sources", []) if x.get("id") == source_id]
    if len(matches) != 1:
        raise SystemExit(f"unknown or duplicate source id: {source_id}")
    return matches[0]


def head_metadata(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Leviathan-training-data/1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {
                "status": getattr(r, "status", None),
                "content_length": int(r.headers["Content-Length"]) if r.headers.get("Content-Length") else None,
                "etag": r.headers.get("ETag"),
                "last_modified": r.headers.get("Last-Modified"),
                "content_type": r.headers.get("Content-Type"),
            }
    except urllib.error.HTTPError as exc:
        # Some object stores reject HEAD while serving GET correctly. Preserve the failure
        # as metadata instead of silently assuming anything about object size.
        return {"head_error": f"HTTP {exc.code}"}
    except Exception as exc:  # pragma: no cover - network/environment dependent
        return {"head_error": str(exc)}


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(source: dict, output_dir: Path, max_bytes: int, force: bool) -> Path:
    url = source["fetch_url"]
    name = Path(urllib.request.urlparse(url).path).name or source["id"]
    target = output_dir / source["id"] / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing data: {target}; use --force")

    meta = head_metadata(url)
    announced = meta.get("content_length")
    if announced is not None and announced > max_bytes:
        raise SystemExit(f"source is {announced} bytes, over --max-bytes={max_bytes}")

    tmp = target.with_suffix(target.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": "Leviathan-training-data/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise RuntimeError(f"download exceeded --max-bytes={max_bytes}")
                out.write(block)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    expected_bytes = source.get("expected_bytes")
    if expected_bytes is not None and total != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"byte-length mismatch: got {total}, expected {expected_bytes}")

    actual_sha256 = sha256_file(tmp)
    expected_sha256 = source.get("expected_sha256")
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("SHA-256 mismatch")

    expected_blob = source.get("expected_git_blob_sha1")
    actual_blob = git_blob_sha1(tmp) if expected_blob else None
    if expected_blob and actual_blob != expected_blob:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Git blob SHA mismatch: got {actual_blob}, expected {expected_blob}")

    os.replace(tmp, target)
    sidecar = target.with_suffix(target.suffix + ".provenance.json")
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "source_id": source["id"],
        "dataset_id": source["dataset_id"],
        "source_url": url,
        "license": source["license"],
        "bytes": total,
        "sha256": actual_sha256,
        "git_blob_sha1": actual_blob,
        "http_metadata": meta,
        "source_lock": source,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(target)
    print(sidecar)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_id", nargs="?")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--output-dir", default=str(ROOT / "training" / "materialized"))
    ap.add_argument("--max-bytes", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", action="store_true", help="HEAD/probe only; never downloads")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    if args.list:
        for item in manifest.get("sources", []):
            print(f"{item['id']}\t{item['kind']}\t{item['fetch_url']}")
        return 0
    if not args.source_id:
        ap.error("source_id is required unless --list is used")
    source = source_by_id(manifest, args.source_id)
    cap = args.max_bytes if args.max_bytes is not None else int(source["default_max_bytes"])
    if cap <= 0:
        ap.error("--max-bytes must be positive")
    if args.probe:
        print(json.dumps({"source_id": source["id"], "url": source["fetch_url"], "http_metadata": head_metadata(source["fetch_url"]), "max_bytes": cap}, indent=2, sort_keys=True))
        return 0
    fetch(source, Path(args.output_dir), cap, args.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"training-data fetch failed: {exc}", file=sys.stderr)
        raise
