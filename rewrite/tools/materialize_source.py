#!/usr/bin/env python3
"""Materialize a pinned donor source lock into a local cache and verify git blobs.

This keeps third-party source out of the Leviathan repository until an actual
adaptation is selected, while still making exact donor code reproducible.

Example:
  python3 rewrite/tools/materialize_source.py rewrite/donors/locks/fathom.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "donors" / "DONOR_REGISTRY.json"


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity, not security


def load_registry():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = data.get("engines", []) + data.get("assets_and_infrastructure", [])
    return {e["id"]: e for e in entries}


def policy(entry):
    return entry.get("direct_code_policy", entry.get("direct_use_policy", "blocked"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lockfile")
    parser.add_argument("--output-dir", default=str(ROOT / ".donor-cache"))
    args = parser.parse_args()

    lock_path = pathlib.Path(args.lockfile)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    donors = load_registry()
    donor_id = lock["donor_id"]
    if donor_id not in donors:
        raise SystemExit(f"unregistered donor: {donor_id}")
    entry = donors[donor_id]
    if policy(entry) not in {"allowed", "allowed_code_only", "allowed_version_gated"}:
        raise SystemExit(f"donor {donor_id} is not approved for direct source materialization")
    if lock["license"] != entry["license"]:
        raise SystemExit(f"license mismatch: lock={lock['license']} registry={entry['license']}")

    repo = lock["source_repo"]
    revision = lock["source_revision"]
    if revision in {"main", "master", "latest", "HEAD", ""}:
        raise SystemExit("source_revision must be immutable")

    outroot = pathlib.Path(args.output_dir) / donor_id / revision
    for item in lock["files"]:
        rel = pathlib.PurePosixPath(item["path"])
        encoded_path = "/".join(urllib.parse.quote(x, safe="") for x in rel.parts)
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{encoded_path}"
        print(f"fetch {repo}@{revision}:{item['path']}")
        with urllib.request.urlopen(url) as response:
            data = response.read()
        actual = git_blob_sha1(data)
        if actual != item["git_blob_sha1"]:
            raise SystemExit(
                f"blob mismatch for {item['path']}: expected {item['git_blob_sha1']} got {actual}"
            )
        dest = outroot.joinpath(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    provenance = {
        "donor_id": donor_id,
        "source_repo": repo,
        "source_revision": revision,
        "license": lock["license"],
        "lockfile": str(lock_path),
        "files": lock["files"],
        "purpose": lock.get("purpose", "")
    }
    (outroot / "LEVIATHAN_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SOURCE OK: {outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
