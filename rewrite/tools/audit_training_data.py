#!/usr/bin/env python3
"""Fail closed on training-data provenance, mutable toolchain refs, bridge dependencies and accidental bulk-data commits."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "training" / "DATASET_REGISTRY.json"
LOCKS = ROOT / "training" / "DATA_SOURCE_LOCKS.json"
SYZYGY3 = ROOT / "training" / "SYZYGY3_LOCK.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BULK_SUFFIXES = {".binpack", ".tar", ".tgz", ".gz", ".zst", ".bz2", ".7z"}
MAX_COMMITTED_DATA_BYTES = 2 * 1024 * 1024
EXPECTED_SYZYGY3 = {
    "KPvK.rtbw", "KPvK.rtbz", "KNvK.rtbw", "KNvK.rtbz", "KBvK.rtbw",
    "KBvK.rtbz", "KRvK.rtbw", "KRvK.rtbz", "KQvK.rtbw", "KQvK.rtbz",
}


def fail(msg: str) -> None:
    print(f"training-data audit: {msg}", file=sys.stderr)
    raise SystemExit(1)


def unique(items: list[dict], field: str, where: str) -> None:
    seen: set[str] = set()
    for item in items:
        value = item.get(field)
        if not isinstance(value, str) or not value:
            fail(f"{where}: missing {field}")
        if value in seen:
            fail(f"{where}: duplicate {field}={value}")
        seen.add(value)


def main() -> int:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    locks = json.loads(LOCKS.read_text(encoding="utf-8"))
    syzygy = json.loads(SYZYGY3.read_text(encoding="utf-8"))
    if reg.get("schema_version") != 2:
        fail("unexpected DATASET_REGISTRY schema")
    if locks.get("schema_version") != 1:
        fail("unexpected DATA_SOURCE_LOCKS schema")
    if syzygy.get("schema_version") != 1:
        fail("unexpected SYZYGY3_LOCK schema")

    datasets = reg.get("datasets", [])
    sources = locks.get("sources", [])
    toolchains = locks.get("toolchains", [])
    unique(datasets, "id", "registry")
    unique(sources, "id", "source locks")
    unique(toolchains, "id", "toolchain locks")
    dataset_ids = {x["id"] for x in datasets}

    required = {"source", "license", "format", "lineage", "selection_rules"}
    for ds in datasets:
        missing = [k for k in required if k not in ds or ds[k] in (None, "")]
        if missing:
            fail(f"dataset {ds['id']} missing fields: {missing}")
        if "unknown" in str(ds["license"]).lower() or ds["license"] == "unspecified":
            fail(f"dataset {ds['id']} has unresolved license")
        parent = ds.get("lineage", {}).get("parent_dataset")
        if parent and parent not in dataset_ids:
            fail(f"dataset {ds['id']} references unknown parent {parent}")

    for src in sources:
        if src.get("dataset_id") not in dataset_ids:
            fail(f"source {src['id']} references unknown dataset {src.get('dataset_id')}")
        url = src.get("fetch_url", "")
        if not url.startswith("https://"):
            fail(f"source {src['id']} must use https")
        if not isinstance(src.get("default_max_bytes"), int) or src["default_max_bytes"] <= 0:
            fail(f"source {src['id']} needs a positive default_max_bytes")
        doc = src.get("documented_by", {})
        revision = doc.get("revision")
        if revision and not HEX40.match(revision):
            fail(f"source {src['id']} uses mutable/non-commit documentation revision {revision}")
        sha = src.get("expected_sha256")
        if sha is not None and not HEX64.match(str(sha)):
            fail(f"source {src['id']} has malformed expected_sha256")

    for tool in toolchains:
        revision = tool.get("revision")
        if not HEX40.match(str(revision or "")):
            fail(f"toolchain {tool['id']} is not pinned to a 40-hex commit")

    syzygy_files = syzygy.get("files", {})
    if set(syzygy_files) != EXPECTED_SYZYGY3:
        fail(f"SYZYGY3_LOCK must contain exactly the 10 three-piece WDL/DTZ files; got {sorted(syzygy_files)}")
    for name, digest in syzygy_files.items():
        if not HEX64.match(str(digest)):
            fail(f"SYZYGY3_LOCK malformed SHA-256 for {name}")
    if syzygy.get("observed_total_bytes") != 25824:
        fail("SYZYGY3_LOCK unexpected total byte count")

    # The repository stores manifests and tiny test fixtures, not training archives.
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in {"build", "build-audit", "build-fathom", "build-sanitize", "materialized", "cache", ".donor-cache"} for part in rel.parts):
            continue
        if path.suffix.lower() in BULK_SUFFIXES and path.stat().st_size > MAX_COMMITTED_DATA_BYTES:
            fail(f"bulk training artifact committed: {rel} ({path.stat().st_size} bytes)")

    print(f"training-data audit OK: {len(datasets)} datasets, {len(sources)} fetch locks, {len(toolchains)} toolchains, 10 Syzygy bridge files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
