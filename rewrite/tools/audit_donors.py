#!/usr/bin/env python3
"""Fail closed on untracked donor code, source locks, and model assets.

Usage:
  python3 rewrite/tools/audit_donors.py \
      rewrite/donors/DONOR_REGISTRY.json rewrite/imports

The audit validates:
- donor/license policy;
- AGPL reference-only gating;
- immutable source-lock revisions and Git blob identities;
- model registry donor/license/hash consistency;
- committed copied/adapted artifacts and their provenance sidecars.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Any

ALLOWED_POLICIES = {"allowed", "allowed_code_only", "allowed_version_gated"}
PROVENANCE_SUFFIX = ".provenance.json"
IGNORED_NAMES = {"README.md", ".gitkeep"}
IMMUTABLE_BAD = {"", "master", "main", "latest", "HEAD"}
REQUIRED_PROVENANCE = {
    "donor_id",
    "source_repo",
    "source_revision",
    "source_path",
    "license",
    "reuse_mode",
    "leviathan_owner",
    "import_reason",
    "tests_required",
}
VALID_REUSE_MODES = {"copied", "adapted", "reimplemented", "model", "dataset", "tool"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX12 = re.compile(r"^[0-9a-f]{12}$")


def fail(msg: str) -> None:
    print(f"DONOR-AUDIT FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit should fail closed
        fail(f"cannot parse {path}: {exc}")


def build_donor_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    donors: dict[str, dict[str, Any]] = {}
    for section in ("engines", "assets_and_infrastructure"):
        for entry in registry.get(section, []):
            donor_id = entry.get("id")
            if not donor_id:
                fail(f"entry in {section} has no id")
            if donor_id in donors:
                fail(f"duplicate donor id: {donor_id}")
            donors[donor_id] = entry
    return donors


def policy_of(entry: dict[str, Any]) -> str:
    return entry.get("direct_code_policy", entry.get("direct_use_policy", "blocked"))


def audit_registry(registry: dict[str, Any], donors: dict[str, dict[str, Any]]) -> None:
    if registry.get("schema_version") != 1:
        fail("unsupported donor registry schema")
    for donor_id, entry in donors.items():
        if not entry.get("license_status", "").startswith("verified"):
            fail(f"{donor_id}: license is not verified")
        if not entry.get("license"):
            fail(f"{donor_id}: missing license")
        policy = policy_of(entry)
        if entry.get("license", "").startswith("AGPL") and policy != "reference_only":
            fail(f"{donor_id}: AGPL donor must remain reference_only under current project policy")
        if policy == "allowed_version_gated" and not entry.get("version_constraint"):
            fail(f"{donor_id}: version-gated donor has no version constraint")


def audit_source_locks(registry_path: pathlib.Path, donors: dict[str, dict[str, Any]]) -> int:
    lock_root = registry_path.parent / "locks"
    count = 0
    if not lock_root.exists():
        return 0
    for path in sorted(lock_root.glob("*.json")):
        lock = load_json(path)
        donor_id = lock.get("donor_id")
        if donor_id not in donors:
            fail(f"{path}: unregistered donor_id {donor_id!r}")
        entry = donors[donor_id]
        if policy_of(entry) not in ALLOWED_POLICIES:
            fail(f"{path}: donor {donor_id} is not approved for direct source materialization")
        if lock.get("license") != entry.get("license"):
            fail(f"{path}: license mismatch with donor registry")
        if lock.get("source_revision") in IMMUTABLE_BAD:
            fail(f"{path}: source_revision must be immutable")
        if not HEX40.fullmatch(lock.get("source_revision", "")):
            fail(f"{path}: source_revision must be a full 40-character commit SHA")
        repo = entry.get("repo")
        if repo and lock.get("source_repo") != repo:
            fail(f"{path}: source_repo does not match donor registry")
        seen: set[str] = set()
        files = lock.get("files", [])
        if not files:
            fail(f"{path}: source lock has no files")
        for item in files:
            rel = item.get("path", "")
            blob = item.get("git_blob_sha1", "")
            if not rel or rel.startswith("/") or ".." in pathlib.PurePosixPath(rel).parts:
                fail(f"{path}: unsafe source path {rel!r}")
            if rel in seen:
                fail(f"{path}: duplicate source path {rel}")
            seen.add(rel)
            if not HEX40.fullmatch(blob):
                fail(f"{path}: {rel} has invalid Git blob SHA-1")
        count += 1
    return count


def audit_model_registry(registry_path: pathlib.Path, donors: dict[str, dict[str, Any]]) -> int:
    model_path = registry_path.parent.parent / "models" / "MODEL_REGISTRY.json"
    if not model_path.exists():
        return 0
    data = load_json(model_path)
    if data.get("schema_version") != 1:
        fail(f"{model_path}: unsupported schema")
    ids: set[str] = set()
    count = 0
    for model in data.get("models", []):
        model_id = model.get("id")
        if not model_id or model_id in ids:
            fail(f"{model_path}: missing/duplicate model id {model_id!r}")
        ids.add(model_id)
        donor_id = model.get("donor_id")
        if donor_id not in donors:
            fail(f"{model_path}: {model_id} references unregistered donor {donor_id!r}")
        donor = donors[donor_id]
        if model.get("license") != donor.get("license"):
            fail(f"{model_path}: {model_id} license mismatches donor registry")
        prefix = model.get("sha256_prefix", "")
        filename = model.get("filename", "")
        if not HEX12.fullmatch(prefix):
            fail(f"{model_path}: {model_id} needs a 12-hex SHA-256 prefix")
        if filename != f"nn-{prefix}.nnue":
            fail(f"{model_path}: {model_id} filename/hash prefix mismatch")
        if "{filename}" not in model.get("fetch_template", ""):
            fail(f"{model_path}: {model_id} fetch template must contain {{filename}}")
        count += 1
    return count


def audit_import(imported: pathlib.Path, donors: dict[str, dict[str, Any]]) -> int:
    donor_id = imported.relative_to(import_root).parts[0]
    if donor_id not in donors:
        fail(f"{imported}: donor directory {donor_id!r} is not registered")
    entry = donors[donor_id]
    policy = policy_of(entry)
    if policy not in ALLOWED_POLICIES:
        fail(f"{imported}: donor {donor_id} is {policy}, so direct import is forbidden")

    sidecar = imported.with_name(imported.name + PROVENANCE_SUFFIX)
    if not sidecar.exists():
        fail(f"{imported}: missing provenance sidecar {sidecar.name}")
    p = load_json(sidecar)
    missing = REQUIRED_PROVENANCE - set(p)
    if missing:
        fail(f"{sidecar}: missing fields {sorted(missing)}")
    if p["donor_id"] != donor_id:
        fail(f"{sidecar}: donor_id does not match directory")
    if p["license"] != entry["license"]:
        fail(f"{sidecar}: license {p['license']} != registry {entry['license']}")
    if p["reuse_mode"] not in VALID_REUSE_MODES:
        fail(f"{sidecar}: unsupported reuse_mode {p['reuse_mode']!r}")
    if policy == "allowed_version_gated" and not p.get("version_gate_verified"):
        fail(f"{sidecar}: version-gated donor requires version_gate_verified=true")
    if p["source_revision"] in IMMUTABLE_BAD:
        fail(f"{sidecar}: source_revision must be immutable")
    if not p["tests_required"]:
        fail(f"{sidecar}: tests_required may not be empty")
    return 1


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    registry_path = pathlib.Path(sys.argv[1])
    global import_root
    import_root = pathlib.Path(sys.argv[2])
    registry = load_json(registry_path)
    donors = build_donor_map(registry)
    audit_registry(registry, donors)
    locks = audit_source_locks(registry_path, donors)
    models = audit_model_registry(registry_path, donors)

    imports = 0
    if import_root.exists():
        for path in sorted(import_root.rglob("*")):
            if not path.is_file() or path.name in IGNORED_NAMES or path.name.endswith(PROVENANCE_SUFFIX):
                continue
            if len(path.relative_to(import_root).parts) < 2:
                fail(f"{path}: imports must be nested under rewrite/imports/<donor-id>/")
            imports += audit_import(path, donors)

    print(
        f"DONOR-AUDIT OK: {len(donors)} donors/assets; {locks} source locks; "
        f"{models} models; {imports} committed imports"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
