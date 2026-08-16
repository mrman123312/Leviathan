#!/usr/bin/env python3
"""Fail closed on untracked donor code/assets.

Usage:
  python3 rewrite/tools/audit_donors.py \
      rewrite/donors/DONOR_REGISTRY.json rewrite/imports

Imported files must live under rewrite/imports/<donor-id>/ and each non-metadata
file must have a sibling `<filename>.provenance.json` describing its exact
origin. AGPL donors are blocked by default. The goal is scientific and legal
traceability, not merely license compliance.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

ALLOWED_POLICIES = {
    "allowed",
    "allowed_code_only",
    "allowed_version_gated",
}
PROVENANCE_SUFFIX = ".provenance.json"
IGNORED_NAMES = {"README.md", ".gitkeep"}
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
        if entry.get("license_status", "").startswith("verified") is False:
            fail(f"{donor_id}: license is not verified")
        if not entry.get("license"):
            fail(f"{donor_id}: missing license")
        policy = policy_of(entry)
        if entry.get("license", "").startswith("AGPL") and policy != "reference_only":
            fail(f"{donor_id}: AGPL donor must remain reference_only under current project policy")
        if policy == "allowed_version_gated" and not entry.get("version_constraint"):
            fail(f"{donor_id}: version-gated donor has no version constraint")


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
    if p["source_revision"] in {"", "master", "main", "latest", "HEAD"}:
        fail(f"{sidecar}: source_revision must be an immutable tag or commit SHA")
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

    imports = 0
    if import_root.exists():
        for path in sorted(import_root.rglob("*")):
            if not path.is_file() or path.name in IGNORED_NAMES or path.name.endswith(PROVENANCE_SUFFIX):
                continue
            if len(path.relative_to(import_root).parts) < 2:
                fail(f"{path}: imports must be nested under rewrite/imports/<donor-id>/")
            imports += audit_import(path, donors)

    print(f"DONOR-AUDIT OK: {len(donors)} registered donors/assets; {imports} imported artifacts audited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
