#!/usr/bin/env python3
"""Fail-closed validator for Leviathan training-data provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "rewrite" / "data" / "DATASET_REGISTRY.json"
DONORS = ROOT / "rewrite" / "donors" / "DONOR_REGISTRY.json"
MODELS = ROOT / "rewrite" / "models" / "MODEL_REGISTRY.json"


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - audit must report malformed input cleanly
        raise SystemExit(f"dataset-audit: cannot parse {path}: {exc}") from exc


def main() -> int:
    dreg = load(DATASETS)
    donors = load(DONORS)
    models = load(MODELS)

    errors: list[str] = []
    allowed_status = set(dreg.get("policy", {}).get("status_values", []))
    if not allowed_status:
        errors.append("policy.status_values must be non-empty")

    donor_ids = {x["id"] for x in donors.get("engines", [])}
    donor_ids |= {x["id"] for x in donors.get("assets_and_infrastructure", [])}
    model_ids = {x["id"] for x in models.get("models", [])}

    seen: set[str] = set()
    for i, item in enumerate(dreg.get("datasets", [])):
        where = f"datasets[{i}]"
        did = item.get("id")
        if not did:
            errors.append(f"{where}: missing id")
            continue
        if did in seen:
            errors.append(f"{where}: duplicate id {did!r}")
        seen.add(did)

        status = item.get("status")
        if status not in allowed_status:
            errors.append(f"{did}: invalid status {status!r}")

        donor_id = item.get("donor_id")
        if donor_id not in donor_ids:
            errors.append(f"{did}: unknown donor_id {donor_id!r}")

        for mid in item.get("model_links", []):
            if mid not in model_ids:
                errors.append(f"{did}: unknown model link {mid!r}")

        license_name = item.get("license")
        source = item.get("source")
        generation = item.get("generation")

        if status in {"AVAILABLE", "MIRRORED", "HASH_PINNED", "RECONSTRUCTIBLE", "PARTIAL"}:
            if not source or not source.get("locator"):
                errors.append(f"{did}: status {status} requires source.locator")
            if not license_name or license_name == "UNKNOWN":
                errors.append(f"{did}: status {status} requires a verified non-UNKNOWN data license")

        if status in {"MIRRORED", "HASH_PINNED"}:
            manifest = source.get("immutable_manifest") if source else None
            if not manifest:
                errors.append(f"{did}: status {status} requires source.immutable_manifest")

        if status == "RECONSTRUCTIBLE":
            if not generation:
                errors.append(f"{did}: RECONSTRUCTIBLE requires generation metadata")
            else:
                required = ("family", "exact_client_revision", "filtering_recipe")
                missing = [k for k in required if generation.get(k) in (None, "")]
                if missing:
                    errors.append(f"{did}: RECONSTRUCTIBLE missing generation fields {missing}")

        if item.get("exact_historical_reproduction"):
            if status not in {"HASH_PINNED", "RECONSTRUCTIBLE"}:
                errors.append(
                    f"{did}: exact_historical_reproduction requires HASH_PINNED or RECONSTRUCTIBLE"
                )
            if not item.get("model_links"):
                errors.append(f"{did}: exact reproduction claim requires at least one model link")
            if not generation:
                errors.append(f"{did}: exact reproduction claim requires generation metadata")

    if not dreg.get("datasets"):
        errors.append("dataset registry must contain at least one dataset")

    if errors:
        print("dataset-audit: FAILED", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        return 1

    print(
        "dataset-audit: PASS "
        f"datasets={len(seen)} donors={len(donor_ids)} models={len(model_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
