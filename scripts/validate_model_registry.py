#!/usr/bin/env python3
"""Validate Leviathan's model registry and Omega transplant references using stdlib only."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "spec" / "model-registry.toml"
OMEGA_PATH = ROOT / "spec" / "omega-transplant.toml"

REQUIRED_MODEL_FIELDS = {
    "id",
    "repo_id",
    "role",
    "stage",
    "license",
    "enabled_for_download",
    "priority",
}


def load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def main() -> int:
    errors: list[str] = []
    registry = load(REGISTRY_PATH)
    omega = load(OMEGA_PATH)

    models = registry.get("models", [])
    if not models:
        errors.append("registry has no [[models]] entries")

    ids: set[str] = set()
    repo_ids: set[str] = set()

    for index, model in enumerate(models):
        missing = REQUIRED_MODEL_FIELDS - set(model)
        if missing:
            errors.append(f"model[{index}] missing fields: {sorted(missing)}")
            continue

        model_id = str(model["id"])
        repo_id = str(model["repo_id"])
        if model_id in ids:
            errors.append(f"duplicate model id: {model_id}")
        ids.add(model_id)

        if repo_id in repo_ids:
            errors.append(f"duplicate repo_id: {repo_id}")
        repo_ids.add(repo_id)

        if "/" not in repo_id:
            errors.append(f"invalid Hugging Face repo_id for {model_id}: {repo_id}")

        total = float(model.get("total_parameters_b", 0.0))
        active = float(model.get("active_parameters_b", 0.0))
        if total < 0 or active < 0:
            errors.append(f"negative parameter count for {model_id}")
        if total > 0 and active > total:
            errors.append(f"active parameters exceed total parameters for {model_id}")

    substrate = omega.get("substrate", {})
    for field, model_id in substrate.items():
        if model_id not in ids:
            errors.append(f"omega substrate.{field} references unknown model: {model_id}")

    teacher_members = omega.get("teacher_ensemble", {}).get("members", [])
    for model_id in teacher_members:
        if model_id not in ids:
            errors.append(f"teacher_ensemble references unknown model: {model_id}")

    if omega.get("invariants", {}).get("raw_experience_updates_core", True):
        errors.append("safety invariant violated: raw_experience_updates_core must be false")

    if not omega.get("invariants", {}).get("rollback_required", False):
        errors.append("safety invariant violated: rollback_required must be true")

    if errors:
        print("Registry validation FAILED:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Registry validation passed: {len(models)} models, {len(teacher_members)} teachers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
