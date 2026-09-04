#!/usr/bin/env python3
"""Validate Leviathan's model registry, Omega references and canonical V4 MoP spec."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "spec" / "model-registry.toml"
OMEGA_PATH = ROOT / "spec" / "omega-transplant.toml"
V4_MOP_PATH = ROOT / "spec" / "deepseek-v4-mop.toml"

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
    v4_mop = load(V4_MOP_PATH)

    models = registry.get("models", [])
    if not models:
        errors.append("registry has no [[models]] entries")

    ids: set[str] = set()
    repo_ids: set[str] = set()
    model_by_id: dict[str, dict] = {}

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
        model_by_id[model_id] = model

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

    experimental_id = substrate.get("experimental")
    if experimental_id in model_by_id:
        experimental = model_by_id[experimental_id]
        if experimental.get("stage") != "base":
            errors.append("omega substrate.experimental must reference a base/pretraining checkpoint")
        if experimental_id != "deepseek-v4-pro-base":
            errors.append(
                "canonical R4 substrate drifted: expected deepseek-v4-pro-base, "
                f"got {experimental_id}"
            )

    teacher_members = omega.get("teacher_ensemble", {}).get("members", [])
    for model_id in teacher_members:
        if model_id not in ids:
            errors.append(f"teacher_ensemble references unknown model: {model_id}")

    invariants = omega.get("invariants", {})
    if invariants.get("raw_experience_updates_core", True):
        errors.append("safety invariant violated: raw_experience_updates_core must be false")
    if not invariants.get("rollback_required", False):
        errors.append("safety invariant violated: rollback_required must be true")

    mop_module = omega.get("modules", {}).get("mixture_of_parameters", {})
    if mop_module.get("substrate") != experimental_id:
        errors.append("MoP substrate must match omega substrate.experimental")
    if mop_module.get("scalar_parameter_routing", True):
        errors.append("R4 forbids scalar parameter routing")
    if mop_module.get("expert_weight_averaging", True):
        errors.append("R4 forbids expert weight averaging")

    model = v4_mop.get("model", {})
    architecture = v4_mop.get("architecture", {})
    quantization = v4_mop.get("quantization", {})
    mop = v4_mop.get("mop", {})

    if model.get("registry_id") != "deepseek-v4-pro-base":
        errors.append("DeepSeek V4 MoP spec must target deepseek-v4-pro-base")
    if model.get("stage") != "base":
        errors.append("DeepSeek V4 MoP spec must target a base checkpoint")

    intermediate = int(architecture.get("moe_intermediate_size", 0))
    tile_width = int(mop.get("tile_width", 0))
    routed_experts = int(architecture.get("n_routed_experts", 0))
    active_experts = int(architecture.get("num_experts_per_tok", 0))
    if intermediate <= 0 or tile_width <= 0 or intermediate % tile_width:
        errors.append("MoP tile_width must exactly divide moe_intermediate_size")
    else:
        tiles_per_expert = intermediate // tile_width
        if tiles_per_expert != 24:
            errors.append(f"unexpected V4 tiles/expert: {tiles_per_expert}")
        if routed_experts * tiles_per_expert != 9216:
            errors.append("unexpected routed tile count for canonical V4 plan")
        if active_experts * tiles_per_expert != 144:
            errors.append("unexpected MoP-0 active routed tile count")

    block_size = quantization.get("weight_block_size", [])
    if tile_width not in block_size:
        errors.append("canonical MoP tile width should align with the V4 FP8 weight block")

    retention = v4_mop.get("evaluation", {}).get("retention", {})
    if not retention.get("heldout_must_never_enter_training", False):
        errors.append("held-out WikiText gate must be excluded from training")
    if float(retention.get("max_relative_public_language_loss_increase", 1.0)) > 0.02:
        errors.append("public-language retention hard gate may not exceed +2%")

    performance = v4_mop.get("evaluation", {}).get("performance", {})
    if not performance.get("wall_clock_must_not_regress", False):
        errors.append("R4 must reject wall-clock regressions")

    if errors:
        print("Registry validation FAILED:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Registry validation passed: "
        f"{len(models)} models, {len(teacher_members)} teachers, "
        "DeepSeek V4 is canonical R4 substrate, MoP-0=24 tiles/expert."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
