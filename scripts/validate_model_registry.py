#!/usr/bin/env python3
"""Validate Leviathan model registry and architecture invariants."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "spec" / "model-registry.toml"
OMEGA_PATH = ROOT / "spec" / "omega-transplant.toml"
DEEPSEEK_MOP_PATH = ROOT / "spec" / "deepseek-v4-mop.toml"
PARAMETER_CELLS_PATH = ROOT / "spec" / "parameter-cells.toml"
MATURITY_PATH = ROOT / "spec" / "architecture-maturity.toml"

REQUIRED_MODEL_FIELDS = {
    "id",
    "repo_id",
    "role",
    "stage",
    "license",
    "enabled_for_download",
    "priority",
}

CANONICAL_MODEL_ID = "deepseek-v4-pro-base"
EXPECTED_V4_FINGERPRINT = {
    "architecture": "DeepseekV4ForCausalLM",
    "model_type": "deepseek_v4",
    "num_hidden_layers": 61,
    "hidden_size": 7168,
    "moe_intermediate_size": 3072,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "max_position_embeddings": 1048576,
    "weight_shards": 64,
}
EXPECTED_MATURITY_GATES = [
    "specification",
    "executable",
    "integrated",
    "learned",
    "demonstrated",
]
EXPECTED_LAYERS = {
    "L0", "L1", "L1.5", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"
}
EXPECTED_CELL_STAGES = list(range(10))


def load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def main() -> int:
    errors: list[str] = []
    registry = load(REGISTRY_PATH)
    omega = load(OMEGA_PATH)
    deepseek_mop = load(DEEPSEEK_MOP_PATH)
    parameter_cells = load(PARAMETER_CELLS_PATH)
    maturity = load(MATURITY_PATH)

    models = registry.get("models", [])
    if not models:
        errors.append("registry has no [[models]] entries")

    ids: set[str] = set()
    repo_ids: set[str] = set()
    canonical_ids: list[str] = []

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

        if model.get("role") == "canonical_semantic_substrate":
            canonical_ids.append(model_id)

    if canonical_ids != [CANONICAL_MODEL_ID]:
        errors.append(
            "registry must contain exactly one canonical_semantic_substrate and it must be "
            f"{CANONICAL_MODEL_ID}; found {canonical_ids}"
        )

    substrate = omega.get("substrate", {})
    for field, model_id in substrate.items():
        if model_id not in ids:
            errors.append(f"omega substrate.{field} references unknown model: {model_id}")

    if substrate.get("canonical") != CANONICAL_MODEL_ID:
        errors.append(f"omega substrate.canonical must be {CANONICAL_MODEL_ID}")
    if substrate.get("experimental") != CANONICAL_MODEL_ID:
        errors.append(f"omega substrate.experimental must be {CANONICAL_MODEL_ID}")

    teacher_members = omega.get("teacher_ensemble", {}).get("members", [])
    for model_id in teacher_members:
        if model_id not in ids:
            errors.append(f"teacher_ensemble references unknown model: {model_id}")

    invariants = omega.get("invariants", {})
    if invariants.get("raw_experience_updates_core", True):
        errors.append("safety invariant violated: raw_experience_updates_core must be false")
    if not invariants.get("rollback_required", False):
        errors.append("safety invariant violated: rollback_required must be true")
    if not invariants.get("single_cognitive_model", False):
        errors.append("architecture invariant violated: single_cognitive_model must be true")
    if not invariants.get("full_deepseek_v4_checkpoint_required", False):
        errors.append("DeepSeek invariant violated: full_deepseek_v4_checkpoint_required must be true")

    if deepseek_mop.get("source_model") != CANONICAL_MODEL_ID:
        errors.append(f"DeepSeek MoP source_model must be {CANONICAL_MODEL_ID}")
    if not deepseek_mop.get("full_checkpoint_required", False):
        errors.append("DeepSeek MoP must require the full checkpoint")
    if not deepseek_mop.get("single_cognitive_model", False):
        errors.append("DeepSeek MoP must preserve the single-cognitive-model invariant")

    fingerprint = deepseek_mop.get("verified_source_fingerprint", {})
    for key, expected in EXPECTED_V4_FINGERPRINT.items():
        actual = fingerprint.get(key)
        if actual != expected:
            errors.append(f"DeepSeek V4 fingerprint {key}={actual!r}; expected {expected!r}")

    conversion = deepseek_mop.get("conversion", {})
    if conversion.get("tile_width") != 128:
        errors.append("DeepSeek MoP tile_width must currently be 128")
    if conversion.get("tiles_per_expert") != 24:
        errors.append("DeepSeek MoP tiles_per_expert must be 24")
    if conversion.get("routed_tiles_per_layer") != 9216:
        errors.append("DeepSeek MoP routed_tiles_per_layer must be 9216")
    if conversion.get("baseline_active_routed_tiles_per_token") != 144:
        errors.append("DeepSeek MoP baseline_active_routed_tiles_per_token must be 144")
    if conversion.get("independent_tile_routing_at_initialization", True):
        errors.append("independent tile routing must be disabled at initialization")

    # Parameter ecology invariants. Cells are parts of one neural system, never subagents.
    if parameter_cells.get("source_substrate") != CANONICAL_MODEL_ID:
        errors.append(f"parameter cells source_substrate must be {CANONICAL_MODEL_ID}")
    if not parameter_cells.get("single_cognitive_model", False):
        errors.append("parameter cells must preserve one cognitive model")
    cell_invariant = parameter_cells.get("invariant", {})
    if cell_invariant.get("initial_cell_influence") != 0.0:
        errors.append("parameter-cell influence must be exactly zero at insertion")
    if cell_invariant.get("independent_agents", True):
        errors.append("parameter cells may not be independent agents")
    for key in (
        "original_router_retained",
        "pretrained_tile_computation_retained",
        "shared_expert_unchanged",
        "one_global_state",
        "one_training_objective",
        "one_parameter_ownership_system",
        "one_final_output",
    ):
        if not cell_invariant.get(key, False):
            errors.append(f"parameter-cell invariant {key} must be true")

    stages = parameter_cells.get("roadmap", {}).get("stages", [])
    if stages != EXPECTED_CELL_STAGES:
        errors.append(f"parameter-cell roadmap must be MoP-0..MoP-9; found {stages}")
    acceptance = parameter_cells.get("acceptance", {})
    for key in (
        "logit_parity_required_before_stage_1",
        "hidden_state_parity_required_before_stage_1",
        "arc_easy_canary_required",
        "wikitext_retention_gate_required",
        "wall_clock_efficiency_required",
        "mathematical_sparsity_alone_is_failure",
    ):
        if not acceptance.get(key, False):
            errors.append(f"parameter-cell acceptance gate {key} must remain true")

    # Five-gate embodiment ledger. This prevents documentation from being mistaken for AGI.
    maturity_gates = maturity.get("maturity", {}).get("gates", [])
    if maturity_gates != EXPECTED_MATURITY_GATES:
        errors.append(f"maturity gates changed unexpectedly: {maturity_gates}")
    layers = maturity.get("layers", {})
    if set(layers) != EXPECTED_LAYERS:
        errors.append(
            f"maturity ledger must cover L0-L10 plus L1.5; found {sorted(layers)}"
        )
    valid_states = {"not_started", "partial", "passed"}
    for layer_id, layer in layers.items():
        gates = layer.get("gates", {})
        if list(gates) != EXPECTED_MATURITY_GATES:
            errors.append(f"{layer_id} maturity gates are incomplete or reordered")
        for gate, state in gates.items():
            if state not in valid_states:
                errors.append(f"{layer_id}.{gate} has invalid state {state!r}")
        if gates.get("demonstrated") == "passed":
            errors.append(
                f"{layer_id} is marked demonstrated without a dedicated empirical promotion record"
            )

    build_order = maturity.get("development", {}).get("build_order", [])
    if build_order[:6] != ["L1", "L1.5", "L2", "L5", "L8", "L6"]:
        errors.append("architecture build order must prioritize L1/L1.5 -> L2 -> L5 -> L8 -> L6")
    if not build_order or build_order[-1] != "L10":
        errors.append("canonical latent L10 must remain after the pretrained-function-preserving stack")

    if errors:
        print("Registry validation FAILED:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Registry validation passed: "
        f"{len(models)} models, {len(teacher_members)} teachers, canonical={CANONICAL_MODEL_ID}, "
        "MoP-0..9 parameter ecology and L0-L10 embodiment ledger valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
