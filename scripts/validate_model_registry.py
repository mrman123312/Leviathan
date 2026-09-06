#!/usr/bin/env python3
"""Validate current consumer identities and retained legacy trust/parity gates."""
from pathlib import Path
import math
import tomllib
ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODEL_ID = "qwen3.8-27b"
LEGACY_MODEL_ID = "deepseek-v4-pro-base"
GATES = ["specification", "executable", "integrated", "learned", "demonstrated"]
PIPELINE = ["representation_compiler", "cognitive_program_compiler", "dynamic_cognitive_graph", "theory_prediction", "evidence_update", "learning_router", "cognitive_compilation"]
DESTINATIONS = ["ignore", "episodic_memory", "semantic_memory", "procedural_memory", "plastic_parameters", "core_parameters"]

def load(path):
    with Path(path).open("rb") as f:
        return tomllib.load(f)

def main():
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    def expect(section, values, label):
        for key, value in values.items():
            check(section.get(key) == value, f"{label}.{key}: expected {value!r}")
    def require(section, names, label):
        expect(section, dict.fromkeys(names, True), label)
    def read(name):
        return load(ROOT / "spec" / name)
    registry, omega = read("model-registry.toml"), read("omega-transplant.toml")
    deepseek, cells = read("deepseek-v4-mop.toml"), read("parameter-cells.toml")
    maturity, kernel = read("architecture-maturity.toml"), read("cognitive-kernel.toml")
    consumer = read("consumer-substrate.toml")
    models = registry.get("models", [])
    check(bool(models), "Model registry is empty")
    ids, repos, canonical = set(), set(), []
    for i, model in enumerate(models):
        required = {"id", "repo_id", "role", "stage", "license", "enabled_for_download", "priority"}
        check(required <= model.keys(), f"model[{i}] missing required fields")
        mid, repo = model.get("id"), model.get("repo_id", "")
        check(mid not in ids, f"Duplicate model id {mid}")
        check(repo not in repos and "/" in repo, f"Invalid/duplicate repo {repo}")
        ids.add(mid); repos.add(repo)
        total, active = float(model.get("total_parameters_b", 0)), float(model.get("active_parameters_b", 0))
        check(math.isfinite(total) and math.isfinite(active) and min(total, active) >= 0 and (not total or active <= total), f"Invalid parameter counts {mid}")
        if model.get("role") == "canonical_semantic_substrate":
            canonical.append(mid)
    check(canonical == [CANONICAL_MODEL_ID], "Exactly one Qwen27B canonical substrate required")
    substrate = omega.get("substrate", {})
    for field, mid in substrate.items():
        check(mid in ids, f"Unknown substrate.{field}: {mid}")
    expect(substrate, {"canonical": CANONICAL_MODEL_ID, "experimental": CANONICAL_MODEL_ID}, "substrate")
    teachers = omega.get("teacher_ensemble", {}).get("members", [])
    check(all(mid in ids for mid in teachers), "Unknown teacher identity")
    expect(omega.get("invariants", {}), {"raw_experience_updates_core": False, "rollback_required": True, "single_cognitive_model": True, "full_canonical_checkpoint_required": True, "legacy_deepseek_full_checkpoint_required": True}, "omega")
    expect(deepseek, {"source_model": LEGACY_MODEL_ID, "full_checkpoint_required": True, "single_cognitive_model": True}, "historical DeepSeek")
    expect(deepseek.get("verified_source_fingerprint", {}), {"architecture": "DeepseekV4ForCausalLM", "model_type": "deepseek_v4", "num_hidden_layers": 61, "hidden_size": 7168, "moe_intermediate_size": 3072, "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6, "max_position_embeddings": 1048576, "weight_shards": 64}, "DeepSeek fingerprint")
    expect(deepseek.get("conversion", {}), {"tile_width": 128, "tiles_per_expert": 24, "routed_tiles_per_layer": 9216, "baseline_active_routed_tiles_per_token": 144, "independent_tile_routing_at_initialization": False}, "DeepSeek conversion")
    expect(cells, {"source_substrate": CANONICAL_MODEL_ID, "single_cognitive_model": True}, "cells")
    expect(cells.get("invariant", {}), {"initial_cell_influence": 0., "independent_agents": False}, "cell invariant")
    require(cells.get("invariant", {}), ["original_router_retained", "pretrained_tile_computation_retained", "shared_expert_unchanged", "one_global_state", "one_training_objective", "one_parameter_ownership_system", "one_final_output"], "cell invariant")
    live = cells.get("live_reference", {})
    expect(live, {name + "_influence_at_insertion": 0. for name in ["independent_route", "communication", "recruitment", "local_state", "refinement"]}, "live cell gates")
    require(live, ["recruited_cell_executes_ancestral_swiglu_tile", "independent_route_executes_ancestral_swiglu_tiles", "peer_communication_runs_inside_packed_expert_forward", "recruited_cells_join_second_token_local_communication_round", "local_state_is_ephemeral", "local_state_requires_explicit_reset_between_sequences_or_tasks", "reference_stage_does_not_imply_maturity_promotion"], "live cell reference")
    expect(cells.get("independent_routing", {}), {"blend_gate_initial": 0., "donor_route_retained_at_insertion": True, "expert_boundaries_are_not_fundamental": True}, "independent routing")
    expect(cells.get("roadmap", {}), {"stages": list(range(10))}, "MoP roadmap")
    require(cells.get("acceptance", {}), ["logit_parity_required_before_stage_1", "hidden_state_parity_required_before_stage_1", "arc_easy_canary_required", "wikitext_retention_gate_required", "wall_clock_efficiency_required", "mathematical_sparsity_alone_is_failure"], "acceptance")
    expect(kernel, {"canonical_model_id": CANONICAL_MODEL_ID, "single_cognitive_model": True}, "kernel")
    expect(kernel.get("invariant", {}), {"semantic_model_count": 1, "subagent_committee": False, "compression_may_raise_trust": False, "raw_experience_updates_core": False}, "kernel invariant")
    expect(kernel.get("pipeline", {}), {"stages": PIPELINE}, "cognitive pipeline")
    expect(kernel.get("learning", {}), {"destinations": DESTINATIONS}, "learning")
    require(kernel.get("learning", {}), ["core_requires_" + x for x in ["independent_verification", "replay", "calibration", "safety", "shadow", "rollback", "external_promotion_authority"]], "core promotion")
    expect(maturity.get("maturity", {}), {"gates": GATES}, "maturity")
    layers = maturity.get("layers", {})
    check(set(layers) == {f"L{i}" for i in range(11)} | {"L1.5"}, "Incomplete layer ledger")
    for lid, layer in layers.items():
        gates = layer.get("gates", {})
        check(list(gates) == GATES, f"Incomplete/reordered gates: {lid}")
        check(all(v in {"not_started", "partial", "passed"} for v in gates.values()), f"Invalid gate state: {lid}")
        check(gates.get("demonstrated") != "passed", f"{lid}: promotion needs a dedicated empirical record")
    order = maturity.get("development", {}).get("build_order", [])
    check(order[:6] == ["L1", "L1.5", "L2", "L5", "L8", "L6"] and order[-1:] == ["L10"], "Build order changed")
    expect(consumer, {"canonical_model_id": CANONICAL_MODEL_ID, "canonical_stage": "posttrained", "pretraining_control": "qwen3-1.7b-base", "single_cognitive_model": True}, "consumer")
    require(consumer.get("invariants", {}), ["request_local_recurrent_state", "no_cross_batch_state_averaging", "opaque_quantized_weights_never_sliced_as_float", "new_path_connected_during_training", "donor_frozen_for_graft_training"], "consumer invariant")
    expect(consumer.get("invariants", {}), {"raw_experience_updates_core": False, "initial_graft_influence": 0.}, "consumer invariant")
    by_id = {m.get("id"): m for m in models}
    for key, mid, stage in [("canonical_revision", CANONICAL_MODEL_ID, "posttrained"), ("control_revision", "qwen3-1.7b-base", "base")]:
        revision = consumer.get(key, "")
        check(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), f"Unpinned {key}")
        expect(by_id.get(mid, {}), {"revision": revision, "stage": stage}, mid)
    if errors:
        print("Registry validation FAILED:\n" + "\n".join("- " + e for e in errors))
        return 1
    print(f"Registry validation passed: {len(models)} models, {len(teachers)} teachers, canonical={CANONICAL_MODEL_ID}; consumer and historical invariants intact.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
