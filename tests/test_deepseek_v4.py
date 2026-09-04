from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from leviathan.deepseek_v4 import (
    DeepSeekV4Fingerprint,
    MixtureOfParametersPlan,
    build_manifest,
    verify_full_checkpoint_files,
)
from leviathan.deepseek_v4_mop import plan_from_spec


CANONICAL_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "num_hidden_layers": 61,
    "hidden_size": 7168,
    "moe_intermediate_size": 3072,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "max_position_embeddings": 1048576,
}


def canonical_weight_map() -> dict[str, str]:
    return {
        f"model.fake_parameter_{index}": f"model-{index:05d}-of-00064.safetensors"
        for index in range(1, 65)
    }


class DeepSeekV4Tests(unittest.TestCase):
    def test_canonical_fingerprint_and_tile_math(self) -> None:
        fingerprint = DeepSeekV4Fingerprint.from_mapping(CANONICAL_CONFIG)
        plan = MixtureOfParametersPlan.from_fingerprint(fingerprint, tile_width=128)

        self.assertEqual(plan.tiles_per_expert, 24)
        self.assertEqual(plan.routed_tiles_per_layer, 9216)
        self.assertEqual(plan.baseline_active_routed_tiles_per_token, 144)
        self.assertEqual(plan.layers, 61)

    def test_tensor_tile_coordinates_match_swiglu_axes(self) -> None:
        fingerprint = DeepSeekV4Fingerprint.from_mapping(CANONICAL_CONFIG)
        plan = MixtureOfParametersPlan.from_fingerprint(fingerprint)

        first = plan.tile_spec(0, 0)
        last = plan.tile_spec(383, 23)

        self.assertEqual(first.global_tile_id, 0)
        self.assertEqual(first.w1_row_bounds, (0, 128))
        self.assertEqual(first.w3_row_bounds, (0, 128))
        self.assertEqual(first.w2_column_bounds, (0, 128))

        self.assertEqual(last.global_tile_id, 9215)
        self.assertEqual(last.w1_row_bounds, (2944, 3072))
        self.assertEqual(last.w3_row_bounds, (2944, 3072))
        self.assertEqual(last.w2_column_bounds, (2944, 3072))

    def test_checkpoint_and_fused_serving_tile_views_are_consistent(self) -> None:
        fingerprint = DeepSeekV4Fingerprint.from_mapping(CANONICAL_CONFIG)
        checkpoint_plan = MixtureOfParametersPlan.from_fingerprint(fingerprint)
        serving_plan = plan_from_spec()

        self.assertEqual(checkpoint_plan.tiles_per_expert, serving_plan.tiles_per_expert)
        self.assertEqual(
            checkpoint_plan.routed_tiles_per_layer,
            serving_plan.routed_tiles_per_layer,
        )
        self.assertEqual(
            checkpoint_plan.baseline_active_routed_tiles_per_token,
            serving_plan.initial_active_routed_tiles,
        )

        for expert_id, tile_index in ((0, 0), (17, 9), (383, 23)):
            checkpoint_tile = checkpoint_plan.tile_spec(expert_id, tile_index)
            serving_tile = serving_plan.tile(expert_id, tile_index)

            self.assertEqual(checkpoint_tile.w1_row_bounds, serving_tile.gate_rows)
            self.assertEqual(checkpoint_tile.w2_column_bounds, serving_tile.down_columns)

            packed_up_start, packed_up_end = serving_tile.up_rows
            logical_up_rows = (
                packed_up_start - serving_tile.intermediate_size,
                packed_up_end - serving_tile.intermediate_size,
            )
            self.assertEqual(checkpoint_tile.w3_row_bounds, logical_up_rows)

    def test_exact_route_expands_all_tiles_of_selected_experts(self) -> None:
        fingerprint = DeepSeekV4Fingerprint.from_mapping(CANONICAL_CONFIG)
        plan = MixtureOfParametersPlan.from_fingerprint(fingerprint)
        route = plan.exact_tile_route((0, 1, 2, 3, 4, 5))

        self.assertEqual(len(route), 144)
        self.assertEqual(route[:24], tuple(range(24)))
        self.assertEqual(route[24:48], tuple(range(24, 48)))

    def test_partial_or_wrong_architecture_is_rejected(self) -> None:
        wrong = dict(CANONICAL_CONFIG)
        wrong["num_hidden_layers"] = 1
        with self.assertRaises(ValueError):
            DeepSeekV4Fingerprint.from_mapping(wrong)

    def test_config_only_manifest_does_not_claim_weights_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(CANONICAL_CONFIG), encoding="utf-8")
            manifest = build_manifest(root, require_weights=False)
            self.assertFalse(manifest.full_checkpoint_verified)
            self.assertEqual(manifest.model_id, "deepseek-v4-pro-base")
            self.assertIn("tensor_tile_contract", manifest.as_dict())

    def test_full_checkpoint_requires_all_64_shards_and_index_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight_map = canonical_weight_map()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}),
                encoding="utf-8",
            )

            for index in range(1, 64):
                (root / f"model-{index:05d}-of-00064.safetensors").touch()
            with self.assertRaises(FileNotFoundError):
                verify_full_checkpoint_files(root)

            (root / "model-00064-of-00064.safetensors").touch()
            verify_full_checkpoint_files(root)

            corrupted_map = dict(weight_map)
            corrupted_map.pop("model.fake_parameter_64")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": corrupted_map}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                verify_full_checkpoint_files(root)


if __name__ == "__main__":
    unittest.main()
