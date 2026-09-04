from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from leviathan.deepseek_v4_mop import (
    architecture_from_spec,
    build_transplant_manifest,
    plan_from_spec,
    validate_deepseek_v4_config,
)


class DeepSeekV4MoPTests(unittest.TestCase):
    def test_full_v4_contract(self) -> None:
        architecture = architecture_from_spec()
        self.assertEqual(architecture.model_type, "deepseek_v4")
        self.assertEqual(architecture.num_hidden_layers, 61)
        self.assertEqual(architecture.n_routed_experts, 384)
        self.assertEqual(architecture.num_experts_per_tok, 6)
        self.assertEqual(architecture.moe_intermediate_size, 3072)
        self.assertEqual(architecture.weight_block_size, (128, 128))

    def test_mop0_tile_counts_and_coverage(self) -> None:
        plan = plan_from_spec()
        self.assertEqual(plan.tiles_per_expert, 24)
        self.assertEqual(plan.routed_tiles_per_layer, 9216)
        self.assertEqual(plan.initial_active_routed_tiles, 144)
        self.assertAlmostEqual(plan.routed_tile_fraction_at_mop0, 6 / 384)
        self.assertTrue(plan.tile_matches_quantization_block)

        first = plan.tile(0, 0)
        last = plan.tile(383, 23)
        self.assertEqual(first.gate_rows, (0, 128))
        self.assertEqual(first.up_rows, (3072, 3200))
        self.assertEqual(first.down_columns, (0, 128))
        self.assertEqual(last.gate_rows, (2944, 3072))
        self.assertEqual(last.up_rows, (6016, 6144))
        self.assertEqual(last.down_columns, (2944, 3072))

    def test_original_six_expert_route_expands_exactly(self) -> None:
        plan = plan_from_spec()
        tiles = plan.expand_expert_route([1, 2, 3, 4, 5, 6])
        self.assertEqual(len(tiles), 144)
        self.assertEqual({tile.expert_index for tile in tiles}, {1, 2, 3, 4, 5, 6})
        for expert_index in range(1, 7):
            expert_tiles = [tile for tile in tiles if tile.expert_index == expert_index]
            self.assertEqual(sum(tile.width for tile in expert_tiles), 3072)

    def test_config_validation_rejects_drift(self) -> None:
        architecture = architecture_from_spec()
        config = {
            "architectures": [architecture.architecture_class],
            "model_type": architecture.model_type,
            "hidden_size": architecture.hidden_size,
            "num_hidden_layers": architecture.num_hidden_layers,
            "moe_intermediate_size": architecture.moe_intermediate_size,
            "n_routed_experts": architecture.n_routed_experts,
            "n_shared_experts": architecture.n_shared_experts,
            "num_experts_per_tok": architecture.num_experts_per_tok,
            "num_attention_heads": architecture.num_attention_heads,
            "num_key_value_heads": architecture.num_key_value_heads,
            "head_dim": architecture.head_dim,
            "max_position_embeddings": architecture.max_position_embeddings,
            "num_nextn_predict_layers": architecture.num_nextn_predict_layers,
            "hc_mult": architecture.hc_mult,
            "vocab_size": architecture.vocab_size,
            "expert_dtype": architecture.expert_dtype,
            "quantization_config": {
                "weight_block_size": list(architecture.weight_block_size)
            },
        }
        self.assertEqual(validate_deepseek_v4_config(config), ())
        config["n_routed_experts"] = 128
        errors = validate_deepseek_v4_config(config)
        self.assertTrue(any("n_routed_experts" in error for error in errors))

    def test_manifest_requires_immutable_revision(self) -> None:
        architecture = architecture_from_spec()
        config = {
            "architectures": [architecture.architecture_class],
            "model_type": architecture.model_type,
            "hidden_size": architecture.hidden_size,
            "num_hidden_layers": architecture.num_hidden_layers,
            "moe_intermediate_size": architecture.moe_intermediate_size,
            "n_routed_experts": architecture.n_routed_experts,
            "n_shared_experts": architecture.n_shared_experts,
            "num_experts_per_tok": architecture.num_experts_per_tok,
            "num_attention_heads": architecture.num_attention_heads,
            "num_key_value_heads": architecture.num_key_value_heads,
            "head_dim": architecture.head_dim,
            "max_position_embeddings": architecture.max_position_embeddings,
            "num_nextn_predict_layers": architecture.num_nextn_predict_layers,
            "hc_mult": architecture.hc_mult,
            "vocab_size": architecture.vocab_size,
            "expert_dtype": architecture.expert_dtype,
            "quantization_config": {
                "weight_block_size": list(architecture.weight_block_size)
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_transplant_manifest(path, revision="main")
            manifest = build_transplant_manifest(path, revision="a" * 40)
            self.assertTrue(manifest["mop"]["function_preserving_mop0"])


if __name__ == "__main__":
    unittest.main()
