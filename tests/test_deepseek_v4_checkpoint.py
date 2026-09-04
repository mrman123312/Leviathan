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


class DeepSeekV4CheckpointTests(unittest.TestCase):
    def test_canonical_fingerprint_and_mop_route(self) -> None:
        fingerprint = DeepSeekV4Fingerprint.from_mapping(CANONICAL_CONFIG)
        plan = MixtureOfParametersPlan.from_fingerprint(fingerprint, tile_width=128)
        self.assertEqual(plan.tiles_per_expert, 24)
        self.assertEqual(plan.routed_tiles_per_layer, 9216)
        self.assertEqual(plan.baseline_active_routed_tiles_per_token, 144)
        route = plan.exact_tile_route([0, 1, 2, 3, 4, 5])
        self.assertEqual(len(route), 144)
        self.assertEqual(route[:24], tuple(range(24)))
        self.assertEqual(route[-24:], tuple(range(120, 144)))

    def test_noncanonical_full_model_is_rejected(self) -> None:
        bad = dict(CANONICAL_CONFIG)
        bad["num_hidden_layers"] = 60
        with self.assertRaises(ValueError):
            DeepSeekV4Fingerprint.from_mapping(bad)

    def test_full_checkpoint_requires_all_64_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                verify_full_checkpoint_files(model_dir)

            for index in range(1, 65):
                (model_dir / f"model-{index:05d}-of-00064.safetensors").touch()
            verify_full_checkpoint_files(model_dir)

    def test_config_only_manifest_does_not_claim_weights_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(CANONICAL_CONFIG), encoding="utf-8"
            )
            manifest = build_manifest(model_dir, require_weights=False)
            self.assertFalse(manifest.full_checkpoint_verified)
            self.assertEqual(manifest.model_id, "deepseek-v4-pro-base")


if __name__ == "__main__":
    unittest.main()
