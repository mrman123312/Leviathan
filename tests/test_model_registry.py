from __future__ import annotations
from pathlib import Path
import tempfile
import textwrap
import unittest
from leviathan.model_registry import ModelRegistry

class ModelRegistryTests(unittest.TestCase):
    def test_project_registry_loads(self):
        registry = ModelRegistry.from_toml()
        ids = {model.id for model in registry.all()}
        self.assertIn("qwen3-30b-a3b-base", ids)
        self.assertIn("deepseek-v4-pro-base", ids)
        self.assertGreaterEqual(len(registry.base_models()), 5)
        self.assertGreaterEqual(len(registry.teachers()), 4)
    def test_qwen_is_canonical_and_true_base_control_is_retained(self):
        registry = ModelRegistry.from_toml()
        model = registry.canonical_substrate()
        self.assertEqual(model.id, "qwen3.8-27b")
        self.assertEqual(model.role, "canonical_semantic_substrate")
        self.assertFalse(model.is_base)
        self.assertTrue(registry.get("qwen3-1.7b-base").is_base)
        self.assertEqual(len(model.revision), 40)
    def test_frontier_download_requires_explicit_override(self):
        registry = ModelRegistry.from_toml()
        with self.assertRaises(PermissionError):
            registry.require_download_permission("deepseek-v4-pro-base")
        model = registry.require_download_permission("deepseek-v4-pro-base", allow_disabled=True)
        self.assertEqual(model.role, "legacy_substrate")
    def test_deepseek_active_fraction_is_sparse(self):
        model = ModelRegistry.from_toml().get("deepseek-v4-pro-base")
        self.assertIsNotNone(model.active_fraction)
        self.assertLess(model.active_fraction, .04)
    def test_duplicate_ids_rejected(self):
        block = '''[[models]]
id = "same"
repo_id = "a/one"
role = "teacher"
stage = "base"
license = "MIT"
total_parameters_b = 1.0
active_parameters_b = 1.0
multimodal = false
enabled_for_download = true
priority = 1
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.toml"
            path.write_text(block + "\n" + block)
            with self.assertRaises(ValueError):
                ModelRegistry.from_toml(path)

if __name__ == "__main__":
    unittest.main()
