from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from leviathan.model_registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def test_project_registry_loads(self) -> None:
        registry = ModelRegistry.from_toml()
        ids = {model.id for model in registry.all()}
        self.assertIn("qwen3-30b-a3b-base", ids)
        self.assertIn("deepseek-v4-pro-base", ids)
        self.assertGreaterEqual(len(registry.base_models()), 5)
        self.assertGreaterEqual(len(registry.teachers()), 4)

    def test_frontier_download_requires_explicit_override(self) -> None:
        registry = ModelRegistry.from_toml()
        with self.assertRaises(PermissionError):
            registry.require_download_permission("deepseek-v4-pro-base")
        model = registry.require_download_permission(
            "deepseek-v4-pro-base", allow_disabled=True
        )
        self.assertEqual(model.role, "frontier_semantic_substrate")

    def test_active_fraction(self) -> None:
        registry = ModelRegistry.from_toml()
        model = registry.get("qwen3-30b-a3b-base")
        self.assertIsNotNone(model.active_fraction)
        assert model.active_fraction is not None
        self.assertLess(model.active_fraction, 0.2)

    def test_duplicate_ids_rejected(self) -> None:
        content = textwrap.dedent(
            """
            [[models]]
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

            [[models]]
            id = "same"
            repo_id = "a/two"
            role = "teacher"
            stage = "base"
            license = "MIT"
            total_parameters_b = 1.0
            active_parameters_b = 1.0
            multimodal = false
            enabled_for_download = true
            priority = 2
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.toml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                ModelRegistry.from_toml(path)


if __name__ == "__main__":
    unittest.main()
