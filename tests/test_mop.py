from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from leviathan.mop import AdamOptimizer, MoPConfig, UnifiedMoP


class UnifiedMoPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MoPConfig(
            input_dim=4,
            context_dim=3,
            output_dim=2,
            basis_count=3,
            rank=2,
            seed=7,
        )
        self.inputs = np.asarray(
            [[0.2, -0.1, 0.4, 0.7], [-0.3, 0.8, 0.1, -0.2]],
            dtype=np.float64,
        )
        self.context = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

    def test_zero_initialized_bank_preserves_base_function_exactly(self) -> None:
        model = UnifiedMoP(self.config)

        base = model.base_prediction(self.inputs)
        mixed = model.forward(self.inputs, self.context).output

        np.testing.assert_array_equal(mixed, base)

    def test_sparse_route_is_normalized_and_activates_exactly_top_k(self) -> None:
        model = UnifiedMoP(self.config)

        gates, selected = model.route(self.context, active_bases=2)

        np.testing.assert_allclose(np.sum(gates, axis=1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(np.count_nonzero(gates, axis=1), np.asarray([2, 2]))
        self.assertEqual(selected.shape, (2, 2))
        self.assertLess(model.active_parameter_count(2), model.parameter_count)

    def test_explicit_gradients_match_finite_differences(self) -> None:
        model = UnifiedMoP(self.config)
        state = model.state_dict()
        rng = np.random.default_rng(11)
        state["basis_up"] = rng.normal(0.0, 0.1, size=state["basis_up"].shape)
        model.load_state_dict(state)
        targets = np.asarray([[0.1, -0.2], [0.4, 0.3]], dtype=np.float64)
        _, gradients = model.loss_and_gradients(self.inputs, self.context, targets)
        epsilon = 1e-6

        for name, index in (
            ("base_weight", (0, 0)),
            ("basis_down", (0, 0, 0)),
            ("basis_up", (1, 0, 1)),
            ("router_weight", (0, 0)),
        ):
            original = model.state_dict()
            plus = {key: value.copy() for key, value in original.items()}
            minus = {key: value.copy() for key, value in original.items()}
            plus[name][index] += epsilon
            minus[name][index] -= epsilon
            model.load_state_dict(plus)
            plus_loss, _ = model.loss_and_gradients(self.inputs, self.context, targets)
            model.load_state_dict(minus)
            minus_loss, _ = model.loss_and_gradients(self.inputs, self.context, targets)
            model.load_state_dict(original)
            numeric = (plus_loss - minus_loss) / (2.0 * epsilon)
            self.assertAlmostEqual(numeric, float(gradients[name][index]), places=6)

    def test_one_optimizer_reduces_loss_for_the_whole_model(self) -> None:
        model = UnifiedMoP(self.config)
        optimizer = AdamOptimizer(model, learning_rate=0.03)
        targets = np.asarray([[0.8, -0.4], [-0.2, 0.9]], dtype=np.float64)
        initial, _ = model.loss_and_gradients(self.inputs, self.context, targets)

        for _ in range(120):
            _, gradients = model.loss_and_gradients(self.inputs, self.context, targets)
            optimizer.step(model, gradients)

        final, _ = model.loss_and_gradients(self.inputs, self.context, targets)
        self.assertLess(final, initial * 1e-3)
        self.assertEqual(optimizer.step_count, 120)

    def test_optimizer_cannot_update_a_second_model(self) -> None:
        first = UnifiedMoP(self.config)
        second = UnifiedMoP(self.config)
        optimizer = AdamOptimizer(first)
        targets = np.asarray([[0.8, -0.4], [-0.2, 0.9]], dtype=np.float64)
        _, gradients = first.loss_and_gradients(self.inputs, self.context, targets)

        with self.assertRaisesRegex(ValueError, "different unified model"):
            optimizer.step(second, gradients)

    def test_one_checkpoint_round_trips_the_complete_function(self) -> None:
        model = UnifiedMoP(self.config)
        optimizer = AdamOptimizer(model)
        targets = np.asarray([[0.5, 0.1], [-0.3, 0.6]], dtype=np.float64)
        for _ in range(3):
            _, gradients = model.loss_and_gradients(self.inputs, self.context, targets)
            optimizer.step(model, gradients)
        expected = model.forward(self.inputs, self.context, active_bases=2).output

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "one-model.npz"
            model.save(checkpoint)
            restored = UnifiedMoP.load(checkpoint)

        actual = restored.forward(self.inputs, self.context, active_bases=2).output
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(restored.parameter_count, model.parameter_count)


if __name__ == "__main__":
    unittest.main()
