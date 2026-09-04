from __future__ import annotations

import unittest

from leviathan.architecture_maturity import GateState, GATE_ORDER, load_maturity_plan


class ArchitectureMaturityTests(unittest.TestCase):
    def test_plan_contains_full_stack_and_five_gates(self) -> None:
        plan = load_maturity_plan()
        self.assertEqual(plan.required_gates, GATE_ORDER)
        ids = {layer.layer_id for layer in plan.layers}
        self.assertEqual(
            ids,
            {"L0", "L1", "L1.5", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"},
        )

    def test_build_order_prioritizes_parameter_ecology_then_cognition(self) -> None:
        plan = load_maturity_plan()
        self.assertEqual(plan.build_order[:6], ("L1", "L1.5", "L2", "L5", "L8", "L6"))
        self.assertEqual(plan.build_order[-1], "L10")

    def test_no_layer_is_falsely_marked_demonstrated(self) -> None:
        plan = load_maturity_plan()
        self.assertEqual(plan.demonstrated_layers, ())
        self.assertFalse(plan.layer("L1.5").fully_demonstrated)
        self.assertEqual(plan.layer("L1.5").gates["learned"], GateState.NOT_STARTED)

    def test_score_is_descriptive_and_partial_aware(self) -> None:
        plan = load_maturity_plan()
        self.assertEqual(plan.layer("L1").score, 2.5)
        self.assertEqual(plan.layer("L1.5").score, 2.5)
        self.assertGreater(plan.layer("L8").score, plan.layer("L6").score)


if __name__ == "__main__":
    unittest.main()
