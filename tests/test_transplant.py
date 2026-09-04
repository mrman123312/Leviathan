from __future__ import annotations

import unittest

from leviathan.transplant import EvaluationGate, TransplantPhase, TransplantRun


class TransplantRunTests(unittest.TestCase):
    def test_new_path_cannot_activate_before_gate_warmup(self) -> None:
        run = TransplantRun(substrate_id="deepseek-v4-pro-base")
        run.advance()  # insert inert modules
        run.advance()  # train new params only
        with self.assertRaises(RuntimeError):
            run.set_gate(0.01)
        run.advance()  # gate warmup
        run.set_gate(0.01)
        self.assertEqual(run.new_module_gate, 0.01)

    def test_core_cannot_unfreeze_too_early(self) -> None:
        run = TransplantRun(substrate_id="deepseek-v4-pro-base")
        with self.assertRaises(RuntimeError):
            run.permit_selective_unfreeze()

    def test_promotion_requires_every_gate_and_rollback(self) -> None:
        run = TransplantRun(substrate_id="deepseek-v4-pro-base")
        while run.phase is not TransplantPhase.SHADOW:
            if run.phase is TransplantPhase.SELECTIVE_UNFREEZE:
                run.permit_selective_unfreeze()
            run.advance()

        with self.assertRaises(RuntimeError):
            run.advance()

        run.evaluation = EvaluationGate(
            capability_pass=True,
            retention_pass=True,
            calibration_pass=True,
            safety_pass=True,
            adversarial_pass=True,
            efficiency_pass=True,
            rollback_verified=True,
        )
        run.rollback_artifact = "checkpoint://previous"
        self.assertEqual(run.advance(), TransplantPhase.PROMOTED)


if __name__ == "__main__":
    unittest.main()
