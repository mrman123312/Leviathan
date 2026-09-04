from __future__ import annotations

import unittest

from leviathan.cognitive_kernel import Evidence, GoalState, LearningDestination
from leviathan.parameter_cells import CellAction, CellTelemetrySummary
from leviathan.runtime import LeviathanRuntime
from leviathan.types import Belief, MetaState, Provenance, ProvenanceKind, UncertaintyKind


def state() -> MetaState:
    return MetaState(
        task_type="diagnosis",
        goal="diagnose",
        success_probability=0.5,
        epistemic_uncertainty=0.6,
        aleatoric_uncertainty=0.1,
        stakes=0.3,
        risk_budget=0.4,
        compute_budget=0.5,
        latency_budget=0.5,
        world_model_confidence=0.7,
        branching_factor_estimate=4,
    )


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = LeviathanRuntime(model_id="deepseek-v4-pro-base")
        self.provenance = Provenance(
            kind=ProvenanceKind.TRUSTED_MEASUREMENT,
            source_id="sensor",
            trust_prior=0.9,
        )

    def test_cell_telemetry_enters_metastate_without_creating_new_model(self) -> None:
        telemetry = CellTelemetrySummary(
            active_cell_token_pairs=100,
            unique_cells_seen=128,
            mean_confidence=0.7,
            mean_abstention=0.2,
            mean_disagreement=0.35,
            max_disagreement=0.8,
            recommended_action=CellAction.RECRUIT,
        )
        updated = self.runtime.incorporate_cell_telemetry(
            state(), telemetry, total_routed_cells=9216, max_active_cells=256
        )
        self.assertAlmostEqual(updated.parameter_cell_disagreement, 0.35)
        self.assertAlmostEqual(updated.parameter_cell_budget_pressure, 0.5)
        self.assertEqual(self.runtime.model_id, "deepseek-v4-pro-base")

    def test_task_evidence_memory_learning_and_compilation_share_one_runtime(self) -> None:
        session = self.runtime.begin_task(
            problem="Why did the pump stop?",
            task_type="diagnosis",
            goal=GoalState(objective="identify cause"),
            meta_state=state(),
        )
        self.runtime.add_belief(
            session.id,
            Belief(
                id="b1",
                value="pressure fault",
                confidence=0.5,
                provenance=self.provenance,
                uncertainty=UncertaintyKind.EPISTEMIC,
                evidence_refs=["sensor-0"],
            ),
            reason_ref="sensor-0",
        )
        posterior = self.runtime.apply_evidence(
            session.id,
            Evidence(
                id="sensor-1",
                target_id="b1",
                supports=True,
                strength=0.8,
                independence=0.9,
                provenance=self.provenance,
            ),
        )
        self.assertGreater(posterior, 0.5)
        self.runtime.record_prediction(session.id, "pred-1")
        self.runtime.record_action(session.id, "action-1")
        self.runtime.record_verification(session.id, "verify-1")
        completion = self.runtime.complete_task(
            session.id,
            outcome_ref="outcome-1",
            verified_success=True,
            provenance=self.provenance,
            truth_quality=0.95,
            novelty=0.7,
            transfer_value=0.8,
            independent_verification=True,
            rollback_available=True,
        )
        self.assertEqual(completion.learning_route.destination, LearningDestination.SEMANTIC)
        self.assertEqual(len(self.runtime.memory.records), 1)
        self.assertTrue(self.runtime.sessions[0].closed)
        self.assertGreater(len(self.runtime.kernel.event_log.events), 0)

    def test_predictions_are_forced_before_outcome_and_completion_requires_verification(self) -> None:
        session = self.runtime.begin_task(
            problem="test",
            task_type="general",
            goal=GoalState(objective="solve"),
            meta_state=state(),
        )
        with self.assertRaises(RuntimeError):
            self.runtime.complete_task(
                session.id,
                outcome_ref="outcome",
                verified_success=True,
                provenance=self.provenance,
                truth_quality=0.9,
                novelty=0.5,
                transfer_value=0.5,
                independent_verification=True,
                rollback_available=True,
            )

        self.runtime.record_verification(session.id, "verify")
        self.runtime.complete_task(
            session.id,
            outcome_ref="outcome",
            verified_success=True,
            provenance=self.provenance,
            truth_quality=0.9,
            novelty=0.5,
            transfer_value=0.5,
            independent_verification=True,
            rollback_available=True,
        )
        with self.assertRaises(RuntimeError):
            self.runtime.record_prediction(session.id, "late-prediction")

    def test_repeated_verified_same_program_becomes_skill_candidate(self) -> None:
        ready = False
        for index in range(8):
            session = self.runtime.begin_task(
                problem="same repeatable problem",
                task_type="general",
                goal=GoalState(objective="solve same class"),
                meta_state=state(),
            )
            self.runtime.record_verification(session.id, f"verify-{index}")
            completion = self.runtime.complete_task(
                session.id,
                outcome_ref=f"outcome-{index}",
                verified_success=True,
                provenance=self.provenance,
                truth_quality=0.95,
                novelty=0.6,
                transfer_value=0.9,
                independent_verification=True,
                rollback_available=True,
            )
            ready = completion.skill_ready_to_compile
        self.assertTrue(ready)
        self.assertEqual(completion.learning_route.destination, LearningDestination.PROCEDURAL)


if __name__ == "__main__":
    unittest.main()
