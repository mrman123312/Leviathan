from __future__ import annotations

import unittest

from leviathan.cells import (
    CellContext,
    CellProposal,
    CognitiveCandidate,
    DeliberationStatus,
    EcologyConfig,
    MetaSnapshot,
    ParameterEcology,
    ScriptedCell,
)
from leviathan.types import CognitiveMode, MetaState


def meta_state() -> MetaState:
    return MetaState(
        task_type="test",
        goal="solve",
        success_probability=0.4,
        epistemic_uncertainty=0.5,
        aleatoric_uncertainty=0.1,
        stakes=0.2,
        risk_budget=0.2,
        compute_budget=1.0,
        latency_budget=1.0,
    )


def context() -> CellContext:
    state = meta_state()
    return CellContext(
        agent_id="leviathan",
        goal_id="goal-1",
        meta=MetaSnapshot.from_state(state, immutable_goal=state.goal),
        observation_id="observation-1",
        observation="input",
        routing_keys=frozenset({"novel", "causal"}),
        allowed_modes=frozenset({CognitiveMode.REASON}),
    )


def candidate(candidate_id: str) -> CognitiveCandidate:
    return CognitiveCandidate(
        id=candidate_id,
        mode=CognitiveMode.REASON,
        payload=candidate_id,
    )


class BrokenCell:
    cell_id = "broken"
    reliability = 0.9

    def activation(self, context: CellContext) -> float:
        return 0.95

    def propose(self, context: CellContext) -> CellProposal:
        raise RuntimeError("experimental backend failed")

    def revise(self, context: CellContext, consensus: object) -> CellProposal:
        raise RuntimeError("experimental backend failed")


class ZeroActivationCell(ScriptedCell):
    def activation(self, context: CellContext) -> float:
        return 0.0


class ParameterEcologyTests(unittest.TestCase):
    def test_agreement_converges(self) -> None:
        answer = candidate("answer")
        ecology = ParameterEcology(
            [
                ScriptedCell("semantic", answer, confidence=0.9),
                ScriptedCell("causal", answer, confidence=0.8),
                ScriptedCell("language", answer, confidence=0.85),
            ]
        )

        trace = ecology.deliberate(context())

        self.assertEqual(trace.status, DeliberationStatus.CONVERGED)
        self.assertEqual(trace.decision, answer)
        self.assertEqual(trace.rounds, 1)
        self.assertEqual(set(trace.coalition_cell_ids), {"semantic", "causal", "language"})

    def test_disagreement_recruits_requested_cells(self) -> None:
        first = candidate("first")
        second = candidate("second")
        cells = [
            ScriptedCell(
                "a_initial",
                first,
                keys=frozenset({"novel"}),
                confidence=0.9,
                request_cell_ids=("z_resolver",),
            ),
            ScriptedCell(
                "b_initial",
                second,
                keys=frozenset({"novel"}),
                confidence=0.9,
            ),
            ScriptedCell(
                "z_resolver",
                first,
                keys=frozenset({"novel"}),
                confidence=0.95,
            ),
            ScriptedCell(
                "z_support",
                first,
                keys=frozenset({"novel"}),
                confidence=0.95,
            ),
        ]
        ecology = ParameterEcology(
            cells,
            config=EcologyConfig(
                initial_cells=2,
                max_active_cells=4,
                max_rounds=3,
                max_cell_calls=10,
                consensus_threshold=0.60,
                disagreement_threshold=0.85,
            ),
        )

        trace = ecology.deliberate(context())

        self.assertEqual(trace.status, DeliberationStatus.CONVERGED)
        self.assertEqual(trace.decision, first)
        self.assertEqual(trace.rounds, 2)
        self.assertIn("z_resolver", trace.active_cell_ids)

    def test_one_cell_failure_does_not_destroy_the_market(self) -> None:
        answer = candidate("answer")
        ecology = ParameterEcology(
            [
                BrokenCell(),
                ScriptedCell("good-1", answer, confidence=0.9),
                ScriptedCell("good-2", answer, confidence=0.9),
            ],
            config=EcologyConfig(initial_cells=3),
        )

        trace = ecology.deliberate(context())

        self.assertEqual(trace.status, DeliberationStatus.CONVERGED)
        self.assertEqual(trace.decision, answer)
        self.assertEqual(trace.failures[0].cell_id, "broken")

    def test_hard_call_budget_stops_disagreement(self) -> None:
        ecology = ParameterEcology(
            [
                ScriptedCell("one", candidate("one"), confidence=0.9),
                ScriptedCell("two", candidate("two"), confidence=0.9),
            ],
            config=EcologyConfig(
                initial_cells=2,
                max_active_cells=2,
                max_rounds=5,
                max_cell_calls=2,
            ),
        )

        trace = ecology.deliberate(context())

        self.assertEqual(trace.status, DeliberationStatus.BUDGET_EXHAUSTED)
        self.assertEqual(trace.cell_calls, 2)

    def test_zero_activation_cannot_create_false_consensus(self) -> None:
        answer = candidate("answer")
        ecology = ParameterEcology(
            [
                ZeroActivationCell("one", answer, confidence=0.99),
                ZeroActivationCell("two", answer, confidence=0.99),
            ],
            config=EcologyConfig(
                initial_cells=2,
                max_active_cells=2,
                max_rounds=2,
                max_cell_calls=2,
            ),
        )

        trace = ecology.deliberate(context())

        self.assertEqual(trace.status, DeliberationStatus.BUDGET_EXHAUSTED)
        self.assertEqual(trace.confidence, 0.0)

    def test_candidate_id_collision_fails_closed(self) -> None:
        ecology = ParameterEcology(
            [
                ScriptedCell("one", candidate("same"), confidence=0.9),
                ScriptedCell(
                    "two",
                    CognitiveCandidate(
                        id="same",
                        mode=CognitiveMode.REASON,
                        payload="different transition",
                    ),
                    confidence=0.9,
                ),
            ],
            config=EcologyConfig(initial_cells=2, max_active_cells=2),
        )

        trace = ecology.deliberate(context())

        self.assertEqual(trace.status, DeliberationStatus.NO_PROPOSAL)
        self.assertIsNone(trace.decision)
        self.assertTrue(
            all(failure.error_type == "CandidateIdCollision" for failure in trace.failures)
        )

    def test_only_repeated_verified_success_compiles_a_coalition(self) -> None:
        answer = candidate("answer")
        ecology = ParameterEcology(
            [
                ScriptedCell("one", answer, confidence=0.9),
                ScriptedCell("two", answer, confidence=0.9),
                ScriptedCell("three", answer, confidence=0.9),
            ]
        )
        ctx = context()
        trace = ecology.deliberate(ctx)

        ecology.record_verified_outcome(
            routing_keys=ctx.routing_keys,
            trace=trace,
            passed=True,
            trust=0.9,
        )
        self.assertEqual(ecology.compiled_coalition(ctx.routing_keys), ())
        ecology.record_verified_outcome(
            routing_keys=ctx.routing_keys,
            trace=trace,
            passed=True,
            trust=0.9,
        )

        self.assertEqual(
            set(ecology.compiled_coalition(ctx.routing_keys)),
            {"one", "two", "three"},
        )


if __name__ == "__main__":
    unittest.main()
