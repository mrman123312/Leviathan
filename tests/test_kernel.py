from __future__ import annotations

import unittest

from leviathan.kernel import (
    CognitiveCandidate,
    CognitiveContext,
    InferenceStatus,
    InferenceTrace,
    KernelManifest,
    MetaSnapshot,
    ScriptedKernel,
)
from leviathan.types import CognitiveMode, MetaState


def context(*, allowed: frozenset[CognitiveMode] | None = None) -> CognitiveContext:
    state = MetaState(
        task_type="unit-test",
        goal="preserve",
        success_probability=0.5,
        epistemic_uncertainty=0.5,
        aleatoric_uncertainty=0.1,
        stakes=0.1,
        risk_budget=0.2,
        compute_budget=1.0,
        latency_budget=1.0,
    )
    return CognitiveContext(
        agent_id="leviathan",
        goal_id="goal",
        meta=MetaSnapshot.from_state(state, immutable_goal="preserve"),
        observation_id="observation",
        observation="input",
        refinement_budget=3,
        allowed_modes=allowed or frozenset(CognitiveMode),
    )


class KernelContractTests(unittest.TestCase):
    def test_manifest_reports_every_non_single_count(self) -> None:
        violations = KernelManifest(
            parameter_owners=2,
            independent_internal_models=1,
        ).violations()

        self.assertIn("parameter_owners=2 (required 1)", violations)
        self.assertIn("independent_internal_models=1 (required 0)", violations)

    def test_scripted_fixture_is_one_call_and_one_decision(self) -> None:
        candidate = CognitiveCandidate("answer", CognitiveMode.REASON, "result")
        trace = ScriptedKernel(candidate).infer(context())

        self.assertEqual(trace.status, InferenceStatus.DECIDED)
        self.assertEqual(trace.decision, candidate)
        self.assertEqual(trace.forward_passes, 1)

    def test_disallowed_mode_becomes_no_decision(self) -> None:
        candidate = CognitiveCandidate("act", CognitiveMode.ACT, "effect")
        trace = ScriptedKernel(candidate).infer(context(allowed=frozenset({CognitiveMode.REASON})))

        self.assertEqual(trace.status, InferenceStatus.NO_DECISION)
        self.assertIsNone(trace.decision)

    def test_decided_trace_requires_exactly_one_decision(self) -> None:
        with self.assertRaises(ValueError):
            InferenceTrace(
                status=InferenceStatus.DECIDED,
                decision=None,
                confidence=0.5,
                uncertainty=0.5,
                refinement_steps=1,
                forward_passes=1,
            )


if __name__ == "__main__":
    unittest.main()
