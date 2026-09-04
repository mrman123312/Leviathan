from __future__ import annotations

import unittest

from leviathan.kernel import CognitiveCandidate, CognitiveContext, InferenceStatus, MetaSnapshot
from leviathan.mop import MoPConfig, UnifiedMoP
from leviathan.mop_kernel import VectorMoPKernel, VectorObservation
from leviathan.types import CognitiveMode, MetaState


def cognitive_context(observation: object) -> CognitiveContext:
    state = MetaState(
        task_type="vector",
        goal="choose",
        success_probability=0.5,
        epistemic_uncertainty=0.5,
        aleatoric_uncertainty=0.0,
        stakes=0.1,
        risk_budget=0.2,
        compute_budget=1.0,
        latency_budget=1.0,
    )
    return CognitiveContext(
        agent_id="leviathan",
        goal_id="goal",
        meta=MetaSnapshot.from_state(state, immutable_goal="choose"),
        observation_id="observation",
        observation=observation,
        allowed_modes=frozenset(CognitiveMode),
    )


def model() -> UnifiedMoP:
    return UnifiedMoP(
        MoPConfig(
            input_dim=3,
            context_dim=2,
            output_dim=2,
            basis_count=2,
            rank=1,
            seed=3,
        )
    )


class VectorMoPKernelTests(unittest.TestCase):
    def test_one_model_output_decodes_to_one_candidate(self) -> None:
        unified = model()
        state = unified.state_dict()
        state["base_weight"][...] = 0.0
        state["base_bias"][...] = (-4.0, 4.0)
        unified.load_state_dict(state)
        candidates = (
            CognitiveCandidate("wait", CognitiveMode.WAIT_OBSERVE, "wait"),
            CognitiveCandidate("reason", CognitiveMode.REASON, "answer"),
        )
        kernel = VectorMoPKernel(unified, candidates, active_bases=1)

        trace = kernel.infer(cognitive_context(VectorObservation((1.0, 0.0, -1.0), (1.0, 0.0))))

        self.assertEqual(trace.status, InferenceStatus.DECIDED)
        self.assertEqual(trace.decision, candidates[1])
        self.assertEqual(trace.forward_passes, 1)
        self.assertEqual(trace.total_parameters, unified.parameter_count)
        self.assertEqual(trace.active_parameters, unified.active_parameter_count(1))

    def test_low_confidence_returns_no_decision(self) -> None:
        unified = model()
        state = unified.state_dict()
        state["base_weight"][...] = 0.0
        state["base_bias"][...] = 0.0
        unified.load_state_dict(state)
        candidates = (
            CognitiveCandidate("wait", CognitiveMode.WAIT_OBSERVE, "wait"),
            CognitiveCandidate("reason", CognitiveMode.REASON, "answer"),
        )
        kernel = VectorMoPKernel(
            unified,
            candidates,
            active_bases=1,
            confidence_threshold=0.8,
        )

        trace = kernel.infer(cognitive_context(VectorObservation((0.0, 0.0, 0.0), (1.0, 0.0))))

        self.assertEqual(trace.status, InferenceStatus.NO_DECISION)
        self.assertIsNone(trace.decision)


if __name__ == "__main__":
    unittest.main()
