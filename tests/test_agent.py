from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from leviathan.agent import (
    ActionContract,
    ActionOutcome,
    AgentObservation,
    AgentPolicy,
    AgentStatus,
    GoalFrame,
    LeviathanAgent,
)
from leviathan.kernel import (
    CognitiveCandidate,
    CognitiveContext,
    InferenceStatus,
    InferenceTrace,
    KernelManifest,
    ScriptedKernel,
)
from leviathan.types import CognitiveMode, MetaState, Provenance, ProvenanceKind, Verification


def provenance(kind: ProvenanceKind = ProvenanceKind.REAL_OBSERVATION) -> Provenance:
    return Provenance(kind=kind, source_id="test", trust_prior=0.9)


def observation(observation_id: str = "obs-1") -> AgentObservation:
    return AgentObservation(
        id=observation_id,
        payload={"state": "ready"},
        provenance=provenance(),
        routing_keys=frozenset({"diagnostic"}),
        evidence_refs=("evidence-1",),
        satisfied_preconditions=frozenset({"sandbox_ready"}),
    )


def meta(goal: str = "diagnose") -> MetaState:
    return MetaState(
        task_type="diagnostic",
        goal=goal,
        success_probability=0.6,
        epistemic_uncertainty=0.5,
        aleatoric_uncertainty=0.1,
        stakes=0.2,
        risk_budget=0.25,
        compute_budget=1.0,
        latency_budget=1.0,
        available_verifiers=("deterministic",),
        available_tools=("sandbox",),
    )


def action_candidate(*, risk: float = 0.1, reversible: bool = True) -> CognitiveCandidate:
    return CognitiveCandidate(
        id="inspect-sandbox",
        mode=CognitiveMode.ACT,
        payload={"operation": "inspect"},
        expected_observation={"state": "inspected"},
        preconditions=("sandbox_ready",),
        risk=risk,
        reversible=reversible,
        authorization_class="sandbox",
        verifier="deterministic",
    )


class RecordingExecutor:
    def __init__(self) -> None:
        self.contracts: list[ActionContract] = []

    def execute(self, contract: ActionContract) -> ActionOutcome:
        self.contracts.append(contract)
        return ActionOutcome(
            contract_id=contract.id,
            result="inspected",
            observation={"state": "inspected"},
            success=True,
        )


class DeterministicVerifier:
    verifier_id = "deterministic"

    def __init__(
        self,
        *,
        passed: bool = True,
        kind: ProvenanceKind = ProvenanceKind.DETERMINISTIC_EXECUTION,
        independence: float = 1.0,
    ) -> None:
        self.passed = passed
        self.kind = kind
        self.independence = independence

    def verify(
        self,
        contract: ActionContract,
        outcome: ActionOutcome,
    ) -> tuple[Verification, ...]:
        return (
            Verification(
                target_id=contract.id,
                verifier_type="deterministic",
                passed=self.passed,
                confidence=1.0,
                independence_score=self.independence,
                provenance=provenance(self.kind),
                verifier_id=self.verifier_id,
                result=outcome.result,
            ),
        )


class OtherVerifier(DeterministicVerifier):
    verifier_id = "other"


class CountingKernel:
    model_id = "counting-single-kernel"
    manifest = KernelManifest()

    def __init__(self, candidate: CognitiveCandidate) -> None:
        self.candidate = candidate
        self.calls = 0

    def infer(self, context: CognitiveContext) -> InferenceTrace:
        self.calls += 1
        return InferenceTrace(
            status=InferenceStatus.DECIDED,
            decision=self.candidate,
            confidence=0.9,
            uncertainty=0.1,
            refinement_steps=1,
            forward_passes=1,
            active_parameters=80,
            total_parameters=100,
        )


class ExhaustedKernel:
    model_id = "exhausted-single-kernel"
    manifest = KernelManifest()

    def infer(self, context: CognitiveContext) -> InferenceTrace:
        return InferenceTrace(
            status=InferenceStatus.BUDGET_EXHAUSTED,
            decision=None,
            confidence=0.0,
            uncertainty=1.0,
            refinement_steps=context.refinement_budget,
            forward_passes=context.refinement_budget,
            reason="refinement budget exhausted",
        )


class FailingKernel:
    model_id = "failing-single-kernel"
    manifest = KernelManifest()

    def infer(self, context: CognitiveContext) -> InferenceTrace:
        raise RuntimeError("model failure")


class LeviathanAgentTests(unittest.TestCase):
    def test_kernel_manifest_rejects_a_hidden_model_population(self) -> None:
        class PopulationKernel:
            model_id = "hidden-population"
            manifest = KernelManifest(
                parameter_owners=5,
                shared_states=5,
                optimizers=5,
                checkpoints=5,
                independent_internal_models=5,
            )

            def infer(self, context: CognitiveContext) -> InferenceTrace:
                raise AssertionError("the invalid kernel must never run")

        with self.assertRaisesRegex(ValueError, "single-model contract"):
            LeviathanAgent(
                agent_id="leviathan-1",
                goal=GoalFrame("diagnose"),
                kernel=PopulationKernel(),
            )

    def test_one_kernel_call_produces_one_agent_decision(self) -> None:
        answer = CognitiveCandidate(
            id="answer",
            mode=CognitiveMode.REASON,
            payload="one coherent result",
        )
        kernel = CountingKernel(answer)
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=kernel,
        )

        result = agent.step(observation(), meta(goal="attempted replacement"))

        self.assertEqual(result.status, AgentStatus.DECIDED)
        self.assertEqual(result.candidate, answer)
        self.assertEqual(kernel.calls, 1)
        self.assertEqual(agent.snapshot.agent_id, "leviathan-1")
        self.assertEqual(agent.snapshot.model_id, kernel.model_id)
        self.assertEqual(agent.snapshot.kernel_manifest.independent_internal_models, 0)
        self.assertEqual(agent.snapshot.goal.original, "diagnose")
        self.assertIn("goal_restored", [event.kind for event in agent.events])

    def test_budget_exhaustion_never_falls_through_to_action(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=ExhaustedKernel(),
            executor=executor,
            verifier=DeterministicVerifier(),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.NO_DECISION)
        self.assertEqual(executor.contracts, [])
        self.assertIn("budget", result.reason or "")

    def test_model_failure_is_explicit_and_cannot_act(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=FailingKernel(),
            executor=executor,
            verifier=DeterministicVerifier(),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(executor.contracts, [])
        self.assertIn("kernel failed", result.reason or "")

    def test_risk_limit_blocks_action(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose", risk_budget=0.2),
            kernel=ScriptedKernel(action_candidate(risk=0.6)),
            executor=executor,
            verifier=DeterministicVerifier(),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.BLOCKED)
        self.assertIn("risk", result.reason or "")
        self.assertEqual(executor.contracts, [])

    def test_required_verifier_must_match_connected_port(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=ScriptedKernel(action_candidate()),
            executor=executor,
            verifier=OtherVerifier(),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.BLOCKED)
        self.assertIn("does not match", result.reason or "")
        self.assertEqual(executor.contracts, [])

    def test_prediction_precedes_verified_action(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=ScriptedKernel(action_candidate()),
            executor=executor,
            verifier=DeterministicVerifier(),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.VERIFIED)
        self.assertIsNotNone(result.contract)
        assert result.contract is not None
        self.assertEqual(result.contract.goal_id, agent.snapshot.goal.id)
        kinds = [event.kind for event in agent.events]
        self.assertLess(kinds.index("prediction_recorded"), kinds.index("action_executed"))
        self.assertLess(kinds.index("action_executed"), kinds.index("verification_recorded"))

    def test_same_model_self_evaluation_cannot_create_learning_evidence(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=ScriptedKernel(action_candidate()),
            executor=executor,
            verifier=DeterministicVerifier(
                kind=ProvenanceKind.SELF_EVALUATION,
                independence=0.1,
            ),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.UNVERIFIED)
        self.assertNotIn("verified_learning_evidence", [event.kind for event in agent.events])

    def test_repeated_external_success_records_evidence_without_online_weight_update(self) -> None:
        executor = RecordingExecutor()
        kernel = CountingKernel(action_candidate())
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=kernel,
            executor=executor,
            verifier=DeterministicVerifier(),
        )

        first = agent.step(observation("obs-1"), meta())
        second = agent.step(observation("obs-2"), meta())

        self.assertEqual(first.status, AgentStatus.VERIFIED)
        self.assertEqual(second.status, AgentStatus.VERIFIED)
        self.assertEqual(kernel.calls, 2)
        self.assertEqual(
            [event.kind for event in agent.events].count("verified_learning_evidence"),
            2,
        )

    def test_negative_independent_verification_rejects_learning(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            kernel=ScriptedKernel(action_candidate()),
            executor=executor,
            verifier=DeterministicVerifier(passed=False),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.REJECTED)

    def test_frozen_governance_cannot_be_rewritten(self) -> None:
        policy = AgentPolicy()
        with self.assertRaises(FrozenInstanceError):
            policy.max_cycles = 999  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
