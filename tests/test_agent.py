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
from leviathan.cells import CognitiveCandidate, EcologyConfig, ParameterEcology, ScriptedCell
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


def ecology_for(candidate: CognitiveCandidate) -> ParameterEcology:
    return ParameterEcology(
        [
            ScriptedCell("causal", candidate, confidence=0.9),
            ScriptedCell("semantic", candidate, confidence=0.9),
            ScriptedCell("planner", candidate, confidence=0.9),
        ]
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


class LeviathanAgentTests(unittest.TestCase):
    def test_internal_cells_produce_one_agent_decision(self) -> None:
        answer = CognitiveCandidate(
            id="answer",
            mode=CognitiveMode.REASON,
            payload="one coherent result",
        )
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            ecology=ecology_for(answer),
        )

        result = agent.step(observation(), meta(goal="attempted replacement"))

        self.assertEqual(result.status, AgentStatus.DECIDED)
        self.assertEqual(result.candidate, answer)
        self.assertEqual(agent.snapshot.agent_id, "leviathan-1")
        self.assertEqual(agent.snapshot.goal.original, "diagnose")
        self.assertIn("goal_restored", [event.kind for event in agent.events])

    def test_nonconverged_market_never_falls_through_to_action(self) -> None:
        executor = RecordingExecutor()
        ecology = ParameterEcology(
            [
                ScriptedCell("one", action_candidate(), confidence=0.9),
                ScriptedCell(
                    "two",
                    CognitiveCandidate(
                        id="other-action",
                        mode=CognitiveMode.ACT,
                        payload="other",
                        expected_observation="other-result",
                        risk=0.1,
                        authorization_class="sandbox",
                    ),
                    confidence=0.9,
                ),
            ],
            config=EcologyConfig(
                initial_cells=2,
                max_active_cells=2,
                max_rounds=4,
                max_cell_calls=2,
            ),
        )
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            ecology=ecology,
            executor=executor,
            verifier=DeterministicVerifier(),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.NO_DECISION)
        self.assertEqual(executor.contracts, [])

    def test_risk_limit_blocks_action(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose", risk_budget=0.2),
            ecology=ecology_for(action_candidate(risk=0.6)),
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
            ecology=ecology_for(action_candidate()),
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
            ecology=ecology_for(action_candidate()),
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

    def test_same_model_self_evaluation_cannot_promote(self) -> None:
        executor = RecordingExecutor()
        ecology = ecology_for(action_candidate())
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            ecology=ecology,
            executor=executor,
            verifier=DeterministicVerifier(
                kind=ProvenanceKind.SELF_EVALUATION,
                independence=0.1,
            ),
        )

        result = agent.step(observation(), meta())

        self.assertEqual(result.status, AgentStatus.UNVERIFIED)
        self.assertEqual(ecology.compiled_coalition(frozenset({"diagnostic"})), ())

    def test_repeated_external_success_compiles_the_internal_coalition(self) -> None:
        executor = RecordingExecutor()
        ecology = ecology_for(action_candidate())
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            ecology=ecology,
            executor=executor,
            verifier=DeterministicVerifier(),
        )

        first = agent.step(observation("obs-1"), meta())
        second = agent.step(observation("obs-2"), meta())

        self.assertEqual(first.status, AgentStatus.VERIFIED)
        self.assertEqual(second.status, AgentStatus.VERIFIED)
        self.assertEqual(
            set(ecology.compiled_coalition(frozenset({"diagnostic"}))),
            {"causal", "semantic", "planner"},
        )

    def test_negative_independent_verification_rejects_learning(self) -> None:
        executor = RecordingExecutor()
        agent = LeviathanAgent(
            agent_id="leviathan-1",
            goal=GoalFrame("diagnose"),
            ecology=ecology_for(action_candidate()),
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
