"""Run one fully local, deterministic Leviathan agent cycle."""

from __future__ import annotations

import json

from leviathan import (
    ActionContract,
    ActionOutcome,
    AgentObservation,
    CognitiveCandidate,
    CognitiveMode,
    GoalFrame,
    LeviathanAgent,
    MetaState,
    ParameterEcology,
    Provenance,
    ProvenanceKind,
    ScriptedCell,
    Verification,
)


class SandboxExecutor:
    def execute(self, contract: ActionContract) -> ActionOutcome:
        return ActionOutcome(
            contract_id=contract.id,
            result={"rule": "switch A controls lamp B"},
            observation={"lamp_b": "on"},
            success=True,
        )


class DeterministicVerifier:
    verifier_id = "environment-check"

    def verify(
        self,
        contract: ActionContract,
        outcome: ActionOutcome,
    ) -> tuple[Verification, ...]:
        return (
            Verification(
                target_id=contract.id,
                verifier_type="deterministic_environment",
                passed=outcome.observation == {"lamp_b": "on"},
                confidence=1.0,
                independence_score=1.0,
                provenance=Provenance(
                    kind=ProvenanceKind.DETERMINISTIC_EXECUTION,
                    source_id="demo-environment",
                    trust_prior=0.97,
                ),
                verifier_id=self.verifier_id,
                result=outcome.observation,
            ),
        )


def main() -> None:
    candidate = CognitiveCandidate(
        id="toggle-switch-a",
        mode=CognitiveMode.EXPERIMENT,
        payload={"switch": "A", "operation": "toggle"},
        expected_observation={"lamp_b": "on"},
        preconditions=("sandbox_ready",),
        risk=0.05,
        reversible=True,
        authorization_class="sandbox",
        verifier="environment-check",
        information_gain=0.8,
        transfer_value=0.6,
    )
    ecology = ParameterEcology(
        [
            ScriptedCell("causal", candidate, frozenset({"hidden-rule"}), confidence=0.90),
            ScriptedCell("experiment", candidate, frozenset({"hidden-rule"}), confidence=0.88),
            ScriptedCell("planner", candidate, frozenset({"hidden-rule"}), confidence=0.86),
        ]
    )
    goal = GoalFrame(
        "discover which switch controls lamp B",
        success_criteria=("produce a verified causal rule",),
        constraints=("use only the reversible sandbox",),
        risk_budget=0.10,
    )
    agent = LeviathanAgent(
        agent_id="leviathan-demo",
        goal=goal,
        ecology=ecology,
        executor=SandboxExecutor(),
        verifier=DeterministicVerifier(),
    )
    observation = AgentObservation(
        id="initial-state",
        payload={"switch_a": "off", "lamp_b": "off"},
        provenance=Provenance(
            kind=ProvenanceKind.REAL_OBSERVATION,
            source_id="demo-environment",
            trust_prior=0.95,
        ),
        routing_keys=frozenset({"hidden-rule", "causal"}),
        satisfied_preconditions=frozenset({"sandbox_ready"}),
    )
    meta = MetaState(
        task_type="novel_environment",
        goal=goal.original,
        success_probability=0.3,
        epistemic_uncertainty=0.8,
        aleatoric_uncertainty=0.0,
        stakes=0.05,
        risk_budget=0.10,
        compute_budget=1.0,
        latency_budget=1.0,
        available_verifiers=("environment-check",),
        expected_information_gain=0.8,
    )

    result = agent.step(observation, meta)
    print(
        json.dumps(
            {
                "agent_id": agent.snapshot.agent_id,
                "status": result.status.value,
                "candidate": result.candidate.id if result.candidate else None,
                "rounds": result.trace.rounds,
                "active_cells": result.trace.active_cell_ids,
                "goal_id": agent.snapshot.goal.id,
                "event_order": [event.kind for event in agent.events],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
