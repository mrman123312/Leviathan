"""Integrated single-model Leviathan cognitive runtime.

This is the reference orchestration boundary connecting the parameter ecology's
telemetry to metacognition, the cognitive program/DAG to persistent belief state,
and verified experience to memory/learning/cognitive compilation.

It owns one semantic model identity. No method creates or registers sub-models or
independent language-model agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from typing import Any

from .cognitive_kernel import (
    CognitiveProgram,
    DynamicCognitiveGraph,
    Evidence,
    GoalState,
    LearningRoute,
    LeviathanCognitiveKernel,
)
from .memory_ecology import BeliefStateStore, MemoryEcology, MemoryKind, MemoryRecord
from .parameter_cells import CellTelemetrySummary
from .types import Belief, MetaState, Provenance


@dataclass(slots=True)
class TaskSession:
    id: str
    problem: str
    task_type: str
    goal: GoalState
    meta_state: MetaState
    program: CognitiveProgram
    graph: DynamicCognitiveGraph
    belief_ids: list[str] = field(default_factory=list)
    prediction_ids: list[str] = field(default_factory=list)
    action_ids: list[str] = field(default_factory=list)
    verification_refs: list[str] = field(default_factory=list)
    episode_id: str | None = None
    outcome_ref: str | None = None
    verified_success: bool | None = None

    @property
    def closed(self) -> bool:
        return self.verified_success is not None


@dataclass(frozen=True, slots=True)
class TaskCompletion:
    session_id: str
    episode_id: str
    learning_route: LearningRoute
    skill_id: str
    skill_trials: int
    skill_success_rate: float
    skill_ready_to_compile: bool


class LeviathanRuntime:
    """One integrated runtime for one Leviathan semantic model."""

    def __init__(
        self,
        *,
        model_id: str,
        memory_journal: str | None = None,
    ) -> None:
        self.kernel = LeviathanCognitiveKernel(model_id=model_id)
        self.beliefs = BeliefStateStore()
        self.memory = MemoryEcology(memory_journal)
        self._sessions: dict[str, TaskSession] = {}
        self._session_counter = 0

    @property
    def model_id(self) -> str:
        return self.kernel.model_id

    @property
    def sessions(self) -> tuple[TaskSession, ...]:
        return tuple(self._sessions.values())

    def _next_session_id(self, *, problem: str, goal: GoalState) -> str:
        payload = json.dumps(
            {
                "model": self.model_id,
                "counter": self._session_counter,
                "problem": problem,
                "goal": goal.objective,
            },
            sort_keys=True,
        ).encode("utf-8")
        self._session_counter += 1
        return "task-" + sha256(payload).hexdigest()[:16]

    @staticmethod
    def incorporate_cell_telemetry(
        state: MetaState,
        telemetry: CellTelemetrySummary,
        *,
        total_routed_cells: int,
        rounds_used: int = 0,
        max_active_cells: int = 256,
    ) -> MetaState:
        if total_routed_cells <= 0 or max_active_cells <= 0:
            raise ValueError("cell-count budgets must be positive")
        if rounds_used < 0:
            raise ValueError("rounds_used cannot be negative")
        active_fraction = min(1.0, telemetry.unique_cells_seen / total_routed_cells)
        budget_pressure = min(1.0, telemetry.unique_cells_seen / max_active_cells)
        return replace(
            state,
            parameter_cell_disagreement=telemetry.mean_disagreement,
            parameter_cell_active_fraction=active_fraction,
            parameter_cell_rounds_used=rounds_used,
            parameter_cell_budget_pressure=budget_pressure,
        )

    def begin_task(
        self,
        *,
        problem: str,
        task_type: str,
        goal: GoalState,
        meta_state: MetaState,
    ) -> TaskSession:
        program, graph = self.kernel.compile_problem(
            problem=problem,
            task_type=task_type,
            goal=goal,
            state=meta_state,
        )
        session = TaskSession(
            id=self._next_session_id(problem=problem, goal=goal),
            problem=problem,
            task_type=task_type,
            goal=goal,
            meta_state=meta_state,
            program=program,
            graph=graph,
        )
        self._sessions[session.id] = session
        self.kernel.event_log.append(
            event_type="task_started",
            module="runtime",
            output_refs=(session.id, program.fingerprint),
            metadata={"model_id": self.model_id, "task_type": task_type},
        )
        return session

    def add_belief(self, session_id: str, belief: Belief, *, reason_ref: str) -> None:
        session = self._sessions[session_id]
        if session.closed:
            raise RuntimeError("cannot add belief to a closed task")
        self.beliefs.put(belief, reason_ref=reason_ref)
        if belief.id not in session.belief_ids:
            session.belief_ids.append(belief.id)
        self.kernel.event_log.append(
            event_type="belief_recorded",
            module="belief_state",
            input_refs=(reason_ref,),
            output_refs=(belief.id,),
            metadata={"confidence": belief.confidence},
        )

    def apply_evidence(self, session_id: str, evidence: Evidence) -> float:
        session = self._sessions[session_id]
        if session.closed:
            raise RuntimeError("cannot update belief in a closed task")
        belief = self.beliefs.get(evidence.target_id)
        update = self.kernel.evidence_updater.update(belief, evidence)
        self.beliefs.apply_confidence_update(
            belief.id,
            posterior_confidence=update.posterior_confidence,
            evidence_ref=evidence.id,
            contradiction=not evidence.supports,
        )
        self.kernel.event_log.append(
            event_type="evidence_applied",
            module="evidence_update",
            input_refs=(belief.id, evidence.id),
            output_refs=(belief.id,),
            metadata={
                "prior_confidence": update.prior_confidence,
                "posterior_confidence": update.posterior_confidence,
                "independence": evidence.independence,
            },
        )
        return update.posterior_confidence

    def record_prediction(self, session_id: str, prediction_id: str) -> None:
        session = self._sessions[session_id]
        if session.outcome_ref is not None:
            raise RuntimeError("predictions must be recorded before outcome")
        if prediction_id not in session.prediction_ids:
            session.prediction_ids.append(prediction_id)
        self.kernel.event_log.append(
            event_type="prediction_recorded",
            module="world_model",
            output_refs=(prediction_id,),
            metadata={"session_id": session_id},
        )

    def record_action(self, session_id: str, action_id: str) -> None:
        session = self._sessions[session_id]
        if session.closed:
            raise RuntimeError("cannot record action on a closed task")
        if action_id not in session.action_ids:
            session.action_ids.append(action_id)
        self.kernel.event_log.append(
            event_type="action_recorded",
            module="agency",
            output_refs=(action_id,),
            metadata={"session_id": session_id},
        )

    def record_verification(self, session_id: str, verification_ref: str) -> None:
        session = self._sessions[session_id]
        if verification_ref not in session.verification_refs:
            session.verification_refs.append(verification_ref)

    def complete_task(
        self,
        session_id: str,
        *,
        outcome_ref: str,
        verified_success: bool,
        provenance: Provenance,
        truth_quality: float,
        novelty: float,
        transfer_value: float,
        independent_verification: bool,
        rollback_available: bool,
        episode_payload: Any | None = None,
    ) -> TaskCompletion:
        session = self._sessions[session_id]
        if session.closed:
            raise RuntimeError("task is already complete")
        if not session.verification_refs:
            raise RuntimeError("task completion requires at least one verification record")

        session.outcome_ref = outcome_ref
        session.verified_success = bool(verified_success)
        episode_id = f"episode-{session.id}"
        payload = episode_payload if episode_payload is not None else {
            "problem": session.problem,
            "task_type": session.task_type,
            "goal": session.goal.objective,
            "program_fingerprint": session.program.fingerprint,
            "belief_ids": tuple(session.belief_ids),
            "prediction_ids": tuple(session.prediction_ids),
            "action_ids": tuple(session.action_ids),
            "verification_refs": tuple(session.verification_refs),
            "outcome_ref": outcome_ref,
            "verified_success": verified_success,
        }
        self.memory.write(
            MemoryRecord(
                id=episode_id,
                kind=MemoryKind.EPISODIC,
                payload=payload,
                confidence=max(0.0, min(1.0, truth_quality)),
                provenance=provenance,
                evidence_refs=tuple(session.verification_refs),
                source_refs=(session.id,),
                tags=(session.task_type,),
                utility=max(0.0, min(1.0, transfer_value)),
                verified=bool(verified_success and independent_verification),
                independent_verifications=1 if independent_verification else 0,
            )
        )
        session.episode_id = episode_id

        # Count previous verified successes for this exact cognitive program before
        # choosing the learning destination. The skill compiler and memory router are
        # therefore driven by the same verified trajectory history.
        skill = self.kernel.cognitive_compiler.observe(
            session.program,
            episode_id=episode_id,
            verified_success=verified_success,
        )
        route = self.kernel.learning_router.route(
            verified=verified_success,
            truth_quality=truth_quality,
            novelty=novelty,
            transfer_value=transfer_value,
            repeated_successes=skill.successes,
            rollback_available=rollback_available,
            independent_verification=independent_verification,
        )

        self.kernel.event_log.append(
            event_type="task_completed",
            module="runtime",
            input_refs=(session.id,),
            output_refs=(episode_id, skill.id),
            metadata={
                "verified_success": verified_success,
                "learning_destination": route.destination.value,
                "skill_trials": skill.trials,
                "skill_success_rate": skill.success_rate,
            },
        )
        return TaskCompletion(
            session_id=session.id,
            episode_id=episode_id,
            learning_route=route,
            skill_id=skill.id,
            skill_trials=skill.trials,
            skill_success_rate=skill.success_rate,
            skill_ready_to_compile=skill.ready_to_compile(),
        )
