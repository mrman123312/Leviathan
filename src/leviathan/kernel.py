"""Contracts for one cognitive model inside one Leviathan agent.

The kernel boundary is deliberately singular.  A kernel owns one parameter state,
reads one immutable context, and emits one decision trace.  Implementations may use
sparse tensor operations or repeat the same block for additional compute, but there
are no internal identities, proposals, messages, votes, or independently-goaled
models at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Protocol, runtime_checkable

from .types import CognitiveMode, MetaState


def _unit_interval(name: str, value: float) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def _non_negative(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True, slots=True)
class MetaSnapshot:
    """Immutable controller state supplied to the single cognitive kernel."""

    task_type: str
    goal: str
    success_probability: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    stakes: float
    risk_budget: float
    compute_budget: float
    latency_budget: float
    available_verifiers: tuple[str, ...]
    available_tools: tuple[str, ...]
    candidate_skills: tuple[str, ...]
    world_model_confidence: float
    branching_factor_estimate: float
    expected_information_gain: float
    hardware_load: float
    recent_failures: int

    @classmethod
    def from_state(cls, state: MetaState, *, immutable_goal: str) -> MetaSnapshot:
        """Copy controller state while restoring the externally anchored goal."""

        return cls(
            task_type=state.task_type,
            goal=immutable_goal,
            success_probability=state.success_probability,
            epistemic_uncertainty=state.epistemic_uncertainty,
            aleatoric_uncertainty=state.aleatoric_uncertainty,
            stakes=state.stakes,
            risk_budget=state.risk_budget,
            compute_budget=state.compute_budget,
            latency_budget=state.latency_budget,
            available_verifiers=tuple(state.available_verifiers),
            available_tools=tuple(state.available_tools),
            candidate_skills=tuple(state.candidate_skills),
            world_model_confidence=state.world_model_confidence,
            branching_factor_estimate=state.branching_factor_estimate,
            expected_information_gain=state.expected_information_gain,
            hardware_load=state.hardware_load,
            recent_failures=state.recent_failures,
        )


@dataclass(frozen=True, slots=True)
class CognitiveCandidate:
    """One state transition, answer, experiment, tool call, or external action."""

    id: str
    mode: CognitiveMode
    payload: Any
    expected_observation: Any | None = None
    preconditions: tuple[str, ...] = ()
    risk: float = 0.0
    reversible: bool = True
    authorization_class: str = "internal"
    verifier: str | None = None
    expected_success_gain: float = 0.0
    information_gain: float = 0.0
    transfer_value: float = 0.0
    compute_cost: float = 0.0
    latency_cost: float = 0.0
    irreversibility_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("candidate id must not be empty")
        if not self.authorization_class.strip():
            raise ValueError("authorization_class must not be empty")
        _unit_interval("risk", self.risk)
        _unit_interval("expected_success_gain", self.expected_success_gain)
        _unit_interval("information_gain", self.information_gain)
        _unit_interval("transfer_value", self.transfer_value)
        _non_negative("compute_cost", self.compute_cost)
        _non_negative("latency_cost", self.latency_cost)
        _non_negative("irreversibility_cost", self.irreversibility_cost)


@dataclass(frozen=True, slots=True)
class CognitiveContext:
    """Read-only state for one invocation of the one cognitive model."""

    agent_id: str
    goal_id: str
    meta: MetaSnapshot
    observation_id: str
    observation: Any
    routing_keys: frozenset[str] = frozenset()
    evidence_refs: tuple[str, ...] = ()
    refinement_budget: int = 1
    allowed_modes: frozenset[CognitiveMode] = field(
        default_factory=lambda: frozenset(CognitiveMode)
    )

    def __post_init__(self) -> None:
        if self.refinement_budget < 1:
            raise ValueError("refinement_budget must be at least 1")


class InferenceStatus(str, Enum):
    DECIDED = "decided"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_DECISION = "no_decision"


@dataclass(frozen=True, slots=True)
class KernelManifest:
    """Machine-checkable declaration of the strict single-model boundary."""

    parameter_owners: int = 1
    shared_states: int = 1
    routers: int = 1
    objectives: int = 1
    optimizers: int = 1
    checkpoints: int = 1
    decision_outputs: int = 1
    independent_internal_models: int = 0

    def violations(self) -> tuple[str, ...]:
        expected = {
            "parameter_owners": 1,
            "shared_states": 1,
            "routers": 1,
            "objectives": 1,
            "optimizers": 1,
            "checkpoints": 1,
            "decision_outputs": 1,
            "independent_internal_models": 0,
        }
        return tuple(
            f"{name}={getattr(self, name)} (required {required})"
            for name, required in expected.items()
            if getattr(self, name) != required
        )


@dataclass(frozen=True, slots=True)
class InferenceTrace:
    """Auditable summary of one model invocation, not a transcript of model parts."""

    status: InferenceStatus
    decision: CognitiveCandidate | None
    confidence: float
    uncertainty: float
    refinement_steps: int
    forward_passes: int
    active_parameters: int = 0
    total_parameters: int = 0
    route_entropy: float = 0.0
    reason: str | None = None

    def __post_init__(self) -> None:
        _unit_interval("inference confidence", self.confidence)
        _unit_interval("inference uncertainty", self.uncertainty)
        _non_negative("route entropy", self.route_entropy)
        if self.refinement_steps < 0:
            raise ValueError("refinement_steps must be non-negative")
        if self.forward_passes < 0:
            raise ValueError("forward_passes must be non-negative")
        if self.active_parameters < 0 or self.total_parameters < 0:
            raise ValueError("parameter counts must be non-negative")
        if self.active_parameters > self.total_parameters:
            raise ValueError("active_parameters cannot exceed total_parameters")
        if self.status is InferenceStatus.DECIDED and self.decision is None:
            raise ValueError("decided traces require a decision")
        if self.status is not InferenceStatus.DECIDED and self.decision is not None:
            raise ValueError("non-decided traces cannot contain a decision")


@runtime_checkable
class CognitiveKernel(Protocol):
    """The agent's single cognitive model boundary."""

    model_id: str
    manifest: KernelManifest

    def infer(self, context: CognitiveContext) -> InferenceTrace:
        """Return one decision trace from one parameter state."""


@dataclass(slots=True)
class ScriptedKernel:
    """Deterministic single-call fixture for boundary and governance tests.

    This is not a collection of simulated specialists.  It stands in for exactly one
    model so the agent envelope can be tested without a neural runtime.
    """

    candidate: CognitiveCandidate | None
    confidence: float = 0.9
    uncertainty: float = 0.1
    model_id: str = "scripted-single-kernel"
    manifest: KernelManifest = field(default_factory=KernelManifest)

    def __post_init__(self) -> None:
        _unit_interval("scripted confidence", self.confidence)
        _unit_interval("scripted uncertainty", self.uncertainty)
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")

    def infer(self, context: CognitiveContext) -> InferenceTrace:
        if self.candidate is None:
            return InferenceTrace(
                status=InferenceStatus.NO_DECISION,
                decision=None,
                confidence=0.0,
                uncertainty=1.0,
                refinement_steps=0,
                forward_passes=1,
                reason="the model emitted no decision",
            )
        if self.candidate.mode not in context.allowed_modes:
            return InferenceTrace(
                status=InferenceStatus.NO_DECISION,
                decision=None,
                confidence=0.0,
                uncertainty=1.0,
                refinement_steps=0,
                forward_passes=1,
                reason="the model selected a disallowed cognitive mode",
            )
        return InferenceTrace(
            status=InferenceStatus.DECIDED,
            decision=self.candidate,
            confidence=self.confidence,
            uncertainty=self.uncertainty,
            refinement_steps=0,
            forward_passes=1,
        )
